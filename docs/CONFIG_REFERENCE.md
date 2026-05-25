# Config Reference — `TitansConfig`

Every knob on `TitansConfig` with its type, valid range, default, and what
happens at the boundary. Source of truth: `config.py`. Detailed rationale per
field in `PLAN.md` §0.2.

> All validation is in `__post_init__` using `raise ValueError`, **not** `assert`
> (G190, G220, G223 — `python -O` strips asserts). Invalid configs fail at
> construction, not at first forward.

## Quick start

```python
from config import TitansConfig

cfg = TitansConfig.gpt2_small()                       # 124M defaults
cfg = TitansConfig.gpt2_small(chunk_size=512)         # override one field
cfg = TitansConfig.gpt2_medium(finetune_mode=False)   # from-scratch flavor
```

Factories: `gpt2_small()`, `gpt2_medium()`, `gpt2_large()`, `gpt2_xl()`. They
merge overrides via `**{**defaults, **overrides}` so any field can be overridden
without a `TypeError: multiple values for keyword argument` (this is a real
footgun if you pass dims as fixed kwargs alongside `**overrides`).

---

## GPT-2 backbone dimensions

| Field | Type | Default (small) | Range | Notes |
|---|---|---|---|---|
| `n_layer` | `int` | 12 | ≥ 1 | Number of transformer blocks |
| `n_head` | `int` | 12 | ≥ 1, must divide `n_embd` | `n_embd % n_head != 0` → `ValueError` (G223) |
| `n_embd` | `int` | 768 | ≥ 1, multiple of `n_head` | Model dim (`d_model`) |
| `block_size` | `int` | 1024 | ≥ 1 | Position embedding table size; max attention context |
| `vocab_size` | `int` | 50257 | matches tokenizer | GPT-2 BPE |
| `dropout` | `float` | 0.0 | [0.0, 1.0) | Standard GPT-2 dropout |

Factory presets:

| Factory | `n_layer` | `n_head` | `n_embd` |
|---|---|---|---|
| `gpt2_small()` | 12 | 12 | 768 |
| `gpt2_medium()` | 24 | 16 | 1024 |
| `gpt2_large()` | 36 | 20 | 1280 |
| `gpt2_xl()` | 48 | 25 | 1600 |

HF model path is derived from `n_embd` in `load_pretrained` (G216) — passing
`n_embd=1024` automatically pulls `openai-community/gpt2-medium`.

---

## NMM hyperparameters

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `nmm_depth` | `int` | 2 | ≥ 1 | `L_M` in paper. `=2` is the SiLU-GLU gated structure we use; `=1` is a single linear map (paper ablation: ≥2 ≫ 1) |
| `nmm_expansion` | `int` | 4 | ≥ 1 | Hidden dim multiplier (`hidden = expansion · n_embd`). `=1` halves state memory but reduces capacity |
| `nmm_conv_kernel` | `int` | 4 | ≥ 1 | Depthwise conv kernel size in Q/K/V projections (§4.4 of paper). `=1` disables temporal mixing |
| `nmm_spectral_norm` | `bool` | `True` | — | Newton-Schulz 5-step on inner gradient. **Toggling this requires also changing inner-loss reduction** (G160 — see below) |
| `nmm_n_persistent` | `int` | 4 | ≥ 0 | Number of learned persistent tokens prepended per block. `=0` disables them |
| `chunk_size` | `int` | 512 | 1 ≤ x ≤ `block_size` | TBPTT chunk length. `> block_size` → `ValueError` (would OOB `wpe`). From-scratch users should set `chunk_size = block_size` to avoid the G163 untrained-`wpe`-rows warning. |
| `nmm_n_heads` | `int` | 1 | ≥ 1, must divide `n_embd` | NUMBER of parallel NMM heads. Default `1` = single-head (current behavior). `>1` instantiates `MultiHeadNMM` wrapping N parallel `NeuralMemoryModule`s on `head_dim = n_embd // n_heads`. NOT in the paper proper — this is a lucidrains enhancement exposed for ablation (G254). When `>1`, the per-layer NMM state becomes a list of per-head `(M, S)` tuples; `detach_states` / `compute_nmm_norm` handle this recursively. |

## Memory-saving knobs (G256 / G257)

Reach for these when training OOMs on the per-token NMM graph (the
typical failure mode at `chunk_size >= 64` on consumer GPUs).
**Disabled by default** so the original numerical and performance
properties are preserved.

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_state_dtype` | `str` | `"fp32"` | Storage dtype for the recurrent `(M, S)` and per-step update buffers. `"bf16"` roughly halves per-step state retention. NS5 still casts to fp32 internally (G226), so the spectral-norm fixed point is preserved. The drift risk is the per-step `M_t = (1-α)·M_{t-1} + S_t` rounding in bf16 — measure loss curves before relying on it. Valid: `"fp32"`, `"bf16"`. `fp16` is rejected (would need loss scaling). |
| `nmm_grad_checkpoint` | `bool` | `False` | When `True`, `_forward_chunk_sequential` runs the per-token inner loop in `nmm_grad_checkpoint_segment_len`-token segments, each wrapped in `torch.utils.checkpoint.checkpoint(use_reentrant=True)`. Backward recomputes inner-loop intermediates instead of storing them. **The unlock for `T ≥ 64` on a 16 GiB consumer card.** Composes with `nmm_state_dtype="bf16"` multiplicatively. The scan path (`_forward_chunk_scan`) ignores this flag. |
| `nmm_grad_checkpoint_segment_len` | `int` | 64 | Segment size when grad-checkpointing is on (no effect otherwise). Smaller = less peak memory + more recompute; larger = more peak memory + less recompute. Tuning guidance: start at `64`; drop to `32` or `16` if OOM persists. |
| `nmm_cpu_offload_segments` | `bool` | `False` | Requires `nmm_grad_checkpoint=True`. Replaces the GPU checkpoint with a CPU-offload variant: saved segment-boundary `(M, S)` tensors are stashed on CPU between forward and backward, and pulled back to GPU for recompute one segment at a time. Removes the `n_blocks × n_segments × per_segment` VRAM ceiling that bounds the GPU checkpoint at long `T`. **Trade:** PCIe transfer cost (~16 GiB/s on PCIe 4.0 x16) added to every backward — expect a 1.5–3× step-time slowdown depending on `T` and `seg_len`. Uses a custom `autograd.Function` (not `save_on_cpu`, which uses `saved_tensors_hooks` rejected by `torch.func.grad`). |
| `nmm_compile_scan_training` | `bool` | `False` | Sets `_allow_scan_training=True` on every block's NMM at construction so the dispatcher in `forward_chunk` takes the associative-scan path **even with autograd enabled**. **You still need to wrap the model in `torch.compile(model)` yourself** — without compile, `associative_scan` has no autograd and silently zeros NMM gradients (G164 / G180). **This is a COMPUTE optimization (parallelism), NOT memory** — the scan path allocates `[T, B, h, d]` gradient tensors upfront, which at T=1024 is *larger* than the sequential path's per-token graph. Useful when paired with `cpu_offload` to fit T=1024 *and* run it faster. **Caveat:** scan is an APPROXIMATION — per-token gradients are computed against chunk-start `M_0`, not paper-faithful `M_{t-1}` — so training loss curves will differ from sequential. |
| `nmm_block_grad_checkpoint` | `bool` | `False` | Wraps each `TitansMAGBlock.forward` in `torch.utils.checkpoint.checkpoint(use_reentrant=True)`. The whole block (attention + NMM forward_chunk + MAG gate + MLP) is recomputed on backward; only block-input/output tensors and the NMM `(M, S)` I/O dicts live in the autograd graph between blocks. Removes the `n_blocks × n_segments × per_segment` boundary term that dominates GPU memory at long T. **Combine with `nmm_grad_checkpoint=True`** — block checkpoint alone would re-build the full per-token NMM graph during a block's backward recompute (≈ 50 GiB at T=1024 gpt2_small), OOMing immediately. The segment checkpoint bounds the in-block transient during recompute. Cost: each block's forward runs twice (forward + backward recompute), ~2× step time on top of inner-segment recompute. Handles both single-head `(M, S)` and multi-head `[(M, S), ...]` states via the `_state_to_flat` / `_flat_to_state` helpers in `model/block.py`. |

**Why `use_reentrant=True`** is *required* (not just convenient):
`use_reentrant=False` calls `disable_saved_tensors_hooks`, and
`torch.func.grad` (used inside `per_sample_grad_fn`) rejects that
at runtime. The reentrant path uses `torch.autograd.function.Function`
which composes with `torch.func`.

**Measured impact** (RTX 5070 Ti, 16 GiB, `gpt2_small` with `bf16` autocast,
end-to-end `train_step` with backward + Adam):

| Configuration | Peak VRAM | Step time | Status |
|---|---|---|---|
| `B=1, T=64`, fp32, no ckpt (baseline) | 12.6 GiB | — | OOM |
| `B=1, T=64`, bf16, no ckpt | 12.6 GiB | — | OOM (bf16 alone ~no help) |
| `B=1, T=64`, fp32, ckpt seg=16 | 10.9 GiB | (fast) | OK |
| `B=1, T=64`, bf16 + ckpt seg=16 | 7.7 GiB | (fast) | OK |
| `B=1, T=128`, bf16 + ckpt seg=16 | 9.0 GiB | (fast) | OK |
| `B=1, T=256`, bf16 + ckpt seg=16 | 11.7 GiB | (fast) | OK ← **fast practical limit** |
| `B=2, T=128`, bf16 + ckpt seg=16 | 13.9 GiB | — | OOM |
| `B=1, T=512`, bf16 + ckpt seg=16 | 12.8 GiB | — | OOM |
| `B=1, T=1024`, bf16 + ckpt seg ∈ {8..64} | ~13 GiB | — | OOM at every seg |
| `B=1, T=256`, bf16 + ckpt seg=32 + cpu_offload | 10.8 GiB | 103 s | OK (slow) |
| `B=1, T=1024`, bf16 + ckpt seg=32 + cpu_offload | **11.1 GiB** | **405 s** | OK ← **T=1024 via cpu_offload** |
| `B=2, T=256`, bf16 + ckpt seg=32 + cpu_offload | 13.8 GiB | — | OOM (in-segment grows with B) |
| `B=1, T=256`, bf16 + ckpt seg=32 + block_ckpt | 11.0 GiB | 127 s | OK |
| `B=1, T=512`, bf16 + ckpt seg=32 + block_ckpt | **11.3 GiB** | 250 s | OK ← **T=512 via block_ckpt** |
| `B=1, T=1024`, bf16 + ckpt seg=32 + block_ckpt | 11.8 GiB | 497 s | OK |
| `B=1, T=1024`, bf16 + ckpt seg=32 + block_ckpt + cpu_offload | 11.0 GiB | 463 s | OK (combined, no big win) |
| `B=2+`, T=512/1024 + block_ckpt + (optional) cpu_offload | ~13.8 GiB | — | OOM (in-segment transient scales with B) |

**Findings:**

- Gradient checkpointing (`nmm_grad_checkpoint`) is the dominant unlock
  for `T ≥ 64`. Required as a baseline for any further memory work.
- `bf16` state (`nmm_state_dtype="bf16"`) stacks on top, ~2× headroom.
- **`cpu_offload` lets T=1024 fit on a 16 GiB card** at ~5–10× step-
  time cost (PCIe + extra recompute). Correctness work, not production.
- **`block_grad_checkpoint` also lets T=1024 fit**, and lets T=512 fit
  (where it didn't before). Comparable step-time penalty to
  `cpu_offload`. Choose one or the other; combining them gives no
  meaningful additional win at T=1024.
- **All optimizations cap at `B=1`** on this 16 GiB card. The in-segment
  per-block transient (~`seg_len × B × per_step_state`) scales with B
  and is *not* reduced by either `cpu_offload` (which only offloads
  saved-input boundaries) or `block_grad_checkpoint` (which only
  reduces cross-block boundary storage). To get B>1 at long T you need
  either a smaller backbone, `nmm_expansion=1`, or hardware with more
  VRAM.

**Recommended configs by goal:**

```python
# Fast-but-short: real training on this card, ~256 tokens / step.
TitansConfig.gpt2_small(
    chunk_size=256, block_size=256,
    nmm_state_dtype="bf16",
    nmm_grad_checkpoint=True,
    nmm_grad_checkpoint_segment_len=16,
)

# Long-but-slow: T=1024 fits, but 5+ minutes per step (cpu_offload path).
TitansConfig.gpt2_small(
    chunk_size=1024, block_size=1024,
    nmm_state_dtype="bf16",
    nmm_grad_checkpoint=True,
    nmm_grad_checkpoint_segment_len=32,
    nmm_cpu_offload_segments=True,
)

# Alternative for T=512 / T=1024: block-level checkpoint instead of
# cpu_offload. Comparable VRAM win, comparable step-time slowdown.
# Pick this if you don't want CPU↔GPU transfers (e.g. PCIe 3.0
# bottleneck) and want full-GPU semantics.
TitansConfig.gpt2_small(
    chunk_size=512, block_size=512,
    nmm_state_dtype="bf16",
    nmm_grad_checkpoint=True,
    nmm_grad_checkpoint_segment_len=32,
    nmm_block_grad_checkpoint=True,
)

# Faster-but-approximate: scan path under torch.compile.
# Caller must `model = torch.compile(model)` after construction.
TitansConfig.gpt2_small(
    chunk_size=512, block_size=512,
    nmm_state_dtype="bf16",
    nmm_compile_scan_training=True,
)
# Note: gradients are computed at chunk-start M_0 (approximation);
# loss curves will differ from sequential training.
```

To do T=1024 + B>1 at practical step times, you really need a card
with ≥24 GiB VRAM. The implemented optimizations get this 16 GiB card
to T=256 at full speed, T=1024 at 5+ min/step, but cannot make T=1024
fast.

**G160 — `nmm_spectral_norm` and inner-loss reduction are linked:**

| `nmm_spectral_norm` | Inner-loss reduction | Why |
|---|---|---|
| `True` | `'sum'` | NS divides by Frobenius — sum is natural |
| `False` | `'mean'` | Without NS, sum gives gradients `d_model×` too large → silent divergence |

If you flip the flag, the reduction must flip too. This is handled inside `NeuralMemoryModule.__init__`, but a user-supplied reduction override would break it.

---

## Attention flags

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `use_swa` | `bool` | `False` | — | Sliding Window Attention. Recommended **off** for GPT-2 fine-tune (pretrained with full attn); **on** for from-scratch long-context experiments |
| `swa_window` | `int` | 256 | ≥ 1 if `use_swa=True` | `use_swa=True and swa_window < 1` → `ValueError` (softmax NaN at step 0) |

Note: field is `swa_window`, **not** `window_size` (G126).

---

## Mode flags

| Field | Type | Default | Notes |
|---|---|---|---|
| `finetune_mode` | `bool` | `True` | Selects the MAG gate formula and `out_scale` init (see below) |

## Paper-vs-lucidrains flags (G254)

These flags expose two deliberate paper/lucidrains divergences as runtime
config. **Defaults prefer paper-strict** (G254 default-flip). Flip to
`False` for lucidrains-flavored ablations.

| Field | Type | Default | Notes |
|---|---|---|---|
| `retrieval_from_M_prev` | `bool` | `True` (paper Eq. 15) | Paper Eq. 15: `y_t = M(q_t)` where M is M_{t-1} (read-then-write). Default `True` = paper-strict. Flip to `False` for lucidrains write-then-read (retrieve from freshly-updated M_t). Applies to all NMM forward paths (`step`, `step_with_conv`, `_forward_chunk_sequential`, `_forward_chunk_scan`). |
| `feed_persistent_to_nmm` | `bool` | `True` (paper Eq. 28) | Paper Eq. 28: `M(x̃)` where x̃ = `concat(persistent, x)`. Default `True` = paper-strict — the block feeds `ln_nmm(x_aug)` to NMM, augments `doc_boundaries` with a False prefix (persistent positions never trigger resets), and slices the prefix off `y_mem` before the residual. Flip to `False` for lucidrains-flavored (NMM sees only real tokens). |

### `finetune_mode=True` (default, recommended for pretrained init)

- MAG gate: `o = y_attn + silu(γ_m · y_mem) * y_attn = y_attn · (1 + silu(γ_m · y_mem))`
- `out_scale` init: **zeros** → `y_mem ≈ 0` at step 0 → `o = y_attn` exactly
- `gamma_attn` is **not created** (only `gamma_mem`)
- Net effect: a freshly built model with HF GPT-2 weights produces identical logits to vanilla GPT-2 (max diff < 1e-4)

### `finetune_mode=False` (from-scratch, paper formula)

- MAG gate: `o = silu(γ_a · y_attn) * silu(γ_m · y_mem)` (Hadamard)
- `out_scale` init: **ones**
- Both `gamma_attn` and `gamma_mem` created, both init to ones
- Use when training from scratch — there is no pretrained residual to preserve

---

## Optimizer and training (constructed outside `TitansConfig`)

These are passed to `train.py` / `apply_lr` directly, not stored on the config object. **Critical: `base_lrs` must be sourced from code-level constants, not from `optimizer.param_groups[i]['lr']`** (G162 — each resume cycle silently compounds LR deflation otherwise).

| Constant | Default | Notes |
|---|---|---|
| `BASE_LR_GPT2` | `3e-4` | LR for `gpt2_decay` / `gpt2_no_decay` groups |
| `BASE_LR_NMM` | `9e-4` | LR for `nmm_decay` / `nmm_no_decay` groups (3× GPT-2 per paper) |
| `WEIGHT_DECAY` | `0.1` | Applied only to `_decay` groups |
| `BETAS` | `(0.9, 0.95)` | AdamW betas |
| `WARMUP_STEPS` | `2000` (recommended for production) | Thread explicitly into `apply_lr` — no module-level constant in `train.py`; `apply_lr` requires it positional (G175). CLI defaults are smaller for quick local runs: `train.py --warmup-steps` defaults to `1000`, `scripts/finetune.py --warmup-steps` defaults to `500`. For real training runs override with `--warmup-steps 2000`. |
| `MAX_STEPS` | run-specific | Thread explicitly into `apply_lr` (G175) |
| `LR_MIN_RATIO` | `0.1` | Cosine schedule floor as fraction of peak |
| `GRAD_CLIP` | `1.0` | Applied in fp32 even under bf16 autocast (G159) |

### Param-group routing rules

| Param name contains... | Group |
|---|---|
| `'nmm'` AND `'bias'`/`'ln'`/`'norm'`/`'out_scale'`/`'gamma'`/`'persistent'` | `nmm_no_decay` |
| `'nmm'` else | `nmm_decay` |
| `'bias'`/`'ln'`/`'norm'`/`'out_scale'`/`'gamma'`/`'persistent'` (no `'nmm'`) | `gpt2_no_decay` |
| else | `gpt2_decay` |

`out_scale`, `gamma_*`, and `persistent_mem` are intentionally no-decay — decay would shrink them toward zero, suppressing the memory branch and prefix capacity.

**Runtime check**: no param may appear in two groups. Add an assertion that
`sum(len(g['params']) for g in groups) == len(list(model.parameters()))`.

---

## Memory footprint (NMM state)

State `(M, S)` per layer scales as:

```
state_bytes_per_layer ≈ 2 (M+S) × 3 (W1, W_gate, W2) × B × 4·d_model² × dtype_bytes
                       = 24 · B · d_model² · dtype_bytes
```

For `gpt2_small` (`d_model=768`), `B=4`, fp32: **~54 MB per layer × 12 layers ≈ 650 MB**.

| Strategy | State reduction |
|---|---|
| `nmm_expansion=1` | 4× smaller (hidden = `d_model` instead of `4·d_model`) |
| bf16 states | 2× smaller (some loss of NS precision — measure first) |
| Lower `batch_size` | linear |

NMM state is **not saved in checkpoints** — it's per-sequence, not model state, and is re-initialized on resume from `memory_mlp.W*.weight`.

---

## Boundary behavior (validation cases)

| Boundary case | Behavior | Detail |
|---|---|---|
| `chunk_size > block_size` | `ValueError` at config build | Would OOB the wpe table |
| `chunk_size < block_size` (from-scratch) | `warnings.warn` | wpe rows beyond `chunk_size` never trained → long-context generation degrades silently |
| `n_embd % n_head != 0` | `ValueError` at config build (G223) | Plus the same check in `CausalSelfAttention.__init__` (G220, defense in depth) |
| `use_swa=True and swa_window < 1` | `ValueError` | Empty window → softmax over zero entries → NaN |
| `nmm_n_persistent < 0` | `ValueError` | Negative tensor dim |
| `nmm_expansion < 1` | `ValueError` | Empty hidden layer |
| `python -O` flag at runtime | **Validation still fires** | We use `raise ValueError`, not `assert` — see G190/G220/G223 |

---

## Cross-references

- Full implementation per phase: `ROADMAP.md` §0.2, `PLAN.md` §0.2
- Why these specific safeguards: `GAP_HISTORY.md` (search for the G-numbers above)
- What to do when training fails despite valid config: `RUNBOOK.md`
- Test coverage for validation: `TEST_PLAN.md` §4 (Phase 0 unit tests)
