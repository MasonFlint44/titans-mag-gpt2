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

**Why `use_reentrant=True`** is *required* (not just convenient):
`use_reentrant=False` calls `disable_saved_tensors_hooks`, and
`torch.func.grad` (used inside `per_sample_grad_fn`) rejects that
at runtime. The reentrant path uses `torch.autograd.function.Function`
which composes with `torch.func`.

**Measured impact** (RTX 5070 Ti, 16 GiB, `gpt2_small` with `bf16` autocast):

| Configuration | Peak VRAM | Status |
|---|---|---|
| `B=1, T=64`, fp32, no ckpt (baseline) | 12.6 GiB | OOM |
| `B=1, T=64`, bf16, no ckpt | 12.6 GiB | OOM (bf16 alone ~no help — activations are already bf16) |
| `B=1, T=64`, fp32, ckpt seg=16 | 10.9 GiB | OK |
| `B=1, T=64`, bf16 + ckpt seg=16 | 7.7 GiB | OK |
| `B=1, T=128`, bf16 + ckpt seg=16 | 9.0 GiB | OK |
| `B=1, T=256`, bf16 + ckpt seg=16 | 11.7 GiB | OK ← **practical limit on this card** |
| `B=2, T=128`, bf16 + ckpt seg=16 | 13.9 GiB | OOM |
| `B=1, T=512`, bf16 + ckpt seg=16 | 12.8 GiB | OOM |
| `B=1, T=1024`, bf16 + ckpt seg ∈ {8, 16, 32, 64} | ~13 GiB | OOM at every seg |

Checkpointing is the dominant unlock — it converts the per-token graph
from "all T steps retained" to "boundary states retained, recompute
within segment." The remaining cost scales as
`n_blocks × (T / seg_len) × per_segment_state_bytes`. For `gpt2_small`
that's `12 × (T/seg) × ~28·B MiB` per layer of M/S boundaries; T=1024
needs >40 GiB of boundary storage at any reasonable seg_len.

**To fit T=1024 on consumer hardware** you'd need: a smaller backbone
(`gpt2_small(n_layer=6)`), `nmm_expansion=1` (halves state), or a
GPU with ≥40 GiB. The implemented optimizations get a 16 GiB card to
T=256, not T=1024.

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
