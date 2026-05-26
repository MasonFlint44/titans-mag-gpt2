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
| `nmm_depth` | `int` | 2 | `== 2` | `L_M` in the paper. Only L_M=2 (single SwiGLU block) is implemented; other values raise `ValueError` at config construction. Wiring up L_M > 2 requires generalizing the analytical-gradient + Triton kernels; the paper's ablations show only marginal gains beyond 2. |
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
| `nmm_state_dtype` | `str` | `"fp32"` | Storage dtype for the recurrent `(M, S)` and per-step update buffers. `"bf16"` roughly halves per-step state retention; `"int8"` (blockwise path only) quarters it. NS5 still casts to fp32 internally (G226), so the spectral-norm fixed point is preserved. The drift risk for bf16 is the per-step `M_t = (1-α)·M_{t-1} + S_t` rounding — measure loss curves before relying on it. Valid: `"fp32"`, `"bf16"`, `"int8"`. `fp16` is rejected (would need loss scaling). |

## Capacity-vs-memory knobs (G261, G262, G263)

These trade NMM capacity for VRAM/speed. Together they're the **big
unlock** for T=1024 on consumer hardware — at gpt2_small d=768 default
expansion=4, the NMM's per-token autograd graph is ~300 MiB; combined
they bring this to ~30 MiB without losing much (`low_rank=64` is well
above the rank-of-meaningful-information for memory_mlp-style maps).

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_expansion` | `int` | `4` | Already an existing knob: the hidden-dim multiplier for `MemoryMLP` (paper has it implicit at 4× from the GPT-2 MLP convention). Setting `nmm_expansion=1` makes the three weight matrices square `[d, d]` and **quarters** the per-step NMM state. Paper ablation says lower expansion hurts capacity but is the smallest viable structure. NS5 still works on square matrices (no transpose). |
| `nmm_layer_indices` | `Optional[list[int]]` | `None` | When set, NMM is wired only on the listed blocks; others become plain GPT-2 blocks (attention + MLP only, no NMM, no persistent prefix, no MAG gate). Linear reduction of NMM-related compute and memory in the number of NMM layers. The paper applies NMM at every block; this is a deliberate departure for VRAM. Validation rejects out-of-range / duplicate / non-int indices. **Decode path is supported**: plain blocks contribute a KV cache only (no NMM conv buffer; their slot in `nmm_conv_buffers` is `None`). |
| `nmm_low_rank` | `Optional[int]` | `None` | When set, factor `MemoryMLP`'s three weight matrices as `A @ B` with intermediate rank `r`. Per-step NMM state drops from `3 × 4d²` to `3 × r × 5d` ≈ `5r/(4d)` of full-rank. At `d=768, r=64` that's ~10× smaller (the biggest single-knob win for long T). State key set goes from 3 to 6; the existing checkpoint plumbing handles this via the `state_keys` discovery at NMM construction time. NS5 still converges on the factored rectangles. Validation rejects `r >= n_embd` (factored form would be larger than full-rank — defeats the purpose). |

**Findings (RTX 5070 Ti, 16 GiB):**

- The per-token sequential path holds the entire chunk's autograd graph
  in memory; on a 16 GiB consumer card it tops out around `B=1, T≈64-128`
  even with `nmm_state_dtype="bf16"`. The right answer for long T is the
  **blockwise path** (`nmm_block_size >= 16`), which engages TC via
  batched matmul and replaces the per-token autograd graph with a
  per-block one (~`T / block_size` smaller).
- `bf16` state (`nmm_state_dtype="bf16"`) ~2× headroom on per-step
  buffers, composable with everything else.
- **`nmm_low_rank=64` is the biggest single-knob memory win at long T.**
  Factoring `memory_mlp` weights shrinks the per-step state ~10×; pair
  with blockwise + bf16 to fit T=1024 at meaningful batch size.
- **`nmm_layer_indices` is the speed knob.** Reducing 12 NMM blocks
  to 4 cuts step time roughly 4×.

**Recommended configs by goal:**

```python
# Fast long-T training on a consumer GPU (RECOMMENDED for T >= 256).
TitansConfig.gpt2_small(
    chunk_size=1024, block_size=1024,
    nmm_state_dtype="bf16",
    nmm_block_size=64,              # TC-engaged blockwise path
    nmm_low_rank=64,                # ~10x smaller per-step state
)

# Maximum speed at T=1024 if you accept reduced NMM capacity.
TitansConfig.gpt2_small(
    chunk_size=1024, block_size=1024,
    nmm_state_dtype="bf16",
    nmm_block_size=64,
    nmm_low_rank=64,
    nmm_layer_indices=[0, 3, 6, 9], # NMM on 4 of 12 blocks
)
```

### CLI flags (`train.py` / `scripts/finetune.py`)

The NMM perf/memory knobs are also exposed as `--nmm-*` CLI flags on both
training entry points (registered via `scripts/_nmm_cli.py::add_nmm_args`).
Omit a flag to keep the factory default; only the flags you set get
passed to `TitansConfig`. Quick reference:

| Flag | Type | Effect |
|---|---|---|
| `--nmm-block-size N` | int | Blockwise NMM aggregation (1 = paper-strict per-token; ≥16 engages TC) |
| `--nmm-state-dtype {fp32,bf16,int8}` | str | Storage dtype for (M, S). int8 requires `--nmm-block-size > 1` |
| `--nmm-low-rank R` | int | Factor MemoryMLP weights as A @ B with rank R |
| `--nmm-expansion N` | int | MemoryMLP hidden-dim multiplier (paper default 4; set 1 for ~4× smaller state at minor capacity cost) |
| `--nmm-layer-indices I,J,K` | csv ints | Subset of blocks that get NMM (others become plain GPT-2 blocks) |
| `--nmm-detach-state-between-blocks` | flag | Truncated BPTT at block boundaries (requires `--nmm-block-size > 1`) |
| `--nmm-compile-inner-loop` | flag | torch.compile the inner loop (~1.7× speedup, 30-60s first-step warm-up) |
| `--nmm-fused-kernel` | flag | Analytical inner gradient (~5-15% on top of `--nmm-compile-inner-loop`) |
| `--nmm-compile-ns5` | flag | Fused NS5 via torch.compile (no effect when `--nmm-compile-inner-loop` is set) |

Recommended invocation for T=1024 on a 16 GiB consumer card:

```bash
python -m train --data corpus.txt \
    --chunk-size 1024 --batch-size 1 --grad-accum 16 \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-low-rank 64 \
    --nmm-compile-inner-loop \
    --nmm-fused-kernel \
    --compile-model \
    --optim8bit
```

If you want to avoid `nmm_low_rank` (keep full-rank MemoryMLP), drop
`--nmm-low-rank 64` and add `--nmm-expansion 1` or
`--nmm-layer-indices 0,3,6,9` if you hit OOM.

---

## Inner-loop speed knobs (G264, G264a)

Capacity knobs above (`low_rank`, `layer_indices`, etc.) fit T=1024 but
the step time is dominated by the **Python-supervised per-token inner
loop**: 12 layers × T tokens × ~70 small ops = ~30K ops per chunk, each
paying ~5-10 µs of PyTorch dispatch overhead. At T=1024 this is
~150 s/step before any algorithmic improvements. These two knobs
attack that overhead directly — sequential semantics preserved, only
the implementation changes.

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_compile_inner_loop` | `bool` | `False` | Wraps `_run_inner_loop` in `torch.compile(mode="default")`. Inductor traces the Python time loop and fuses adjacent ops into batched Triton kernels, eliminating the per-token dispatch overhead. **Dominant ~1.7× speedup** on its own. First step pays a one-time compile cost (~30-60 s); subsequent steps are fast. `mode="default"` (not `"reduce-overhead"`) because the inner loop builds new dicts each step which cudagraphs reject without explicit `cudagraph_mark_step_begin()` calls. |
| `nmm_fused_kernel` | `bool` | `False` | Replaces `vmap(grad(inner_loss))` + autograd-through-NS5 with hand-derived analytical gradients (see `model/nmm_fused.py`). Adds **~5-15% on top of `compile_inner_loop`**. The hand-derived ops also trace more cleanly through Inductor than `torch.func.grad`. Correctness oracle is the reference path; `tests/unit/test_nmm_fused.py` locks the numerical match to fp32 round-off. |

**Measured impact** (RTX 5070 Ti, gpt2_small, bf16 autocast, end-to-end
step with backward + Adam, `nmm_low_rank=64`, grad checkpoint on):

| Config | T | B | Step time | tok/s | Speedup vs ref |
|---|---|---|---|---|---|
| reference | 256 | 1 | 40.0 s | 6.4 | 1.0× |
| `+fused_kernel` | 256 | 1 | 33.2 s | 7.7 | 1.20× |
| `+compile_inner_loop` | 256 | 1 | 21.9 s | 11.7 | **1.82×** |
| `+fused_kernel +compile_inner_loop` | 256 | 1 | **21.0 s** | **12.2** | **1.90×** |
| reference | 1024 | 1 | 158.1 s | 6.5 | 1.0× |
| `+fused_kernel` | 1024 | 1 | 136.0 s | 7.5 | 1.16× |
| `+compile_inner_loop` | 1024 | 1 | 93.0 s | 11.0 | **1.70×** |
| `+fused_kernel +compile_inner_loop` | 1024 | 1 | **83.3 s** | **12.3** | **1.90×** |

**Why isn't `compile_inner_loop` already 10×?** The remaining cost is
the actual *compute* of the per-token NMM update plus the unavoidable
sequential dependency on `(M, S)`. Inductor fuses adjacent ops into ~1
Triton kernel per logical block, but it cannot parallelise across the
time dimension (state is recurrent). The matmul shapes at our scale
(W [H, D] @ k [D], outer products u [H] ⊗ v [D]) don't engage tensor
cores — cuBLAS uses the same non-TC kernels for these shapes. The
algorithmic answer is **`nmm_block_size > 1`** (blockwise path),
which batches matmuls across tokens to engage TC.

### Blockwise NMM (G266, G267) — chunk-as-update for tensor-core engagement

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_block_size` | `int` | `1` | Chunk-as-update aggregation. At `1` (default), the inner loop is paper-strict per-token. At `>1`, every `nmm_block_size` consecutive tokens produce ONE memory update via batched matmul. **TC engages at `block_size >= 16`** — every forward matmul (pre1, preg, y, retrieval) becomes a `[B, H, D] @ [B, D, block_size] → [B, H, block_size]` GEMM where the `block_size` is the N dim. Gradient accumulation across the block is a single `einsum("bth,btd->bhd", d_pre1, k)` — also TC-eligible. Trailing block may be smaller than `block_size`; math handles it (just runs without TC on that one block). |
| `nmm_per_token_ns5` | `bool` | `False` | G267 paper-faithful refinement. When `True` + `block_size > 1`, the blockwise update applies NS5 PER TOKEN and weights by per-token θ before summing — matching paper Eq 16's `Σ_t θ_t · NS5(∇_t)` exactly. The default (`False`) uses v1's `θ_mean · NS5(Σ_t ∇_t)` simplification. **Note**: a tempting "cheap" approach — folding θ into the pre-NS5 aggregation — does NOT work because NS5 normalises the Frobenius norm, cancelling any positive scalar applied before it. Real per-token θ weighting requires per-token NS5, which materialises per-token gradient tensors of shape `[B, block, H, D]` per key. **Cost**: ~5-10× slower than `per_token_ns5=False` and significantly higher peak memory; at gpt2_small T=1024 with `block_size=64` it OOMs at 16 GiB. Use only when paper-strict per-token θ matters and your config has the memory budget (small T or rented compute) — reach for `nmm_low_rank` if memory is tight. |

**Approximation cost at `block_size > 1`**:
- All tokens within a block share the block-start M for surprise gradient AND retrieval. Paper's per-token M_{t-1} resolution becomes per-block M_{block-1}.
- One theta/eta/alpha per block (mean over the block's tokens) instead of per-token.
- At `block_size = 1`, math is identical to the sequential path (locked by tests).

**Measured impact for `per_token_ns5=True`** (G267, paper-faithful):

| Config | Step time | tok/s | Peak VRAM |
|---|---|---|---|
| T=512, B=1, block=32 | 6.32 s | 81 | 12.31 GiB |
| T=512, B=1, block=16 | 8.07 s | 63 | 12.43 GiB |
| T=256, B=1, block=64 | 3.17 s | 81 | 7.50 GiB |
| T=1024, B=1, block=64 | OOM | — | 13.9 GiB+ |

~5-10× slower than `per_token_ns5=False` but math-faithful to paper Eq 16.
Useful when paper-strict per-token θ semantics are required for an
ablation or when comparing against published TITANS numbers.

**Measured impact for `per_token_ns5=False`** (v1, the recommended path)
(RTX 5070 Ti, gpt2_small, T=1024, B=1, low_rank=64, bf16+ckpt):

| Path | block_size | Step time | tok/s | Peak VRAM | Speedup vs sequential |
|---|---|---|---|---|---|
| sequential (compile + fused_kernel) | 1 | 84.7 s | 12 | 4.90 GiB | 1.0× |
| **blockwise** | 32 | 3.77 s | 272 | 13.75 GiB | **22×** |
| **blockwise** | 64 | 1.88 s | 545 | 9.62 GiB | **45×** |
| **blockwise** | 128 | 0.99 s | 1,037 | 7.56 GiB | **86×** |
| **blockwise** | 256 | 0.53 s | 1,920 | 6.55 GiB | **160×** |
| **blockwise** | 512 | 0.31 s | 3,290 | 6.04 GiB | **274×** |

**Loss trajectory check** at the recommended block_size=64: same-batch training for 20 steps drives loss from 10.94 → 6.94 cleanly, no NaN, no instability. The chunk-as-update approximation trains effectively.

**Memory paradox**: peak VRAM DROPS as block_size grows. Larger blocks mean fewer iterations of the Python loop, so fewer intermediate gradient tensors are simultaneously alive in the autograd graph. block_size=32 has 32 blocks per chunk → 32 sets of saved-for-backward state.

**Recommended starting point at T=1024**:
```python
TitansConfig.gpt2_small(
    block_size=1024, chunk_size=1024,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
    nmm_block_size=64,        # ← 45× over sequential, 6 GiB headroom
    # If you want even MORE speed and can tolerate coarser approximation:
    # nmm_block_size=128 → 86× / 7.6 GiB
    # nmm_block_size=512 → 274× / 6.0 GiB
)
```

At `nmm_block_size=64`, a 50k-step training run goes from **~7 weeks (sequential)** to **~31 hours**. At `nmm_block_size=128`, ~14 hours. This is the breakthrough that makes real training on a consumer card viable.

### Full-model `torch.compile` (G277)

CLI flag (not a config field — runtime concern): `--compile-model`.

Wraps the full `TitansMAGGPT2` in `torch.compile(mode="default", dynamic=False)` after construction (and BEFORE DDP). Inductor traces the embedding + N transformer blocks + LN + LM head into one graph per shape. Composes with `nmm_compile_inner_loop` — the inner compile runs first, the outer compile traces around the compiled inner call.

**Expected impact**: 10-20% throughput on top of the inner-loop compile alone. The outer attention + MLP paths haven't been compiled separately before; full-model compile picks them up. Adds 1-3 minutes of warm-up compile time on the first training step.

**Composability**:
- `_unwrap` (in `model/__init__.py`) already strips `_orig_mod.` prefixes so checkpoint save/load survives compile wrapping.
- DDP is applied AFTER compile (model = DDP(torch.compile(model))).
- Dropout, gradient checkpointing, autocast all compose normally.

### 8-bit AdamW (G278)

CLI flag: `--optim8bit`. Requires `bitsandbytes` installed (`pip install bitsandbytes` or use the `optim8bit` optional dependency group in `pyproject.toml`).

Switches the optimizer from `torch.optim.AdamW` to `bnb.optim.AdamW8bit`. The 4-group layout (gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay) is preserved — bnb's AdamW8bit accepts the same param-groups API.

**Memory savings**: optimizer state (m, v moments) quantizes to 8-bit block-wise. Fp32 master weights are unaffected. At gpt2_small (~124M params), this drops optimizer memory from ~700 MB (fp32 state) to ~175 MB.

**Quality drift**: bitsandbytes' own benchmarks show <1% loss-curve drift vs fp32 AdamW at standard transformer training. We have NOT empirically validated this for TITANS' inner-loop training dynamics; the NMM's per-token surprise gradient has different magnitude statistics than standard transformer gradients. Treat as a memory-savings option; measure loss curves on your own data before relying on it for long runs.

### Int8 KV cache for decode (G279)

API-level option (not a config field — decode-time concern): `int8_kv_cache=True` argument to `prepare_decode`, `prepare_decode_chunked`, and the `generate()` function.

Quantizes the attention K and V caches to int8 with per-`(batch, head, token)` fp16 scales. The cache containers (`KVCacheInt8`) store int8 tensors + scale tensors; dequantization to bf16 happens only at the SDPA call. The new token's K and V are quantized on append.

**Memory savings**: at gpt2_small (n_head=12, head_dim=64), per cached token:
- bf16: 1.5 KB
- int8 with per-(B,h,t) scale: ~0.79 KB (~2× smaller)

For long-context generation (e.g., 8K tokens cached across 12 layers), this is ~7 MB → ~4 MB per layer of cache memory.

**Quality drift**: per-(B, head, token) scaling gives ~127 levels of resolution per row. For typical KV magnitudes the round-trip relative error is <1/127 per element. Empirically the logit drift over short decode runs is <5% relative (locked by test). For very long decode runs the noise can compound; pair with shorter generation horizons or per-element fp16 scales if you see drift.

**Composability**: dense (bf16) and int8 caches go through the same `forward_with_kv_cache` path; `isinstance(cache, KVCacheInt8)` branches the append + dequant logic. Plain GPT-2 blocks (no NMM) support it too.

### Fused Newton-Schulz via torch.compile (G274)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_compile_ns5` | `bool` | `False` | Routes NS5 calls through a module-level `torch.compile`-wrapped variant. The 5 NS5 iterations are 10 matmuls + 10 elementwise ops — without compile each is a separate CUDA kernel launch (~5-10 μs each → 50-100 μs of pure launch overhead per call). Compile collapses them into one graph. **No effect when `nmm_compile_inner_loop=True`** (which already wraps NS5 transitively). Most useful for the blockwise path (per-block NS5 calls), `per_token_ns5=True` (per-token NS5), and decode-time `step_with_conv()`. First call pays a 1-3 s warm-up. **At gpt2_small dims the win is ~5-10%** because matmul time dominates over launch overhead at H=3072. Larger at low_rank=64 and decode-time (~30-50%). |

### Int8 state (G275)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_state_dtype="int8"` | `str` | `"fp32"` | Third option alongside `"fp32"` and `"bf16"`. State dicts carry `key` (int8 tensor) plus `key_qs` (fp16 per-sample scale) companion entries. Per-sample, per-tensor symmetric quantization (scale = max_abs / 127). Dequantize on chunk entry, run blockwise math in fp32, requantize on chunk exit. **Blockwise path only** — sequential, scan v2, `step()`, `step_with_conv()` all raise `NotImplementedError` (those paths update state per-token; dequantize/requantize on every step would be both slow and noisy). Validation requires `nmm_block_size > 1`. Memory savings: ~2× smaller than bf16 for the state value tensors (the `_qs` scale companions are negligible). Quantization noise enters once per block and accumulates over many blocks — validate loss curves on your data when training at long T. |

### Truncated BPTT — detach state between blocks (G268)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_detach_state_between_blocks` | `bool` | `False` | When `True` AND `nmm_block_size > 1`, detaches `(M, S)` at every block boundary in the blockwise path. Backward graph spans ONE block instead of the whole chunk — peak transient memory drops roughly proportionally to `T / block_size`. Standard truncated-BPTT trade: outer params (k_proj, q_proj, v_proj, W_*, gamma_mem, NMM weight inits) only learn from within-block gradients; cross-block "remember earlier in chunk" signal via the recurrent (M, S) is cut. Validation: rejected at `block_size = 1` (would silently no-op). |

**When to enable**: training contexts longer than what one chunk can hold — e.g., 4K context split into 4 chunks of 1024. Without detach, backward through 4 chunks accumulates the full 4× per-chunk graph. With detach, only one chunk's worth of state is alive at a time. At single-chunk training (T = chunk_size = block_size) the effect is bounded by the number of blocks in a chunk.

### Lookahead value (G269)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_lookahead_value` | `bool` | `False` | When `True`, NMM's inner reconstruction loss uses `v_{t+1}` as the target for token `t` instead of `v_t` (predictive instead of reconstructive). The last token has no `v_{t+1}` within the chunk; its surprise contribution is dropped (M update is `(1-α)·M_{t-1}` decay only). Affects all three forward paths (blockwise, sequential, scan v2). Lucidrains' `store_with_lookahead_value` flag. Use when your downstream task benefits from a next-token-prediction-style inner loss instead of key→value reconstruction. |

### Per-parameter LR modulation (G270)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_per_param_lr_modulation` | `bool` | `False` | When `True`, `W_theta`'s output expands from 1 scalar per token to K scalars (one per state-key — 3 for full-rank, 6 for low-rank). Each weight matrix's update uses its OWN data-dependent θ. Tiny parameter overhead (~hundreds extra params at gpt2_small); meaningful expressivity gain when different state keys have different gradient scales. Affects all paths except scan v1 / scan v2 (those raise `NotImplementedError`). Lucidrains' `per_parameter_lr_modulation`. |

### Higher-order momentum (G272)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_momentum_order` | `int` | `1` | Order N of the momentum recurrence on S. Default 1 = paper-strict `S_t = η·S_{t-1} - θ·g`. With `N > 1`, N nested momenta are maintained with independently-learned η projections (W_eta gains N output dims): `S_k_t = η_k · S_k_{t-1} + S_{k-1}_t` for k=2..N. M update uses S_N (the smoothest). Per-step state grows linearly in N. Affects all paths except scan v1 / scan v2. Lucidrains' `momentum_order`. |

**State structure changes**: at `nmm_momentum_order > 1`, the per-layer NMM state's S field becomes a `tuple` of N dicts (one per momentum level) instead of a single dict. `init_state`, `reset_state`, `detach_states` all handle both shapes transparently — but callers reading state internals should branch on `isinstance(S, tuple)`.

### Per-head shared MemoryMLP (G271)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_per_head_learned_params` | `bool` | `True` | When `False` AND `nmm_n_heads > 1`, every head's `MemoryMLP` recurrent weights point at the SAME `nn.Module` instance; parameter count for the inner weights drops by ~n_heads×. Per-head (M, S) recurrent state remains independent at runtime; LayerNorm / out_scale / Q/K/V projections / update-param Linears remain head-private. Validation: rejected at `n_heads = 1` (nothing to share). Lucidrains' `per_head_learned_parameters`. |

### Soft norm clamping (G265)

| Field | Type | Default | Notes |
|---|---|---|---|
| `nmm_softclamp_max` | `Optional[float]` | `None` | Apply tanh-based soft norm clamping to per-token surprise gradients BEFORE Newton-Schulz. Smooth analog of `clip_grad_norm_` — no zero-gradient region, no threshold discontinuity. Off by default (paper-strict NS5 alone suffices). Useful with fresh-init or scan-path configs where gradient magnitude can spike. Lucidrains' `titans-pytorch` uses this as a default safety net; recommended values ~5.0 to 20.0. |

---

**Recommended starting point for T ≥ 256 training:** enable
`nmm_compile_inner_loop` and `nmm_fused_kernel`. Net cost: one-time
~30 s compile at first iteration, then ~1.9× faster forever.

```python
TitansConfig.gpt2_small(
    chunk_size=1024, block_size=1024,
    nmm_state_dtype="bf16",
    nmm_block_size=64,
    nmm_low_rank=64,
    nmm_compile_inner_loop=True,    # the big speedup
    nmm_fused_kernel=True,          # +5% on top, free correctness oracle
)
```

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
| `retrieval_from_M_prev` | `bool` | `True` (paper Eq. 15) | Paper Eq. 15: `y_t = M(q_t)` where M is M_{t-1} (read-then-write). Default `True` = paper-strict. Flip to `False` for lucidrains write-then-read (retrieve from freshly-updated M_t). Applies to all NMM forward paths (`step`, `step_with_conv`, `_forward_chunk_sequential`, `_forward_chunk_blockwise`). |
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
