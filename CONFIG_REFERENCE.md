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
| `chunk_size` | `int` | 1024 | 1 ≤ x ≤ `block_size` | TBPTT chunk length. `> block_size` → `ValueError` (would OOB `wpe`) |
| `nmm_grad_checkpoint` | `bool` | `False` | — | Rematerialize per-token updates on backward (saves memory, ~2× backward cost) |

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
| `WARMUP_STEPS` | `2000` | Thread explicitly into `apply_lr`, don't rely on defaults (G175) |
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
