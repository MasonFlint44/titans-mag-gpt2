# TITANS MAG + GPT-2 — Implementation Specification

This document is the authoritative specification for what the code in this
repository actually does. It is the single source of truth for the
implementation; sister documents in [`docs/`](docs/) (`ARCHITECTURE.md`,
`CONFIG_REFERENCE.md`, `RUNBOOK.md`, `GLOSSARY.md`, `PLAN.md`,
`TEST_PLAN.md`) describe design
rationale, knobs, operations, terminology, planning history, and tests
respectively. Where any of those disagree with this file, this file is
correct and the others should be updated.

Paper reference: Sun et al., *Titans: Learning to Memorize at Test Time*
(arXiv:2501.00663). All equation numbers below refer to that paper.

---

## 1. Scope and high-level architecture

Implements the **Memory as a Gate (MAG)** TITANS variant on top of GPT-2.
Every transformer block is augmented with a **Neural Memory Module (NMM)**
whose weights update online (test-time learning) via surprise-driven
gradient descent. Memory and attention outputs are combined per token by a
**MAG gate** before the standard MLP residual.

```
TitansMAGGPT2
├── wte   Embedding[vocab_size, d_model]
├── wpe   Embedding[block_size, d_model]
├── drop  Dropout(p)
├── blocks: ModuleList[N × TitansMAGBlock]
└── ln_f  LayerNorm(d_model)
        logits = ln_f(x) @ wte.weight.T          # tied LM head
```

```
TitansMAGBlock(config)
├── persistent_mem  Parameter[N_p, d_model]                # learned prefix
├── ln_1            LayerNorm(d_model)
├── attn            CausalSelfAttention(full causal; SWA optional)
├── ln_nmm          LayerNorm(d_model)                     # separate from ln_1
├── nmm             NeuralMemoryModule or MultiHeadNMM
├── gamma_mem       Parameter[d_model]                     # MAG scale, always
├── gamma_attn      Parameter[d_model]                     # only if finetune_mode=False
├── ln_2            LayerNorm(d_model)
└── mlp             GPT2MLP(d_model)                       # gelu approximate='tanh'
```

The block's `persistent_mem` is prepended to `x` before attention (paper
Eq. 19); the persistent prefix is sliced off the attention output before the
MAG gate. Whether the NMM also sees the prefix is governed by
`feed_persistent_to_nmm` (§5.4).

---

## 2. Neural Memory Module (NMM)

### 2.1 Components

The NMM is implemented in `model/nmm.py` as `NeuralMemoryModule(nn.Module)`.

| Component | Type / shape | Purpose |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` | `NMMProjection` = `Linear(d, d, bias=False)` then `CausalDepthwiseConv1d(d, kernel=k)` | Per-token query/key/value with local context |
| `W_theta`, `W_eta`, `W_alpha` | `Linear(d, 1, bias=False)` | Per-token data-dependent learning-rate / momentum-decay / forgetting-rate |
| `memory_mlp` | `MemoryMLP(d, expansion=E)` | The recurrent MLP; its weights ARE the meta-learned initial M |
| `out_scale` | `Parameter[d]` | Per-channel output scale (zeros if `finetune_mode=True`, else ones) |
| `per_sample_grad_fn` | `vmap(grad(inner_loss))` cached at `__init__` | Per-sample inner gradient |
| `_batched_retrieve` | `vmap(functional_call(memory_mlp, ...))` | Per-sample retrieval `f_M(q)` |

`MemoryMLP` is a SiLU-GLU gated two-layer MLP (paper depth `L_M = 2`) with a
ResidualNorm tail:

```
h = silu(W1·x) ⊙ sigmoid(W_gate·x)            # SiLU-GLU (NOT SwiGLU)
y = norm(W2·h) + x                              # ResidualNorm (LN + residual)
```

`W1, W_gate, W2` are 2D nn.Linear weights (no bias) initialized **Xavier
uniform**. Only these three matrices appear in the recurrent state — the
LayerNorm parameters are not recurrent (Newton-Schulz needs 2D matrices and
LN params would break the per-key uniform-shape invariant).

`NMMProjection` separates the linear and the causal depthwise conv so the
two can be reused independently at decode time (see §6). SiLU and L2-norm
are applied **at the call site**, not inside `NMMProjection.forward` —
putting them inside would yield `silu(silu(x))` at the call site.

`CausalDepthwiseConv1d`: depthwise `nn.Conv1d(d, d, kernel_size, groups=d,
padding=0)` with left-only `F.pad(x, (k-1, 0))`. Built-in conv padding would
be symmetric → leaks future tokens; we pad left manually.

### 2.2 State

The per-layer state is a tuple `(M, S)` where each is a dict keyed by
`"W1.weight"`, `"W_gate.weight"`, `"W2.weight"` with shapes
`[B, expansion·d, d]` (W1, W_gate) and `[B, d, expansion·d]` (W2). `S` is
the momentum buffer; same shapes as `M`.

`init_state(B, device)` builds `M` by replicating `memory_mlp.W*.weight`
across the batch (`.expand(B, -1, -1).to(device).clone()` — `.expand` is a
stride-0 view; `.clone()` materializes normal strides so vmap with
`in_dims=0` is well-defined). `S` is zeros.

`(M, S)` is **never saved in checkpoints**. It is per-sequence state, not
model state; resume re-initializes from `memory_mlp.W*.weight`. The
checkpoint contract is in §8.5.

### 2.3 Per-token update rule

Given input token vector `x_t ∈ ℝ^d` and prior state `(M_{t-1}, S_{t-1})`:

```
q̂_t = l2_norm(silu(conv(linear_q(x_t))))                         # retrieval query
k̂_t = l2_norm(silu(conv(linear_k(x_t))))                         # key
v_t  =        silu(conv(linear_v(x_t)))                           # value (not L2-normed)

ℓ_t  = ||f_{M_{t-1}}(k̂_t) - v_t||²_{red}                          # surprise (Eq. 12)
g_t  = ∇_{M_{t-1}} ℓ_t                                            # via torch.func.grad + vmap

g̃_t  = NewtonSchulz5(g_t)            if nmm_spectral_norm else g_t

θ_t  = sigmoid(W_θ · x_t)                                         # per-token LR
η_t  = sigmoid(W_η · x_t)                                         # per-token momentum decay
α_t  = sigmoid(W_α · x_t)                                         # per-token forgetting rate

S_t  = η_t · S_{t-1}        − θ_t · g̃_t                          # Eq. 14
M_t  = (1 − α_t) · M_{t-1}  + S_t                                 # Eq. 13

# Retrieval source (§5.4):
M_for_retrieval = M_{t-1} if retrieval_from_M_prev else M_t
y_t  = out_scale ⊙ f_{M_for_retrieval}(q̂_t)                       # Eq. 15 / Eq. 16
```

Ordering invariants:

1. **NS before θ.** Newton-Schulz divides by the Frobenius norm; pre-scaling
   by θ inside the loss would be cancelled exactly (`NS(θ·g) = NS(g)`). θ
   must be applied **after** NS, in the momentum update.
2. **Reduction is locked at construction.** `inner_loss` uses `'sum'` when
   `nmm_spectral_norm=True` (paper Eq. 12 in spirit; the d_model factor
   cancels after NS) and `'mean'` when False (otherwise gradients scale with
   d_model and the effective LR explodes). The reduction is baked into the
   cached `per_sample_grad_fn` at `__init__`; mutating
   `self.nmm_spectral_norm` after construction is a silent miscalibration
   and is detected by a runtime guard in every NMM forward path that raises
   `RuntimeError`.
3. **Retrieval source is config-driven.** Default `retrieval_from_M_prev=True`
   (paper Eq. 15, read-then-write). Set False for lucidrains-style
   write-then-read.

### 2.4 Newton-Schulz 5 (NS5)

`newton_schulz5(G, steps=5, eps=1e-7)` in `model/nmm.py` drives the
spectral norm of `G` toward 1 using the Jordan / Muon coefficients
`(a, b, c) = (3.4445, −4.7750, 2.0315)`.

Two correctness guards:

1. The whole iteration runs under `torch.amp.autocast(..., enabled=False)`
   and the input is cast to fp32 first. A bare `G.float()` under an ambient
   bf16 autocast is silently undone by autocast re-casting matmul inputs to
   bf16; the iteration then accumulates in bf16 and the fixed point ends up
   in [0.7, 1.4] instead of ~1.
2. NS converges on **wide** matrices (cols ≥ rows). Tall matrices (W1 /
   W_gate gradients with shape `[expansion·d, d]`) are transposed before
   the iteration and transposed back after. `W2 [d, expansion·d]` is already
   wide.

Output is cast back to `G`'s original dtype.

### 2.5 Per-sample gradient

```python
def inner_loss(params, k_hat, v):
    pred = functional_call(memory_mlp, params, k_hat)
    return F.mse_loss(pred, v, reduction=reduction)

per_sample_grad_fn = vmap(grad(inner_loss), in_dims=(0, 0, 0))
```

`argnums=0` (default of `grad`) → gradient is w.r.t. the params dict.
`functional_call` reads `self.memory_mlp` at call time, so device moves of
the NMM after `__init__` are still honored. `vmap` over the batch dim
gives per-sample gradients `{key: [B, ...]}` in one fused kernel call.

The function is **built once at `__init__`** and cached. Recreating
`vmap(grad(...))` per forward is measurably slower at T=512.

### 2.6 Single-token step (`step`)

Inference-time path. Pre-projects `[B, 1, d]` through Q/K/V (so the conv
sees a 1-token window with `k−1` zero-pad), then performs the per-token
update of §2.3.

**Do NOT call in a training loop.** With kernel_size=4, the conv only sees
a 1-token window per call (3 of 4 kernel weights masked by left-pad). The
training path is `_forward_chunk_sequential` which pre-projects the **full
chunk** so the conv sees up to k tokens of context for every output.

### 2.7 Decode-time step with conv buffer (`step_with_conv`)

Maintains an **explicit conv buffer** of the last `k−1` Linear projections
per Q/K/V so the depthwise conv at decode time sees a full k-token window
matching training-time behavior.

`init_conv_buffer_from_prompt(x_chunk)` seeds the buffer by re-projecting
the last `k−1` tokens of the warm-up input through `q_proj.linear /
k_proj.linear / v_proj.linear` (no conv, no activation). Returns
`{"q": [B, k-1, d], "k": [B, k-1, d], "v": [B, k-1, d]}`. Left-pads with
zeros if `T < k−1`.

`step_with_conv(x_t, state, conv_buffer)`:

```
q_lin, k_lin, v_lin = q_proj.linear(x), k_proj.linear(x), v_proj.linear(x)  # [B, 1, d] each
q_full = cat([buffer.q, q_lin], dim=1)                                       # [B, k, d]
q_conv = q_proj.conv(q_full)[:, -1, :]                                       # [B, d]
# same for k, v
... [SiLU + L2 + update rule as §2.3]
new_buffer = {q: cat([buffer.q[:, 1:], q_lin]), ...}
return y_t, (M_t, S_t), new_buffer
```

The conv buffer is **not** part of `(M, S)`. NS operates on 2D matrices; a
`[k−1, d]` buffer would break the per-key uniform-shape invariant. The
buffer is a separate per-block decode cache (§6).

### 2.8 Chunked forward — sequential path

`_forward_chunk_sequential(x_chunk, state_in, doc_boundaries)`:

1. Full-chunk pre-projection: `k_proj(x)`, `q_proj(x)`, `v_proj(x)` over
   the whole `[B, T, d]` chunk so the conv sees up to k tokens for every
   output position. Compute SiLU + L2 (q, k) and SiLU (v).
2. Compute `θ_chunk`, `η_chunk`, `α_chunk` for the whole chunk in one shot.
3. Build a CPU-side per-position "any boundary?" mask once with
   `doc_boundaries.any(dim=0).cpu().tolist()` to avoid T implicit
   GPU↔CPU syncs from `tensor.any()` inside a Python `if`.
4. Loop `t = 0..T−1`:
   - If `any_boundary_per_t[t]`: call `reset_state((M, S), doc_boundaries[:, t], init_M)`.
     `init_M` is **lazy-built** the first time a boundary fires (zero-cost
     on the common no-boundary chunk).
   - Capture `M_prev = M` (needed if `retrieval_from_M_prev`).
   - Compute `g_t`, NS, update `S` and `M` per §2.3.
   - Retrieve from `M_prev` or `M` per `retrieval_from_M_prev`.
5. Stack `y_list` into `[B, T, d]` and return `(y_chunk, (M, S))`.

`reset_state(state, mask, init_M)` uses `torch.where` (NOT in-place
assignment): in-place index assignment on tensors in the autograd graph
raises `RuntimeError`. `where` is non-mutating and differentiable.

### 2.9 Chunked forward — associative-scan path

`_forward_chunk_scan(x_chunk, state_in, doc_boundaries)`. Available iff
`torch.associative_scan` (or `torch._higher_order_ops.associative_scan` on
older 2.x) resolves; the import is wrapped in `try/except`.

**Approximation.** All per-token gradients are computed against the
chunk-start `M_0` (not against `M_{t-1}`). This breaks the recurrence's
true sequential dependency in exchange for parallelism. The scan path is
inference-only unless the model is wrapped in `torch.compile` (without it,
`associative_scan` lacks autograd and silently zeros NMM gradients —
G164/G180).

Update recurrence is decomposed into two scans of the form
`x_t = decay_t · x_{t−1} + delta_t`:

```
S_t = η_t · S_{t-1} + (−θ_t · g̃_t)    # decay = η, delta = -θ·g̃
M_t = (1 − α_t) · M_{t-1} + S_t        # decay = 1−α, delta = S
```

Associative operator: `(a₁, d₁) ⊗ (a₂, d₂) = (a₂·a₁, a₂·d₁ + d₂)`.

To incorporate the initial `S_0` / `M_0`, each scan prepends a synthetic
`(1, X_0)` element and slices `[1:]` off the result. For paper-Eq.-15
retrieval ordering (read `M_{t-1}` at position t), the retrieval source is
constructed by prepending `M_state[k]` to `M_chunk[k]` and dropping the
last entry. Retrieval is `vmap(_batched_retrieve, in_dims=(0, 1))` over T.

### 2.10 Dispatcher (`forward_chunk`)

`forward_chunk(x_chunk, state_in, doc_boundaries)` decides scan vs.
sequential:

```
can_scan = _HAS_ASSOC_SCAN and (doc_boundaries is None or not doc_boundaries.any())
if torch.is_grad_enabled() and not getattr(self, "_allow_scan_training", False):
    can_scan = False
```

`torch.is_grad_enabled()` is used **instead of `self.training`** (G164).
The two are independent: a forgotten `model.train()` after a probe
`model.eval()` leaves `self.training=False` during a training loop with
autograd on. Probing autograd directly is robust to that.

Opt into the scan path during compiled training via
`allow_scan_training(model, True)`.

### 2.11 Multi-head NMM

`MultiHeadNMM(n_embd, n_heads, ...)` instantiates `n_heads` parallel
`NeuralMemoryModule` instances on `head_dim = n_embd // n_heads`. Same API
surface as `NeuralMemoryModule`: `init_state`, `forward_chunk`,
`init_conv_buffer_from_prompt`, `step_with_conv`, `step`. State per layer
is a `list[n_heads]` of `(M_h, S_h)` tuples instead of a single tuple.

`memory_mlp` property returns `heads[0].memory_mlp` for shape / dtype
consumers (`init_conv_buffer_from_prompt` dtype inference,
`_apply_gpt2_init`'s NMM-skip-by-id).

The block constructs `MultiHeadNMM` only when `nmm_n_heads > 1`. Default 1
uses `NeuralMemoryModule` directly with no wrapper.

`detach_states` and `compute_nmm_norm` are recursive via
`isinstance(layer_state, list)` to handle the nested structure.

Multi-head NMM is **not in the paper proper** — it is a lucidrains
enhancement exposed for ablation.

---

## 3. CausalSelfAttention

Standard GPT-2 multi-head self-attention. Q/K/V projections are **split**
(`q_proj`, `k_proj`, `v_proj` as separate `nn.Linear(d, d, bias=True)`),
not fused into a single `c_attn`, so HF weight loading can copy each
independently. Output projection is `proj` (also bias=True). All four
biases are required for HF parity.

`forward(x, mask)`: standard SDPA. The mask is passed in by the caller —
this module does **not** impose causality on its own. Caller (the block)
constructs the `[N_p+T, N_p+T]` block-structured mask via `_aug_mask`.

`forward(x, mask)` uses `F.scaled_dot_product_attention(q, k, v,
attn_mask=mask, dropout_p=self.resid_dropout.p if self.training else 0)`
and applies the residual dropout once at the projection output.

`project_kv(x)` returns `(K, V)` shaped `[B, n_head, T, head_dim]` without
computing attention. Used during decode warm-up to seed the KV cache from
`ln_1(x_aug)` over the prompt.

`forward_with_kv_cache(x_new, k_cache, v_cache, swa_window, n_persistent)`
projects the new token's Q/K/V (T=1), appends K/V to the cache, runs SDPA
of Q against the full cached K, V, returns `(y, k_full, v_full)`. When
`swa_window` is set, builds a single-row additive mask that opens on
persistent positions `[0, n_persistent)` and on the most recent
`swa_window` real positions; -inf elsewhere. Without this branch, decode
under a model trained with SWA would silently attend over the full
history (G243-adjacent invariant).

---

## 4. MAG Block

### 4.1 Block-structured causal mask (`_aug_mask`)

`_aug_mask(T, dtype)` builds the additive `[N_p+T, N_p+T]` mask used by
attention. Block structure:

| | Persistent (col) | Real (col) |
|---|---|---|
| **Persistent (row)** | 0 (open, bidirectional among themselves) | −∞ (masked) |
| **Real (row)** | 0 (open, always visible) | upper-triangular causal (banded if `use_swa`) |

When `use_swa=True`, the real-to-real block also masks positions `≥
swa_window` steps in the past (`torch.tril(..., diagonal=-swa_window)`).
Persistent prefix stays fully visible per paper Fig. 3b.

`dtype` matches `x.dtype` to avoid an implicit cast inside SDPA under
autocast.

### 4.2 Forward

```python
def forward(x, nmm_state, doc_boundaries=None):
    B, T, _ = x.shape
    x_aug = cat([persistent_mem.expand(B, -1, -1), x], dim=1)             # [B, N_p+T, d]
    y_attn = attn(ln_1(x_aug), mask=_aug_mask(T, x.dtype))[:, N_p:, :]    # [B, T, d]

    if feed_persistent_to_nmm:                                            # paper Eq. 28 (default)
        db_aug = cat([zeros(B, N_p, bool), doc_boundaries], dim=1) if doc_boundaries is not None else None
        y_mem_full, nmm_state = nmm.forward_chunk(ln_nmm(x_aug), nmm_state, db_aug)
        y_mem = y_mem_full[:, N_p:, :]
    else:                                                                 # lucidrains-flavored
        y_mem, nmm_state = nmm.forward_chunk(ln_nmm(x), nmm_state, doc_boundaries)

    if finetune_mode:
        o = y_attn + silu(gamma_mem * y_mem) * y_attn       # = y_attn * (1 + silu(γ_m · y_mem))
    else:
        o = silu(gamma_attn * y_attn) * silu(gamma_mem * y_mem)

    x = x + o
    x = x + mlp(ln_2(x))
    return x, nmm_state
```

Two MAG gate variants:

- **`finetune_mode=True` (default).** `o = y_attn · (1 + silu(γ_m · y_mem))`.
  At init `out_scale=0 ⇒ y_mem=0 ⇒ silu(0)=0 ⇒ o = y_attn` exactly,
  preserving the pretrained GPT-2 residual stream so logits match vanilla
  GPT-2 at step 0. `gamma_attn` is **not created** in this mode (otherwise
  it would leak unused params into the state_dict and break finetune↔scratch
  checkpoint interchange).
- **`finetune_mode=False`.** Pure paper formula
  `o = silu(γ_a · y_attn) · silu(γ_m · y_mem)`. Both `gamma_attn` and
  `gamma_mem` exist, both init to ones.

### 4.3 Block decode methods

`init_decode_cache(x_prompt, nmm_state)`: called once per block AFTER
`forward` has run on the prompt. Captures
`(k_cache, v_cache, nmm_conv_buffer)`:

- `k_cache, v_cache` come from `attn.project_kv(ln_1(x_aug_prompt))`;
  length `N_p + T_prompt`, includes the persistent prefix.
- `nmm_conv_buffer` is `nmm.init_conv_buffer_from_prompt(ln_nmm(x_for_nmm))`
  where `x_for_nmm` is `x_aug` if `feed_persistent_to_nmm` else `x_prompt`.
  This matches whatever the warm-up's conv actually saw, so decode-time conv
  sees the same `k`-token window across the persistent→real boundary.

`forward_step(x_new, nmm_state, k_cache, v_cache, nmm_conv_buffer)`:

- `x_new`: `[B, 1, d]` — already embedded (wte + wpe at the new position).
- Attention via `attn.forward_with_kv_cache(ln_1(x_new), k_cache, v_cache,
  swa_window=swa_window if use_swa else None, n_persistent=N_p)`. No
  `x_aug` concat — the persistent prefix is already in the cache.
- NMM via `nmm.step_with_conv(ln_nmm(x_new).squeeze(1), nmm_state,
  nmm_conv_buffer)`: one update per token, full k-token conv context.
- Same MAG gate variants as §4.2.
- Returns `(x_out, new_nmm_state, new_k_cache, new_v_cache, new_conv_buf)`.

---

## 5. Configuration

`TitansConfig` is a `@dataclass` in `config.py`. Full field-by-field
reference is in [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md);
this section codifies the
**invariants the implementation depends on**.

### 5.1 Validation (`__post_init__`)

All checks use `raise ValueError`, not `assert`. `python -O` strips
asserts, which would let invalid configs ship silently in production.

| Check | Behavior |
|---|---|
| `chunk_size > block_size` | `ValueError` |
| `nmm_n_persistent < 0` | `ValueError` |
| `nmm_expansion < 1` | `ValueError` |
| `n_embd % n_head != 0` | `ValueError` |
| `use_swa and swa_window < 1` | `ValueError` (empty window → softmax NaN) |
| `nmm_n_heads < 1` | `ValueError` |
| `n_embd % nmm_n_heads != 0` | `ValueError` |
| `not finetune_mode and chunk_size < block_size` | `warnings.warn` — wpe rows beyond `chunk_size` will never train (G163) |

`CausalSelfAttention.__init__` re-checks `n_embd % n_head` as defense in
depth.

### 5.2 Factory presets

```python
TitansConfig.gpt2_small()   # n_layer=12, n_head=12, n_embd=768
TitansConfig.gpt2_medium()  # n_layer=24, n_head=16, n_embd=1024
TitansConfig.gpt2_large()   # n_layer=36, n_head=20, n_embd=1280
TitansConfig.gpt2_xl()      # n_layer=48, n_head=25, n_embd=1600
```

All factories accept `**overrides` and merge via `**{**defaults,
**overrides}` so callers can override backbone dims without hitting
`TypeError: multiple values for keyword argument`.

`load_pretrained` selects the HF checkpoint by `config.n_embd` via the
`_HF_GPT2_NAMES` table (768→`gpt2`, 1024→`gpt2-medium`, 1280→`gpt2-large`,
1600→`gpt2-xl`).

### 5.3 NMM hyperparameter contract

| Field | Default | Constraint |
|---|---|---|
| `nmm_depth` | 2 | **Documentation only.** `MemoryMLP` hard-codes L_M=2 (W1+W_gate→W2). Changing this field has no effect; to change depth, modify `MemoryMLP.__init__`. |
| `nmm_expansion` | 4 | Hidden dim = `expansion · d_model` |
| `nmm_conv_kernel` | 4 | Kernel size for all three NMMProjections (`q_proj.conv`, `k_proj.conv`, `v_proj.conv`) |
| `nmm_spectral_norm` | `True` | **Construction-time only.** Inner-loop reduction (`sum`/`mean`) is locked into `per_sample_grad_fn` at `__init__`; mutating after construction raises `RuntimeError` on the next NMM forward (any path). |
| `nmm_n_persistent` | 4 | `N_p`; 0 disables persistent tokens |
| `chunk_size` | 512 | `1 ≤ chunk_size ≤ block_size` |

`nmm_spectral_norm` + reduction coupling:

| `nmm_spectral_norm` | Inner-loss reduction | Why |
|---|---|---|
| `True` | `'sum'` | NS divides by Frobenius — sum is natural |
| `False` | `'mean'` | Without NS, sum gives gradients ~d_model× too large → silent divergence |

### 5.4 Paper-vs-lucidrains flags (G254 / G255)

These flags expose deliberate paper/lucidrains divergences as runtime
config. **Defaults prefer the paper** (G255 default flip).

| Field | Default | Paper-strict (default) | Flip-to-`False` (lucidrains-flavored) |
|---|---|---|---|
| `retrieval_from_M_prev` | `True` | Paper Eq. 15: `y_t = M(q_t)` where `M = M_{t-1}` (read-then-write). Applies in `step`, `step_with_conv`, `_forward_chunk_sequential`, and `_forward_chunk_scan`. | Write-then-read: retrieve from freshly-updated `M_t`. |
| `feed_persistent_to_nmm` | `True` | Paper Eq. 28: `M(x̃)` where `x̃ = concat(persistent, x)`. Block feeds `ln_nmm(x_aug)` to NMM, augments `doc_boundaries` with a False prefix of length `N_p` (persistent positions never trigger resets), and slices `N_p` positions off `y_mem` before the residual. | NMM sees `ln_nmm(x)` only (real tokens); `doc_boundaries` is passed verbatim. Slightly lower memory; updates only on real tokens. |
| `nmm_n_heads` | `1` | Paper single-head (implicit). | `>1` instantiates `MultiHeadNMM` (lucidrains enhancement). Must divide `n_embd`. |

At `finetune_mode=True` with `out_scale=0`, the NMM contributes 0 at init
regardless of these flags, so HF parity at init is unaffected.

### 5.5 Attention / mode flags

| Field | Default | Notes |
|---|---|---|
| `use_swa` | `False` | Sliding Window Attention. **Off by default** for GPT-2 fine-tune (pretrained with full attn). |
| `swa_window` | 256 | Field name is `swa_window`, NOT `window_size` (G126). |
| `finetune_mode` | `True` | Controls MAG gate variant + `out_scale` init + presence of `gamma_attn`. |

---

## 6. Decode pipeline

Cached single-token decoding lives in `TitansMAGGPT2`:

```
prepare_decode(prompt_idx, initial_nmm_states=None) -> cache
prepare_decode_chunked(prompt_idx) -> cache          # any prompt length
forward_step(token_id, cache) -> (logits, new_cache)
```

Cache structure:

```python
{
    "last_logits":      Tensor[B, 1, V],         # logits at last prompt token
    "nmm_states":       list[n_layer] of state,  # post-warmup NMM state per layer
    "kv_caches":        list[n_layer] of (K, V), # each [B, n_head, N_p+P, head_dim]
    "nmm_conv_buffers": list[n_layer] of buf,    # each {"q","k","v"}: [B, k-1, d]
    "position":         int,                     # = prompt length P (next decode pos)
}
```

### 6.1 `prepare_decode`

1. **Eval-mode contract** (G243). Raises `RuntimeError` if `self.training`.
   In train mode with dropout > 0, `forward_with_kv_cache` skips
   resid_dropout / SDPA dropout while the warm-up block forward applies
   both. The two paths' attention outputs would diverge silently and break
   the decode-vs-full-forward parity invariant.
2. Reject `prompt_idx.size(1) > block_size` (would OOB the wpe table).
3. Embed `wte(prompt_idx) + wpe(0..P-1)`; build `nmm_states` from
   `initial_nmm_states` or `init_state(B, device)` per block.
4. For each block, in order: capture `(k_cache, v_cache, conv_buf)` via
   `block.init_decode_cache(x, nmm_state)` **BEFORE** the block mutates
   `x`, then run `x, nmm_state = block(x, nmm_state, None)`.
5. Compute `last_logits = ln_f(x)[..., -1:, :] @ wte.weight.T`.
6. Return the cache dict with `position = P`.

`initial_nmm_states` (when not None) is validated up front for
(a) length matching `n_layer` and (b) leading batch dim matching the
prompt; for multi-head states the check digs into `first_layer[0][0]`.

### 6.2 `prepare_decode_chunked`

Encapsulates the chunked-warm-up + tail-`prepare_decode` pipeline so
`generate.py`, `eval.needle_in_haystack`, and behavior tests don't each
re-implement it (G249).

- `prompt_len ≤ block_size`: equivalent to `prepare_decode(prompt_idx)`.
- `prompt_len > block_size`: chunks the prefix through `forward()` in
  `block_size` slices so the NMM accumulates state across the full prompt,
  then calls `prepare_decode(tail, initial_nmm_states=nmm_states)` on the
  last `block_size` tokens. The returned `cache["position"]` lands at
  `block_size`; callers can sample AT MOST ONE token from
  `cache["last_logits"]` (any `forward_step` call would wpe-OOB).

Same eval-mode contract as `prepare_decode`.

### 6.3 `forward_step`

1. Eval-mode contract (G243).
2. Reject `cache["position"] >= block_size` (wpe OOB).
3. Embed `wte(token_id) + wpe([position])` → `[B, 1, d]`.
4. For each block, in order: `block.forward_step(x, nmm_state, k_cache,
   v_cache, conv_buf)` advances all four caches by one token.
5. Compute `logits = ln_f(x) @ wte.weight.T` → `[B, 1, V]`.
6. Return `(logits, new_cache)` with `new_cache["position"] = pos + 1`.

### 6.4 `generate(prompt, max_new_tokens, temperature, top_k, tokenizer)`

Wraps the cache pipeline (`generate.py`). Sampling order is
**temperature → top-k mask → softmax → multinomial** (G173). `temperature
≤ 0` collapses to argmax. Mode is captured-and-restored via `try/finally`
(G161).

`max_new_tokens` is capped against block_size:

- Short prompt (`prompt_len ≤ block_size`): `max_new = min(max_new_tokens,
  block_size − prompt_len + 1)`. The `+1` accounts for the first sampled
  token coming from `cache["last_logits"]` (no `forward_step` call → no
  wpe lookup at a new position).
- Long prompt (`prompt_len > block_size`): `max_new = 1` if
  `max_new_tokens >= 1` else 0.

The last iteration skips the trailing `forward_step` (we already have the
sampled token).

---

## 7. HF GPT-2 weight loading (`scripts/load_pretrained.py`)

`load_pretrained(model, config)` overwrites the GPT-2 backbone weights
from an HF checkpoint **without touching any NMM parameters**.

HF `Conv1D(out, in)` stores weight as `[in, out]` — the transpose of
`nn.Linear`. Every Conv1D weight (`c_attn`, `c_proj`, `c_fc`,
`mlp.c_proj`) must be transposed on copy. LayerNorm and Embedding layouts
match (no transpose).

Per block:

```
HF c_attn.weight  [n_embd, 3*n_embd]  ──split dim=1──> W_q, W_k, W_v  ──.T──> our q/k/v_proj.weight
HF c_attn.bias    [3*n_embd]          ──split dim=0──> b_q, b_k, b_v           our q/k/v_proj.bias
HF c_proj.weight                                       ──.T──>               our attn.proj.weight
HF ln_1.weight/bias                                                          our ln_1.weight/bias
HF ln_2.weight/bias                                                          our ln_2.weight/bias
HF mlp.c_fc.weight, mlp.c_proj.weight                  ──.T──>               our mlp.c_fc.weight, c_proj.weight
```

Plus `wte`, `wpe`, `ln_f` at the top level. HF's `n_layer` and `n_head`
are checked against `config` up front; mismatches raise `ValueError`
loudly (otherwise `zip(self.blocks, hf_model.transformer.h)` silently
truncates).

**HARD GATE** (Phase 2.6): with `N_p=0` and `out_scale=0`, the loaded
model must produce logits within `1e-4` of HF GPT-2 on the same inputs.
This invariant is preserved by `nmm_spectral_norm`, `nmm_n_heads`,
`feed_persistent_to_nmm`, and `retrieval_from_M_prev` because at
`out_scale=0` the NMM contributes exactly zero.

---

## 8. Training contract

### 8.1 Optimizer (4 groups)

`build_optimizer(model, ...)` in `train.py` routes every parameter into
exactly one of four AdamW param groups:

| Group | LR | weight_decay | Members |
|---|---|---|---|
| `gpt2_decay` | `BASE_LR_GPT2 = 3e-4` | 0.1 | GPT-2 backbone weights |
| `gpt2_no_decay` | `BASE_LR_GPT2` | 0.0 | GPT-2 biases / LayerNorm |
| `nmm_decay` | `BASE_LR_NMM = 9e-4` (= 3× GPT-2 per paper) | 0.1 | NMM weights |
| `nmm_no_decay` | `BASE_LR_NMM` | 0.0 | NMM biases / norms / `out_scale` / `gamma_*` / `persistent_mem` |

Routing substrings:

```python
NO_DECAY_SUBSTRINGS = ("bias", "ln", "norm", "out_scale", "gamma", "persistent")
NMM_SUBSTRINGS      = ("nmm", "gamma", "persistent", "ln_nmm")
```

`out_scale`, `gamma_*`, and `persistent_mem` are intentionally no-decay —
decay would shrink them toward zero, suppressing the memory branch and
prefix capacity.

`build_optimizer` asserts `sum(len(g['params']) for g in groups) ==
len(list(model.parameters()))` so no param is missing or double-counted.

AdamW betas are `(0.9, 0.95)` — **not** PyTorch default `(0.9, 0.999)`
(G153). `eps=1e-8`.

### 8.2 LR schedule (`apply_lr`)

Linear warmup to peak over `warmup_steps`, then cosine decay to
`min_ratio · peak` over `(max_steps − warmup_steps)`.

`warmup_steps` and `max_steps` are **required positional args** (no
defaults) so callers cannot silently inherit a 1k/100k schedule on a
200-step overfit (G175). `max_steps <= warmup_steps` raises `ValueError`
(G197).

`base_lrs` MUST come from code-level constants (use
`base_lrs_from_constants()`), never from
`optimizer.param_groups[i]['lr']` after `load_state_dict` — that captures
the mid-cosine deflated value and compounds the deflation every resume
(G162).

### 8.3 TBPTT step (`train_step` / `run_training`)

Per micro-batch:

1. `detach_states(nmm_states)` — bounds the autograd graph (no-op if None).
2. Forward (optionally in `torch.autocast(dtype=bf16)` on CUDA):
   `logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)`.
3. Loss: `F.cross_entropy(logits[:, :-1].reshape(-1, V),
   input_ids[:, 1:].reshape(-1)) / accum_steps`.
4. `loss.backward()`.

Per accumulation cycle (every `accum_steps` micro-batches):

1. `apply_lr(...)`.
2. `grad_norm = clip_grad_norm_(model.parameters(), GRAD_CLIP=1.0)`.
   Always run **outside autocast in fp32** (G159); clip inside bf16
   computes the norm in low precision and defeats clipping.
3. If `torch.isfinite(grad_norm)`: `optimizer.step()`.
   Else: **NaN-skip** — `optimizer.zero_grad(set_to_none=True)` and reset
   `nmm_states = None` (G158/G213/G217). Returning the existing state
   would propagate a NaN-tainted M through the next forward and livelock
   until the next document boundary.
4. `optimizer.zero_grad(set_to_none=True)`.

### 8.4 DDP semantics

- `is_distributed=True`: caller wraps `model` with `DDP`. All but the last
  micro-batch in an accumulation cycle run inside `model.no_sync()` to
  suppress per-microbatch all-reduce; the last micro-batch triggers the
  all-reduce (G200).
- **Partial-cycle guard** (`is_partial_cycle(batch, accum_i)`): when
  `batch is None and accum_i > 0`, the loader exhausted mid-cycle. Under
  DDP, the per-rank `.grad` buffers were never AllReduce'd; stepping
  would diverge ranks permanently. Discard the cycle's accumulated grads
  (`optimizer.zero_grad`), reset `nmm_states = None`, and return. Single-
  GPU treats partial as complete (G214/G222).
- **Checkpoint barrier**: rank 0 owns the write; all ranks `dist.barrier()`
  afterwards so non-rank-0 doesn't race into the next iteration while
  rank 0 is still flushing to disk (G199).
- **NCCL teardown**: `dist.destroy_process_group()` in the entry-point
  script's `try/finally` (G225/G227).
- **Per-rank seed**: all ranks build identical params from the seed, then
  re-seed `seed + rank` after model construction so dropout masks diverge
  (G204).

### 8.5 Checkpoints

`save_checkpoint(path, model, optimizer, step, config)`:

```python
torch.save({
    "state_dict": _unwrap(model).state_dict(),
    "optimizer":  optimizer.state_dict(),
    "step":       step,
    "config":     dataclasses.asdict(config),
}, path)
```

- `_unwrap(model)` strips `torch.compile` (`_orig_mod`) and `DDP/FSDP`
  (`module`) prefixes so the saved state_dict is portable across
  wrapping choices on resume (G184/G186/G195).
- `config` is **required** (not optional). `finetune_mode` controls block
  topology (`gamma_attn` presence, `out_scale` init); resume needs it to
  rebuild the same structure.
- **NMM `(M, S)` states are intentionally NOT saved.** They are
  per-sequence accumulators; resume re-initializes from
  `memory_mlp.W*.weight`.

`load_checkpoint(path, device)`: `torch.load(path,
weights_only=False)` (G168 — PyTorch 2.6+ flipped the default; our nested
optimizer state would be rejected at `weights_only=True`). Caller rebuilds
the model from `ckpt["config"]`, then `model.load_state_dict`, then
optimizer construct + (optional) `optimizer.load_state_dict`.

### 8.6 Data loader contract

`ParallelStreamLoader(token_stream, batch_size B, chunk_size T, eot_id,
rank, world_size)` yields `(idx [B, T], doc_boundaries [B, T])` such
that position-i is contiguous across consecutive batches in the same
sub-stream. Each rank reads a contiguous segment of the corpus, rounded
down to a multiple of `B · T`.

`doc_boundaries[:, t]` is True iff `streams[:, t-1] == eot_id`; position
0 of every rank's segment is also True (no cross-rank or pre-corpus NMM
continuity to preserve).

A naive `DataLoader(shuffle=False, batch_size=B)` does NOT give this
property: it would collate chunks [0..B-1] into batch 0, [B..2B-1] into
batch 1, ..., and the carried `nmm_states[i]` would jump over B-1 chunks
between batches — silent cross-document state corruption (G151).

---

## 9. Model-level init (`_apply_gpt2_init`)

Runs once at `TitansMAGGPT2.__init__` after blocks are constructed. In
fine-tune mode this is overwritten by `load_pretrained`; running it
unconditionally keeps construction deterministic.

```
nn.Linear         std = 0.02
nn.Linear (.proj / .c_proj — residual output projections)
                  std = 0.02 / sqrt(2 * n_layer)
nn.Embedding      std = 0.02
LayerNorm         left at default (weight=1, bias=0) by isinstance filter
```

**NMM-internal modules are skipped by IDENTITY** (not name substring): an
explicit set of `id(submodule)` is built by walking every
`NeuralMemoryModule` in the model. A substring check would silently miss
NMM submodules if a future refactor renamed `self.nmm` to `self.memory`.

The import is relative (`from .nmm import ...`) so it survives a top-level
package rename.

---

## 10. Forward of `TitansMAGGPT2`

```python
def forward(idx, nmm_states=None, doc_boundaries=None):
    B, T = idx.shape
    pos = torch.arange(0, T, device=idx.device)
    x = drop(wte(idx) + wpe(pos))                      # NOTE: positions restart at 0 each call
    if nmm_states is None:
        nmm_states = [b.nmm.init_state(B, idx.device) for b in blocks]
    new_states = []
    for block, ns in zip(blocks, nmm_states):
        x, ns = block(x, ns, doc_boundaries)
        new_states.append(ns)
    x = ln_f(x)
    logits = x @ wte.weight.T                          # tied LM head
    return logits, new_states
```

Position embeddings restart at 0 each `forward` call. Cross-chunk context
lives in the NMM state, not in attention. This is what `chunk_size ≤
block_size` enforces.

---

## 11. Invariants the implementation upholds

1. **HF parity at init** (`finetune_mode=True`, `out_scale=0`, `N_p=0`):
   max logit diff vs HF GPT-2 < 1e-4 across `n_embd ∈ {768, 1024, 1280,
   1600}`. Unaffected by `nmm_spectral_norm`, `nmm_n_heads`,
   `feed_persistent_to_nmm`, `retrieval_from_M_prev`.
2. **No mid-NMM-forward state-shape change.** `nmm_spectral_norm` is
   construction-time only; mutating it after `__init__` raises
   `RuntimeError` in every NMM forward path.
3. **No in-place state mutation on autograd tensors.** `reset_state` uses
   `torch.where`; state dict updates rebuild new dicts.
4. **NMM state is never saved in checkpoints.**
5. **TBPTT autograd graph is bounded.** `detach_states` between chunks
   is mandatory (handled by `train_step` / `run_training`).
6. **NaN-tainted state is dropped.** NaN-skip path zeroes grads and resets
   `nmm_states = None`; the next forward re-inits.
7. **DDP partial cycles never step.** `is_partial_cycle` discards the
   cycle and ends training rather than risking rank divergence.
8. **Decode parity.** `prepare_decode` / `forward_step` produce the same
   logits as a single full `forward(prompt + [token])` to within float
   precision; preserved by the eval-mode contract (G243).
9. **Long-prompt decode is bounded.** `prepare_decode_chunked` caps
   `forward_step` calls to `block_size − P + 1` (short prompt) or 1
   (long prompt — wpe OOB otherwise).
10. **Optimizer 4-group coverage.** Every param lands in exactly one
    group; `build_optimizer` asserts at construction.

---

## 12. File structure

```
titans-mag-gpt2/
├── README.md                Quickstart, install, headlines.
├── SPEC.md                  This document.
├── LICENSE                  MIT.
├── docs/
│   ├── ARCHITECTURE.md          Design decisions, equations, block diagram.
│   ├── PLAN.md                  Phase-by-phase implementation guide.
│   ├── CONFIG_REFERENCE.md      Every config knob with range / defaults.
│   ├── TEST_PLAN.md             Unit / integration / parity / DDP test plan.
│   ├── RUNBOOK.md               What to do when training breaks.
│   ├── GLOSSARY.md              TITANS terminology.
│   ├── EXPERIMENTS.md           Ablation plan and success criteria.
│   ├── GAP_HISTORY.md           Audit log (background reading).
│   ├── ROADMAP.md               Phase-by-phase delivery plan.
│   └── IMPLEMENTATION_PROMPT.md One-shot bootstrap prompt for fresh agents.
├── config.py                TitansConfig dataclass + factories.
├── model/
│   ├── __init__.py          _unwrap helper.
│   ├── nmm.py               NeuralMemoryModule, MultiHeadNMM, NS5, helpers.
│   ├── block.py             TitansMAGBlock, CausalSelfAttention, GPT2MLP.
│   └── titans_gpt2.py       TitansMAGGPT2 + decode pipeline.
├── data/
│   ├── tokenizer.py         tiktoken wrapper.
│   ├── dataset.py           Document stream helpers.
│   └── dataloader.py        ParallelStreamLoader.
├── scripts/
│   ├── load_pretrained.py   HF GPT-2 weight loader (Conv1D → Linear transpose).
│   └── finetune.py          Fine-tune entry point.
├── train.py                 build_optimizer, train_step, run_training, schedule, ckpts.
├── generate.py              Cached autoregressive sampling.
├── eval.py                  Perplexity + needle-in-haystack.
├── tests/                   See docs/TEST_PLAN.md.
└── diagrams/                Mermaid diagrams.
```

---

## 13. References

- Sun et al. (2025). *Titans: Learning to Memorize at Test Time.*
  arXiv:2501.00663 — primary reference for NMM, MAG, and all architectural
  decisions.
- Radford et al. (2019). *Language Models are Unsupervised Multitask
  Learners.* (GPT-2.)
- Di Nepi et al. (2025). *Titans Revisited.* arXiv:2510.09551 — frozen
  backbone + NMM training fails; persistent tokens alone ineffective.
- Jordan et al. *Muon optimizer / nanogpt* — Newton-Schulz coefficients
  `(3.4445, −4.7750, 2.0315)` and the wide-matrix convergence regime.
- `lucidrains/titans-pytorch` (GitHub) — reference implementation;
  spectral norm, `torch.func.grad`, ResidualNorm, and multi-head NMM are
  additions beyond the paper that improve stability or capacity. **Not
  used as a runtime dependency.**
