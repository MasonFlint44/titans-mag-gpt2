# TITANS MAG + GPT-2 Architecture

## Overview

Implements the **Memory as a Gate (MAG)** variant from the TITANS architecture
("Titans: Learning to Memorize at Test Time", Sun et al. 2025, arXiv:2501.00663)
on top of GPT-2.

The core idea: augment each transformer block with a Neural Memory Module (NMM) whose
*weights* update online during the forward pass via surprise-driven gradient descent.
The memory output is combined with the attention output via a learned gate, letting the
model blend long-range associative memory with local attention.

---

## Neural Memory Module (NMM)

### Forward pass (retrieval)

The NMM is a gated MLP (SiLU-GLU: `silu(W1·x) * sigmoid(W_gate·x)`) with weights
`M = {W1, W_gate, W2}` that serve as the associative memory. The weights are
**recurrent state** — they update at every token via surprise-driven gradient descent
and are carried forward across tokens. They also serve as the **meta-learned initial
state**: `memory_mlp.W*.weight` (their initial values) are trained by the outer optimizer
at sequence/document starts (via TBPTT), but not through the per-token surprise updates
(which are inner-loop only).

```
q̂_t  = l2_norm(conv(act(q_proj(x_t))))                      # retrieval query
h_t   = silu(W1 · q̂_t) ⊙ sigmoid(W_gate · q̂_t)            # gated hidden
y_t   = out_scale ⊙ (norm(W2 · h_t) + q̂_t)                 # ResidualNorm + out_scale
```

Two components beyond the bare gated MLP:
- **ResidualNorm**: `MemoryMLP.forward` returns `norm(W2(h)) + x` (residual connection with
  LayerNorm). The `+ q̂_t` term passes the query through even if W2≈0.
- **`out_scale`**: a learnable `[d_model]` parameter in `NeuralMemoryModule`, initialized to
  zeros for `finetune_mode=True`. Multiplied element-wise onto the MLP output. At init:
  `out_scale=0 → y_t=0` exactly, preserving the pretrained GPT-2 residual stream unchanged.

### Projections and convolution (Section 4.4)

After each of the Q, K, V linear projections, a **1D depthwise convolution**
is applied before the activation and L2-normalization:

```
proj_out = conv1d_dw(linear(x))              # depthwise conv, kernel_size=4
k̂_t = l2_norm(act(proj_out_k))              # L2 after activation
q̂_t = l2_norm(act(proj_out_q))
v_t  = act(proj_out_v)                        # values: activated, not L2-normed
```

The convolution captures local dependencies before the memory update. The paper ablates
this component (+1.24 ppl without it). Kernel size is not specified in the paper;
kernel_size=4 is standard for similar linear recurrent models (Mamba, GLA).

### Memory update rule

At each token `t`, the NMM updates its weights to minimize a surprise signal:

```
ℓ_t = ||f_M(k̂_t) - v_t||²_2                # associative memory loss (Eq. 12)

g_t  = ∇_{M_{t-1}} ℓ_t                      # gradient via torch.func.grad + vmap
g̃_t  = NewtonSchulz5(g_t)                   # spectral normalization (see note below)

θ_t  = sigmoid(W_θ · x_t)                    # per-token data-dependent learning rate
η_t  = sigmoid(W_η · x_t)                    # per-token data-dependent momentum decay
α_t  = sigmoid(W_α · x_t)                    # per-token data-dependent forgetting rate

S_t  = η_t · S_{t-1} - θ_t · g̃_t           # momentum buffer (Eq. 14); θ applied POST-NS
M_t  = (1 - α_t) · M_{t-1} + S_t            # memory weights with forgetting (Eq. 13)
```

All three gating parameters (θ, η, α) are **per-token, data-dependent** linear
projections `W_θ, W_η, W_α ∈ ℝ^{d_model × 1}` (the paper is explicit: "In our
experiments, we make these parameters as the functions of tokens", §3.2).

Apply **spectral normalization** (Newton-Schulz 5-step) to `g_t` first, THEN scale by
`θ_t`. This ordering is critical: Newton-Schulz normalizes magnitude to spectral norm ≈ 1
by dividing by the Frobenius norm. If θ_t were applied before NS (e.g., inside the loss
function), NS would cancel it entirely (`NS(θ·g) = NS(g)` since the scale divides out).
θ_t must be the last multiplier before the momentum accumulation.

### Gradient computation

Use `torch.func.grad` + `torch.func.vmap` — do NOT hand-code the backward:

```python
from torch.func import grad, vmap, functional_call

def inner_loss(params, k_hat, v):
    pred = functional_call(memory_mlp, params, k_hat)
    return F.mse_loss(pred, v, reduction='sum')

per_sample_grad_fn = vmap(grad(inner_loss), in_dims=(0, 0, 0))
grads = per_sample_grad_fn(params_per_sample, k_hat, v)
```

Functional autograd is correct by construction and handles any NMM architecture.

### NMM dimensions

- **Input/output**: `d_model` (same as embedding dim)
- **Hidden**: `4 × d_model` (expansion factor 4, consistent with GPT-2 MLP)
- **Depth**: `L_M = 2` — our SiLU-GLU structure (W1 + W_gate → hidden → W2) constitutes a
  two-layer MLP (input→hidden→output), which is L_M=2 in paper terminology. L_M=1 would
  be a single linear map with no hidden layer. Paper ablation: L_M≥2 >> L_M=1.
- **State**: `(M, S)` per layer, each a dict of 3 entries: `{W1: [B,4d,d], W_gate: [B,4d,d], W2: [B,d,4d]}`.
  LayerNorm (ResidualNorm) params are **not** recurrent — they are fixed module parameters
  trained only by the outer optimizer. This keeps all state entries 2D (required for Newton-Schulz).
  `memory_mlp.W1`, `memory_mlp.W_gate`, `memory_mlp.W2` weights initialized **Xavier uniform** (confirmed in lucidrains).

*Multi-head NMM* (n_heads parallel heads each on d_head dims) is not in the paper but
is implemented in lucidrains' reference as an optional enhancement. We do not use it
by default; it can be added via config if ablations warrant it.

---

## MAG: Memory as a Gate

The paper's formulation (Equations 26–28):

```
x̃  = concat([P, x], dim=T)                  # prepend persistent tokens
y   = SW-Attn*(x̃)                           # sliding window attention output
o   = y ⊗ M(x̃)                              # gate (Eq. 28)
```

The `⊗` gating is implemented as:

```
y_n = σ(γ_a ⊙ y)                            # normalize + nonlinearity (learnable scale)
m_n = σ(γ_m ⊙ M(x̃))                        # normalize + nonlinearity (learnable scale)
o   = y_n ⊗ m_n                              # element-wise product
```

`γ_a, γ_m ∈ ℝ^{d_model}` are learned scale vectors initialized to 1. `σ` is SiLU.

**Fine-tuning modification**: when initializing from pretrained GPT-2, this pure
formula silences the attention output at init (y_mem≈0 → o≈0). Instead, use:

```
o = y_attn + silu(γ_m ⊙ y_mem) * y_attn    # = y_attn · (1 + silu(γ_m · y_mem))
```

At init (y_mem≈0): `o = y_attn`. Standard GPT-2 residual preserved. The memory
contribution is additive on top of the standard residual, starting at zero.
Controlled by `finetune_mode=True` in config.

### Position embeddings and chunk_size

GPT-2's position embedding table covers positions 0…1023 (`block_size=1024`).
Chunks are processed independently with positions restarting at 0 each chunk boundary.
This is semantically correct: the NMM state provides cross-chunk memory, so
within-chunk positions are sufficient. `chunk_size` must be ≤ `block_size`.

For experiments targeting sequences longer than 1024 tokens trained from scratch,
either interpolate/extend the position embedding table or replace with RoPE.

### Attention: SWA vs full causal

The paper specifies **Sliding Window Attention (SWA)** for the short-term memory
component of MAG (Figure 3b, Eq. 27: "SW-Attn*"). SWA limits each token to attending
to the most recent `swa_window` tokens, making attention O(n · swa_window).

**For our GPT-2 adaptation**: we default to **full causal attention** for two reasons:
1. GPT-2 is pretrained with full attention — switching to SWA breaks the pretrained
   distribution and requires longer fine-tuning to recover.
2. GPT-2's max context is 1024 tokens; SWA's efficiency advantage only matters at
   lengths well beyond that.

SWA is available as a config option (`use_swa=True`, `swa_window=256`) for experiments
targeting longer contexts trained from scratch. The config field is `swa_window` (not
`window_size`); see PLAN.md task 0.2.

---

## Persistent Memory Tokens

All TITANS variants use `N_p` learnable, input-independent tokens prepended to each
sequence (Eq. 19). These are distinct from the NMM — they are static parameters, not
updated at test time:

```
x̃ = concat([P, x], dim=T)                   # P ∈ ℝ^{N_p × d_model}
```

Attention is computed over `x̃`. Causal mask (block-structured):
- Persistent tokens → persistent tokens: full (bidirectional among themselves)
- Persistent tokens → real tokens: masked (−∞)
- Real tokens → persistent tokens: full (always visible)
- Real tokens → real tokens: standard causal mask

After attention, the persistent token positions are dropped before the MAG combination.

Paper ablation: persistent tokens alone have negligible or negative effect; they only
contribute when combined with the NMM. Keep `N_p` small (default 4).

---

## Block Architecture

```
TitansMAGBlock
├── persistent_mem     P ∈ ℝ^{N_p × d_model}    (learned, prepended before attn)
├── ln_1               LayerNorm(d_model)
├── attn               CausalSelfAttention (full causal; SWA optional)
├── ln_nmm             LayerNorm(d_model)          (separate from ln_1)
├── nmm                NeuralMemoryModule
│   ├── q_proj         Linear(d_model, d_model, bias=False) + Conv1d_dw  (SiLU applied at call site, NOT inside module)
│   ├── k_proj         Linear(d_model, d_model, bias=False) + Conv1d_dw  (SiLU applied at call site, NOT inside module)
│   ├── v_proj         Linear(d_model, d_model, bias=False) + Conv1d_dw  (SiLU applied at call site, NOT inside module)
│   ├── memory_mlp     MemoryMLP(d_model, expansion=4)  ← W*.weight ARE the learned initial state
│   │   ├── W1         Linear(d_model, 4*d_model, bias=False)  Xavier-uniform init
│   │   ├── W_gate     Linear(d_model, 4*d_model, bias=False)  Xavier-uniform init
│   │   ├── W2         Linear(4*d_model, d_model, bias=False)  Xavier-uniform init
│   │   └── norm       LayerNorm(d_model)               (fixed; NOT recurrent — see state note)
│   ├── W_θ, W_η, W_α  Linear(d_model, 1, bias=False)  (per-token update params)
│   └── out_scale      Parameter [d_model]              (init=zeros when finetune_mode=True; init=ones when False)
├── gamma_attn         Parameter [d_model]         (MAG scale, init=1; created only when finetune_mode=False)
├── gamma_mem          Parameter [d_model]         (MAG scale, init=1; always created)
├── ln_2               LayerNorm(d_model)
└── mlp                GPT-2 MLP (unchanged)
```

### Block forward pass

```
x̃     = concat([P.expand(B,-1,-1), x], dim=1)  # prepend persistent tokens
x̃_norm = ln_1(x̃)
y_attn = attn(x̃_norm, mask)[:, N_p:, :]        # drop persistent prefix from output

x_norm         = ln_nmm(x)                       # NMM receives real tokens only (not x̃)
y_mem, state_t = nmm.forward_chunk(x_norm, state_{t-1}, doc_boundaries)
# 3-arg call: NMM needs doc_boundaries to reset state at within-chunk document starts
# (see PLAN.md task 1.8 / 2.4). Calling with only (x_norm, state) drops boundary handling.

# finetune_mode=True (default — initialize from pretrained GPT-2):
#   At init: out_scale=0 → y_mem=0 → silu(0)=0 → o = y_attn. GPT-2 residual preserved.
o = y_attn + silu(gamma_mem ⊙ y_mem) * y_attn   # = y_attn * (1 + silu(gamma_mem * y_mem))

# finetune_mode=False (training from scratch — pure paper formula):
# o = silu(gamma_attn ⊙ y_attn) * silu(gamma_mem ⊙ y_mem)

x = x + o                                        # residual
x = x + mlp(ln_2(x))                             # MLP residual
```

---

## Full Model

```
TitansMAGGPT2
├── wte       Token embedding  [vocab_size, d_model]
├── wpe       Position embedding [block_size, d_model]
├── drop
├── blocks    N × TitansMAGBlock
└── ln_f      Final LayerNorm
```

```python
def forward(idx, nmm_states=None, doc_boundaries=None):
    # nmm_states: list of (M, S) per layer; None → init via init_state() from memory_mlp.W*.weight
    # doc_boundaries: bool [B, T]; True at document starts → reset NMM state
    # returns: logits [B, T, vocab_size], new_nmm_states
```

Language model head: `lm_head = wte.weight.T` (tied weights, as in GPT-2).

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Gradient computation | `torch.func.grad` + `vmap` | Correct by construction; handles any NMM arch |
| Spectral norm on updates | Newton-Schulz 5-step (optional flag, default ON) | Prevents update blow-up; primary cause of naive-impl divergence |
| NS applied per-gradient | Before momentum accumulation (not post-scan) | Per-token stability; lucidrains applies NS to accumulated update instead — both valid |
| NS transpose guard | Transpose tall matrices (rows>cols) before NS, back after | NS converges on fat/wide matrices. W1/W_gate [4d,d] are tall → transposed. W2 [d,4d] is wide → no transpose. Without this guard, NS on tall W1/W_gate gradients is suboptimal |
| θ_t application order | After Newton-Schulz, in momentum update | NS divides by Frobenius norm — pre-scaling by θ is cancelled exactly; must scale post-NS |
| `memory_mlp.W*.weight` initialization | Xavier uniform | Prevents activation saturation; confirmed in lucidrains. No separate "W_init" object exists — W*.weight in memory_mlp ARE the initial weights |
| Gated MLP activation | silu(W1·x) * sigmoid(W_gate·x) | Paper says "gated MLP". This is SiLU-GLU (sigmoid gate), NOT SwiGLU (SwiGLU uses a linear gate, no sigmoid). Lucidrains MemoryMLP uses GELU between layers. Our choice is valid; call it "SiLU-GLU" not "SwiGLU" |
| Attention type | Full causal (SWA optional) | GPT-2 pretrained with full attn; SWA available for long-context |
| 1D conv in NMM | Depthwise (no pointwise), kernel_size=4 | Paper §4.4 says "depthwise-**separable**" (includes pointwise); we omit pointwise as a simplification — lucidrains also omits the conv entirely |
| Retrieval ordering | Write-then-read (retrieve from M_t) | Paper Eq. 15 uses M_{t-1} (read-then-write); lucidrains retrieves from M_t (write-then-read). We follow lucidrains — the current token's query sees the freshly-updated memory |
| MAG gate σ | SiLU | Paper §4.2 says "normalize using learnable vectors, followed by non-linearity σ" without naming σ; SiLU is our choice. Lucidrains uses sigmoid on the memory branch only |
| θ/η/α granularity | All per-token, linear projection → scalar | Paper: "functions of tokens" §3.2 |
| NMM depth | L_M = 2 | Paper ablation: L_M ≥ 2 >> L_M = 1 |
| NMM hidden dim | 4 × d_model | Matches GPT-2 MLP expansion; paper unspecified |
| Multi-head NMM | No (single head, d_model) | Not in paper; lucidrains enhancement; add via config if needed |
| MAG gate | silu(γ·y_attn) ⊗ silu(γ·y_mem) | Paper §4.2; Hadamard after learnable normalization |
| Persistent tokens | N_p = 4, per-block | Paper uses all variants; small N_p sufficient; per-block is our choice — paper does not specify granularity |
| Memory reset | torch.where at doc boundaries (non-mutating) | Prevent cross-document leakage; in-place assignment on autograd tensors raises RuntimeError |
| Frozen backbone warm-up | Do NOT freeze GPT-2 weights | Titans Revisited: NMM-only training against frozen backbone fails |
| `persistent_mem` weight decay | Excluded (no_decay `'persistent'`) | Learnable prefix embeddings — analogous to learned position embeddings; decay shrinks them toward zero, reducing representational capacity |
| NMM input in MAG block | `ln_nmm(x)` (real tokens only, not x̃) | Paper Eq. 28 has `M(x̃)` where x̃ includes persistent tokens. We feed only real tokens: persistent tokens are input-independent so updating memory on them adds noise with no semantic benefit; also avoids slicing their prefix from NMM output |
| Conv buffer in state | Not included; stateless conv | CausalDepthwiseConv1d buffer is not part of (M, S). `step()` at T=1 sees a 1-token conv window (vs. T-token window during training). Known train/inference discrepancy; mitigated by sliding-window context strategy in generate.py |
| MAG gate (fine-tuning) | `o = y_attn·(1 + silu(γ_m·y_mem))` | Pure MAG gates away attention at init when W2≈0; additive form preserves GPT-2 residual |
| NMM output scaling | `out_scale ∈ ℝ^{d_model}`; init=**zeros** when `finetune_mode=True`, init=**ones** when `finetune_mode=False` | ResidualNorm passes x through even with W2≈0 (output = norm(W2h)+x ≈ x); zero-init out_scale is the only reliable way to get y_mem=0 at fine-tuning start. For training from scratch there is no pretrained residual to preserve, so the NMM contributes from step 1 (ones init). Conditional in `NeuralMemoryModule.__init__` (Pass 19 / G123) |
| MAG gate (from scratch) | `silu(γ_a·y_attn) ⊗ silu(γ_m·y_mem)` | Paper formula; use when not starting from pretrained weights |
| Scan approximation | Gradients pre-computed at chunk-start M_0 | g_t depends on M_{t-1}; pre-computing at M_0 makes scan feasible; is an approximation. `torch.associative_scan` requires `torch.compile` for autograd; inference-only without it |
| chunk_size | ≤ block_size (1024); positions reset per chunk | GPT-2 PE table covers 0–1023; NMM carries cross-chunk context |
| NMM state memory | ~54MB per layer at B=4, d=768, exp=4 (fp32) | Use bf16 states or nmm_expansion=1 for larger models/batches |
| nmm_states in checkpoint | Not saved; reset to init on resume | Per-sequence state, not model state; saving rarely justified |
| NMM output norm | ResidualNorm (LN + residual) | Lucidrains; stabilizes memory output scale |

---

## File Structure

```
titans-mag-gpt2/
├── ARCHITECTURE.md
├── PLAN.md
├── config.py
├── model/
│   ├── __init__.py
│   ├── nmm.py              NeuralMemoryModule
│   ├── block.py            TitansMAGBlock
│   └── titans_gpt2.py      TitansMAGGPT2
├── data/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── dataset.py
│   └── dataloader.py
├── train.py
├── generate.py
├── eval.py
├── scripts/
│   ├── load_pretrained.py
│   └── finetune.py
└── requirements.txt
```

---

## References

- Sun et al. (2025). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663
  — primary reference for NMM, MAG, and all architectural decisions
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
- Di Nepi et al. (2025). *Titans Revisited.* arXiv:2510.09551
  — frozen backbone + NMM training fails; persistent tokens alone ineffective
- Zohar et al. (2025). *TPTT.* arXiv:2506.17671
  — uses DeltaProduct (Householder linear attn), **not** original NMM; not a cross-reference
- lucidrains/titans-pytorch (GitHub)
  — reference implementation; spectral norm, `torch.func.grad`, ResidualNorm are additions
    beyond the paper that improve stability
