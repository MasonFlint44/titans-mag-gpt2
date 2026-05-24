# Implementation Plan

Ordered tasks from blank repo to a trainable, evaluable TITANS MAG GPT-2.
Each task has an unambiguous done-condition. Complete phases in order; within a phase,
tasks with no dependency on each other can be parallelized.

> **New here?** Read `ROADMAP.md` first — it's the high-level companion (one page per
> phase, key gotchas only) and points back to specific tasks in this file when you
> need the full detail. This file is the reference manual; `ROADMAP.md` is the tour.

---

## Phase 0 — Scaffolding

### 0.1 Directory skeleton
Create empty files matching the file structure in ARCHITECTURE.md.

**G182 — `requirements.txt` MUST pin lower bounds.** Earlier versions of this task
listed dependencies without version constraints (`tiktoken`, `transformers`, etc.).
Unpinned dependencies are a silent-failure source: a future minor version of
`transformers` could change the internal `GPT2Model.transformer.h[i].attn.c_attn`
attribute path (a Conv1D module our task 2.6 unpacks via `.chunk(3, dim=1)`), and
the weight-load would silently load garbage (or raise a less-helpful error one
PyTorch version up). Lock the surface area we depend on:

```
torch>=2.3,<3        # Phase 6 scan also requires >=2.8; otherwise 2.3+
tiktoken>=0.5        # Encoding.eot_token attribute (task 3.1)
transformers>=4.30   # AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
                     # + the c_attn / c_proj Conv1D weight layout we transpose in task 2.6
datasets>=2.14       # streaming=True / .iter() support used in task 4.4
numpy>=1.24          # np.memmap dtype=int32 patterns used in G170
```

Note: phases 0–5 work with `torch>=2.3`. Phase 6 (`torch.associative_scan`) requires
`torch>=2.8` (experimental API). The fallback sequential loop works on any 2.3+
version, so the `torch>=2.3` lower bound is correct for the base model — only the
scan branch needs the higher bound. The `<3` upper bound is a defensive guard
against a major-version compatibility break.

**Done:** `python -c "import torch, tiktoken, transformers, datasets"` exits 0;
`pip show transformers | grep Version` returns >= 4.30.

### 0.2 `config.py` — `TitansConfig`
Standard `@dataclass`:

```python
from dataclasses import dataclass

@dataclass
class TitansConfig:
    # GPT-2 dimensions
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    vocab_size: int   = 50257
    block_size: int   = 1024
    dropout:    float = 0.0

    # NMM
    nmm_depth:         int  = 2      # L_M; our SiLU-GLU (W1+W_gate→W2) is L_M=2 (two-layer MLP
                                     # with gated activation). L_M=1 would be linear (no hidden layer).
                                     # Paper ablation: L_M>=2 >> L_M=1. Field kept for ablations.
                                     # WARNING: this field is DOCUMENTATION-ONLY. MemoryMLP is
                                     # hardcoded to W1+W_gate+W2 (always L_M=2). Changing nmm_depth
                                     # has NO effect on the running code — to actually change depth,
                                     # you must modify MemoryMLP.__init__ to add/remove layers.
    nmm_expansion:     int  = 4      # hidden = nmm_expansion * n_embd
    nmm_conv_kernel:   int  = 4      # depthwise conv kernel size
    nmm_spectral_norm: bool = True   # Newton-Schulz on gradient updates
    nmm_n_persistent:  int  = 4      # persistent memory tokens per block
    chunk_size:        int  = 512    # TBPTT chunk; larger = better, more memory
                                     # MUST be ≤ block_size (1024 for GPT-2) — enforced in __post_init__

    # Attention
    use_swa:    bool = False  # sliding window attention (paper default for MAG); False = full causal.
                              # Consumed by TitansMAGBlock._aug_mask — when True, real tokens attend
                              # only to the most recent `swa_window` real tokens. Persistent tokens
                              # remain fully visible regardless (paper Figure 3b).
    swa_window: int  = 256    # window size when use_swa=True. Has NO effect when use_swa=False.

    # Fine-tuning
    finetune_mode: bool = True  # use additive MAG gate to preserve GPT-2 residual at init
                                # set False for training from scratch

    def __post_init__(self):
        # G190: config validation MUST use explicit `if not <cond>: raise ValueError`
        # rather than `assert <cond>`. The Python `-O` (optimization) flag strips
        # `assert` statements entirely from bytecode. A user running
        #     python -O train.py
        # (common in production, especially when packaged via PyInstaller or running
        # under a service manager that sets PYTHONOPTIMIZE=1) silently bypasses every
        # `assert` in `__post_init__`. The model then constructs with an INVALID config
        # — e.g., chunk_size > block_size → wpe(pos) goes OOB at the first training
        # forward (loud), or use_swa=True + swa_window=0 → softmax NaN at step 0
        # (silent loss=nan, see G166), or nmm_expansion=0 → MemoryMLP construction
        # raises with a confusing "shape mismatch" error far from the config site.
        # Worse: a user who tested in dev mode (no -O) saw all assertions firing
        # correctly, ships to production with -O, and the silent skip is invisible
        # until the bad config is supplied.
        #
        # `raise ValueError(msg)` is NOT stripped by -O. Always prefer it for
        # invariants the program must maintain to be correct. `assert` is only
        # appropriate for development-only checks (e.g., debug sanity assertions
        # inside a hot loop, where the -O strip is intentional for perf).

        # chunk_size > block_size would overflow wpe's [block_size, n_embd] embedding table
        # in TitansMAGGPT2.forward (task 2.5: pos = arange(T); x = wte + wpe(pos)).
        if self.chunk_size > self.block_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be <= block_size ({self.block_size}); "
                f"otherwise wpe(pos) at training time goes out of bounds."
            )
        # swa_window only meaningful when use_swa=True, but allowed unconditionally so callers
        # can flip use_swa without touching swa_window.
        if self.nmm_n_persistent < 0:
            raise ValueError("nmm_n_persistent must be non-negative")
        if self.nmm_expansion < 1:
            raise ValueError("nmm_expansion must be >= 1 (MemoryMLP needs a hidden dim)")

        # G223 — also validate `n_embd % n_head == 0` at config time, not just at
        # block-construction time. CausalSelfAttention.__init__ already enforces this
        # (G220) using the same `raise ValueError` pattern, but that check only fires
        # when a TitansMAGGPT2 is actually constructed. A config-only validation step
        # — e.g., a CI smoke test that does
        #     TitansConfig.gpt2_small(n_head=10)
        # to verify the factory + override path — silently accepts the misconfig and
        # only catches it on model construction. By that point the user may have
        # already done expensive setup (data preprocessing, DDP init) that has to be
        # torn down before they can fix the config. Symmetric with the existing
        # config-time invariants here (chunk_size > block_size, swa_window < 1 when
        # SWA on, nmm_n_persistent < 0, nmm_expansion < 1) — every other class
        # invariant `TitansMAGGPT2` depends on is checked here; n_embd/n_head should
        # be no exception. CausalSelfAttention's runtime check stays as
        # defense-in-depth (someone could construct a CausalSelfAttention directly
        # with mismatched dims, bypassing TitansConfig).
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}). "
                f"Got n_embd % n_head = {self.n_embd % self.n_head} (head_dim would be "
                f"{self.n_embd // self.n_head}, which yields "
                f"{(self.n_embd // self.n_head) * self.n_head}, not {self.n_embd}). "
                f"(See G223 / G220 in GAP_HISTORY.md.)"
            )

        # G166: with use_swa=True and swa_window<1, the SWA mask construction in task 2.4's
        # _aug_mask makes the entire real-to-real attention block all -inf. Concretely:
        #     far_past = tril(full(T,T,-inf), diagonal=-swa_window)
        #     causal   = causal + far_past
        # When swa_window<=0, tril(diagonal=0) returns -inf on AND below the main diagonal;
        # adding to `causal` (which is -inf strictly above the diagonal) yields -inf for
        # every (i, j) — including the (i, i) self-attention diagonal that should be the
        # one position every token can always attend to. softmax(all_minus_inf) is then
        # 0/0 = NaN, which propagates through the rest of the forward and produces NaN
        # logits + NaN loss with no informative error message — the user only sees
        # "loss is NaN at step 0" and chases the wrong cause (LR, init, mixed precision).
        # swa_window must be a positive integer. The default of 256 is safe; this guards
        # the case where a user explicitly overrides to 0 (or, less likely, a negative int).
        if self.use_swa and self.swa_window < 1:
            raise ValueError(
                f"swa_window must be >= 1 when use_swa=True (got swa_window={self.swa_window}). "
                f"swa_window=0 makes the real-to-real attention mask all -inf → softmax NaN. "
                f"(See G166 in GAP_HISTORY.md.)"
            )

        # G163: from-scratch training with chunk_size < block_size leaves
        # wpe.weight[chunk_size:block_size] at random init forever. During training, each
        # forward only looks up positions 0..chunk_size-1 (see TitansMAGGPT2.forward:
        # `pos = arange(T); wte(idx) + wpe(pos)`). Positions beyond chunk_size never
        # receive gradient. Then at generation (task 5.1's sliding-window of up to
        # block_size recent tokens), once the context grows past chunk_size, the model
        # accesses untrained random position embeddings — generation quality silently
        # degrades for the second half of long contexts.
        #
        # The finetune path is immune: task 2.6 overwrites wpe.weight with HF GPT-2's
        # pretrained 1024-position table, so positions chunk_size..1023 are already
        # trained (by HF). We only warn when finetune_mode=False AND chunk_size <
        # block_size. The user can either:
        #   - set chunk_size = block_size (preferred — all positions trained)
        #   - lower block_size to chunk_size (effective context limit at chunk_size)
        #   - accept the limitation and cap generation context at chunk_size manually
        if not self.finetune_mode and self.chunk_size < self.block_size:
            import warnings
            warnings.warn(
                f"From-scratch training (finetune_mode=False) with "
                f"chunk_size={self.chunk_size} < block_size={self.block_size}: "
                f"wpe.weight[{self.chunk_size}:{self.block_size}] will never be trained. "
                f"Generation at context > {self.chunk_size} will access untrained "
                f"position embeddings and silently degrade. Recommend "
                f"chunk_size == block_size (e.g., both 1024) for from-scratch training. "
                f"(See G163 in GAP_HISTORY.md.)",
                UserWarning,
                stacklevel=2,
            )

    # ---------- Factory methods (G143) ----------
    # Dimensions match HF's openai-community/gpt2{,-medium,-large,-xl}. These are the
    # ONLY sizes for which the weight loader in task 2.6 has a corresponding HF model.
    # IMPORTANT (G150): merge defaults+overrides via dict so callers can override the backbone
    # dims (e.g., for ablations). Passing dims as fixed kwargs alongside **overrides raises
    # `TypeError: multiple values for keyword argument` whenever a caller passes the same key.
    @classmethod
    def gpt2_small(cls, **overrides):
        return cls(**{**dict(n_layer=12, n_head=12, n_embd=768),  **overrides})
    @classmethod
    def gpt2_medium(cls, **overrides):
        return cls(**{**dict(n_layer=24, n_head=16, n_embd=1024), **overrides})
    @classmethod
    def gpt2_large(cls, **overrides):
        return cls(**{**dict(n_layer=36, n_head=20, n_embd=1280), **overrides})
    @classmethod
    def gpt2_xl(cls, **overrides):
        return cls(**{**dict(n_layer=48, n_head=25, n_embd=1600), **overrides})
```

Constraint: `chunk_size ≤ block_size`. GPT-2's position embedding table has
`block_size=1024` entries. Each chunk is fed independently with positions 0…chunk_size-1
(positions reset at each chunk boundary). The NMM state provides cross-chunk memory,
so this is semantically correct — position embeddings encode within-chunk position,
not global position. For inputs longer than block_size, no change is needed; TITANS
achieves long-context through NMM state, not extended positions.

**Done:** `TitansConfig.gpt2_small().n_embd == 768`; `TitansConfig(chunk_size=2048)`
raises `ValueError` (post_init enforcement). G206 — earlier wording said
"raises AssertionError" but G190 changed `assert` to `raise ValueError(...)` for
`-O` safety. A test asserting on `AssertionError` would now FAIL because the
actual exception type is `ValueError`. Assertion type matters — the standard
test pattern `with pytest.raises(AssertionError): TitansConfig(chunk_size=2048)`
silently fails the wrong way (no exception is caught, the constructor raises
ValueError which propagates past the test). Update test code to
`pytest.raises(ValueError)`.

---

## Phase 1 — Neural Memory Module

### 1.1 Depthwise 1D convolution helper
`model/nmm.py`: implement `CausalDepthwiseConv1d(dim, kernel_size)`:

```python
class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        # No padding in the conv — we pad manually on the left only
        self.conv = nn.Conv1d(dim, dim, kernel_size,
                              padding=0, groups=dim, bias=False)

    def forward(self, x):                        # x: [B, T, dim]
        x = x.transpose(1, 2)                    # [B, dim, T]
        x = F.pad(x, (self.kernel_size - 1, 0)) # left-pad only → [B, dim, T + k-1]
        x = self.conv(x)                          # [B, dim, T]
        return x.transpose(1, 2)                 # [B, T, dim]
```

Left-only padding ensures the conv is strictly causal: output at position t depends
only on inputs at positions ≤ t. No right-side trim needed.

**Done:** output shape `[B, T, dim]` equals input shape for any T; causal property
verified by confirming output[t] is identical whether or not inputs after t change.

### 1.2 Q/K/V projection modules
Three identical `NMMProjection(n_embd, kernel_size)` modules. Concrete class:

```python
class NMMProjection(nn.Module):
    def __init__(self, n_embd, kernel_size=4):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.conv   = CausalDepthwiseConv1d(n_embd, kernel_size)

    def forward(self, x):                # x: [B, T, d]
        return self.conv(self.linear(x)) # [B, T, d]
```

Structure: `Linear(n_embd, n_embd, bias=False) → DepthwiseConv1d`.
Submodule names (`linear`, `conv`) matter for optimizer parameter grouping (task 4.1) —
the `'nmm'` substring in the parent path (`blocks.X.nmm.k_proj.linear.weight`) routes these
to the NMM parameter group correctly. Do not rename to `proj` or anything containing `'norm'`,
`'bias'`, or `'gamma'` — these substrings collide with the `no_decay` set.

**No activation inside the module.** SiLU and L2-norm are applied at the call site:

```python
k_hat = F.normalize(F.silu(self.k_proj(x)), dim=-1)  # [B, T, d]
q_hat = F.normalize(F.silu(self.q_proj(x)), dim=-1)  # [B, T, d]
v     = F.silu(self.v_proj(x))                         # [B, T, d] — no L2
```

This matches ARCHITECTURE.md: `proj_out = conv1d_dw(linear(x))` then `k̂_t = l2_norm(act(proj_out_k))`.
**Do NOT put SiLU inside NMMProjection** — if SiLU is inside the module AND applied again at the call
site (e.g., `F.silu(self.k_proj(x))`), it is applied twice (double SiLU), producing a
`silu(silu(x)) * silu(silu(x))` gate that is empirically damaging and not what the paper specifies.

**Done:** `‖q̂‖ ≈ 1`, `‖k̂‖ ≈ 1`, `‖v‖` unconstrained.

### 1.3 Data-dependent update parameters
Three `nn.Linear(n_embd, 1, bias=False)` modules (`W_θ`, `W_η`, `W_α`).

```python
θ = torch.sigmoid(self.W_theta(x)).squeeze(-1)   # [B, T]
η = torch.sigmoid(self.W_eta(x)).squeeze(-1)     # [B, T]
α = torch.sigmoid(self.W_alpha(x)).squeeze(-1)   # [B, T]
```

All three are per-token (paper §3.2: "functions of tokens"). Squeeze the trailing
dimension to `[B, T]` (not `[B, T, 1]`) so that per-token slices are `[B]` (scalars
per batch entry). This is required for correct broadcasting over weight matrices in
task 1.7 and for `grad()` to receive a scalar loss in task 1.5.

**Done:** all three outputs are in `(0, 1)` for arbitrary input; shape is `[B, T]`.

### 1.4 Memory MLP (gated, L_M = 2)
`MemoryMLP(d_in, expansion=4)` — architecture module used via `functional_call`:

```python
class MemoryMLP(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        h = d * expansion
        self.W1     = nn.Linear(d, h, bias=False)    # key: 'W1.weight'
        self.W_gate = nn.Linear(d, h, bias=False)    # key: 'W_gate.weight'
        self.W2     = nn.Linear(h, d, bias=False)    # key: 'W2.weight'
        self.norm   = nn.LayerNorm(d)                # fixed — NOT part of recurrent state

    def forward(self, x):
        h = F.silu(self.W1(x)) * torch.sigmoid(self.W_gate(x))
        y = self.W2(h)
        return self.norm(y) + x                      # ResidualNorm
```

**Critical**: `norm.weight` and `norm.bias` are **NOT** part of the recurrent memory
state. Only `{W1.weight, W_gate.weight, W2.weight}` are recurrent. Reasons:
1. Newton-Schulz operates via `G.norm(dim=(-2,-1))` — valid only for 2D matrices.
   1D LayerNorm params would cause incorrect normalization.
2. The norm is a fixed stabilizer, not associative memory. Its params are trained by
   the outer optimizer (standard gradient descent), not the inner update rule.

`functional_call(mlp, state_M, k_hat)` where `state_M` contains only the three weight
matrices will override W1/W_gate/W2 while leaving `norm` using the module's fixed
registered parameters. This is the correct behavior.

**W_init and memory_mlp are the same object**: `NeuralMemoryModule` holds one
`MemoryMLP` instance (`self.memory_mlp`). Its W1.weight, W_gate.weight, W2.weight
*are* the learned initial weights — there is no separate set of W_init parameters.
The outer optimizer trains these parameters. `init_state` builds M directly from them:

```python
# Sketch — see the consolidated __init__ below for the full constructor.
# Both init_state and _forward_chunk_sequential need init_M; factor into one helper
# (_build_init_M) to keep device + clone behavior in a single place (G147).

# Method on NeuralMemoryModule:
def _build_init_M(self, B, device):
    # .clone() required: vmap with in_dims=0 on zero-stride expand tensors can cause
    # undefined behavior in the batched autograd interpreter; clone gives normal strides.
    # G207 — ordering: `.to(device).clone()` (or `.to(device, copy=True)`) NOT
    # `.clone().to(device)`. The two are semantically equivalent ONLY when source
    # and target devices match (the common co-located case). When the caller passes
    # `device` differing from `self.memory_mlp.W1.weight.device` (DataParallel
    # replicas, manual cross-device construction, multi-GPU model partitioning),
    # the order matters:
    #   - `.clone().to(device)`: allocates a CLONE on the SOURCE device (1 copy),
    #     then transfers the cloned tensor to the TARGET device (1 H2D copy + new
    #     allocation). Total: 2 allocations on source, 1 allocation on target.
    #   - `.to(device).clone()`: transfers the EXPAND view to the target device
    #     (1 H2D copy + materialization, since `.to` on a non-contiguous expand
    #     materializes), then clones on target. Total: 1 allocation on source
    #     (via `.to`'s temp), 1 clone on target. Same total work in the cross-
    #     device case but no wasted source-device allocation.
    # In the co-located case (W*.weight.device == device), both orderings reduce
    # to "clone on the same device once" — no perf diff. The change is defensive
    # and costs nothing in the common path.
    # Note: `.expand(B, -1, -1)` produces a stride-0 view; `.to()` on a stride-0
    # view will materialize (no longer stride-0 after the device transfer), which
    # is what we want — the subsequent `.clone()` is then a same-device copy.
    return {
        'W1.weight':     self.memory_mlp.W1.weight.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
        'W_gate.weight': self.memory_mlp.W_gate.weight.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
        'W2.weight':     self.memory_mlp.W2.weight.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
    }

def init_state(self, B, device):
    M = self._build_init_M(B, device)
    S = {k: torch.zeros_like(v) for k, v in M.items()}
    return (M, S)
```

The gradient from the loss can flow back through `functional_call` → M_t → M_{t-1} →
... → M_0 → init_state → `memory_mlp.W1.weight` (via `.expand()`, which sums gradients
across the batch dim). However, TBPTT detaches state between chunks, so this path is
active only at sequence/document starts (see task 1.5 for the full training design note).
The outer optimizer primarily trains projections and W_θ/η/α; memory_mlp.W*.weight is
the meta-learned initialization updated infrequently by the outer loop.

**Activation choice**: `silu(W1·x) * sigmoid(W_gate·x)`. This is a SiLU-gated GLU:
SiLU on the main branch, sigmoid on the gate branch. It is NOT true SwiGLU — SwiGLU
(Shazeer 2020) uses `silu(W1·x) * (W_gate·x)` with a linear (no-sigmoid) gate branch.
The lucidrains `MemoryMLP` uses GELU between layers (no gating). The paper says "gated
MLP" without specifying the activation. Our silu+sigmoid choice is a valid gated form
with no known disadvantage. Document as "SiLU-GLU" in config comments, not "SwiGLU".

**Output scaling for fine-tuning**: ResidualNorm passes x through (`output = norm(W2(h)) + x`),
so y_mem ≈ x (the query) even with W2≈0 — NOT zero. To achieve y_mem≈0 at init for
fine-tuning, `NeuralMemoryModule` applies a learnable output scale, **conditionally initialized**
based on `finetune_mode`:

```python
# in NeuralMemoryModule.__init__ — finetune_mode comes from the constructor arg:
if finetune_mode:
    # zero-init: y_mem = 0 at start → preserves pretrained GPT-2 residual exactly
    self.out_scale = nn.Parameter(torch.zeros(n_embd))
else:
    # ones-init: NMM contributes from step 1 (training from scratch, no residual to preserve)
    self.out_scale = nn.Parameter(torch.ones(n_embd))
# in forward / step:
y_mem = self.out_scale * functional_call(mlp, state_M, q_hat)
```

**`NeuralMemoryModule` must take `finetune_mode` as a constructor argument.** It is not derivable
from the other args; the caller (`TitansMAGBlock.__init__`) passes it through from `config.finetune_mode`.
At checkpoint resume, the saved `out_scale` tensor restores whatever value it learned — the
init branch only matters at first construction; mode mismatch at resume is the caller's responsibility.

**Consolidated `NeuralMemoryModule.__init__`** — the full constructor must assemble all pieces
described across tasks 1.2–1.7. Implementer should write exactly this:

```python
class NeuralMemoryModule(nn.Module):
    def __init__(self, n_embd, expansion=4, kernel_size=4,
                 spectral_norm=True, finetune_mode=True):
        super().__init__()
        self.n_embd = n_embd
        self.nmm_spectral_norm = spectral_norm   # accessed as self.nmm_spectral_norm in step/scan
        self.finetune_mode = finetune_mode

        # Q/K/V projections (task 1.2 — NMMProjection: Linear → CausalDepthwiseConv1d, no act)
        self.k_proj = NMMProjection(n_embd, kernel_size)
        self.q_proj = NMMProjection(n_embd, kernel_size)
        self.v_proj = NMMProjection(n_embd, kernel_size)

        # Data-dependent update parameters (task 1.3 — bias=False, output [B,T,1] → squeeze to [B,T])
        self.W_theta = nn.Linear(n_embd, 1, bias=False)
        self.W_eta   = nn.Linear(n_embd, 1, bias=False)
        self.W_alpha = nn.Linear(n_embd, 1, bias=False)

        # Memory MLP (task 1.4) — its W*.weight ARE the meta-learned initial state
        self.memory_mlp = MemoryMLP(n_embd, expansion)
        nn.init.xavier_uniform_(self.memory_mlp.W1.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W_gate.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W2.weight)

        # Output scale (conditional init — see above)
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(n_embd))
        else:
            self.out_scale = nn.Parameter(torch.ones(n_embd))

        # Cache the per-sample gradient function ONCE — recreating per call is slow (task 1.5).
        # Pass `spectral_norm` so the inner-loss reduction matches the post-NS pipeline (G160):
        # 'sum' when NS is on (NS cancels the d-factor); 'mean' when NS is off (avoids a 768x
        # blow-up of W_θ's effective LR).
        #
        # G165 — `nmm_spectral_norm` is BAKED INTO THE CACHED grad function and cannot be
        # safely changed after construction. The NS branch later in step()/forward_chunk
        # reads `self.nmm_spectral_norm` dynamically (so toggling it appears to "work"),
        # but the cached `per_sample_grad_fn` already chose its reduction at __init__ time.
        # Mismatch case: construct with spectral_norm=True (reduction='sum'), then mutate
        # `self.nmm_spectral_norm = False` for an ablation. The NS branch is now skipped,
        # so the d_model factor in the 'sum' reduction is no longer cancelled by NS.
        # Gradient on W_θ's per-token output is ~d_model (=768) times larger than the
        # author's mental model; sigmoid(W_θ x) ∈ (0,1) scales this huge gradient and the
        # effective per-token LR is ~768× too high. Training silently diverges within tens
        # of steps and the user (correctly toggling a single flag) blames spectral norm,
        # not the cached-reduction stale-state issue.
        # Hard rule: TREAT nmm_spectral_norm AS CONSTRUCTION-TIME-ONLY. To run an ablation
        # without NS, build a SECOND NMM/model with spectral_norm=False. Do not mutate the
        # field on an existing instance. The assertion below makes the rule machine-checkable.
        self.per_sample_grad_fn = _make_grad_fn(self.memory_mlp, spectral_norm=self.nmm_spectral_norm)
        # Lock the spectral_norm value into the grad function's closure. A later sanity
        # check in step() and _forward_chunk_sequential should assert that
        # `self.nmm_spectral_norm == self._spectral_norm_at_init` before running the loop,
        # so mid-training mutation surfaces as a clear AssertionError instead of silent
        # gradient blow-up.
        self._spectral_norm_at_init = self.nmm_spectral_norm

        # Cache the batched retrieval function ONCE — same reason as per_sample_grad_fn (G140).
        # Recreating `vmap(...)` inside step() / _forward_chunk_scan's `retrieve_one_token` per
        # call rebuilds the batching closure on every token, which is a measurable hit at
        # T=512. The closure captures self.memory_mlp; after .to(device), the captured module
        # is moved with it (functorch reads the module at call time), so device transfer is safe.
        def _retrieve_one_sample(m_dict, q):
            return functional_call(self.memory_mlp, m_dict, q.unsqueeze(0)).squeeze(0)
        self._batched_retrieve = vmap(_retrieve_one_sample, in_dims=(0, 0))
```

This is the single source of truth for `__init__`; do not also rely on prose-fragment lists
elsewhere in Phase 1. `MemoryMLP`, `NMMProjection`, `CausalDepthwiseConv1d`, and `_make_grad_fn`
are defined in their own tasks (1.4, 1.2, 1.1, 1.5 respectively).

**Additional methods** on `NeuralMemoryModule` (defined as peers to `__init__`, not inside it):
- `_build_init_M(self, B, device)` — sketch above; the shared init-M builder (G147).
- `init_state(self, B, device)` — sketch above; returns `(M, S)` using `_build_init_M`.
- `step(self, x_t, state)` — task 1.7.
- `_forward_chunk_sequential(self, x_chunk, state_in, doc_boundaries)` — task 1.8.
- `_forward_chunk_scan(self, x_chunk, state_in, doc_boundaries)` — task 6.1 (Phase 6 only).
- `forward_chunk(self, x_chunk, state_in, doc_boundaries)` — Phase 1 wraps the sequential
  path; Phase 6 (task 6.2) replaces with a dispatcher.

**Done:** `functional_call(mlp, params, x)` produces output `[B, d]`; with `out_scale=0`,
`y_mem` is exactly zero for any input.

### 1.5 Gradient computation via `torch.func`
Build `self.per_sample_grad_fn` via the `_make_grad_fn(memory_mlp)` factory below — there is
NO separately-named `compute_surprise_grad` function; the per-sample grad function IS the
artifact (cached in `__init__`). Do NOT include theta in the loss. θ_t must be applied
AFTER Newton-Schulz (task 1.6), not inside the loss:

```python
from torch.func import grad, vmap, functional_call

# IMPORTANT: inner_loss and per_sample_grad_fn MUST be created once in NeuralMemoryModule.__init__,
# NOT in forward() or step(). Creating vmap(grad(...)) on every forward call is slow.
# inner_loss closes over self.memory_mlp (the module instance, captured at __init__ time).

# G160: _make_grad_fn MUST take the `spectral_norm` flag so the reduction can be chosen
# at construction time. Earlier passes had a comment that said "With nmm_spectral_norm=False:
# switch to reduction='mean'" but the code was unconditionally `reduction='sum'`. A user
# toggling `nmm_spectral_norm=False` for an ablation (and NOT manually editing the inner
# loss — the natural assumption "the code reads the config") got gradients d_model = 768×
# larger than intended; W_θ's per-token LR (sigmoid output in [0,1]) then scaled this huge
# gradient, producing an effective LR ~768× too high → divergence with no error. The user
# would (wrongly) blame the spectral-norm hypothesis when the real issue was the scale.

def _make_grad_fn(memory_mlp, spectral_norm: bool):
    # Reduction choice — derived from spectral_norm, NOT hard-coded:
    # - spectral_norm=True (default): use 'sum' to match paper Eq. 12 (squared L2 norm).
    #   NS normalizes the resulting gradient's spectral norm to 1, so the d_model factor
    #   cancels exactly — sum vs mean is invisible to the downstream momentum update.
    # - spectral_norm=False: use 'mean' to keep gradient scale independent of d_model.
    #   Without NS, 'sum' would make the gradient grow with d, blowing up W_θ's effective
    #   LR by a factor of d. 'mean' is equivalent to dividing the paper's loss by d, which
    #   uniformly scales the gradient by 1/d — fine when paired with W_θ initialized to
    #   produce sigmoid(W_θ x) ≈ 0.5 at init (the default kaiming init).
    reduction = 'sum' if spectral_norm else 'mean'

    def inner_loss(params, k_hat, v):
        # params: dict of per-sample weights [h, d]; k_hat: [d]; v: [d]
        pred = functional_call(memory_mlp, params, k_hat)
        return F.mse_loss(pred, v, reduction=reduction)   # must return a scalar

    # grad() defaults to argnums=0 → differentiates w.r.t. the FIRST argument (params).
    # params MUST stay the first arg in inner_loss — reordering breaks gradient computation.
    # vectorize over batch dim (dim 0 for all inputs)
    return vmap(grad(inner_loss), in_dims=(0, 0, 0))

# In NeuralMemoryModule.__init__:
# self.per_sample_grad_fn = _make_grad_fn(self.memory_mlp, spectral_norm=self.nmm_spectral_norm)
# call: grads = self.per_sample_grad_fn(M_batched, k_hat, v)
# M_batched: dict of [B, ...] tensors; k_hat, v: [B, d]
```

**Why theta is NOT in the loss:** Newton-Schulz normalizes the gradient magnitude
to spectral norm ≈ 1 regardless of input scale (its first step divides by ‖G‖_F).
If theta is inside the loss, the returned gradient is `θ_t · g`, but Newton-Schulz
computes `NS(θ_t · g) = NS(g)` — θ_t cancels exactly. Theta must be applied
POST-spectral-norm in the momentum update (task 1.7).

The returned `grads` dict has the same keys as `params`, each tensor shaped `[B, ...]`.

Full backprop through this call trains the Q/K/V projections and the data-dependent
parameter networks (W_θ/η/α) via the outer optimizer — the outer loss flows through
the retrieved `y_t` values (which depend on M_t, which depends on the grads).

**`memory_mlp.W*.weight` and outer BPTT**: `memory_mlp.W*.weight` serves as the
*learned initial state* for the NMM. It can in principle receive outer gradients through
the decay path (M_t = (1-α)*M_{t-1} + S_t traces back to M_0 = init_state → W*.weight),
but TBPTT detaches state at every chunk boundary, severing this path for all but the
first chunk of each document. In practice, W*.weight is trained almost exclusively by the
outer optimizer at sequence/document starts, and the per-token surprise updates happen
entirely within the inner loop. This is intentional: W*.weight is the meta-learned
initialization, not a weight updated on every token. Do NOT add `stop_gradient` to the
init_state path — this would prevent any learning of the initial memory; but also do NOT
remove the TBPTT detach, as it is required for the inner-loop design to be the primary
update mechanism.

**Done:** gradient output matches `torch.autograd.functional.jacobian` to 1e-4;
`grad()` receives a scalar return value from `inner_loss`; with the SAME random
`(M, k_hat, v)` and `spectral_norm=False`, the gradient produced by `_make_grad_fn(...,
spectral_norm=False)` has Frobenius norm approximately `1/d_model` times the Frobenius
norm of `_make_grad_fn(..., spectral_norm=True)`'s output — verifies the reduction
switch (G160) is wired through, not just documented.

### 1.6 Newton-Schulz spectral normalization
`newton_schulz5(G, steps=5)` — normalizes a matrix so its spectral norm ≈ 1:

```python
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    # G198 — force fp32 inside the iteration. `newton_schulz5` is called from
    # `_forward_chunk_sequential` (task 1.8) and `_forward_chunk_scan` (task 6.1),
    # both of which run under `train_step`'s `torch.autocast(dtype=torch.bfloat16)`
    # scope (task 4.2 / G159). Without the explicit fp32 cast below, the matmuls
    # `G @ G.mT` and `(b*A + c*A@A) @ G` execute in bf16, and the iteration
    # accumulates rounding errors that break the spectral-norm-≈1 fixed point:
    #   - The Frobenius-norm divisor `G / (G.norm + eps)`: norm in bf16 over a
    #     [4d, d] = [3072, 768] tensor sums ~2.4M squared values; the accumulated
    #     rounding error is ~5% relative. The first iteration starts with G whose
    #     spectral norm is already wrong by 5% before the polynomial runs.
    #   - The polynomial `a*G + (b*A + c*A@A) @ G` with a=3.4445, b=-4.7750,
    #     c=2.0315 is tuned (Jordan et al.) to converge to the spectral-norm-1
    #     fixed point IN FP32. In bf16, the iteration overshoots or undershoots
    #     depending on the input — empirically the post-NS spectral norm spreads
    #     across [0.7, 1.4] instead of converging to ~1.
    # Silent failure mode: post-NS gradient magnitude is no longer tightly bounded.
    # θ_t (the per-token learning rate, applied POST-NS — see task 1.5) interprets
    # the gradient as if it were spectral-norm-1; the actual scale is 30%+ off.
    # Some tokens over-amplify (NMM weight blowup → NaN downstream), others
    # under-amplify (no learning). Training looks "noisy" but the user blames LR
    # or seed, not the dtype inside NS. We adopt fp32-internal NS for this reason.
    # G226 — `G = G.float()` ALONE is not enough under ambient bf16 autocast.
    # PyTorch's autocast policy for matmul is: when matmul inputs are fp32 under
    # an autocast-enabled region, CAST THE INPUTS BACK TO THE AUTOCAST DTYPE
    # (bf16/fp16) before running, then return autocast-dtype output. So:
    #     G = G.float()      # G is fp32 ✓
    #     A = G @ G.mT       # under autocast bf16: inputs CAST to bf16 internally,
    #                        # output is bf16. The polynomial accumulates in bf16
    #                        # from here on, and the next iteration's G arrives bf16.
    #     G = a*G + (b*A + c*(A@A)) @ G   # the `@ G` matmul again sees bf16 inputs
    #                                       # (A is bf16, G upcast or cast); product
    #                                       # is bf16. The G.float() upcast at the
    #                                       # top is silently undone on the FIRST
    #                                       # matmul iteration, and never recovered.
    # The cast is "absorbed" by autocast's policy. The G198 fix as originally
    # written looks like it forces fp32 but actually runs the iteration in bf16
    # — same failure mode G198 was added to prevent. Effectively a no-op.
    # The fix that actually works: explicitly DISABLE autocast for the duration
    # of the NS iteration. `torch.amp.autocast(device_type=..., enabled=False)`
    # nests inside a parent autocast scope and turns off auto-casting for ops
    # inside. The `G.float()` cast is then respected: `G @ G.mT` sees fp32
    # inputs, runs in fp32, returns fp32. The full iteration is genuinely fp32.
    # On exit from the disabled-autocast scope, the parent autocast (bf16) is
    # restored for subsequent ops. We use `G.device.type` so the same code
    # works for CPU and CUDA (matters for unit tests / debugging on CPU).
    orig_dtype = G.dtype
    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        G = G.float()
        # NS converges for fat/wide matrices (more cols than rows).
        # Tall matrices (more rows than cols) must be transposed first.
        # W1/W_gate are [4d, d] (tall) → need transpose. W2 is [d, 4d] (wide) → no transpose.
        should_transpose = G.shape[-2] > G.shape[-1]
        if should_transpose:
            G = G.mT
        G = G / (G.norm(dim=(-2,-1), keepdim=True) + eps)
        for _ in range(steps):
            A = G @ G.mT
            G = a * G + (b * A + c * (A @ A)) @ G
        if should_transpose:
            G = G.mT
    return G.to(orig_dtype)
```

Applied to each **individual gradient matrix** before the momentum accumulation when
`spectral_norm=True`. Coefficients a=3.4445, b=-4.7750, c=2.0315 are from Jordan et al.
(Muon optimizer / nanogpt), confirmed in the lucidrains implementation.

**Structural divergence from lucidrains NS**: lucidrains' `newtonschulz5` contains an early
exit `if t.ndim <= 3: return t`, which skips NS entirely for 3D inputs `[B, h, d]`. Our
implementation intentionally omits this guard — in the sequential path, per-sample gradients
are exactly `[B, 4d, d]` (ndim=3) and must have NS applied. Only the coefficients are
confirmed from the lucidrains source; the function body is our own.

**Transpose guard**: NS converges when applied to fat (wide) matrices — more columns than rows.
W1 [4d, d] and W_gate [4d, d] are tall; their gradients are transposed before NS and
transposed back after. W2 [d, 4d] is wide; no transpose needed. Without this guard, NS
applied to tall matrices produces a different spectral normalization result and may converge
more slowly or diverge. This guard pattern is from the lucidrains Muon implementation.

**Design choice vs. lucidrains**: The lucidrains reference applies Newton-Schulz to the
*accumulated momentum vector* (after the scan, before applying to M). We apply it to each
raw gradient g_t individually, before it enters the momentum buffer. Our approach provides
per-token gradient stability guarantees; lucidrains' approach normalizes the batch-level
accumulated update. Both are valid; neither is specified in the paper (Newton-Schulz is
not in the paper at all). We prefer per-gradient NS because:
1. Bounded gradient magnitude at every token → M weights stay well-conditioned throughout
2. Cleaner interaction with θ_t (which then acts as a true per-token learning rate)

**Done:** `‖newton_schulz5(G)‖_2` lies in the post-NS5 basin `~[0.85, 1.20]` for random matrices of any shape (G230 — the Muon coefficients have `a + b + c = 0.701`, so σ=1 is NOT a fixed point; "≈ 1" is engineering shorthand for "bounded near 1, tight enough for inner-loop stability", not the literal "1 ± 0.01" the diagram annotation originally suggested).

### 1.7 Sequential memory step (single-token inference ONLY)
`NeuralMemoryModule.step(x_t, state) -> (y_t, new_state)` — used for single-token
generation (`generate.py`, task 5.1) where exactly one new token is being processed.
**DO NOT call this in a training loop** — calling `step()` T times over a chunk causes
the CausalDepthwiseConv1d (kernel_size=4) to see a length-1 input every iteration,
left-padded with 3 zeros, so 3 of 4 kernel weights have no effect (G154). Training-path
chunked forward (task 1.8) pre-projects the full chunk so the conv sees T-token context.
The 1-token conv discrepancy at inference time is an acknowledged limitation, not a bug
specific to `step()`'s contract — the contract IS "one token, one step."

```
k̂_t, q̂_t, v_t  ← F.normalize(F.silu(proj(x_t))) / F.silu(v_proj(x_t))  # see task 1.2
θ_t, η_t, α_t   ← W_θ/η/α(x_t).squeeze(-1) via sigmoid   # [B] each (scalars per sample)
g_t              ← per_sample_grad_fn(M, k̂_t, v_t)          # [B, h, d] per weight
g̃_t              ← newton_schulz5(g_t)  [if spectral_norm]  # [B, h, d], spectral norm ≈ 1
S_t              = scale(η_t, S_{t-1}) - scale(θ_t, g̃_t)   # θ applied post-NS
M_t              = scale(1 - α_t, M_{t-1}) + S_t
y_t              = out_scale * _retrieve(M_t, q̂_t)           # vmap over B — see note below
```

**θ_t is applied AFTER Newton-Schulz** (not inside the loss function — see task 1.5
for the rationale: NS normalizes magnitude to spectral norm ≈ 1, so pre-scaling by θ
would be cancelled). The paper's formula is `S_t = η_t·S_{t-1} - θ_t·g̃_t`.

**Broadcasting helper** — θ_t, η_t, α_t are `[B]` (scalars per sample). The state
tensors are dicts with entries of shape `[B, d_out, d_in]` (2D weights only — norm is
not recurrent). Apply a helper to reshape the scalar to broadcast correctly:

```python
def scale(scalar_B, tensor_dict):
    # scalar_B: [B]; tensor_dict: dict of [B, ...] tensors
    result = {}
    for k, g in tensor_dict.items():
        s = scalar_B.view(scalar_B.shape[0], *([1] * (g.ndim - 1)))
        result[k] = s * g
    return result

def dict_add(a, b):
    return {k: a[k] + b[k] for k in a}

def dict_sub(a, b):
    return {k: a[k] - b[k] for k in a}
```

**Critical**: Python dicts do not support `+` or `-` operators. The pseudocode
notation `scale(η_t, S_{t-1}) - scale(θ_t, g̃_t)` is mathematical shorthand that
CANNOT be written as a Python expression. Use `dict_sub` and `dict_add` instead:

```python
# Correct Python for the momentum update:
S_t = dict_sub(scale(η_t, S_prev), scale(θ_t, g_tilde))  # S - θg
M_t = dict_add(scale(1 - α_t, M_prev), S_t)              # (1-α)M + S
```

`state = (M: dict[str, Tensor], S: dict[str, Tensor])` — both dicts have keys
matching `MemoryMLP` parameter names; each tensor shaped `[B, d_out, d_in]`.

Memory concern: for GPT-2 small (d=768, expansion=4), each dict entry has shape
`[B, 3072, 768]`. At B=4, M+S together ≈ 54MB per layer in float32, ≈ 650MB for
12 layers. Use bfloat16 for states to halve this. Set `nmm_expansion=1` for larger
models or if memory is tight.

**G212 — NMM state dtype: the default IS fp32 even under bf16 autocast; explicit
opt-in is required to actually run states in bf16.** The "use bfloat16 for states
to halve this" recommendation above is aspirational without code support. The
implementation path:

```
init_state → _build_init_M → self.memory_mlp.W*.weight  (fp32, model master copy)
                            ↓
                         .to(device).clone()             (preserves fp32)
                            ↓
                         state tuple (M, S)              (fp32)
```

Once the state is built fp32, it STAYS fp32 throughout the chunk loop even under
`torch.autocast(dtype=torch.bfloat16)` (G159). Reason: autocast only intervenes
for ops in its registered list (matmul, conv2d, linear, …); the dict-of-tensor
state plumbing uses `*`/`+`/`-`/`torch.where` which are NOT in any autocast list.
Mixed-dtype operations (bf16 sigmoid output × fp32 state) follow PyTorch's
normal promotion rules → result is fp32 → state stays fp32. The bf16-state
expectation in the comment above is wrong for the default implementation.

This is intentional default: fp32 states are more stable across the T=512 token
recurrence (NS5 already requires fp32 internally per G198; states in bf16 would
re-quantize the post-NS spectral-norm-≈1 gradient back to ~6-bit mantissa).
But the memory cost is real: at GPT-2-small B=4 n_layer=12, that's
~2.7 GB of NMM state (vs ~1.4 GB in bf16) competing with optimizer + activation
memory. On 24 GB GPUs this can be the difference between fitting and OOM.

To explicitly opt into bf16 states, override `_build_init_M` and `detach_states`
to cast at construction/detach time:

```python
# Caller-side: monkeypatch or subclass to cast state to bf16
NMM_STATE_DTYPE = torch.bfloat16  # or torch.float32 (default)

def _build_init_M_bf16(self, B, device):
    return {
        'W1.weight':     self.memory_mlp.W1.weight.unsqueeze(0).expand(B, -1, -1)
                              .to(device, dtype=NMM_STATE_DTYPE).clone(),
        'W_gate.weight': self.memory_mlp.W_gate.weight.unsqueeze(0).expand(B, -1, -1)
                              .to(device, dtype=NMM_STATE_DTYPE).clone(),
        'W2.weight':     self.memory_mlp.W2.weight.unsqueeze(0).expand(B, -1, -1)
                              .to(device, dtype=NMM_STATE_DTYPE).clone(),
    }
```

Caveats when opting into bf16 states:
- NS5 still runs fp32 internally (G198), then casts BACK to bf16 before return.
  The state momentum (S) accumulates bf16 quantization noise over T tokens —
  for T=512, the cumulative quantization can exceed 0.5% per element, drifting
  the post-NS spectral-norm-≈1 effective magnitude further from 1.
- `out_scale * _batched_retrieve(M, q_hat)` with bf16 M may produce bf16 y_t;
  out_scale=0 at finetune init then computes `0 * bf16_value = 0` (no NaN
  hazard for finite values), but if `_batched_retrieve` ever produces +/-inf
  in bf16, the multiplication becomes NaN regardless of out_scale.
- The `torch.where(mask, init_M, M)` reset call requires init_M and M to have
  matching dtypes (auto-promotes otherwise). If both are bf16, the reset is
  straightforward. If mismatched, the result promotes and subsequent ops carry
  the promoted dtype.

Recommend keeping fp32 (default) unless OOM is imminent; the 1.3 GB savings
vs the stability cost is rarely worth it for GPT-2-scale training. Document
the override explicitly so users who need bf16 know how to opt in without
trial and error.

**Single-token input note (step vs. forward_chunk)**: The projection modules (task 1.2)
expect `[B, T, dim]` input (the CausalDepthwiseConv1d is defined over a sequence dim T).
In `step()`, `x_t` is `[B, d]` — unsqueeze before projections, squeeze after.

Full `step` method, assembling all pieces above:

```python
def step(self, x_t, state):
    # x_t: [B, d] (single-token input)
    # state: (M_prev, S_prev) where each is a dict {'W1.weight': [B,h,d], ...}
    # returns: (y_t, new_state) where y_t: [B, d], new_state matches state shape
    M_prev, S_prev = state

    x_seq = x_t.unsqueeze(1)               # [B, d] → [B, 1, d]
    # NMMProjection returns Linear+Conv1d output (no activation inside — see task 1.2):
    k_raw = self.k_proj(x_seq).squeeze(1)  # [B, 1, d] → [B, d]
    q_raw = self.q_proj(x_seq).squeeze(1)
    v_raw = self.v_proj(x_seq).squeeze(1)
    # Apply activation + L2-norm OUTSIDE the module (exactly once — task 1.2 warning):
    k_hat = F.normalize(F.silu(k_raw), dim=-1)  # [B, d]
    q_hat = F.normalize(F.silu(q_raw), dim=-1)  # [B, d]
    v     = F.silu(v_raw)                        # [B, d] — no L2 on values
    # W_theta/eta/alpha are plain nn.Linear — they accept [B, d] directly:
    theta_t = torch.sigmoid(self.W_theta(x_t)).squeeze(-1)   # [B, 1] → [B]
    eta_t   = torch.sigmoid(self.W_eta(x_t)).squeeze(-1)
    alpha_t = torch.sigmoid(self.W_alpha(x_t)).squeeze(-1)

    # Per-sample inner-loss gradient (task 1.5)
    g_t = self.per_sample_grad_fn(M_prev, k_hat, v)        # dict of [B, h, d]
    if self.nmm_spectral_norm:                              # task 1.6
        g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
    else:
        g_tilde = g_t

    # Momentum + memory update (paper Eq. 13/14; θ POST-NS — see task 1.5)
    S_t = dict_sub(scale(eta_t, S_prev),  scale(theta_t, g_tilde))
    M_t = dict_add(scale(1 - alpha_t, M_prev), S_t)

    # Retrieve via cached batched-vmap (task 1.4 — write-then-read, G140 caching)
    y_t = self.out_scale * self._batched_retrieve(M_t, q_hat)  # [B, d]

    return y_t, (M_t, S_t)
```

**Why retrieval requires vmap over B** (rationale for `_retrieve_step` above): `M_t` is a
dict of `[B, h, d]` tensors (batch-stacked weights). A bare
`functional_call(memory_mlp, M_t, q_hat)` would fail — `nn.Linear` expects weight `[out, in]`,
not `[B, out, in]`. Vmap unwraps the batch dim and applies one-sample functional_call per
batch entry, then re-stacks. This is the same pattern as Phase 6 Step 4's `retrieve_one_token`
— both sequential and scan paths need vmap over B for retrieval. Phase 6's double-vmap (outer T,
inner B) is the generalization; `step()` needs only the inner-B vmap since T=1.

Note: with T=1 and kernel_size=4 left-padding, the conv sees `[0, 0, 0, x[0]]`. The
output uses only the last kernel weight element (the element aligned with the current
token) — a per-channel learned affine transform of x[0]. This differs from training
where the conv blends context from up to 4 recent tokens. The train/inference discrepancy
is documented as a known limitation in task 5.1.

**Done:** T sequential steps produce the same result as a reference manual loop.

### 1.8 Chunked forward (training mode)
`NeuralMemoryModule._forward_chunk_sequential(x_chunk, state_in, doc_boundaries) -> (y_chunk, state_out)`.
Phase 6 adds a `_forward_chunk_scan` variant and a dispatching `forward_chunk` (task 6.2) that
picks between them. In Phase 1, define this method as `_forward_chunk_sequential` directly
(not `forward_chunk`) so the Phase 6 dispatcher can wrap both implementations without renaming.

Until Phase 6 ships, define `forward_chunk` as a **thin wrapper method** so Phase 1–5 callers
(`TitansMAGBlock.forward` calls `self.nmm.forward_chunk(...)` per task 2.4) work:

```python
# Add to NeuralMemoryModule as a peer method to _forward_chunk_sequential.
# Do NOT alias via `self.forward_chunk = self._forward_chunk_sequential` in __init__ —
# at __init__ execution time the method may not yet be bound on the instance, and
# state_dict introspection treats bound-method attributes inconsistently.
def forward_chunk(self, x_chunk, state_in, doc_boundaries):
    return self._forward_chunk_sequential(x_chunk, state_in, doc_boundaries)
```

Phase 6 (task 6.2) REPLACES this thin wrapper with the dispatcher that selects between
the scan and sequential paths. The wrapper above is just enough to make Phase 1–5 work.

**CRITICAL — DO NOT loop `step()` over T for training (G154).** The natural-looking
implementation
```python
for t in range(T):
    y_t, state = self.step(x_chunk[:, t, :], state)   # WRONG for training
```
is **silently incorrect**: `step()` unsqueezes its `[B, d]` input to `[B, 1, d]` before
calling `NMMProjection`, so the `CausalDepthwiseConv1d` (kernel_size=4) sees a length-1
sequence and left-pads with 3 zeros. The conv at that single position sees `[0, 0, 0, x_t]`
per channel — output is `last_kernel_weight · linear(x_t)`. **Three of four kernel weights
have no effect**, equivalent to disabling the conv entirely at training time. Kernel
positions 0..k-2 receive no gradient and never learn. The paper's conv ablation reports
+1.24 ppl WITHOUT the conv — that regression becomes our baseline, silently, with no
error. Same silent-bug class as G151/G152.

The conv MUST see chunk-level context during training. The canonical training-path
implementation pre-projects the full `[B, T, d]` chunk (so the conv sees the full T-token
sequence and produces a causal output for every position), then loops only over the
*recurrent* state update — gradient, NS, momentum, retrieval — which is genuinely
sequential and cannot be batched across T without losing the per-token data dependence.

```python
def _forward_chunk_sequential(self, x_chunk, state_in, doc_boundaries):
    # x_chunk: [B, T, d]; state_in: (M, S) dicts of [B, h, d]; doc_boundaries: [B, T] bool or None
    # returns: (y_chunk [B, T, d], state_out (M, S) at last token)
    B, T, _ = x_chunk.shape

    # ----- Pre-projection over the full chunk (G154 — required for conv correctness) -----
    # NMMProjection = Linear → CausalDepthwiseConv1d. Running it on the full [B, T, d]
    # chunk lets the conv see T-token causal context per output position. Running it
    # on per-token [B, 1, d] slices (as `step()` does) loses 3 of 4 kernel weights.
    # SiLU + L2-norm are applied at the call site (task 1.2's "no activation inside" rule).
    k_hat_chunk = F.normalize(F.silu(self.k_proj(x_chunk)), dim=-1)  # [B, T, d]
    q_hat_chunk = F.normalize(F.silu(self.q_proj(x_chunk)), dim=-1)  # [B, T, d]
    v_chunk     = F.silu(self.v_proj(x_chunk))                        # [B, T, d]
    theta_chunk = torch.sigmoid(self.W_theta(x_chunk)).squeeze(-1)    # [B, T]
    eta_chunk   = torch.sigmoid(self.W_eta(x_chunk)).squeeze(-1)      # [B, T]
    alpha_chunk = torch.sigmoid(self.W_alpha(x_chunk)).squeeze(-1)    # [B, T]

    # G211 — `init_M` is allocated LAZILY: only when a doc boundary actually fires
    # within the chunk. Pre-G211 the unconditional `init_M = self._build_init_M(...)`
    # ran at the top of every forward, eagerly allocating 3 × [B, h, d] tensors
    # (the W1/W_gate/W2 batched initial weights) regardless of whether they would
    # ever be used. At GPT-2-small (d=768, h=3072, B=4) that's:
    #     3 weights × B × h × d × 4 bytes ≈ 113 MB per layer per forward
    #     × n_layer = 12 ⇒ ~1.36 GB allocated and freed every step
    # The vast majority of training chunks have NO doc boundaries (typical doc
    # length ≫ chunk_size of 512). Allocator pressure from this churn:
    #     - CUDA caching allocator handles it cheaply if the size is stable,
    #       but on first use after a different-shape forward, the allocator
    #       hunts/splits blocks (~100µs overhead per call).
    #     - GC of fp32 [B,h,d] tensors stresses the autograd graph.
    #     - On constrained-memory setups (gradient checkpointing + autocast)
    #       this 1.4GB sits on the GPU for the duration of the chunk's autograd
    #       graph, competing with checkpoint stash and Adam state.
    # The lazy form below allocates init_M only the FIRST time a boundary fires
    # in a given chunk; subsequent boundaries in the same chunk reuse it.
    # When no boundary fires (common case), init_M is never built — zero cost.
    # Correctness invariant: init_M is a function of (B, device, current
    # memory_mlp.W*.weight). Within one chunk, the weights don't change (outer
    # update happens between chunks), so reusing the cached init_M across
    # multiple boundaries in the same chunk is correct.
    init_M = None
    M, S = state_in
    y_list = []

    # G202 — precompute per-position "any boundary in this column?" mask ONCE,
    # transferred to CPU, instead of calling `.any()` on a CUDA tensor inside
    # the for-t loop. The naive form `if doc_boundaries[:, t].any():` evaluates
    # a 0-d CUDA bool tensor in a Python `if`, which forces an implicit GPU→CPU
    # sync (the `__bool__` method on a CUDA tensor must transfer the value to
    # CPU memory before the Python interpreter can branch on it). At T=512,
    # that's 512 GPU↔CPU syncs per chunk per layer per forward — every sync
    # stalls the CUDA stream waiting for the prior kernel to finish, then
    # blocks on the DMA roundtrip. Measured: a 512-token chunk takes ~20-30ms
    # of pure sync overhead per layer × 12 layers = 240-360ms wasted per
    # training step on GPT-2-small, which can easily exceed the per-step
    # compute budget for the rest of the recurrence.
    # The fix: collapse the batch dim once with `.any(dim=0)` to get [T] bool,
    # transfer to CPU once (a single DMA of T bytes), then index with a CPU
    # bool inside the loop — no implicit syncs. The `reset_state` call still
    # needs the GPU-side `doc_boundaries[:, t]` (it's used as a `torch.where`
    # mask on GPU tensors), so we keep the original tensor around.
    if doc_boundaries is not None:
        any_boundary_per_t = doc_boundaries.any(dim=0).cpu().tolist()  # length T
    else:
        any_boundary_per_t = [False] * T

    # ----- Recurrent loop (state-only — no projection calls inside) -----
    for t in range(T):
        # Reset on doc boundaries BEFORE this token's update (G149 — same rule as step()).
        if any_boundary_per_t[t]:
            # G211 — lazy-build init_M on first boundary in this chunk; reuse for
            # subsequent boundaries. None-check IS the gate; no other side effects.
            if init_M is None:
                init_M = self._build_init_M(B, x_chunk.device)
            M, S = reset_state((M, S), doc_boundaries[:, t], init_M)

        k_hat_t = k_hat_chunk[:, t, :]                   # [B, d]
        q_hat_t = q_hat_chunk[:, t, :]
        v_t     = v_chunk[:, t, :]
        theta_t = theta_chunk[:, t]                       # [B]
        eta_t   = eta_chunk[:, t]
        alpha_t = alpha_chunk[:, t]

        # Per-sample inner-loss gradient (task 1.5)
        g_t = self.per_sample_grad_fn(M, k_hat_t, v_t)   # dict of [B, h, d]
        if self.nmm_spectral_norm:                        # task 1.6
            g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
        else:
            g_tilde = g_t

        # Momentum + memory update (paper Eq. 13/14; θ POST-NS — see task 1.5)
        S = dict_sub(scale(eta_t, S),  scale(theta_t, g_tilde))
        M = dict_add(scale(1 - alpha_t, M), S)

        # Retrieve via cached batched-vmap (task 1.4 — write-then-read, G140 caching)
        y_t = self.out_scale * self._batched_retrieve(M, q_hat_t)   # [B, d]
        y_list.append(y_t)

    y_chunk = torch.stack(y_list, dim=1)   # [B, T, d]
    return y_chunk, (M, S)
```

- Autograd flows through all steps for outer-loop learning
- State is detached between chunks externally (TBPTT — not inside this function)
- `step()` (task 1.7) is for SINGLE-TOKEN INFERENCE only (one new token at decode time,
  where the 1-token conv discrepancy is acceptable and documented as a known limitation).
  Do NOT call `step()` in a training loop — that's the G154 bug.

Memory note: the sequential loop stores T intermediate states for backprop.
At T=512, B=4, this is significant. If OOM occurs, apply gradient checkpointing
to the per-token recurrent step (NOT to `step()` — see G154; checkpointing must wrap
the state-only update so it does not re-introduce the 1-token conv bug):

```python
# Gradient checkpointing: rematerialize each per-token recurrent update on backward.
# Wrap the GRADIENT + NS + MOMENTUM + RETRIEVAL block (which already consumes the
# pre-projected k_hat_t, q_hat_t, v_t, theta_t, eta_t, alpha_t — see the canonical
# loop body above) — NOT step(), which would recompute the conv on a 1-token slice.
def _recurrent_update(M, S, k_hat_t, q_hat_t, v_t, theta_t, eta_t, alpha_t):
    g_t = self.per_sample_grad_fn(M, k_hat_t, v_t)
    if self.nmm_spectral_norm:
        g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
    else:
        g_tilde = g_t
    S = dict_sub(scale(eta_t, S),  scale(theta_t, g_tilde))
    M = dict_add(scale(1 - alpha_t, M), S)
    y_t = self.out_scale * self._batched_retrieve(M, q_hat_t)
    return y_t, M, S

# Inside the for-t loop, replace the inline update with:
y_t, M, S = torch.utils.checkpoint.checkpoint(
    _recurrent_update, M, S,
    k_hat_chunk[:, t, :], q_hat_chunk[:, t, :], v_chunk[:, t, :],
    theta_chunk[:, t], eta_chunk[:, t], alpha_chunk[:, t],
    use_reentrant=False,
)
```

Gradient checkpointing trades memory for compute; use only if OOM occurs.

**Done:** loss decreases on a single-token memorization task (overfit test).

### 1.9 `init_state` / `reset_state` / `detach_states`
- `init_state(batch, device)` → `(M, S)` where:
  - `M` = `{'W1.weight': self.memory_mlp.W1.weight.unsqueeze(0).expand(B,-1,-1).clone(), ...}` (3 entries only —
    W1, W_gate, W2; norm params are NOT in state, see task 1.4)
  - `S` = zeros matching M shapes

- `reset_state(state, mask, init_M)` → return a NEW state with masked batch entries
  replaced by their init values. **Do NOT use in-place index assignment** — tensors in
  the autograd graph raise `RuntimeError` on in-place mutation. Use `torch.where`:

  ```python
  def reset_state(state, mask, init_M):
      # mask: [B] bool; state = (M_dict, S_dict); init_M: dict of [B, h, d]
      M, S = state
      zeros_S = {k: torch.zeros_like(v) for k, v in S.items()}
      def where_dict(new_dict, old_dict):
          result = {}
          for k, new_v in new_dict.items():
              old_v = old_dict[k]
              # mask [B] → [B, 1, 1] to broadcast over [B, h, d]
              m = mask.view(mask.shape[0], *([1] * (old_v.ndim - 1)))
              result[k] = torch.where(m, new_v, old_v)
          return result
      return (where_dict(init_M, M), where_dict(zeros_S, S))
  ```

  Called inside `forward_chunk` at each token position where `doc_boundaries[:, t]`
  has any True entries.

- `detach_states(states)` → return new per-layer list of `(M, S)` dicts with all tensors
  detached. **Must handle `None`** — on the very first training step, `nmm_states=None`
  and `model.forward` initializes state internally; `detach_states` must pass `None`
  through so `model.forward` can detect and initialize it:

  ```python
  def detach_states(states):
      if states is None:
          return None   # first step: model.forward will call init_state
      return [
          ({k: v.detach() for k, v in M.items()},
           {k: v.detach() for k, v in S.items()})
          for M, S in states
      ]
  ```

  Called between chunks in the TBPTT loop. Does NOT modify in-place — returns new lists/dicts.

**Done:** masked reset leaves unmasked entries byte-identical; detach preserves values
but severs gradient tape; no in-place operations anywhere in state management.

---

## Phase 2 — Block and Full Model

### 2.0 `CausalSelfAttention` and `GPT2MLP`
`model/block.py` must define these standard GPT-2 modules before `TitansMAGBlock`.
Both are referenced throughout the plan but were never explicitly specified.

**`CausalSelfAttention(n_embd, n_head, dropout)`** — split Q/K/V projections (not fused like GPT-2's
`c_attn`) so that HF weight loading in task 2.6 can copy each projection independently.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        # G220 — match G190's `-O`-safety rule: `assert n_embd % n_head == 0`
        # is stripped under `python -O` (PYTHONOPTIMIZE=1), so a misconfig
        # (e.g., n_embd=768 with n_head=10 → head_dim=76, 76*10=760≠768)
        # slips through __init__. The downstream
        #     q = self.q_proj(x).view(B, T, n, d)
        # raises a confusing shape error ("shape '[B, T, 10, 76]' invalid
        # for input of size B*T*768") instead of the helpful "n_head must
        # divide n_embd" message. Under non-O the bare `assert` raises
        # `AssertionError` with no detail. Either way the user has to
        # debug from a non-obvious symptom.
        # Same `raise ValueError(msg)` pattern G190 applies to
        # `TitansConfig.__post_init__`: explicit error with the actual
        # values, not stripped by `-O`. Apply consistently across the
        # codebase — anywhere a class invariant is enforced at
        # construction.
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head}). "
                f"Got n_embd % n_head = {n_embd % n_head} (head_dim would be "
                f"{n_embd // n_head}, which yields {(n_embd // n_head) * n_head}, "
                f"not {n_embd}). See G220 / G190 in GAP_HISTORY.md."
            )
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        # bias=True required — HF GPT-2 c_attn and c_proj both have biases (task 2.6)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.proj   = nn.Linear(n_embd, n_embd, bias=True)   # output projection
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x:    [B, T, C]  where T = N_p + seq_len when called with augmented input
        # mask: [T, T] additive float mask (0 = attend, -inf = block); None = no mask (no causal)
        B, T, C = x.shape
        n, d = self.n_head, self.head_dim
        q = self.q_proj(x).view(B, T, n, d).transpose(1, 2)  # [B, n, T, d]
        k = self.k_proj(x).view(B, T, n, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n, d).transpose(1, 2)
        # attn_mask: float additive mask broadcast over [B, n, T, T]
        # None means no masking — always pass _aug_mask(T) or a causal mask explicitly.
        dp = self.resid_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        y = y.transpose(1, 2).contiguous().view(B, T, C)     # [B, T, C]
        return self.resid_dropout(self.proj(y))
```

`F.scaled_dot_product_attention` requires `torch>=2.0`. The caller is responsible for passing the
correct mask — `attn_mask=None` does NOT apply a causal mask; it attends to all positions.
Task 2.4's `_aug_mask(T)` provides the block-structured causal + persistent-token mask.
Instantiated in `TitansMAGBlock.__init__` as `self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.dropout)`.

**`GPT2MLP(n_embd, dropout)`** — standard GPT-2 feedforward (unchanged from original GPT-2):

```python
class GPT2MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))
```

**`approximate='tanh'`** is required for HF GPT-2 compatibility. HF GPT-2 uses `gelu_new` (the tanh
approximation of GELU). Standard `F.gelu(x)` (error-function approximation) gives numerically different
outputs and produces a perplexity mismatch vs. HF GPT-2 even with identical weights. The `tanh`
approximation is `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
Instantiated as `self.mlp = GPT2MLP(config.n_embd, config.dropout)`.

**Done:** `CausalSelfAttention` on a `[B, N_p+T, d]` input with `_aug_mask(T)` produces attention
patterns where persistent tokens attend to each other, real tokens attend to persistent tokens and
all preceding real tokens, and no future tokens attend. `GPT2MLP` output matches
`F.gelu(c_fc(x), approximate='tanh')` elementwise.

### 2.1 Persistent memory tokens + causal mask
In `TitansMAGBlock`:

```python
self.persistent_mem = nn.Parameter(torch.randn(N_p, n_embd) * 0.02)
```

Prepend before attention: `x_aug = cat([P.expand(B,-1,-1), x], dim=1)`.

Causal mask for `x_aug` (shape `[N_p + T, N_p + T]`):
- Top-left `[N_p, N_p]`: zeros (persistent see each other)
- Top-right `[N_p, T]`: −inf (persistent don't see real tokens)
- Bottom-left `[T, N_p]`: zeros (real tokens always see persistent)
- Bottom-right `[T, T]`: standard upper-triangle −inf causal mask

After attention, slice off the persistent prefix: `y_attn = out[:, N_p:, :]`.

**Done:** output shape unchanged; attention over 8 tokens with N_p=4 produces correct
         causal pattern (verified by inspecting attention weights).

### 2.2 Separate `ln_nmm`
`self.ln_nmm = nn.LayerNorm(n_embd)` — independent of `ln_1`.
NMM receives `self.ln_nmm(x)` where `x` is the pre-residual real-token tensor.

**Done:** `ln_nmm` appears in `state_dict` with its own tracked weight/bias.

### 2.3 MAG combination — fine-tuning compatible gate
The paper's pure MAG formula `o = silu(γ_a·y_attn) ⊗ silu(γ_m·y_mem)` has an
initialization problem: with y_mem≈0 at init (W2_init≈0), silu(γ_m·0)=0, so o=0.
This silences the attention residual entirely, breaking the pretrained GPT-2 behavior.

For fine-tuning from pretrained weights, use an additive formulation that preserves
the standard attention residual at init:

```python
gamma_mem = nn.Parameter(torch.ones(n_embd))

# Standard attention residual (always present — preserves pretrained behavior)
# Additional memory contribution (starts at zero when W2_init≈0)
o = y_attn + F.silu(self.gamma_mem * y_mem) * y_attn
```

This equals `y_attn * (1 + silu(gamma_mem * y_mem))`. At init: y_mem≈0, silu(0)=0,
o = y_attn. Standard GPT-2 residual. ✓ As the NMM learns, it multiplicatively scales
the attention output, which is close to the paper's intent.

For training from scratch (not from GPT-2 weights), the pure paper formula can be
used. Add `finetune_mode: bool = True` to `TitansConfig`.

**Done:** with W2_init≈0, output exactly matches standard GPT-2 attention residual;
with W2_init large, output diverges from it in a controlled way.

### 2.4 `TitansMAGBlock.forward`

**Consolidated `TitansMAGBlock.__init__`** — single source of truth. The pieces in tasks 2.0–2.3
(persistent_mem, ln_nmm, gamma_mem, attention module, MLP module) must all be assembled here.
Implementer should write exactly this:

```python
class TitansMAGBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Constants needed by forward / _aug_mask
        self.N_p           = config.nmm_n_persistent
        self.finetune_mode = config.finetune_mode
        self.use_swa       = config.use_swa        # consumed by _aug_mask (G136)
        self.swa_window    = config.swa_window     # only consulted when use_swa=True

        # Persistent prefix tokens (task 2.1) — small init like GPT-2 token embeddings
        self.persistent_mem = nn.Parameter(
            torch.randn(config.nmm_n_persistent, config.n_embd) * 0.02
        )

        # Attention path (task 2.0) + its LayerNorm (HF GPT-2 calls these ln_1, attn)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.dropout)

        # NMM path — task 2.2's ln_nmm is independent of ln_1.
        # Pass finetune_mode to NMM: it controls out_scale init (task 1.4 — G123).
        self.ln_nmm = nn.LayerNorm(config.n_embd)
        self.nmm    = NeuralMemoryModule(
            n_embd        = config.n_embd,
            expansion     = config.nmm_expansion,
            kernel_size   = config.nmm_conv_kernel,
            spectral_norm = config.nmm_spectral_norm,
            finetune_mode = config.finetune_mode,
        )

        # MAG combination gates (task 2.3 / 2.4) — gamma_mem always; gamma_attn only in scratch mode.
        # init=ones for identity scaling at start.
        self.gamma_mem = nn.Parameter(torch.ones(config.n_embd))
        if not config.finetune_mode:
            self.gamma_attn = nn.Parameter(torch.ones(config.n_embd))
        # NOTE: do NOT create gamma_attn when finetune_mode=True — it would be an unused
        # parameter taking up optimizer state and (worse) state_dict keys that would not
        # exist when loading a from-scratch checkpoint into a finetune model or vice versa.

        # MLP path (task 2.0)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = GPT2MLP(config.n_embd, config.dropout)
```

Block must define `self.N_p = config.nmm_n_persistent`, `self.finetune_mode = config.finetune_mode`, and the `_aug_mask(T)` helper:

```python
def _aug_mask(self, T, dtype=None):
    # Returns additive attention mask for [N_p + T, N_p + T] sequence.
    # 0 = attend; -inf = masked.
    # dtype: pass q.dtype (e.g., bf16) to avoid a cast inside scaled_dot_product_attention.
    #        If None, defaults to float32 — works with autocast but not with explicit half models.
    device = self.persistent_mem.device
    N = self.N_p + T
    mask = torch.full((N, N), float('-inf'), device=device, dtype=dtype)
    # Persistent tokens see each other (top-left block)
    mask[:self.N_p, :self.N_p] = 0
    # Real tokens always see all persistent tokens (bottom-left block)
    mask[self.N_p:, :self.N_p] = 0
    # Real tokens apply standard upper-triangle causal mask (bottom-right block).
    # device= and dtype= MUST be set on the inner full() — without them the causal block is
    # built on CPU/fp32 and triggers an implicit device+dtype cast on assignment, which is
    # both slow and (in some autocast paths) reduces correctness.
    causal = torch.triu(torch.full((T, T), float('-inf'), device=device, dtype=dtype),
                        diagonal=1)
    if self.use_swa:
        # Sliding Window Attention: each real token attends only to its most recent
        # `swa_window` real tokens (including itself). Achieved by ALSO masking positions
        # at offset <= -swa_window below the diagonal. `tril(... , diagonal=-window)`
        # produces a tensor whose entries are -inf where j <= i - window and 0 elsewhere;
        # adding it to `causal` (which is already -inf above the diagonal) masks the
        # too-old positions. Persistent tokens remain fully visible (top-left block of mask),
        # matching paper Figure 3b — SWA only restricts real-to-real attention.
        far_past = torch.tril(
            torch.full((T, T), float('-inf'), device=device, dtype=dtype),
            diagonal=-self.swa_window,
        )
        causal = causal + far_past
    mask[self.N_p:, self.N_p:] = causal
    # Top-right block stays -inf: persistent tokens don't attend to real tokens
    return mask   # [N_p+T, N_p+T]
```

`forward` should pass `self._aug_mask(T, dtype=x.dtype)` (or rely on the float32 default if running
under `torch.autocast`, which casts attention internally). Building the mask in the model's
running dtype avoids implicit casts on every forward.

G183 — the snippets in this task and elsewhere use unqualified `cat`, `F`, and
similar names. The implementer should write at the top of `model/block.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat            # convenience; equivalent to torch.cat
```

The plan uses `F.silu`, `F.gelu`, `F.cross_entropy`, `cat(...)`, `nn.Linear`,
`nn.LayerNorm`, etc. consistently. The same conventions apply to `model/nmm.py`
(needs `from torch.func import grad, vmap, functional_call` in addition).

```python
def forward(self, x, nmm_state, doc_boundaries=None):
    # x: [B, T, d]
    B, T, _ = x.shape
    x_aug  = cat([self.persistent_mem.expand(B,-1,-1), x], dim=1)
    # dtype=x.dtype ensures the mask matches the running precision (bf16/fp16) — see _aug_mask.
    y_attn = self.attn(self.ln_1(x_aug), mask=self._aug_mask(T, dtype=x.dtype))[:, self.N_p:, :]
    # NMM receives real tokens only (not x_aug) — deliberate deviation from paper Eq. 28.
    # Paper has M(x̃) where x̃ includes persistent tokens. We feed only ln_nmm(x) because
    # persistent tokens are input-independent; updating memory on them adds noise with no
    # semantic benefit. Design decision documented in ARCHITECTURE.md design table.
    y_mem, nmm_state = self.nmm.forward_chunk(self.ln_nmm(x), nmm_state, doc_boundaries)
    if self.finetune_mode:
        # Additive gate: at init (out_scale=0 → y_mem=0), o = y_attn exactly.
        # Preserves pretrained GPT-2 attention residual. Memory contribution grows
        # multiplicatively as out_scale is learned.
        o = y_attn + F.silu(self.gamma_mem * y_mem) * y_attn
    else:
        # Pure paper formula (training from scratch — no pretrained residual to preserve)
        o = F.silu(self.gamma_attn * y_attn) * F.silu(self.gamma_mem * y_mem)
    x = x + o
    x = x + self.mlp(self.ln_2(x))
    return x, nmm_state
```

Parameters:
- `gamma_mem ∈ ℝ^{d_model}` init=ones — created in `__init__` always (both modes)
- `gamma_attn ∈ ℝ^{d_model}` init=ones — created conditionally: `if not finetune_mode`.
  If finetune_mode=True, `gamma_attn` does NOT exist; do not reference it in forward.
  Creating it unconditionally wastes parameters that never receive gradients.

**Done:** block is `torch.jit.trace`-able on a fixed-size input.

### 2.5 `TitansMAGGPT2`

**Consolidated `TitansMAGGPT2.__init__`** — single source of truth. Like task 2.4, the
embeddings/blocks/final-norm pieces are scattered elsewhere; assemble them here:

```python
class TitansMAGGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config            # keep config on self for state_dict introspection /
                                        # checkpoint round-trips (task 4.3 — G134).

        # Embeddings (standard GPT-2 layout — wte: [vocab, d], wpe: [block_size, d])
        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe  = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)   # post-embedding dropout (see note below)

        # N TitansMAG blocks
        self.blocks = nn.ModuleList([
            TitansMAGBlock(config) for _ in range(config.n_layer)
        ])

        # Final LayerNorm — MUST be applied before the LM head (see forward note below).
        self.ln_f = nn.LayerNorm(config.n_embd)

        # No separate lm_head module — task 2.5 forward ties weights via self.wte.weight.T.

        # G155: GPT-2-style backbone init. Must come AFTER block construction so that
        # named_modules() can find every Linear/Embedding. Skips anything inside an NMM
        # submodule (those have their own explicit inits in NMM.__init__: Xavier for
        # memory_mlp.W*, zeros/ones for out_scale, ones for gamma_*, randn*0.02 for
        # persistent_mem). For finetune_mode=True, task 2.6 then overwrites all backbone
        # params with HF GPT-2 weights — this init is effectively a no-op in that path
        # but is still applied unconditionally to keep the construction deterministic
        # (HF load is decoupled from model construction; running construction without
        # the load must produce a trainable model, not a randomly-divergent one).
        self._apply_gpt2_init()

    def _apply_gpt2_init(self):
        """GPT-2-style backbone init. The PyTorch defaults are wrong for transformers:
        - `nn.Embedding` defaults to `N(0, 1)` — std=1, which is 50× too large. With
          wte/wpe at std=1, initial pre-softmax logits have std ~sqrt(n_embd) ≈ 28 →
          softmax is one-hot on whatever vocab id wins the random init at each
          position. Cross-entropy gradient becomes essentially random; from-scratch
          training takes drastically longer to converge (often diverges).
        - `nn.Linear` defaults to `kaiming_uniform_(a=sqrt(5))` which gives a std of
          roughly `1/sqrt(3*fan_in)` ≈ 0.021 at n_embd=768. That's accidentally close
          to GPT-2's 0.02, but the value depends on fan_in (so for c_fc with fan_in=d,
          c_proj with fan_in=4d, etc., they all end up at different scales). Explicit
          `N(0, 0.02)` fixes this.
        - GPT-2 / nanoGPT also scale OUTPUT projections (`attn.proj`, `mlp.c_proj`) by
          `1/sqrt(2*n_layer)` to keep residual stream variance stable through deep
          stacks — without this, variance grows linearly with depth and later layers
          under-train.
        Silent failure mode if omitted: from-scratch training (task 4.5) appears to
        run but loss decays much slower than expected, with high seed variance.
        """
        import math
        n_layer = len(self.blocks)
        # G203 — identify NMM-internal modules by IDENTITY, not by name substring.
        # Earlier passes skipped via `if 'nmm' in name: continue`, which is correct
        # for the current naming (`blocks.0.nmm.*`, `blocks.0.ln_nmm`) but BREAKS
        # silently on any rename. Concretely, if a future refactor changes
        # `TitansMAGBlock.nmm = NeuralMemoryModule(...)` to `self.memory =
        # NeuralMemoryModule(...)`, the named_modules() walk yields
        # `blocks.0.memory.k_proj.linear`, `blocks.0.memory.memory_mlp.W1`, etc.
        # None of these contain 'nmm' → the skip fails → the NMM-internal Linears
        # (NMMProjection, MemoryMLP's W1/W_gate/W2, W_theta/W_eta/W_alpha) all get
        # overwritten with N(0, 0.02). The Xavier-uniform inits set inside
        # NeuralMemoryModule.__init__ (task 1.4) are destroyed silently.
        # Downstream impact: NMM weights start at the wrong scale; the gated MLP
        # produces near-zero outputs (since N(0, 0.02) × N(0, 0.02) ≈ 4e-4 with
        # gain factors absent); per-token gradient norms are tiny; θ_t saturates
        # near zero; the NMM never effectively trains and the user sees ~baseline
        # perplexity from-scratch with no error.
        # The robust pattern: collect the id()s of all modules that are inside
        # any NeuralMemoryModule, then skip those by identity. This is invariant
        # to naming — if a renamed `self.memory` still wraps a NeuralMemoryModule,
        # all its submodules are still excluded correctly.
        # G224 — use a RELATIVE import (`from .nmm`) instead of an absolute
        # `from model.nmm` import. Both break the `__init__` → nmm circular
        # dependency by deferring until method-call time, but they differ in
        # robustness to package renaming:
        #   - `from model.nmm import NeuralMemoryModule` (absolute) works ONLY
        #     when `model` is importable as a top-level package — either the
        #     project is being run from a CWD where `model/` is a top-level
        #     directory on sys.path, OR `pip install -e .` registered a package
        #     literally named `model`.
        #   - If the project is later packaged with a more conventional name
        #     (e.g., `pyproject.toml` declares `name = "titans_mag_gpt2"` so the
        #     top-level package becomes `titans_mag_gpt2.model.*`), the absolute
        #     import raises `ModuleNotFoundError: No module named 'model'`
        #     during the FIRST `TitansMAGGPT2.__init__` call. The model
        #     construction itself fails — the user never even reaches the
        #     training step. The fix is non-obvious to a reader skimming the
        #     traceback because the rest of the package loads fine; only the
        #     deferred import inside this one method breaks.
        # Relative imports (`from .nmm import ...`) depend only on the file's
        # location within the package, which is invariant to the outer package
        # name. The form below works under any top-level rename. The downside:
        # the file containing this method MUST be inside a package (have an
        # adjacent `__init__.py`), which it already does (`model/__init__.py`
        # per the directory skeleton in task 0.1 / ARCHITECTURE.md).
        from .nmm import NeuralMemoryModule
        nmm_internal_ids = set()
        for module in self.modules():
            if isinstance(module, NeuralMemoryModule):
                for sub in module.modules():
                    nmm_internal_ids.add(id(sub))
        # Additionally exclude `ln_nmm` (LayerNorm on the NMM input) — it's not
        # INSIDE the NeuralMemoryModule but IS on the NMM path. LayerNorm's
        # default init (weight=1, bias=0) is already what we want for it, so
        # leaving it alone (or hitting the LayerNorm branch below, which is a
        # no-op since we don't touch LayerNorm) is fine either way. We keep the
        # name-based skip for `ln_nmm` as a belt-and-suspenders documentation
        # of intent — the substring check is the FALLBACK, not the primary.
        for name, module in self.named_modules():
            if id(module) in nmm_internal_ids:
                continue
            if 'ln_nmm' in name:   # belt-and-suspenders; LayerNorm's default is correct anyway
                continue
            if isinstance(module, nn.Linear):
                # Default std = 0.02. Output projections in residual blocks get scaled.
                # The naming convention from task 2.0 puts the attention output projection
                # at `attn.proj` and the MLP output projection at `mlp.c_proj`. Q/K/V are
                # `attn.{q,k,v}_proj` — those end with `q_proj`/`k_proj`/`v_proj`, NOT
                # `.proj`, so the endswith check below correctly excludes them.
                if name.endswith('.proj') or name.endswith('.c_proj'):
                    std = 0.02 / math.sqrt(2 * n_layer)
                else:
                    std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # wte, wpe → N(0, 0.02). MUST NOT use the default N(0, 1).
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # nn.LayerNorm: defaults (weight=1, bias=0) are already correct — leave alone.
            # gamma_mem, gamma_attn, out_scale, persistent_mem are Parameters not Modules,
            # so they aren't visited here — their inits stay as set in Block/NMM __init__.
```

`self.drop = nn.Dropout(config.dropout)` is used in the forward pass below — the
embedding output is passed through dropout before the first block (standard GPT-2 pattern).

```python
def forward(self, idx, nmm_states=None, doc_boundaries=None):
    # idx: [B, T] token ids
    # nmm_states: list[layer] of (M, S) or None
    # doc_boundaries: [B, T] bool or None
    # returns: logits [B, T, vocab_size], new_nmm_states
    B, T = idx.shape
    pos = torch.arange(0, T, device=idx.device)  # [T]
    x = self.drop(self.wte(idx) + self.wpe(pos)) # [B, T, d]
    if nmm_states is None:
        nmm_states = [block.nmm.init_state(B, idx.device) for block in self.blocks]
    new_nmm_states = []
    for block, nmm_state in zip(self.blocks, nmm_states):
        x, nmm_state = block(x, nmm_state, doc_boundaries)
        new_nmm_states.append(nmm_state)
    x = self.ln_f(x)                             # MUST apply ln_f before LM head
    logits = x @ self.wte.weight.T               # tied weights [B, T, vocab_size]
    return logits, new_nmm_states
```

**Critical**: `self.ln_f(x)` MUST be applied before the LM head. GPT-2's architecture
applies a final LayerNorm to the residual stream before projecting to vocab. Omitting
`ln_f` produces unnormalized logits — the model will appear to "work" (no error) but
perplexity will be far above baseline and weights loaded from HF GPT-2 will not transfer
correctly (HF GPT-2 always applies `ln_f`).

Position embeddings use positions 0…T-1 within each chunk. Positions restart at each
chunk boundary (see task 0.2 for rationale).

**Done:** forward on random tokens produces finite logits; `new_nmm_states` length = n_layer;
post-init `model.wte.weight.std()` ≈ 0.02 (NOT ≈ 1) — verifies `_apply_gpt2_init` ran and
overrode `nn.Embedding`'s default `N(0, 1)` (G155); `model.blocks[0].attn.proj.weight.std()`
is roughly `0.02/sqrt(2*n_layer)` (scaled output projection); the NMM params still have
their own inits (`model.blocks[0].nmm.out_scale.abs().sum() == 0` in finetune_mode=True,
`model.blocks[0].nmm.memory_mlp.W1.weight.std()` consistent with Xavier).

### 2.6 GPT-2 weight loading
`scripts/load_pretrained.py`:

1. Load `openai-community/gpt2` (or medium/large/xl) via HuggingFace `transformers`
2. Map weights — GPT-2 uses fused `c_attn` (QKV) and `c_proj`; map to our split attn.
   HuggingFace GPT-2 implements `c_attn` as a custom `Conv1D(3*n_embd, n_embd)` whose
   weight is shaped `[n_embd, 3*n_embd]` (input-first, the transpose of `nn.Linear`):
   ```python
   c_attn_w = hf_block.attn.c_attn.weight   # [n_embd, 3*n_embd]
   c_attn_b = hf_block.attn.c_attn.bias     # [3*n_embd]
   # Split along the output (cols) dimension:
   W_q, W_k, W_v = c_attn_w.chunk(3, dim=1)  # each [n_embd, n_embd]
   b_q, b_k, b_v = c_attn_b.chunk(3, dim=0)  # each [n_embd]
   # Conv1D does x @ W; nn.Linear does x @ W.T — so transpose for our Linear.
   # IMPORTANT: use .data.copy_() or nn.Parameter(.clone().detach()), NOT plain assignment.
   # Direct assignment (our_attn.q_proj.weight = W_q.T) replaces the nn.Parameter with
   # a plain tensor, removing it from model.parameters() → not trained, no gradients.
   with torch.no_grad():
       our_attn.q_proj.weight.copy_(W_q.T)          # [n_embd, n_embd]
       our_attn.q_proj.bias.copy_(b_q)              # [n_embd]
       our_attn.k_proj.weight.copy_(W_k.T)
       our_attn.k_proj.bias.copy_(b_k)
       our_attn.v_proj.weight.copy_(W_v.T)
       our_attn.v_proj.bias.copy_(b_v)
       # c_proj is Conv1D(n_embd, n_embd): weight [n_embd, n_embd] — transpose similarly
       our_attn.proj.weight.copy_(hf_block.attn.c_proj.weight.T)
       our_attn.proj.bias.copy_(hf_block.attn.c_proj.bias)
   ```
   Attention Q/K/V and proj use `bias=True` (matching HF GPT-2's c_attn and c_proj biases).
   NMM projections (nmm.k_proj, nmm.q_proj, nmm.v_proj) are NOT loaded from HF weights;
   they are randomly initialized (kaiming uniform, the PyTorch Linear default).

   The attention code above applies to each block. Structure the full loop as:
   ```python
   # G216 — derive the HF model name from `config.n_embd` instead of hard-coding
   # "openai-community/gpt2" (which only matches the small variant). The four
   # GPT-2 factory configs map 1:1 to four HF checkpoints:
   #   n_embd=768  → openai-community/gpt2           (12 layers, 12 heads, small)
   #   n_embd=1024 → openai-community/gpt2-medium    (24 layers, 16 heads)
   #   n_embd=1280 → openai-community/gpt2-large     (36 layers, 20 heads)
   #   n_embd=1600 → openai-community/gpt2-xl        (48 layers, 25 heads)
   # Pre-G216 the script hard-coded "openai-community/gpt2" — user constructs
   # `our_model = TitansMAGGPT2(TitansConfig.gpt2_large())` (36 layers, 1280
   # hidden), then load_pretrained.py downloads the SMALL checkpoint (12
   # layers, 768 hidden) and the zip(our_model.blocks, hf_model.transformer.h)
   # iterates only 12 of our 36 blocks. The first iteration's c_attn copy
   # raises a shape-mismatch ("expected [768, 2304] but got [1280, 3840]")
   # at .copy_ — loud failure, but the error message points at the weight
   # copy, not the model-name mismatch. The user blames their config or our
   # attention code rather than the hard-coded model name.
   # Map size → HF name explicitly so each factory variant loads its own
   # checkpoint:
   _HF_GPT2_NAMES = {
       768:  "openai-community/gpt2",
       1024: "openai-community/gpt2-medium",
       1280: "openai-community/gpt2-large",
       1600: "openai-community/gpt2-xl",
   }
   try:
       hf_name = _HF_GPT2_NAMES[config.n_embd]
   except KeyError:
       raise ValueError(
           f"No HF GPT-2 checkpoint maps to n_embd={config.n_embd}. "
           f"Supported variants: {list(_HF_GPT2_NAMES)}. Use one of the "
           f"TitansConfig.gpt2_{{small,medium,large,xl}} factory methods, "
           f"or implement custom weight init for from-scratch training."
       )
   hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
   # Defensive sanity check: HF's actual config must match our config's
   # n_layer / n_head. If the user passes overrides (e.g.,
   # TitansConfig.gpt2_small(n_layer=14)), the zip would silently truncate
   # at 12 of our 14 blocks — blocks 12-13 stay at random init with no
   # warning. Catch this here.
   if hf_model.config.n_layer != config.n_layer:
       raise ValueError(
           f"Our n_layer={config.n_layer} but HF {hf_name} has "
           f"n_layer={hf_model.config.n_layer}. Override TitansConfig "
           f"factories at your own risk — load_pretrained cannot transfer "
           f"weights when layer counts differ."
       )
   if hf_model.config.n_head != config.n_head:
       raise ValueError(
           f"Our n_head={config.n_head} but HF {hf_name} has "
           f"n_head={hf_model.config.n_head}. Head-count mismatch makes "
           f"the QKV split incompatible."
       )
   for i, (our_block, hf_block) in enumerate(
           zip(our_model.blocks, hf_model.transformer.h)):
       our_attn = our_block.attn
       # --- Attention: split fused c_attn into q/k/v ---
       c_attn_w = hf_block.attn.c_attn.weight          # [n_embd, 3*n_embd]
       c_attn_b = hf_block.attn.c_attn.bias            # [3*n_embd]
       W_q, W_k, W_v = c_attn_w.chunk(3, dim=1)        # each [n_embd, n_embd]
       b_q, b_k, b_v = c_attn_b.chunk(3, dim=0)        # each [n_embd]
       with torch.no_grad():
           our_attn.q_proj.weight.copy_(W_q.T); our_attn.q_proj.bias.copy_(b_q)
           our_attn.k_proj.weight.copy_(W_k.T); our_attn.k_proj.bias.copy_(b_k)
           our_attn.v_proj.weight.copy_(W_v.T); our_attn.v_proj.bias.copy_(b_v)
           our_attn.proj.weight.copy_(hf_block.attn.c_proj.weight.T)
           our_attn.proj.bias.copy_(hf_block.attn.c_proj.bias)
       # --- LayerNorms (nn.LayerNorm matches HF layout — direct copy) ---
       with torch.no_grad():
           our_block.ln_1.weight.copy_(hf_block.ln_1.weight)
           our_block.ln_1.bias.copy_(hf_block.ln_1.bias)
           our_block.ln_2.weight.copy_(hf_block.ln_2.weight)
           our_block.ln_2.bias.copy_(hf_block.ln_2.bias)
       # --- MLP: HF Conv1D weight [in, out]; nn.Linear weight [out, in] — transpose ---
       with torch.no_grad():
           our_block.mlp.c_fc.weight.copy_(hf_block.mlp.c_fc.weight.T)      # [4d, d]
           our_block.mlp.c_fc.bias.copy_(hf_block.mlp.c_fc.bias)            # [4d]
           our_block.mlp.c_proj.weight.copy_(hf_block.mlp.c_proj.weight.T)  # [d, 4d]
           our_block.mlp.c_proj.bias.copy_(hf_block.mlp.c_proj.bias)        # [d]

   # --- Embeddings and final LayerNorm ---
   # nn.Embedding.weight layout [vocab, d] matches HF — no transpose needed
   with torch.no_grad():
       our_model.wte.weight.copy_(hf_model.transformer.wte.weight)
       our_model.wpe.weight.copy_(hf_model.transformer.wpe.weight)
       our_model.ln_f.weight.copy_(hf_model.transformer.ln_f.weight)
       our_model.ln_f.bias.copy_(hf_model.transformer.ln_f.bias)
   ```
   **All Conv1D weights require transposing** (HF GPT-2 Conv1D stores weight as `[in, out]`
   regardless of the layer — attn QKV, attn output, MLP c_fc, MLP c_proj all use Conv1D).
   LayerNorm weights and embedding weights do NOT need transposing (same layout as PyTorch).
3. Initialize all NMM params (specific initializations matter — do NOT use a generic random-init):
   - `memory_mlp.W*.weight`: Xavier uniform (prevents activation saturation)
   - `W_θ/η/α`: default PyTorch Linear init (kaiming uniform)
   - `gamma_mem`: ones (always exists — init to 1 for identity scale)
   - `gamma_attn`: ones (only if `finetune_mode=False` — conditional creation, see task 2.4)
   - `persistent_mem`: randn * 0.02 (small random, like GPT-2 token embeddings)
   - `ln_nmm.weight`: ones, `ln_nmm.bias`: zeros (LayerNorm identity init)
   - conv weights: default PyTorch Conv1d init
   - `out_scale`: **zeros** (critical for `finetune_mode=True` — ensures y_mem=0 at init,
     preserving pretrained GPT-2 output exactly; use ones for training from scratch)
4. Save `TitansMAGGPT2` checkpoint using the same format as task 4.3:
   ```python
   torch.save({
       'state_dict': our_model.state_dict(),
       'config':     dataclasses.asdict(config),   # required — see task 4.3 / G134
       'step':       0,                              # fresh init — fine-tuning starts here
   }, 'titans_gpt2_init.pt')
   ```
   Saving `config` is mandatory so `scripts/finetune.py` and `eval.py` can rebuild the
   correct block structure (finetune_mode controls gamma_attn creation and out_scale init).

Sanity check: create a config with `nmm_n_persistent=0` (N_p=0) and `out_scale=0` or
`gamma_mem=0`. With N_p=0, `x_aug = x` (no persistent prefix) and the attention is
standard causal. With NMM zeroed, each block reduces to `x = x + attn(ln_1(x))` followed
by `x = x + mlp(ln_2(x))` — identical to HF GPT-2. With N_p>0, persistent tokens change
the softmax denominator even when zeroed, producing a small but nonzero discrepancy
(expected; not a bug). Use `nmm_n_persistent=0` for the clean equivalence check.

**Done:** sanity check with N_p=0, out_scale=0 passes; logit max-diff < 1e-4 vs HF GPT-2.

---

## Phase 3 — Data Pipeline

### 3.1 Tokenizer
`data/tokenizer.py`: thin wrapper around `tiktoken.get_encoding("gpt2")`. Concrete class:

```python
import tiktoken
import torch

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # G187: portable EOT-id lookup. Earlier passes wrote
        #     self.eot_token = self.enc.eot_token
        # which assumes `Encoding.eot_token` is a public attribute. This is true
        # for newer tiktoken versions (~0.7+) but NOT for older 0.5/0.6 — the
        # Encoding class exposes `_special_tokens` (private) and methods like
        # `encode_single_token` and `encode_special_token`, not a top-level
        # `eot_token` field. A reader on tiktoken 0.5 or 0.6 hits
        #     AttributeError: 'Encoding' object has no attribute 'eot_token'
        # at Tokenizer construction time — a confusing failure point.
        #
        # `encode_single_token("<|endoftext|>")` is the documented public API and
        # has been stable across all tiktoken versions we care about (≥0.5).
        # It returns 50256 for the "gpt2" encoding. This makes the Tokenizer
        # work regardless of which tiktoken version is installed, freeing us
        # from having to pin to ≥0.7 in requirements.txt (G182 still pins ≥0.5
        # for general encoding stability).
        self.eot_token = self.enc.encode_single_token("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        # `disallowed_special=()` permits any special token in `text` to be encoded
        # as regular BPE tokens. We do NOT pass `allowed_special={'<|endoftext|>'}`
        # here because users may pass arbitrary document text — special-token-aware
        # encoding belongs in `encode_corpus`, not the generic `encode`.
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def encode_corpus(self, documents) -> torch.Tensor:
        """Tokenize each document separately and join with the EOT token id.

        Returns a 1-D int64 LongTensor — the `token_stream` argument expected by
        `ParallelStreamLoader` (task 3.3).

        CRITICAL — silent-bug zone (G152): the naive shortcut
            ids = self.enc.encode("<|endoftext|>".join(documents))
        crashes with `ValueError: Encountered text corresponding to disallowed
        special token <|endoftext|>` because tiktoken's default `disallowed_special`
        is `"all"`. A reader fixing this by adding `disallowed_special=()` then
        encodes `<|endoftext|>` as ORDINARY BPE tokens (the literal characters
        `<`, `|`, `e`, ...), NOT as the EOT token id 50256. Downstream,
        ParallelStreamLoader's `(streams == eot_id)` mask is then all-False;
        `reset_state` is NEVER called at document boundaries; the NMM accumulates
        memory across the entire corpus → cross-document state leakage with no
        error and only a subtle quality regression. Same failure class as G151.

        The correct pattern: tokenize each doc individually, append the EOT id
        directly to the integer list (skipping the text-encoding step entirely
        for the special token). This is robust regardless of tiktoken's special-
        token policy.
        """
        ids: list[int] = []
        for doc in documents:
            ids.extend(self.enc.encode(doc, disallowed_special=()))
            ids.append(self.eot_token)   # append the integer id, NOT the text
        return torch.tensor(ids, dtype=torch.long)
```

**Done:** `decode(encode(s)) == s` for any ASCII string; `tok.eot_token == 50256`;
`tok.encode_corpus(["a", "b"])[-1].item() == tok.eot_token` (last token is EOT);
`(tok.encode_corpus(["a<|endoftext|>b"]) == tok.eot_token).sum().item() == 1`
(literal `<|endoftext|>` in user text is encoded as BPE characters, not the
special id — so the ONLY EOT in the output is the one appended after the doc).

### 3.2 Chunked document dataset (OPTIONAL — see G169 below)
`data/dataset.py`: `ChunkedDocumentDataset(path_or_hf_name, chunk_size)`:

**G169 — this task is OPTIONAL.** The primary loader for B>1 training is
`ParallelStreamLoader` (task 3.3), which consumes the 1-D token stream produced by
`Tokenizer.encode_corpus` directly — it does NOT wrap `ChunkedDocumentDataset`.
Earlier passes kept this task as a separate deliverable because the original plan
threaded `ChunkedDocumentDataset` into a vanilla `DataLoader`. After G151 introduced
`ParallelStreamLoader` to fix the parallel-streams TBPTT bug, this dataset became
orphaned but stayed in the plan, confusing readers who would build it and then never
use it. To remove ambiguity:

- For B>1 training and eval: build the 1-D token stream via `Tokenizer.encode_corpus`
  and pass it directly to `ParallelStreamLoader`. Skip task 3.2 entirely.
- For B=1 inference or a single-document analysis (e.g., a needle-in-haystack harness
  that only needs one continuous stream): `ChunkedDocumentDataset` is a valid primitive.
  Implement it only if you actually need it.

If implemented:

- Tokenize and concatenate documents using `Tokenizer.encode_corpus` (task 3.1).
  Do NOT call `tokenizer.encode` on a pre-joined string with literal `<|endoftext|>`
  separators — see the silent-bug warning in `encode_corpus` (G152).
- Yield `(input_ids [chunk_size], doc_boundaries [chunk_size])` where
  `doc_boundaries[t]` is True at the first token of each document
- Non-overlapping chunks; last short chunk **dropped** (not padded)

Dropping is strongly preferred over padding: padding requires propagating a padding mask
into the cross-entropy loss (`ignore_index` in task 4.2) and through the NMM gradients.
Dropping the last short chunk is simpler and loses at most `chunk_size - 1` tokens per
document — negligible at scale. If padding is desired, add a `pad_id` token and pass
`ignore_index=pad_id` to `F.cross_entropy` in task 4.2.

**This dataset is the single-stream primitive. For B>1 training, USE `ParallelStreamLoader`
DIRECTLY (task 3.3); do NOT wrap this in a vanilla `DataLoader(batch_size=B)`. The naive
batched-DataLoader path is the G151 silent bug — chunks 0..B-1 get collated into batch 0
but `nmm_states[i]` carries from batch N to batch N+1 along the wrong tokens.**

**Done (if implementing):** iterating over 1000 chunks produces no shape errors; boundaries
are correct (True at exactly the positions following `<|endoftext|>`); all yielded chunks
are exactly `chunk_size` tokens (no padding).

### 3.3 DataLoader — parallel-stream TBPTT batching
`data/dataloader.py`: a **parallel-stream batched** IterableDataset (NOT a plain DataLoader
on top of task 3.2). This is the correctness-critical piece — see G151.

**Why the obvious approach is wrong**: A plain `DataLoader(ChunkedDocumentDataset, batch_size=B)`
with `shuffle=False` collates chunks `[0, 1, …, B-1]` into batch 0, then `[B, B+1, …, 2B-1]`
into batch 1. The training loop carries `nmm_states[i]` from batch 0 position i to batch 1
position i. But batch 0 position 0 is chunk 0 and batch 1 position 0 is chunk B — they are
NOT consecutive (chunks 1..B-1 came between). The carried NMM state at position 0 is the
memory accumulated over chunk 0; feeding it into chunk B as if the state belonged to chunk B-1's
end silently corrupts memory. With B=4 chunk_size=512, position 0 sees states from a token
~1500 tokens in its past. The NMM behaves like random noise.

**Correct approach**: split the token stream into B parallel sub-streams, each yielding
chunks in order. Each batch is one chunk from each sub-stream. Then `nmm_states[i]` at
batch N+1 IS the continuation of `nmm_states[i]` at batch N for every i.

```python
class ParallelStreamLoader:
    def __init__(self, token_stream, batch_size, chunk_size, eot_id):
        # token_stream: 1-D LongTensor of the full corpus (concatenated, EOT-separated)
        N = (len(token_stream) // (batch_size * chunk_size)) * batch_size * chunk_size
        # Reshape into B parallel sub-streams of equal length (drops trailing remainder)
        self.streams = token_stream[:N].view(batch_size, -1)        # [B, S]
        self.B, self.S = self.streams.shape
        self.chunk_size = chunk_size
        self.num_chunks = self.S // chunk_size                       # chunks per stream
        # doc_boundaries: True at every token IMMEDIATELY FOLLOWING an EOT (start of new doc).
        # Precompute once over the full reshaped streams so chunk slicing is O(1).
        eot_mask = (self.streams == eot_id)                          # [B, S]
        boundaries = torch.zeros_like(self.streams, dtype=torch.bool)
        boundaries[:, 1:] = eot_mask[:, :-1]                         # shift right by 1
        # Also flag position 0 of stream as a doc boundary (start of the very first doc).
        boundaries[:, 0] = True
        self.boundaries = boundaries                                  # [B, S]

    def __iter__(self):
        for c in range(self.num_chunks):
            s = slice(c * self.chunk_size, (c + 1) * self.chunk_size)
            yield self.streams[:, s].contiguous(), self.boundaries[:, s].contiguous()

    def __len__(self):
        return self.num_chunks
```

This yields tuples `(input_ids [B, T], doc_boundaries [B, T])` directly — the training
loop iterates `for batch in loader:` and passes through to `train_step` unchanged.

**Call-site wiring** — the only correct way to construct the loader. Both `token_stream`
and `eot_id` MUST come from the same `Tokenizer` instance (task 3.1) so the EOT id used
to flag boundaries is the same id that was appended between documents during tokenization.
Mismatch — e.g., hard-coding `eot_id=50256` while constructing `token_stream` with a
different tokenizer or vocabulary — would silently produce all-False boundaries (G152):

```python
tok = Tokenizer()
token_stream = tok.encode_corpus(documents)             # 1-D int64 LongTensor
loader = ParallelStreamLoader(
    token_stream,
    batch_size = 4,
    chunk_size = config.chunk_size,
    eot_id     = tok.eot_token,                          # SAME tokenizer → matches token_stream
)
```

**Shuffling**: shuffle at the **stream level**, not chunk level — i.e., randomize the
mapping from stream index to corpus offset between epochs (rebuild the loader between
epochs with a different starting offset per stream). Within an epoch, each stream's chunks
stay in order so TBPTT state is valid. Do NOT shuffle inside `__iter__`.

**Document-level shuffle** (an alternative): shuffle documents BEFORE concatenating into
`token_stream`. Cross-document state leakage is bounded by the reset_state mechanism at
document boundaries (task 1.9).

**Done:** for a synthetic corpus of `B * chunk_size * 100` tokens with batch_size=4,
chunk_size=512: loader yields 100 batches, each shaped `([4, 512], [4, 512] bool)`;
position-i sequence across consecutive batches forms a contiguous token stream (verifiable
by `torch.equal(batch_n[:, i, -1].roll(1), batch_n_plus_1[:, i, 0])`-style check on
position-i token boundary).

**G191 — DDP / multi-GPU sharding.** `ParallelStreamLoader` as written above is
single-process. With `DistributedDataParallel(model)` and `torch.distributed.launch`
spawning N ranks, the natural pattern is:

```python
# WRONG — each rank constructs the same loader and sees the same data:
tok = Tokenizer()
token_stream = tok.encode_corpus(documents)
loader = ParallelStreamLoader(token_stream, batch_size=4, chunk_size=512,
                              eot_id=tok.eot_token)
for batch in loader:
    train_step(model, batch, ...)
```

Every rank yields IDENTICAL batches. The effective batch size stays at `batch_size`
(=4) regardless of N — DDP averages identical gradients across ranks, no statistical
benefit. The user sees "linear scaling didn't help" and blames the model architecture
or NMM state coordination, when really the data loading is the bug. This is silent:
no error, training proceeds, gradients are averaged correctly (just over duplicate
data), per-rank metrics look identical (because they ARE identical).

The fix: each rank reads a DIFFERENT slice of the token stream, but EACH rank still
internally maintains `batch_size` parallel sub-streams. The way to do this without
losing TBPTT continuity is to **partition the corpus into N contiguous segments and
give rank-r the r-th segment**, then run ParallelStreamLoader inside that segment.

```python
import torch.distributed as dist

class ParallelStreamLoader:
    def __init__(self, token_stream, batch_size, chunk_size, eot_id,
                 rank=None, world_size=None):
        # G191: rank/world_size args partition the corpus across DDP ranks.
        # Default to single-process when not in a distributed context (the
        # standalone-script user path).
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Split the token stream into `world_size` contiguous segments. Rank r
        # gets segment r. Within its segment, rank r still maintains
        # `batch_size` parallel sub-streams.
        N_total = len(token_stream) // world_size
        # Round N_total down to a multiple of (batch_size * chunk_size) so each
        # rank's loader has the same number of batches per epoch — required for
        # synchronous DDP step-counting (all ranks must reach optimizer.step()
        # together; differing batch counts cause hang at the AllReduce barrier).
        N_per_rank = (N_total // (batch_size * chunk_size)) * batch_size * chunk_size
        seg_start = rank * N_per_rank
        seg_end   = seg_start + N_per_rank
        seg = token_stream[seg_start:seg_end]

        self.streams = seg.view(batch_size, -1)
        self.B, self.S = self.streams.shape
        self.chunk_size = chunk_size
        self.num_chunks = self.S // chunk_size
        eot_mask = (self.streams == eot_id)
        boundaries = torch.zeros_like(self.streams, dtype=torch.bool)
        boundaries[:, 1:] = eot_mask[:, :-1]
        # boundaries[:, 0] is True on rank 0 (start of corpus) but ALSO on every
        # other rank (start of segment — the previous rank's last token in the
        # corpus is NOT this rank's "previous token"). This is intentional: each
        # rank starts fresh, no cross-rank NMM state continuity. The
        # `boundaries[:, 0] = True` ensures the per-rank loop resets the NMM
        # state at the start of its segment regardless of what came before.
        boundaries[:, 0] = True
        self.boundaries = boundaries
```

Per-rank gradient sync via DDP still works as expected: each rank's gradient is
computed on its own data, AllReduce averages them. Effective batch is
`batch_size * world_size`. ✓

Caveats:
- Stream count: each rank has `batch_size` streams. Global stream count is
  `batch_size * world_size`. If the user wanted exactly `batch_size` total
  streams (e.g., for hyperparameter equivalence to a single-GPU run), they
  should use `batch_size = total_batch / world_size` per rank.
- NMM state continuity: per-rank only. Tokens at the boundary between rank r's
  segment and rank r+1's segment are NOT consecutive from the NMM's POV — but
  this is fine because each rank's `boundaries[:, 0] = True` flag resets at
  segment start. Cross-rank continuity isn't needed for TBPTT correctness.
- Determinism: the same `torch.manual_seed(seed)` produces the same partitioning
  on every rank, so the segment assignment is deterministic without extra
  coordination.

**G170 — RAM at production scale.** `Tokenizer.encode_corpus` (task 3.1) returns a 1-D
int64 `LongTensor`. For FineWebEdu-10BT (the paper's training corpus), that's
`10e9 * 8 bytes ≈ 80 GB` — does not fit in RAM on a typical training machine.
`ParallelStreamLoader.__init__` then calls `.view(B, -1)` on this tensor (no copy —
view is free) and computes `eot_mask = (self.streams == eot_id)` (allocates a `[B, S]`
bool tensor — another ~10 GB at 10B tokens). The implementation as-written is fine for
development-scale corpora (OpenWebText, WikiText-103, FineWeb-EDU-sample at <1B
tokens), but for the paper-scale 10BT run you need streaming/memmap.

The recommended scaling path:

1. **Streaming tokenization to disk.** Tokenize the corpus document-by-document and
   append directly to an `np.memmap` file of dtype `int32` (50257 < 2^31, so int32 is
   sufficient — halves the on-disk size vs int64). Write the EOT id between documents
   as part of the same append.

   ```python
   import numpy as np
   N_DOCS = number_of_docs_in_iterable
   approx_tokens = N_DOCS * AVG_TOKENS_PER_DOC * 1.05   # 5% slack
   mm = np.memmap('corpus.bin', dtype=np.int32, mode='w+', shape=(approx_tokens,))
   tok = Tokenizer()
   write_idx = 0
   for doc in iter_documents():                          # streaming reader, NOT list
       ids = tok.encode(doc)
       mm[write_idx : write_idx + len(ids)] = ids
       write_idx += len(ids)
       mm[write_idx] = tok.eot_token
       write_idx += 1
   mm.flush()
   # Record actual final length somewhere (a sidecar .meta JSON is the usual pattern).
   ```

2. **Reading via memmap.** `ParallelStreamLoader` then takes a memmapped array (NOT a
   torch tensor). Modify `__init__` to accept either a `torch.LongTensor` (small corpus,
   in-RAM) OR a numpy memmap (large corpus). For the memmap branch, the per-batch slice
   `self.streams[c*chunk_size:(c+1)*chunk_size]` reads only the chunk's worth of bytes
   from disk — the OS page cache handles prefetching, and total RAM is
   ~`B * chunk_size * 8 bytes` per batch (kilobytes, not gigabytes). The precomputed
   `[B, S]` bool boundaries tensor would itself be 10 GB at 10BT, so we compute
   boundaries on-the-fly per chunk.

   **G192 — cross-chunk boundary continuity.** The naive per-chunk boundary
   computation:
   ```python
   eot_mask_chunk = (chunk == eot_id)
   boundaries_chunk = torch.zeros_like(chunk, dtype=torch.bool)
   boundaries_chunk[:, 1:] = eot_mask_chunk[:, :-1]
   ```
   is INCORRECT at the chunk's first position. `boundaries[:, t]` is True iff
   `streams[:, t-1]` was an EOT — but the chunk's position 0 has no `t-1` IN
   THIS CHUNK; it has `t-1` in the PREVIOUS chunk. The naive version always reports
   `boundaries_chunk[:, 0] = False`, missing every doc boundary that falls
   exactly at a chunk boundary. The NMM state then DOESN'T reset at those
   boundaries → cross-document state leakage exactly like G152, except triggered
   only at chunk boundaries (rare but not negligible: at chunk_size=512 and
   average doc length 1000 tokens, ~50% of doc boundaries land on a chunk
   boundary).

   The fix: carry a one-token "previous chunk last token" buffer across iterations
   and use it to compute `boundaries_chunk[:, 0]` from the prior chunk's last
   token. For the very first chunk of a rank's segment, `boundaries[:, 0] = True`
   unconditionally (matches G191's per-rank reset semantics).

   ```python
   class ParallelStreamLoader:
       def __init__(self, source, batch_size, chunk_size, eot_id,
                    rank=None, world_size=None):
           # G193 — store eot_id as self.eot_id. The per-chunk boundary computation
           # in __iter__ references it; earlier drafts of this snippet referenced bare
           # `eot_id` (NameError at runtime). The in-RAM variant only needs eot_id in
           # __init__ (precomputes boundaries upfront), but the memmap variant needs
           # it per-chunk.
           self.eot_id = eot_id

           # G196 — memmap + DDP combined. G191 introduced rank/world_size for the
           # in-RAM variant; this memmap variant accepts the same args. The pipeline is:
           #   1. Determine `total_tokens = len(source)` — works for both torch.LongTensor
           #      (.shape[0] via len()) and numpy memmap (.shape[0] via len()).
           #   2. Round down to a multiple of (B*chunk_size*world_size) so each rank has
           #      the same number of batches (synchronous DDP step counting requires this).
           #   3. Slice the per-rank segment. For torch tensor: `source[start:end]`.
           #      For memmap: `source[start:end]` ALSO works — numpy memmap supports
           #      slicing, and the slice IS still a memmap view (no copy, no RAM blowup).
           #   4. Reshape to [batch_size, S_per_stream]. For torch tensor: `.view(...)`.
           #      For memmap: `.reshape(...)` — view raises if the memmap is non-
           #      contiguous, but a contiguous slice supports reshape directly.
           import torch.distributed as dist
           if rank is None:
               rank = dist.get_rank() if dist.is_initialized() else 0
           if world_size is None:
               world_size = dist.get_world_size() if dist.is_initialized() else 1

           total_tokens = len(source)
           N_per_rank = (total_tokens // (batch_size * chunk_size * world_size)) \
                        * (batch_size * chunk_size)
           seg_start = rank * N_per_rank
           seg_end   = seg_start + N_per_rank
           seg = source[seg_start:seg_end]

           # Reshape into B parallel sub-streams. Both torch.view and numpy.reshape
           # produce a [B, S] view on contiguous memory. We use `.reshape(...)` for
           # numpy (memmap's .reshape returns a memmap view if contiguous, a copy
           # otherwise — slicing a contiguous segment gives a contiguous view, so
           # reshape stays a view).
           if isinstance(seg, torch.Tensor):
               self.streams = seg.view(batch_size, -1)
           else:                                                # numpy memmap
               self.streams = seg.reshape(batch_size, -1)
           # Both expose `.shape`, `.__getitem__` consistently. Below, isinstance
           # checks in __iter__ branch on tensor vs memmap.

           # Don't precompute the [B, S] boundaries tensor — at 10BT scale it's
           # itself ~10GB, defeating the memmap RAM-savings.
           self.chunk_size = chunk_size
           self.num_chunks = self.streams.shape[1] // chunk_size

       def __iter__(self):
           # Track the last token per stream from the PRIOR chunk, used to compute
           # boundaries[:, 0] of the next chunk. Initial value None signals "first
           # chunk — boundaries[:, 0] = True unconditionally (G191 semantics)."
           prev_last = None
           for c in range(self.num_chunks):
               s = slice(c * self.chunk_size, (c + 1) * self.chunk_size)
               # Slice from memmap (or tensor; both support [:, s] notation when
               # `streams` is shaped [B, S]).
               chunk_raw = self.streams[:, s]                      # [B, T] int32 (memmap) or int64
               chunk = (torch.from_numpy(chunk_raw).long()
                        if isinstance(chunk_raw, np.ndarray)
                        else chunk_raw.long())
               # Per-chunk boundaries — G193: use self.eot_id, not bare eot_id.
               eot_mask = (chunk == self.eot_id)                   # [B, T]
               boundaries = torch.zeros_like(chunk, dtype=torch.bool)
               boundaries[:, 1:] = eot_mask[:, :-1]                # standard shift-right
               # Cross-chunk position 0: True if the PRIOR chunk's last token was EOT,
               # OR if this is the very first chunk of the rank's segment (G191).
               if prev_last is None:
                   boundaries[:, 0] = True
               else:
                   boundaries[:, 0] = (prev_last == self.eot_id)   # [B] broadcast → [B, 1]
               # Update prev_last for the next iteration.
               prev_last = chunk[:, -1]
               yield chunk, boundaries

       def __len__(self):
           return self.num_chunks
   ```

   The `prev_last` buffer is `[B]` int — negligible RAM. The first-chunk
   special case matches the in-RAM ParallelStreamLoader's `boundaries[:, 0] = True`
   semantics. Cross-chunk boundary detection now works correctly for memmap.

3. **Conversion at yield time.** `torch.from_numpy(slice).long()`. The int32 memmap
   slice is widened to int64 on the host, then transferred to GPU. This adds
   negligible latency vs. compute.

The default implementation in the snippet above is the in-RAM version — keep it for
development clarity. Add a memmap branch to `__init__` when targeting >2-3 GB of token
ids. The G170 line in GAP_HISTORY documents the threshold and the failure mode if you
try the in-RAM version at 10BT (the tokenization itself OOMs before the loader is
constructed, so the failure is loud — but the user wastes a multi-hour tokenization
run before discovering it).

---

## Phase 4 — Training

### 4.1 Optimizer — four parameter groups (gpt2/nmm × decay/no-decay)
```python
no_decay = {'bias', 'ln', 'norm', 'out_scale', 'gamma', 'persistent'}
# 'ln' catches ln_1, ln_2, ln_nmm, ln_f (LayerNorm params)
# 'norm' catches memory_mlp.norm (ResidualNorm — fixed stabilizer)
# 'out_scale' — magnitude gate init=0, decay would resist learning
# 'gamma' — scale gates init=1, decay would suppress memory branch toward zero
# 'persistent' — learnable prefix embeddings (like learned position embeddings — never decayed)

def _make_param_groups(named_params, lr):
    """Split named params into decay and no-decay sub-groups at the given lr."""
    decay, no_dc = [], []
    for n, p in named_params:
        if any(nd in n for nd in no_decay):
            no_dc.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'lr': lr, 'weight_decay': 0.1},
        {'params': no_dc, 'lr': lr, 'weight_decay': 0.0},
    ]

gpt2_named = [(n, p) for n, p in model.named_parameters()
              if not any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]
nmm_named  = [(n, p) for n, p in model.named_parameters()
              if any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]

optimizer = AdamW(
    _make_param_groups(gpt2_named, lr=1e-4) +
    _make_param_groups(nmm_named,  lr=3e-4),
    betas = (0.9, 0.95),   # G153: NOT the PyTorch default (0.9, 0.999) — see below.
    eps   = 1e-8,
)
```

**AdamW betas**: We pass `betas=(0.9, 0.95)`, NOT PyTorch's default `(0.9, 0.999)`. β2=0.999
has a ~1000-step adaptation horizon — far too smooth for the per-token NMM surprise gradients,
which pass through Newton-Schulz (which bounds spectral norm) but whose per-element values
are still noisy and non-stationary. β2=0.95 has a ~20-step horizon: matches the noise scale,
gives stable updates without lagging behind. This is the standard LM-training value used by
both the Titans paper and nanoGPT; if you omit `betas=`, the silent failure mode is over-smoothed
second moments that under-estimate the effective gradient variance early in training, producing
oversized updates and divergence on a fraction of runs (no error — just unlucky seeds).

**Critical**: The `no_decay` set must be IMPLEMENTED in the optimizer, not just documented.
The previous 2-group pattern (`AdamW([gpt2_params, nmm_params], weight_decay=0.1)`) applies
`weight_decay=0.1` uniformly — LayerNorms, biases, `out_scale`, `gamma_*`, and
`persistent_mem` would all receive weight decay. The 4-group pattern above correctly applies
`weight_decay=0.0` to no-decay params within each learning-rate group. Total: 4 param groups.

`persistent_mem` holds learnable prefix embeddings (analogous to learned position embeddings).
Weight decay shrinks them toward zero, reducing their representational capacity. Excluded via
the `'persistent'` substring match on `persistent_mem`.

Do NOT freeze GPT-2 weights — Titans Revisited shows NMM-only training against a frozen
backbone fails because KV projections are misaligned with how memory evolves.

**Done:** `len(optimizer.param_groups) == 4`; groups have correct lr and weight_decay pairs;
no parameter appears in more than one group; no parameter is missing.

### 4.2 TBPTT training step
```python
# Before the training loop: nmm_states = None  (model.forward initializes on first call)
def train_step(model, batch, nmm_states, optimizer, device):
    # G167: device is an explicit argument; the loader yields CPU tensors. See the
    # bf16-autocast train_step below for the full rationale + non_blocking note.
    input_ids, doc_boundaries = batch       # [B, T] on CPU
    input_ids      = input_ids.to(device, non_blocking=True)
    doc_boundaries = doc_boundaries.to(device, non_blocking=True)
    nmm_states = detach_states(nmm_states)  # None-safe (see task 1.9); breaks gradient TBPTT
    logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
    # Use logits.size(-1) for vocab dim; `vocab_size` is not in scope here unless explicitly
    # passed/closed over. Avoid the NameError by reading the dim from the tensor itself.
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1)
    )
    loss.backward()

    # Capture the pre-clip grad_norm — task 4.3 logs it; without capturing here the
    # user has to re-implement clipping in their training loop to get the number.
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # G158: NaN/Inf guard. `clip_grad_norm_` does NOT clip when the total norm is
    # non-finite — the comparison `NaN > max_norm` evaluates to False, so the
    # clipping branch is skipped and `optimizer.step()` would then apply the NaN
    # gradients directly to the parameters. One bad batch silently corrupts every
    # weight in the model; all subsequent forward passes produce NaN logits and the
    # loss stays NaN forever. The script doesn't crash — the only signal is `loss=nan`
    # in the next log line. With our setup (torch.func.grad's second-order graph over
    # a T=512 chunked recurrence, Newton-Schulz on per-token gradient matrices), one-in-
    # a-thousand-step NaN is plausible enough to need a guard, not a hope.
    if not torch.isfinite(grad_norm):
        # Skip this step entirely: don't apply the bad gradients to params, but DO
        # zero them so the next backward starts clean. Returning loss.item() as NaN
        # lets the caller log/alert on it. G213 — return `None` for nmm_states
        # (NOT the existing state) so the caller's next call triggers
        # `init_state` via model.forward's None-detection path. The current
        # state is almost certainly NaN-tainted (a NaN gradient norm implies
        # some param's gradient was NaN, which means the forward chain
        # produced NaN — typically inside the NMM recurrence, contaminating
        # M and S). Returning the NaN state would propagate NaN through the
        # next chunk's forward → NaN loss → NaN grad → G158 skip again →
        # infinite NaN-stuck loop until the next doc boundary fires
        # reset_state. The trade-off: NaN events lose the in-flight memory
        # state. Better than a permanently-dead NMM.
        optimizer.zero_grad(set_to_none=True)
        return loss.item(), None, grad_norm.item()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item(), nmm_states, grad_norm.item()
```

`doc_boundaries` is passed into the model so the NMM resets state at every document
start within the chunk, not just at chunk boundaries.

`detach_states` recursively calls `.detach()` on all tensors in the `(M, S)` state
tuples (see task 1.9). This is what implements TBPTT — gradients do NOT flow from
one chunk to the previous chunk's parameters, but DO flow through all steps within
the current chunk.

**Return-tuple change (G158)**: `train_step` now returns `(loss, nmm_states, grad_norm)`
— grad_norm is the pre-clip total norm, needed by task 4.3's logger. Callers that
unpack as `loss, nmm_states = train_step(...)` (the earlier 2-tuple signature) will
get a `ValueError: too many values to unpack` on the first step — loud failure, not
silent. If you'd rather not refactor the caller, accept `grad_norm` with `_`:
`loss, nmm_states, _ = train_step(...)`.

**Mixed precision — bf16 autocast pattern (G159).** GPT-2-small in fp32 with our NMM
state (~2.7 GB for B=4, n_layer=12) plus optimizer state plus T=512 activation
graph hits 24 GB GPU memory fast. Mixed precision is in practice required. The plan
sprinkles hints about bf16/fp16 support (`_aug_mask` takes a `dtype` arg, "use
bfloat16 for states to halve this" in task 1.7) but the safe RECIPE is not obvious
from those hints, and three silent failure modes await a reader who guesses:

- **`model.half()` (full fp16 weights, no master fp32 copy).** AdamW updates of size
  `lr * m / sqrt(v+eps)` ≈ `1e-4 * small` are below fp16's ~6e-5 representable
  minimum and round to zero. After thousands of steps, parameters silently flatline.
  Loss curves look "smooth" but the model isn't actually learning. NEVER do this for
  our setup.
- **`torch.autocast(dtype=torch.float16)` without `GradScaler`.** fp16 has ~6 orders
  of magnitude dynamic range. Gradients through our T=512-deep chunked recurrence
  with `torch.func.grad`'s second-order graph easily underflow. The optimizer sees
  zero gradient → no update. Same silent flatline mode. If you MUST use fp16 (e.g.,
  hardware without bf16), use `torch.cuda.amp.GradScaler` and call `scaler.scale(loss).backward()`,
  `scaler.unscale_(optimizer)` (before `clip_grad_norm_`), `scaler.step(optimizer)`,
  `scaler.update()`. But bf16 is strictly preferable on hardware that supports it.
- **Wrapping `loss.backward()` inside autocast.** Autocast's scope rules for backward
  are subtle — different ops are registered for forward vs. backward, and `clip_grad_norm_`
  inside autocast computes the norm in low precision. Keep backward and clipping
  OUTSIDE the autocast block.

The safe pattern is bf16 autocast for forward+loss only, with model parameters in
fp32 (the master copy). bf16 has fp32's dynamic range — gradients don't underflow,
no scaler needed. Hopper / Ampere / RDNA3+ all support it natively.

```python
@torch.no_grad()  # NO — that's eval; for train we want grad
def _placeholder_to_avoid_confusion(): ...

def train_step(model, batch, nmm_states, optimizer, device):
    # G167: the loader yields CPU tensors (ParallelStreamLoader's `self.streams` is the
    # original `token_stream` tensor reshaped — created on CPU and never moved). The
    # model lives on `device`. Without the explicit transfer, `model(input_ids, ...)`
    # raises a "tensors on different devices" error on the first batch.
    # We accept `device` as an explicit arg (NOT inferred from
    # `next(model.parameters()).device`) so the caller's intent is visible at the
    # call site — the perplexity helper in task 5.2 takes the same shape, and keeping
    # them symmetric makes the train/eval pair easy to grep.
    # non_blocking=True is a hint to overlap the H2D copy with compute. It only helps
    # if the upstream tensor is in pinned memory; ParallelStreamLoader does NOT pin,
    # so the hint is effectively a no-op today. It's set anyway because flipping the
    # loader to pinned memory later requires zero code changes here.
    input_ids, doc_boundaries = batch
    input_ids      = input_ids.to(device, non_blocking=True)
    doc_boundaries = doc_boundaries.to(device, non_blocking=True)
    nmm_states     = detach_states(nmm_states)

    # G159: forward + loss in bf16 autocast. Backward/clip/step in fp32.
    # device_type='cuda' is required; on CPU you'd use 'cpu' with bf16.
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

    # Backward + clipping outside autocast: gradient computation uses each op's
    # natural dtype (bf16 for ops registered in autocast forward), but the
    # accumulated `.grad` tensors on the fp32 parameters are themselves fp32
    # (PyTorch handles this automatically). clip_grad_norm_ then computes the
    # norm in fp32 — what we want.
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        # G213 — also RESET nmm_states to None on NaN so the caller's next call
        # re-initializes them via init_state. Pre-G213 we returned the existing
        # nmm_states unchanged, but the state IS LIKELY NaN-tainted at this
        # point: a NaN in grad_norm means at least one parameter's gradient
        # was NaN, which means SOMETHING in the forward graph produced NaN
        # (NMM gradient blowup, division pathologies in F.normalize on near-
        # zero vectors, NS5 instability on extreme inputs, autocast overflow
        # in a hot loop). The most common source is the NMM recurrence itself
        # — a single bad token's gradient blows up M_t to inf, then NS5 of
        # inf produces NaN, then S, M, y are all NaN. The state tuple
        # (M, S) returned to the caller IS the post-forward state — fully
        # NaN-contaminated.
        # Without the reset: next forward starts with NaN M → NaN g_t → NaN
        # g_tilde → NaN S, M → NaN logits → NaN loss → NaN grad_norm →
        # G158 skip → return same NaN state. Loop FOREVER (until the next
        # doc boundary fires reset_state, which uses init_M and recovers).
        # For a long document without any EOTs, the NMM is dead for the
        # remainder of that document. The user sees `loss=nan` for many
        # steps in a row with no obvious recovery path. With the reset:
        # next forward sees nmm_states=None → init_state builds fresh M
        # from memory_mlp.W*.weight → recovery.
        # The reset costs one chunk's worth of NMM memory continuity. That's
        # cheap insurance against an indefinite NaN-stuck training loop.
        # Trade-off: NaN events lose the in-flight memory state. Better to
        # lose a few thousand tokens of memory than to silently train with
        # a permanently-dead NMM until the next doc boundary.
        optimizer.zero_grad(set_to_none=True)
        return loss.item(), None, grad_norm.item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item(), nmm_states, grad_norm.item()
```

The autocast block does NOT need to be inside a `with torch.no_grad()`, because
training requires gradients. The eval pattern in task 5.2 uses both decorators
(`@torch.no_grad()` outermost, autocast for inference acceleration). For eval, the
autocast wrap is optional — eval doesn't have the bf16-vs-fp32 memory pressure
that training does.

For our NMM specifically: `torch.func.grad` inside the chunked forward is invoked
within the autocast scope. PyTorch's torch.func supports autocast, but the
second-order gradient through bf16 ops has slightly different precision than fp32.
Empirically this is fine for our small-scale runs; if you see training instabilities
that don't appear in fp32, the second-order grad is a suspect — try fp32 first
before assuming Newton-Schulz or chunk_size are the cause.

**Done:** loss decreases monotonically on a 100-step overfit run on a single batch;
injecting `loss = loss + float('nan')` at step 50 of an overfit run does NOT corrupt
the parameters (verifiable: `all(torch.isfinite(p).all() for p in model.parameters())`
remains True after the NaN-injected step, and the loop continues training normally on
subsequent good batches). Without the G158 guard, that assertion fails at step 51.

### 4.3 LR schedule and logging
- Cosine decay with linear warmup (1000 steps warmup). Concrete code below.
- Log every 50 steps: `loss`, `grad_norm`, `lr`, `nmm_state_norm` (‖M‖_F per layer)

**LR schedule — multi-group update (G157).** With four param groups (task 4.1) at two
different base LRs (1e-4 for gpt2, 3e-4 for nmm), the LR schedule MUST scale all groups
in unison. Compute a scalar multiplier (0 → 1) and apply it to each group's *initial*
LR, preserving the gpt2-vs-nmm ratio. Two silent traps the natural code falls into:

- **Single-group update** (`optimizer.param_groups[0]['lr'] = lr`): only updates group 0.
  Groups 1/2/3 keep their initial LRs forever — never warmed up, never decayed. By
  end of cosine, group 0 → 0 while nmm groups still run at 3e-4.
- **Uniform LR clobber** (`for g in optimizer.param_groups: g['lr'] = lr`): destroys the
  1:1:3:3 ratio. All groups end up at the same LR, defeating the 3× NMM LR rationale.

```python
import math

def get_lr_multiplier(step, warmup_steps=1000, max_steps=100_000, min_ratio=0.1):
    """Returns a scalar in [min_ratio, 1.0]. Multiply each param_group's base LR by this."""
    # G197: validate max_steps > warmup_steps. The two natural misconfigurations:
    #   - max_steps == warmup_steps: cosine has zero length. progress = (step -
    #     warmup_steps) / 0 → ZeroDivisionError on the first cosine-branch step.
    #     But before that, the `if step >= max_steps: return min_ratio` clamp at
    #     step == max_steps == warmup_steps catches it; off-by-one at step =
    #     warmup_steps - 1 = max_steps - 1 stays in linear branch and works.
    #     So in this degenerate case the function actually works — but the user
    #     intended a cosine that never decays, which is misleading.
    #   - max_steps < warmup_steps: at step in [max_steps, warmup_steps), the
    #     `if step < warmup_steps` check fires FIRST and returns the linear
    #     warmup value. The `step >= max_steps` cosine-clamp is never reached.
    #     Result: LR keeps linearly warming up past max_steps, NEVER decays.
    #     User sees "LR doesn't decay" and blames the cosine implementation.
    # Both are config bugs that produce silently-wrong schedules. Raise
    # explicitly to surface the mis-config at first call.
    if max_steps <= warmup_steps:
        raise ValueError(
            f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps}). "
            f"With max_steps <= warmup_steps, the cosine branch is degenerate "
            f"(divide-by-zero or never reached). For a warmup-only schedule "
            f"with no decay, set min_ratio=1.0 and max_steps just beyond your "
            f"intended training duration. (See G197 in GAP_HISTORY.md.)"
        )
    if step < warmup_steps:
        return step / warmup_steps                # linear warmup: 0 → 1 over warmup_steps
    if step >= max_steps:
        return min_ratio                            # floor for stability post-decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    # Cosine from 1.0 (at warmup end) down to min_ratio (at max_steps).
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

# G162: derive `base_lrs` from CODE-LEVEL CONSTANTS, NOT from `optimizer.param_groups[i]['lr']`.
# An earlier version of this section said "regenerate base_lrs from optimizer.param_groups
# after construction." That is silently wrong on checkpoint resume:
# `optimizer.load_state_dict(ckpt['optimizer'])` restores the saved `param_groups` —
# including the 'lr' field — which is whatever apply_lr had set at save time (mid-cosine,
# not peak). Capturing `[g['lr'] for g in optimizer.param_groups]` AFTER load then captures
# the deflated mid-cosine value, and every subsequent apply_lr call multiplies the deflated
# base by lr_mul. Each resume compounds the deflation; after a few save/resume cycles, the
# effective LR is essentially zero, training plateaus, and the user blames the cosine
# schedule.
#
# The fix: peak LRs are configuration, not state. Define them as constants in code, use
# them BOTH when constructing the optimizer AND when defining base_lrs. Then it doesn't
# matter whether base_lrs is captured before or after load_state_dict — neither path
# reads the (mutable, deflated) `param_groups[i]['lr']`.

GPT2_PEAK_LR = 1e-4
NMM_PEAK_LR  = 3e-4

optimizer = AdamW(
    _make_param_groups(gpt2_named, lr=GPT2_PEAK_LR) +
    _make_param_groups(nmm_named,  lr=NMM_PEAK_LR),
    betas = (0.9, 0.95),
    eps   = 1e-8,
)

# base_lrs MUST mirror the construction order above:
#   [gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay]
# Derived from constants — resume-safe. NEVER write
#     base_lrs = [g['lr'] for g in optimizer.param_groups]
# unless you do it strictly BEFORE `optimizer.load_state_dict(...)` (and even then, the
# constants-from-code form is preferred for clarity).
base_lrs = [GPT2_PEAK_LR, GPT2_PEAK_LR, NMM_PEAK_LR, NMM_PEAK_LR]

# Inside the training loop, BEFORE optimizer.step():
def apply_lr(optimizer, base_lrs, step, warmup_steps=1000, max_steps=100_000, min_ratio=0.1):
    # G175: the schedule's `max_steps`/`warmup_steps` MUST match the user's actual
    # training duration. Earlier versions of this snippet called
    # `get_lr_multiplier(step)` with no arguments, silently picking up the function
    # default `max_steps=100_000`. If the user's loop runs for FEWER steps (e.g., 20K
    # for a fast fine-tune), the cosine is only ~20% progressed when training ends —
    # the LR stays near peak the whole run, the model trains at higher-than-intended
    # LR throughout, and "loss seems noisy at the end" gets blamed on optimization
    # noise rather than the schedule mismatch. If the user's loop runs LONGER (e.g.,
    # 500K), the schedule hits `step >= max_steps` at step 100K and clamps to
    # `min_ratio` for the remaining 400K steps — the user effectively trains the
    # last 80% of the run at 0.1× peak with no further decay (sometimes desirable,
    # often unintended). Same kind of silent miscalibration if `warmup_steps` is
    # decoupled: the natural reading "I'll train for 200 steps to overfit a batch"
    # would have warmup=1000 default → entire 200-step run is in warmup → LR never
    # exceeds 200/1000 = 0.2× peak.
    #
    # Pass the SAME `warmup_steps`/`max_steps` here that you used to derive your
    # loop's termination. The consolidated training loop in task 4.5 (G171) shows
    # the full wiring: max_steps = N_EPOCHS * num_batches_per_epoch, passed both
    # to apply_lr and to the loop's `if step >= max_steps: break` check.
    lr_mul = get_lr_multiplier(step, warmup_steps=warmup_steps,
                               max_steps=max_steps, min_ratio=min_ratio)
    for g, base_lr in zip(optimizer.param_groups, base_lrs):
        g['lr'] = base_lr * lr_mul
    return lr_mul                                       # for logging

# Logging emits lr_mul; per-group LRs are derivable as base_lrs * lr_mul.
```

`max_steps` should be set to your full training budget so cosine completes at the end of
training. `min_ratio=0.1` keeps a 10% floor — cosine-to-zero is harsher and often
underperforms a small-but-nonzero terminal LR.

On checkpoint resume (task 4.3's resume sequence), `base_lrs` is **redefined from the
same `GPT2_PEAK_LR` / `NMM_PEAK_LR` constants** that the optimizer was built with — NOT
re-read from `optimizer.param_groups` after `load_state_dict`. The schedule then picks
up at `step = ckpt['step']` and produces the correct multiplier for that step. **Do not
save `base_lrs` in the checkpoint** — peak LRs are configuration, not state; the
constants live in code.

**G172 — `compute_nmm_norm` helper.** The log line specifies `nmm_state_norm` (‖M‖_F
per layer) but earlier passes never provided a concrete implementation. Without one,
users either skip the metric or roll inconsistent ad-hoc versions (some sum across W
keys, some report only W1, some compute on the SQUARED Frobenius — all giving different
numbers that can't be compared across runs). Standardize:

```python
def compute_nmm_norm(nmm_states):
    """Per-layer Frobenius norm of M, averaged across batch.

    Returns a list of floats, one per layer, suitable for logging or plotting.
    Use this for tracking NMM weight magnitude over training — a useful sanity
    signal for whether Newton-Schulz + momentum is keeping weights bounded
    (paper §3.2 stability claim) or if updates are exploding/decaying.

    Aggregation choices, fixed:
      - Sum the squared Frobenius norms across the three W keys (W1, W_gate, W2),
        then sqrt — this gives ‖M‖_F treating the three matrices as one block.
      - Mean across the batch dimension — easier to compare runs at different B.
      - Detach + float — caller is logging, not computing gradient through this.

    Returns `None` for layers where state is None (first step before init).
    """
    if nmm_states is None:
        return None
    out = []
    for M, _S in nmm_states:
        # M: dict of [B, h, d] tensors
        sq_sum = sum((v.float() ** 2).sum(dim=(-2, -1)) for v in M.values())  # [B]
        out.append(sq_sum.sqrt().mean().detach().item())
    return out
```

Use at logging time:
```python
if step % 50 == 0:
    norms = compute_nmm_norm(nmm_states)
    print(f"step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} "
          f"lr_mul={lr_mul:.4f} nmm_norms={norms}")
```

If you see `nmm_norms` growing unbounded (>100× the init magnitude after a few
thousand steps), Newton-Schulz isn't kicking in (check `nmm_spectral_norm`) OR the
forgetting α_t is collapsed near zero (check W_α's logits — should sigmoid-distribute,
not saturate at one extreme). If they decay to ~0, α_t is collapsed near 1 (everything
forgotten) OR the inner-loss gradient is vanishing (check k̂_t, v_t aren't being
zeroed by L2-norm pathology on near-zero inputs).

- Checkpoint every 1000 steps. **Required fields:**
  ```python
  # G186: use _unwrap(model).state_dict() — see task 6.2's _unwrap helper. If `model`
  # is wrapped with torch.compile, model.state_dict() emits keys prefixed
  # `_orig_mod.*` (e.g., `_orig_mod.blocks.0.attn.q_proj.weight`). The resume code
  # below rebuilds an unwrapped model and calls load_state_dict, which fails on
  # the mismatched keys. The _unwrap helper strips the wrapper unconditionally
  # so the saved state_dict matches the unwrapped layout regardless of whether
  # torch.compile was applied at save time. See G184 in GAP_HISTORY.md for the
  # detailed failure mode (including the strict=False footgun).
  torch.save({
      'state_dict':     _unwrap(model).state_dict(),
      'optimizer':      optimizer.state_dict(),
      'step':           step,
      'config':         dataclasses.asdict(model.config),  # MUST be saved — see below
  }, ckpt_path)
  ```
  **Critical**: `config` (or at minimum `config.finetune_mode`) must be in the checkpoint.
  `finetune_mode` controls block structure: `gamma_attn` is created only when
  `finetune_mode=False`, and `out_scale` is initialized to zeros vs. ones based on it
  (tasks 1.4/2.4). Without saving the config, resume code cannot know which structure to
  build; constructing a `TitansMAGGPT2` with the wrong `finetune_mode` produces either
  `state_dict` mismatch errors (missing/extra `gamma_attn`) or silently-wrong inits.

- **Resume code** must rebuild the model from the saved config BEFORE loading state_dict,
  AND restore optimizer state + step counter. The model-only resume sketched in earlier
  passes is silently incorrect (G153) — re-initializing the optimizer drops the saved Adam
  `m` (first-moment) and `v` (second-moment) buffers. With β2=0.95 the moments need ~20
  steps to re-warm; with β2=0.999 (the wrong default — see task 4.1's betas note) it would
  be ~1000 steps. Either way, post-resume training proceeds with effectively random update
  statistics for the warm-up period, combined with `weight_decay=0.1`, which silently
  destabilizes parameters. The Done condition below only tests the immediate forward pass
  (`identical loss on the next fresh document`), so this slips past testing.
  ```python
  # 1. Load checkpoint blob.
  # G168: weights_only=False is REQUIRED. PyTorch 2.6+ changed the default of
  # `torch.load` from weights_only=False to weights_only=True. With the new default,
  # loading a checkpoint that contains anything beyond bare tensors / OrderedDicts /
  # primitives raises `UnpicklingError: Weights only load failed. ... Use
  # weights_only=False ...`. Our checkpoint includes `dataclasses.asdict(config)`
  # (a plain dict — usually OK), `optimizer.state_dict()` (nested dicts of tensors
  # plus param_group metadata — usually OK), and `step` (int — OK), so weights_only
  # =True might happen to succeed in some PyTorch versions. But the optimizer state
  # may contain torch.Tensor subclasses or version-specific structures that the
  # weights_only allowlist rejects on certain combinations of (PyTorch version,
  # accelerator, optimizer class). Set `weights_only=False` explicitly so the
  # behavior is stable across PyTorch versions and so a future addition to the
  # checkpoint blob doesn't introduce a silent (or noisy-but-mysterious) regression.
  # This is safe because we are loading OUR OWN checkpoints written by OUR code —
  # the security rationale for weights_only=True (refusing arbitrary pickle code
  # from untrusted sources) does not apply.
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

  # 2. Rebuild the model FROM THE SAVED CONFIG. Must happen before load_state_dict so
  #    block structure (gamma_attn / out_scale init) matches what was saved.
  config = TitansConfig(**ckpt['config'])
  model  = TitansMAGGPT2(config).to(device)
  model.load_state_dict(ckpt['state_dict'])

  # G209 — wrap with DDP AFTER load_state_dict and BEFORE optimizer build, when
  # resuming distributed training. The saved state_dict was UNWRAPPED at save
  # time (G186/G195 — `_unwrap(model).state_dict()` strips `module.` and
  # `_orig_mod.` prefixes), so load_state_dict targets the unwrapped layout.
  # Wrapping AFTER load is the correct order. Skipping the wrap on resume
  # produces silent single-GPU training — same failure mode as G201, just on
  # the resume path: ranks construct independent models from the same
  # state_dict, never AllReduce, diverge. The user notices nothing until eval
  # shows N-rank training underperformed the expected compute budget.
  # Mirror the launch-time `is_distributed` check (set during the original
  # device-selection step at G201) — under torchrun the resume script reads
  # LOCAL_RANK the same way the launch script did.
  from torch.nn.parallel import DistributedDataParallel as DDP
  if is_distributed:
      model = DDP(model, device_ids=[local_rank])

  # G221 — explicit `model.train()` after the wrap. The consolidated training
  # loop's step 4 (G171) calls `model.train()` right after model construction
  # for exactly this reason: defense against any prior code (a sanity-check
  # eval, a perplexity probe, an in-flight generation) having left the model
  # in eval mode. `scripts/finetune.py` BUILDS its model via this resume
  # sequence INSTEAD OF the consolidated loop's step 4, so without the
  # explicit `model.train()` here the finetune path silently skips that
  # defense — if anything between this resume block and the training-loop
  # entry runs `model.eval()` (a smoke-test perplexity, a pre-training
  # sample generation, an external evaluator), the subsequent loop runs in
  # eval mode. Dropout silently stays disabled (regularization gone),
  # logging shows "loss looks fine" because the loss IS fine — just
  # measured without dropout — and any seed/regularization-sensitive
  # phenomena diverge from the train-mode reference. G164's dispatcher
  # gate now keys on `torch.is_grad_enabled()` (not `self.training`), so
  # the scan-during-training hazard is closed, but the dropout-disabled
  # hazard remains. Belt-and-suspenders: explicitly set train mode here
  # at the end of the resume block. Idempotent — safe regardless of
  # caller's mode at entry. `nn.Module.train()` propagates to all
  # submodules (and through DDP's wrapper).
  model.train()

  # 3. Rebuild the optimizer using the same 4-group recipe from task 4.1.
  #    `optimizer.load_state_dict` requires that param_groups already match what was
  #    saved — same group count, same params per group, in the same order. The
  #    `_make_param_groups` helper produces this deterministically given the same
  #    model topology, which we just rebuilt from the saved config.
  # G209 — use `_unwrap(model).named_parameters()` for symmetry with the
  # launch-time loop (task 4.5). Substring matches like `'nmm' in n` work
  # against both `blocks.0.nmm.*` and `module.blocks.0.nmm.*` (DDP prefix),
  # so this is robust either way; the explicit `_unwrap` is for clarity and
  # to insulate against future filter changes that move from substring to
  # exact-prefix matching.
  gpt2_named = [(n, p) for n, p in _unwrap(model).named_parameters()
                if not any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]
  nmm_named  = [(n, p) for n, p in _unwrap(model).named_parameters()
                if     any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]
  # Use the SAME peak-LR constants as task 4.1 — see G162. Hard-coding 1e-4 / 3e-4
  # here would work but drifts on re-tuning; pulling from the constants the rest of
  # the code uses keeps the resume path locked to the training-loop path.
  optimizer  = AdamW(
      _make_param_groups(gpt2_named, lr=GPT2_PEAK_LR) +
      _make_param_groups(nmm_named,  lr=NMM_PEAK_LR),
      betas = (0.9, 0.95),    # must match the original (see task 4.1's betas note)
      eps   = 1e-8,
  )
  # G219 — guard the optimizer-restore against checkpoints that don't carry
  # optimizer state. `scripts/load_pretrained.py` (task 2.6) saves a fresh
  # HF-init checkpoint with only `{state_dict, config, step}` keys — NO
  # `optimizer` key, because there's no training step that has produced m/v
  # moments yet. When `scripts/finetune.py` later uses THIS resume sequence
  # to load that HF-init checkpoint, a bare
  #     optimizer.load_state_dict(ckpt['optimizer'])
  # raises KeyError: 'optimizer' on the very first run of finetune.py. The
  # error message points at the resume code, not the load_pretrained side,
  # so the user blames the resume sequence and may "fix" it by skipping the
  # optimizer restore entirely — which then silently breaks resume from
  # MID-training checkpoints (G153's bug: m/v moments dropped → effective
  # LR de-warms over ~20 steps with β2=0.95).
  # Correct: detect the missing key and skip the restore for fresh-init
  # checkpoints. The optimizer keeps its just-constructed empty state (no
  # m, v entries; lazy-populated on first .step()). For finetune-from-HF
  # this is the intended behavior — training starts cold for the optimizer
  # while warm for the model weights. For mid-training resume, ckpt
  # ALWAYS has 'optimizer' (the save block in this task always emits it),
  # so the restore fires as expected.
  if 'optimizer' in ckpt:
      optimizer.load_state_dict(ckpt['optimizer'])  # restores Adam m/v moments AND
                                                     # overwrites 'lr' in each param_group
                                                     # with whatever apply_lr last wrote
                                                     # before save — that's fine; the
                                                     # next apply_lr call (step 5 below)
                                                     # rewrites them deterministically.
  # If 'optimizer' is missing, this is an HF-init checkpoint (load_pretrained
  # output) — optimizer stays at fresh-construction state.

  # 4. Restore the training step counter so any step-based LR schedule, logging
  #    interval, and checkpoint counter pick up from the right point. Recompute LR
  #    by feeding `step` into your schedule function (we use a manual cosine — no
  #    torch.optim.lr_scheduler state to deserialize).
  step = ckpt['step']

  # 5. Re-derive base_lrs from CONSTANTS, NOT from optimizer.param_groups (G162).
  #    `optimizer.load_state_dict` above overwrote `g['lr']` with the saved mid-cosine
  #    values. Reading `[g['lr'] for g in optimizer.param_groups]` here would silently
  #    capture the deflated mid-cosine LR as the "peak", and every subsequent apply_lr
  #    would scale a deflated base by lr_mul — the effective LR after each resume
  #    halves (or more), compounding across resumes until training plateaus.
  base_lrs = [GPT2_PEAK_LR, GPT2_PEAK_LR, NMM_PEAK_LR, NMM_PEAK_LR]
  ```
  Missing any of steps 3, 4, or 5 above is a silent correctness regression — the model
  loads but the training dynamics restart cold (3), the LR schedule jumps (4), or every
  resume silently deflates the peak LR a little more (5).

**Do NOT save `nmm_states` in the checkpoint.** NMM states are per-sequence running
accumulators, not model parameters. At checkpoint resume, reset them to `init_state`.
The model will lose any in-progress memory state, but since most documents are shorter
than a few chunks, this is acceptable. Saving them would require storing B × n_layers
large tensors and managing their serialization — not worth it.

**Done:** (a) checkpoint save/load round-trip produces identical loss on the next fresh
document (states reset to init, then run) — verifies model state_dict + config restore;
(b) after the resume sequence above, `optimizer.state_dict()['state'].values()` is
non-empty (i.e., Adam `exp_avg` and `exp_avg_sq` are populated for at least one param) —
verifies G153's optimizer state restore. Without (b), step (a) alone would pass even
when the optimizer is silently re-initialized.

### 4.4 Fine-tuning entry point
`scripts/finetune.py`: loads pretrained checkpoint from Phase 2.6, runs training loop.

**Done:** 100 steps on FineWebEdu (or OpenWebText as a local substitute) without OOM,
GPT-2 small, chunk_size=512, batch=4, single GPU.

Note: the paper trains on FineWebEdu-10BT. OpenWebText is a reasonable local substitute
for development; switch to FineWebEdu via `HuggingFaceFW/fineweb-edu` on HuggingFace for
final experiments.

### 4.5 Training-from-scratch entry point
`train.py`: train a `TitansMAGGPT2` with `finetune_mode=False` from random initialization.
No pretrained checkpoint; initializes all weights fresh via `TitansMAGGPT2._apply_gpt2_init`
(task 2.5) — GPT-2-style N(0, 0.02) with the 1/sqrt(2*n_layer) residual-init scaling on
output projections. **Do NOT skip this**: PyTorch's `nn.Embedding` default is `N(0, 1)`,
which sends from-scratch training off a cliff (G155). The finetune path masks this
because task 2.6 overwrites embeddings with HF weights; the from-scratch path is exactly
where the bug would hit. Reuses the same training loop from task 4.2 and optimizer from
task 4.1. Key differences from `scripts/finetune.py`:
- `finetune_mode=False` in config → pure paper MAG formula; `gamma_attn` created
- `out_scale` initialized to ones (not zeros) — NMM contributes at step 1
- Full learning-rate schedule from warm-up start (no backbone warm-up stage)
- Larger batch size / longer training budget expected for competitive perplexity
- **Set `chunk_size == block_size` (G163).** The default config has `block_size=1024,
  chunk_size=512` — designed for finetune mode (where HF GPT-2 has already trained all
  1024 position embeddings). From-scratch training only looks up positions
  `0..chunk_size-1` during the forward, so `wpe.weight[chunk_size:block_size]` never
  gets a gradient and stays at random init. Generating past `chunk_size` tokens then
  accesses untrained random position embeddings — quality silently degrades. The
  `__post_init__` warning will fire; either suppress it intentionally (and cap
  generation at `chunk_size`) or set both equal:
  ```python
  config = TitansConfig.gpt2_small(
      finetune_mode = False,
      chunk_size    = 1024,   # match block_size so all wpe positions are trained
      block_size    = 1024,
  )
  ```

`train.py` is the top-level CLI entry point; `scripts/finetune.py` is a specialized
script for starting from HF GPT-2 weights. Both share the same dataset, optimizer, and
TBPTT loop implementations from tasks 3.x and 4.1–4.3.

**G171 — consolidated training loop.** Earlier passes scattered the loop's
ingredients across tasks: `apply_lr` and `base_lrs` in 4.3, `train_step` in 4.2,
`detach_states` in 1.9, `compute_nmm_norm` in 4.3 (added in G172), the device transfer
in train_step (G167), and the implicit `model.train()` call nowhere. The natural
assembly is non-obvious and several common readings produce silent bugs:

- `apply_lr` AFTER `train_step` → step 0 runs at peak LR (no warmup), every later step
  uses the previous step's LR. Off-by-one warmup, minor but visible in early-loss curves.
- `model.train()` never called → if a sanity-check `model.eval()` ran first, the
  training loop runs in eval mode. G164 closed the scan-during-training hazard at the
  dispatcher level, but dropout (when `config.dropout > 0`) silently stays off and
  LayerNorm/Dropout-mode-dependent behaviors diverge from the intended training-time
  setup. Always call `model.train()` once before the loop.
- `nmm_states` initialized to anything other than `None` → if the user does
  `nmm_states = [block.nmm.init_state(B, device) for ...]` to "pre-warm", the first
  call's None-detection path in model.forward is skipped — fine in theory, but if B
  doesn't match what the loader produces, the shapes mismatch on the first chunk.
  Start with `None` and let model.forward initialize lazily.
- `max_steps` mismatch with actual data length → cosine completes before the data does
  (LR floors at min_ratio for the remainder, wasting compute) or never completes (LR
  stays in warmup phase forever if max_steps > len(loader)).

The canonical assembly:

```python
import torch
import torch.nn.functional as F
import dataclasses

# --- 0. User-provided inputs (placeholders — fill these in for your run) ---
# G177 — these were referenced but undefined in the earlier snippet, producing a
# NameError on the first run. They're inputs, not derivations; supply them up front.
seed = 42                                         # any int; reused by manual_seed

# G189 — bind `documents` to a resource-managed iterable. The earlier sketch
# used `documents = (line for line in open('corpus.txt'))` which leaks the file
# handle (Python only closes it when the generator is GC'd, which happens at
# program exit — fine for a short script, bad for long-running multi-job
# workflows where many such generators leak handles until ulimit is hit). The
# usual idiom is to open the file in a `with` block and EITHER pass the open
# file object directly to `encode_corpus` (Tokenizer iterates it line-by-line)
# OR materialize the list inside the `with` block:
#
#   with open('corpus.txt') as f:
#       token_stream = tok.encode_corpus(f)
#
# This pattern closes the file as soon as tokenization is done. For an HF
# dataset (no file handle), no resource management is needed:
#
#   ds = datasets.load_dataset("openwebtext", split="train", streaming=True)
#   documents = (row['text'] for row in ds)
#   token_stream = tok.encode_corpus(documents)
#
# G178 — `documents` MUST be an iterable of STRINGS (Tokenizer.encode_corpus
# calls `self.enc.encode(doc, ...)`). For HF datasets, the
# `(row['text'] for row in ds)` adapter above produces strings. Passing the
# raw `ds` object yields dicts → `enc.encode(dict)` raises TypeError.

# --- 1. Set up reproducibility BEFORE constructing the model ---
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# G194 — device selection. The bare `torch.device('cuda')` defaults to cuda:0
# regardless of which rank is running this code. Under DDP / torchrun, this means
# every rank tries to use GPU 0 → out-of-memory on first allocation past the first
# rank, or kernel contention (multiple processes serializing on the same device).
# The standard torchrun pattern is:
#     export LOCAL_RANK=...
#     torchrun --nproc_per_node=N train.py
# which sets `LOCAL_RANK` per process. Read it and select the matching device.
# `torch.cuda.set_device(local_rank)` also sets the default device for any
# `cuda` tensor created without explicit device — critical because not every
# code path passes `device=` explicitly (e.g., `torch.empty(...)` in a third-party
# layer would use the default).
import os
import torch.distributed as dist
is_distributed = 'LOCAL_RANK' in os.environ and torch.cuda.is_available()
if is_distributed:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    # G201 — `init_process_group` MUST be called before any other distributed
    # operation. Without it:
    #   - `dist.is_initialized()` returns False everywhere → G191's data loader
    #     defaults to rank=0, world_size=1 on every process → every rank trains
    #     on the FULL corpus (same data partition). The very bug G191 was added
    #     to fix re-emerges silently: gradients are averaged across duplicate
    #     data, effective batch size doesn't scale with N_ranks, user blames
    #     "DDP didn't help" instead of the missing init.
    #   - `DistributedDataParallel(model, device_ids=[local_rank])` raises
    #     "Default process group has not been initialized" loudly — but only
    #     IF the user remembers to wrap (see G201 model-wrap step below).
    # `backend='nccl'` is the GPU collective backend (NVIDIA NCCL). For CPU-only
    # DDP (rare), use `backend='gloo'`. `torchrun` automatically sets `RANK`,
    # `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`; init_process_group reads them.
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
elif torch.cuda.is_available():
    device = torch.device('cuda:0')          # single-GPU explicit
    rank = 0
    world_size = 1
else:
    device = torch.device('cpu')
    rank = 0
    world_size = 1

# --- 2. Build config FIRST (loader needs config.chunk_size; model needs config). ---
# G205: config MUST be defined before the loader's chunk_size is referenced. Earlier
# versions of this snippet ordered "build data → build model" with `config` defined
# inside step 3 — but step 2's `ParallelStreamLoader(chunk_size=config.chunk_size)`
# and the empty-stream guard `token_stream.numel() < 4 * config.chunk_size` both
# read `config` before it exists. NameError on the very first run (caught at
# `loader = ParallelStreamLoader(...)`, but the error message is "name 'config' is
# not defined" which is unhelpfully far from the real issue — the implementer
# might think they forgot to import something rather than that the assembly order
# is wrong). Move config construction here, ahead of both loader and model.
# For from-scratch (task 4.5), use chunk_size == block_size (G163).
config = TitansConfig.gpt2_small(
    finetune_mode = False,
    chunk_size    = 1024,
    block_size    = 1024,
)

# --- 3. Build data ---
tok = Tokenizer()
# G189: open inside `with` so the file handle closes as soon as tokenization finishes.
# Swap this block for the HF-dataset adapter shown in the G178 comment above as needed.
# G210 — file iteration yields LINES, not documents. The naive
#     with open('corpus.txt') as f:
#         token_stream = tok.encode_corpus(f)         # WRONG: each line becomes a "doc"
# silently treats every newline as a document boundary because `for doc in f`
# yields one line per iteration; `encode_corpus` then appends `self.eot_token`
# after EVERY line. Downstream consequences in the training loop:
#   - ParallelStreamLoader's boundary mask fires on every line break.
#   - `_forward_chunk_sequential` calls reset_state on every line boundary.
#   - The NMM state never accumulates more than ~one line of context — the
#     entire point of TITANS (long-range memory) is silently disabled.
# No error message; loss curves look "as expected" because the backbone still
# trains; NMM training is effectively reset per line so memory metrics stay
# near init magnitude. The user only notices on a needle-in-haystack eval that
# the model has no long-range recall.
# The correct pattern depends on what "document" means for your corpus:
#   - Whole-file-as-one-document (e.g., a single long text):
#         with open('corpus.txt') as f:
#             token_stream = tok.encode_corpus([f.read()])
#   - Blank-line-separated paragraphs as documents (typical for Wikipedia
#     dumps; matches the BLANK_LINE_RE = re.compile(r'\n\s*\n') idiom):
#         with open('corpus.txt') as f:
#             text = f.read()
#         docs = re.split(r'\n\s*\n', text)
#         token_stream = tok.encode_corpus(docs)
#   - HuggingFace dataset with one row per document (the `row['text']` adapter
#     from G178's comment): each row IS a document — works directly.
# All three patterns yield ONE EOT between LOGICAL documents, not after every
# line. The example below uses the whole-file form as the safe default for an
# unfamiliar corpus; downgrade only if you're confident your file's newlines
# ARE document boundaries (rare).
with open('corpus.txt') as f:
    token_stream = tok.encode_corpus([f.read()])  # whole file = 1 document
# G179: guard against silently-empty loaders. An empty `documents` iterable yields
# `token_stream` of length 0 → `(0 // (B*chunk)) * B*chunk = 0` → loader yields zero
# batches → the training loop completes immediately with no error and no log lines.
# Catch this early. Uses `raise` (not `assert`) so `python -O` doesn't strip it — see
# G190.
if token_stream.numel() < 4 * config.chunk_size:
    raise ValueError(
        f"Token stream too small ({token_stream.numel()} tokens) for the configured "
        f"loader (batch_size=4, chunk_size={config.chunk_size}). "
        f"ParallelStreamLoader would yield zero batches. Check that `documents` "
        f"actually produces text (a common bug is forgetting the `(row['text'] for ...)` "
        f"adapter when passing an HF dataset — see G178)."
    )
loader = ParallelStreamLoader(
    token_stream,
    batch_size = 4,
    chunk_size = config.chunk_size,
    eot_id     = tok.eot_token,
)
num_batches_per_epoch = len(loader)               # exact count, drops the trailing remainder

# --- 4. Build model (config already built in step 2) ---
model = TitansMAGGPT2(config).to(device)
model.train()                                     # G171 — explicit; required if any prior
                                                  # code (sanity sample, validation) left
                                                  # the model in eval mode.

# G201 — wrap with DistributedDataParallel AFTER model construction + .to(device)
# and BEFORE optimizer construction. The earlier passes named DDP repeatedly
# (G191's loader sharding, G194's device selection, G195's _unwrap) but never
# wrote out the wrap step itself. Skipping it produces silent single-GPU
# training even under torchrun: each rank constructs its own independent model,
# trains it on its own data shard, and never AllReduces gradients. The N "ranks"
# are N independent trainers diverging into different minima; final checkpoints
# don't average across ranks (each rank saves its own); only rank 0's checkpoint
# is the one anyone uses, so the other ranks' compute is pure waste. Loss curves
# look fine on each rank individually — they're learning, just not together.
# `device_ids=[local_rank]` tells DDP which GPU this rank owns; `find_unused_
# parameters=False` is the default and matches our model (no conditional
# parameter use — gamma_attn is created conditionally at __init__ time, not
# conditionally consumed per forward, so it stays "used").
# Order matters: DDP MUST wrap before the optimizer is built because the
# optimizer captures `.parameters()` at construction. With DDP wrapping after,
# the optimizer holds the unwrapped Parameter objects — which still works (DDP
# doesn't replace params, it adds reduction hooks) — but the conventional order
# is wrap-then-optimizer and matches every PyTorch example. Either works
# functionally; we keep wrap-then-optimize for convention.
from torch.nn.parallel import DistributedDataParallel as DDP
if is_distributed:
    model = DDP(model, device_ids=[local_rank])

# G204 — re-seed per-rank AFTER model construction so stochastic ops diverge
# across DDP ranks. The setup-time `torch.manual_seed(seed)` block above
# (before model construction) MUST use the SAME seed on every rank so each
# rank constructs an IDENTICAL model — DDP requires bit-identical initial
# weights to maintain gradient-sync invariants. But after construction, the
# same global RNG state means every rank's dropout draws the SAME mask, every
# augmentation/shuffle call samples the SAME positions, etc.
# Failure mode: with `config.dropout = 0.1`, the user expects K=N independent
# dropout masks per step (giving N× the effective regularization signal of
# a single-GPU run). Instead they get the SAME mask repeated N times — the
# regularization effect collapses to a single-mask equivalent, and the
# implicit "more dropout draws = better regularization" benefit of multi-GPU
# data parallelism vanishes. Loss curves look "as expected" because the user
# never compared against the alternative; subtle generalization degradation
# only shows up on held-out evals. Same hazard for any random data sampling
# inside the training loop (e.g., random crop, random span masking, dropout
# inside the model).
# `seed + rank` is the standard pattern: ranks see distinct RNG streams while
# the SAME `seed` reruns reproducibly. Re-seed both CPU and CUDA generators;
# `manual_seed` advances the default generator only, so we also re-seed the
# current device's CUDA generator (the autocast/dropout paths both pull from
# this stream).
torch.manual_seed(seed + rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed + rank)

# --- 5. Build optimizer using the 4-group recipe from task 4.1 ---
# Use `_unwrap(model)` so DDP-wrapped params route into the same `nmm`/non-`nmm`
# splits as the unwrapped path. DDP exposes parameters with `module.` prefix
# (e.g., `module.blocks.0.nmm.k_proj.linear.weight`); the substring checks
# `'nmm' in n` still match correctly, so unwrap is not strictly required here
# — but if a future refactor switches to exact-prefix matching, the
# `_unwrap(model).named_parameters()` form is robust to wrapping.
gpt2_named = [(n, p) for n, p in _unwrap(model).named_parameters()
              if not any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]
nmm_named  = [(n, p) for n, p in _unwrap(model).named_parameters()
              if     any(k in n for k in ['nmm', 'gamma', 'persistent', 'ln_nmm'])]
optimizer = AdamW(
    _make_param_groups(gpt2_named, lr=GPT2_PEAK_LR) +
    _make_param_groups(nmm_named,  lr=NMM_PEAK_LR),
    betas = (0.9, 0.95),
    eps   = 1e-8,
)
# G162: base_lrs derived from CONSTANTS, NOT from optimizer.param_groups (see task 4.3).
base_lrs = [GPT2_PEAK_LR, GPT2_PEAK_LR, NMM_PEAK_LR, NMM_PEAK_LR]

# --- 6. Coordinate max_steps with the actual data length ---
# Set max_steps to the cosine-decay end point. Aim for the cosine to bottom out at
# roughly the end of training. If you'll run N_EPOCHS, max_steps = N_EPOCHS *
# num_batches_per_epoch. The schedule will then warm up (warmup_steps) → cosine to
# min_ratio over the remaining (max_steps - warmup_steps) steps.
N_EPOCHS     = 1
warmup_steps = 1000
max_steps    = N_EPOCHS * num_batches_per_epoch
# G175: max_steps and warmup_steps are explicitly threaded into apply_lr below. The
# pre-G175 snippet called apply_lr without these and silently used the function
# defaults (max_steps=100_000, warmup_steps=1000), which only happened to be right
# for a specific training-budget assumption.

# --- 7. The actual training loop ---
# G225 — wrap the training loop in try/finally so `dist.destroy_process_group()`
# fires on the failure path too, not only on clean termination. Without this,
# any exception inside the loop (OOM mid-step, KeyboardInterrupt during a hang,
# NaN-stuck-loop manually terminated, a downstream library crash) bypasses the
# cleanup below. Failure modes the leak enables:
#   - Jupyter / IPython kernels: the kernel survives the exception. The next
#     training cell calls `dist.init_process_group(...)` and either fails with
#     "default process group has already been initialized" (loud), OR succeeds
#     into a stale-but-resuable group that mixes communicators with the dead
#     job's residue (silent, intermittent NCCL hangs).
#   - Hyperparameter sweeps / wrapper scripts that catch+log+continue: the next
#     iteration's `init_process_group` hangs waiting for the prior NCCL group's
#     release (NCCL communicators are reference-counted at the OS level; a
#     leaked group ties them up).
#   - Long-running CI runners: leaked groups eventually exhaust NCCL's
#     communicator pool / shared-memory budget; later jobs in the same runner
#     fail with cryptic "cudaErrorMemoryAllocation" or
#     "ncclSystemError: System call failed" messages with no relation to
#     the actual cause.
# On the success path the cleanup is identical; the try/finally is idempotent.
# Bare `try/finally` (no `except`) re-raises the exception after running the
# finally block — we DO want the exception to propagate to the user, just with
# the resources released first.
# G227 — keep CONSISTENT 4-space indentation inside the try block. An earlier
# G225 sketch used 2-space indent for `for epoch` (col 2) but kept the inner
# body at the pre-G225 column 8, producing mixed step sizes (2-2-4) within the
# nested blocks. Python parses both correctly because each block's body is
# internally consistent at its own indent level; the discrepancy is the indent
# STEP between levels (2 vs 4). PEP-8 specifies 4-space indent consistently,
# and the mixed-step variant is jarring to readers who copy-paste fragments
# into their own files (assuming uniform 4-space) — they end up with broken
# indentation that Python rejects at parse time, with an error pointing at
# the wrong line. Use 4-space steps throughout so the snippet is uniformly
# portable.
nmm_states = None                                     # model.forward initializes lazily
step = 0
try:
    for epoch in range(N_EPOCHS):
        for batch in loader:
            # G171: apply_lr BEFORE train_step so the new LR is in effect for this step's
            # optimizer.step() call (which happens inside train_step).
            # G175: pass the SAME max_steps/warmup_steps that govern the loop termination
            # below — keeps the schedule's cosine endpoint matched to the actual training
            # budget.
            lr_mul = apply_lr(optimizer, base_lrs, step,
                              warmup_steps=warmup_steps, max_steps=max_steps)
            # G167: train_step takes device explicitly; it does the .to(device) transfer
            # internally. batch is a CPU tuple from the loader.
            loss, nmm_states, grad_norm = train_step(
                model, batch, nmm_states, optimizer, device,
            )
            if step % 50 == 0:
                nmm_norms = compute_nmm_norm(nmm_states)         # G172
                print(f"epoch={epoch} step={step}/{max_steps} loss={loss:.4f} "
                      f"grad_norm={grad_norm:.4f} lr_mul={lr_mul:.4f} nmm_norms={nmm_norms}")
            if step % 1000 == 0 and step > 0:
                # G199 — rank-guard the save. Under DDP, every rank runs this block;
                # without the `rank == 0` guard, N processes race to write the same
                # checkpoint file. Failure modes:
                #   - Local SSD: N parallel writes to the same path interleave bytes →
                #     corrupted checkpoint (torch.load later raises "PytorchStreamReader
                #     failed" or "unexpected EOF" depending on which write won the race).
                #   - NFS / shared FS: even worse — flock semantics vary, partial writes
                #     can survive, and the post-load model loads silently-corrupted
                #     weights (random subset of ranks' partial writes).
                #   - Best case (local SSD with overwrite ordering): N writes succeed
                #     sequentially, only the last one survives — N× IO bandwidth wasted
                #     for identical data, and the last write's timing skews the training
                #     loop's per-step latency (rank 0 finishes early, sits at barrier).
                # The state_dict is IDENTICAL across ranks (DDP keeps params synced
                # via AllReduce), so saving from any one rank captures the same data.
                # By convention save from rank 0. The dist.barrier() AFTER the save
                # prevents other ranks from racing into the next training step while
                # rank 0 is still flushing to disk — without it, the I/O contention
                # for that node's filesystem can stall the next AllReduce.
                # G186: _unwrap(model).state_dict() — see task 6.2 helper and G184.
                # G195: _unwrap now also strips DDP's `.module` wrapper iteratively.
                if rank == 0:
                    torch.save({
                        'state_dict': _unwrap(model).state_dict(),
                        'optimizer':  optimizer.state_dict(),
                        'step':       step,
                        'config':     dataclasses.asdict(config),
                    }, f'ckpt_step_{step}.pt')
                if is_distributed:
                    dist.barrier()
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break
finally:
    # G201 / G225 — cleanup runs on BOTH the success path and the exception path.
    # Tear down the process group at end of training. Without this, the script
    # can hang at exit on some PyTorch/NCCL versions waiting for the group to
    # be released; also keeps NCCL communicators from leaking into a subsequent
    # script run in the same process (notebooks, sweepers, CI runners — see
    # G225's failure-mode list). The finally-block ensures the cleanup runs
    # even when the training loop raised mid-step.
    if is_distributed:
        dist.destroy_process_group()
```

This is the canonical loop. `scripts/finetune.py` differs ONLY in step 4
(model construction goes through `scripts/load_pretrained.py` to apply HF
weights — i.e., load the HF-init checkpoint via the task 4.3 resume
sequence INSTEAD OF `TitansMAGGPT2(config).to(device)`) and step 6 (often a
smaller `max_steps` for finetuning, since the model is already near a good
init and over-training degrades). G218 — these step numbers reflect the
post-G205 renumbering: build-config is step 2, build-data is step 3,
build-model is step 4, build-optimizer is step 5, max_steps coordination
is step 6, the training loop is step 7. The pre-G205 plan called them
steps 3 and 5; that wording is stale and a reader using it as a TOC ends
up pointing at "build-data" and "build-optimizer" instead of the intended
"build-model" and "max_steps coordination."

**G174 — gradient accumulation.** If GPU memory caps your batch size below what's
useful for training stability, accumulate gradients over `K` micro-batches before
stepping. Do NOT call `train_step` K times — it steps and zero_grads internally, so
K calls produce K independent optimizer steps at micro-batch granularity, not one
accumulated step. Refactor inline:

```python
import contextlib

ACCUM_STEPS = 4   # effective batch = batch_size * ACCUM_STEPS
nmm_states = None
step = 0
for epoch in range(N_EPOCHS):
    micro_batches = iter(loader)
    while True:
        # Accumulate gradients over ACCUM_STEPS micro-batches BEFORE stepping.
        for accum_i in range(ACCUM_STEPS):
            try:
                batch = next(micro_batches)
            except StopIteration:
                batch = None
                break
            input_ids, doc_boundaries = batch
            input_ids      = input_ids.to(device, non_blocking=True)
            doc_boundaries = doc_boundaries.to(device, non_blocking=True)
            nmm_states     = detach_states(nmm_states)
            # G200 — under DDP, EVERY `loss.backward()` triggers an AllReduce of
            # gradients across ranks. With ACCUM_STEPS=4 micro-batches between
            # optimizer.step()s, that's 4 AllReduces per step — but only the
            # LAST gradient state matters (the prior 3 are intermediate
            # accumulations). The other 3 AllReduces are pure communication
            # waste — at 12 GPT-2-small blocks × 4 ranks × ~125M params/rank,
            # each AllReduce is ~500MB; wasting 3 per step at scale means
            # interconnect saturation and the optimizer.step() loop becomes
            # comms-bound, not compute-bound. Effective throughput drops by
            # 2-3× on multi-node setups.
            # The fix: wrap all but the LAST micro-batch's backward in
            # `model.no_sync()`, which disables the per-step AllReduce. The
            # final backward (NOT in no_sync) does the single AllReduce of
            # the accumulated `.grad` buffers, which is mathematically
            # equivalent to averaging K gradients at the end. Same total
            # gradient, 1 AllReduce instead of K.
            # `no_sync` is a method on DDP wrapper, not the base nn.Module.
            # When the model isn't DDP-wrapped (single-GPU), use a null
            # contextmanager — no-op semantics.
            is_last_accum = (accum_i == ACCUM_STEPS - 1)
            if is_distributed and not is_last_accum:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = contextlib.nullcontext()
            # Nesting order matters: sync_ctx wraps backward (so DDP's no_sync
            # hook actually fires during gradient accumulation), but autocast
            # is scoped to forward+loss only (per G159; backward in autocast
            # would compute clip_grad_norm_ in low precision later, and on some
            # PyTorch versions the bf16 forward graph's backward triggers
            # autocast cache issues). Separating the two `with`s keeps both
            # invariants.
            with sync_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
                    # Divide by ACCUM_STEPS so the SUM of K backward passes equals
                    # what one big batch would give. Without /ACCUM_STEPS, gradients
                    # are K× too large and the effective LR is K× too high.
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                    ) / ACCUM_STEPS
                loss.backward()       # inside sync_ctx (DDP gradient-sync gate)
                                       # but outside autocast (fp32 .grad accum)
        if batch is None and accum_i == 0:
            break
        # G214 / G222 — handle the PARTIAL-CYCLE case under DDP. When StopIteration
        # fires at iteration J ∈ [1, K-1] (the corpus has 1..K-1 leftover
        # micro-batches at epoch end OR exhausts EXACTLY at the boundary that
        # leaves iter K-1 with nothing to consume), iterations 0..J-1 all ran
        # INSIDE `model.no_sync()` (G200's pattern picks no_sync for every iter
        # except accum_i == ACCUM_STEPS-1). The final-iter "sync" backward NEVER
        # ran because the loop broke before reaching it. Result: every rank's
        # `.grad` holds its OWN per-rank accumulated gradient with NO
        # AllReduce. The post-loop `optimizer.step()` then applies per-rank
        # gradients independently → ranks diverge.
        # AdamW's m, v moments diverge along with the params (each rank
        # updates its own copy of m, v using its own gradient). On
        # subsequent FULL cycles, gradients ARE AllReduced — but the
        # diverged optimizer state means each rank applies a different
        # update from the synced gradient → ranks stay diverged. The
        # divergence is permanent until manually re-synced (which we never
        # do).
        # The fix: detect "partial cycle under DDP" and SKIP the optimizer
        # step entirely. Discard the partial gradients (zero_grad), break
        # out of the while loop, let the epoch boundary or N_EPOCHS loop
        # reset. We sacrifice 1..K-1 micro-batches of work at the end of
        # each epoch — bounded loss, vs. unbounded divergence.
        # Single-GPU partial cycles are SAFE (no AllReduce involved). The
        # gradient is K/J of intended (since loss was scaled by /ACCUM_STEPS
        # but only J of K backwards ran), making the effective LR J/K of
        # the requested. That's a minor calibration drift but not a
        # correctness bug, so we still step in the single-GPU case.
        # G222 — the original condition `accum_i < ACCUM_STEPS - 1` had an
        # off-by-one: it MISSED the J = ACCUM_STEPS - 1 case. When
        # StopIteration fires on the VERY LAST iteration (accum_i = K-1):
        #   - Iterations 0..K-2 ran, ALL with `is_last_accum = False` →
        #     all backward calls in `no_sync` → no AllReduce.
        #   - Iteration K-1 hit StopIteration before its backward could
        #     run — the would-be "sync" backward (which is the ONLY
        #     iteration that runs in nullcontext / triggers AllReduce)
        #     never fired.
        #   - Total: K-1 no_sync'd backwards, 0 sync'd backwards. Same
        #     no-AllReduce hazard as J ∈ [1, K-2] — just one position over.
        # The old check `accum_i < K - 1` evaluated to (K-1 < K-1) = False
        # for this case, falsely classifying it as "complete cycle" and
        # falling through to optimizer.step. Per-rank `.grad` has K-1
        # accumulated micro-batches' worth of unsynced gradient; ranks
        # apply their own gradients → silent divergence (and from then on,
        # subsequent full cycles' AllReduce'd updates compound on top of
        # diverged params + diverged AdamW state — permanent rank drift).
        # The probability of triggering: corpus length N satisfying
        #     (N // (B * chunk_size)) mod ACCUM_STEPS == K - 1
        # (i.e., the leftover micro-batches after the last full cycle is
        # exactly 0, with the for loop entering iter K-1 and finding the
        # iterator empty). For a uniform distribution of corpus lengths
        # this is 1/K of epochs. For corpora deliberately chosen as
        # multiples of (B * chunk_size * ACCUM_STEPS), it happens EVERY
        # epoch.
        # The fix is to drop the `< ACCUM_STEPS - 1` constraint and gate
        # on `accum_i > 0` instead (or equivalently, no further constraint
        # — the `accum_i == 0` case is already handled by the outer break
        # above). This catches J ∈ [1, K-1] uniformly. When `batch is None`
        # at this point, the cycle is partial regardless of where in the
        # iteration space the StopIteration fired.
        is_partial_cycle = (batch is None) and (accum_i > 0)
        if is_distributed and is_partial_cycle:
            # Partial DDP cycle: per-rank gradients never synced.
            # Discard and break to avoid rank divergence.
            optimizer.zero_grad(set_to_none=True)
            break
        # One step per ACCUM_STEPS micro-batches, with the accumulated gradient.
        # G185: pass max_steps/warmup_steps explicitly. The accumulation example
        # initially read `apply_lr(optimizer, base_lrs, step)` which silently used
        # the function defaults (max_steps=100_000, warmup_steps=1000) regardless
        # of the user's actual training budget — same bug as G175, propagated here.
        # Same `max_steps` value must be used both in apply_lr and in the
        # `if step >= max_steps: break` termination check below; otherwise the
        # schedule's cosine endpoint and the loop's stop point diverge.
        lr_mul = apply_lr(optimizer, base_lrs, step,
                          warmup_steps=warmup_steps, max_steps=max_steps)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            # G217 — same NaN-recovery as G213's single-step train_step: when
            # the accumulated gradient is non-finite, the post-cycle nmm_states
            # is almost certainly NaN-tainted (a NaN gradient anywhere in the
            # K-micro-batch accumulation means at least one forward pass
            # produced NaN, contaminating M and S; later micro-batches in the
            # same cycle ran with the corrupted state, propagating NaN
            # forward). Pre-G217 the accumulation block only had the
            # `if torch.isfinite(grad_norm): optimizer.step()` skip — it
            # SKIPPED the parameter update but RETAINED the NaN-state via
            # the local `nmm_states` variable, so the next cycle started
            # with a NaN M, produced NaN forward again, repeated forever.
            # Same persistent-NaN-stuck loop as the single-step variant.
            # Reset to None so the next cycle's first micro-batch's forward
            # hits model.forward's None-detection path and calls init_state
            # for a fresh M from memory_mlp.W*.weight. One chunk of memory
            # continuity lost; indefinite NaN loop avoided.
            nmm_states = None
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step >= max_steps:
            break
```

Two non-obvious points:
1. **`detach_states` runs PER MICRO-BATCH, not per accumulation cycle.** TBPTT detaches
   between chunks, regardless of accumulation. The K backward passes within one accum
   cycle each cover one chunk's worth of unrolled NMM state.
2. **The `/ ACCUM_STEPS` on the loss is what makes accumulation equivalent to a larger
   batch.** Cross-entropy with `reduction='mean'` averages over the (B × T-1) target
   tokens. Without scaling, K calls to `loss.backward()` accumulate K full means,
   producing a gradient K× too large. The /K rescales it back. (This is the standard
   accumulation pattern; the trap is that omitting it silently sextuples-or-more your
   effective LR with no error.)

**Done:** loss decreases on a from-scratch training run with `finetune_mode=False`; config
serialized to checkpoint so mode is recoverable at evaluation time.

---

## Phase 5 — Generation and Evaluation

### 5.1 Autoregressive generation
`generate.py`:

```python
@torch.no_grad()                       # G156 — see "Eval mode" note below.
def generate(model, prompt, max_new_tokens=200, temperature=1.0, top_k=50, tokenizer=None):
    # G208 — `tokenizer` is an OPTIONAL argument so the caller can pass the
    # same Tokenizer instance used during training/eval. Pre-G208 the function
    # built `tok = Tokenizer()` internally on every call, which has three
    # problems:
    #   (1) Reproducibility — if training extended the tokenizer (special
    #       tokens, additional BPE merges), a fresh `Tokenizer()` inside
    #       generate() doesn't see those modifications. EOT id and special
    #       tokens silently differ between train and gen; generated samples
    #       drift in unexpected ways.
    #   (2) Performance — `tiktoken.get_encoding("gpt2")` lazy-loads BPE
    #       merges from disk. Cheap per call (~10ms), but repeated calls in
    #       a generation harness compound. Reusing one instance saves it.
    #   (3) Coupling — generate() shouldn't need to know how to construct a
    #       Tokenizer. Inverting the dependency (caller passes it) lets the
    #       generate() API stay stable while Tokenizer's signature evolves.
    # Default to constructing a fresh one to preserve the old call signature.
    # G161: capture-and-restore the mode. Without `try/finally`, an eval call inside a
    # training loop can leave the model in eval mode after this helper returns. The
    # specific dispatcher hazard was first identified pre-G164: the Phase 6 guard then
    # keyed on `self.training`, so a sticky eval mode silently re-routed training-time
    # forwards through the scan path (which lacks autograd outside torch.compile),
    # zeroing NMM gradients with no visible signal.
    #
    # G164 hardened the dispatcher to gate on `torch.is_grad_enabled()` instead of
    # `self.training`, so the scan-during-training hazard is now blocked regardless of
    # the mode flag. The try/finally below is STILL required, however — dropout and
    # other train/eval-mode-sensitive behaviors (and the public contract that this
    # helper is side-effect-free w.r.t. model state) depend on it. Defense in depth:
    # G161 keeps the mode consistent for the caller; G164 makes the dispatcher robust
    # to the mode being wrong anyway.
    was_training = model.training
    model.eval()
    try:
        # G173 — concrete implementation. The earlier passes left this as a
        # three-bullet comment, which led readers to roll inconsistent samplers
        # (some missing temperature scaling on the logits, some applying top_k
        # AFTER softmax — both common silent bugs that produce degenerate output
        # with no error).
        # G208 — accept caller-supplied tokenizer (default-construct only if
        # not provided, preserving the old single-arg call shape).
        tok = tokenizer if tokenizer is not None else Tokenizer()
        device = next(model.parameters()).device

        # Tokenize prompt. Empty prompt → BOS-equivalent start (just EOT).
        context_ids = torch.tensor(
            tok.encode(prompt) if prompt else [tok.eot_token],
            dtype=torch.long, device=device,
        ).unsqueeze(0)                                              # [1, P]
        block_size = model.config.block_size

        # Warm-up pass: process the prompt so the NMM sees full conv context.
        # The model returns logits over the whole prompt; we sample from the LAST
        # position. nmm_states is initialized inside model.forward when None passed.
        # doc_boundaries=None at inference — the prompt is one continuous context.
        nmm_states = None

        # G176 — for prompts > block_size, CHUNK the warm-up rather than truncating
        # to the last block_size tokens. The original snippet did:
        #     prompt_window = context_ids[:, -block_size:]
        #     logits, nmm_states = model(prompt_window, nmm_states, None)
        # which silently DROPPED everything before the last block_size tokens. For a
        # long-context QA prompt (e.g., "Here is a 5000-token document. Question:
        # what was on line 7?"), the model never sees lines 1–4500 of the document.
        # The NMM is the whole point of TITANS — let it actually process the prompt.
        # We feed the prompt in `block_size`-sized chunks (block_size keeps wpe in
        # bounds; using config.chunk_size would also work but is unnecessarily small).
        # `nmm_states` carries between chunks; positions restart at 0 each chunk,
        # matching the training-time chunking behavior.
        prompt_len = context_ids.size(1)
        for start in range(0, prompt_len, block_size):
            end = min(start + block_size, prompt_len)
            chunk = context_ids[:, start:end]
            logits, nmm_states = model(chunk, nmm_states, None)     # [1, T, V]
        next_logits = logits[:, -1, :]                              # [1, V]

        generated = []
        for _ in range(max_new_tokens):
            # ----- Sample one token from `next_logits` -----
            # 1. Temperature scaling: divide BEFORE softmax. temperature=0 is a
            #    degenerate "argmax" mode; clamp to a tiny epsilon to avoid div-by-zero.
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            else:
                scaled = next_logits / max(temperature, 1e-8)
                # 2. Top-k filtering BEFORE softmax. Mask logits outside top-k
                #    with -inf so they get zero probability after softmax. Common
                #    silent bug: applying top_k AFTER softmax then re-normalizing
                #    by hand — works but is unnecessarily complex and easy to
                #    get wrong (e.g., not handling ties or applying it after
                #    temperature instead of before).
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float('-inf'))
                probs = F.softmax(scaled, dim=-1)                   # [1, V]
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            generated.append(next_token.item())

            # EOS handling: stop when the EOT token is sampled. (For prompts that
            # explicitly want unbounded continuation across doc boundaries, the
            # caller can decode whatever was emitted and ignore this — but the
            # default is to respect the EOT.)
            if next_token.item() == tok.eot_token:
                break

            # ----- Update context for the next step -----
            # Append the new token. Slide the window to keep length <= block_size
            # so wpe(pos) stays in-bounds (G163 still applies if chunk_size <
            # block_size in from-scratch mode — see config warning).
            context_ids = torch.cat([context_ids, next_token], dim=1)
            window = context_ids[:, -block_size:]
            # NMM state continuity across calls: do NOT detach (we have no
            # gradients to break at inference under @torch.no_grad), but DO carry
            # `nmm_states` so memory accumulates. Caveat documented in the
            # "NMM reprocessing limitation" note below: the sliding window
            # re-feeds already-seen tokens through the NMM each step, double-
            # counting their updates. Acknowledged approximation for v1.
            logits, nmm_states = model(window, nmm_states, None)
            next_logits = logits[:, -1, :]

        return tok.decode(generated)
    finally:
        if was_training:
            model.train()              # G161 — restore original mode so any enclosing
                                       # training loop continues with the correct dispatcher
                                       # behavior.
```

**Eval mode (G156)** — both this script and `eval.py` (tasks 5.2, 5.3) MUST call
`model.eval()` AND wrap the inference loop in `torch.no_grad()` (or use the
`@torch.no_grad()` decorator above). Skipping either is a silent failure:
- **`model.eval()` missing** → dropout stays on (10% activations dropped at the
  HF GPT-2 default `config.dropout=0.1`). Log-probs are silently under-estimated,
  reported perplexity is silently inflated, and generated samples have extra
  randomness on top of the temperature/top-k mixing. A reader comparing perplexity
  to HF GPT-2's published numbers sees an apparent regression that is really a
  measurement bug. (Note: the Phase 6 scan dispatcher gates on
  `torch.is_grad_enabled()` post-G164, not on `self.training`, so the
  scan-vs-sequential choice is correctly driven by the surrounding `@torch.no_grad()`
  wrapper. The remaining reason to call `model.eval()` is dropout/LayerNorm-stats
  behavior, not the scan path.)
- **`torch.no_grad()` missing** → autograd builds the full chunked-forward graph
  including the second-order graph from `torch.func.grad` inside the NMM gradient
  computation. Memory blows up 5-10× compared to gradient-free inference. Long eval
  loops hit OOM on the same hardware that handled training.

**Position embedding strategy**: GPT-2's `wpe` covers positions 0…block_size-1. Each
`model.forward` call must receive a context with positions 0…T-1. During generation,
maintain a context buffer and always feed `context[-block_size:]` with positions
`arange(len(context[-block_size:]))`. This is correct as long as context ≤ block_size.
For longer generation, positions wrap (context drops old tokens), which is acceptable
because the NMM state carries long-range memory.

**Conv buffer limitation (known)**: `CausalDepthwiseConv1d` is stateless — its buffer is
NOT part of the NMM state. During token-by-token generation via `step()`, the conv
receives a 1-token window (3 zero-padding + 1 real token) vs. the T-token window used
during training. This is a train/inference discrepancy that may slightly degrade
generation quality for the NMM projections. Mitigations:
- Use the sliding-window approach above (process all T recent tokens through forward_chunk
  each step) — correct conv behavior, O(T) per step
- Or carry the conv buffer (a ring buffer of last kernel_size-1 activations) in the NMM
  state — pure O(1) per step but adds state complexity
- Or accept the discrepancy (conv mainly helps training dynamics; empirically often minor)

Default implementation: sliding-window approach (correctness over efficiency). The NMM
state carries memory across the full generation, not just the window.

**NMM reprocessing limitation (known)**: The sliding-window approach has a subtle
correctness issue for the NMM. When generating token t+1, we call `model.forward`
with the window `[tokens 0..t]` and carry the NMM state from the previous step.
But the previous step already processed tokens `[0..t-1]` through the NMM and updated
the state; now we process those same tokens again in this new call. Each old token
accumulates extra NMM updates at every subsequent generation step — a form of
double-counting that grows with each generated token.

The root cause: attention wants the full window context but NMM should process each
token exactly once. Reconciling these requires KV-cached attention + `step()` for the
single new token only:
```
attention: look up KV cache for old tokens + compute KV for new token
NMM:       call step(new_token, nmm_state) — exactly one new update
```
This is the correct architecture for generation but requires implementing a KV cache.
For a first version, the sliding-window reprocessing is an acknowledged approximation.
Document the discrepancy in comments at the `model.forward` call site in `generate.py`.

**Done:** produces non-degenerate (non-repeating) output from a GPT-2 prompt; perplexity
on a held-out set matches eval.py to within 0.5 bits/char (verifying correct position
management and NMM state continuity).

### 5.2 Perplexity evaluation
`eval.py`: stream dataset in chunks, carry NMM state within documents, reset at
document boundaries. Report perplexity and bits-per-character.

Must run under `model.eval()` + `torch.no_grad()` — see task 5.1's "Eval mode (G156)"
note. Skipping either silently inflates reported perplexity (dropout) or OOMs
(autograd graph). Aggregate as `total_nll / total_tokens` then `exp(...)`:

```python
@torch.no_grad()
def perplexity(model, loader, device):
    # G161: capture-and-restore the original training mode. A bare `model.eval()` leaves
    # the model in eval mode after this function returns. Pre-G164 this was a silent
    # correctness hazard: the Phase 6 dispatcher gated on `self.training`, so a sticky
    # eval mode silently re-routed training forwards through the autograd-broken scan
    # path, zeroing NMM gradients with loss still appearing to decrease (backbone-only
    # training). G164 changed the dispatcher to gate on `torch.is_grad_enabled()`, which
    # closes that specific hazard. The try/finally is still needed: dropout, LayerNorm
    # stats, and the public contract that `perplexity` is side-effect-free w.r.t.
    # model.training all require the mode to be restored. Defense in depth.
    was_training = model.training
    model.eval()
    try:
        total_nll  = 0.0
        total_toks = 0
        nmm_states = None
        for input_ids, doc_boundaries in loader:                  # ParallelStreamLoader is fine
            input_ids = input_ids.to(device, non_blocking=True)
            doc_boundaries = doc_boundaries.to(device, non_blocking=True)
            logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
            # reduction='sum' so we can aggregate across batches with the correct total_toks
            # divisor. reduction='mean' would average per-batch, masking variable batch lengths.
            nll = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction='sum',
            )
            total_nll  += nll.item()
            total_toks += input_ids[:, 1:].numel()
            # nmm_states do NOT need detach_states here — no graph is being built under no_grad.
        return math.exp(total_nll / total_toks)
    finally:
        if was_training:
            model.train()
```

Baseline: GPT-2 pretrained weights + untrained NMM should be within 5% of HF GPT-2
perplexity on Wikitext-103 (NMM initialized to near-zero output).

**Done:** baseline perplexity check passes; eval loop runs to completion on Wikitext-103;
torch.cuda.max_memory_allocated() during eval is bounded (no autograd-graph balloon).

### 5.3 Needle-in-haystack
`eval.py`: insert a random key-value fact at a random position in a long document;
measure whether the model correctly predicts the value when the key is repeated later.
Report accuracy vs. context length at {2K, 4K, 8K, 16K} tokens.

Same eval-mode requirement as 5.1/5.2 — wrap in `@torch.no_grad()` and call
`model.eval()` before the harness loop (G156).

**Done:** harness runs and produces a result table for untrained model (random accuracy baseline).

---

## Phase 6 — Associative Scan (Inference Speed Optimization)

The sequential token loop in `forward_chunk` is the main throughput bottleneck at
inference. The NMM update has the form of a linear recurrence that can be parallelized
with an associative scan.

**Critical autograd limitation**: `torch.associative_scan` explicitly states in its docs:
"It currently does not support autograd." Without `torch.compile`, using the scan path
during training silently breaks gradient flow — the outer optimizer cannot train Q/K/V
projections or W_θ/η/α through the scan. For **training**, either (a) use the sequential
loop (tasks 1.7/1.8, always correct), or (b) wrap the full model with `torch.compile`
before training, which converts the scan to a fused CUDA kernel that supports gradients.

Phase 6 provides reliable speedup for **inference** (generation, eval) where gradient
flow is not needed. It also provides training speedup when used inside a `torch.compile`
model on CUDA.

### 6.1 Pre-compute gradients, then scan
The key obstacle: `g_t = ∇_{M_{t-1}} ℓ_t` depends on `M_{t-1}`, which depends on
prior gradients. This makes `S_t = η_t·S_{t-1} - g̃_t` NOT a fixed-coefficient
linear recurrence — the "delta" term `g̃_t` changes as M evolves.

The standard approximation (used in lucidrains): compute ALL gradients at the start
of the chunk using the chunk's initial weights `M_0`, then run the linear scan:

```python
def _forward_chunk_scan(self, x_chunk, state_in, doc_boundaries):
    # x_chunk: [B, T, d]; state_in: (M, S) dicts of [B, h, d]; doc_boundaries: [B, T] bool or None
    # returns: (y_chunk [B, T, d], state_out (M, S) at last token)
    # CALLER MUST GUARD: doc_boundaries is None or has no True entries (the scan cannot perform
    # mid-chunk state resets). The `forward_chunk` dispatcher in task 6.2 enforces this.
    # CALLER MUST ALSO GUARD: not in training without torch.compile (autograd unsupported).
    M_state, S_state = state_in  # dicts of [B, h, d] — must unpack BEFORE Step 1

    # Projection pass: compute all T-length projections upfront (same pattern as task 1.8).
    # These tensors are then consumed by Step 1 (grad) and Step 4 (retrieval).
    k_hat_chunk = F.normalize(F.silu(self.k_proj(x_chunk)), dim=-1)  # [B, T, d]
    q_hat_chunk = F.normalize(F.silu(self.q_proj(x_chunk)), dim=-1)  # [B, T, d]
    v_chunk     = F.silu(self.v_proj(x_chunk))                        # [B, T, d]
    theta_chunk = torch.sigmoid(self.W_theta(x_chunk)).squeeze(-1)    # [B, T]
    eta_chunk   = torch.sigmoid(self.W_eta(x_chunk)).squeeze(-1)      # [B, T]
    alpha_chunk = torch.sigmoid(self.W_alpha(x_chunk)).squeeze(-1)    # [B, T]

    # Step 1: compute all gradients in parallel using M_state (chunk start weights).
    # per_sample_grad_fn = vmap(grad(inner_loss), in_dims=(0,0,0)) handles batch dim B.
    # Outer vmap over T (dim 1 of chunk tensors), M_state shared across T (None).
    all_grads = vmap(self.per_sample_grad_fn, in_dims=(None, 1, 1))(
        M_state, k_hat_chunk, v_chunk
    )  # dict of [T, B, h, d] tensors (T from outer vmap, B from inner)

    # Step 2: apply Newton-Schulz and theta scaling.
    # newton_schulz5 handles batched [..., h, d] inputs via dim=(-2,-1) naturally.
    # theta_chunk is [B, T]; outer vmap produced all_grads [T, B, h, d]
    # → transpose to [T, B] before unsqueeze so shapes align.
    th = theta_chunk.T.unsqueeze(-1).unsqueeze(-1)                    # [T, B, 1, 1]
    eta = eta_chunk.T.unsqueeze(-1).unsqueeze(-1)                     # [T, B, 1, 1]
    one_minus_alpha = (1 - alpha_chunk.T).unsqueeze(-1).unsqueeze(-1) # [T, B, 1, 1]

    scaled_grads = {}
    for key, g in all_grads.items():
        if self.nmm_spectral_norm:
            # batched NS over [T, B, h, d]: norm(dim=(-2,-1)) on last 2 dims.
            # The transpose guard in newton_schulz5 fires on dim=(-2,-1) shape — e.g.
            # W1 grads [T, B, 4d, d] (tall) are transposed per-NS to [T, B, d, 4d] and back.
            g = newton_schulz5(g)
        scaled_grads[key] = th * g  # θ applied POST-NS per paper + task 1.5/1.7

    # Step 3: associative scan for S and M per weight matrix.
    # Recurrence: S_t = η_t · S_{t-1} + δ_t   where δ_t = -θ_t · g̃_t
    # Associative op: (decay_a, δ_a) ⊗ (decay_b, δ_b) = (decay_b·decay_a, decay_b·δ_a + δ_b)
    def assoc_op(carry_a, carry_b):
        decay_a, delta_a = carry_a
        decay_b, delta_b = carry_b
        return (decay_b * decay_a, decay_b * delta_a + delta_b)

    S_chunk = {}  # S at each token within chunk: dict of [T, B, h, d]
    M_chunk = {}  # M at each token within chunk: dict of [T, B, h, d]

    # Incorporate state_in.S and state_in.M via the augmented-scan trick:
    # Prepend a synthetic element (decay=1, δ=S_0) before the real sequence.
    # The scan produces S_0 at position 0 and S_1..S_T at positions 1..T.
    # Slicing [1:] gives the correct T outputs, each incorporating S_0.
    # Same trick is applied to the M scan using S_chunk (with S_0 already embedded).
    one_B11 = eta.new_ones(1, *eta.shape[1:])   # [1, B, 1, 1] — identity decay for prepended step

    for key, dg in scaled_grads.items():
        S0 = S_state[key].unsqueeze(0)           # [1, B, h, d]
        M0 = M_state[key].unsqueeze(0)           # [1, B, h, d]

        # S scan incorporating S_0: prepend (1, S0) then drop position 0 from output.
        eta_aug = torch.cat([one_B11, eta], dim=0)          # [T+1, B, 1, 1]
        neg_dg_aug = torch.cat([S0, -dg], dim=0)            # [T+1, B, h, d]
        # combine_mode="generic" is REQUIRED: our assoc_op unpacks a tuple (carry_a, carry_b).
        # The default "pointwise" mode does not support tuple inputs and also requires CUDA.
        # torch.associative_scan returns the same pytree structure as xs (not a (carry, ys) pair).
        # So the return is (eta_accumulated, S_accumulated); we only need S_accumulated.
        # G215: use the resolved `_associative_scan` symbol (see Phase 6 task 6.2's
        # version-aware import) instead of bare `torch.associative_scan` so 2.6/2.7
        # users hit the private-path fallback transparently.
        _, S_aug = _associative_scan(
            assoc_op, (eta_aug, neg_dg_aug), dim=0, combine_mode="generic"
        )
        S_chunk[key] = S_aug[1:]                             # [T, B, h, d]

        # M scan incorporating M_0: prepend (1, M0) and use full S_chunk as deltas.
        alpha_aug = torch.cat([one_B11, one_minus_alpha], dim=0)  # [T+1, B, 1, 1]
        S_aug_delta = torch.cat([M0, S_chunk[key]], dim=0)        # [T+1, B, h, d]
        _, M_aug = _associative_scan(
            assoc_op, (alpha_aug, S_aug_delta), dim=0, combine_mode="generic"
        )
        M_chunk[key] = M_aug[1:]                             # [T, B, h, d]

    # Step 4: retrieve y_chunk using M_chunk via outer-T vmap + cached inner-B vmap (G140).
    # M_chunk: dict of [T, B, h, d]; q_hat_chunk: [B, T, d].
    # self._batched_retrieve is the cached vmap over B (built in NeuralMemoryModule.__init__).
    # outer vmap: M_chunk values have T at dim 0, q_hat_chunk has T at dim 1
    y_raw = vmap(self._batched_retrieve, in_dims=(0, 1))(M_chunk, q_hat_chunk)  # [T, B, d]
    y_chunk = self.out_scale * y_raw.transpose(0, 1)                             # [B, T, d]

    # Extract final state for cross-chunk continuity (TBPTT caller will detach these).
    # M_chunk / S_chunk are dicts of [T, B, h, d]; index [-1] gives the last token's state.
    state_out = (
        {k: v[-1] for k, v in M_chunk.items()},   # final M [B, h, d]
        {k: v[-1] for k, v in S_chunk.items()},   # final S [B, h, d]
    )
    return y_chunk, state_out
```

**Phase 6 doc_boundary limitation**: The associative scan cannot perform mid-chunk
state resets. If `doc_boundaries` contains True entries within a chunk, the scan path
silently ignores them — a correctness regression vs. the sequential path (task 1.8).
**Fix**: before calling `_forward_chunk_scan`, check for any True boundary within the
chunk. If found, fall back to `_forward_chunk_sequential` for that chunk:

```python
def forward_chunk(self, x_chunk, state_in, doc_boundaries):
    # `_allow_scan_training` is a flag on the module, default False. Set to True ONLY when the
    # entire model is wrapped in torch.compile — then associative_scan gets a fused kernel that
    # supports autograd. Without torch.compile, `torch.associative_scan` "does not support
    # autograd" (per its docs) and the scan path silently breaks gradient flow back through the
    # NMM projections and W_θ/η/α. The sequential path is always autograd-safe.
    can_scan = _HAS_ASSOC_SCAN and (doc_boundaries is None or not doc_boundaries.any())
    # G164: gate on torch.is_grad_enabled(), NOT self.training. These two flags are independently
    # controlled and need not agree:
    #   - self.training is the Python-level mode flag toggled by model.train()/model.eval()
    #   - torch.is_grad_enabled() reflects the ACTUAL autograd state (off inside torch.no_grad
    #     or torch.inference_mode; on otherwise)
    # The scan path's incompatibility is with AUTOGRAD, not with training-mode-the-flag, so the
    # correct condition is is_grad_enabled. The earlier `self.training`-only guard had a silent
    # failure mode: a user who called model.eval() for a sanity check (e.g., `generate(...)`)
    # and forgot to call model.train() before starting the training loop ended up with
    # self.training=False but autograd ENABLED in the training loop. The guard `if self.training
    # and not _allow_scan_training` evaluated False, can_scan stayed True, and the scan ran in
    # an autograd-enabled context — silently zeroing gradients on Q/K/V/W_θ/W_η/W_α/memory_mlp.W*.
    # The backbone (attention/MLP/embeddings) still trained because those paths don't touch the
    # scan, so loss decreased normally and the user had no visible signal that the NMM was
    # effectively frozen at init. Notably G161's try/finally in generate() does NOT defend
    # against this: it captures `was_training = model.training` and only restores train mode
    # `if was_training` — if the caller was already in eval mode at entry, the helper leaves
    # the model in eval mode on exit.
    #
    # The is_grad_enabled gate is robust to any train/eval/no_grad combination:
    #   - eval-mode helpers wrapped in @torch.no_grad: grad_enabled=False → scan allowed (correct)
    #   - training with autograd on: grad_enabled=True, opt-in False → sequential (correct)
    #   - training with torch.compile + _allow_scan_training=True: scan (correct)
    #   - accidental model.eval() during training: still grad_enabled=True → sequential (safe)
    if torch.is_grad_enabled() and not getattr(self, '_allow_scan_training', False):
        can_scan = False
    if can_scan:
        return self._forward_chunk_scan(x_chunk, state_in, doc_boundaries)
    return self._forward_chunk_sequential(x_chunk, state_in, doc_boundaries)
```

This is the approximation (gradients at M_0 not M_{t-1}), but it is the practical
approach. The S_0/M_0 incorporation from state_in is required for cross-chunk continuity.

**Done:** scan output approximates sequential reference; max relative error < 5% on
a random sequence, monotonically decreasing with shorter chunk_size.

### 6.2 Integrate with `torch.associative_scan`
Replace the Python loop in `forward_chunk` with `torch.associative_scan`.

**Version requirement**: `torch.associative_scan` was added in **PyTorch ≥ 2.8** as an
experimental/prototype API (`[API-Unstable]`). The base model (phases 0–5) works with
`torch>=2.3`. Phase 6 specifically requires `torch>=2.8`.

**Additional requirements**: `torch.associative_scan` requires `torch.compile` for
autograd support and currently only supports CUDA for `combine_mode="pointwise"`. We use
`combine_mode="generic"` (see Phase 6 task 6.1 scan calls), which supports CPU and CUDA
but uses `vmap` internally — slower than the CUDA `pointwise` kernel on GPU. For peak
training throughput on CUDA, switch to a custom triton kernel or wait for the API to
stabilize.

Implementation strategy:
```python
import torch
# G215 — `torch.associative_scan` was introduced as a prototype API and its
# canonical import path has shifted across PyTorch versions:
#   - 2.8.0+: exposed as `torch.associative_scan` (the documented path).
#   - 2.6.0–2.7.x: lived under `torch._higher_order_ops.associative_scan`
#     (private namespace; user code reaching here got DeprecationWarning).
#   - 2.5.x and earlier: did not exist at all.
# A bare `hasattr(torch, 'associative_scan')` returns False on 2.6/2.7 even
# though the API IS available under the private path — users on those
# versions silently fall back to sequential when they could have used scan.
# Worse, a future PyTorch release could move it back to a private namespace
# during refactoring; we'd silently lose the speedup with no warning.
# Defensive probe: try the documented path first, fall back to the private
# path, only set False when both miss. Bind the resolved function to a
# module-level name (`_associative_scan`) so the rest of the file can call
# it without re-doing the lookup. This isolates the version-dependence to
# one place.
try:
    from torch import associative_scan as _associative_scan
    _HAS_ASSOC_SCAN = True
except ImportError:
    try:
        from torch._higher_order_ops import associative_scan as _associative_scan
        _HAS_ASSOC_SCAN = True
    except ImportError:
        _associative_scan = None
        _HAS_ASSOC_SCAN = False
# Callers in task 6.1 (`_forward_chunk_scan`) should use `_associative_scan(...)`
# instead of `torch.associative_scan(...)` so the private-path fallback is
# transparent. Sequential path is taken whenever _HAS_ASSOC_SCAN is False.

def forward_chunk(self, x_chunk, state_in, doc_boundaries):
    can_scan = _HAS_ASSOC_SCAN and (doc_boundaries is None or not doc_boundaries.any())
    # G164: gate on torch.is_grad_enabled(), NOT self.training. See task 6.1 for the
    # full failure-mode write-up. Short version: self.training and is_grad_enabled are
    # independently controlled; the scan's incompatibility is with autograd, so probe
    # the autograd flag directly. A `model.eval()`-then-forget-to-`model.train()` pattern
    # silently breaks NMM gradient flow under the older self.training gate.
    if torch.is_grad_enabled() and not getattr(self, '_allow_scan_training', False):
        can_scan = False
    if can_scan:
        return self._forward_chunk_scan(x_chunk, state_in, doc_boundaries)
    return self._forward_chunk_sequential(x_chunk, state_in, doc_boundaries)
```

Update `requirements.txt` to note: `torch>=2.3  # Phase 6 scan requires >=2.8`.

**G180 — `_allow_scan_training` is a per-NMM flag.** The dispatcher reads
`getattr(self, '_allow_scan_training', False)` where `self` is the individual NMM
submodule. To opt into scan-during-training, the flag must be set on EVERY NMM in the
model (one per block). The natural-but-wrong pattern is:

```python
model = TitansMAGGPT2(config)
model._allow_scan_training = True       # wrong — sets on the top-level model, not NMMs
model = torch.compile(model)
```

This silently does nothing — `forward_chunk` is on the NMM, and `self._allow_scan_training`
there will still resolve to the default `False`. Provide a helper to do the right thing
in one call:

```python
def allow_scan_training(model, enabled=True):
    """Opt in/out of the scan path during training for every NMM in the model.

    Only call this AFTER wrapping the model with torch.compile (or knowing that you
    will). The scan path lacks autograd outside torch.compile; without compile, this
    silently zeros NMM gradients at training time (see G164's full failure mode).
    """
    for block in model.blocks:
        block.nmm._allow_scan_training = bool(enabled)

# Usage:
model = TitansMAGGPT2(config).to(device)
model = torch.compile(model)            # gives associative_scan autograd via fused kernel
allow_scan_training(model, True)
```

The helper iterates `model.blocks`. If the user wrapped the model in `torch.compile`,
the wrapper's attribute access pass-through resolves `.blocks` to the inner model's
list, so the same code works pre- and post-compile.

**G184 — `torch.compile` state_dict prefix.** `torch.compile(model)` returns an
`OptimizedModule` whose `state_dict()` prefixes every key with `_orig_mod.` (e.g.,
`_orig_mod.blocks.0.attn.q_proj.weight`). Saving this directly to a checkpoint and
loading into a non-compiled model fails with a "Missing key(s) in state_dict" /
"Unexpected key(s) in state_dict" error from `load_state_dict(strict=True)`. The fix
on the save side is to unwrap before saving:

**G188 — define `_unwrap` as a top-level utility.** This helper is referenced from
both the consolidated training loop (task 4.5) and the periodic-checkpoint save in
task 4.3. Define it once in `model/__init__.py` (or a tiny `utils.py`) so all call
sites import the same function rather than re-defining it inline. Example placement:

```python
# in model/__init__.py:
def _unwrap(m):
    """Return the underlying nn.Module behind torch.compile / DDP / FSDP wrappers.

    G184 — torch.compile(model) returns an OptimizedModule whose state_dict
    prefixes every key with `_orig_mod.`. Incompatible with load_state_dict on
    an uncompiled model.

    G195 — DistributedDataParallel(model) and FullyShardedDataParallel(model)
    both wrap the inner model in `.module`. Their state_dict prefixes every key
    with `module.` (a separate prefix from torch.compile's `_orig_mod.`).
    Earlier versions of _unwrap only stripped `_orig_mod.`, leaving DDP-wrapped
    saves with `module.*` keys that fail to load into an unwrapped model with
    the same "Missing key(s)" / "Unexpected key(s)" error. Worse: stacked
    DDP(torch.compile(model)) wraps as `module._orig_mod.<key>` — needs both
    strips.

    Iterate to handle any wrapper stacking (DDP-over-compile, compile-over-DDP,
    deeper if someone gets creative). The loop terminates because each strip
    removes one wrapper layer; the inner model has neither attribute.
    """
    while hasattr(m, 'module') or hasattr(m, '_orig_mod'):
        m = getattr(m, 'module', m)        # DDP / FSDP
        m = getattr(m, '_orig_mod', m)     # torch.compile
    return m
```

Usage:

```python
torch.save({
    'state_dict': _unwrap(model).state_dict(),    # never has _orig_mod. prefix
    'optimizer':  optimizer.state_dict(),
    'step':       step,
    'config':     dataclasses.asdict(config),
}, ckpt_path)
```

On the load side, the resume sequence in task 4.3 rebuilds an UNCOMPILED model from the
saved config, calls `load_state_dict`, and only then optionally wraps with
`torch.compile`. The unwrapped state_dict matches the uncompiled model's keys
unconditionally. If the user instead saved with the compiled wrapper (no `_unwrap`),
the keys carry the `_orig_mod.` prefix and resume fails loudly — but the error message
is unhelpful ("Missing key(s): blocks.0.attn.q_proj.weight; Unexpected key(s):
_orig_mod.blocks.0.attn.q_proj.weight"), so users blame their config or model
structure rather than the wrapping. Always use `_unwrap` at save time when `torch.compile`
is in play.

**Done:** inference (generation) step wall-clock time improves ≥ 20% on T=1024 sequences on
PyTorch ≥ 2.8 CUDA; falls back to sequential loop on earlier versions or when doc boundaries
are present. Training speedup requires wrapping model with `torch.compile` before training.

---

## Testing Checkpoints

> **For the full test plan** — including unit/integration/parity/behavior/DDP/performance/
> failure-mode coverage, test infrastructure, CI tiers, and a complete G-number →
> defending-test regression matrix — see `TEST_PLAN.md`. The table below is the
> condensed task-ordered checkpoint list; `TEST_PLAN.md` is the authoritative
> superset organized by component.

**G181** — earlier versions of this table covered only a fraction of the gaps logged
in GAP_HISTORY.md. Each table row below now points to a specific gap-driven assertion
that, if violated, surfaces the silent failure it defends against. Run each test
right after the listed task lands; together they form a regression net for every
correctness-critical decision the audit found.

| Test | After task | Defends |
|---|---|---|
| ✓ Config `__post_init__` rejects chunk_size > block_size (raises ValueError, NOT AssertionError) | 0.2 | wpe OOB / G206 |
| ✓ Factory `TitansConfig.gpt2_small().n_embd == 768` | 0.2 | G143/G150 |
| ✓ `__post_init__` warns when from-scratch + chunk_size < block_size | 0.2 | G163 |
| ✓ `__post_init__` rejects `use_swa=True, swa_window=0` | 0.2 | G166 |
| ✓ Conv output shape invariant; conv strictly causal | 1.1 | conv design |
| ✓ q̂/k̂ L2 normalized, ‖v‖ unconstrained | 1.2 | activation/L2 placement |
| ✓ W_θ/η/α outputs shape `[B, T]` (not `[B, T, 1]`) | 1.3 | broadcasting bug |
| ✓ out_scale init: zeros when finetune_mode=True, ones when False | 1.4 | G123 |
| ✓ Inner-loss gradient matches `autograd.functional.jacobian` to 1e-4 | 1.5 | torch.func correctness |
| ✓ Reduction switch: spectral_norm=False gradient ≈ 1/d × spectral_norm=True | 1.5 | G160 |
| ✓ Newton-Schulz spectral norm bound ≈ 1 for random matrices any shape | 1.6 | NS5 correctness |
| ✓ NS transpose guard: tall vs wide matrices both work | 1.6 | NS convergence |
| ✓ NS5 forces fp32 internally; post-NS spectral norm ≈ 1 even under bf16 autocast | 1.6 | G198 |
| ✓ NS5 explicitly disables autocast — `G @ G.mT` runs fp32 (NOT bf16) under ambient autocast scope | 1.6 | G226 |
| ✓ Consolidated try/finally block uses CONSISTENT 4-space indentation (not mixed 2-2-4) | 4.5 | G227 |
| ✓ `step()` T-token sequential matches reference manual loop | 1.7 | step semantics |
| ✓ `_forward_chunk_sequential` ≠ T-many `step()` calls (conv sees full chunk) | 1.8 | G154 |
| ✓ `_forward_chunk_sequential` precomputes per-t boundary mask on CPU (no per-token GPU sync) | 1.8 | G202 |
| ✓ `_forward_chunk_sequential` lazy-builds `init_M` only when a doc boundary fires | 1.8 | G211 |
| ✓ NMM memorizes single key→value pair (overfit) | 1.8 | end-to-end correctness |
| ✓ reset_state on doc boundaries leaves unmasked entries byte-identical | 1.9 | G149 |
| ✓ detach_states handles `None` (first-step case) | 1.9 | G149 |
| CausalSelfAttention with `_aug_mask(T)` produces correct persistent/causal pattern | 2.0 | attention mask |
| ✓ CausalSelfAttention rejects `n_head` not dividing `n_embd` via `ValueError` (NOT AssertionError, NOT silent under -O) | 2.0 | G220 |
| ✓ GPT2MLP matches `F.gelu(c_fc(x), approximate='tanh')` elementwise | 2.0 | HF parity |
| ✓ Persistent token mask: top-right block is -inf (persistent ⊥ real) | 2.1 | mask block structure |
| ✓ ln_nmm appears in state_dict, independent of ln_1 | 2.2 | separate norms |
| ✓ MAG additive gate at out_scale=0 equals y_attn exactly | 2.3 | finetune init |
| ✓ SWA banded mask: row i attends to j ∈ (i-swa_window, i] | 2.4 | G136 |
| ✓ Block forward shape-invariant; jit-traceable | 2.4 | block correctness |
| ✓ `_apply_gpt2_init`: `wte.weight.std() ≈ 0.02` (NOT ≈ 1) post-init | 2.5 | G155 |
| ✓ `_apply_gpt2_init`: `attn.proj.weight.std()` ≈ 0.02/√(2·n_layer) | 2.5 | G155 residual scaling |
| ✓ `_apply_gpt2_init` skips NMM-internal modules by id (renaming `self.nmm` doesn't break it) | 2.5 | G203 |
| ✓ `_build_init_M` uses `.to(device).clone()` order (no wasted source-device copy) | 1.4 | G207 |
| ✓ GPT-2 weight parity vs HF (NMM zeroed, N_p=0); max-logit-diff < 1e-4 | 2.6 | weight load correctness |
| ✓ `load_pretrained` derives HF model name from `config.n_embd` (medium/large/xl work) | 2.6 | G216 |
| ✓ Tokenizer: `decode(encode(s)) == s` for ASCII; `eot_token == 50256` | 3.1 | tokenizer basics |
| ✓ Tokenizer: literal `<|endoftext|>` in text is BPE-encoded (NOT EOT id) | 3.1 | G152 |
| ✓ `encode_corpus(open(path))` is treated as line-per-doc (warn user); whole-file pattern recommended | 3.1 | G210 |
| ✓ ParallelStreamLoader: position-i stream is contiguous across batches | 3.3 | G151 |
| ✓ Optimizer has exactly 4 param groups; no param in multiple groups | 4.1 | G117 / 4-group split |
| ✓ Optimizer betas == (0.9, 0.95); no_decay set excludes 'bias'/'ln'/'norm'/'out_scale'/'gamma'/'persistent' | 4.1 | G153 / G117 |
| ✓ `train_step` does the `.to(device)` transfer (CPU batch → GPU works) | 4.2 | G167 |
| ✓ Loss decreases monotonically on 100-step overfit batch | 4.2 | end-to-end |
| ✓ Injected `loss = loss + nan` does NOT corrupt parameters | 4.2 | G158 |
| ✓ NaN-skip resets `nmm_states` to None (caller's next call re-initializes) | 4.2 | G213 |
| ✓ NaN-skip in accumulation block also resets `nmm_states` to None | 4.5 | G217 |
| ✓ bf16 autocast: backward + clip + step run in fp32 | 4.2 | G159 |
| ✓ `apply_lr` scales all 4 param groups; preserves 1:1:3:3 ratio | 4.3 | G157 |
| ✓ `apply_lr` respects user-supplied `max_steps`/`warmup_steps` (NOT defaults) | 4.3 | G175 |
| ✓ `base_lrs` constants survive `optimizer.load_state_dict` deflation | 4.3 | G162 |
| ✓ DDP-aware checkpoint save fires only on rank 0; barrier follows | 4.3, 4.5 | G199 (structural test; ddp runtime untested) |
| Consolidated loop wraps with DDP after `.to(device)`, before optimizer | 4.5 | G201 (ddp-marked) |
| `init_process_group` called once before DDP wrap; `destroy_process_group` at end | 4.5 | G201 (ddp-marked) |
| Per-rank seed differs after model construction (dropout masks diverge) | 4.5 | G204 (ddp-marked) |
| Gradient accumulation under DDP uses `model.no_sync()` for all but last micro-batch | 4.5 | G200 (ddp-marked) |
| Partial DDP accumulation cycle skips optimizer.step (avoids rank divergence) | 4.5 | G214 (ddp-marked) |
| ✓ Partial cycle detected ALSO when StopIteration fires at iter K-1 (off-by-one defended) | 4.5 | G222 |
| ✓ `TitansConfig(n_embd=768, n_head=10)` raises `ValueError` at config time (NOT model time) | 0.2 | G223 |
| ✓ `_apply_gpt2_init` uses relative import `from .nmm` (works under any top-level package name) | 2.5 | G224 |
| ✓ Training loop wrapped in try/finally; `destroy_process_group` runs on exception path | 4.5 | G225 (structural; ddp runtime untested) |
| ✓ Consolidated loop builds `config` BEFORE the loader references `config.chunk_size` | 4.5 | G205 |
| Resume sequence wraps DDP after `load_state_dict` and before optimizer build | 4.3 | G209 (ddp-marked) |
| ✓ Checkpoint save/load round-trip: state_dict + optimizer state populated post-resume | 4.3 | G153 / G134 / G168 |
| ✓ Resume from HF-init checkpoint (no 'optimizer' key) does not raise KeyError | 4.3 | G219 |
| ✓ Resume sequence ends with `model.train()` (defense against prior eval-mode code) | 4.3 | G221 |
| ✓ `torch.load(..., weights_only=False)` succeeds on saved checkpoint | 4.3 | G168 |
| ✓ `compute_nmm_norm(None) is None`; non-None returns one float per layer | 4.3 | G172 |
| ✓ `generate` chunks prompts > block_size (full prompt seen by NMM) | 5.1 | G176 |
| ✓ `generate` applies temperature, then top_k, THEN softmax | 5.1 | G173 |
| ✓ `generate(model, prompt, tokenizer=tok)` reuses caller's tokenizer instance | 5.1 | G208 |
| ✓ `generate` / `perplexity` restore `model.training` post-exit (try/finally) | 5.1, 5.2 | G161 |
| ✓ Perplexity baseline within 5% of HF GPT-2 (NMM zeroed, eval mode + no_grad) | 5.2 | G156 |
| ✓ Scan path matches all-grads-at-M_0 sequential exactly (implementation correctness); approximation cost vs true sequential is loose at random init (G232) | 6.1 | scan approximation |
| ✓ Scan dispatcher gates on `torch.is_grad_enabled()` (NOT `self.training`) | 6.2 | G164 |
| ✓ `_HAS_ASSOC_SCAN` resolves via documented or private import path; both succeed | 6.2 | G215 |
| ✓ `allow_scan_training(model, True)` sets the flag on every block.nmm | 6.2 | G180 |
| ✓ `torch.compile` checkpoint: `_unwrap(model).state_dict()` has no `_orig_mod.` keys | 6.2 | G184 |

---

## Gap History

See [GAP_HISTORY.md](GAP_HISTORY.md) for the full audit log (227 gaps across 53 passes).

<!-- Removed from this file to keep PLAN.md focused on implementation. -->
<!-- PLACEHOLDER — do not add gap entries here; append to GAP_HISTORY.md instead. -->

