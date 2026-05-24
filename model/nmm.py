"""Neural Memory Module: SiLU-GLU gated MLP with online surprise-driven weight updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def _make_grad_fn(memory_mlp: nn.Module, spectral_norm: bool):
    """Build the cached vmap(grad(inner_loss)) per-sample gradient function.

    The reduction is derived from `spectral_norm`, NOT hardcoded:
      - True: 'sum' — Newton-Schulz normalises the gradient's spectral norm
        to 1, so the d_model factor cancels and sum vs mean is invisible
        downstream. Matches paper Eq. 12 (squared L2 norm).
      - False: 'mean' — without NS, 'sum' would let the gradient grow with
        d_model, scaling W_theta's effective LR by ~d. 'mean' keeps the
        gradient magnitude independent of d.

    Mismatch between this flag and the downstream NS branch is silent and
    catastrophic: a user toggling nmm_spectral_norm=False after construction
    keeps the cached 'sum' reduction (no NS to cancel the d factor),
    inflating per-token LR ~768x at gpt2_small dims. Treat
    nmm_spectral_norm as construction-time-only; rebuild the NMM to change it.
    """
    reduction = "sum" if spectral_norm else "mean"

    def inner_loss(params, k_hat, v):
        # params: dict of per-sample weights ([h,d] / [d,h]); k_hat, v: [d]
        pred = functional_call(memory_mlp, params, k_hat)
        return F.mse_loss(pred, v, reduction=reduction)

    # argnums=0 (default) -> grad w.r.t. params (must stay first arg).
    return vmap(grad(inner_loss), in_dims=(0, 0, 0))


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D conv with strict causality (left-only padding).

    Public API operates on `[B, T, dim]` (transformer convention); the
    underlying nn.Conv1d expects `[B, dim, T]`. Padding inside Conv1d would
    be symmetric, which leaks future tokens; we pad left-only by `k-1`.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1 (got {kernel_size})")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=0, groups=dim, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        x = x.transpose(1, 2)                              # [B, dim, T]
        x = F.pad(x, (self.kernel_size - 1, 0))            # left-pad only
        x = self.conv(x)                                    # [B, dim, T]
        return x.transpose(1, 2)                           # [B, T, dim]


class NMMProjection(nn.Module):
    """Q/K/V projection: Linear -> CausalDepthwiseConv1d, no activation.

    SiLU + L2-norm are applied at the call site, not inside the module —
    putting SiLU inside would silently produce silu(silu(x)) at the call site.

    Submodule names `linear` and `conv` are load-bearing for §4.1 optimizer
    routing: param paths like `blocks.X.nmm.k_proj.linear.weight` get into
    the NMM decay group via the `'nmm'` substring; renaming to anything
    containing `'norm'`, `'bias'`, or `'gamma'` would misroute to no_decay.
    """

    def __init__(self, n_embd: int, kernel_size: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.conv = CausalDepthwiseConv1d(n_embd, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.linear(x))


class MemoryMLP(nn.Module):
    """SiLU-GLU gated two-layer MLP (L_M = 2) with ResidualNorm.

    silu(W1 x) * sigmoid(W_gate x)  ->  W2  ->  norm(.) + x

    norm is a fixed stabilizer trained by the outer optimizer only — it is
    NOT recurrent state. Only the three 2D weights {W1, W_gate, W2} live in
    the recurrent (M, S); Newton-Schulz operates on 2D matrices.
    """

    def __init__(self, d: int, expansion: int = 4):
        super().__init__()
        h = d * expansion
        self.W1 = nn.Linear(d, h, bias=False)
        self.W_gate = nn.Linear(d, h, bias=False)
        self.W2 = nn.Linear(h, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.W1(x)) * torch.sigmoid(self.W_gate(x))
        y = self.W2(h)
        return self.norm(y) + x


class NeuralMemoryModule(nn.Module):
    """Online memory module with surprise-driven weight updates.

    The constructor wires together the components from PLAN.md §1.2–§1.4:
    Q/K/V projections, the three data-dependent update params, the
    MemoryMLP (whose W*.weight ARE the meta-learned initial state), and
    a learnable per-channel output scale.

    Phase 1.4 surface: __init__, _build_init_M, init_state. The cached
    per-sample gradient function (§1.5), Newton-Schulz (§1.6), step
    (§1.7), and forward_chunk (§1.8) are added by subsequent tasks.
    """

    def __init__(
        self,
        n_embd: int,
        expansion: int = 4,
        kernel_size: int = 4,
        spectral_norm: bool = True,
        finetune_mode: bool = True,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.nmm_spectral_norm = spectral_norm
        self.finetune_mode = finetune_mode

        # Q/K/V projections — SiLU/L2 applied at call site, not inside.
        self.k_proj = NMMProjection(n_embd, kernel_size)
        self.q_proj = NMMProjection(n_embd, kernel_size)
        self.v_proj = NMMProjection(n_embd, kernel_size)

        # Per-token data-dependent update params (sigmoid+squeeze at call site).
        self.W_theta = nn.Linear(n_embd, 1, bias=False)
        self.W_eta = nn.Linear(n_embd, 1, bias=False)
        self.W_alpha = nn.Linear(n_embd, 1, bias=False)

        # MemoryMLP. Its W*.weight ARE the meta-learned initial values of M;
        # _build_init_M reads them at sequence/document start to seed state.
        self.memory_mlp = MemoryMLP(n_embd, expansion)
        nn.init.xavier_uniform_(self.memory_mlp.W1.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W_gate.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W2.weight)

        # finetune_mode=True: zero-init silences y_mem at step 0, preserving
        # the pretrained GPT-2 residual exactly (ResidualNorm makes a bare
        # W2=0 not enough; out_scale is the only reliable mute).
        # finetune_mode=False: NMM contributes from step 1, no residual to preserve.
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(n_embd))
        else:
            self.out_scale = nn.Parameter(torch.ones(n_embd))

        # Per-sample inner gradient — built ONCE; recreating vmap(grad(...)) in
        # forward/step is measurably slower at T=512. Reduction is baked in via
        # spectral_norm at construction time; mutating self.nmm_spectral_norm
        # later does NOT change the cached reduction. Lock the construction
        # value so downstream sanity checks can detect drift.
        self.per_sample_grad_fn = _make_grad_fn(
            self.memory_mlp, spectral_norm=self.nmm_spectral_norm
        )
        self._spectral_norm_at_init = self.nmm_spectral_norm

        # Per-sample retrieval — same caching rationale. functional_call reads
        # self.memory_mlp at call time, so device moves of NMM after __init__
        # are still respected.
        def _retrieve_one_sample(m_dict, q):
            return functional_call(
                self.memory_mlp, m_dict, q.unsqueeze(0)
            ).squeeze(0)

        self._batched_retrieve = vmap(_retrieve_one_sample, in_dims=(0, 0))

    def _build_init_M(self, B: int, device) -> dict:
        """Per-sample-batched initial M dict from memory_mlp.W*.weight.

        `.expand(B, -1, -1)` is a stride-0 view; vmap with in_dims=0 over
        such views is undefined in the batched autograd interpreter, so
        we `.clone()` to materialize normal strides. Order is
        `.to(device).clone()` (not `.clone().to(device)`): on the same
        device the two are equivalent, but cross-device the former avoids
        a wasted source-device allocation.
        """
        W1 = self.memory_mlp.W1.weight
        W_gate = self.memory_mlp.W_gate.weight
        W2 = self.memory_mlp.W2.weight
        return {
            "W1.weight":     W1.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
            "W_gate.weight": W_gate.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
            "W2.weight":     W2.unsqueeze(0).expand(B, -1, -1).to(device).clone(),
        }

    def init_state(self, B: int, device) -> tuple:
        M = self._build_init_M(B, device)
        S = {k: torch.zeros_like(v) for k, v in M.items()}
        return (M, S)
