"""Neural Memory Module: SiLU-GLU gated MLP with online surprise-driven weight updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NeuralMemoryModule(nn.Module):
    """Online memory module with surprise-driven weight updates.

    Holds the data-dependent update params W_theta, W_eta, W_alpha:
    three Linear(d, 1, bias=False) modules whose sigmoid outputs are
    per-token scalars (learning rate, momentum decay, forgetting rate).
    Call site is `sigmoid(self.W_theta(x)).squeeze(-1)` — squeezing to
    [B, T] (not [B, T, 1]) is required for downstream broadcasting and
    for grad() to receive a scalar inner loss.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.W_theta = nn.Linear(n_embd, 1, bias=False)
        self.W_eta = nn.Linear(n_embd, 1, bias=False)
        self.W_alpha = nn.Linear(n_embd, 1, bias=False)
