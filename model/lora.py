"""LoRA: low-rank adaptation for backbone attention projections.

TPTT recipe (arxiv 2506.17671 §4.1): rank 8, alpha=16, dropout 0.05, applied
to q/k/v/o projections of the backbone attention layers. Base weights are
frozen; only the rank-r adapters train. Memory pathway is unaffected (still
fully trainable).

The mechanism: each adapted Linear produces
    y = W·x + (alpha/rank) · B·A·x
where A is rank×in_features (init: Kaiming uniform), B is out×rank (init:
zeros). At step 0, the LoRA contribution is exactly zero so the model
behaves identically to the unwrapped backbone; gradient descent then moves
A and B to adapt attention behavior within a rank-r-bounded perturbation.

Why this is in the recipe at all: full-fine-tune of backbone attention
gives the optimizer the easy path of tweaking attention's existing
behavior, which can drown out the memory pathway's gradient signal. LoRA
bottlenecks backbone adaptation to a small subspace, making the memory
pathway the optimizer's path of least resistance for any improvement the
limited-rank attention can't deliver. TPTT credits this for stable
fine-tuning convergence; whether it actually helps cross-chunk retrieval is
the question we're testing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """nn.Linear + rank-r LoRA adapter.

    The base `nn.Linear(in_features, out_features, bias)` is created and
    its weight/bias are exposed as `self.linear.weight` / `self.linear.bias`
    so existing pretrained-weight loaders (which `copy_` directly into
    `block.attn.q_proj.weight`) continue to work — they hit
    `block.attn.q_proj.linear.weight` after one extra `.linear` traversal,
    which we plumb in `model/load_pretrained.py`'s LoRA-aware path.

    When `freeze_base=True`, `self.linear.weight` and `self.linear.bias`
    get `requires_grad=False`. The LoRA adapter params (`lora_A`, `lora_B`)
    stay trainable regardless.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        freeze_base: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(
                f"LoRA rank must be > 0 (got {rank}); use plain nn.Linear if "
                f"you don't want LoRA."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if freeze_base:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False

        # LoRA matrices. Standard init: A is Kaiming uniform, B is zero so
        # the LoRA contribution starts at exactly zero (model behaves
        # identically to base at step 0).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = (
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # LoRA path: (x @ A^T) @ B^T, scaled by alpha/rank.
        # F.linear(input, weight, bias=None) is equivalent to input @ weight.T.
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base + self.scaling * lora_out
