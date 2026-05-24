"""TitansMAGBlock: attention + NMM combined via a learned MAG gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Standard GPT-2 multi-head self-attention.

    Q/K/V projections are split (not fused into a single c_attn) so HF weight
    loading in Phase 2.6 can copy each independently. The mask is passed in
    by the caller — this module does NOT impose causality on its own.
    `attn_mask=None` attends to all positions; always pass an explicit mask.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head != 0:
            head_dim = n_embd // n_head
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head}). "
                f"Got n_embd % n_head = {n_embd % n_head} (head_dim would be "
                f"{head_dim}, which yields {head_dim * n_head}, not {n_embd})."
            )
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # HF GPT-2 c_attn and c_proj both have biases — required for parity.
        self.q_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        n, d = self.n_head, self.head_dim
        q = self.q_proj(x).view(B, T, n, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, n, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n, d).transpose(1, 2)
        dp = self.resid_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class GPT2MLP(nn.Module):
    """Standard GPT-2 feedforward. Uses gelu approximate='tanh' (HF's gelu_new)
    — error-function gelu produces logit drift vs HF GPT-2."""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))
