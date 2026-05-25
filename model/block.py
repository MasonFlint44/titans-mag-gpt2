"""TitansMAGBlock: attention + NMM combined via a learned MAG gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from model.nmm import NeuralMemoryModule


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

    def project_kv(self, x: torch.Tensor) -> tuple:
        """Project x into (K, V) tensors shaped [B, n_head, T, head_dim].

        Used to seed the KV cache during warm-up: caller runs ln_1(x_aug),
        passes that here, gets the K, V that the attention would have used,
        stores them. No attention is computed — that runs separately during
        the warm-up forward.
        """
        B, T, C = x.shape
        n, d = self.n_head, self.head_dim
        k = self.k_proj(x).view(B, T, n, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n, d).transpose(1, 2)
        return k, v

    def forward_with_kv_cache(
        self,
        x_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple:
        """Decode-path attention: project a single new token's Q/K/V,
        append the new K, V to the cache, run SDPA with Q against the
        full cached K, V.

        x_new: [B, 1, C] — the new token's pre-attention input (already
        ln_1-normalized by the caller).
        k_cache, v_cache: [B, n_head, T_seen, head_dim] — prior K, V.

        Returns (y [B, 1, C], new_k_cache, new_v_cache).

        No mask is needed: the new token attends to everything in the
        cache (all of which is in its causal past or is a persistent
        token, both of which it's allowed to see). dropout_p=0 because
        decode is inference-time.
        """
        B, T_new, C = x_new.shape
        assert T_new == 1, f"forward_with_kv_cache expects T=1, got T={T_new}"
        n, d = self.n_head, self.head_dim

        q = self.q_proj(x_new).view(B, 1, n, d).transpose(1, 2)  # [B, n, 1, d]
        k_new = self.k_proj(x_new).view(B, 1, n, d).transpose(1, 2)
        v_new = self.v_proj(x_new).view(B, 1, n, d).transpose(1, 2)

        # Append to cache along the T dim.
        k_full = torch.cat([k_cache, k_new], dim=2)  # [B, n, T_seen + 1, d]
        v_full = torch.cat([v_cache, v_new], dim=2)

        y = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=None, dropout_p=0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, 1, C)
        y = self.proj(y)
        # resid_dropout at p=0 (eval) is a no-op; skip for clarity at decode.
        return y, k_full, v_full


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


class TitansMAGBlock(nn.Module):
    """One transformer block with persistent prefix + NMM combined via MAG gate.

    Block flow:
      x_aug    = concat([persistent_mem, x], dim=T)
      y_attn   = attn(ln_1(x_aug), mask=block-structured causal)[:, N_p:, :]
      y_mem, s = nmm.forward_chunk(ln_nmm(x), nmm_state, doc_boundaries)
      o        = MAG-gate(y_attn, y_mem)                  # mode-dependent
      x        = x + o + mlp(ln_2(x + o))

    The mask is built block-structured:
      persistent <-> persistent : open
      persistent  -> real       : masked
      real        -> persistent : open
      real        -> real       : standard upper-triangular causal
                                  (plus banded-far-past mask when use_swa)

    NMM is fed only `ln_nmm(x)` (real tokens), not the persistent-augmented
    `x_aug`. Persistent tokens are input-independent; updating memory on
    them would add noise without semantic benefit.
    """

    def __init__(self, config):
        super().__init__()
        self.N_p = config.nmm_n_persistent
        self.finetune_mode = config.finetune_mode
        self.use_swa = config.use_swa
        self.swa_window = config.swa_window

        # Small init like GPT-2 wte; learned, no weight decay (routed in §4.1).
        self.persistent_mem = nn.Parameter(
            torch.randn(config.nmm_n_persistent, config.n_embd) * 0.02
        )

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd, config.n_head, config.dropout
        )

        # Separate from ln_1; NMM has its own pre-norm.
        self.ln_nmm = nn.LayerNorm(config.n_embd)
        self.nmm = NeuralMemoryModule(
            n_embd=config.n_embd,
            expansion=config.nmm_expansion,
            kernel_size=config.nmm_conv_kernel,
            spectral_norm=config.nmm_spectral_norm,
            finetune_mode=config.finetune_mode,
        )

        # MAG gates: gamma_mem always; gamma_attn only when training from scratch.
        # Creating gamma_attn unconditionally would leak unused params into the
        # state_dict, breaking finetune<->scratch checkpoint interchange.
        self.gamma_mem = nn.Parameter(torch.ones(config.n_embd))
        if not config.finetune_mode:
            self.gamma_attn = nn.Parameter(torch.ones(config.n_embd))

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config.n_embd, config.dropout)

    def _aug_mask(self, T: int, dtype: torch.dtype = None) -> torch.Tensor:
        """Build the [N_p+T, N_p+T] additive attention mask (0 attend, -inf block).

        dtype matches x.dtype to avoid an implicit cast inside
        scaled_dot_product_attention under autocast.
        """
        device = self.persistent_mem.device
        N = self.N_p + T
        mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
        mask[: self.N_p, : self.N_p] = 0
        mask[self.N_p :, : self.N_p] = 0
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        if self.use_swa:
            # Sliding Window Attention: mask positions >= swa_window steps in the past.
            # Persistent prefix stays fully visible per paper Fig. 3b.
            far_past = torch.tril(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype),
                diagonal=-self.swa_window,
            )
            causal = causal + far_past
        mask[self.N_p :, self.N_p :] = causal
        return mask

    def forward(self, x: torch.Tensor, nmm_state, doc_boundaries=None):
        B, T, _ = x.shape
        x_aug = cat([self.persistent_mem.expand(B, -1, -1), x], dim=1)
        y_attn = self.attn(
            self.ln_1(x_aug), mask=self._aug_mask(T, dtype=x.dtype)
        )[:, self.N_p :, :]

        y_mem, nmm_state = self.nmm.forward_chunk(
            self.ln_nmm(x), nmm_state, doc_boundaries
        )

        if self.finetune_mode:
            # Additive gate: at out_scale=0 -> y_mem=0 -> o = y_attn exactly.
            # Pretrained GPT-2 residual preserved at init.
            o = y_attn + F.silu(self.gamma_mem * y_mem) * y_attn
        else:
            # Paper's pure multiplicative gate (from-scratch only).
            o = F.silu(self.gamma_attn * y_attn) * F.silu(self.gamma_mem * y_mem)

        x = x + o
        x = x + self.mlp(self.ln_2(x))
        return x, nmm_state
