"""TitansMAGBlock: attention + NMM combined via a learned MAG gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from model.nmm import MultiHeadNMM, NeuralMemoryModule


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
        swa_window: int = None,
        n_persistent: int = 0,
    ) -> tuple:
        """Decode-path attention: project a single new token's Q/K/V,
        append the new K, V to the cache, run SDPA with Q against the
        full cached K, V.

        x_new: [B, 1, C] — the new token's pre-attention input (already
        ln_1-normalized by the caller).
        k_cache, v_cache: [B, n_head, T_seen, head_dim] — prior K, V.
        swa_window: when set, the new real token attends only to the most
          recent `swa_window` real positions (matches warm-up's banded
          mask via `_aug_mask`). When None, full causal — every prior
          position is allowed.
        n_persistent: number of persistent-prefix positions at the START
          of k_cache. Persistent positions are ALWAYS visible regardless
          of `swa_window` (paper Fig. 3b). Only consulted when SWA fires.

        Returns (y [B, 1, C], new_k_cache, new_v_cache).
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

        # SWA at decode: matches warm-up's `_aug_mask` semantics. The new
        # real token (at the LAST absolute position) attends to:
        #   - All persistent positions (0..n_persistent-1) — always open.
        #   - The most recent `swa_window` real positions only.
        # Without this branch, decode silently attends to every prior real
        # token regardless of swa_window — i.e., a model trained with SWA
        # would have learned a banded distribution but suddenly see the
        # full history at generation time.
        attn_mask = None
        if swa_window is not None:
            T_full = k_full.size(2)
            # New token is at absolute position T_full - 1 in the cache.
            # First REAL allowed position = max(n_persistent, T_full - swa_window).
            first_real_allowed = max(n_persistent, T_full - swa_window)
            # Build a length-T_full single-row additive mask: open on
            # [0, n_persistent) ∪ [first_real_allowed, T_full); -inf elsewhere.
            mask_row = torch.zeros(T_full, device=q.device, dtype=q.dtype)
            if first_real_allowed > n_persistent:
                mask_row[n_persistent:first_real_allowed] = float("-inf")
            # SDPA expects mask shape broadcastable to [B, n, 1, T_full].
            attn_mask = mask_row.view(1, 1, 1, T_full)

        y = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=attn_mask, dropout_p=0.0
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
        # Paper Eq. 28 says M(x̃) — feed persistent-augmented input to NMM.
        # Default False (our lucidrains-flavored choice): NMM sees only real
        # tokens. True = paper-strict.
        self.feed_persistent_to_nmm = config.feed_persistent_to_nmm

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
        # Build NMM — single-head (default) or multi-head wrapper.
        nmm_kwargs = dict(
            n_embd=config.n_embd,
            expansion=config.nmm_expansion,
            kernel_size=config.nmm_conv_kernel,
            spectral_norm=config.nmm_spectral_norm,
            finetune_mode=config.finetune_mode,
            retrieval_from_M_prev=config.retrieval_from_M_prev,
            state_dtype=config.nmm_state_dtype,
            grad_checkpoint=config.nmm_grad_checkpoint,
            grad_checkpoint_segment_len=config.nmm_grad_checkpoint_segment_len,
        )
        if config.nmm_n_heads > 1:
            self.nmm = MultiHeadNMM(n_heads=config.nmm_n_heads, **nmm_kwargs)
        else:
            self.nmm = NeuralMemoryModule(**nmm_kwargs)

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

        if self.feed_persistent_to_nmm:
            # Paper Eq. 28 strict: M(x̃). Feed the persistent-augmented input
            # through ln_nmm + NMM; slice the persistent prefix off the OUTPUT
            # so the residual stream only sees y_mem for real tokens.
            # doc_boundaries must also gain a False prefix (persistent
            # positions never trigger doc resets — they're input-independent
            # and identical across documents).
            if doc_boundaries is not None:
                db_aug = cat(
                    [torch.zeros(B, self.N_p, dtype=torch.bool, device=x.device),
                     doc_boundaries], dim=1,
                )
            else:
                db_aug = None
            y_mem_full, nmm_state = self.nmm.forward_chunk(
                self.ln_nmm(x_aug), nmm_state, db_aug,
            )
            y_mem = y_mem_full[:, self.N_p :, :]
        else:
            # Default: NMM sees only real tokens (lucidrains-flavored).
            y_mem, nmm_state = self.nmm.forward_chunk(
                self.ln_nmm(x), nmm_state, doc_boundaries,
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

    def init_decode_cache(self, x_prompt: torch.Tensor, nmm_state: tuple) -> tuple:
        """Seed the per-block decode caches from a warm-up prompt.

        Called once per block AFTER block.forward has already updated
        nmm_state on the prompt. Computes:
        - (k_cache, v_cache): K, V from attn over ln_1(x_aug_prompt),
          length N_p + T_prompt — includes the persistent prefix.
        - nmm_conv_buffer: dict for NMM step_with_conv (last k-1 Linear
          projections of ln_nmm(prompt)).

        x_prompt: [B, T, d] — the block's INPUT prompt (pre-block, not
        post-block). nmm_state is the post-warmup NMM state.
        Returns (k_cache, v_cache, nmm_conv_buffer).
        """
        B, T, _ = x_prompt.shape
        # KV cache: project K, V from ln_1(x_aug) where x_aug includes the
        # persistent prefix. Decode-time queries against this cache see the
        # persistent positions exactly the way warm-up's attention saw them.
        x_aug = cat([self.persistent_mem.expand(B, -1, -1), x_prompt], dim=1)
        x_aug_norm = self.ln_1(x_aug)
        k_cache, v_cache = self.attn.project_kv(x_aug_norm)

        # NMM conv buffer: last (k-1) Linear projections of ln_nmm input.
        # Under feed_persistent_to_nmm, the conv must have seen the persistent
        # prefix during warm-up — so we seed the buffer from ln_nmm(x_aug)
        # (matching what _forward_chunk_sequential saw). Without this branch
        # the decode-time conv would diverge from warm-up at exactly the
        # boundary between persistent prefix and the new decoded token.
        if self.feed_persistent_to_nmm:
            x_nmm_norm = self.ln_nmm(x_aug)
        else:
            x_nmm_norm = self.ln_nmm(x_prompt)
        nmm_conv_buffer = self.nmm.init_conv_buffer_from_prompt(x_nmm_norm)
        return k_cache, v_cache, nmm_conv_buffer

    def forward_step(
        self,
        x_new: torch.Tensor,
        nmm_state: tuple,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        nmm_conv_buffer: dict,
    ) -> tuple:
        """Single-token decode forward through one block.

        x_new: [B, 1, d] — embedded new token (wte+wpe).
        nmm_state: (M, S) from prior step.
        k_cache, v_cache: [B, n_head, T_seen, head_dim] — prior K, V.
        nmm_conv_buffer: dict from prior step.

        Returns (x_out [B, 1, d], new_nmm_state, new_k_cache, new_v_cache,
        new_nmm_conv_buffer).
        """
        # Attention via KV cache. ln_1 on the new token; no x_aug concat
        # (persistent prefix is already in the cache). Forward the SWA
        # window + persistent-prefix length so SWA semantics survive at
        # decode time (matches the warm-up path's `_aug_mask`).
        x_norm_attn = self.ln_1(x_new)
        y_attn, new_k_cache, new_v_cache = self.attn.forward_with_kv_cache(
            x_norm_attn, k_cache, v_cache,
            swa_window=self.swa_window if self.use_swa else None,
            n_persistent=self.N_p,
        )

        # NMM via step_with_conv: one update per token, full conv context.
        x_norm_nmm = self.ln_nmm(x_new).squeeze(1)  # [B, d]
        y_mem_t, new_nmm_state, new_nmm_conv_buffer = self.nmm.step_with_conv(
            x_norm_nmm, nmm_state, nmm_conv_buffer
        )
        y_mem = y_mem_t.unsqueeze(1)  # [B, 1, d]

        if self.finetune_mode:
            o = y_attn + F.silu(self.gamma_mem * y_mem) * y_attn
        else:
            o = F.silu(self.gamma_attn * y_attn) * F.silu(self.gamma_mem * y_mem)

        x = x_new + o
        x = x + self.mlp(self.ln_2(x))
        return x, new_nmm_state, new_k_cache, new_v_cache, new_nmm_conv_buffer
