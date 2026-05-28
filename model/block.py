"""TitansMAGBlock: attention + NMM combined via a learned MAG gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from model.nmm import MultiHeadNMM, NeuralMemoryModule


def _build_aug_mask(
    N_p: int,
    T_real: int,
    use_swa: bool,
    swa_window: int,
    device,
    dtype,
) -> torch.Tensor:
    """Block-structured additive attention mask `[N_p+T_real, N_p+T_real]`.

    Used by both `TitansMAGBlock` (always) and `PlainGPT2Block` (only in
    `persistent_prefix_mode="model_wide"` — see `PlainGPT2Block.forward`).

    Block layout:
      - persistent ↔ persistent : 0 (bidirectional among themselves)
      - persistent → real       : −∞ (persistent can't see real)
      - real → persistent       : 0 (real always sees persistent — Fig 3b)
      - real → real             : upper-triangular causal, optionally
                                  banded `swa_window` positions back if
                                  `use_swa` is on. Persistent prefix stays
                                  fully visible under SWA per paper Fig 3b.

    `dtype` matches the input's dtype to avoid an implicit cast inside
    SDPA under autocast.
    """
    N = N_p + T_real
    mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
    if N_p > 0:
        mask[:N_p, :N_p] = 0
        mask[N_p:, :N_p] = 0
    causal = torch.triu(
        torch.full((T_real, T_real), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    if use_swa:
        far_past = torch.tril(
            torch.full((T_real, T_real), float("-inf"), device=device, dtype=dtype),
            diagonal=-swa_window,
        )
        causal = causal + far_past
    mask[N_p:, N_p:] = causal
    return mask


# int8 KV cache (decode-time only).
#
# Each cached K or V tensor at a transformer block has shape
# [B, n_head, T_seen, head_dim]. Per-head, per-token symmetric int8
# quantization compresses these to ~half the bf16 footprint (int8
# values + per-(B,h,t) fp16 scale).
#
# Why per-(B, h, t) scale and not per-tensor: KV magnitudes vary
# strongly across heads (some heads carry larger activations than
# others) and across token positions (early vs late positions in long
# sequences). Per-tensor would lose precision; per-(B,h,t) keeps
# resolution per-row. Per-(B,h,t,d) would be max-precision but the
# scale tensor would dominate the memory savings.
#
# Decode-only: training-time attention uses dense bf16 K, V. The cache
# is only built at `prepare_decode` and only consumed at `forward_step`,
# both in eval mode.


def _quantize_int8_kv(x: torch.Tensor):
    """Per-(B, head, T) symmetric int8 quantization.

    Args:
        x: [B, n_head, T, head_dim] tensor (any float dtype).

    Returns:
        (int8 [B, n_head, T, head_dim], scale [B, n_head, T, 1] fp16)
        where dequantized = int8.float() * scale.float().
    """
    # Operate on a detached fp32 copy — KV cache is built in no_grad
    # context (eval mode + prepare_decode), so detach is free.
    x_f = x.detach().float()
    max_abs = x_f.abs().amax(dim=-1, keepdim=True)        # [B, n_head, T, 1]
    scale = (max_abs / 127.0).clamp(min=1e-8)
    q = (x_f / scale).round().clamp(min=-128, max=127).to(torch.int8)
    return q, scale.to(torch.float16)


def _dequantize_int8_kv(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Inverse of `_quantize_int8_kv`. Casts to `dtype` for the
    downstream SDPA call (matches the rest of the decode-path dtype)."""
    return (q.float() * scale.float()).to(dtype)


class KVCacheInt8:
    """quantized container for one (K or V) decode-time cache.

    The class wraps `(int8_tensor, scale_tensor)` and exposes:
      - `dense(dtype)`: full dequantized tensor, ready for SDPA.
      - `append(new_fp_tensor)`: quantize a new token's K/V and append
        along the T (time) dim, returning a new container.

    Using a class (not a bare tuple) lets `isinstance(cache, KVCacheInt8)`
    branch cleanly in `forward_with_kv_cache`; existing callers that
    receive plain bf16 tensors keep working unchanged.

    Memory: at gpt2_small (n_head=12, head_dim=64), per cached token:
      bf16:  12 × 64 × 2 = 1.5 KB
      int8:  12 × 64 × 1 + 12 × 1 × 2 = 0.79 KB  (~2× smaller)
    """
    __slots__ = ("int8", "scale")

    def __init__(self, int8_tensor: torch.Tensor, scale_tensor: torch.Tensor):
        self.int8 = int8_tensor
        self.scale = scale_tensor

    def dense(self, dtype=torch.bfloat16) -> torch.Tensor:
        return _dequantize_int8_kv(self.int8, self.scale, dtype=dtype)

    def append(self, new_fp_tensor: torch.Tensor) -> "KVCacheInt8":
        q_new, s_new = _quantize_int8_kv(new_fp_tensor)
        return KVCacheInt8(
            torch.cat([self.int8, q_new], dim=2),
            torch.cat([self.scale, s_new], dim=2),
        )

    @property
    def shape(self):
        return self.int8.shape

    @property
    def device(self):
        return self.int8.device

    @classmethod
    def from_dense(cls, x: torch.Tensor) -> "KVCacheInt8":
        q, s = _quantize_int8_kv(x)
        return cls(q, s)


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

    def project_kv(self, x: torch.Tensor, int8_kv_cache: bool = False) -> tuple:
        """Project x into (K, V) tensors shaped [B, n_head, T, head_dim].

        Used to seed the KV cache during warm-up: caller runs ln_1(x_aug),
        passes that here, gets the K, V that the attention would have used,
        stores them. No attention is computed — that runs separately during
        the warm-up forward.

        When `int8_kv_cache=True`, the returned K, V are `KVCacheInt8`
        objects instead of dense tensors. The dense tensor would be ~2×
        the size of the int8 representation, which matters most for long
        prompts at gpt2_small (~1.5 KB / token at n_head=12, head_dim=64).
        """
        B, T, C = x.shape
        n, d = self.n_head, self.head_dim
        k = self.k_proj(x).view(B, T, n, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n, d).transpose(1, 2)
        if int8_kv_cache:
            return KVCacheInt8.from_dense(k), KVCacheInt8.from_dense(v)
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

        # int8 KV cache branch: append new K, V in int8 form; dequant
        # to bf16 for SDPA. The returned cache stays in int8 form to keep
        # the memory footprint small across decode steps.
        if isinstance(k_cache, KVCacheInt8):
            new_k_cache = k_cache.append(k_new)
            new_v_cache = v_cache.append(v_new)
            # Dequantize ONLY for the SDPA call. Q is fp/bf16 (not cached);
            # picking bf16 for the dense view matches the decode-path
            # dtype contract.
            sdpa_dtype = q.dtype
            k_full = new_k_cache.dense(dtype=sdpa_dtype)
            v_full = new_v_cache.dense(dtype=sdpa_dtype)
        else:
            # Dense (bf16) cache — original behavior.
            k_full = torch.cat([k_cache, k_new], dim=2)  # [B, n, T_seen + 1, d]
            v_full = torch.cat([v_cache, v_new], dim=2)
            new_k_cache = k_full
            new_v_cache = v_full

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
        return y, new_k_cache, new_v_cache


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


class PlainGPT2Block(nn.Module):
    """Standard GPT-2 transformer block — attn + MLP, no NMM, no persistent
    prefix, no MAG gate.

    Used when `config.nmm_layer_indices` is set and the current block's
    position is NOT in that list. Same forward signature as
    `TitansMAGBlock` so the model's per-block loop is uniform:
        x, state = block(x, state, doc_boundaries)
    State is passed through unchanged (None for plain blocks); the
    `nmm_states` list across the model has None entries at plain-block
    positions.

    Why a separate class rather than a flag on TitansMAGBlock: the latter
    would carry dead `persistent_mem`, `ln_nmm`, `nmm`, and `gamma_mem`
    parameters in the state_dict, breaking the simple invariant that
    parameter counts match an equivalent subset config.
    """

    def __init__(self, config):
        super().__init__()
        self.use_swa = config.use_swa
        self.swa_window = config.swa_window
        # Plain blocks must honor `persistent_prefix_mode` because in
        # `model_wide` mode the persistent prefix is prepended at the model
        # input and arrives at EVERY block — including this one. Without
        # block-structured masking, plain blocks would apply standard
        # upper-triangular causal mask over the augmented sequence, which:
        #   - makes persistent-to-persistent attention causal instead of
        #     bidirectional (paper Fig 3b wants bidirectional);
        #   - under SWA, can mask out persistent positions when they fall
        #     outside the sliding window (paper Fig 3b: persistent always
        #     visible).
        # In `per_block` mode the plain block sees only real tokens (the
        # NMM blocks prepend/slice locally) and standard causal works.
        self.persistent_prefix_mode = config.persistent_prefix_mode
        self.N_p = config.nmm_n_persistent
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd, config.n_head, config.dropout
        )
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config.n_embd, config.dropout)

    def _causal_mask(self, T: int, device, dtype):
        """Standard causal mask (with optional banded SWA), no persistent
        prefix. Used in `per_block` mode where plain blocks see only real
        tokens."""
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        if self.use_swa:
            far_past = torch.tril(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype),
                diagonal=-self.swa_window,
            )
            mask = mask + far_past
        return mask

    def _aug_mask(self, T_real: int, device, dtype):
        """Block-structured mask for the `model_wide` path — delegates to
        the module-level `_build_aug_mask`. See that function's docstring
        for the block layout."""
        return _build_aug_mask(
            self.N_p, T_real, self.use_swa, self.swa_window, device, dtype,
        )

    def forward(self, x: torch.Tensor, nmm_state=None, doc_boundaries=None):
        # nmm_state and doc_boundaries are accepted for signature uniformity
        # with TitansMAGBlock; both are ignored. State passes through.
        T_in = x.size(1)
        if self.persistent_prefix_mode == "model_wide" and self.N_p > 0:
            # In `model_wide` mode the model has already prepended the
            # persistent prefix; `T_in == N_p + T_real`. Build the block-
            # structured mask so persistent positions stay bidirectional
            # and always-visible to real positions.
            T_real = T_in - self.N_p
            mask = self._aug_mask(T_real, x.device, x.dtype)
        else:
            mask = self._causal_mask(T_in, x.device, x.dtype)
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x, nmm_state  # state pass-through (typically None)

    def init_decode_cache(
        self, x_prompt: torch.Tensor, nmm_state=None, int8_kv_cache: bool = False,
    ) -> tuple:
        """Decode-cache seed for the plain block: just the KV cache. No NMM
        state involvement. Signature mirrors `TitansMAGBlock.init_decode_cache`
        so the model's per-block decode loop is uniform.

        `int8_kv_cache=True` returns `KVCacheInt8` containers for
        the K and V caches; the rest of the block API is dtype-agnostic.
        """
        B, T, _ = x_prompt.shape
        x_norm = self.ln_1(x_prompt)
        k_cache, v_cache = self.attn.project_kv(x_norm, int8_kv_cache=int8_kv_cache)
        return k_cache, v_cache

    def forward_step(
        self,
        x_new: torch.Tensor,
        nmm_state,  # ignored (None)
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple:
        """Single-token decode through a plain block."""
        x_norm = self.ln_1(x_new)
        # In `model_wide` mode the KV cache already contains the persistent
        # prefix at positions [0, N_p) (captured during warmup against the
        # model-augmented input). Pass `n_persistent=self.N_p` so the SWA
        # mask keeps those positions always-visible. In `per_block` mode
        # the cache contains real tokens only — `n_persistent=0`.
        n_persistent = (
            self.N_p if self.persistent_prefix_mode == "model_wide" else 0
        )
        y_attn, new_k_cache, new_v_cache = self.attn.forward_with_kv_cache(
            x_norm, k_cache, v_cache,
            swa_window=self.swa_window if self.use_swa else None,
            n_persistent=n_persistent,
        )
        x = x_new + y_attn
        x = x + self.mlp(self.ln_2(x))
        return x, None, new_k_cache, new_v_cache


class TitansMAGBlock(nn.Module):
    """One transformer block with persistent prefix + NMM combined via MAG gate.

    Block flow (paper-strict default — `feed_persistent_to_nmm=True`):
      x_aug      = concat([persistent_mem, x], dim=T)
      y_attn     = attn(ln_1(x_aug), mask=block-structured causal)[:, N_p:, :]
      y_mem_aug, s = nmm.forward_chunk(ln_nmm(x_aug), state, db_aug)
      y_mem      = y_mem_aug[:, N_p:, :]
      o          = MAG-gate(y_attn, y_mem)              # mode-dependent
      x          = x + o + mlp(ln_2(x + o))

    With `feed_persistent_to_nmm=False` (lucidrains-flavored) the NMM is fed
    only `ln_nmm(x)` (real tokens) and `doc_boundaries` is passed verbatim;
    no output slice. This is a slight memory win but updates memory only on
    real tokens.

    The mask is built block-structured:
      persistent <-> persistent : open
      persistent  -> real       : masked
      real        -> persistent : open
      real        -> real       : standard upper-triangular causal
                                  (plus banded-far-past mask when use_swa)
    """

    def __init__(self, config):
        super().__init__()
        self.N_p = config.nmm_n_persistent
        self.finetune_mode = config.finetune_mode
        self.use_swa = config.use_swa
        self.swa_window = config.swa_window
        # Paper Eq. 28 says M(x̃) — feed persistent-augmented input to NMM.
        # Config default True = paper-strict (feed ln_nmm(x_aug) to NMM, slice
        # the persistent prefix off the output, augment doc_boundaries with a
        # False prefix). Flip to False for the lucidrains-flavored "real
        # tokens only" path: NMM sees ln_nmm(x) and never updates on
        # persistent positions.
        self.feed_persistent_to_nmm = config.feed_persistent_to_nmm

        # persistent_prefix_mode controls whether THIS block owns its own
        # learned prefix (`per_block`) or whether the prefix is model-wide
        # and arrives pre-prepended in the input (`model_wide`, default,
        # paper Eq. 19). In `model_wide` mode the block does no prepend /
        # slice — its forward receives [B, N_p+T, d] and produces the same
        # shape unchanged.
        self.persistent_prefix_mode = config.persistent_prefix_mode
        if self.persistent_prefix_mode == "per_block":
            # Small init like GPT-2 wte; learned, no weight decay (routed
            # in §4.1).
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
            low_rank=config.nmm_low_rank,
            softclamp_max=config.nmm_softclamp_max,
            block_size=config.nmm_block_size,
            per_token_ns5=config.nmm_per_token_ns5,
            detach_state_between_blocks=config.nmm_detach_state_between_blocks,
            lookahead_value=config.nmm_lookahead_value,
            per_param_lr_modulation=config.nmm_per_param_lr_modulation,
            momentum_order=config.nmm_momentum_order,
            ns5_steps=config.nmm_ns5_steps,
            use_gram_ns5=config.nmm_use_gram_ns5,
            use_cans=config.nmm_use_cans,
        )
        if config.nmm_n_heads > 1:
            self.nmm = MultiHeadNMM(
                n_heads=config.nmm_n_heads,
                per_head_learned_params=config.nmm_per_head_learned_params,
                **nmm_kwargs,
            )
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
        """Build the `[N_p+T, N_p+T]` additive attention mask — delegates
        to the module-level `_build_aug_mask`. See that function's
        docstring for the block layout.

        `T` here is the real-token count; the persistent rows/cols are
        added internally based on `self.N_p`. `dtype` matches x.dtype to
        avoid an implicit cast inside SDPA under autocast.

        Device is picked off `ln_1.weight` because `persistent_mem` is
        only present on this block in `per_block` mode (in `model_wide`
        mode the prefix lives on the model). `ln_1.weight` is always
        present and always lives on the block's device.
        """
        return _build_aug_mask(
            self.N_p, T, self.use_swa, self.swa_window,
            self.ln_1.weight.device, dtype,
        )

    def forward(self, x: torch.Tensor, nmm_state, doc_boundaries=None):
        B, T_in, _ = x.shape

        if self.persistent_prefix_mode == "per_block":
            # Per-block: prepend this block's persistent_mem; slice persistent
            # positions off the attention output before the residual / MAG /
            # MLP. T_real == T_in.
            x_aug = cat([self.persistent_mem.expand(B, -1, -1), x], dim=1)
            T_real = T_in
            mode_model_wide = False
        else:
            # Model-wide: x already includes the persistent prefix at
            # positions [0, N_p). The block doesn't prepend or slice — the
            # persistent positions stay in the residual through this block
            # and accumulate information from real tokens.
            x_aug = x
            T_real = T_in - self.N_p
            mode_model_wide = True

        y_attn_aug = self.attn(
            self.ln_1(x_aug), mask=self._aug_mask(T_real, dtype=x.dtype),
        )

        # NMM input + doc_boundaries handling per (feed_persistent_to_nmm,
        # persistent_prefix_mode).
        if self.feed_persistent_to_nmm:
            # NMM sees the persistent-augmented input (paper Eq. 28).
            if mode_model_wide:
                # doc_boundaries was augmented at the model level.
                db_for_nmm = doc_boundaries
            else:
                # Per-block: augment locally with a False prefix.
                if doc_boundaries is not None:
                    db_for_nmm = cat(
                        [torch.zeros(B, self.N_p, dtype=torch.bool, device=x.device),
                         doc_boundaries], dim=1,
                    )
                else:
                    db_for_nmm = None
            y_mem_aug, nmm_state = self.nmm.forward_chunk(
                self.ln_nmm(x_aug), nmm_state, db_for_nmm,
            )
        else:
            # NMM sees only real tokens.
            if mode_model_wide:
                x_real = x_aug[:, self.N_p:, :]
                db_real = (
                    doc_boundaries[:, self.N_p:]
                    if doc_boundaries is not None else None
                )
            else:
                x_real = x
                db_real = doc_boundaries
            y_mem_real, nmm_state = self.nmm.forward_chunk(
                self.ln_nmm(x_real), nmm_state, db_real,
            )
            # Pad to augmented shape so MAG composes; persistent positions
            # get zero memory contribution (consistent with "NMM sees only
            # real tokens").
            if self.N_p > 0:
                zero_persist = torch.zeros(
                    B, self.N_p, y_mem_real.shape[-1],
                    device=x.device, dtype=y_mem_real.dtype,
                )
                y_mem_aug = cat([zero_persist, y_mem_real], dim=1)
            else:
                y_mem_aug = y_mem_real

        if self.finetune_mode:
            # Additive gate: at out_scale=0 -> y_mem=0 -> o = y_attn exactly.
            # Pretrained GPT-2 residual preserved at init.
            o_aug = y_attn_aug + F.silu(self.gamma_mem * y_mem_aug) * y_attn_aug
        else:
            # Paper's pure multiplicative gate (from-scratch only).
            o_aug = F.silu(self.gamma_attn * y_attn_aug) * F.silu(self.gamma_mem * y_mem_aug)

        if mode_model_wide:
            # MAG and MLP apply over the full augmented sequence; persistent
            # positions evolve through the residual.
            x_out = x_aug + o_aug
            x_out = x_out + self.mlp(self.ln_2(x_out))
        else:
            # Slice persistent positions off before the residual on x (real
            # tokens). Equivalent to the previous "slice y_attn / y_mem
            # early and combine on real-token shape".
            o = o_aug[:, self.N_p:, :]
            x_out = x + o
            x_out = x_out + self.mlp(self.ln_2(x_out))
        return x_out, nmm_state

    def init_decode_cache(
        self, x_prompt: torch.Tensor, nmm_state: tuple, int8_kv_cache: bool = False,
    ) -> tuple:
        """Seed the per-block KV cache from a warm-up prompt.

        Called once per block BEFORE block.forward runs on the prompt
        (caller's loop captures KV cache from the input then runs the
        block, which mutates nmm_state). Returns (k_cache, v_cache).

        Item 6: the NMM conv buffer is no longer captured here — it's
        part of `nmm_state` and gets rolled forward naturally by
        `block.forward` / `nmm.forward_chunk` during warm-up.

        x_prompt: [B, T, d] — the block's INPUT prompt (pre-block).
        Returns (k_cache, v_cache).
        """
        B, T_in, _ = x_prompt.shape
        # In per_block mode the prompt arrives as real tokens; the block
        # prepends its own persistent_mem here. In model_wide mode the
        # prompt is already augmented at the model level.
        if self.persistent_prefix_mode == "per_block":
            x_aug = cat([self.persistent_mem.expand(B, -1, -1), x_prompt], dim=1)
        else:
            x_aug = x_prompt
        x_aug_norm = self.ln_1(x_aug)
        k_cache, v_cache = self.attn.project_kv(x_aug_norm, int8_kv_cache=int8_kv_cache)
        return k_cache, v_cache

    def forward_step(
        self,
        x_new: torch.Tensor,
        nmm_state: tuple,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple:
        """Single-token decode forward through one block.

        Item 6: the NMM conv buffer is part of `nmm_state` now; no separate
        `nmm_conv_buffer` argument. The signature shrinks correspondingly.

        x_new: [B, 1, d] — embedded new token (wte+wpe).
        nmm_state: (M, S, conv_buf) from prior step.
        k_cache, v_cache: [B, n_head, T_seen, head_dim] — prior K, V.

        Returns (x_out [B, 1, d], new_nmm_state, new_k_cache, new_v_cache).
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
        # conv_buf inside nmm_state rolls automatically.
        x_norm_nmm = self.ln_nmm(x_new).squeeze(1)  # [B, d]
        y_mem_t, new_nmm_state = self.nmm.step_with_conv(x_norm_nmm, nmm_state)
        y_mem = y_mem_t.unsqueeze(1)  # [B, 1, d]

        if self.finetune_mode:
            o = y_attn + F.silu(self.gamma_mem * y_mem) * y_attn
        else:
            o = F.silu(self.gamma_attn * y_attn) * F.silu(self.gamma_mem * y_mem)

        x = x_new + o
        x = x + self.mlp(self.ln_2(x))
        return x, new_nmm_state, new_k_cache, new_v_cache
