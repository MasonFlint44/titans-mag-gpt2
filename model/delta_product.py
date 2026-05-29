"""DeltaProduct fast-weight memory — TPTT-aligned formulation.

Replaces the NMM's surprise-driven inner-loop gradient update with the
explicit delta rule (Yang et al., NeurIPS 2024 / arxiv 2406.06484) of
order N (Siems et al., ICLR 2025 / arxiv 2502.10297). The specific
projection / gating / normalization conventions follow TPTT (Furfaro
2025, arxiv 2506.17671), which is the published recipe for adapting
*pretrained* transformers — exactly our setting.

TPTT-specific design (see model/delta_product.py for the original
formulation, since-deleted):

  Projections (from TPTT eq. 4):
    q_normed = L2_normalize(SiLU(q_raw))     — per-head, per-token
    k_normed = L2_normalize(SiLU(k_raw))
    v_scaled = v_raw / sqrt(head_dim)        — just attention-scaling

  β gating (from TPTT eq. 5, default `beta_gate="k"`):
    β = sigmoid(CausalAvgPool_3(k_raw))      — vector, per-component

    CausalAvgPool is a fixed-weight kernel-3 moving average — NOT
    learnable. β is computed from the RAW K projection (before
    SiLU+L2) so the gating sees the un-normalized signal.

  Per-token update with vector β:
    M ← M + (β⊙v) k^T − M (β⊙k) k^T

  Output (from TPTT eq. 6):
    y_out = RMSNorm(y_raw) · out_proj   — manual RMSNorm + Linear

  We additionally keep a per-channel `out_scale` gain after `out_proj`
  to preserve the gate-ramp logic in cli/train.py and the finetune-mode
  "y = 0 at step 0" invariant.

Math derivation for the chunkwise WY solve with vector β:

  u_t = (β_t ⊙ v_t) − M_{t-1} (β_t ⊙ k_t)
  G[t, j] = (β_t ⊙ k_t) · k_j        for j < t
  R[t]    = (β_t ⊙ v_t) − M_in (β_t ⊙ k_t)
  (I + G) U = R                       (unit-lower-tri solve)
  M_out = M_in + Uᵀ K                 (K = un-gated keys)
  y[t]  = M_in q_t + Σ_{s ≤ (t+1)N − 1} (q_t · k_s) u_s

Multi-head fusion: `n_heads > 1` runs all heads in parallel via batched
einsum projections + WY solve with batch dim = B·n_heads.

State: a 1-tuple `(M,)` with `M ∈ R^(B, n_heads, head_dim, head_dim)`.

Interface mirrors `NeuralMemoryModule` so `TitansMAGBlock` can pick
between the two via `config.memory_type` without other changes:
    init_state(B, device) -> state
    forward_chunk(x_chunk, state_in, doc_boundaries) -> (y_mem, new_state)
    step_with_conv(x_t, state) -> (y_t, new_state)
"""

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=128)
def _chunkwise_aux_tensors(device, T: int, N: int, dtype):
    """Static auxiliary tensors used by every WY solve.

    Returns `(mask_lt, I_TN, real_mask)`:
      - `mask_lt`: strict-lower-tri bool mask of shape [TN, TN] used to
        keep only the j<s entries of G.
      - `I_TN`: identity of shape [TN, TN] in `dtype` for the (I + G)
        triangular system.
      - `real_mask`: [T, TN] mask of `(s_idx < (t_idx + 1) * N)` in
        `dtype`, used to scope each real token's read to its own
        write-window.

    Caching by `(device, T, N, dtype)` because they're deterministic
    functions of those args and re-allocating per layer per forward
    costs ~150 MB/s of allocator pressure at training scale.
    """
    TN = T * N
    mask_lt = torch.tril(
        torch.ones(TN, TN, device=device, dtype=torch.bool),
        diagonal=-1,
    )
    I_TN = torch.eye(TN, device=device, dtype=dtype)
    s_idx = torch.arange(TN, device=device).unsqueeze(0)
    t_idx = torch.arange(T, device=device).unsqueeze(1)
    real_mask = (s_idx < (t_idx + 1) * N).to(dtype)
    return mask_lt, I_TN, real_mask


def _causal_avg_pool_3(x: torch.Tensor) -> torch.Tensor:
    """Causal 3-token moving average along the T (dim=1) axis.

    Replicate-pads the leading 2 positions (so the boundary entries
    see their own value rather than zeros). Matches TPTT's
    `CausalAvgPool1d` semantics (kernel size 3, fixed weights
    [1/3, 1/3, 1/3], replicate padding). Implemented as direct shifts +
    broadcast-add to avoid a Conv1d kernel launch (which would be
    pointless for a fixed average).

    Args:
        x: [B, T, ...] — input tensor; T is the time axis.

    Returns:
        Tensor of the same shape: `y[t] = (x[t] + x[t-1] + x[t-2]) / 3`
        with `x[t-k]` replicated to `x[0]` for any `t-k < 0`.
    """
    # x[:, 0] replicated as the first two "past" positions.
    first = x[:, :1]
    x_pad = torch.cat([first, first, x], dim=1)  # [B, T+2, ...]
    return (x_pad[:, :-2] + x_pad[:, 1:-1] + x_pad[:, 2:]) / 3.0


class DeltaProductMemory(nn.Module):
    """Multi-head DeltaProduct memory, order >= 1, TPTT-aligned.

    `n_heads=1` is the single-head case. `n_heads>1` runs n_heads parallel
    head_dim-sized memory modules, fused via batched einsum projections
    and a batched WY solve.

    Args:
        n_embd: input/output dim.
        n_heads: number of parallel memory heads. Must divide n_embd.
            Default 1 = single-head with `head_dim = n_embd`.
        order: number of delta sub-steps applied per token (1 = DeltaNet,
            2 = matches Titans expressivity per TPTT). Default 2.
        finetune_mode: when True, `out_scale` initializes to zero so that
            y_mem = 0 at step 0 and the pretrained backbone's residual is
            preserved exactly until the gate ramp / optimizer takes over.
        block_size: forward-path selector. `block_size=1` runs the
            sequential per-token recurrence (reference correctness path,
            slow but bit-exact; useful as a baseline and as the decode
            path). `block_size>1` routes through the closed-form
            chunkwise WY parallel path (training-time speed path, bit-
            equivalent to sequential, one triangular solve per document
            segment).
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int = 1,
        order: int = 2,
        finetune_mode: bool = True,
        block_size: int = 1,
    ):
        super().__init__()
        if order < 1:
            raise ValueError(f"order must be >= 1 (got {order})")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1 (got {n_heads})")
        if n_embd % n_heads != 0:
            head_dim = n_embd // n_heads
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads}); "
                f"head_dim would be {head_dim} but {n_heads} * {head_dim} = "
                f"{n_heads * head_dim}, not {n_embd}."
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1 (got {block_size})")

        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = self.n_embd // self.n_heads
        self.order = int(order)
        self.block_size = int(block_size)
        self.finetune_mode = bool(finetune_mode)
        # V scaling factor (1/√head_dim) per TPTT's `prepare_attention_input`.
        self._v_scale = 1.0 / math.sqrt(self.head_dim)

        H, hd = self.n_heads, self.head_dim

        # Stacked per-head projection weights. One Q (read query, single
        # projection — the read happens after all N writes), N K and N V
        # for the N delta sub-steps. β is NOT learnable in TPTT — it's
        # computed from K_raw via fixed CausalAvgPool + sigmoid.
        self.q_proj_weight = nn.Parameter(torch.empty(H, hd, hd))
        self.k_proj_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, hd, hd)) for _ in range(order)]
        )
        self.v_proj_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, hd, hd)) for _ in range(order)]
        )

        # Output projection — applied to the merged-head output after
        # the RMSNorm, matching TPTT's `merge_head_output`. No bias.
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Per-channel output gain. Zero-init under finetune_mode so the
        # pretrained backbone's residual is preserved at step 0. Survives
        # cli/train.py's gate-ramp logic by name (`*.out_scale`).
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(H, hd))
        else:
            self.out_scale = nn.Parameter(torch.ones(H, hd))

        self._init_weights()

    def _init_weights(self) -> None:
        """Match PyTorch Linear's default kaiming init for each per-head
        slice, plus the standard Linear init on `out_proj`."""
        hd = self.head_dim
        bound = 1.0 / math.sqrt(hd)
        with torch.no_grad():
            nn.init.uniform_(self.q_proj_weight, -bound, bound)
            for i in range(self.order):
                nn.init.uniform_(self.k_proj_weights[i], -bound, bound)
                nn.init.uniform_(self.v_proj_weights[i], -bound, bound)
        # out_proj uses PyTorch's Linear default (kaiming_uniform_) on
        # construction — no override needed.

    # ------------------------------------------------------------------
    # State + projections
    # ------------------------------------------------------------------

    def init_state(self, B: int, device) -> tuple:
        """Per-sample-batched zero-initialized M plus per-order rolling
        buffers of the last 2 raw K values.

        Returns a 2-tuple `(M, k_raw_history)`:
          - `M` shape `[B, n_heads, head_dim, head_dim]` — recurrent state.
          - `k_raw_history` list of `order` tensors, each shape
            `[B, 2, n_heads, head_dim]` — the last 2 raw K projections
            for each sub-step, threaded across forward_chunk calls so the
            CausalAvgPool sees continuous context. Zero-initialized at
            fresh state, which is equivalent to "no prior tokens" (the
            pool's first-position replicate-pad reproduces from
            x[:, 0] either way).
        """
        M = torch.zeros(
            B, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float32,
        )
        k_buf = [
            torch.zeros(
                B, 2, self.n_heads, self.head_dim,
                device=device, dtype=torch.float32,
            )
            for _ in range(self.order)
        ]
        return (M, k_buf)

    def _project_kvb(self, x_chunk: torch.Tensor, k_buf_in: list):
        """Batched per-head projections for the whole chunk.

        x_chunk: [B, T, n_embd]
        k_buf_in: list[order] of [B, 2, H, hd] — prior-call raw K values
            for continuous CausalAvgPool context.

        Returns (q, ks, vs, bs, k_buf_out):
            q  [B, T, H, hd] — SiLU + L2-normalized
            ks list[order] of [B, T, H, hd] — SiLU + L2-normalized
            vs list[order] of [B, T, H, hd] — scaled by 1/√head_dim
            bs list[order] of [B, T, H, hd] — vector β = σ(CausalAvgPool(k_raw_extended))
            k_buf_out list[order] of [B, 2, H, hd] — last 2 raw K values
                of the combined (buffer + chunk) sequence, ready for the
                next call.

        β is computed from the RAW K projection (before SiLU+L2) so the
        gating depends on the un-normalized signal, matching TPTT's
        `compute_gate(k_raw, v_raw)` placement before SiLU+L2 normalize.

        The CausalAvgPool runs over the BUFFER-EXTENDED sequence
        `[k_buf_in, k_raw]` so the leading 2 positions of the chunk see
        the prior call's last 2 raw K values, not replicate-padded zeros.
        Without this, splitting a continuous stream into multiple
        forward_chunk calls would produce different β at the chunk head
        than a single-shot forward — see test_state_continuity_split_chunk_matches_full.
        """
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim
        x_h = x_chunk.view(B, T, H, hd)

        # Q: project, then SiLU + L2 normalize.
        q_raw = torch.einsum("bthd,hde->bthe", x_h, self.q_proj_weight)
        q = F.normalize(F.silu(q_raw), p=2, dim=-1, eps=1e-6)

        ks, vs, bs = [], [], []
        k_buf_out = []
        for i in range(self.order):
            # K_raw: project x; concat with prior buffer for continuous
            # pool context across forward_chunk boundaries.
            k_raw = torch.einsum("bthd,hde->bthe", x_h, self.k_proj_weights[i])
            k_raw_ext = torch.cat([k_buf_in[i], k_raw], dim=1)  # [B, 2+T, H, hd]

            # β = σ(CausalAvgPool(k_raw_ext))[buffer_len:]  -> matches the
            # current chunk's T positions exactly.
            beta_full = torch.sigmoid(_causal_avg_pool_3(k_raw_ext))
            beta_i = beta_full[:, 2:]  # [B, T, H, hd]

            # k_normed for the WY math (computed from the current chunk's
            # k_raw, NOT the buffer-extended version — the buffer is
            # purely for pool continuity).
            k_normed = F.normalize(F.silu(k_raw), p=2, dim=-1, eps=1e-6)
            ks.append(k_normed)
            bs.append(beta_i)

            # New buffer: last 2 raw K values of the combined sequence.
            # For T >= 2 this is just k_raw[:, -2:]; for T < 2 the cat
            # form handles the short-chunk case correctly.
            k_buf_out.append(k_raw_ext[:, -2:])

            # V: project, scale by 1/√head_dim (no SiLU, no L2 normalize).
            v_i = (
                torch.einsum("bthd,hde->bthe", x_h, self.v_proj_weights[i])
                * self._v_scale
            )
            vs.append(v_i)

        return q, ks, vs, bs, k_buf_out

    def _apply_out_norm_and_proj(
        self, y_BTHd: torch.Tensor, dtype,
    ) -> torch.Tensor:
        """TPTT-style output: merge heads, manual RMSNorm, Linear out_proj,
        per-channel `out_scale` gain.

        The per-channel `out_scale` is our addition to TPTT — it preserves
        the gate-ramp logic in cli/train.py and the finetune-mode "y = 0
        at step 0" invariant. TPTT relies on LoRA's near-zero init for
        the equivalent guarantee.
        """
        B, T = y_BTHd.shape[:2]
        H, hd = self.n_heads, self.head_dim

        # Merge heads to [B, T, n_embd].
        y = y_BTHd.reshape(B, T, H * hd)

        # Manual RMSNorm (no learnable scale, matches TPTT eq. 6 verbatim).
        rms = y.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        y = y / rms

        # Out projection (Linear, no bias).
        y = self.out_proj(y.to(dtype))

        # Per-channel gain. out_scale is [H, hd] for parity with multi-
        # head shape; reshape to [n_embd] for the merged-head application.
        y = y * self.out_scale.to(dtype).reshape(H * hd)
        return y

    # ------------------------------------------------------------------
    # Sequential reference path
    # ------------------------------------------------------------------

    def _forward_chunk_sequential(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries=None,
    ) -> tuple:
        """Per-token recurrent forward — reference correctness path.

        Bit-equivalent to T calls to `step_with_conv`. The chunkwise path
        (block_size>1) is checked against this for correctness.

        Per-token update with vector β:
            k_β = β_i ⊙ k_i,    v_β = β_i ⊙ v_i
            M ← M + (v_β − M·k_β) · k_iᵀ       — un-gated k on the right
        """
        M, k_buf_in = state_in  # M: [B, H, hd, hd], k_buf_in: list of [B, 2, H, hd]
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim

        M_dtype = M.dtype
        q, ks, vs, bs, k_buf_out = self._project_kvb(x_chunk, k_buf_in)

        # Flatten (B, H) into the bmm batch dim.
        M_v = M.reshape(B * H, hd, hd)

        y_steps = []
        for t in range(T):
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]  # [B] bool
                if reset_mask.any():
                    rm = reset_mask.view(B, 1).expand(B, H).reshape(B * H, 1, 1)
                    M_v = torch.where(rm, torch.zeros_like(M_v), M_v)

            for i in range(self.order):
                k_i = ks[i][:, t].to(M_dtype).reshape(B * H, hd)
                v_i = vs[i][:, t].to(M_dtype).reshape(B * H, hd)
                beta_i = bs[i][:, t].to(M_dtype).reshape(B * H, hd)

                k_beta = beta_i * k_i              # [B*H, hd]
                v_beta = beta_i * v_i              # [B*H, hd]
                # M · (β ⊙ k)  →  [B*H, hd]
                Mk = torch.bmm(M_v, k_beta.unsqueeze(-1)).squeeze(-1)
                err = v_beta - Mk
                # Rank-1 outer product with UN-GATED k on the right.
                delta = torch.bmm(err.unsqueeze(-1), k_i.unsqueeze(-2))
                M_v = M_v + delta

            q_t = q[:, t].to(M_dtype).reshape(B * H, hd)
            y_t = torch.bmm(M_v, q_t.unsqueeze(-1)).squeeze(-1)
            y_steps.append(y_t.view(B, H, hd))

        y_BTHhd = torch.stack(y_steps, dim=1)  # [B, T, H, hd]
        y = self._apply_out_norm_and_proj(y_BTHhd, x_chunk.dtype)
        M_out = M_v.view(B, H, hd, hd)
        return y, (M_out, k_buf_out)

    # ------------------------------------------------------------------
    # Chunkwise WY parallel path
    # ------------------------------------------------------------------

    def _forward_chunk_blockwise(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries=None,
    ) -> tuple:
        """Chunkwise parallel forward — closed-form WY representation
        with vector β.

        Bit-equivalent to the sequential per-token recurrence: one WY
        triangular solve per contiguous document segment.
        """
        M, k_buf_in = state_in  # M: [B, H, hd, hd], k_buf_in: list of [B, 2, H, hd]
        B, T, _ = x_chunk.shape

        q_all, ks_all, vs_all, bs_all, k_buf_out = self._project_kvb(
            x_chunk, k_buf_in,
        )

        splits = {0, T}
        if doc_boundaries is not None:
            any_boundary = doc_boundaries.any(dim=0)
            boundary_positions = (
                torch.nonzero(any_boundary, as_tuple=False).flatten().tolist()
            )
            splits.update(boundary_positions)
        split_list = sorted(splits)

        y_segments = []
        for seg_i in range(len(split_list) - 1):
            t_lo, t_hi = split_list[seg_i], split_list[seg_i + 1]
            if t_hi == t_lo:
                continue
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t_lo]
                M = torch.where(
                    reset_mask.view(B, 1, 1, 1),
                    torch.zeros_like(M),
                    M,
                )
            q_seg = q_all[:, t_lo:t_hi]
            ks_seg = [k[:, t_lo:t_hi] for k in ks_all]
            vs_seg = [v[:, t_lo:t_hi] for v in vs_all]
            bs_seg = [b[:, t_lo:t_hi] for b in bs_all]
            y_seg_raw, M = self._chunkwise_solve_raw(
                M, q_seg, ks_seg, vs_seg, bs_seg,
            )
            y_segments.append(y_seg_raw)

        y_raw = (
            y_segments[0] if len(y_segments) == 1
            else torch.cat(y_segments, dim=1)
        )
        y = self._apply_out_norm_and_proj(y_raw, x_chunk.dtype)
        return y, (M, k_buf_out)

    def _chunkwise_solve_raw(
        self,
        M_in: torch.Tensor,
        q: torch.Tensor,
        ks: list,
        vs: list,
        bs: list,
    ) -> tuple:
        """Core WY chunkwise solve with vector β, all heads batched.

        M_in: [B, H, hd, hd]; q: [B, T, H, hd]; ks/vs/bs: list of [B, T, H, hd].

        Returns (y_raw [B, T, H, hd], M_out [B, H, hd, hd]).

        Math (per (B, H) slice):
            K_β  = β ⊙ K,     V_β = β ⊙ V          (element-wise)
            G[s, j] = (β_s ⊙ k_s) · k_j = (K_β @ Kᵀ)[s, j]   for j < s
            R[s]    = (β_s ⊙ v_s) − M_in (β_s ⊙ k_s) = V_β − K_β @ M_inᵀ
            (I + G) U = R                          (unit-lower-tri solve)
            M_out   = M_in + Uᵀ K                  (un-gated K on the right)
            y[t]    = M_in q_t + Σ_{s ≤ (t+1)N − 1} (q_t · k_s) u_s
        """
        M_dtype = M_in.dtype
        B, T, H, hd = q.shape
        N = len(ks)
        TN = T * N

        # Stack to virtual sequence [B, T, N, H, hd]; permute to put H
        # before TN; flatten (B, H) into the bmm batch dim.
        K_stack = torch.stack(ks, dim=2)
        V_stack = torch.stack(vs, dim=2)
        beta_stack = torch.stack(bs, dim=2)

        K_virt = (
            K_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        V_virt = (
            V_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        beta_virt = (
            beta_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        Q_BH = q.permute(0, 2, 1, 3).to(M_dtype)  # [B, H, T, hd]

        K_v = K_virt.reshape(B * H, TN, hd)
        V_v = V_virt.reshape(B * H, TN, hd)
        beta_v = beta_virt.reshape(B * H, TN, hd)
        Q_v = Q_BH.reshape(B * H, T, hd)
        M_in_v = M_in.reshape(B * H, hd, hd)

        # β-gated K and V (element-wise on the head_dim axis).
        K_beta_v = beta_v * K_v
        V_beta_v = beta_v * V_v

        # Mk_β[b, s] = M_in · (β_s ⊙ k_s) — uses K_beta on the row side.
        Mk_beta = torch.bmm(K_beta_v, M_in_v.transpose(-1, -2))
        R = V_beta_v - Mk_beta

        # G[b, s, j] = (β_s ⊙ k_s) · k_j  —  asymmetric Gram.
        # K_beta on rows, K (un-gated) on cols.
        mask_lt, I_TN, real_mask = _chunkwise_aux_tensors(
            K_v.device, T, N, M_dtype,
        )
        G_full = torch.bmm(K_beta_v, K_v.transpose(-1, -2))
        G = G_full * mask_lt

        # (I + G) U = R via unit-lower-triangular solve.
        LhS = I_TN.unsqueeze(0) + G
        U = torch.linalg.solve_triangular(
            LhS, R, upper=False, unitriangular=True,
        )

        # M_out = M_in + Uᵀ K  (K un-gated — rank-1 outer products use
        # the right-side k unchanged).
        M_out_v = M_in_v + torch.bmm(U.transpose(-1, -2), K_v)

        # Reads: y_t uses M_t = M_in + Σ_{s ≤ Nt+N-1} u_s k_sᵀ.
        QKt = torch.bmm(Q_v, K_v.transpose(-1, -2))  # [B*H, T, TN]
        A_rv = QKt * real_mask
        y_init = torch.bmm(Q_v, M_in_v.transpose(-1, -2))
        y_acc = torch.bmm(A_rv, U)
        y_raw_v = y_init + y_acc  # [B*H, T, hd]

        M_out = M_out_v.view(B, H, hd, hd)
        y_raw = y_raw_v.view(B, H, T, hd).permute(0, 2, 1, 3).contiguous()
        return y_raw, M_out

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def forward_chunk(self, x_chunk, state_in, doc_boundaries=None) -> tuple:
        """Dispatch sequential (block_size=1) vs blockwise (block_size>1)."""
        if self.block_size > 1:
            return self._forward_chunk_blockwise(
                x_chunk, state_in, doc_boundaries,
            )
        return self._forward_chunk_sequential(
            x_chunk, state_in, doc_boundaries,
        )

    def step_with_conv(self, x_t: torch.Tensor, state: tuple) -> tuple:
        """Decode-path single-token forward.

        Same per-token recurrence as `_forward_chunk_sequential` with T=1.
        Name kept as `step_with_conv` for polymorphism with NMM at the
        call site — DeltaProduct itself has no conv preprocessing.

        Args:
            x_t: [B, n_embd] — single-token post-LN input.
            state: (M,) — current recurrent state.

        Returns:
            (y_t [B, n_embd], new_state).
        """
        x_unsq = x_t.unsqueeze(1)
        y_chunk, new_state = self._forward_chunk_sequential(
            x_unsq, state, doc_boundaries=None,
        )
        return y_chunk.squeeze(1), new_state
