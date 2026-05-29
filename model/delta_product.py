"""DeltaProduct fast-weight memory — TPTT formulation.

Replaces the NMM's surprise-driven inner-loop gradient update with the
explicit delta rule (Yang et al., NeurIPS 2024 / arxiv 2406.06484) of
order N (Siems et al., ICLR 2025 / arxiv 2502.10297). The specific
projection / gating / normalization / order-N-expansion conventions
follow TPTT (Furfaro 2025, arxiv 2506.17671), which is the published
recipe for adapting *pretrained* transformers.

TPTT-specific design (see TPTT Section 3.1 for the equations):

  Projections (TPTT eq. 4):
    q_normed = L2_normalize(SiLU(q_raw))
    k_normed = L2_normalize(SiLU(k_raw))
    v_scaled = v_raw / sqrt(head_dim)

  β gating (TPTT eq. 5, default `beta_gate="k"`):
    β = sigmoid(CausalAvgPool_3(k_raw))   — vector, per-component
    pool kernel = [1/3, 1/3, 1/3], fixed (not learnable)

  Order-N expansion (VirtualTokenExpander, "dt" / derivative trick):
    deriv_kernel[k] = (-1)^k · C(n-1, k), normalized by sum of abs
    For each (q, k, v, β):
      virtual[s, k] = x_padded[s + (n-1-k)] · deriv_kernel[n-1-k]
    Reshaped to virtual sequence: T·N writes per chunk.

  Per-token update with vector β (virtual sequence):
    M ← M + (β⊙v − M·(β⊙k)) · (β⊙k)ᵀ
    The β-gated key appears on BOTH sides of the rank-1 update,
    matching TPTT's recurrence (w · u_valᵀ outer product in
    `modeling_tptt.py:1490`, where `w = inv_hh @ k_beta` is the
    WY-transformed β⊙k). For scalar β this is equivalent to "β on the
    read side only"; for vector β they differ, and TPTT's choice
    is what we mirror.

  Read: at virtual position N·t + N−1 (last sub-step of real token t),
    using virtual_q at that position. The output for real token t is
    y[t] = virtual_q_last_substep[t] · M.

  Output (TPTT eq. 6):
    y_out = RMSNorm(y_raw) · out_proj(bias=True) · out_scale

  out_proj has a learnable bias (matches TPTT default). out_scale is
  our addition — preserves cli/train.py's gate-ramp logic and the
  finetune-mode "y = 0 at step 0" invariant.

State: 3-tuple `(M, k_raw_buf, qkvb_buf)`:
  - `M`: [B, n_heads, head_dim, head_dim] — recurrent state, TPTT-style
    1e-6 fill at init (numerical stability for un-linear activation).
  - `k_raw_buf`: [B, 2, n_heads, head_dim] — last 2 raw K projections,
    threaded across forward_chunk calls so the CausalAvgPool sees
    continuous context.
  - `qkvb_buf`: tuple of (q, k, v, β) each [B, n-1, n_heads, head_dim] —
    last (n-1) post-projection tensors, threaded so the
    VirtualTokenExpander sees continuous context across chunks. None at
    order=1 (no expansion = no continuity buffer needed).

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

    Replicate-pads the leading 2 positions. Matches TPTT's
    `CausalAvgPool1d` semantics (kernel size 3, weights [1/3, 1/3, 1/3],
    fixed/not-learnable, replicate padding).
    """
    first = x[:, :1]
    x_pad = torch.cat([first, first, x], dim=1)  # [B, T+2, ...]
    return (x_pad[:, :-2] + x_pad[:, 1:-1] + x_pad[:, 2:]) / 3.0


def _virtual_token_expand(
    x: torch.Tensor, n: int, kernel: torch.Tensor,
) -> torch.Tensor:
    """TPTT's `VirtualTokenExpander` for the "dt" (derivative) trick.

    Args:
        x: [B, T, H, hd] — post-projection tensor.
        n: expansion order (the same as `order`).
        kernel: [n] — pre-normalized binomial-coefficient derivative
            kernel: `deriv[k] = (-1)^k · C(n-1, k) / sum(|coeffs|)`.

    Returns:
        Tensor of shape [B, T, n, H, hd]. For each real position s and
        sub-step k:
            virtual[s, k] = x_padded[s + (n-1-k)] · kernel[n-1-k]
        where `x_padded` is `x` prepended with (n-1) zeros for the
        unfold-from-left padding TPTT uses internally.

    Notes:
        - The implementation mirrors TPTT's `VirtualTokenExpander._apply_derivative_conv`
          and the subsequent `flip(-1).permute(0, 1, 2, 4, 3)` exactly,
          modulo the input being head-split [B, T, H, hd] in our layout
          vs TPTT's [B, H, T, hd]. We unfold along dim=1.
        - For n=1, kernel = [1.0] and the expansion is the identity
          (input wrapped as [B, T, 1, H, hd]) — short-circuit at the
          call site for clarity.
    """
    B, T, H, hd = x.shape
    # Internal padding: prepend (n-1) zeros along T for unfold.
    if n > 1:
        pad = torch.zeros(B, n - 1, H, hd, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([pad, x], dim=1)  # [B, T+n-1, H, hd]
    else:
        x_padded = x
    # Unfold along T with size=n step=1: result [B, T, H, hd, n].
    windows = x_padded.unfold(dimension=1, size=n, step=1)
    # Multiply by kernel (broadcast over B, T, H, hd; index by k).
    conv_out = windows * kernel.view(1, 1, 1, 1, n).to(x.dtype)
    # Flip the last (kernel-index) dim so the natural order (k=0 →
    # current token, k=n-1 → oldest token) is reversed: TPTT's
    # convention has k=n-1 as the oldest. The flip + permute produces
    # the layout [B, T, n, H, hd] with sub-step indexed by dim=2.
    conv_out = conv_out.flip(-1)
    out = conv_out.permute(0, 1, 4, 2, 3)  # [B, T, n, H, hd]
    return out


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
            sequential per-token recurrence (reference correctness path).
            `block_size>1` routes through the closed-form chunkwise WY
            parallel path (training-time speed path, bit-equivalent to
            sequential, one triangular solve per document segment).
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
        self._v_scale = 1.0 / math.sqrt(self.head_dim)

        H, hd = self.n_heads, self.head_dim

        # SINGLE Q, K, V projections (TPTT design — N virtual tokens
        # come from the VirtualTokenExpander, not N independent learned
        # projections). β is computed from K_raw via fixed CausalAvgPool.
        self.q_proj_weight = nn.Parameter(torch.empty(H, hd, hd))
        self.k_proj_weight = nn.Parameter(torch.empty(H, hd, hd))
        self.v_proj_weight = nn.Parameter(torch.empty(H, hd, hd))

        # Output projection — matches TPTT's `out_proj = Linear(...,
        # bias=True)`. The bias is multiplied by `out_scale` post-norm,
        # so finetune-mode "y = 0 at step 0" still holds.
        self.out_proj = nn.Linear(n_embd, n_embd, bias=True)

        # Per-channel output gain. Zero-init under finetune_mode.
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(H, hd))
        else:
            self.out_scale = nn.Parameter(torch.ones(H, hd))

        # VirtualTokenExpander kernel — pre-normalized binomial
        # coefficients with alternating signs. Fixed buffer (NOT a
        # parameter). Only needed for order > 1; order=1 short-circuits.
        if order > 1:
            coeffs = [(-1) ** k * math.comb(order - 1, k) for k in range(order)]
            kernel = torch.tensor(coeffs, dtype=torch.float32)
            kernel = kernel / kernel.abs().sum()
            self.register_buffer("_virtual_kernel", kernel, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Match PyTorch Linear's default kaiming init for each per-head
        slice on stacked tensors; out_proj inherits PyTorch's Linear
        default."""
        hd = self.head_dim
        bound = 1.0 / math.sqrt(hd)
        with torch.no_grad():
            nn.init.uniform_(self.q_proj_weight, -bound, bound)
            nn.init.uniform_(self.k_proj_weight, -bound, bound)
            nn.init.uniform_(self.v_proj_weight, -bound, bound)

    # ------------------------------------------------------------------
    # State + projections
    # ------------------------------------------------------------------

    def init_state(self, B: int, device) -> tuple:
        """Per-sample-batched initial state.

        Returns a 3-tuple `(M, k_raw_buf, qkvb_buf)`:
          - `M`: [B, n_heads, head_dim, head_dim] — 1e-6 fill (TPTT's
            `state` initial fill_value for "stability if unlinear
            activation"). Pretrained-mode finetuning preserves the
            backbone's residual at step 0 via `out_scale = 0`, not via
            M = 0, so 1e-6 here doesn't affect step-0 behavior.
          - `k_raw_buf`: [B, 2, n_heads, head_dim] — last 2 raw K
            projections for the CausalAvgPool's 3-token window.
          - `qkvb_buf`: tuple (q, k, v, β) each [B, order-1, n_heads,
            head_dim] for the VirtualTokenExpander's (order-1)-token
            window. None at order=1.
        """
        H, hd = self.n_heads, self.head_dim
        M = torch.full(
            (B, H, hd, hd),
            fill_value=1e-6,
            device=device, dtype=torch.float32,
        )
        k_raw_buf = torch.zeros(
            B, 2, H, hd, device=device, dtype=torch.float32,
        )
        if self.order > 1:
            n_minus_1 = self.order - 1
            qkvb_buf = tuple(
                torch.zeros(
                    B, n_minus_1, H, hd,
                    device=device, dtype=torch.float32,
                )
                for _ in range(4)  # q, k, v, β
            )
        else:
            qkvb_buf = None
        return (M, k_raw_buf, qkvb_buf)

    def _project_kvb(
        self, x_chunk: torch.Tensor, k_raw_buf_in: torch.Tensor, qkvb_buf_in,
    ):
        """Project x_chunk → (virtual_q, virtual_k, virtual_v, virtual_β)
        with cross-chunk continuity for both the CausalAvgPool gating
        and the VirtualTokenExpander.

        Returns:
            q_virt   [B, T, n, H, hd] — virtual queries from expander
            k_virt   [B, T, n, H, hd] — virtual keys (SiLU+L2 then expander)
            v_virt   [B, T, n, H, hd] — virtual values (scaled then expander)
            beta_virt [B, T, n, H, hd] — virtual β (vector, expanded)
            k_raw_buf_out [B, 2, H, hd] — new pool buffer
            qkvb_buf_out  tuple(4) of [B, order-1, H, hd] — new expander
                buffer (None at order=1)
        """
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim
        n = self.order
        x_h = x_chunk.view(B, T, H, hd)

        # Single Q, K, V projections.
        q_raw = torch.einsum("bthd,hde->bthe", x_h, self.q_proj_weight)
        k_raw = torch.einsum("bthd,hde->bthe", x_h, self.k_proj_weight)
        v_raw = torch.einsum("bthd,hde->bthe", x_h, self.v_proj_weight)

        # β = σ(CausalAvgPool(k_raw_extended_with_buf)).
        k_raw_pool_input = torch.cat([k_raw_buf_in, k_raw], dim=1)
        beta_full = torch.sigmoid(_causal_avg_pool_3(k_raw_pool_input))
        beta = beta_full[:, 2:]  # [B, T, H, hd]
        k_raw_buf_out = k_raw_pool_input[:, -2:]

        # SiLU+L2 on Q and K; V scaled by 1/√head_dim.
        q = F.normalize(F.silu(q_raw), p=2, dim=-1, eps=1e-6)
        k = F.normalize(F.silu(k_raw), p=2, dim=-1, eps=1e-6)
        v = v_raw * self._v_scale

        # VirtualTokenExpander on q, k, v, β. For n=1 the expansion is
        # an identity (add a length-1 sub-step dim). For n>1 we prepend
        # the per-tensor (n-1)-token buffer for cross-chunk continuity.
        if n == 1:
            q_virt = q.unsqueeze(2)        # [B, T, 1, H, hd]
            k_virt = k.unsqueeze(2)
            v_virt = v.unsqueeze(2)
            beta_virt = beta.unsqueeze(2)
            qkvb_buf_out = None
        else:
            # Concat per-tensor expander buffers; expand the longer
            # sequence; slice off the prepended portion so the output
            # length is exactly T.
            q_ext = torch.cat([qkvb_buf_in[0], q], dim=1)
            k_ext = torch.cat([qkvb_buf_in[1], k], dim=1)
            v_ext = torch.cat([qkvb_buf_in[2], v], dim=1)
            beta_ext = torch.cat([qkvb_buf_in[3], beta], dim=1)

            kernel = self._virtual_kernel
            q_virt_full = _virtual_token_expand(q_ext, n, kernel)
            k_virt_full = _virtual_token_expand(k_ext, n, kernel)
            v_virt_full = _virtual_token_expand(v_ext, n, kernel)
            beta_virt_full = _virtual_token_expand(beta_ext, n, kernel)

            # The expander output has length (T + n - 1). Slice the last
            # T positions, which correspond to the current chunk's real
            # tokens with the prior-chunk context baked into the early
            # sub-steps.
            q_virt = q_virt_full[:, -T:]
            k_virt = k_virt_full[:, -T:]
            v_virt = v_virt_full[:, -T:]
            beta_virt = beta_virt_full[:, -T:]

            # New buffer: last (n-1) of the extended post-projection
            # tensors (TPTT saves them BEFORE the expander, line ~553
            # in modeling_tptt.py).
            qkvb_buf_out = (
                q_ext[:, -(n - 1):],
                k_ext[:, -(n - 1):],
                v_ext[:, -(n - 1):],
                beta_ext[:, -(n - 1):],
            )

        return (
            q_virt, k_virt, v_virt, beta_virt,
            k_raw_buf_out, qkvb_buf_out,
        )

    def _apply_out_norm_and_proj(
        self, y_BTHd: torch.Tensor, dtype,
    ) -> torch.Tensor:
        """TPTT-style output: merge heads, manual RMSNorm, Linear out_proj
        (bias=True), per-channel `out_scale` gain.
        """
        B, T = y_BTHd.shape[:2]
        H, hd = self.n_heads, self.head_dim

        y = y_BTHd.reshape(B, T, H * hd)
        rms = y.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        y = y / rms
        y = self.out_proj(y.to(dtype))
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

        The chunkwise path (block_size>1) is checked against this for
        correctness. Iterates over the T·n virtual writes with the
        delta-rule update, reading at every n-th virtual position using
        the corresponding virtual Q.
        """
        M, k_raw_buf_in, qkvb_buf_in = state_in
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim
        n = self.order

        M_dtype = M.dtype
        q_virt, k_virt, v_virt, beta_virt, k_raw_buf_out, qkvb_buf_out = (
            self._project_kvb(x_chunk, k_raw_buf_in, qkvb_buf_in)
        )

        M_v = M.reshape(B * H, hd, hd)

        y_steps = []
        for t in range(T):
            # Doc-boundary reset for this real token (BEFORE its first
            # virtual write). M is per-row; broadcast over heads.
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]
                if reset_mask.any():
                    rm = reset_mask.view(B, 1).expand(B, H).reshape(B * H, 1, 1)
                    M_v = torch.where(rm, torch.zeros_like(M_v), M_v)

            # n virtual writes for real token t.
            for k in range(n):
                k_i = k_virt[:, t, k].to(M_dtype).reshape(B * H, hd)
                v_i = v_virt[:, t, k].to(M_dtype).reshape(B * H, hd)
                beta_i = beta_virt[:, t, k].to(M_dtype).reshape(B * H, hd)
                k_beta = beta_i * k_i
                v_beta = beta_i * v_i
                Mk = torch.bmm(M_v, k_beta.unsqueeze(-1)).squeeze(-1)
                err = v_beta - Mk
                # Rank-1 outer product with β-GATED k on the right
                # (TPTT's `w · u_valᵀ` with `w = inv_hh @ k_beta`).
                delta = torch.bmm(err.unsqueeze(-1), k_beta.unsqueeze(-2))
                M_v = M_v + delta

            # Read at the last sub-step using virtual_q[t, n-1].
            q_t = q_virt[:, t, n - 1].to(M_dtype).reshape(B * H, hd)
            y_t = torch.bmm(M_v, q_t.unsqueeze(-1)).squeeze(-1)
            y_steps.append(y_t.view(B, H, hd))

        y_BTHhd = torch.stack(y_steps, dim=1)  # [B, T, H, hd]
        y = self._apply_out_norm_and_proj(y_BTHhd, x_chunk.dtype)
        M_out = M_v.view(B, H, hd, hd)
        return y, (M_out, k_raw_buf_out, qkvb_buf_out)

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
        with vector β over the virtual sequence.

        One WY solve per contiguous document segment.
        """
        M, k_raw_buf_in, qkvb_buf_in = state_in
        B, T, _ = x_chunk.shape

        q_virt, k_virt, v_virt, beta_virt, k_raw_buf_out, qkvb_buf_out = (
            self._project_kvb(x_chunk, k_raw_buf_in, qkvb_buf_in)
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
            q_seg = q_virt[:, t_lo:t_hi]
            k_seg = k_virt[:, t_lo:t_hi]
            v_seg = v_virt[:, t_lo:t_hi]
            b_seg = beta_virt[:, t_lo:t_hi]
            y_seg_raw, M = self._chunkwise_solve_raw(
                M, q_seg, k_seg, v_seg, b_seg,
            )
            y_segments.append(y_seg_raw)

        y_raw = (
            y_segments[0] if len(y_segments) == 1
            else torch.cat(y_segments, dim=1)
        )
        y = self._apply_out_norm_and_proj(y_raw, x_chunk.dtype)
        return y, (M, k_raw_buf_out, qkvb_buf_out)

    def _chunkwise_solve_raw(
        self,
        M_in: torch.Tensor,
        q_virt: torch.Tensor,
        k_virt: torch.Tensor,
        v_virt: torch.Tensor,
        beta_virt: torch.Tensor,
    ) -> tuple:
        """Core WY chunkwise solve with vector β over the virtual sequence.

        Inputs are pre-expanded virtual tensors of shape [B, T, n, H, hd].

        Returns (y_raw [B, T, H, hd], M_out [B, H, hd, hd]).

        The read at each real token t uses the virtual Q at the LAST
        sub-step `q_virt[:, t, n-1]`. M_out reflects all T·n virtual
        writes for the chunk.

        TPTT-faithful recurrence — β-gated K appears on BOTH sides of
        the rank-1 update (G's column side, M_out's right factor, the
        read inner-product):
            G[s, j]  = (β_s ⊙ k_s) · (β_j ⊙ k_j) = (K_β @ K_βᵀ)[s, j]
            R[s]     = (β_s ⊙ v_s) − M_in · (β_s ⊙ k_s)
            (I + G) U = R                          (unit-lower-tri solve)
            M_out    = M_in + Uᵀ K_β               (β-gated K on right)
            y[t]     = M_in q_t + Σ_{s ≤ (t+1)N − 1} (q_t · (β_s ⊙ k_s)) u_s

        For scalar β this is equivalent to "β only on the read side";
        for vector β the two formulations differ, and we mirror TPTT.
        """
        M_dtype = M_in.dtype
        B, T, n, H, hd = q_virt.shape
        TN = T * n

        # Reshape virtual sequence: collapse the (T, n) dims into TN.
        # Layout: virtual position N·t + k for real t, sub-step k.
        # The natural reshape gives this layout when the n dim is
        # adjacent to T (which it is: [B, T, n, H, hd]).
        K_virt = (
            k_virt.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        V_virt = (
            v_virt.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        beta_virt_r = (
            beta_virt.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )

        # Virtual queries at the last sub-step for each real token.
        # Shape: [B, T, H, hd] → permute → [B, H, T, hd].
        q_last = q_virt[:, :, n - 1].permute(0, 2, 1, 3).to(M_dtype)

        K_v = K_virt.reshape(B * H, TN, hd)
        V_v = V_virt.reshape(B * H, TN, hd)
        beta_v = beta_virt_r.reshape(B * H, TN, hd)
        Q_v = q_last.reshape(B * H, T, hd)
        M_in_v = M_in.reshape(B * H, hd, hd)

        K_beta_v = beta_v * K_v
        V_beta_v = beta_v * V_v

        Mk_beta = torch.bmm(K_beta_v, M_in_v.transpose(-1, -2))
        R = V_beta_v - Mk_beta

        mask_lt, I_TN, real_mask = _chunkwise_aux_tensors(
            K_v.device, T, n, M_dtype,
        )
        # Gram of β-gated keys on BOTH sides (TPTT recurrence).
        G_full = torch.bmm(K_beta_v, K_beta_v.transpose(-1, -2))
        G = G_full * mask_lt
        LhS = I_TN.unsqueeze(0) + G
        U = torch.linalg.solve_triangular(
            LhS, R, upper=False, unitriangular=True,
        )

        # M_out uses β-gated K on the right of the rank-1 outer product.
        M_out_v = M_in_v + torch.bmm(U.transpose(-1, -2), K_beta_v)

        # Reads' inner-product is q_t · (β_s ⊙ k_s), so use K_β here too.
        QKt = torch.bmm(Q_v, K_beta_v.transpose(-1, -2))  # [B*H, T, TN]
        A_rv = QKt * real_mask
        y_init = torch.bmm(Q_v, M_in_v.transpose(-1, -2))
        y_acc = torch.bmm(A_rv, U)
        y_raw_v = y_init + y_acc

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
        """
        x_unsq = x_t.unsqueeze(1)
        y_chunk, new_state = self._forward_chunk_sequential(
            x_unsq, state, doc_boundaries=None,
        )
        return y_chunk.squeeze(1), new_state
