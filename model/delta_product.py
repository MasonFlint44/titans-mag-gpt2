"""DeltaProduct fast-weight memory — alternative to NeuralMemoryModule.

Replaces the NMM's surprise-driven inner-loop gradient update with the
explicit delta rule (Yang et al., NeurIPS 2024 / arxiv 2406.06484) of
order N (Siems et al., ICLR 2025 / arxiv 2502.10297).

Order=1 reduces to DeltaNet. Order=N>=2 is DeltaProduct. The TPTT paper
(arxiv 2506.17671) shows order=2 matches Titans expressivity, and uses
this family as the production "Memory as Gate" mechanism for retrofitting
pretrained transformers — exactly the setting we're targeting.

Per-token recurrence (single head, order N), write-then-read:

    for i in 1..N:
        k_i, v_i, β_i = K_i(x_t), V_i(x_t), σ(B_i(x_t))
        M ← M + β_i · (v_i − M·k_i) · k_iᵀ
    y_t = M · Q(x_t)

Multi-head fusion: `n_heads > 1` splits `n_embd` into `n_heads × head_dim`
parallel memory heads, each with its own [head_dim × head_dim] M matrix
and independent per-head projection weights. The forward runs all heads
in parallel via batched einsum projections and a WY solve with batch
dim = B·n_heads — substantially fewer kernel launches than running
n_heads single-head modules in a Python loop.

State: a 1-tuple `(M,)` with `M ∈ R^(B, n_heads, head_dim, head_dim)`.
No momentum stack, no depthwise conv buffer — DeltaProduct's update is
closed-form per token.

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

    Caching by `(device, T, N, dtype)` because:
      - At training scale (T·N ≈ 2056) `mask_lt` alone is ~4 MB, so
        re-allocating per layer per forward is ~150 MB/s of allocator
        pressure for purely deterministic tensors.
      - `(T, N, device, dtype)` is the full identifier of the tensor
        values — there's no per-call variation.

    `lru_cache(maxsize=128)` bounds cache memory if the caller exercises
    many distinct chunk lengths (e.g., variable-length doc segments at
    training time). In practice most calls share one (T, N) so the cache
    is hot after the first few forwards.
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


class DeltaProductMemory(nn.Module):
    """Multi-head DeltaProduct memory, order >= 1.

    `n_heads=1` is the single-head case (memory operates on the full
    n_embd dim). `n_heads>1` runs n_heads parallel head_dim-sized memory
    modules, fused into batched einsum projections + batched WY solve.
    The math per head is identical to running n_heads independent
    single-head modules; the speedup is purely in kernel launches and
    tensor-core utilization.

    Args:
        n_embd: input/output dim.
        n_heads: number of parallel memory heads. Must divide n_embd.
            Default 1 = single-head with `head_dim = n_embd`.
        order: number of delta sub-steps applied per token (1 = DeltaNet,
            2 = matches Titans per TPTT). Default 2.
        finetune_mode: when True, `out_scale` initializes to zero so that
            y_mem = 0 at step 0 and the pretrained backbone's residual is
            preserved exactly until training picks up the gate. Mirrors
            NMM's `out_scale` semantics.
        block_size: forward-path selector. `block_size=1` runs the
            sequential per-token recurrence (reference correctness path,
            slow but bit-exact; useful as a baseline and as the decode
            path). `block_size>1` routes through the closed-form
            chunkwise WY parallel path (training-time speed path, bit-
            equivalent to sequential, one triangular solve per document
            segment). The numeric value above 1 is currently unused —
            reserved for a future memory-bounded sub-chunking path; see
            `config.delta_block_size` docstring.
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

        H, hd = self.n_heads, self.head_dim

        # Stacked per-head projection weights. Mathematically equivalent
        # to N independent per-head Linear modules — same parameter
        # count, same per-head [hd, hd] mapping — but stored as one
        # contiguous tensor so the forward fuses N projections into one
        # einsum kernel per role. ~3-5× kernel-launch reduction vs the
        # earlier Python-loop-over-heads design.
        self.q_proj_weight = nn.Parameter(torch.empty(H, hd, hd))
        self.k_proj_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, hd, hd)) for _ in range(order)]
        )
        self.v_proj_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, hd, hd)) for _ in range(order)]
        )
        # β: per-token, per-head scalar write strength. Weight is
        # [H, hd, 1] (head_dim → 1 per head); bias is [H, 1].
        self.beta_proj_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, hd, 1)) for _ in range(order)]
        )
        self.beta_proj_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(H, 1)) for _ in range(order)]
        )

        # Per-head output scale gate. Zero-init under finetune_mode so
        # the pretrained backbone's residual is preserved at step 0.
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(H, hd))
        else:
            self.out_scale = nn.Parameter(torch.ones(H, hd))

        self._init_weights()

        # Backward-compat: migrate legacy state-dict layouts to the
        # stacked-parameter layout used by this class. Two legacy shapes
        # are recognized:
        #   1. Pre-fusion multi-head: MultiHeadDeltaProduct wrapped N
        #      per-head DeltaProductMemory submodules under `heads.{h}.`.
        #   2. Pre-fusion single-head: DeltaProductMemory used `q_proj`,
        #      `k_projs.{i}`, etc. as Linear modules.
        # See `_migrate_legacy_format` for the per-key mapping.
        self.register_load_state_dict_pre_hook(self._migrate_legacy_format)

    def _init_weights(self) -> None:
        """Match PyTorch Linear's default init for each per-head slice.

        Linear(in_features=hd) uses kaiming_uniform_(a=√5) on its weight
        — uniform(-1/√hd, 1/√hd) — and the same bound for the bias.
        We replicate that distribution per-head on the stacked tensors
        so n_heads × stacked-DeltaProduct(n_heads=H) is statistically
        indistinguishable from H independent Linear-based single-head
        DeltaProducts at init.
        """
        hd = self.head_dim
        bound_proj = 1.0 / math.sqrt(hd)
        with torch.no_grad():
            nn.init.uniform_(self.q_proj_weight, -bound_proj, bound_proj)
            for i in range(self.order):
                nn.init.uniform_(self.k_proj_weights[i], -bound_proj, bound_proj)
                nn.init.uniform_(self.v_proj_weights[i], -bound_proj, bound_proj)
                nn.init.uniform_(self.beta_proj_weights[i], -bound_proj, bound_proj)
                nn.init.uniform_(self.beta_proj_biases[i], -bound_proj, bound_proj)

    # ------------------------------------------------------------------
    # Legacy state-dict migration
    # ------------------------------------------------------------------

    def _migrate_legacy_format(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Detect-and-rewrite pre-fusion checkpoint layouts.

        New format keys are unchanged; we only act when sentinel keys
        from a legacy layout are present. State_dict is mutated in
        place (PyTorch's load contract).

        Signature note: PyTorch's public `register_load_state_dict_pre_hook`
        wraps the hook with `with_module=True`, so `module` is passed in
        first. It refers to the same instance as `self` here; the
        parameter exists to match the API contract.
        """
        H = self.n_heads
        order = self.order
        hd = self.head_dim

        if f"{prefix}heads.0.q_proj.weight" in state_dict:
            # Multi-head: MultiHeadDeltaProduct wrapping per-head submodules.
            self._migrate_multihead_legacy(state_dict, prefix, H, order)
            return

        if f"{prefix}q_proj.weight" in state_dict and H == 1:
            # Single-head pre-fusion: q_proj was a Linear module.
            self._migrate_singlehead_legacy(state_dict, prefix, order, hd)
            return

    @staticmethod
    def _migrate_multihead_legacy(state_dict, prefix, H, order) -> None:
        """Stack per-head Linear weights into the [H, *, *] layout."""
        q_list = [
            state_dict.pop(f"{prefix}heads.{h}.q_proj.weight") for h in range(H)
        ]
        state_dict[f"{prefix}q_proj_weight"] = torch.stack(q_list, dim=0)

        for i in range(order):
            k_list = [
                state_dict.pop(f"{prefix}heads.{h}.k_projs.{i}.weight")
                for h in range(H)
            ]
            state_dict[f"{prefix}k_proj_weights.{i}"] = torch.stack(k_list, dim=0)

            v_list = [
                state_dict.pop(f"{prefix}heads.{h}.v_projs.{i}.weight")
                for h in range(H)
            ]
            state_dict[f"{prefix}v_proj_weights.{i}"] = torch.stack(v_list, dim=0)

            # β: old per-head Linear(hd, 1) stored weight as [1, hd];
            # new layout uses [H, hd, 1]. Transpose each, then stack.
            bw_list = [
                state_dict.pop(f"{prefix}heads.{h}.beta_heads.{i}.weight")
                for h in range(H)
            ]
            state_dict[f"{prefix}beta_proj_weights.{i}"] = torch.stack(
                [w.t() for w in bw_list], dim=0,
            )

            bb_list = [
                state_dict.pop(f"{prefix}heads.{h}.beta_heads.{i}.bias")
                for h in range(H)
            ]
            state_dict[f"{prefix}beta_proj_biases.{i}"] = torch.stack(bb_list, dim=0)

        os_list = [
            state_dict.pop(f"{prefix}heads.{h}.out_scale") for h in range(H)
        ]
        state_dict[f"{prefix}out_scale"] = torch.stack(os_list, dim=0)

    @staticmethod
    def _migrate_singlehead_legacy(state_dict, prefix, order, hd) -> None:
        """Wrap single-head Linear weights into the [1, *, *] layout."""
        q_w = state_dict.pop(f"{prefix}q_proj.weight")  # [hd, hd]
        state_dict[f"{prefix}q_proj_weight"] = q_w.unsqueeze(0)

        for i in range(order):
            k_w = state_dict.pop(f"{prefix}k_projs.{i}.weight")
            state_dict[f"{prefix}k_proj_weights.{i}"] = k_w.unsqueeze(0)

            v_w = state_dict.pop(f"{prefix}v_projs.{i}.weight")
            state_dict[f"{prefix}v_proj_weights.{i}"] = v_w.unsqueeze(0)

            bw = state_dict.pop(f"{prefix}beta_heads.{i}.weight")  # [1, hd]
            state_dict[f"{prefix}beta_proj_weights.{i}"] = bw.t().unsqueeze(0)

            bb = state_dict.pop(f"{prefix}beta_heads.{i}.bias")  # [1]
            state_dict[f"{prefix}beta_proj_biases.{i}"] = bb.unsqueeze(0)

        os_p = state_dict.pop(f"{prefix}out_scale")  # [hd]
        state_dict[f"{prefix}out_scale"] = os_p.unsqueeze(0)

    # ------------------------------------------------------------------
    # State + projections
    # ------------------------------------------------------------------

    def init_state(self, B: int, device) -> tuple:
        """Per-sample-batched zero-initialized M.

        Returns a 1-tuple `(M,)` with M shape `[B, n_heads, head_dim,
        head_dim]`. Tuple-of-one layout is kept for parity with the NMM's
        `(M, S, conv_buf)` triple at the model-level dispatch layer (see
        `model/nmm.py::_detach_per_layer`).
        """
        M = torch.zeros(
            B, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float32,
        )
        return (M,)

    def _project_kvb(self, x_chunk: torch.Tensor):
        """Batched per-head projections for the whole chunk.

        x_chunk: [B, T, n_embd]
        Returns:
            q  [B, T, H, hd]
            ks list[order] of [B, T, H, hd] — L2-normalized along hd
            vs list[order] of [B, T, H, hd]
            bs list[order] of [B, T, H, 1] (after sigmoid)

        Keys are L2-normalized so ||k|| = 1 and the Gram K·Kᵀ has
        entries bounded in [-1, 1]. This bounds the spectral norm of
        β·K·Kᵀ inside the WY solve and keeps `(I + G)` well-conditioned
        at any T·N — required for the chunkwise path to stay finite at
        training scale.
        """
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim
        x_h = x_chunk.view(B, T, H, hd)

        q = torch.einsum("bthd,hde->bthe", x_h, self.q_proj_weight)

        ks = []
        for i in range(self.order):
            k_raw = torch.einsum("bthd,hde->bthe", x_h, self.k_proj_weights[i])
            ks.append(torch.nn.functional.normalize(k_raw, dim=-1))

        vs = [
            torch.einsum("bthd,hde->bthe", x_h, self.v_proj_weights[i])
            for i in range(self.order)
        ]

        bs = []
        for i in range(self.order):
            b_raw = torch.einsum("bthd,hde->bthe", x_h, self.beta_proj_weights[i])
            # bias broadcasts over (B, T): [H, 1] -> [1, 1, H, 1]
            b_raw = b_raw + self.beta_proj_biases[i].view(1, 1, H, 1)
            bs.append(torch.sigmoid(b_raw))

        return q, ks, vs, bs

    def _apply_out_scale(self, y_BHTd: torch.Tensor, dtype) -> torch.Tensor:
        """Apply per-head out_scale and reshape to [B, T, n_embd].

        y_BHTd: [B, T, H, hd] (raw read).
        """
        H, hd = self.n_heads, self.head_dim
        y = y_BHTd.to(dtype) * self.out_scale.to(dtype).view(1, 1, H, hd)
        return y.reshape(y.shape[0], y.shape[1], H * hd)

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
        """
        (M,) = state_in  # [B, H, hd, hd]
        B, T, _ = x_chunk.shape
        H, hd = self.n_heads, self.head_dim

        M_dtype = M.dtype
        q, ks, vs, bs = self._project_kvb(x_chunk)

        # Flatten (B, H) into the bmm batch dim.
        M_v = M.reshape(B * H, hd, hd)

        y_steps = []
        for t in range(T):
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]  # [B] bool
                if reset_mask.any():
                    # Broadcast per-batch mask across the H dim, then
                    # flatten to (B*H, 1, 1) for the where.
                    rm = reset_mask.view(B, 1).expand(B, H).reshape(B * H, 1, 1)
                    M_v = torch.where(rm, torch.zeros_like(M_v), M_v)

            for i in range(self.order):
                k_i = ks[i][:, t].to(M_dtype).reshape(B * H, hd)
                v_i = vs[i][:, t].to(M_dtype).reshape(B * H, hd)
                beta_i = bs[i][:, t].to(M_dtype).reshape(B * H, 1)

                Mk = torch.bmm(M_v, k_i.unsqueeze(-1)).squeeze(-1)
                err = v_i - Mk
                delta = torch.bmm(
                    (beta_i * err).unsqueeze(-1),
                    k_i.unsqueeze(-2),
                )
                M_v = M_v + delta

            q_t = q[:, t].to(M_dtype).reshape(B * H, hd)
            y_t = torch.bmm(M_v, q_t.unsqueeze(-1)).squeeze(-1)
            y_steps.append(y_t.view(B, H, hd))

        y_BTHhd = torch.stack(y_steps, dim=1)  # [B, T, H, hd]
        y = self._apply_out_scale(y_BTHhd, x_chunk.dtype)
        M_out = M_v.view(B, H, hd, hd)
        return y, (M_out,)

    # ------------------------------------------------------------------
    # Chunkwise WY parallel path
    # ------------------------------------------------------------------

    def _forward_chunk_blockwise(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries=None,
    ) -> tuple:
        """Chunkwise parallel forward — closed-form WY representation.

        Bit-equivalent to the sequential per-token recurrence: one WY
        triangular solve per contiguous document segment. With no doc
        boundaries that's one solve over the whole chunk; with
        boundaries we split at boundary positions and run one solve per
        segment, threading M between segments with per-row reset where
        needed.
        """
        (M,) = state_in  # [B, H, hd, hd]
        B, T, _ = x_chunk.shape

        q_all, ks_all, vs_all, bs_all = self._project_kvb(x_chunk)

        # Split positions: 0 (implicit start), T (implicit end), and any
        # position where any row has a doc boundary. Each segment gets
        # one WY solve; the head of each segment may reset M per-row.
        #
        # One CUDA sync per chunk (`.tolist()`) instead of one sync per
        # boundary position (the previous `int(t.item())` loop) — `.tolist()`
        # synchronously reads the whole 1-D tensor in a single round-trip.
        splits = {0, T}
        if doc_boundaries is not None:
            any_boundary = doc_boundaries.any(dim=0)  # [T]
            boundary_positions = (
                torch.nonzero(any_boundary, as_tuple=False).flatten().tolist()
            )
            splits.update(boundary_positions)
        split_list = sorted(splits)

        # Reset M unconditionally at every segment head where a boundary
        # fires — INCLUDING t_lo=0. The sequential path resets per token
        # without a t>0 guard; the chunkwise must match to stay bit-
        # equivalent. The data loader sets doc_boundaries[:, 0]=True at
        # every per-rank stream start and whenever EOT aligns with a
        # chunk boundary, so position-0 resets fire more than they might
        # seem to.
        #
        # The per-segment `torch.where` runs unconditionally instead of
        # being gated by `reset_mask.any()`. The kernel is a no-op when
        # the mask is all-False (cheap), but the `.any()` reads a 0-d
        # bool back to CPU — a sync per segment. For our use (few
        # segments per chunk) the trade is a wash on bandwidth and a win
        # on sync count.
        y_segments = []
        for seg_i in range(len(split_list) - 1):
            t_lo, t_hi = split_list[seg_i], split_list[seg_i + 1]
            if t_hi == t_lo:
                continue
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t_lo]  # [B]
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
        y = self._apply_out_scale(y_raw, x_chunk.dtype)
        return y, (M,)

    def _chunkwise_solve_raw(
        self,
        M_in: torch.Tensor,
        q: torch.Tensor,
        ks: list,
        vs: list,
        bs: list,
    ) -> tuple:
        """Core WY chunkwise solve, all heads batched.

        M_in: [B, H, hd, hd]; q: [B, T, H, hd]; ks/vs: list of [B, T, H, hd];
        bs: list of [B, T, H, 1].

        Returns (y_raw [B, T, H, hd], M_out [B, H, hd, hd]).

        Math derivation (per (B, H) slice):
            u_t = β_t · (v_t − M_{t-1} · k_t)
            (I + G) U = R   where
              G[t, j] = β_t · (k_t · k_j)   for j < t
              R[t]    = β_t · (v_t − M_in · k_t)
            M_out = M_in + Uᵀ · K
            y[t]  = M_in · q_t + Σ_{s ≤ (t+1)N − 1} (q_t · k_s) · u_s
        """
        M_dtype = M_in.dtype
        B, T, H, hd = q.shape
        N = len(ks)
        TN = T * N

        # Stack into virtual sequence: [B, T, N, H, hd] then transpose
        # head before the virtual time so (B, H) flattens cleanly.
        K_stack = torch.stack(ks, dim=2)
        V_stack = torch.stack(vs, dim=2)
        beta_stack = torch.stack(bs, dim=2)  # [B, T, N, H, 1]

        K_virt = (
            K_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        V_virt = (
            V_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, hd).to(M_dtype)
        )
        beta_virt = (
            beta_stack.permute(0, 3, 1, 2, 4).reshape(B, H, TN, 1).to(M_dtype)
        )
        Q_BH = q.permute(0, 2, 1, 3).to(M_dtype)  # [B, H, T, hd]

        # Flatten (B, H) into the bmm/solve batch dim.
        K_v = K_virt.reshape(B * H, TN, hd)
        V_v = V_virt.reshape(B * H, TN, hd)
        beta_v = beta_virt.reshape(B * H, TN, 1)
        Q_v = Q_BH.reshape(B * H, T, hd)
        M_in_v = M_in.reshape(B * H, hd, hd)

        # Static auxiliary tensors (mask_lt, I_TN, real_mask) depend only
        # on (T, N, device, dtype) — fetch from the module-level cache
        # instead of re-allocating every forward.
        mask_lt, I_TN, real_mask = _chunkwise_aux_tensors(
            K_v.device, T, N, M_dtype,
        )

        # R[b, s] = β_s · (v_s − M_in · k_s)
        Mk_virt = torch.bmm(K_v, M_in_v.transpose(-1, -2))
        R = beta_v * (V_v - Mk_virt)

        # G[b, s, j] = β_s · (k_s · k_j) for j < s (strict lower tri)
        KKt = torch.bmm(K_v, K_v.transpose(-1, -2))
        G = (beta_v * KKt) * mask_lt

        # (I + G) U = R  via unit-lower-triangular solve.
        LhS = I_TN.unsqueeze(0) + G
        U = torch.linalg.solve_triangular(
            LhS, R, upper=False, unitriangular=True,
        )

        # M_out = M_in + Uᵀ K
        M_out_v = M_in_v + torch.bmm(U.transpose(-1, -2), K_v)

        # Reads
        QKt = torch.bmm(Q_v, K_v.transpose(-1, -2))  # [B*H, T, TN]
        A_rv = QKt * real_mask
        y_init = torch.bmm(Q_v, M_in_v.transpose(-1, -2))
        y_acc = torch.bmm(A_rv, U)
        y_raw_v = y_init + y_acc  # [B*H, T, hd]

        # Unflatten back to [B, T, H, hd].
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
