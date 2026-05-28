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

State: a 1-tuple `(M,)` with `M ∈ R^(B, d, d)`. No momentum stack, no
depthwise conv buffer — DeltaProduct's update is closed-form per token.

Interface mirrors `NeuralMemoryModule` so `TitansMAGBlock` can pick
between the two via `config.memory_type` without other changes:
    init_state(B, device) -> state
    forward_chunk(x_chunk, state_in, doc_boundaries) -> (y_mem, new_state)
    step_with_conv(x_t, state) -> (y_t, new_state)
"""

import torch
import torch.nn as nn


class DeltaProductMemory(nn.Module):
    """Single-head DeltaProduct memory, order >= 1.

    For multi-head use, wrap copies of this module via
    `MultiHeadDeltaProduct` (separate file / class).

    Args:
        n_embd: input dim. M lives in R^(n_embd × n_embd).
        order: number of delta sub-steps applied per token (1 = DeltaNet,
            2 = matches Titans per TPTT).
        finetune_mode: when True, `out_scale` initializes to zero so that
            y_mem = 0 at step 0 and the pretrained backbone's residual is
            preserved exactly until training picks up the gate. Mirrors
            NMM's `out_scale` semantics.
        block_size: chunked-update aggregation size for the blockwise
            forward path. `block_size=1` always runs the sequential
            per-token recurrence (paper-strict reference). `block_size>1`
            routes through the blockwise parallel path (added in a
            separate task; raises NotImplementedError until then).
    """

    def __init__(
        self,
        n_embd: int,
        order: int = 2,
        finetune_mode: bool = True,
        block_size: int = 1,
    ):
        super().__init__()
        if order < 1:
            raise ValueError(f"order must be >= 1 (got {order})")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1 (got {block_size})")

        self.n_embd = int(n_embd)
        self.order = int(order)
        self.block_size = int(block_size)
        self.finetune_mode = bool(finetune_mode)

        # One read query per token — the read happens AFTER all N
        # write sub-steps, so a single Q projection suffices.
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)

        # N independent (K, V, β) projection sets — each sub-step writes
        # into M with its own key, value, and write-strength. β passes
        # through a sigmoid at call time (so β ∈ [0, 1]).
        self.k_projs = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(self.order)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(self.order)]
        )
        self.beta_heads = nn.ModuleList(
            [nn.Linear(n_embd, 1, bias=True) for _ in range(self.order)]
        )

        # Output scale gate. Zero-init under finetune_mode so the
        # pretrained backbone's residual is preserved at step 0 (matches
        # NMM `out_scale` convention).
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(n_embd))
        else:
            self.out_scale = nn.Parameter(torch.ones(n_embd))

    def init_state(self, B: int, device) -> tuple:
        """Per-sample-batched zero-initialized M.

        Returns a 1-tuple `(M,)` — kept as a tuple for shape parity with
        the NMM's `(M, S, conv_buf)` triple at the call-site dispatch
        layer (model/nmm.py reset_state / _detach_per_layer dispatch on
        tuple length).
        """
        M = torch.zeros(
            B, self.n_embd, self.n_embd, device=device, dtype=torch.float32,
        )
        return (M,)

    def _project_kvb(self, x_chunk: torch.Tensor):
        """Batched projections for the whole chunk, all order sub-steps.

        Returns:
            q [B, T, d]
            ks: list[order] of [B, T, d] — L2-normalized along last dim
            vs: list[order] of [B, T, d]
            bs: list[order] of [B, T, 1] (after sigmoid)

        Keys are L2-normalized so ||k_t|| = 1 and the Gram matrix K·Kᵀ
        has entries bounded in [-1, 1]. This bounds the spectral norm of
        β·K·Kᵀ inside the chunkwise WY solve and keeps `(I + G)` well-
        conditioned regardless of chunk length. Without this the
        triangular solve diverges at training scale (T·N >> 1) — the
        canonical DeltaNet / DeltaProduct formulation requires it. The
        sequential path also benefits: M's spectral norm stays bounded
        across the chunk.

        Done outside the per-token loop so the loop body only does the
        recurrent state update — every projection is one batched matmul
        across the (B, T) dims.
        """
        q = self.q_proj(x_chunk)
        ks = [
            torch.nn.functional.normalize(self.k_projs[i](x_chunk), dim=-1)
            for i in range(self.order)
        ]
        vs = [self.v_projs[i](x_chunk) for i in range(self.order)]
        bs = [
            torch.sigmoid(self.beta_heads[i](x_chunk))
            for i in range(self.order)
        ]
        return q, ks, vs, bs

    def _forward_chunk_sequential(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries=None,
    ) -> tuple:
        """Per-token recurrent forward — reference correctness path.

        Bit-equivalent to T calls to `step_with_conv`. The blockwise path
        (added separately) is checked against this for correctness.

        Args:
            x_chunk: [B, T, d] — post-LN inputs (ln_nmm applied upstream
                by `TitansMAGBlock`).
            state_in: (M,) 1-tuple with M ∈ [B, d, d].
            doc_boundaries: [B, T] bool or None. When True at (b, t), M[b]
                is reset to zero BEFORE applying token t's update.

        Returns:
            (y_mem [B, T, d], (M_out,)) where M_out is the post-chunk state.
        """
        (M,) = state_in
        B, T, d = x_chunk.shape

        # Run M arithmetic in M's dtype (typically fp32); cast projections
        # in as needed. autocast-aware: under bf16 autocast the Linear
        # projections will be bf16 and we promote on the bmm input.
        M_dtype = M.dtype

        q, ks, vs, bs = self._project_kvb(x_chunk)

        y_steps = []
        for t in range(T):
            # Doc-boundary reset for any batch row that starts a new doc.
            if doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t]  # [B] bool
                if reset_mask.any():
                    M = torch.where(
                        reset_mask.view(B, 1, 1),
                        torch.zeros_like(M),
                        M,
                    )

            # N sequential delta sub-steps. Each uses the M produced by
            # the previous sub-step (order matters).
            for i in range(self.order):
                k_i = ks[i][:, t, :].to(M_dtype)          # [B, d]
                v_i = vs[i][:, t, :].to(M_dtype)          # [B, d]
                beta_i = bs[i][:, t, :].to(M_dtype)       # [B, 1]

                # M · k_i  ->  [B, d, d] @ [B, d, 1] -> [B, d, 1] -> [B, d]
                Mk = torch.bmm(M, k_i.unsqueeze(-1)).squeeze(-1)
                # error = v_i - M·k_i        [B, d]
                err = v_i - Mk
                # rank-1 outer-product update, scaled by β_i:
                #   delta = β_i · err ⊗ k_i        [B, d, d]
                delta = torch.bmm(
                    (beta_i * err).unsqueeze(-1),     # [B, d, 1]
                    k_i.unsqueeze(-2),                 # [B, 1, d]
                )
                M = M + delta

            # Write-then-read: y_t uses M AFTER all N sub-steps.
            q_t = q[:, t, :].to(M_dtype)               # [B, d]
            y_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)  # [B, d]
            y_steps.append(y_t)

        y = torch.stack(y_steps, dim=1)                # [B, T, d]
        # Cast back to chunk's dtype before the out_scale gate (which is
        # in module's parameter dtype, typically fp32 — autocast handles
        # the rest).
        y = y.to(x_chunk.dtype) * self.out_scale.to(x_chunk.dtype)
        return y, (M,)

    def _forward_chunk_blockwise(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries=None,
    ) -> tuple:
        """Chunkwise parallel forward — closed-form WY representation.

        Bit-equivalent to the sequential per-token recurrence: the WY
        solve produces exactly the sequential outputs for the whole
        chunk (L2-normalized K bounds the Gram matrix and keeps the
        triangular solve well-conditioned at any T·N).

        One WY solve per contiguous document segment. With no doc
        boundaries, the whole chunk is solved in a single WY system —
        one large, tensor-core-friendly triangular solve per layer per
        head per forward. With doc boundaries, the chunk is split at
        boundary positions and one WY solve runs per segment, threading
        M between segments with per-row reset where needed.

        Why one solve instead of fixed-size sub-chunks: sub-chunking
        adds Python-loop overhead and replaces one large solve with many
        small ones, with ~6× wall-clock penalty at chunk_size=1024 (the
        smoke-test observed regression). The WY solve's O(T²·N²) cost
        is dominated by the single triangular solve kernel, which
        parallelizes natively on tensor cores; sub-chunking trades that
        for kernel-launch overhead with no compensating throughput win.

        Memory bound (peak): the (I + G) matrix is [B, T·N, T·N] fp32.
        At chunk_size=1024, T·N=2048 → ~16 MB per head per layer per
        batch element — well within budget. Watch this if you raise
        chunk_size past ~4096; we may then need a memory-bounded
        chunked path.
        """
        (M,) = state_in
        B, T, d = x_chunk.shape

        # Project once for the whole chunk.
        q_all, ks_all, vs_all, bs_all = self._project_kvb(x_chunk)

        # Split set: just doc-boundary positions (no fixed-size sub-chunks).
        # 0 and T frame the segments implicitly.
        splits = {0, T}
        if doc_boundaries is not None and doc_boundaries.any():
            any_boundary = doc_boundaries.any(dim=0)  # [T]
            for t in torch.nonzero(any_boundary, as_tuple=False).flatten():
                splits.add(int(t.item()))
        split_list = sorted(splits)

        # Fast path: no doc boundaries → one solve over the whole chunk.
        if len(split_list) == 2:
            y_raw, M_out = self._chunkwise_solve_raw(
                M, q_all, ks_all, vs_all, bs_all,
            )
            y = y_raw.to(x_chunk.dtype) * self.out_scale.to(x_chunk.dtype)
            return y, (M_out,)

        # Slow path: per-document-segment WY solves with M reset at
        # boundary positions for any row that crossed one.
        y_segments = []
        for seg_i in range(len(split_list) - 1):
            t_lo, t_hi = split_list[seg_i], split_list[seg_i + 1]
            if t_hi == t_lo:
                continue
            if t_lo > 0 and doc_boundaries is not None:
                reset_mask = doc_boundaries[:, t_lo]  # [B] bool
                if reset_mask.any():
                    M = torch.where(
                        reset_mask.view(B, 1, 1),
                        torch.zeros_like(M),
                        M,
                    )
            q_seg = q_all[:, t_lo:t_hi, :]
            ks_seg = [k[:, t_lo:t_hi, :] for k in ks_all]
            vs_seg = [v[:, t_lo:t_hi, :] for v in vs_all]
            bs_seg = [b[:, t_lo:t_hi, :] for b in bs_all]
            y_seg_raw, M = self._chunkwise_solve_raw(
                M, q_seg, ks_seg, vs_seg, bs_seg,
            )
            y_segments.append(y_seg_raw)

        y_raw = torch.cat(y_segments, dim=1)
        y = y_raw.to(x_chunk.dtype) * self.out_scale.to(x_chunk.dtype)
        return y, (M,)

    def _chunkwise_solve_raw(
        self,
        M_in: torch.Tensor,
        q: torch.Tensor,
        ks: list,
        vs: list,
        bs: list,
    ) -> tuple:
        """Core WY chunkwise solve — no gate, no dtype cast on output.

        Used by both the no-boundary path (one solve covers the whole
        chunk) and the boundary-aware path (one solve per segment).
        Returns (y_raw [B, T, d], M_out [B, d, d]) — caller composes the
        gate / cast at the end.
        """
        M_dtype = M_in.dtype
        B, T, d = q.shape
        N = len(ks)
        TN = T * N

        # Stack into virtual sequence: virtual position s = N*t + i
        # corresponds to real token t, sub-step i.
        K_stack = torch.stack(ks, dim=2)         # [B, T, N, d]
        V_stack = torch.stack(vs, dim=2)         # [B, T, N, d]
        beta_stack = torch.stack(bs, dim=2)      # [B, T, N, 1]
        K_virt = K_stack.reshape(B, TN, d).to(M_dtype)        # [B, TN, d]
        V_virt = V_stack.reshape(B, TN, d).to(M_dtype)        # [B, TN, d]
        beta_virt = beta_stack.reshape(B, TN, 1).to(M_dtype)  # [B, TN, 1]
        Q = q.to(M_dtype)                                     # [B, T, d]

        # R[b, s] = β_s · (v_s − M_in · k_s)
        # Mk_virt = K_virt @ M_in^T  ->  [B, TN, d]
        Mk_virt = torch.bmm(K_virt, M_in.transpose(-1, -2))
        R = beta_virt * (V_virt - Mk_virt)                    # [B, TN, d]

        # G[b, s, j] = β_s · (k_s · k_j) for j < s (strict lower tri)
        KKt = torch.bmm(K_virt, K_virt.transpose(-1, -2))     # [B, TN, TN]
        # Mask to strict lower triangle then apply β scaling on the s
        # axis (the "row index" of G).
        mask_lt = torch.tril(
            torch.ones(TN, TN, device=K_virt.device, dtype=torch.bool),
            diagonal=-1,
        )
        # KKt is symmetric — masking + β broadcast over rows gives the
        # strict-lower-triangular G we want.
        G = (beta_virt * KKt) * mask_lt  # [B, TN, TN]

        # Solve (I + G) U = R for U via triangular solve.
        I_TN = torch.eye(TN, device=K_virt.device, dtype=M_dtype)
        LhS = I_TN.unsqueeze(0) + G                            # [B, TN, TN]
        U = torch.linalg.solve_triangular(
            LhS, R, upper=False, unitriangular=True,
        )  # [B, TN, d]

        # Final state: M_out = M_in + U^T @ K_virt
        # U^T: [B, d, TN]; K_virt: [B, TN, d]; product: [B, d, d]
        M_out = M_in + torch.bmm(U.transpose(-1, -2), K_virt)

        # Reads — y_t uses M after token t's N writes.
        # y[t] = M_in · q_t + Σ_{s < (t+1)·N} (q_t · k_s) · u_s
        QKt = torch.bmm(Q, K_virt.transpose(-1, -2))           # [B, T, TN]
        # Mask: row t allows columns 0 .. (t+1)·N - 1.
        s_idx = torch.arange(TN, device=Q.device).unsqueeze(0)  # [1, TN]
        t_idx = torch.arange(T, device=Q.device).unsqueeze(1)   # [T, 1]
        real_mask = (s_idx < (t_idx + 1) * N).to(M_dtype)       # [T, TN]
        A_rv = QKt * real_mask                                  # [B, T, TN]

        y_init = torch.bmm(Q, M_in.transpose(-1, -2))           # [B, T, d]
        y_acc = torch.bmm(A_rv, U)                              # [B, T, d]
        y_raw = y_init + y_acc                                  # [B, T, d]

        return y_raw, M_out

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
            x_t: [B, d] — single-token post-LN input.
            state: (M,) — current recurrent state.

        Returns:
            (y_t [B, d], new_state).
        """
        # Reuse sequential path with T=1 to avoid code duplication.
        x_unsq = x_t.unsqueeze(1)  # [B, 1, d]
        y_chunk, new_state = self._forward_chunk_sequential(
            x_unsq, state, doc_boundaries=None,
        )
        return y_chunk.squeeze(1), new_state


class MultiHeadDeltaProduct(nn.Module):
    """N parallel `DeltaProductMemory` heads on `head_dim = n_embd // n_heads`.

    Mirrors `MultiHeadNMM`: splits the input along the last dim into
    n_heads × head_dim, dispatches per-head, concatenates outputs.

    State is `list[n_heads]` of per-head `(M,)` tuples — same nested
    layout the multi-head NMM uses (model/nmm.py `MultiHeadNMM`), so the
    block-level state plumbing handles both polymorphically via tuple-vs-
    list dispatch.

    Why parallel single-head modules rather than a fused multi-head
    matmul: matches the NMM's per-head layout (each head has its own
    Q/K/V projections), keeps per-head DeltaProductMemory the single
    source of truth for the recurrence, and makes
    forward(MultiHeadDeltaProduct) == concat per-head forward(DPM) an
    exact identity that's straightforward to test.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        order: int = 2,
        finetune_mode: bool = True,
        block_size: int = 1,
    ):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1 (got {n_heads})")
        if n_embd % n_heads != 0:
            head_dim = n_embd // n_heads
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads}); "
                f"head_dim would be {head_dim} but {n_heads} * {head_dim} = "
                f"{n_heads * head_dim}, not {n_embd}."
            )
        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = self.n_embd // self.n_heads
        self.order = int(order)
        self.block_size = int(block_size)
        self.finetune_mode = bool(finetune_mode)

        self.heads = nn.ModuleList(
            [
                DeltaProductMemory(
                    n_embd=self.head_dim,
                    order=order,
                    finetune_mode=finetune_mode,
                    block_size=block_size,
                )
                for _ in range(self.n_heads)
            ]
        )

    def init_state(self, B: int, device) -> list:
        """Per-head init states. Returns `list[n_heads]` of `(M,)` tuples."""
        return [h.init_state(B, device) for h in self.heads]

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape last dim into (n_heads, head_dim). Works for [B, T, d]
        or [B, d] inputs (the step path)."""
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    def _merge_heads(self, head_outputs: list) -> torch.Tensor:
        """Concatenate per-head outputs back into a d_model tensor along
        the last dim."""
        return torch.cat(head_outputs, dim=-1)

    def forward_chunk(self, x_chunk, state_in, doc_boundaries=None) -> tuple:
        """Per-head dispatch of `forward_chunk`. doc_boundaries is shared
        across heads (a token-level event is the same for every head)."""
        x_split = self._split_heads(x_chunk)  # [B, T, n_heads, head_dim]
        outputs = []
        new_states = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()  # [B, T, head_dim]
            y_h, state_h = head.forward_chunk(x_h, state_in[i], doc_boundaries)
            outputs.append(y_h)
            new_states.append(state_h)
        return self._merge_heads(outputs), new_states

    def step_with_conv(self, x_t, state) -> tuple:
        """Per-head decode-step dispatch."""
        x_split = self._split_heads(x_t)  # [B, n_heads, head_dim]
        outputs = []
        new_states = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()  # [B, head_dim]
            y_h, s_h = head.step_with_conv(x_h, state[i])
            outputs.append(y_h)
            new_states.append(s_h)
        return self._merge_heads(outputs), new_states
