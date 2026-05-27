"""TitansConfig dataclass + GPT-2 factory presets."""

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Optional


# Keys removed from TitansConfig over time. `from_dict` silently drops these
# when loading older checkpoints so a refactor that cuts a knob doesn't
# strand users on their old checkpoints. Keys NOT in this set are still
# rejected loudly (typo / schema mismatch). Append to this set as knobs
# are removed; never remove entries.
_REMOVED_CONFIG_KEYS: frozenset = frozenset({})


@dataclass
class TitansConfig:
    # GPT-2 backbone
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    vocab_size: int   = 50257
    block_size: int   = 1024
    dropout:    float = 0.0

    # NMM
    # L_M in the paper. Today only L_M=2 is implemented (single SwiGLU
    # block: silu(W1 x) * sigmoid(W_gate x) -> W2 -> norm -> residual).
    # Wiring up L_M > 2 requires generalizing the analytical-gradient
    # fused kernel (model/nmm_fused.py) and the state-key discovery —
    # non-trivial work for a knob the paper shows gives only marginal
    # gains beyond L_M=2. We keep the field for paper-vocabulary
    # alignment but validate that callers ask for the only value we
    # actually support.
    nmm_depth:         int  = 2
    nmm_expansion:     int  = 4
    nmm_conv_kernel:   int  = 4
    nmm_spectral_norm: bool = True
    nmm_n_persistent:  int  = 4
    chunk_size:        int  = 512

    # Attention
    use_swa:    bool = False
    swa_window: int  = 256

    # Fine-tuning
    finetune_mode: bool = True

    # Paper-strict vs lucidrains flags. Defaults PREFER PAPER for the two
    # paper-vs-lucidrains divergences; single-head NMM is itself paper-
    # aligned so the third flag's default also matches the paper. Flip
    # to False to recover the lucidrains-flavored behavior for ablation.
    #
    # retrieval_from_M_prev: paper Eq. 15 specifies `y_t = M(q_t)` where M
    #   is M_{t-1} (read-then-write). Our default (True) is paper-strict.
    #   Set False for lucidrains "write-then-read" (retrieve from M_t).
    #
    # feed_persistent_to_nmm: paper Eq. 28 specifies `M(x̃)` where x̃ = concat(
    #   persistent, x). Our default (True) is paper-strict — feeds the
    #   persistent-augmented input to NMM, slices the prefix off the output.
    #   Set False for lucidrains-flavored "real tokens only" (NMM receives
    #   ln_nmm(x), persistent tokens never influence the surprise update).
    #
    # nmm_n_heads: NOT in the paper proper (single-head implicit; multi-head
    #   is a lucidrains enhancement). Default 1 = paper-aligned single-head.
    #   Setting >1 instantiates `MultiHeadNMM` (lucidrains-style); must
    #   divide n_embd.
    retrieval_from_M_prev: bool = True
    feed_persistent_to_nmm: bool = True
    nmm_n_heads: int = 1

    # Memory-saving knobs (G256). Defaults preserve the original
    # fp32 / no-checkpoint behavior; flip when you hit OOM training the
    # NMM with realistic chunk sizes.
    #
    # nmm_state_dtype: storage dtype for the recurrent (M, S) tensors and
    #   the per-step update buffers in `_forward_chunk_sequential`. Use
    #   "bf16" to roughly halve per-step state retention (~2x larger
    #   feasible chunk_size on a fixed VRAM budget). NS5 still casts in/out
    #   of fp32 internally (G226 — bf16 NS5 drifts to spectral norm 0.7-1.4
    #   instead of ~1), so this is safe-by-construction; the only drift
    #   risk is the per-step M_t = (1-a)*M_{t-1} + S_t update rounding in
    #   bf16. Measure loss curves before relying on it for full training.
    #   Valid values: "fp32", "bf16", "int8" (int8 requires blockwise; see
    #   the validator in `__post_init__`).
    nmm_state_dtype: str = "fp32"

    # nmm_layer_indices (G261): if not None, NMM is wired only on the
    # listed transformer blocks; other blocks are plain GPT-2 blocks
    # (attn + MLP only, no persistent prefix, no MAG gate). Cuts NMM-
    # related memory/time roughly proportionally — at n_layer=12 with
    # nmm_layer_indices=[0, 4, 8] (3 NMM blocks), the per-step NMM
    # transient drops 4x. Paper applies NMM at every block, so subset
    # is an ablation departure; useful when the alternative is "can't
    # train at all" because of VRAM. None = all blocks have NMM (default).
    nmm_layer_indices: Optional[list] = None

    # nmm_block_size (G266): chunk-as-update aggregation for the NMM inner
    # loop. When >1, the per-token sequential recurrence is REPLACED by a
    # blockwise recurrence: every `nmm_block_size` consecutive tokens
    # produce ONE memory update instead of `nmm_block_size` separate
    # updates. The per-block forward through MemoryMLP runs as a batched
    # matmul over the block's tokens — TC engages at block_size ≥ 16.
    #
    # Default 1 = paper-strict per-token recurrence. Bit-equivalent to the
    # current sequential path at this setting (locked by tests).
    #
    # Approximation cost at block_size > 1:
    # - Within a block: all `nmm_block_size` tokens share the block-start
    #   M for both surprise gradient and retrieval. Paper's per-token
    #   M_{t-1} resolution is replaced by per-block M_{block-1}.
    # - One theta/eta/alpha per block (mean over the block) instead of
    #   per-token. The fine-grained adaptive-rate signal is averaged out.
    #
    # Throughput win (vs sequential at block_size=1) comes from:
    # - Forward matmuls (pre1, preg, y, retrieval) become bmm with N=block_size,
    #   engaging tensor cores. At gpt2_small with block_size=64: ~5-10×
    #   speedup on each matmul.
    # - Gradient accumulation across the block is a single GEMM
    #   (`einsum("bth,btd->bhd", d_pre1, k)`) instead of T outer products.
    #
    # Valid values: any positive int. Must divide `chunk_size` for clean
    # block alignment; validation enforces this.
    #
    # When using `nmm_block_size > 1`, the per-token state-key dimensions
    # (`pre1`, `preg`, `a`) get an extra leading T dim — block-checkpoint /
    # cpu-offload / `compile_inner_loop` paths all compose.
    nmm_block_size: int = 1

    # nmm_detach_state_between_blocks (G268 — lucidrains' `detach_mem_state`):
    # when True AND `nmm_block_size > 1`, the blockwise path detaches (M, S)
    # at the START of each block. Backward graph spans ONE block instead of
    # the full chunk — peak transient memory drops ~proportional to (T /
    # block_size). Standard truncated BPTT trade-off: outer params
    # (k_proj, q_proj, v_proj, W_theta, W_eta, W_alpha, gamma_mem, NMM
    # weight inits) only learn from gradients within a single block; the
    # cross-block "remember earlier in this chunk" signal is lost. Good
    # fit for TITANS' memorize-at-test-time framing, where the inner loop
    # is the load-bearing learner and outer params just need to learn TO
    # memorize effectively.
    #
    # Semantics:
    # - At block boundary, M and S are detached before being fed into the
    #   next block's gradient and retrieval ops. The block produces a new
    #   (M_new, S_new) with a fresh graph rooted at the detached predecessor.
    # - The CHUNK output state (last block's M, S) still has a graph for
    #   that block; only the inter-block links are cut.
    # - At `block_size=1`, the v1 blockwise path with detach effectively
    #   makes the NMM per-token-only-trained from the outer optimizer's
    #   perspective. Usually not what you want — pair with `block_size >= 16`
    #   to keep meaningful per-block training signal.
    #
    # No-op when `block_size = 1` (only one block per chunk, no boundaries
    # to detach at) and when the path isn't blockwise (sequential / scan
    # paths ignore this flag — they have their own memory-management
    # mechanisms via grad_checkpoint / segment boundaries).
    nmm_detach_state_between_blocks: bool = False

    # nmm_lookahead_value (G269 — lucidrains' `store_with_lookahead_value`):
    # when True, the NMM's inner reconstruction loss uses v_{t+1} as the
    # target for token t (predictive) instead of v_t (reconstructive).
    # The inner gradient becomes ∇ℓ(M; k_t, v_{t+1}), so the memory learns
    # to predict the NEXT value-projected token from the current
    # key-projected one — closer to a small inner LM loss than to a
    # key→value reconstruction.
    #
    # Boundary handling: the last token in each chunk has no v_{t+1}
    # within the chunk; we DROP its inner-loss contribution (the update
    # from that final position is skipped — the M and S advance only by
    # the (1-α), η decay terms with zero surprise gradient). This is the
    # cleanest boundary: no cross-chunk peek, no synthetic padding.
    # Cross-chunk semantics: a token's v_{t+1} would, in principle, come
    # from the FIRST token of the next chunk, but threading that across
    # the chunk boundary would require restructuring chunked training to
    # peek ahead. We accept the within-chunk-only semantics, which is
    # what lucidrains does too.
    #
    # Affects: blockwise, sequential, scan paths — value-side shift applied
    # at the chunk level before each path runs. Composes with all other
    # NMM knobs.
    nmm_lookahead_value: bool = False

    # nmm_per_param_lr_modulation (G270 — lucidrains'
    # `per_parameter_lr_modulation`): when True, the data-dependent
    # learning rate θ_t becomes per-state-key instead of a single scalar.
    # `W_theta` projects x_t to K independent scalars (one per recurrent
    # weight key — 3 for full-rank, 6 for low-rank), and each weight's
    # update uses its OWN θ. The downside is K times the W_theta param
    # count (still tiny — a few thousand params total at gpt2_small).
    # The upside is more expressivity: different state keys can adapt at
    # different rates, which can matter when (e.g.) W2 has different
    # gradient scales than W1.
    #
    # Affects all paths: step, step_with_conv, sequential, blockwise,
    # scan. At construction time, `W_theta`'s output dim grows from 1
    # to K = len(state_keys); the per-token θ tensor becomes [B, T, K]
    # and is split per key when applying the update.
    #
    # NOT a no-op when `nmm_n_heads > 1`: each head gets its own
    # per-key W_theta (independently learned per head).
    nmm_per_param_lr_modulation: bool = False

    # nmm_per_head_learned_params (G271 — lucidrains'
    # `per_head_learned_parameters`): when False AND `nmm_n_heads > 1`,
    # the MemoryMLP weights are SHARED across heads instead of replicated.
    # Per-head recurrent state is still independent (each head threads its
    # own (M, S) — required for distinct head outputs), but the INIT M
    # comes from the SAME shared weight matrix, and the LayerNorm /
    # `out_scale` are still per-head.
    #
    # Default True = current behavior (each head fully independent).
    # Setting False reduces per-block NMM parameter count by ~`n_heads`×
    # for the recurrent weight inits. At n_heads=8 with full-rank
    # MemoryMLP that's ~7× param reduction on the inner weights.
    #
    # No-op when `nmm_n_heads = 1` (single-head NMM, no replication to
    # share). Wired through `MultiHeadNMM.__init__`.
    nmm_per_head_learned_params: bool = True

    # nmm_momentum_order (G272 — lucidrains' `momentum_order`): order of
    # the momentum recurrence on S. Default 1 = paper-strict
    # `S_t = η·S_{t-1} - θ·g_t`. With order N > 1, N nested momenta
    # are maintained, each decaying the next; the formula is recursive:
    #   S1_t = η1·S1_{t-1} - θ·g_t
    #   S2_t = η2·S2_{t-1} + S1_t
    #   ...
    #   SN_t = ηN·SN_{t-1} + S{N-1}_t
    #   M_t  = (1-α)·M_{t-1} + SN_t
    # Each level uses an INDEPENDENTLY-learned η projection (W_eta
    # becomes a Linear with output dim N instead of 1).
    #
    # Memory cost: N× the S state. At N=2, the recurrent state doubles
    # in size. Compute is negligible per level (broadcast multiply +
    # add). Use only when stuck on convergence — the paper doesn't
    # endorse this strongly, and lucidrains exposes it as an experimental
    # knob.
    #
    # Affects: every path. Validated to be >= 1.
    nmm_momentum_order: int = 1

    # nmm_ns5_steps: number of Newton-Schulz iterations applied to the
    # surprise gradient before each memory update (paper Eq 16).
    #
    # Default 5 matches Jordan et al. (Muon / nanogpt). The polynomial
    # coefficients (a=3.4445, b=-4.7750, c=2.0315) are TUNED for the
    # fp32 fixed point reached in 5 iterations — at fewer steps, the
    # spectral norm of NS5(g) drifts away from 1, scaling every memory
    # update by the same factor. Concretely (measured at gpt2_small
    # dims on random Gaussian gradients):
    #   - steps=5: |sv_max - 1| ~ 0 (converged, paper-faithful)
    #   - steps=4: |sv_max - 1| ~ 0.12 (~12% LR drift)
    #   - steps=3: |sv_max - 1| ~ 0.20 (~20% LR drift)
    #
    # Speed wins are large at gpt2_small (NS5 fp32 GEMMs are ~75% of
    # CUDA time at block_size=64): steps=4 ≈ -16% step time, steps=3 ≈
    # -33%. But the LR drift is the same scale that broke training
    # under bf16 NS5 (G226), so do NOT lower this without a convergence
    # study on your own data. Sanity bounds: [1, 10].
    nmm_ns5_steps: int = 5

    # nmm_compile_ns5 (G274 — fused Newton-Schulz via torch.compile): when
    # True, every NS5 call in this NMM's forward paths goes through a
    # torch.compile-wrapped variant. The 5 NS5 iterations execute as 10
    # matmuls + 10 elementwise ops; without compile each is a separate
    # CUDA kernel launch (~5-10 μs each), so 50-100 μs of pure launch
    # overhead per NS5 call. Compile collapses this into a single graph.
    #
    # When meaningful:
    # - block_size=1 sequential path WITHOUT nmm_compile_inner_loop
    #   (which already wraps NS5 transitively).
    # - blockwise path: NS5 is called per-block; compile saves the
    #   per-block launch overhead.
    # - per_token_ns5=True (G267): NS5 called per token within a block.
    # - decode-time step_with_conv() where NS5 launch overhead is a
    #   bigger fraction of step time.
    #
    # No effect when nmm_compile_inner_loop=True (compile already
    # transitively traces the NS5 ops). Composable with every other
    # path.
    #
    # First call pays a 1-3 s warm-up cost (Inductor traces the
    # iteration); subsequent calls reuse the cached compiled graph.
    # Module-level singleton cache so all NMMs share one warm-up.
    nmm_compile_ns5: bool = False

    # nmm_use_gram_ns5: when True, replace the stock Newton-Schulz polar
    # decomposition with Tri Dao's Gram-Newton-Schulz (Dao-AILab/
    # gram-newton-schulz). Standard NS5 does 2T rectangular matmuls
    # (T=5 iterations × 2 each); Gram-NS5 does 2 rectangular matmuls
    # total + T iterations on the small n×n Gram matrix. At gpt2_small
    # dims (m=4d=3072, n=d=768, α=4): claimed 42% FLOP reduction,
    # measured 1.17-3.07× wall-clock speedup on a consumer Blackwell
    # (RTX 5070 Ti, sm_120). The library also runs the iteration in
    # fp16 (better than TF32 tensor-core throughput).
    #
    # Convergence: the library uses Polar Express coefficients (different
    # per iteration) with a restart at iteration 2 to reset Gram-matrix
    # drift. On random Gaussian inputs, |sv - 1| ≈ 0.12-0.15 — comparable
    # to stock NS5-steps=5. Authors claim perplexity preserved within
    # 0.01 on trillion-parameter Muon training.
    #
    # Optional dependency: install via `pip install
    # 'titans-mag-gpt2[gram_ns5]'` or `pip install gram-newton-schulz`.
    # Targets Hopper / Blackwell GPUs and requires PyTorch 2.7+, CUDA 12.9+.
    # Construction-time flag, NOT runtime — set at config time so we
    # can build the Gram-NS module once and reuse it across all NS calls.
    #
    # When this flag is set, `nmm_ns5_steps` is IGNORED (Gram-NS5 has its
    # own per-iteration coefficient table) and `nmm_compile_ns5` is
    # redundant (Gram-NS5 has its own optimized kernels).
    nmm_use_gram_ns5: bool = False

    # nmm_per_token_ns5 (G267): when True AND `nmm_block_size > 1`, the
    # blockwise path uses per-token NS5 + per-token θ weighting, matching
    # paper Eq 16's `Σ_t θ_t · NS5(∇_t)` exactly. Default False, which
    # uses v1's `θ_mean · NS5(Σ_t ∇_t)` simplification.
    #
    # Why this matters: putting θ INSIDE the aggregation pre-NS5 (a
    # tempting "cheap" approach) does NOT preserve θ's effect — NS5
    # normalises the Frobenius norm, cancelling any positive scalar
    # applied before it. To get per-token θ to actually weight the
    # update, NS5 must be applied PER TOKEN.
    #
    # Cost: stores per-token gradient tensors of shape [B, block, ...]
    # per state-key for the duration of one block's forward. At
    # gpt2_small full-rank, block_size=64: ~1.8 GiB per layer; at
    # low_rank=64: ~300 MiB per layer. Reach for `nmm_low_rank` if
    # memory is tight.
    #
    # At block_size=1, per_token_ns5 is a no-op (single-token block has
    # θ_mean = θ_t and per-token NS5 = single NS5).
    nmm_per_token_ns5: bool = False

    # nmm_softclamp_max (G265): when set to a float, applies tanh-based
    # soft norm clamping to the per-token surprise gradient BEFORE NS5.
    # Smooth analog of hard clip_grad_norm — never has zero gradient,
    # never has a discontinuous threshold. Off (None) by default since
    # paper-strict NS5 alone suffices when the inner loss is well-behaved.
    # Enable to add a safety net when training a fresh-init model where
    # the first few steps' gradients can spike (NS5 input far from its
    # convergence regime). Reasonable values: 5.0 to 20.0 (matches
    # lucidrains/titans-pytorch default of ~5).
    nmm_softclamp_max: Optional[float] = None

    # nmm_compile_inner_loop (G264a): wrap `_run_inner_loop` in
    # `torch.compile(mode="default")`. This is the dominant speedup mechanism
    # — at gpt2_small, T=256, low_rank=64 it cuts step time ~1.8x (40s -> 22s
    # on RTX 5070 Ti). Inductor traces the Python time loop and fuses
    # adjacent ops into batched Triton kernels, eliminating the per-token
    # dispatch overhead that dominates the reference path. Composes with
    # nmm_fused_kernel for an extra ~5% on top of compile alone (the
    # analytical-gradient ops trace more cleanly than `torch.func.grad`).
    #
    # Why not mode="reduce-overhead" (which enables cudagraphs): the inner
    # loop builds new dicts each step (`{k: a + b for k, b in ...}`) which
    # cudagraph's "no input-buffer overwrite" contract rejects without
    # explicit `torch.compiler.cudagraph_mark_step_begin()` calls per step.
    # Default mode is the safe + always-works choice; future work could
    # rewrite the loop to use pre-allocated buffers for reduce-overhead.
    nmm_compile_inner_loop: bool = False

    # nmm_fused_kernel (G264): swap the per-token NMM inner update from the
    # reference `vmap(grad(...)) + newton_schulz5` path to a fused
    # implementation that uses analytical inner gradients (no `torch.func.grad`
    # overhead) and an analytical NS5 backward. Optionally dispatches into a
    # Triton kernel when CUDA is available, otherwise falls back to the
    # analytical-PyTorch path (still ~5-20x faster than the reference).
    #
    # The reference path remains the correctness oracle — equivalence tests
    # in `tests/unit/test_nmm_fused.py` lock the numerical match to within
    # fp32/bf16 tolerance. Default False so users opt into the optimization
    # explicitly; flip to True for any T >= 256 training run.
    #
    # Composes with every existing memory knob (nmm_low_rank,
    # nmm_state_dtype="bf16", nmm_layer_indices).
    nmm_fused_kernel: bool = False

    # nmm_low_rank (G262): factor the MemoryMLP weights as A @ B with an
    # intermediate dim of `nmm_low_rank`. At gpt2_small d=768, default
    # expansion=4: full-rank state per layer is 3 × [4d, d] = ~28 MB bf16;
    # low-rank state per layer is 3 × ([4d, r] + [r, d]) = 3 × r × 5d.
    # At r=64: ~7.5 MB per layer per token, ~10x smaller. Most impactful
    # knob for per-step NMM memory at long T. NS5 still works on the
    # factored matrices (it transposes tall rectangles). The factored
    # MLP has slightly different expressiveness; treat as an ablation
    # and measure loss curves vs full-rank baseline. None = full-rank
    # (default, paper-faithful).
    nmm_low_rank: Optional[int] = None

    def __post_init__(self):
        # raise ValueError (never assert): `python -O` strips asserts, which
        # would let invalid configs ship silently in production.

        if self.chunk_size > self.block_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be <= block_size "
                f"({self.block_size}); otherwise wpe(pos) at training time "
                f"goes out of bounds."
            )

        if self.nmm_n_persistent < 0:
            raise ValueError(
                f"nmm_n_persistent must be >= 0 (got {self.nmm_n_persistent})"
            )

        if self.nmm_expansion < 1:
            raise ValueError(
                f"nmm_expansion must be >= 1 (got {self.nmm_expansion}); "
                f"MemoryMLP needs a hidden dim."
            )

        # L_M only supports the paper's recommended value of 2 today.
        # Wiring up L_M > 2 requires generalizing the analytical-gradient
        # kernel + Triton kernels — see `model/nmm_fused.py` and
        # `model/nmm.py::MemoryMLP`. Reject other values loudly rather
        # than silently no-op (a previous failure mode of this field).
        if self.nmm_depth != 2:
            raise ValueError(
                f"nmm_depth={self.nmm_depth} is not supported. Only L_M=2 "
                f"(single SwiGLU block) is implemented today; the paper's "
                f"L_M ablations beyond 2 give only marginal gains. To add "
                f"support, generalize `MemoryMLP` and the analytical-gradient "
                f"kernel in `model/nmm_fused.py`."
            )

        if self.n_embd % self.n_head != 0:
            head_dim = self.n_embd // self.n_head
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head "
                f"({self.n_head}); head_dim would be {head_dim} but "
                f"{self.n_head} * {head_dim} = {self.n_head * head_dim}, "
                f"not {self.n_embd}."
            )

        # swa_window=0 makes the real-to-real mask all -inf -> softmax NaN
        # at step 0, surfacing as a confusing "loss is NaN" with no other signal.
        if self.use_swa and self.swa_window < 1:
            raise ValueError(
                f"swa_window must be >= 1 when use_swa=True "
                f"(got swa_window={self.swa_window}); empty window "
                f"yields softmax NaN."
            )

        if self.nmm_n_heads < 1:
            raise ValueError(
                f"nmm_n_heads must be >= 1 (got {self.nmm_n_heads}); "
                f"use 1 for single-head NMM (current default behavior)."
            )

        if self.n_embd % self.nmm_n_heads != 0:
            head_dim = self.n_embd // self.nmm_n_heads
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by nmm_n_heads "
                f"({self.nmm_n_heads}); head_dim would be {head_dim} but "
                f"{self.nmm_n_heads} * {head_dim} = "
                f"{self.nmm_n_heads * head_dim}, not {self.n_embd}."
            )

        # Memory-saving knob validation (G256 / G257 / G275).
        if self.nmm_state_dtype not in ("fp32", "bf16", "int8"):
            raise ValueError(
                f"nmm_state_dtype must be 'fp32', 'bf16', or 'int8' (got "
                f"{self.nmm_state_dtype!r}). fp16 is NOT supported — it "
                f"needs loss scaling that this codebase doesn't wire. "
                f"bf16 halves state memory vs fp32; int8 quarters it (G275)."
            )
        # G275 int8 state: only the blockwise path supports it.
        if self.nmm_state_dtype == "int8" and self.nmm_block_size <= 1:
            raise ValueError(
                "nmm_state_dtype='int8' requires nmm_block_size > 1. The "
                "sequential per-token path would dequantize/requantize on "
                "every step. Use the blockwise path "
                "(block_size >= 16 recommended for TC engagement)."
            )
        # nmm_layer_indices: must reference valid block indices, no dupes.
        if self.nmm_layer_indices is not None:
            if not isinstance(self.nmm_layer_indices, (list, tuple)):
                raise ValueError(
                    f"nmm_layer_indices must be a list/tuple of ints or None "
                    f"(got {type(self.nmm_layer_indices).__name__})."
                )
            idx_list = list(self.nmm_layer_indices)
            if len(set(idx_list)) != len(idx_list):
                raise ValueError(
                    f"nmm_layer_indices has duplicate entries: {idx_list}."
                )
            for i in idx_list:
                if not isinstance(i, int):
                    raise ValueError(
                        f"nmm_layer_indices entries must be ints (got {i!r})."
                    )
                if i < 0 or i >= self.n_layer:
                    raise ValueError(
                        f"nmm_layer_indices entry {i} out of range "
                        f"[0, {self.n_layer}). With n_layer={self.n_layer}, "
                        f"valid indices are 0..{self.n_layer - 1}."
                    )
            # Normalize to sorted list for deterministic iteration order.
            self.nmm_layer_indices = sorted(idx_list)

        # nmm_low_rank: must be a positive int; reject ranks >= d_model
        # (low-rank wouldn't actually be low-rank — would have MORE params
        # than full-rank because of the doubled matmul).
        if self.nmm_low_rank is not None:
            if not isinstance(self.nmm_low_rank, int) or self.nmm_low_rank < 1:
                raise ValueError(
                    f"nmm_low_rank must be a positive int or None "
                    f"(got {self.nmm_low_rank!r})."
                )
            # Threshold: low-rank with r >= d_model is wasteful.
            # At r = d_model the factored form has more params than full-rank
            # (each [4d, d] becomes [4d, d] + [d, d] = 5d² > 4d²).
            if self.nmm_low_rank >= self.n_embd:
                raise ValueError(
                    f"nmm_low_rank={self.nmm_low_rank} >= n_embd={self.n_embd}. "
                    f"At this rank the factored form has MORE parameters than "
                    f"full-rank, defeating the purpose. Use r << d_model "
                    f"(e.g. 32, 64, 128 for d=768)."
                )

        # nmm_block_size: must be a positive int that divides chunk_size.
        if not isinstance(self.nmm_block_size, int) or self.nmm_block_size < 1:
            raise ValueError(
                f"nmm_block_size must be a positive int (got "
                f"{self.nmm_block_size!r}). Use 1 for paper-strict "
                f"per-token recurrence; larger for blockwise (approximate)."
            )
        # NOTE on alignment: we DO NOT require chunk_size % block_size == 0.
        # The NMM sees `chunk_size + nmm_n_persistent` tokens per call (the
        # persistent prefix is prepended), so requiring strict alignment at
        # the config level would force users to do off-by-n_persistent math
        # to pick a valid block_size. Instead, `_forward_chunk_blockwise`
        # handles a possibly-smaller TRAILING block gracefully — the math
        # is correct for any block size, just less TC-efficient on that
        # trailing block. Common configs (block_size=64, chunk_size=1024,
        # n_persistent=4) give 16 full blocks of 64 + 1 trailing block of 4,
        # which is fine.
        # nmm_momentum_order: must be a positive int. 1 = paper-default; >1
        # enables higher-order momentum (G272).
        if not isinstance(self.nmm_momentum_order, int) or self.nmm_momentum_order < 1:
            raise ValueError(
                f"nmm_momentum_order must be a positive int (got "
                f"{self.nmm_momentum_order!r}). Use 1 for the paper-default "
                f"first-order momentum; >1 stacks additional momentum levels."
            )

        # nmm_detach_state_between_blocks only meaningful when blockwise (G268).
        # Fail loud when set without blockwise — silent no-op would mislead
        # users into thinking they enabled truncated BPTT.
        if self.nmm_detach_state_between_blocks and self.nmm_block_size <= 1:
            raise ValueError(
                "nmm_detach_state_between_blocks=True requires "
                "nmm_block_size > 1. The blockwise path is the only one that "
                "exposes block boundaries to detach at; with block_size=1 the "
                "blockwise path isn't taken (sequential is) and this flag "
                "would be a silent no-op. Set nmm_block_size >= 16 (for TC "
                "engagement) or set nmm_detach_state_between_blocks=False."
            )

        # nmm_compile_inner_loop / nmm_fused_kernel only affect
        # `_run_inner_loop`, which is the SEQUENTIAL path (`block_size=1`).
        # On the blockwise path (`_forward_chunk_blockwise`) they're silent
        # no-ops — bench-confirmed: all four combinations within 1% at
        # block_size=64. The compile flag also costs ~20s of warm-up on the
        # first step. Warn (not raise) so existing configs keep working;
        # users can drop the flags or move to the sequential path.
        if self.nmm_block_size > 1:
            wasted = []
            if self.nmm_compile_inner_loop:
                wasted.append("nmm_compile_inner_loop")
            if self.nmm_fused_kernel:
                wasted.append("nmm_fused_kernel")
            if wasted:
                warnings.warn(
                    f"{', '.join(wasted)} only affect the sequential path "
                    f"(nmm_block_size=1) — at nmm_block_size={self.nmm_block_size} "
                    f"the blockwise path is taken and these flag(s) have no "
                    f"effect. Drop them to skip the torch.compile warm-up cost.",
                    UserWarning,
                    stacklevel=2,
                )

        # nmm_per_head_learned_params=False meaningless at n_heads=1: nothing
        # to share. Loud rather than silent.
        if (not self.nmm_per_head_learned_params) and self.nmm_n_heads <= 1:
            raise ValueError(
                "nmm_per_head_learned_params=False requires nmm_n_heads > 1. "
                "Sharing learned parameters across heads is only meaningful "
                "when there are multiple heads; with n_heads=1 there's nothing "
                "to share. Either set nmm_n_heads > 1 to use multi-head NMM, "
                "or leave nmm_per_head_learned_params=True (default)."
            )

        # nmm_use_gram_ns5: warn about ignored knobs so the user knows
        # their nmm_ns5_steps / nmm_compile_ns5 settings don't apply.
        if self.nmm_use_gram_ns5:
            redundant = []
            if self.nmm_ns5_steps != 5:
                redundant.append(f"nmm_ns5_steps={self.nmm_ns5_steps}")
            if self.nmm_compile_ns5:
                redundant.append("nmm_compile_ns5=True")
            if redundant:
                warnings.warn(
                    f"nmm_use_gram_ns5=True overrides {', '.join(redundant)} — "
                    f"Gram-NS5 has its own per-iteration coefficient table "
                    f"and optimized kernels. Drop those knobs to silence "
                    f"this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        # nmm_ns5_steps: paper default is 5; sanity bounds [1, 10].
        if not isinstance(self.nmm_ns5_steps, int) or not (1 <= self.nmm_ns5_steps <= 10):
            raise ValueError(
                f"nmm_ns5_steps must be an int in [1, 10] (got "
                f"{self.nmm_ns5_steps!r}). Default is 5 (paper-faithful, "
                f"Jordan/Muon coefficients tuned for this fixed point). "
                f"Lower values speed up training but the spectral norm of "
                f"NS5(g) drifts away from 1, scaling every memory update."
            )

        # nmm_softclamp_max: must be positive if set.
        if self.nmm_softclamp_max is not None:
            if (not isinstance(self.nmm_softclamp_max, (int, float))
                or self.nmm_softclamp_max <= 0):
                raise ValueError(
                    f"nmm_softclamp_max must be a positive float or None "
                    f"(got {self.nmm_softclamp_max!r}). Use None to disable, "
                    f"or e.g. 5.0 for lucidrains-style soft norm clamping."
                )

        # From-scratch with chunk_size < block_size leaves wpe rows above
        # chunk_size untrained -- generation beyond chunk_size hits random
        # position embeddings and silently degrades. Fine-tune path is immune
        # because load_pretrained overwrites wpe with HF's trained 1024-row table.
        if not self.finetune_mode and self.chunk_size < self.block_size:
            warnings.warn(
                f"From-scratch training (finetune_mode=False) with "
                f"chunk_size={self.chunk_size} < block_size={self.block_size}: "
                f"wpe.weight[{self.chunk_size}:{self.block_size}] will never "
                f"be trained. Generation at context > {self.chunk_size} will "
                f"access untrained position embeddings.",
                UserWarning,
                stacklevel=2,
            )

    # Forward-compat loader for saved checkpoints. `save_checkpoint` writes
    # `dataclasses.asdict(config)` (a dict with every current field). When
    # a later schema removes a field, the dict still carries it — and
    # `TitansConfig(**ckpt["config"])` raises TypeError on the unknown key.
    # `from_dict` accepts the dict, silently drops keys listed in
    # `_REMOVED_CONFIG_KEYS` (known intentional removals), and raises loud
    # on any OTHER unknown key (likely typo or genuine schema mismatch).
    @classmethod
    def from_dict(cls, d: dict) -> "TitansConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - valid
        bad = unknown - _REMOVED_CONFIG_KEYS
        if bad:
            raise TypeError(
                f"TitansConfig.from_dict: unrecognized config keys "
                f"{sorted(bad)}. Either the saved checkpoint comes from "
                f"an incompatible build, or someone misspelled a field. "
                f"Known-deprecated keys that ARE silently dropped on load: "
                f"{sorted(_REMOVED_CONFIG_KEYS)}."
            )
        return cls(**{k: v for k, v in d.items() if k in valid})

    # Factory methods. Merging via **{**defaults, **overrides} (not fixed
    # kwargs + **overrides) lets callers override backbone dims without
    # hitting "TypeError: multiple values for keyword argument".
    @classmethod
    def gpt2_small(cls, **overrides):
        return cls(**{**dict(n_layer=12, n_head=12, n_embd=768),  **overrides})

    @classmethod
    def gpt2_medium(cls, **overrides):
        return cls(**{**dict(n_layer=24, n_head=16, n_embd=1024), **overrides})

    @classmethod
    def gpt2_large(cls, **overrides):
        return cls(**{**dict(n_layer=36, n_head=20, n_embd=1280), **overrides})

    @classmethod
    def gpt2_xl(cls, **overrides):
        return cls(**{**dict(n_layer=48, n_head=25, n_embd=1600), **overrides})
