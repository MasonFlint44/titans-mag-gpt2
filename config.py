"""TitansConfig dataclass + GPT-2 factory presets."""

import warnings
from dataclasses import dataclass


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
    # nmm_depth is documentation-only: MemoryMLP is hardcoded to L_M=2
    # (W1 + W_gate -> W2). Changing this field has no effect on the running
    # code; to truly change depth, modify MemoryMLP.__init__.
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

    # Memory-saving knobs (G256, G257). Defaults preserve the original
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
    #   Valid values: "fp32", "bf16".
    #
    # nmm_grad_checkpoint: when True, `_forward_chunk_sequential` runs the
    #   per-token inner loop in segments of `nmm_grad_checkpoint_segment_len`
    #   tokens; each segment is wrapped in `torch.utils.checkpoint.checkpoint`
    #   so backward recomputes the inner-loop intermediates instead of
    #   storing them. ~5-10x larger feasible chunk_size at the cost of an
    #   extra forward pass through each segment during backward. Composes
    #   with nmm_state_dtype="bf16" multiplicatively. The scan path
    #   (`_forward_chunk_scan`) ignores this flag — it has a different
    #   memory-vs-compute trade and doesn't share the per-token graph.
    #
    # nmm_grad_checkpoint_segment_len: segment size when grad-checkpointing
    #   is on. Smaller = less peak memory + more recompute; larger = more
    #   peak memory + less recompute. 64 is a reasonable default that
    #   roughly matches "checkpoint every 64 tokens" guidance from other
    #   sequence-model checkpoint implementations.
    nmm_state_dtype: str = "fp32"
    nmm_grad_checkpoint: bool = False
    nmm_grad_checkpoint_segment_len: int = 64

    # Two more memory / compute knobs (G258, G259).
    #
    # nmm_compile_scan_training: opt the per-block NMM into the
    #   `_forward_chunk_scan` path during training. This is the
    #   associative-scan-based parallel path; it pre-computes all T per-
    #   token gradients at chunk-start M_0 (an APPROXIMATION — does NOT
    #   match the paper's M_{t-1}-conditioned gradients). The scan path
    #   needs torch.compile to have autograd; without it autograd is
    #   silently zeroed for the NMM (G164/G180). Setting this flag only
    #   flips `_allow_scan_training` on each block's NMM at construction;
    #   YOU STILL HAVE TO call `torch.compile(model)` yourself to actually
    #   enable the scan path under autograd.
    #   Note: scan is a COMPUTE optimization (parallelism), NOT a memory
    #   optimization. It allocates `[T, B, h, d]` gradient tensors upfront,
    #   which at T=1024 can be larger than the sequential path's per-token
    #   graph — combine with cpu_offload below if VRAM is the bottleneck.
    #
    # nmm_cpu_offload_segments: when True, gradient-checkpoint boundary
    #   (M, S) tensors are stashed on CPU between forward and backward
    #   instead of staying on GPU. Backward moves them back to GPU one
    #   segment at a time, recomputes, and discards. ~20x reduction in
    #   GPU memory used by checkpoint boundaries at the cost of CPU↔GPU
    #   transfer time (PCIe 4.0 x16: ~16 GB/s realistic). Requires
    #   `nmm_grad_checkpoint=True` — without it there are no boundary
    #   tensors to offload. Composes with `nmm_state_dtype="bf16"`
    #   (offloaded tensors are bf16, transfer is half the size).
    nmm_compile_scan_training: bool = False
    nmm_cpu_offload_segments: bool = False

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

        # Memory-saving knob validation (G256 / G257).
        if self.nmm_state_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"nmm_state_dtype must be 'fp32' or 'bf16' (got "
                f"{self.nmm_state_dtype!r}). fp16 is NOT supported — it "
                f"needs loss scaling that this codebase doesn't wire; "
                f"bf16 is the safe choice for halved NMM state memory."
            )
        if self.nmm_grad_checkpoint_segment_len < 1:
            raise ValueError(
                f"nmm_grad_checkpoint_segment_len must be >= 1 (got "
                f"{self.nmm_grad_checkpoint_segment_len})."
            )

        # cpu_offload only makes sense when grad_checkpoint is on — without
        # checkpointing there are no boundary tensors to offload (the full
        # graph is on GPU). Fail loud so callers don't enable cpu_offload
        # alone and wonder why memory didn't drop.
        if self.nmm_cpu_offload_segments and not self.nmm_grad_checkpoint:
            raise ValueError(
                "nmm_cpu_offload_segments=True requires "
                "nmm_grad_checkpoint=True. The CPU-offload only stashes "
                "checkpoint-boundary tensors; with checkpointing off there "
                "are no boundaries to offload (the full per-token graph "
                "lives on GPU). Set nmm_grad_checkpoint=True too, or set "
                "nmm_cpu_offload_segments=False."
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
