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
