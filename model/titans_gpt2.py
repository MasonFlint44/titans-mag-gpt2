"""TitansMAGGPT2: full model with HF GPT-2 weight-loading hook."""

import math

import torch
import torch.nn as nn

from model.block import TitansMAGBlock


class TitansMAGGPT2(nn.Module):
    """Full TITANS MAG model on a GPT-2 backbone.

    Architecture: wte + wpe -> dropout -> N x TitansMAGBlock -> ln_f -> tied LM head.

    forward(idx, nmm_states=None, doc_boundaries=None) -> (logits, new_nmm_states)
      - nmm_states=None: each block's NMM is initialized from memory_mlp.W*.weight.
      - new_nmm_states is the per-layer list of (M, S) after this chunk; the caller
        feeds it into the next chunk (TBPTT) after `detach_states` between chunks.

    Final ln_f BEFORE the LM head matches HF GPT-2 — omitting it produces unnormalized
    logits and breaks weight-load parity even with correct weights.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TitansMAGBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        # GPT-2 init must run AFTER block construction so named_modules sees everything.
        # In finetune mode this gets overwritten by load_pretrained; running it
        # unconditionally keeps construction deterministic and produces a trainable
        # model even without the HF load.
        self._apply_gpt2_init()

    def _apply_gpt2_init(self):
        """GPT-2-style backbone init.

        - Default nn.Embedding init is N(0, 1) — 50x too large for a transformer.
          Without overriding, initial logits have std ~sqrt(n_embd), softmax
          collapses to ~one-hot, and from-scratch training is wildly unstable.
        - Default nn.Linear init varies with fan_in; explicit N(0, 0.02) keeps
          everything on the same scale.
        - Output projections in residual blocks are further scaled by
          1/sqrt(2*n_layer) so residual stream variance doesn't grow with depth.

        NMM-internal modules are skipped by IDENTITY (not name substring). A
        substring check would silently miss them if a future refactor renamed
        `self.nmm` to `self.memory`. Identity is invariant to renames.

        Relative `from .nmm import ...` (not `from model.nmm`) so the import
        survives any top-level package rename.
        """
        from .nmm import NeuralMemoryModule

        n_layer = len(self.blocks)

        nmm_internal_ids = set()
        for module in self.modules():
            if isinstance(module, NeuralMemoryModule):
                for sub in module.modules():
                    nmm_internal_ids.add(id(sub))

        for name, module in self.named_modules():
            if id(module) in nmm_internal_ids:
                continue
            if "ln_nmm" in name:
                # LayerNorm default (weight=1, bias=0) is already what we want.
                continue
            if isinstance(module, nn.Linear):
                # Output projections (attn.proj, mlp.c_proj) get residual scaling.
                # Note: NMMProjection's k/q/v_proj end with "_proj" but are skipped
                # above via the NMM id set; here `.proj` matches only attn.proj.
                if name.endswith(".proj") or name.endswith(".c_proj"):
                    std = 0.02 / math.sqrt(2 * n_layer)
                else:
                    std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, nmm_states=None, doc_boundaries=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        if nmm_states is None:
            nmm_states = [
                block.nmm.init_state(B, idx.device) for block in self.blocks
            ]

        new_nmm_states = []
        for block, nmm_state in zip(self.blocks, nmm_states):
            x, nmm_state = block(x, nmm_state, doc_boundaries)
            new_nmm_states.append(nmm_state)

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # tied weights
        return logits, new_nmm_states
