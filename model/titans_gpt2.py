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

    def prepare_decode(
        self,
        prompt_idx: torch.Tensor,
        initial_nmm_states=None,
    ) -> dict:
        """Warm up on a prompt, return the full DecodeCache for forward_step.

        Runs forward() over the prompt (NMM gets exactly one update per
        prompt token via forward_chunk), then captures per-block
        (k_cache, v_cache, nmm_conv_buffer) by re-projecting the prompt's
        ln_1/ln_nmm outputs. The KV cache for each block includes the
        persistent prefix at positions 0..N_p-1.

        prompt_idx: [B, P] token ids. P must be <= block_size.
        initial_nmm_states: optional list[n_layer] of (M, S) — for long
        prompts where an earlier chunked warm-up established NMM state
        that this prepare_decode call should continue from.

        Returns dict:
          last_logits:  [B, 1, vocab_size]
          nmm_states:   list[n_layer] of (M, S)
          kv_caches:    list[n_layer] of (k_cache, v_cache)
          nmm_conv_buffers: list[n_layer] of dict {'q', 'k', 'v'}
          position:     int — position index of the next decoded token
                        (= prompt length P)
        """
        B, P = prompt_idx.shape
        if P > self.config.block_size:
            raise ValueError(
                f"prompt length {P} exceeds block_size {self.config.block_size}. "
                f"Truncate prompt or use the non-cached generate path."
            )

        pos = torch.arange(0, P, device=prompt_idx.device)
        x = self.drop(self.wte(prompt_idx) + self.wpe(pos))

        if initial_nmm_states is None:
            nmm_states = [
                block.nmm.init_state(B, prompt_idx.device) for block in self.blocks
            ]
        else:
            nmm_states = list(initial_nmm_states)
        kv_caches = []
        nmm_conv_buffers = []
        for block, nmm_state in zip(self.blocks, nmm_states):
            # Capture decode caches BEFORE the block mutates x — they're
            # functions of the block's INPUT, not its output.
            k_cache, v_cache, conv_buf = block.init_decode_cache(x, nmm_state)
            kv_caches.append((k_cache, v_cache))
            nmm_conv_buffers.append(conv_buf)
            x, nmm_state = block(x, nmm_state, None)
            nmm_states[len(kv_caches) - 1] = nmm_state

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        last_logits = logits[:, -1:, :]
        return {
            "last_logits": last_logits,
            "nmm_states": nmm_states,
            "kv_caches": kv_caches,
            "nmm_conv_buffers": nmm_conv_buffers,
            "position": P,
        }

    def forward_step(self, token_id: torch.Tensor, cache: dict) -> tuple:
        """Single-token decode forward.

        token_id: [B, 1] new token id.
        cache: dict from prepare_decode (or from a prior forward_step).

        Returns (logits [B, 1, vocab_size], new_cache).

        Mutates the cache structure (in spirit; returns a new dict). Decode
        position bounded by block_size — wpe lookup goes OOB past that.
        Caller should respect that bound.
        """
        B, T_new = token_id.shape
        assert T_new == 1, f"forward_step expects single token, got T={T_new}"
        pos_idx = cache["position"]
        if pos_idx >= self.config.block_size:
            raise ValueError(
                f"decode position {pos_idx} >= block_size {self.config.block_size}; "
                f"wpe lookup would go out of bounds. Cap max_new_tokens at "
                f"block_size - prompt_len."
            )

        pos = torch.tensor([pos_idx], device=token_id.device)
        x = self.drop(self.wte(token_id) + self.wpe(pos))  # [B, 1, d]

        new_nmm_states = []
        new_kv_caches = []
        new_nmm_conv_buffers = []
        for i, block in enumerate(self.blocks):
            k_cache, v_cache = cache["kv_caches"][i]
            x, nmm_state, k_cache, v_cache, conv_buf = block.forward_step(
                x,
                cache["nmm_states"][i],
                k_cache,
                v_cache,
                cache["nmm_conv_buffers"][i],
            )
            new_nmm_states.append(nmm_state)
            new_kv_caches.append((k_cache, v_cache))
            new_nmm_conv_buffers.append(conv_buf)

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # [B, 1, vocab_size]
        new_cache = {
            "last_logits": logits,
            "nmm_states": new_nmm_states,
            "kv_caches": new_kv_caches,
            "nmm_conv_buffers": new_nmm_conv_buffers,
            "position": pos_idx + 1,
        }
        return logits, new_cache
