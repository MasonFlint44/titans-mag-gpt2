"""TitansMAGGPT2: full model with HF GPT-2 weight-loading hook."""

import math

import torch
import torch.nn as nn

from model.block import PlainGPT2Block, TitansMAGBlock


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

        # Paper Eq. 19 persistent-memory prefix at the model level. One set
        # of learned tokens prepended ONCE to wte+wpe; persistent positions
        # then propagate through every block's residual stream and pick up
        # information from real tokens at each layer. Sliced off before
        # ln_f / LM head so logits are only over real-token positions.
        # In "per_block" mode this is absent and each block owns its own
        # smaller prefix.
        self._model_wide_persistent = (
            config.persistent_prefix_mode == "model_wide"
        )
        if self._model_wide_persistent:
            self.persistent_mem = nn.Parameter(
                torch.randn(config.nmm_n_persistent, config.n_embd) * 0.02
            )

        # Determine which blocks get the full TitansMAGBlock (with NMM /
        # persistent / MAG gate) vs. PlainGPT2Block (attn + MLP only).
        # nmm_layer_indices=None means every block has NMM (default,
        # paper-faithful). When set, only listed indices get NMM. G261.
        if config.nmm_layer_indices is None:
            nmm_idx_set = set(range(config.n_layer))
        else:
            nmm_idx_set = set(config.nmm_layer_indices)
        self.blocks = nn.ModuleList([
            TitansMAGBlock(config) if i in nmm_idx_set else PlainGPT2Block(config)
            for i in range(config.n_layer)
        ])
        # Cache the boolean mask for fast per-block dispatch in forward.
        self._block_has_nmm = [i in nmm_idx_set for i in range(config.n_layer)]

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
            # NOTE: no explicit ln_nmm skip needed — LayerNorm matches neither
            # nn.Linear nor nn.Embedding below, so it's left at the default
            # (weight=1, bias=0) by virtue of the isinstance filter. Same is
            # true for ln_1, ln_2, ln_f — all correctly skipped implicitly.
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

        # Model-wide persistent prefix (paper Eq. 19): prepend ONCE here.
        # Each block sees the augmented sequence; the prefix accumulates
        # information from real tokens at every layer via the residual
        # stream. doc_boundaries is augmented in parallel with a False
        # prefix so persistent positions never trigger NMM resets.
        N_p = self.config.nmm_n_persistent
        if self._model_wide_persistent and N_p > 0:
            persistent = self.persistent_mem.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([persistent, x], dim=1)
            if doc_boundaries is not None:
                pad = torch.zeros(B, N_p, dtype=torch.bool, device=idx.device)
                doc_boundaries = torch.cat([pad, doc_boundaries], dim=1)

        if nmm_states is None:
            # Plain blocks have no NMM — their slot is None. The per-block
            # forward signature is uniform; PlainGPT2Block ignores the state.
            nmm_states = [
                block.nmm.init_state(B, idx.device) if has_nmm else None
                for block, has_nmm in zip(self.blocks, self._block_has_nmm)
            ]

        new_nmm_states = []
        for block, nmm_state in zip(self.blocks, nmm_states):
            x, nmm_state = block(x, nmm_state, doc_boundaries)
            new_nmm_states.append(nmm_state)

        # Slice persistent positions off before ln_f / LM head — logits
        # are only over real-token positions.
        if self._model_wide_persistent and N_p > 0:
            x = x[:, N_p:, :]

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # tied weights
        return logits, new_nmm_states

    def prepare_decode(
        self,
        prompt_idx: torch.Tensor,
        initial_nmm_states=None,
        int8_kv_cache: bool = False,
    ) -> dict:
        """Warm up on a prompt, return the full DecodeCache for forward_step.

        Runs forward() over the prompt (NMM gets exactly one update per
        prompt token via forward_chunk), then captures per-block
        (k_cache, v_cache, nmm_conv_buffer) by re-projecting the prompt's
        ln_1/ln_nmm outputs. The KV cache for each block includes the
        persistent prefix at positions 0..N_p-1.

        prompt_idx: [B, P] token ids. P must be <= block_size.
        initial_nmm_states: optional list[n_layer] of (M, S, conv_buf) —
        for long prompts where an earlier chunked warm-up established NMM
        state that this prepare_decode call should continue from.

        Returns dict:
          last_logits:  [B, 1, vocab_size]
          nmm_states:   list[n_layer] of (M, S, conv_buf) (item 6 —
                        conv buffer is folded into per-layer state)
          kv_caches:    list[n_layer] of (k_cache, v_cache)
          position:     int — position index of the next decoded token
                        (= prompt length P)
        """
        # Eval-mode is part of the contract. In train mode with dropout > 0,
        # init_decode_cache captures K, V from project_kv (no dropout), but
        # subsequent forward_step's `forward_with_kv_cache` hardcodes
        # `dropout_p=0` and skips resid_dropout — while the warm-up block
        # forward DOES apply both dropouts. The two paths' attention outputs
        # then differ, and any decode-vs-full-forward parity invariant breaks
        # silently. (mlp.dropout also still fires at decode but not at the
        # K, V capture, compounding the divergence.) Forcing eval here keeps
        # the "decode matches a single full forward" invariant a hard guarantee
        # rather than a hidden precondition.
        if self.training:
            raise RuntimeError(
                "prepare_decode requires model.eval() mode. In train mode "
                "with dropout > 0 the cached path's attention output differs "
                "from a full forward's (decode skips resid_dropout / SDPA "
                "dropout while warm-up applies them), silently breaking the "
                "decode-vs-full-forward parity invariant. Call model.eval() "
                "first, or use generate()/needle_in_haystack() which manage "
                "the mode for you (G243)."
            )
        B, P = prompt_idx.shape
        if P > self.config.block_size:
            raise ValueError(
                f"prompt length {P} exceeds block_size {self.config.block_size}. "
                f"Truncate prompt or use the non-cached generate path."
            )

        pos = torch.arange(0, P, device=prompt_idx.device)
        x = self.drop(self.wte(prompt_idx) + self.wpe(pos))

        # Model-wide persistent prefix: prepend ONCE at the model level so
        # every block sees the augmented sequence. The KV caches captured
        # by `init_decode_cache` will then include the persistent positions
        # naturally; `forward_step` only feeds new real tokens.
        N_p = self.config.nmm_n_persistent
        if self._model_wide_persistent and N_p > 0:
            persistent = self.persistent_mem.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([persistent, x], dim=1)

        if initial_nmm_states is None:
            nmm_states = [
                block.nmm.init_state(B, prompt_idx.device) if has_nmm else None
                for block, has_nmm in zip(self.blocks, self._block_has_nmm)
            ]
        else:
            nmm_states = list(initial_nmm_states)
            # Without this check, zip(self.blocks, nmm_states) below silently
            # iterates to the shorter list — leaving the cache with fewer
            # per-block entries than n_layer. The first forward_step then
            # IndexErrors deep in the per-block loop, pointing at the wrong
            # call site. Validate eagerly so the error surfaces here.
            if len(nmm_states) != len(self.blocks):
                raise ValueError(
                    f"initial_nmm_states length {len(nmm_states)} does not "
                    f"match model n_layer {len(self.blocks)}. Each block "
                    f"needs its own (M, S) pair; pass the full per-layer "
                    f"list returned by an earlier forward() / prepare_decode()."
                )
            # Same eager-failure rationale (G244): a mismatched B between the
            # passed nmm_states and the prompt produces a deep, confusing
            # shape error inside the first block's NMM forward. Find the
            # first non-None layer state to check batch dim (G261: with
            # nmm_layer_indices, some entries are None).
            # Multi-head (G254): each non-None entry is a list-of-states.
            for first_layer in nmm_states:
                if first_layer is None:
                    continue
                if isinstance(first_layer, list):
                    first_layer_M = first_layer[0][0]
                else:
                    first_layer_M = first_layer[0]
                any_W = next(iter(first_layer_M.values()))
                if any_W.shape[0] != B:
                    raise ValueError(
                        f"initial_nmm_states batch dim {any_W.shape[0]} does not "
                        f"match prompt batch dim {B}. The state's B must equal "
                        f"the prompt's leading dim; rebuild the state with "
                        f"nmm.init_state(B={B}, ...) or pass a prompt whose "
                        f"leading dim matches the state's."
                    )
                break  # one non-None layer is enough to verify B
        kv_caches = []
        for block, nmm_state in zip(self.blocks, nmm_states):
            # Capture KV cache BEFORE the block mutates x — the cache is
            # a function of the block's INPUT, not its output. NMM conv
            # buffer is part of nmm_state now (item 6); it gets rolled
            # forward naturally by `block(x, nmm_state, None)`.
            k_cache, v_cache = block.init_decode_cache(
                x, nmm_state, int8_kv_cache=int8_kv_cache,
            )
            kv_caches.append((k_cache, v_cache))
            x, nmm_state = block(x, nmm_state, None)
            nmm_states[len(kv_caches) - 1] = nmm_state

        # Slice persistent positions off before ln_f (model_wide only;
        # per_block mode has already produced real-token-only output).
        if self._model_wide_persistent and N_p > 0:
            x = x[:, N_p:, :]
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        last_logits = logits[:, -1:, :]
        return {
            "last_logits": last_logits,
            "nmm_states": nmm_states,
            "kv_caches": kv_caches,
            "position": P,
        }

    def prepare_decode_chunked(
        self,
        prompt_idx: torch.Tensor,
        initial_nmm_states=None,
        int8_kv_cache: bool = False,
    ) -> dict:
        """Prepare a decode cache for ANY prompt length (short or long).

        Encapsulates the chunked-warm-up + tail-prepare_decode pipeline that
        previously lived inline in `generate.py`, `eval.needle_in_haystack`,
        and `tests/behavior/test_cached_generate_parity.py` (G249 — three-way
        DRY violation, future-divergence risk).

        Behavior:
          - prompt_len <= block_size: equivalent to `prepare_decode(prompt_idx)`.
          - prompt_len > block_size: chunks the prefix through `forward()` so
            the NMM accumulates state across the full prompt, then calls
            `prepare_decode(tail, initial_nmm_states=...)` on the last
            block_size tokens. In this case the returned cache's `position` is
            at `block_size`; callers can sample AT MOST ONE token from
            `cache["last_logits"]` (any `forward_step` call would wpe-OOB).

        Caller still owns the `max_new` cap, since the appropriate value
        depends on what they want to do with the cache.

        `initial_nmm_states`: optional starting NMM state — used by
        `generate.py`'s `--nmm-state-file` persistent-session flow. When
        provided, the NMM remembers context from a previous session: the
        prompt chunks update on top of this state instead of starting from
        the model's init weights. Default `None` = init from scratch
        (existing behavior; backward compatible).

        Same eval-mode contract as prepare_decode (G243): asserted up front
        so the long-prompt prefix chunks don't run their dropout-different
        forward path before the final prepare_decode would have rejected the
        whole thing.
        """
        if self.training:
            raise RuntimeError(
                "prepare_decode_chunked requires model.eval() mode (G243). "
                "Call model.eval() first, or use generate() / "
                "needle_in_haystack() which manage the mode for you."
            )
        block_size = self.config.block_size
        prompt_len = prompt_idx.size(1)
        if prompt_len <= block_size:
            return self.prepare_decode(
                prompt_idx,
                initial_nmm_states=initial_nmm_states,
                int8_kv_cache=int8_kv_cache,
            )

        # Long-prompt path: chunk prefix through forward() so the NMM sees
        # every token; then prepare_decode on the trailing block_size tokens.
        tail_start = prompt_len - block_size
        nmm_states = initial_nmm_states
        for start in range(0, tail_start, block_size):
            end = min(start + block_size, tail_start)
            chunk = prompt_idx[:, start:end]
            _, nmm_states = self(chunk, nmm_states, None)
        tail = prompt_idx[:, tail_start:]
        return self.prepare_decode(
            tail, initial_nmm_states=nmm_states, int8_kv_cache=int8_kv_cache,
        )

    def forward_step(self, token_id: torch.Tensor, cache: dict) -> tuple:
        """Single-token decode forward.

        token_id: [B, 1] new token id.
        cache: dict from prepare_decode (or from a prior forward_step).

        Returns (logits [B, 1, vocab_size], new_cache).

        Mutates the cache structure (in spirit; returns a new dict). Decode
        position bounded by block_size — wpe lookup goes OOB past that.
        Caller should respect that bound.
        """
        # Same eval-mode contract as prepare_decode (see G243). The two
        # methods share the cache structure; if forward_step were allowed
        # in train mode while prepare_decode required eval, a caller could
        # silently combine an eval-mode cache with train-mode decode steps
        # and hit the same parity divergence.
        if self.training:
            raise RuntimeError(
                "forward_step requires model.eval() mode (G243). Call "
                "model.eval() first, or use generate()/needle_in_haystack() "
                "which manage the mode for you."
            )
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
        for i, block in enumerate(self.blocks):
            k_cache, v_cache = cache["kv_caches"][i]
            x, nmm_state, k_cache, v_cache = block.forward_step(
                x,
                cache["nmm_states"][i],
                k_cache,
                v_cache,
            )
            new_nmm_states.append(nmm_state)
            new_kv_caches.append((k_cache, v_cache))

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # [B, 1, vocab_size]
        new_cache = {
            "last_logits": logits,
            "nmm_states": new_nmm_states,
            "kv_caches": new_kv_caches,
            "position": pos_idx + 1,
        }
        return logits, new_cache
