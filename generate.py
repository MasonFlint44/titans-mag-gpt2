"""Autoregressive generation via KV-cache attention + single-token NMM step.

Option B from the audit: each decoded token gets exactly ONE NMM update
(matching the TITANS spec) instead of re-feeding the full sliding window
through the NMM at every step (the old behavior). See PLAN.md §5.1 +
RUNBOOK.md "Long-context generation drift" §4.

Pipeline:
  1. prepare_decode: chunked warm-up on the prompt (existing forward_chunk
     for the NMM; KV cache + conv buffer captured at the end).
  2. Decode loop: for each new token, call model.forward_step — one NMM
     update via step_with_conv (full k-token conv context), one attention
     pass via KV cache.

Bounded by block_size: total tokens (prompt + generated) <= block_size,
because GPT-2's wpe table only covers positions 0..block_size-1.
"""

import torch
import torch.nn.functional as F

from data.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    tokenizer: Tokenizer = None,
) -> str:
    """Autoregressive sampling using KV-cache + single-token NMM step.

    Sampling order: temperature -> top-k mask -> softmax -> multinomial (G173).
    Temperature <= 0 collapses to argmax.

    `tokenizer` is optional (G208); pass the same Tokenizer instance used
    at training/eval time to avoid reproducibility drift.

    Mode is captured-and-restored via try/finally (G161).

    max_new_tokens is capped to `block_size - prompt_len` because the
    KV-cache decode path uses absolute positions for wpe and would go OOB
    past block_size. For longer generation you'd need RoPE or extrapolation
    (not implemented).
    """
    was_training = model.training
    model.eval()
    try:
        tok = tokenizer if tokenizer is not None else Tokenizer()
        device = next(model.parameters()).device

        prompt_ids = tok.encode(prompt) if prompt else [tok.eot_token]
        context_ids = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        block_size = model.config.block_size
        prompt_len = context_ids.size(1)

        if prompt_len > block_size:
            # Long-prompt path (G176): NMM sees the full prompt; KV cache
            # holds only the last block_size tokens (wpe table is bounded).
            # Chunk the prefix through forward() so the NMM accumulates
            # state, then prepare_decode on the tail with that state.
            tail_start = prompt_len - block_size
            nmm_states = None
            for start in range(0, tail_start, block_size):
                end = min(start + block_size, tail_start)
                chunk = context_ids[:, start:end]
                _, nmm_states = model(chunk, nmm_states, None)
            tail = context_ids[:, tail_start:]
            cache = model.prepare_decode(tail, initial_nmm_states=nmm_states)
            # Position in cache is block_size; KV cache holds block_size
            # real positions + N_p persistent. forward_step would go OOB on
            # wpe immediately. The user can sample at most ONE new token
            # from the cache's last_logits (which IS valid for the position
            # immediately AFTER block_size - 1, the last prompt token); any
            # further generation requires shortening the prompt.
            max_new = 1 if max_new_tokens >= 1 else 0
        else:
            # Single-shot warm-up.
            cache = model.prepare_decode(context_ids)
            max_new = min(max_new_tokens, block_size - prompt_len)
        next_logits = cache["last_logits"].squeeze(1)  # [B, vocab]

        generated = []
        for i in range(max_new):
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            if next_token.item() == tok.eot_token:
                break

            # No need to run forward_step on the LAST iteration — we already
            # have the sampled token, no next_logits needed. Saves one
            # NMM update + attention call. Also guards against OOB wpe when
            # the long-prompt path has already pushed position to block_size.
            if i + 1 < max_new and cache["position"] < block_size:
                new_logits, cache = model.forward_step(next_token, cache)
                next_logits = new_logits.squeeze(1)

        return tok.decode(generated)
    finally:
        if was_training:
            model.train()
