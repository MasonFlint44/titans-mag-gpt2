"""Autoregressive generation via KV-cache attention + single-token NMM step.

Option B from the audit: each decoded token gets exactly ONE NMM update
(matching the TITANS spec) instead of re-feeding the full sliding window
through the NMM at every step (the old behavior). See docs/PLAN.md §5.1 +
docs/RUNBOOK.md "Long-context generation drift" §4.

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
    int8_kv_cache: bool = False,
) -> str:
    """Autoregressive sampling using KV-cache + single-token NMM step.

    Sampling order: temperature -> top-k mask -> softmax -> multinomial (G173).
    Temperature <= 0 collapses to argmax.

    `tokenizer` is optional (G208); pass the same Tokenizer instance used
    at training/eval time to avoid reproducibility drift.

    Mode is captured-and-restored via try/finally (G161).

    max_new_tokens is capped to `block_size - prompt_len + 1` because the
    KV-cache decode path uses absolute positions for wpe and would go OOB
    past block_size. The +1 accounts for the first sampled token coming
    from `cache["last_logits"]` (which needs no `forward_step` call and
    therefore no wpe lookup at the new position). For longer generation
    you'd need RoPE or extrapolation (not implemented).
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

        # Single call handles both short prompts (one-shot prepare_decode) and
        # long prompts (chunked-warm-up + tail prepare_decode). G249.
        # G279: `int8_kv_cache=True` quantizes the cache to int8 + per-(B,h,t)
        # scale — ~2× smaller, decode quality drift bounded for short runs.
        cache = model.prepare_decode_chunked(context_ids, int8_kv_cache=int8_kv_cache)
        if prompt_len > block_size:
            # Long-prompt path: cache position is at block_size; forward_step
            # would wpe-OOB immediately. The user can sample at most ONE new
            # token from cache["last_logits"] (valid for the position right
            # after the last prompt token); further generation requires
            # shortening the prompt.
            max_new = 1 if max_new_tokens >= 1 else 0
        else:
            # +1 because the FIRST sampled token comes from cache["last_logits"]
            # — it doesn't require a forward_step (no wpe lookup at a new
            # position). Only the remaining (max_new - 1) tokens hit
            # forward_step, which needs `cache["position"] + k < block_size`
            # for k = 0..max_new-2. The highest position used is therefore
            # prompt_len + max_new - 2, giving max_new <= block_size - prompt_len + 1.
            # At prompt_len == block_size, this yields max_new == 1 — the user
            # can still sample one token from last_logits without going OOB.
            max_new = min(max_new_tokens, block_size - prompt_len + 1)
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
