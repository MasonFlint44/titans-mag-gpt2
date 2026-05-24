"""Autoregressive generation with chunked prompt warm-up and conv-window mitigation."""

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
    """Autoregressive sampling. Carries NMM state across calls so test-time
    learning accumulates across the full generation.

    Sampling order is: temperature scale -> top-k mask -> softmax -> multinomial
    (G173). Top-k AFTER softmax requires manual renormalization and is a
    common silent bug. Temperature <= 0 collapses to argmax.

    For prompts longer than block_size, the warm-up runs in block_size-sized
    chunks (G176) — truncating to `[-block_size:]` would silently drop the
    long-context prefix that the NMM is supposed to memorize. wpe(pos) wraps
    per chunk so positions stay in-bounds.

    `tokenizer` is optional (G208); pass the same Tokenizer instance used at
    training/eval time to avoid reproducibility drift between train and gen.

    Mode is captured-and-restored via try/finally (G161) so an enclosing
    training loop continues in train mode regardless of what this helper saw.
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
        nmm_states = None

        # Chunked warm-up so the NMM sees the full prompt, not just the tail.
        prompt_len = context_ids.size(1)
        for start in range(0, prompt_len, block_size):
            end = min(start + block_size, prompt_len)
            chunk = context_ids[:, start:end]
            logits, nmm_states = model(chunk, nmm_states, None)
        next_logits = logits[:, -1, :]

        generated = []
        for _ in range(max_new_tokens):
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

            context_ids = torch.cat([context_ids, next_token], dim=1)
            window = context_ids[:, -block_size:]
            # Sliding-window reprocessing: the NMM sees the window's old
            # tokens again, which double-counts their updates. Acknowledged
            # approximation — proper fix is a KV cache + step() for the new
            # token only.
            logits, nmm_states = model(window, nmm_states, None)
            next_logits = logits[:, -1, :]

        return tok.decode(generated)
    finally:
        if was_training:
            model.train()
