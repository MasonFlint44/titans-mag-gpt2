"""Perplexity and needle-in-haystack evaluation."""

import math
import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, loader, device: torch.device) -> float:
    """Aggregate NLL / tokens across the loader, then exp.

    reduction='sum' (not 'mean') so variable batch lengths don't bias the
    average. NMM state carries across the loader (the loader provides
    `doc_boundaries`; the NMM resets at each one).

    Mode captured-and-restored via try/finally (G161). Wrapped in
    @torch.no_grad to keep the chunked-forward + torch.func.grad graph
    from being built — eval otherwise OOMs at 5-10x training memory.
    """
    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        total_toks = 0
        nmm_states = None
        for input_ids, doc_boundaries in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            doc_boundaries = doc_boundaries.to(device, non_blocking=True)
            logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
            nll = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_toks += input_ids[:, 1:].numel()
        return math.exp(total_nll / total_toks)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def needle_in_haystack(
    model,
    tokenizer,
    device: torch.device,
    haystack: str,
    needle_template: str = "The secret password is {}.",
    secret: str = "alpha-7-zebra",
    probe: str = "The secret password is",
    insert_fraction: float = 0.5,
    block_size: int = None,
) -> bool:
    """Inject a needle into a haystack at the given fractional position, then
    probe the model. Returns True iff the model's greedy completion of `probe`
    contains `secret`.

    A simple binary recall test. For a real harness this would batch many
    (position, secret) pairs and report accuracy vs context length.

    Mode captured-and-restored via try/finally (G161).
    """
    was_training = model.training
    model.eval()
    try:
        # Build the haystack with the needle inserted at `insert_fraction`.
        idx = int(len(haystack) * insert_fraction)
        needle = needle_template.format(secret)
        full = haystack[:idx] + " " + needle + " " + haystack[idx:] + " " + probe

        if block_size is None:
            block_size = model.config.block_size

        ids = torch.tensor(
            tokenizer.encode(full), dtype=torch.long, device=device
        ).unsqueeze(0)

        # Chunked warm-up (matches generate()).
        nmm_states = None
        for start in range(0, ids.size(1), block_size):
            end = min(start + block_size, ids.size(1))
            logits, nmm_states = model(ids[:, start:end], nmm_states, None)

        # Greedy-decode a short continuation; check whether secret appears.
        out_ids = []
        next_logits = logits[:, -1, :]
        for _ in range(len(tokenizer.encode(secret)) + 4):
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            out_ids.append(next_token.item())
            ids = torch.cat([ids, next_token], dim=1)
            window = ids[:, -block_size:]
            logits, nmm_states = model(window, nmm_states, None)
            next_logits = logits[:, -1, :]
        completion = tokenizer.decode(out_ids)
        return secret in completion
    finally:
        if was_training:
            model.train()
