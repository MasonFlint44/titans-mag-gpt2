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

    Decode path: same Option B cached pipeline as `generate()`. Earlier
    versions re-fed the trailing `block_size` window through the NMM at
    every decoded token, compounding NMM state updates ~block_size× per
    sampled token — that's exactly the architectural bug the Phase 7
    rewrite of `generate.py` fixed. Using `prepare_decode` +
    `forward_step` here gives each decoded token exactly ONE NMM update
    via `step_with_conv` (matching training-time semantics) and one
    attention pass via KV cache.

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
        prompt_len = ids.size(1)
        n_decode = len(tokenizer.encode(secret)) + 4

        # Warm-up + cached decode (mirrors generate.py).
        if prompt_len > block_size:
            # Long prompt: chunk the prefix through forward() to thread
            # NMM state across all prompt tokens, then prepare_decode on
            # the trailing block_size tokens. As in generate.py, the
            # cache's `last_logits` is valid for sampling ONE token; we
            # can't run forward_step at the boundary without wpe OOB.
            tail_start = prompt_len - block_size
            nmm_states = None
            for start in range(0, tail_start, block_size):
                end = min(start + block_size, tail_start)
                _, nmm_states = model(ids[:, start:end], nmm_states, None)
            tail = ids[:, tail_start:]
            cache = model.prepare_decode(tail, initial_nmm_states=nmm_states)
            # At the boundary we can only sample 1 token; subsequent ones
            # would need RoPE. Probe is short by construction so this
            # caps recall verification rather than enabling it for long
            # haystacks — caller should choose haystack so prompt fits.
            max_new = 1
        else:
            cache = model.prepare_decode(ids)
            max_new = min(n_decode, block_size - prompt_len + 1)

        out_ids = []
        next_logits = cache["last_logits"].squeeze(1)  # [B, vocab]
        for i in range(max_new):
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            out_ids.append(next_token.item())
            # Skip forward_step on the last iter (saves one NMM update +
            # attention call; also guards the wpe OOB at the boundary).
            if i + 1 < max_new and cache["position"] < block_size:
                new_logits, cache = model.forward_step(next_token, cache)
                next_logits = new_logits.squeeze(1)

        completion = tokenizer.decode(out_ids)
        return secret in completion
    finally:
        if was_training:
            model.train()
