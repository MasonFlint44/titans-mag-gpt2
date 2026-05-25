"""Perplexity and needle-in-haystack evaluation."""

import math
import random
import string

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

        # Single call handles both short and long prompts (G249). Long
        # prompts get a chunked-warm-up + tail prepare_decode internally;
        # the resulting cache's `position` lands at block_size, so we cap
        # decode at 1 token in that branch (anything further would need
        # RoPE). Probe is short by construction; for long haystacks this
        # caps recall verification rather than enabling it.
        cache = model.prepare_decode_chunked(ids)
        if prompt_len > block_size:
            max_new = 1
        else:
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


def _random_secret(rng: random.Random, n_chars: int = 5) -> str:
    """Generate a random alphanumeric secret. Restricted to uppercase letters
    and digits so the BPE tokenization is dense and deterministic — no
    leading-space-vs-no-space drift, no rare-byte fallbacks."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choices(alphabet, k=n_chars))


@torch.no_grad()
def needle_in_haystack_sweep(
    model,
    tokenizer,
    device: torch.device,
    haystack: str,
    insert_fractions: list = None,
    secrets: list = None,
    n_positions: int = 9,
    n_secrets: int = 5,
    needle_template: str = "The secret password is {}.",
    probe: str = "The secret password is",
    block_size: int = None,
    seed: int = 0,
) -> dict:
    """Sweep N (insert_fraction, secret) pairs and aggregate recall.

    For each pair: build the full prompt with the needle injected at
    `insert_fraction`, decode the probe completion, check whether the
    secret appears. Aggregate per-position and per-secret recall.

    The TEST_PLAN.md §10 specification ("Top-1 token includes the secret
    for ≥80% of positions") is implemented here as `result["recall"]`.

    Args:
      insert_fractions: positions in [0, 1] to inject the needle. Default:
        `n_positions` evenly spaced points in (0, 1).
      secrets: secret strings to test. Default: `n_secrets` random
        5-character alphanumeric strings generated from `seed`.
      n_positions, n_secrets: only consulted when the corresponding
        explicit list is None.
      seed: RNG seed for default secret generation. Has no effect when
        `secrets` is supplied explicitly.

    Returns:
      dict with:
        recall: float — overall fraction of pairs matched
        n_pairs: int — total pairs evaluated (= len(insert_fractions) * len(secrets))
        n_matched: int — pairs whose decoded completion contained the secret
        per_position: dict[float, float] — recall at each insert_fraction
        per_secret: dict[str, float] — recall for each secret across positions
        details: list[tuple[float, str, bool]] — per-pair (fraction, secret, matched)
        insert_fractions: list[float] — the actual positions evaluated
        secrets: list[str] — the actual secrets evaluated

    Untrained models will produce near-zero recall (the secret is unlikely
    to be the argmax of an untrained vocab distribution); this harness is
    intended for trained-checkpoint evaluation. See TEST_PLAN §10.

    Mode is captured-and-restored by the per-pair `needle_in_haystack`
    call (G161); this function adds no additional mode mutation.
    """
    rng = random.Random(seed)

    if insert_fractions is None:
        # Evenly spaced in (0, 1), avoiding the exact endpoints where the
        # needle would land outside the haystack character bounds.
        step = 1.0 / (n_positions + 1)
        insert_fractions = [round(step * (i + 1), 3) for i in range(n_positions)]
    if secrets is None:
        secrets = [_random_secret(rng) for _ in range(n_secrets)]

    details = []
    for f in insert_fractions:
        for s in secrets:
            matched = needle_in_haystack(
                model=model,
                tokenizer=tokenizer,
                device=device,
                haystack=haystack,
                needle_template=needle_template,
                secret=s,
                probe=probe,
                insert_fraction=f,
                block_size=block_size,
            )
            details.append((f, s, matched))

    n_pairs = len(details)
    n_matched = sum(1 for _, _, m in details if m)
    recall = n_matched / n_pairs if n_pairs > 0 else 0.0

    per_position = {}
    for f in insert_fractions:
        pos_results = [m for ff, _, m in details if ff == f]
        per_position[f] = (
            sum(pos_results) / len(pos_results) if pos_results else 0.0
        )

    per_secret = {}
    for s in secrets:
        sec_results = [m for _, ss, m in details if ss == s]
        per_secret[s] = (
            sum(sec_results) / len(sec_results) if sec_results else 0.0
        )

    return {
        "recall": recall,
        "n_pairs": n_pairs,
        "n_matched": n_matched,
        "per_position": per_position,
        "per_secret": per_secret,
        "details": details,
        "insert_fractions": insert_fractions,
        "secrets": secrets,
    }
