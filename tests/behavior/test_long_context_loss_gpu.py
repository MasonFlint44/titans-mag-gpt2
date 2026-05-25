"""docs/TEST_PLAN.md §10: per-position loss across a long sequence.

With HF GPT-2 weights loaded and the NMM zeroed (out_scale=0), the
backbone is vanilla GPT-2 and per-position cross-entropy should look
like HF GPT-2's: roughly stable across positions (no sharp jump at any
specific position would indicate a state-reset bug or a positional
embedding bug).

A trained NMM should make later-position loss DECREASE (more context →
better predictions); we don't have a trained checkpoint here, so the
weaker invariant we can verify is: no PATHOLOGICAL position-dependent
spike (e.g., a 10x jump at any position would indicate a state-reset
or positional-embedding bug).
"""

import pytest
import torch
import torch.nn.functional as F

from config import TitansConfig
from data.tokenizer import Tokenizer
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def _per_position_loss(model, idx: torch.Tensor) -> torch.Tensor:
    """Returns [B, T-1] cross-entropy loss per position."""
    db = torch.zeros_like(idx, dtype=torch.bool)
    db[:, 0] = True  # force sequential path; no-op observationally
    with torch.no_grad():
        logits, _ = model(idx, nmm_states=None, doc_boundaries=db)
    # CE between logits[:, :-1, :] and idx[:, 1:].
    flat_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    flat_targets = idx[:, 1:].reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return per_token.view(idx.size(0), -1)  # [B, T-1]


def test_loss_decreases_across_repetitions_with_HF_loaded():
    """docs/TEST_PLAN.md §10: loss should DECREASE as more context accumulates
    (model has more to condition on). On a repeating text, the second
    repetition should have substantially lower loss than the first because
    the pattern is now in the attention context.

    A state-reset bug would manifest as the second-repetition loss being
    similar to or higher than the first-repetition loss (state was wiped
    between repetitions). Use HF-loaded GPT-2 with NMM zeroed so we
    isolate the attention/positional path.
    """
    _gpu_or_skip()
    cfg = TitansConfig.gpt2_small(nmm_n_persistent=0)
    model = TitansMAGGPT2(cfg).cuda()
    load_pretrained(model, cfg)
    model.eval()

    tok = Tokenizer()
    # One repetition is ~32 tokens; 4 reps fit comfortably in T=128.
    one_rep = (
        "The quick brown fox jumps over the lazy dog. "
        "She sells seashells by the seashore. "
        "Peter Piper picked a peck of pickled peppers. "
    )
    text = one_rep * 4
    ids = torch.tensor(tok.encode(text), dtype=torch.long, device="cuda").unsqueeze(0)
    ids = ids[:, :128]

    losses = _per_position_loss(model, ids)[0]  # [T-1]
    T = losses.size(0)
    first_quarter = losses[: T // 4].mean().item()
    last_quarter = losses[3 * T // 4 :].mean().item()
    assert last_quarter < first_quarter, (
        f"loss did not decrease across repetitions: first quarter mean "
        f"= {first_quarter:.3f}, last quarter mean = {last_quarter:.3f}. "
        f"A state-reset or positional-embedding bug would manifest like this."
    )
    # And the decrease should be substantial — at least 30% drop.
    assert last_quarter < 0.7 * first_quarter, (
        f"loss decreased only marginally: first={first_quarter:.3f}, "
        f"last={last_quarter:.3f}. Memory/context not flowing as expected."
    )


def test_position_loss_is_bounded_for_HF_loaded_model():
    """Sanity: per-position loss for HF GPT-2 weights should be in a
    reasonable range (English text -> CE < 8 per token typically; certainly
    not 30+ which would indicate broken weights)."""
    _gpu_or_skip()
    cfg = TitansConfig.gpt2_small(nmm_n_persistent=0)
    model = TitansMAGGPT2(cfg).cuda()
    load_pretrained(model, cfg)
    model.eval()

    tok = Tokenizer()
    text = "The capital of France is Paris. " * 16
    ids = torch.tensor(tok.encode(text), dtype=torch.long, device="cuda").unsqueeze(0)
    ids = ids[:, :64]
    losses = _per_position_loss(model, ids)
    # On repetitive text the loss should be quite low after a few tokens.
    assert losses.mean().item() < 8.0
    assert losses.max().item() < 20.0
