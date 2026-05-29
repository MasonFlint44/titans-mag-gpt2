"""Tests for the anti-marginal-output contrastive needle loss.

Two parts: (1) `find_answer_positions` correctly locates "A:" markers
in input streams and aligns its mask to the standard LM-loss slice
shape; (2) `compute_contrastive_needle_loss` produces the InfoNCE
top-K-hard-negative loss we expect, including zero on empty masks and
strictly higher loss when the model assigns low probability to the
correct answer.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cli.train import (
    NEEDLE_ANSWER_MARKER_TOKENS,
    compute_contrastive_needle_loss,
    find_answer_positions,
)


# ---------------------------------------------------------------------------
# find_answer_positions
# ---------------------------------------------------------------------------


def test_marker_constants_match_GPT2_BPE_for_A_colon():
    """The hard-coded marker (32, 25) must equal what GPT-2's BPE
    actually produces for "A:" — drift here would silently disable
    the contrastive loss (no answer positions detected)."""
    from data.tokenizer import Tokenizer
    tok = Tokenizer()
    assert tuple(tok.encode("A:")) == NEEDLE_ANSWER_MARKER_TOKENS


def test_find_answer_positions_locates_single_marker():
    """Input with exactly one "A:" marker should produce a mask with
    exactly one True at the position whose label is the post-marker
    token."""
    A, COL = NEEDLE_ANSWER_MARKER_TOKENS
    # Construct: [random, A, :, answer, random]
    # Position 2 (the :) has label = position 3 = answer. Mask[2]=True.
    input_ids = torch.tensor([[100, A, COL, 7777, 9999]])
    mask = find_answer_positions(input_ids)
    assert mask.shape == (1, 4)  # [B, T-1]
    expected = torch.tensor([[False, False, True, False]])
    assert torch.equal(mask, expected)


def test_find_answer_positions_locates_multiple_markers():
    """A batched input with two "A:" markers in one row should produce
    True at both colon positions."""
    A, COL = NEEDLE_ANSWER_MARKER_TOKENS
    input_ids = torch.tensor([
        [100, A, COL, 1111, 200, A, COL, 2222],
    ])
    mask = find_answer_positions(input_ids)
    expected = torch.tensor([
        [False, False, True, False, False, False, True],
    ])
    assert torch.equal(mask, expected)


def test_find_answer_positions_per_batch_row_independence():
    """Different rows in the batch can have answer markers at different
    positions — the mask must reflect each row independently."""
    A, COL = NEEDLE_ANSWER_MARKER_TOKENS
    input_ids = torch.tensor([
        [A, COL, 1111, 200, 300],  # marker at positions 0-1, label at 2
        [400, A, COL, 1111, 200],  # marker at positions 1-2, label at 3
    ])
    mask = find_answer_positions(input_ids)
    expected = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
    ])
    assert torch.equal(mask, expected)


def test_find_answer_positions_lone_colon_not_a_marker():
    """A naked ":" without a preceding "A" is not an answer marker;
    `find_answer_positions` must not flag it."""
    _, COL = NEEDLE_ANSWER_MARKER_TOKENS
    input_ids = torch.tensor([[100, 200, COL, 1111, 9999]])
    mask = find_answer_positions(input_ids)
    expected = torch.tensor([[False, False, False, False]])
    assert torch.equal(mask, expected)


def test_find_answer_positions_empty_input_safe():
    """Input shorter than the marker length must not crash — return an
    all-False mask of the LM-loss-aligned shape `[B, T-1]` or empty."""
    A, _ = NEEDLE_ANSWER_MARKER_TOKENS
    input_ids = torch.tensor([[A]])  # 1 token
    mask = find_answer_positions(input_ids)
    # T=1 → T-1=0, no positions to compare
    assert mask.shape == (1, 0)


# ---------------------------------------------------------------------------
# compute_contrastive_needle_loss
# ---------------------------------------------------------------------------


def test_contrastive_loss_zero_when_mask_empty():
    """No answer positions → no loss to compute. Returns finite zero
    rather than NaN (which `mean()` on empty would yield)."""
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    mask = torch.zeros(2, 5, dtype=torch.bool)
    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=10)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_contrastive_loss_is_zero_when_correct_dominates_top_k():
    """If the correct answer's logit is much larger than every other
    logit, the InfoNCE softmax over [correct, top_k_wrong] puts ~all
    mass on correct, so the cross-entropy approaches zero."""
    V = 50
    logits = torch.full((1, 3, V), -10.0)
    labels = torch.tensor([[0, 7, 0]])
    mask = torch.tensor([[False, True, False]])  # one answer position
    # Make correct (logit at index 7) dominate.
    logits[0, 1, 7] = 100.0
    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=5)
    assert loss.item() < 1e-3, (
        f"expected near-zero loss when correct logit dominates; got {loss.item()}"
    )


def test_contrastive_loss_is_high_when_correct_underranked():
    """If a wrong token has much higher logit than the correct one, the
    contrastive loss should be large (model paying for ranking error)."""
    V = 50
    logits = torch.full((1, 3, V), 0.0)
    labels = torch.tensor([[0, 7, 0]])
    mask = torch.tensor([[False, True, False]])
    # Wrong token at index 3 outscores correct (7).
    logits[0, 1, 3] = 100.0
    logits[0, 1, 7] = -100.0
    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=5)
    assert loss.item() > 5.0, (
        f"expected large loss when correct is dominated by a wrong "
        f"prediction; got {loss.item()}"
    )


def test_contrastive_loss_does_not_include_correct_in_negatives():
    """The correct token's logit must be masked out before top-K
    selection — otherwise it could appear as both 'correct' and 'a
    top-K negative', and the InfoNCE softmax would compare it against
    itself (trivially making the loss too easy)."""
    V = 5
    # Set up: correct token has logit 10, all others have logit 0. With
    # top_k=4 we'd pick the 4 zero-logit indices as negatives — the
    # correct should NOT be one of them.
    logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0]])  # correct is index 0
    mask = torch.tensor([[True]])
    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=4)
    # Expected: softmax over [10, 0, 0, 0, 0], target=0
    # = -log(exp(10)/(exp(10)+4*exp(0))) ≈ -log(1/(1+4e-10)) ≈ ~0
    assert loss.item() < 1e-3


def test_contrastive_loss_matches_hand_computed_value():
    """End-to-end correctness check: a hand-constructed [N, V] logit
    matrix produces the loss value computed by the standard
    F.cross_entropy on the [N, K+1] softmax."""
    # 2 answer positions, vocab=10, correct labels [3, 5].
    logits = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5]],
    ])
    # Replicate to get 2 answer positions in one batch row.
    logits = logits.repeat(1, 2, 1)
    labels = torch.tensor([[3, 5]])
    mask = torch.tensor([[True, True]])

    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=3)

    # Hand computation:
    # For each row, mask correct, take top-3 wrong logits.
    # Row 0 (correct=3, value=4): top-3 wrong are [9, 8, 7] (values from
    #   indices 8, 7, 6). Combined: [4, 9, 8, 7]. softmax target=0.
    # Row 1 (correct=5, value=6): top-3 wrong are [9, 8, 7]. Combined:
    #   [6, 9, 8, 7]. softmax target=0.
    expected_row_0 = F.cross_entropy(
        torch.tensor([[4.0, 9.0, 8.0, 7.0]]),
        torch.tensor([0]),
        reduction="none",
    )
    expected_row_1 = F.cross_entropy(
        torch.tensor([[6.0, 9.0, 8.0, 7.0]]),
        torch.tensor([0]),
        reduction="none",
    )
    expected = (expected_row_0 + expected_row_1) / 2
    assert torch.allclose(loss, expected[0], atol=1e-5), (
        f"computed {loss.item()}, expected {expected.item()}"
    )


def test_contrastive_loss_gradient_flows_to_logits():
    """Backward through the contrastive loss must produce gradients on
    the logits — defends against an accidental detach somewhere in the
    top-K / scatter machinery."""
    V = 20
    logits = torch.randn(1, 4, V, requires_grad=True)
    labels = torch.randint(0, V, (1, 4))
    mask = torch.tensor([[False, True, False, True]])
    loss = compute_contrastive_needle_loss(logits, labels, mask, top_k=5)
    loss.backward()
    assert logits.grad is not None
    # Gradient should be non-zero specifically at masked positions; at
    # unmasked positions we don't require any particular gradient (it'll
    # be zero since they don't participate in the loss).
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() > 0


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def test_finetune_cli_exposes_contrastive_flags():
    """The two new flags must show up in --help so users discover them
    without reading source. Regression guard against the parser being
    rewritten and the flags silently dropped."""
    from cli.finetune import build_parser
    parser = build_parser()
    help_text = parser.format_help()
    assert "--needle-contrastive-loss-weight" in help_text
    assert "--needle-contrastive-top-k" in help_text


def test_run_training_accepts_contrastive_kwargs():
    """`run_training`'s signature must accept the two contrastive kwargs.
    Defends against a signature-rename drift between finetune.py and
    train.py."""
    import inspect
    from cli.train import run_training
    sig = inspect.signature(run_training)
    assert "needle_contrastive_loss_weight" in sig.parameters
    assert "needle_contrastive_top_k" in sig.parameters
