"""Phase 2.0 — CausalSelfAttention."""

import subprocess
import sys

import pytest
import torch

from model.block import CausalSelfAttention


def test_attention_shape_no_mask():
    attn = CausalSelfAttention(n_embd=32, n_head=4)
    x = torch.randn(2, 16, 32)
    y = attn(x, mask=None)
    assert y.shape == x.shape


def test_attention_shape_with_explicit_causal_mask():
    attn = CausalSelfAttention(n_embd=32, n_head=4)
    T = 16
    causal = torch.full((T, T), float("-inf"))
    causal = torch.triu(causal, diagonal=1)
    x = torch.randn(2, T, 32)
    y = attn(x, mask=causal)
    assert y.shape == x.shape


def test_causal_mask_actually_enforces_causality():
    """With a causal mask, perturbing inputs at t+k must not change output at t."""
    attn = CausalSelfAttention(n_embd=16, n_head=2)
    attn.eval()  # no dropout
    T = 8
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    x1 = torch.randn(1, T, 16)
    x2 = x1.clone()
    x2[0, 4:, :] = torch.randn(4, 16)
    with torch.no_grad():
        y1 = attn(x1, mask=causal)
        y2 = attn(x2, mask=causal)
    assert torch.allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-5)


def test_no_mask_attends_to_all_positions():
    """Without a mask, output at t=0 depends on the full sequence."""
    attn = CausalSelfAttention(n_embd=16, n_head=2)
    attn.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[0, 4:, :] = torch.randn(4, 16)
    with torch.no_grad():
        y1 = attn(x1, mask=None)
        y2 = attn(x2, mask=None)
    # Should DIFFER at t=0 — proves no implicit causal mask.
    assert not torch.allclose(y1[:, 0, :], y2[:, 0, :], atol=1e-5)


# ---------------------------------------------------------------------------
# ValueError (not AssertionError) for bad dims, survives -O
# ---------------------------------------------------------------------------

def test_rejects_n_head_not_dividing_n_embd_via_ValueError():
    with pytest.raises(ValueError, match="divisible by n_head"):
        CausalSelfAttention(n_embd=768, n_head=10)


def test_attention_validation_survives_python_O():
    """must use raise ValueError, not assert — survives -O strip."""
    result = subprocess.run(
        [
            sys.executable, "-O", "-c",
            "from model.block import CausalSelfAttention; "
            "CausalSelfAttention(n_embd=768, n_head=10)",
        ],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr


# ---------------------------------------------------------------------------
# Linear-layer bias presence for HF parity (Phase 2.6)
# ---------------------------------------------------------------------------

def test_all_linear_layers_have_bias():
    """HF GPT-2 c_attn and c_proj have biases. Must match for weight-load parity."""
    attn = CausalSelfAttention(n_embd=32, n_head=4)
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.proj.bias is not None


def test_qkv_are_split_not_fused():
    """Phase 2.6 copies HF's fused c_attn into q_proj/k_proj/v_proj independently;
    they must be three separate Linear modules."""
    attn = CausalSelfAttention(n_embd=32, n_head=4)
    assert attn.q_proj is not attn.k_proj
    assert attn.q_proj is not attn.v_proj
    assert attn.k_proj is not attn.v_proj
    # Independent weights at init.
    assert not torch.equal(attn.q_proj.weight, attn.k_proj.weight)


# ---------------------------------------------------------------------------
# T15 — determinism at dropout=0 (TEST_PLAN §4 spec)
# ---------------------------------------------------------------------------

def test_attention_output_deterministic_at_dropout_zero():
    """T15 — `CausalSelfAttention(..., dropout=0.0)` must produce
    bit-identical outputs for the same input across two forward calls.
    No hidden RNG source should leak into the attention path when
    dropout is disabled.

    Catches: a refactor that accidentally introduces randomness (e.g.,
    via a stochastic attention variant, a randomized k/v shuffle, or
    a misconfigured nn.Dropout with p > 0)."""
    attn = CausalSelfAttention(n_embd=16, n_head=2, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y1 = attn(x, mask=None)
        y2 = attn(x, mask=None)
    assert torch.equal(y1, y2), (
        "attention output differs across identical calls at dropout=0.0 "
        "with model.eval() — some RNG source is leaking in."
    )


def test_attention_output_deterministic_at_dropout_zero_with_causal_mask():
    """T15 — same as above but with an explicit causal mask. The mask
    path goes through a different SDPA codepath and could have its own
    RNG bug."""
    attn = CausalSelfAttention(n_embd=16, n_head=2, dropout=0.0)
    attn.eval()
    T = 8
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    x = torch.randn(2, T, 16)
    with torch.no_grad():
        y1 = attn(x, mask=causal)
        y2 = attn(x, mask=causal)
    assert torch.equal(y1, y2), (
        "attention output differs across identical calls at dropout=0.0 "
        "with an explicit causal mask."
    )


def test_attention_output_deterministic_in_train_mode_at_dropout_zero():
    """T15 — even in TRAIN mode, dropout=0.0 means no stochastic op
    should fire. Verify that `.train()` doesn't accidentally enable
    randomness when the rate is exactly 0.

    The internal `self.resid_dropout.p if self.training else 0.0`
    branch in CausalSelfAttention.forward picks up self.resid_dropout.p
    in train mode; with that p=0 we should still get identical outputs.
    """
    attn = CausalSelfAttention(n_embd=16, n_head=2, dropout=0.0)
    attn.train()  # train mode is the typical place dropout fires
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        y1 = attn(x, mask=None)
        y2 = attn(x, mask=None)
    assert torch.equal(y1, y2), (
        "attention in train mode at dropout=0.0 is not deterministic — "
        "the p=0 fast path is broken."
    )
