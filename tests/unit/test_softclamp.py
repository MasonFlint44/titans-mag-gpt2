"""Tests for softclamp_grad_norm (G265 — lucidrains-style soft norm clamp).

Validates: (a) no-op for small-norm inputs, (b) saturation for large-norm,
(c) smoothness across the transition, (d) autograd traceability,
(e) per-matrix batched application, (f) end-to-end NMM integration.
"""
import pytest
import torch

from config import TitansConfig
from model.nmm import softclamp_grad_norm, NeuralMemoryModule
from model.titans_gpt2 import TitansMAGGPT2


def _frob(t: torch.Tensor) -> torch.Tensor:
    """Per-matrix Frobenius norm over last two dims."""
    return t.norm(dim=(-2, -1))


def test_noop_when_norm_below_threshold():
    """When ||t|| << max_value, softclamp should be near-identity.
    tanh(x) ≈ x for x near 0, so scale ≈ norm/norm = 1."""
    torch.manual_seed(0)
    t = torch.randn(4, 16, 8) * 0.1   # small entries -> small Frobenius
    max_value = 100.0  # way above any matrix's norm
    out = softclamp_grad_norm(t, max_value)
    # Within 1% — tanh(x)≈x to second order for x<0.1, our x is ~norm/100 ≈ 0.01.
    assert torch.allclose(out, t, atol=1e-3, rtol=1e-3)


def test_saturation_when_norm_above_threshold():
    """When ||t|| >> max_value, output norm should approach max_value."""
    torch.manual_seed(1)
    t = torch.randn(4, 16, 8) * 50.0   # huge entries
    max_value = 5.0
    out = softclamp_grad_norm(t, max_value)
    out_norms = _frob(out)
    # tanh(N/M) -> 1 as N/M grows; output norm -> max_value.
    # At norm/max ≈ 200/5 = 40, tanh ≈ 1.0 to 17 decimal places, so
    # output norm should be max_value to fp32 precision.
    assert torch.allclose(out_norms, torch.full_like(out_norms, max_value), rtol=1e-3)


def test_direction_preserved():
    """Softclamp scales the tensor uniformly — out = scale * t, where scale
    is a per-matrix [B, 1, 1] scalar. So `out / t` is constant for non-tiny
    entries (the constant being `scale`)."""
    torch.manual_seed(2)
    t = torch.randn(4, 16, 8) * 10.0
    max_value = 5.0
    out = softclamp_grad_norm(t, max_value)
    for b in range(t.shape[0]):
        # Compute expected scale analytically from the norm + clamp formula.
        norm_b = t[b].norm()
        expected_scale = (max_value * torch.tanh(norm_b / max_value)) / norm_b
        # All entries should satisfy out[b] == expected_scale * t[b].
        assert torch.allclose(out[b], expected_scale * t[b], atol=1e-5, rtol=1e-5)


def test_autograd_safe():
    """The function must be autograd-traceable end-to-end. No graph breaks."""
    torch.manual_seed(3)
    t = (torch.randn(2, 4, 4) * 10.0).requires_grad_(True)
    out = softclamp_grad_norm(t, max_value=2.0)
    loss = out.pow(2).sum()
    loss.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()
    # Saturation regime: gradient should be small (we're at the tanh plateau)
    # but nonzero (unlike hard clip which has gradient = 0 here).
    assert t.grad.abs().sum().item() > 0


def test_smooth_transition():
    """No discontinuity at the threshold. Sweep norms from below to above
    max_value and check the output norm changes continuously."""
    max_value = 1.0
    # Build a sequence of tensors with increasing Frobenius norm.
    direction = torch.randn(1, 16, 16)
    direction = direction / direction.norm()  # unit Frobenius
    norms = torch.linspace(0.01, 5.0, 50)
    out_norms = []
    for n in norms:
        t = (direction * n).clone()
        out = softclamp_grad_norm(t, max_value)
        out_norms.append(out.norm().item())
    out_norms = torch.tensor(out_norms)
    # Monotone increasing (or at least non-decreasing).
    diffs = out_norms[1:] - out_norms[:-1]
    assert (diffs >= -1e-6).all(), "output norm should not decrease with input norm"
    # Asymptotes to ~max_value.
    assert out_norms[-1] < max_value * 1.01
    # No jump in the derivative — check max step in output is bounded.
    # If discontinuous, one diff would be huge. With smooth tanh, all are small.
    assert diffs.max() < 0.15


def test_batched_per_matrix():
    """A per-matrix Frobenius norm — different matrices in a batch get
    different scaling. Catches a bug where the norm is computed over
    the wrong axes."""
    big = torch.randn(8, 8) * 100.0     # ||big|| ≈ 800
    small = torch.randn(8, 8) * 0.01    # ||small|| ≈ 0.08
    stacked = torch.stack([big, small], dim=0)  # [2, 8, 8]
    out = softclamp_grad_norm(stacked, max_value=5.0)
    assert _frob(out)[0] < 5.1   # big saturated
    assert torch.allclose(out[1], small, atol=1e-3, rtol=1e-3)  # small unchanged


# ---------------------------------------------------------------------------
# Config flag + propagation
# ---------------------------------------------------------------------------


def test_config_default_none():
    cfg = TitansConfig.gpt2_small(block_size=64, chunk_size=64)
    assert cfg.nmm_softclamp_max is None


def test_config_accepts_positive_float():
    cfg = TitansConfig.gpt2_small(
        block_size=64, chunk_size=64,
        nmm_softclamp_max=5.0,
    )
    assert cfg.nmm_softclamp_max == 5.0


def test_config_rejects_zero():
    with pytest.raises(ValueError, match="positive float or None"):
        TitansConfig.gpt2_small(block_size=64, chunk_size=64, nmm_softclamp_max=0.0)


def test_config_rejects_negative():
    with pytest.raises(ValueError, match="positive float or None"):
        TitansConfig.gpt2_small(block_size=64, chunk_size=64, nmm_softclamp_max=-1.0)


def test_propagates_to_every_nmm():
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=32, chunk_size=32,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
        nmm_softclamp_max=3.0,
    )
    model = TitansMAGGPT2(cfg)
    for blk in model.blocks:
        if hasattr(blk, "nmm"):
            assert blk.nmm.softclamp_max == 3.0


# ---------------------------------------------------------------------------
# Integration: NMM trains end-to-end with softclamp enabled.
# ---------------------------------------------------------------------------


def test_nmm_trains_with_softclamp_enabled():
    """Run a forward+backward step with softclamp_max set; gradients should
    flow to every parameter and finite values throughout."""
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=8, chunk_size=8,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
        nmm_softclamp_max=5.0,
    )
    torch.manual_seed(0)
    model = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.chunk_size))
    db = torch.zeros_like(ids, dtype=torch.bool); db[:, 0] = True
    logits, _ = model(ids, None, db)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad == n_total
    assert torch.isfinite(loss).all()


def test_softclamp_disabled_matches_no_softclamp():
    """With softclamp_max=None, output should be bit-identical to the
    same model built without the flag — lock the 'no-op' contract."""
    cfg_kwargs = dict(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=8, chunk_size=8,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
    )
    torch.manual_seed(0)
    m_default = TitansMAGGPT2(TitansConfig(**cfg_kwargs))
    torch.manual_seed(0)
    m_none = TitansMAGGPT2(TitansConfig(**cfg_kwargs, nmm_softclamp_max=None))
    m_none.load_state_dict(m_default.state_dict())
    ids = torch.randint(0, cfg_kwargs["vocab_size"], (2, 8))
    db = torch.zeros_like(ids, dtype=torch.bool); db[:, 0] = True
    with torch.no_grad():
        out_default, _ = m_default(ids, None, db)
        out_none, _ = m_none(ids, None, db)
    assert torch.equal(out_default, out_none)
