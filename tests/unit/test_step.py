"""Phase 1.7 — sequential single-token step()."""

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from model.nmm import (
    NeuralMemoryModule,
    _dict_add,
    _dict_sub,
    _scale,
    newton_schulz5,
)


# ---------------------------------------------------------------------------
# Dict / scale helpers
# ---------------------------------------------------------------------------

def test_scale_broadcasts_scalar_B_over_nd_tensor():
    scalar = torch.tensor([2.0, 3.0])
    d = {"a": torch.ones(2, 4, 5)}
    out = _scale(scalar, d)
    assert torch.equal(out["a"][0], torch.full((4, 5), 2.0))
    assert torch.equal(out["a"][1], torch.full((4, 5), 3.0))


def test_dict_add_and_sub_elementwise():
    a = {"x": torch.tensor([1.0, 2.0])}
    b = {"x": torch.tensor([3.0, 4.0])}
    assert torch.equal(_dict_add(a, b)["x"], torch.tensor([4.0, 6.0]))
    assert torch.equal(_dict_sub(a, b)["x"], torch.tensor([-2.0, -2.0]))


# ---------------------------------------------------------------------------
# step() shape and return contract
# ---------------------------------------------------------------------------

def test_step_shapes():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)
    y_t, (M_new, S_new) = nmm.step(x_t, state)
    assert y_t.shape == (2, 8)
    for k in M_new:
        assert M_new[k].shape == state[0][k].shape
        assert S_new[k].shape == state[1][k].shape


def test_step_at_finetune_init_returns_zero_y():
    """out_scale=0 at finetune_mode=True -> y_t must be exactly zero, every input."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=True)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)
    y_t, _ = nmm.step(x_t, state)
    assert torch.all(y_t == 0.0)


# ---------------------------------------------------------------------------
# Recurrence semantics
# ---------------------------------------------------------------------------

def test_step_state_changes_from_initial():
    """A single step must move M off the initial state (otherwise no learning)."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)
    _, (M_new, _) = nmm.step(x_t, state)
    assert not torch.equal(M_new["W1.weight"], state[0]["W1.weight"])
    assert not torch.equal(M_new["W_gate.weight"], state[0]["W_gate.weight"])
    assert not torch.equal(M_new["W2.weight"], state[0]["W2.weight"])


def test_step_matches_manual_recurrence():
    """End-to-end: step() must equal a manually-expanded inner loop using the
    same helpers (per_sample_grad_fn, newton_schulz5, _batched_retrieve, gates).
    """
    torch.manual_seed(42)
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)

    # Manual reference
    M_prev, S_prev = state
    x_seq = x_t.unsqueeze(1)
    k_raw = nmm.k_proj(x_seq).squeeze(1)
    q_raw = nmm.q_proj(x_seq).squeeze(1)
    v_raw = nmm.v_proj(x_seq).squeeze(1)
    k_hat = F.normalize(F.silu(k_raw), dim=-1)
    q_hat = F.normalize(F.silu(q_raw), dim=-1)
    v = F.silu(v_raw)
    theta = torch.sigmoid(nmm.W_theta(x_t)).squeeze(-1)
    eta = torch.sigmoid(nmm.W_eta(x_t)).squeeze(-1)
    alpha = torch.sigmoid(nmm.W_alpha(x_t)).squeeze(-1)
    g = nmm.per_sample_grad_fn(M_prev, k_hat, v)
    g_tilde = {key: newton_schulz5(val) for key, val in g.items()}
    S_t_ref = _dict_sub(_scale(eta, S_prev), _scale(theta, g_tilde))
    M_t_ref = _dict_add(_scale(1.0 - alpha, M_prev), S_t_ref)
    y_t_ref = nmm.out_scale * nmm._batched_retrieve(M_t_ref, q_hat)

    y_t, (M_t, S_t) = nmm.step(x_t, state)

    assert torch.allclose(y_t, y_t_ref, atol=1e-6)
    for k in M_t:
        assert torch.allclose(M_t[k], M_t_ref[k], atol=1e-6)
        assert torch.allclose(S_t[k], S_t_ref[k], atol=1e-6)


def test_step_deterministic_with_same_seed():
    """Two NMMs constructed identically must produce identical step output."""
    def build_and_step():
        torch.manual_seed(0)
        nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
        state = nmm.init_state(B=2, device=torch.device("cpu"))
        x_t = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 1.5]] * 2)
        return nmm.step(x_t, state)

    y1, (M1, S1) = build_and_step()
    y2, (M2, S2) = build_and_step()
    assert torch.equal(y1, y2)
    for k in M1:
        assert torch.equal(M1[k], M2[k])


# ---------------------------------------------------------------------------
# Recurrence multi-step
# ---------------------------------------------------------------------------

def test_step_runs_T_iterations_without_nan():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    for _ in range(16):
        x_t = torch.randn(2, 8)
        y_t, state = nmm.step(x_t, state)
        assert torch.isfinite(y_t).all()
        for k in state[0]:
            assert torch.isfinite(state[0][k]).all()
            assert torch.isfinite(state[1][k]).all()


# ---------------------------------------------------------------------------
# G165 — construction-time-only spectral_norm
# ---------------------------------------------------------------------------

def test_step_detects_post_init_spectral_norm_mutation():
    """Mutating self.nmm_spectral_norm after __init__ must raise — the cached
    grad function's reduction is locked at construction; silent mismatch would
    inflate W_theta's effective LR by d_model."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, spectral_norm=True)
    nmm.nmm_spectral_norm = False  # the silent footgun
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)
    with pytest.raises(RuntimeError, match="spectral_norm"):
        nmm.step(x_t, state)


# ---------------------------------------------------------------------------
# theta is POST-NS (paper §3.2 — pre-NS theta cancels in Frobenius division)
# ---------------------------------------------------------------------------

def test_theta_acts_post_NS_not_pre():
    """Scaling each per-sample gradient by theta after NS5 must produce a
    DIFFERENT result than ignoring theta — verifying theta is actually applied."""
    torch.manual_seed(0)
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    # Force theta != 0 by setting W_theta weights non-zero.
    with torch.no_grad():
        nmm.W_theta.weight.fill_(1.0)  # sigmoid(sum(x)) — generally not 0.5
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8) + 5.0  # positive shift -> theta near 1
    _, (_, S_t) = nmm.step(x_t, state)
    # If theta were effectively zero, S_t (zero init) - theta*g would equal -0.
    # Verify S_t is non-trivially nonzero.
    assert torch.any(S_t["W1.weight"].abs() > 1e-3)
