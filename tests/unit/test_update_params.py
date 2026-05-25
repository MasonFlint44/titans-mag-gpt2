import torch

from model.nmm import NeuralMemoryModule


def _call_pattern(linear: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """The canonical call-site form from docs/PLAN.md §1.3 / §1.7."""
    return torch.sigmoid(linear(x)).squeeze(-1)


def test_outputs_in_unit_interval():
    """Sigmoid output is in [0, 1]; fp32 saturates exactly at the boundaries
    for extreme inputs, so the closed interval is the runtime-correct bound."""
    nmm = NeuralMemoryModule(n_embd=8)
    x = torch.randn(2, 16, 8)
    for linear in (nmm.W_theta, nmm.W_eta, nmm.W_alpha):
        out = _call_pattern(linear, x)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_outputs_strictly_inside_open_interval_for_moderate_input():
    """For non-extreme input, sigmoid output should be strictly inside (0, 1)."""
    nmm = NeuralMemoryModule(n_embd=8)
    x = torch.randn(2, 16, 8)  # roughly N(0,1) logits -> sigmoid output well inside (0,1)
    for linear in (nmm.W_theta, nmm.W_eta, nmm.W_alpha):
        out = _call_pattern(linear, x)
        assert torch.all(out > 0.0) and torch.all(out < 1.0)


def test_output_shape_is_B_T_not_B_T_1():
    """[B, T] shape (post-squeeze) is required for §1.5 scalar loss and §1.7 broadcasting.

    Pre-squeeze the linear produces [B, T, 1]; if the call site forgot the
    squeeze, downstream code would either broadcast wrong or grad() would
    receive a non-scalar loss.
    """
    nmm = NeuralMemoryModule(n_embd=8)
    x = torch.randn(3, 7, 8)
    for linear in (nmm.W_theta, nmm.W_eta, nmm.W_alpha):
        out = _call_pattern(linear, x)
        assert out.shape == (3, 7), f"Expected [B=3, T=7], got {tuple(out.shape)}"
        assert out.dim() == 2


def test_pre_squeeze_shape_is_B_T_1():
    """Sanity: the linear itself emits [B, T, 1]; the call site is responsible for squeeze."""
    nmm = NeuralMemoryModule(n_embd=8)
    x = torch.randn(2, 5, 8)
    pre = nmm.W_theta(x)
    assert pre.shape == (2, 5, 1)


def test_three_linears_are_independent_modules():
    nmm = NeuralMemoryModule(n_embd=8)
    # Each Linear should be a separate parameter set.
    assert nmm.W_theta is not nmm.W_eta
    assert nmm.W_theta is not nmm.W_alpha
    assert nmm.W_eta is not nmm.W_alpha
    # Independent weights (Xavier-uniform default differs by seeding order).
    assert not torch.equal(nmm.W_theta.weight, nmm.W_eta.weight)
    assert not torch.equal(nmm.W_eta.weight, nmm.W_alpha.weight)


def test_perturbing_W_alpha_does_not_change_theta_output():
    """The three projections must operate on independent weights — perturbing
    one must leave the other two's outputs unchanged.
    """
    nmm = NeuralMemoryModule(n_embd=8)
    x = torch.randn(2, 4, 8)
    theta_before = _call_pattern(nmm.W_theta, x).clone()
    with torch.no_grad():
        nmm.W_alpha.weight.add_(torch.randn_like(nmm.W_alpha.weight))
    theta_after = _call_pattern(nmm.W_theta, x)
    assert torch.equal(theta_before, theta_after)


def test_no_bias_on_update_params():
    """bias=False per docs/PLAN.md §1.3 — keeps init-time output at sigmoid(0) = 0.5."""
    nmm = NeuralMemoryModule(n_embd=8)
    assert nmm.W_theta.bias is None
    assert nmm.W_eta.bias is None
    assert nmm.W_alpha.bias is None
