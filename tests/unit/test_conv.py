import pytest
import torch

from model.nmm import CausalDepthwiseConv1d


def test_shape_preservation_default_kernel():
    conv = CausalDepthwiseConv1d(dim=32, kernel_size=4)
    x = torch.randn(2, 16, 32)
    y = conv(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("T", [1, 2, 4, 16, 64, 1024])
def test_shape_preservation_varying_T(T: int):
    conv = CausalDepthwiseConv1d(dim=8, kernel_size=4)
    x = torch.randn(2, T, 8)
    y = conv(x)
    assert y.shape == (2, T, 8)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_shape_preservation_varying_kernel(k: int):
    conv = CausalDepthwiseConv1d(dim=8, kernel_size=k)
    x = torch.randn(2, 16, 8)
    y = conv(x)
    assert y.shape == x.shape


def test_strict_causality_perturbing_future_does_not_change_past():
    """Output at position t must NOT change when input at position t+k is perturbed."""
    conv = CausalDepthwiseConv1d(dim=4, kernel_size=4)
    x1 = torch.randn(1, 16, 4)
    x2 = x1.clone()
    x2[0, 8:, :] = torch.randn(8, 4)  # perturb positions >= 8
    with torch.no_grad():
        y1 = conv(x1)
        y2 = conv(x2)
    # Positions 0..7 must be bit-identical (depend only on inputs <= 7).
    assert torch.equal(y1[:, :8, :], y2[:, :8, :])


def test_strict_causality_via_jacobian_pattern():
    """For each output position t, the Jacobian wrt input position s must be zero for s > t."""
    conv = CausalDepthwiseConv1d(dim=4, kernel_size=4)
    x = torch.randn(1, 8, 4, requires_grad=True)
    y = conv(x)
    T = x.shape[1]
    for t in range(T):
        # Sum the output at position t across channels; backprop.
        x.grad = None
        y[0, t, :].sum().backward(retain_graph=True)
        # Inputs at positions > t must have zero gradient (no causal dependency).
        future_grad = x.grad[0, t + 1:, :]
        assert torch.all(future_grad == 0), (
            f"Output at t={t} has non-zero gradient wrt future inputs"
        )


def test_depthwise_no_cross_channel_mixing():
    """A perturbation on input channel c must not affect output of channel c' != c."""
    conv = CausalDepthwiseConv1d(dim=8, kernel_size=4)
    x = torch.zeros(1, 16, 8)
    x[0, 4, 2] = 1.0  # spike on channel 2 at t=4
    with torch.no_grad():
        y = conv(x)
    # All channels other than 2 must be zero everywhere (output stays in-channel).
    other_channels = torch.cat([y[..., :2], y[..., 3:]], dim=-1)
    assert torch.allclose(other_channels, torch.zeros_like(other_channels), atol=0.0)
    # Channel 2 must have some non-zero response.
    assert torch.any(y[..., 2] != 0)


def test_left_padding_at_t0_sees_only_x0():
    """Output at t=0 must be unaffected by inputs at t>=1."""
    conv = CausalDepthwiseConv1d(dim=4, kernel_size=4)
    x1 = torch.zeros(1, 8, 4)
    x1[0, 0] = torch.randn(4)
    x2 = x1.clone()
    x2[0, 1:] = torch.randn(7, 4)  # any future content; t=0 must be unaffected
    with torch.no_grad():
        y1 = conv(x1)
        y2 = conv(x2)
    assert torch.equal(y1[0, 0], y2[0, 0])


def test_rejects_invalid_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        CausalDepthwiseConv1d(dim=8, kernel_size=0)


def test_kernel_size_1_is_pointwise_in_time():
    """At k=1 the conv is purely a per-channel linear with no temporal mixing."""
    conv = CausalDepthwiseConv1d(dim=4, kernel_size=1)
    x = torch.randn(2, 8, 4)
    y = conv(x)
    # output[t] depends only on x[t]; replacing x[t+1:] must not change y[t].
    x2 = x.clone()
    x2[:, 1:] = 0
    with torch.no_grad():
        y2 = conv(x2)
    assert torch.equal(y[:, 0, :], y2[:, 0, :])
