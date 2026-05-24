"""Phase 2.0 — GPT2MLP."""

import torch
import torch.nn.functional as F

from model.block import GPT2MLP


def test_shape_invariant():
    mlp = GPT2MLP(n_embd=32, dropout=0.0)
    x = torch.randn(2, 16, 32)
    assert mlp(x).shape == x.shape


def test_uses_tanh_approximate_gelu_not_erf():
    """HF GPT-2 uses gelu_new (tanh-approx). The default F.gelu (erf) is
    numerically different — would cause logit drift in 2.6 parity tests.

    Concretely: with c_fc.bias=0, c_proj=identity, the MLP output should
    match F.gelu(c_fc(x), approximate='tanh') — NOT F.gelu(c_fc(x)).
    """
    mlp = GPT2MLP(n_embd=8, dropout=0.0)
    with torch.no_grad():
        # Set c_proj = Identity-like (W = 4x8 identity-tiled, bias=0)
        # so y = gelu(c_fc(x)) directly.
        mlp.c_proj.weight.zero_()
        for i in range(8):
            mlp.c_proj.weight[i, i] = 1.0  # picks the first 8 of 32 hidden units
        mlp.c_proj.bias.zero_()

    x = torch.randn(2, 4, 8)
    mlp.eval()
    y = mlp(x)
    h = mlp.c_fc(x)
    expected_tanh = F.gelu(h, approximate="tanh")[..., :8]
    expected_erf = F.gelu(h)[..., :8]  # default approximate='none'

    assert torch.allclose(y, expected_tanh, atol=1e-6)
    # If we accidentally used erf gelu, this would also pass — sanity-check
    # the two approximations actually differ for our input.
    assert not torch.allclose(expected_tanh, expected_erf, atol=1e-4)


def test_biases_present_on_c_fc_and_c_proj():
    """HF GPT-2 MLP has biases on both linears. Required for weight-load parity."""
    mlp = GPT2MLP(n_embd=32, dropout=0.0)
    assert mlp.c_fc.bias is not None
    assert mlp.c_proj.bias is not None


def test_hidden_expansion_is_4x():
    mlp = GPT2MLP(n_embd=32, dropout=0.0)
    assert mlp.c_fc.weight.shape == (4 * 32, 32)
    assert mlp.c_proj.weight.shape == (32, 4 * 32)
