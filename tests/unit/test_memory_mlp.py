import math

import pytest
import torch

from model.nmm import MemoryMLP, NeuralMemoryModule


# ---------------------------------------------------------------------------
# MemoryMLP (the gated MLP itself)
# ---------------------------------------------------------------------------

def test_output_shape_matches_input():
    mlp = MemoryMLP(d=32, expansion=4)
    x = torch.randn(2, 16, 32)
    y = mlp(x)
    assert y.shape == x.shape


def test_residualnorm_passes_x_through_when_W2_is_zero():
    """With W2.weight=0 and bias=False, W2(h)=0 -> norm(0)=0 -> output = 0 + x = x."""
    mlp = MemoryMLP(d=32, expansion=4)
    with torch.no_grad():
        mlp.W2.weight.zero_()
    x = torch.randn(2, 16, 32)
    y = mlp(x)
    assert torch.allclose(y, x, atol=1e-6)


@pytest.mark.parametrize("d,expansion", [(32, 1), (32, 2), (32, 4), (64, 4), (16, 8)])
def test_xavier_uniform_init_via_NMM_constructor(d: int, expansion: int):
    """NMM.__init__ Xavier-uniform-inits MemoryMLP weights.

    Xavier-uniform: U(-a, a) with a = sqrt(6/(fan_in+fan_out)).
    Variance = a^2/3 = 2/(fan_in+fan_out), so std = sqrt(2/(fan_in+fan_out)).
    """
    nmm = NeuralMemoryModule(n_embd=d, expansion=expansion)
    for name, weight, fan_in, fan_out in [
        ("W1",     nmm.memory_mlp.W1.weight,     d,             d * expansion),
        ("W_gate", nmm.memory_mlp.W_gate.weight, d,             d * expansion),
        ("W2",     nmm.memory_mlp.W2.weight,     d * expansion, d),
    ]:
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        actual_std = weight.std().item()
        # 5% tolerance per docs/TEST_PLAN.md.
        assert abs(actual_std - expected_std) / expected_std < 0.05, (
            f"{name}: expected std ~{expected_std:.4f}, got {actual_std:.4f}"
        )


def test_memorymlp_has_no_bias_on_W_layers():
    mlp = MemoryMLP(d=8, expansion=2)
    assert mlp.W1.bias is None
    assert mlp.W_gate.bias is None
    assert mlp.W2.bias is None


def test_layernorm_is_present_and_d_sized():
    mlp = MemoryMLP(d=32, expansion=4)
    assert isinstance(mlp.norm, torch.nn.LayerNorm)
    assert mlp.norm.normalized_shape == (32,)


# ---------------------------------------------------------------------------
# out_scale init: zeros for finetune_mode=True, ones for False (G123)
# ---------------------------------------------------------------------------

def test_out_scale_init_zeros_in_finetune_mode():
    nmm = NeuralMemoryModule(n_embd=16, finetune_mode=True)
    assert torch.equal(nmm.out_scale, torch.zeros(16))


def test_out_scale_init_ones_when_not_finetune_mode():
    nmm = NeuralMemoryModule(n_embd=16, finetune_mode=False)
    assert torch.equal(nmm.out_scale, torch.ones(16))


def test_out_scale_is_a_parameter():
    nmm = NeuralMemoryModule(n_embd=8, finetune_mode=True)
    assert isinstance(nmm.out_scale, torch.nn.Parameter)
    assert nmm.out_scale.requires_grad


# ---------------------------------------------------------------------------
# _build_init_M / init_state
# ---------------------------------------------------------------------------

def test_build_init_M_keys_are_only_2D_weights():
    """norm.weight / norm.bias must NOT be present — they are not recurrent state.
    NS5 operates via Frobenius norm over (-2,-1), which is only meaningful for 2D.
    """
    nmm = NeuralMemoryModule(n_embd=8, expansion=2)
    M = nmm._build_init_M(B=2, device=torch.device("cpu"))
    assert set(M.keys()) == {"W1.weight", "W_gate.weight", "W2.weight"}
    assert "norm.weight" not in M
    assert "norm.bias" not in M


def test_build_init_M_shapes_are_per_sample():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2)
    B = 4
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))
    # h = d * expansion = 16
    assert M["W1.weight"].shape == (B, 16, 8)
    assert M["W_gate.weight"].shape == (B, 16, 8)
    assert M["W2.weight"].shape == (B, 8, 16)


def test_build_init_M_seeded_from_memory_mlp_weights():
    """At init, M[k][b] must equal memory_mlp.<k> for every b — the meta-learned
    initial state is broadcast across the batch."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2)
    B = 3
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))
    for b in range(B):
        assert torch.equal(M["W1.weight"][b], nmm.memory_mlp.W1.weight)
        assert torch.equal(M["W_gate.weight"][b], nmm.memory_mlp.W_gate.weight)
        assert torch.equal(M["W2.weight"][b], nmm.memory_mlp.W2.weight)


def test_build_init_M_is_cloned_not_a_view():
    """Per-sample entries must be independent allocations — mutating one must
    not affect the source weight or any other batch row."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2)
    M = nmm._build_init_M(B=3, device=torch.device("cpu"))
    src_before = nmm.memory_mlp.W1.weight.clone()
    M["W1.weight"][0].add_(1.0)
    assert torch.equal(nmm.memory_mlp.W1.weight, src_before)  # source untouched
    assert not torch.equal(M["W1.weight"][0], M["W1.weight"][1])  # rows independent


def test_init_state_returns_M_and_zero_S():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2)
    M, S, _ = nmm.init_state(B=2, device=torch.device("cpu"))
    assert set(M.keys()) == set(S.keys())
    for k in M:
        assert M[k].shape == S[k].shape
        assert torch.all(S[k] == 0)


def test_init_state_shapes_match_build_init_M():
    nmm = NeuralMemoryModule(n_embd=16, expansion=4)
    M, S, _ = nmm.init_state(B=2, device=torch.device("cpu"))
    M_direct = nmm._build_init_M(B=2, device=torch.device("cpu"))
    for k in M:
        assert M[k].shape == M_direct[k].shape
