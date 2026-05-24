"""Phase 1.5 — torch.func gradient cached per-sample function."""

import torch
import torch.nn.functional as F
from torch.func import functional_call

from model.nmm import NeuralMemoryModule, _make_grad_fn


# ---------------------------------------------------------------------------
# Output shape and dict structure
# ---------------------------------------------------------------------------

def test_per_sample_grad_returns_dict_keyed_like_params():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, spectral_norm=True)
    B, d = 3, 8
    k_hat = torch.randn(B, d)
    v = torch.randn(B, d)
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))
    grads = nmm.per_sample_grad_fn(M, k_hat, v)
    assert isinstance(grads, dict)
    assert set(grads.keys()) == set(M.keys())
    for k in grads:
        assert grads[k].shape == M[k].shape


# ---------------------------------------------------------------------------
# Correctness vs torch.autograd
# ---------------------------------------------------------------------------

def test_per_sample_grad_matches_torch_autograd():
    """For B=1 the per-sample grad must match plain autograd through functional_call."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2, spectral_norm=True)
    B, d = 1, 4
    k_hat = torch.randn(B, d)
    v = torch.randn(B, d)
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))

    grads_func = nmm.per_sample_grad_fn(M, k_hat, v)

    params = {k: M[k][0].detach().clone().requires_grad_(True) for k in M}
    pred = functional_call(nmm.memory_mlp, params, k_hat[0])
    loss = F.mse_loss(pred, v[0], reduction="sum")
    grads_auto = torch.autograd.grad(loss, list(params.values()))
    grads_auto_dict = {k: g for k, g in zip(params, grads_auto)}

    for k in grads_func:
        assert torch.allclose(
            grads_func[k][0], grads_auto_dict[k], atol=1e-5
        ), f"key={k}: vmap-grad and torch.autograd.grad disagree"


def test_per_sample_grad_is_per_sample_not_summed():
    """Each batch row's gradient must reflect only its own (k_hat, v)."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2, spectral_norm=True)
    B, d = 2, 4
    k_hat = torch.randn(B, d)
    v = torch.randn(B, d)
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))

    grads_batched = nmm.per_sample_grad_fn(M, k_hat, v)

    # Compute each row's gradient via a separate B=1 call; must match the batched result.
    for b in range(B):
        M_b = {k: M[k][b:b + 1] for k in M}
        g_b = nmm.per_sample_grad_fn(M_b, k_hat[b:b + 1], v[b:b + 1])
        for k in grads_batched:
            assert torch.allclose(grads_batched[k][b], g_b[k][0], atol=1e-5)


# ---------------------------------------------------------------------------
# G160 — reduction switch matches spectral_norm flag
# ---------------------------------------------------------------------------

def test_reduction_switch_scales_gradient_by_inverse_d():
    """spectral_norm=False ('mean' reduction) gradients are ~1/d_model the
    Frobenius norm of spectral_norm=True ('sum' reduction) gradients —
    confirms the wiring is real, not just documented.
    """
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, spectral_norm=True)
    d = nmm.n_embd
    B = 4
    k_hat = torch.randn(B, d)
    v = torch.randn(B, d)
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))

    grad_sum = nmm.per_sample_grad_fn(M, k_hat, v)

    # Build a second grad_fn against the same MLP but with reduction='mean'.
    grad_fn_mean = _make_grad_fn(nmm.memory_mlp, spectral_norm=False)
    grad_mean = grad_fn_mean(M, k_hat, v)

    for k in grad_sum:
        ratio = grad_mean[k].norm() / grad_sum[k].norm()
        # mse_loss reduction='mean' divides the squared-error sum by d.
        # Expect grad_mean ≈ grad_sum / d in Frobenius norm.
        assert abs(ratio.item() - 1.0 / d) < 1e-5, (
            f"key={k}: expected ratio ~{1.0 / d:.5f}, got {ratio.item():.5f}"
        )


# ---------------------------------------------------------------------------
# Caching behavior
# ---------------------------------------------------------------------------

def test_grad_fn_is_cached_not_rebuilt_per_call():
    """per_sample_grad_fn must be the SAME callable across multiple .__init__
    inspections — not rebuilt per forward."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    fn_id_1 = id(nmm.per_sample_grad_fn)
    fn_id_2 = id(nmm.per_sample_grad_fn)
    assert fn_id_1 == fn_id_2


def test_spectral_norm_at_init_locked():
    """_spectral_norm_at_init records the construction value so step()/
    forward_chunk can detect drift after mutation."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2, spectral_norm=True)
    assert nmm._spectral_norm_at_init is True
    nmm2 = NeuralMemoryModule(n_embd=4, expansion=2, spectral_norm=False)
    assert nmm2._spectral_norm_at_init is False


# ---------------------------------------------------------------------------
# _batched_retrieve cache
# ---------------------------------------------------------------------------

def test_batched_retrieve_returns_per_sample_y():
    """_batched_retrieve(M, q) — for each batch row b, returns memory_mlp(M[b]; q[b])."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    B, d = 3, 4
    q = torch.randn(B, d)
    M = nmm._build_init_M(B=B, device=torch.device("cpu"))
    y = nmm._batched_retrieve(M, q)
    assert y.shape == (B, d)

    # Match a manual single-sample functional_call.
    for b in range(B):
        params_b = {k: M[k][b] for k in M}
        y_b = functional_call(nmm.memory_mlp, params_b, q[b:b + 1]).squeeze(0)
        assert torch.allclose(y[b], y_b, atol=1e-6)
