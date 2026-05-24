"""Phase 1.6 — Newton-Schulz 5-step spectral normalization."""

import pytest
import torch

from model.nmm import newton_schulz5


def _spectral_norm(G: torch.Tensor) -> float:
    """Largest singular value of G (matrix 2-norm)."""
    return torch.linalg.svdvals(G.float()).max().item()


# ---------------------------------------------------------------------------
# Spectral norm bound across shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (16, 16),       # square
    (8, 32),        # wide
    (32, 8),        # tall
    (4, 64),        # very wide
    (64, 4),        # very tall
])
def test_spectral_norm_bound_for_any_shape(shape):
    """Post-NS spectral norm must be ~1 regardless of input aspect ratio."""
    G = torch.randn(shape) * 0.5
    G_ns = newton_schulz5(G)
    s = _spectral_norm(G_ns)
    # NS5 with Muon coefficients does NOT have sigma=1 as a fixed point
    # (a+b+c = 0.701, not 1). For random inputs the 5-step iteration
    # converges to a basin near sigma ~ 1.0-1.20 — bounded, not exact.
    # The bound is what we need for inner-loop stability; "approximately 1"
    # in PLAN.md / the diagram refers to this loose basin. See G230.
    assert 0.80 < s < 1.25, f"shape={shape}: out of post-NS5 basin, got {s}"


def test_spectral_norm_bound_with_batch_dim():
    """3D inputs [B, h, d] (per-sample gradients) — each batch row in basin."""
    G = torch.randn(4, 32, 8) * 0.5
    G_ns = newton_schulz5(G)
    for b in range(G.shape[0]):
        s = _spectral_norm(G_ns[b])
        assert 0.80 < s < 1.25, f"batch={b}: out of post-NS5 basin, got {s}"


# ---------------------------------------------------------------------------
# Transpose-tall guard (G198)
# ---------------------------------------------------------------------------

def test_transpose_guard_tall_matrix_W1_shape():
    """[4d, d] (W1/W_gate gradient shape) — must land in NS5 basin."""
    d, expansion = 8, 4
    G = torch.randn(d * expansion, d) * 0.3
    G_ns = newton_schulz5(G)
    assert G_ns.shape == G.shape
    assert 0.80 < _spectral_norm(G_ns) < 1.25


def test_transpose_guard_wide_matrix_W2_shape():
    """[d, 4d] (W2 gradient shape) — no transpose, must still land in basin."""
    d, expansion = 8, 4
    G = torch.randn(d, d * expansion) * 0.3
    G_ns = newton_schulz5(G)
    assert G_ns.shape == G.shape
    assert 0.80 < _spectral_norm(G_ns) < 1.25


def test_transpose_guard_preserves_output_shape():
    """Output shape must equal input shape (transpose-back is correct)."""
    for shape in [(8, 4), (4, 8), (16, 16), (32, 7), (7, 32)]:
        G = torch.randn(shape)
        assert newton_schulz5(G).shape == G.shape


# ---------------------------------------------------------------------------
# Theta-cancellation property (paper §3.2 / PLAN.md §1.5)
# ---------------------------------------------------------------------------

def test_pre_scaling_is_cancelled_by_NS():
    """NS divides by Frobenius norm — uniform pre-scaling cancels exactly.
    This is WHY theta must be applied POST-NS, not pre-NS.
    """
    G = torch.randn(16, 8)
    G_ns = newton_schulz5(G)
    G_scaled_ns = newton_schulz5(G * 7.3)  # arbitrary positive scale
    assert torch.allclose(G_ns, G_scaled_ns, atol=1e-4)


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_output_dtype_matches_input_dtype(dtype):
    G = torch.randn(8, 16).to(dtype)
    G_ns = newton_schulz5(G)
    assert G_ns.dtype == dtype


# ---------------------------------------------------------------------------
# G226 — internal matmuls run in fp32 under ambient bf16 autocast
# ---------------------------------------------------------------------------

def test_internal_matmul_runs_fp32_under_bf16_autocast():
    """Direct G226 defence: instrument __matmul__ and verify every NS matmul
    output is fp32 even with the parent autocast enabled in bf16.
    """
    seen_dtypes: list[torch.dtype] = []
    orig_matmul = torch.Tensor.__matmul__

    def patched_matmul(self, other):
        out = orig_matmul(self, other)
        seen_dtypes.append(out.dtype)
        return out

    torch.Tensor.__matmul__ = patched_matmul
    try:
        G = torch.randn(8, 64).to(torch.bfloat16)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            _ = newton_schulz5(G)
    finally:
        torch.Tensor.__matmul__ = orig_matmul

    assert len(seen_dtypes) >= 5, (
        f"Expected >=5 matmuls inside the NS iteration, saw {len(seen_dtypes)}"
    )
    assert all(d == torch.float32 for d in seen_dtypes), (
        f"At least one matmul ran outside fp32: {seen_dtypes}"
    )


def test_spectral_norm_bound_holds_under_bf16_autocast():
    """Behavioural twin of the matmul-dtype check: with G226 in place, the
    NS output's spectral norm should be ~1 even when called from within a
    bf16 autocast region. Without the autocast-disable wrap, the iteration
    would run in bf16 and the spectral norm would spread to [0.7, 1.4].
    """
    G = torch.randn(8, 64).to(torch.bfloat16) * 0.3
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        G_ns = newton_schulz5(G)
    s = _spectral_norm(G_ns)
    # Without G226 the iteration would run in bf16 and spread to [0.7, 1.4];
    # with G226 it stays in the fp32 NS5 basin (~0.80-1.25).
    assert 0.80 < s < 1.25, f"out of post-NS5 basin (G226 broken?), got {s}"


# ---------------------------------------------------------------------------
# Stress test — pathological inputs
# ---------------------------------------------------------------------------

def test_tiny_input_with_eps_does_not_explode():
    """G norm near zero — `+ eps` must keep the divide finite."""
    G = torch.randn(8, 16) * 1e-9
    G_ns = newton_schulz5(G)
    assert torch.isfinite(G_ns).all()


def test_large_input_converges():
    G = torch.randn(8, 16) * 1e4
    G_ns = newton_schulz5(G)
    # Initial Frobenius normalization cancels the 1e4 scale; output lands in basin.
    assert 0.80 < _spectral_norm(G_ns) < 1.25
