"""Phase 1.6 — Newton-Schulz 5-step spectral normalization."""

import pytest
import torch

from model.nmm import cans_stationary, gram_newton_schulz, newton_schulz5


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
    # in docs/PLAN.md / the diagram refers to this loose basin. See G230.
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
# Theta-cancellation property (paper §3.2 / docs/PLAN.md §1.5)
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


# ---------------------------------------------------------------------------
# CANS-stationary (3-step Chebyshev-optimised; arxiv 2506.10935)
# ---------------------------------------------------------------------------
#
# CANS coefficients (3.8641, -9.7196, 9.7101) are minimax-optimal over the
# post-F-norm sv range [0.0228, 0.0542] observed at gpt2_small NMM weight
# shapes (768x3072 and 3072x768). On those shapes CANS-3 outperforms NS5-5.
# At arbitrary shapes the sv range shifts and the polynomial may diverge —
# this is a documented recipe-specific limitation, NOT a bug. Convergence
# tests here therefore use the NMM weight shapes; structural tests
# (shape/dtype/finiteness) hold for any input.


def test_cans_better_than_ns5_at_nmm_shapes():
    """At the actual NMM weight shapes — the regime the coefficients were
    tuned for — CANS-3 must achieve lower orthogonalisation error than
    NS5-5 (benchmark: ~1.81 vs ~8.30). This is the core property that
    justifies CANS as the default for our recipe."""
    torch.manual_seed(42)
    for rows, cols in [(768, 3072), (3072, 768)]:
        G = torch.randn(rows, cols)
        X_ns5 = newton_schulz5(G)
        X_cans = cans_stationary(G)

        def _orth_err(X):
            Xf = X.float()
            gram = Xf.mT @ Xf if Xf.size(-2) >= Xf.size(-1) else Xf @ Xf.mT
            I = torch.eye(gram.size(-1), dtype=torch.float32)
            return (gram - I).norm().item()

        ns5_err = _orth_err(X_ns5)
        cans_err = _orth_err(X_cans)
        assert cans_err < ns5_err, (
            f"shape=({rows},{cols}): CANS error {cans_err:.3f} should be "
            f"< NS5 error {ns5_err:.3f}"
        )


def test_cans_spectral_norm_near_one_at_nmm_shapes():
    """Post-CANS spectral norm must land near 1 at the regime the
    coefficients were tuned for. Empirically tight (≈1) because the
    minimax objective drives σ → 1 exactly."""
    torch.manual_seed(42)
    for rows, cols in [(768, 3072), (3072, 768)]:
        G = torch.randn(rows, cols) * 0.5
        s = _spectral_norm(cans_stationary(G))
        assert 0.95 < s < 1.05, f"shape=({rows},{cols}): out of basin, got {s}"


@pytest.mark.parametrize("shape", [
    (768, 3072), (3072, 768),  # the shapes the coefficients are tuned for
    (16, 16), (8, 32), (32, 8),
])
def test_cans_output_is_finite(shape):
    """Even outside the tuned sv range, the output must be finite — no
    NaN/Inf — so a misconfigured run fails loud at training time instead
    of silently producing garbage."""
    G = torch.randn(shape) * 0.5
    assert torch.isfinite(cans_stationary(G)).all()


def test_cans_output_shape_matches_input():
    for shape in [(8, 4), (4, 8), (16, 16), (32, 7), (7, 32)]:
        G = torch.randn(shape)
        assert cans_stationary(G).shape == G.shape


def test_cans_with_batch_dim_at_nmm_shapes():
    """3D inputs [B, h, d] — each batch row must converge to the basin at
    NMM shapes (vmap of NS5 is the actual production call path)."""
    G = torch.randn(4, 768, 3072) * 0.5
    G_out = cans_stationary(G)
    for b in range(G.shape[0]):
        s = _spectral_norm(G_out[b])
        assert 0.95 < s < 1.05, f"batch={b}: out of basin, got {s}"


def test_cans_pre_scaling_is_cancelled():
    """F-norm normalisation must cancel positive pre-scaling, same as NS5."""
    G = torch.randn(768, 3072)
    a = cans_stationary(G)
    b = cans_stationary(G * 7.3)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_cans_output_dtype_matches_input_dtype(dtype):
    G = torch.randn(8, 16).to(dtype)
    assert cans_stationary(G).dtype == dtype


def test_cans_internal_matmul_runs_fp32_under_bf16_autocast():
    """G226 defence: CANS must keep its iteration in fp32 even under an
    ambient bf16 autocast. CANS-3 does 3 iterations × 2 matmuls = 6 inside
    matmuls (plus the F-norm has no matmul)."""
    seen_dtypes: list[torch.dtype] = []
    orig_matmul = torch.Tensor.__matmul__

    def patched(self, other):
        out = orig_matmul(self, other)
        seen_dtypes.append(out.dtype)
        return out

    torch.Tensor.__matmul__ = patched
    try:
        G = torch.randn(8, 64).to(torch.bfloat16)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            _ = cans_stationary(G)
    finally:
        torch.Tensor.__matmul__ = orig_matmul

    assert len(seen_dtypes) >= 6, (
        f"Expected >=6 matmuls inside the CANS-3 iteration, saw {len(seen_dtypes)}"
    )
    assert all(d == torch.float32 for d in seen_dtypes), (
        f"At least one matmul ran outside fp32: {seen_dtypes}"
    )


def test_cans_tiny_input_with_eps_does_not_explode():
    """G norm near zero — `+ eps` must keep the divide finite."""
    G = torch.randn(8, 16) * 1e-9
    assert torch.isfinite(cans_stationary(G)).all()


# ---------------------------------------------------------------------------
# gram_newton_schulz — local Gram-iteration impl with POLAR_EXPRESS coefficients
# ---------------------------------------------------------------------------


def test_gram_ns5_spectral_norm_near_one_at_nmm_shapes():
    """POLAR_EXPRESS coefficients should drive spectral norm toward 1 at
    rectangular NMM gradient shapes."""
    torch.manual_seed(0)
    for rows, cols in [(768, 3072), (3072, 768)]:
        G = torch.randn(rows, cols) * 0.5
        s = _spectral_norm(gram_newton_schulz(G))
        assert 0.70 < s < 1.30, f"shape=({rows},{cols}): out of basin, got {s}"


def test_gram_ns5_with_batch_dim():
    """3D inputs [B, h, d] — each batch row must converge to the basin."""
    G = torch.randn(4, 768, 3072) * 0.5
    out = gram_newton_schulz(G)
    for b in range(G.shape[0]):
        s = _spectral_norm(out[b])
        assert 0.70 < s < 1.30, f"batch={b}: out of basin, got {s}"


def test_gram_ns5_output_shape_matches_input():
    for shape in [(8, 4), (4, 8), (16, 16), (32, 7), (7, 32), (2, 4, 8)]:
        G = torch.randn(*shape)
        assert gram_newton_schulz(G).shape == G.shape


def test_gram_ns5_pre_scaling_is_cancelled():
    """F-norm normalisation must cancel positive pre-scaling."""
    G = torch.randn(768, 3072)
    a = gram_newton_schulz(G)
    b = gram_newton_schulz(G * 7.3)
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_gram_ns5_output_dtype_matches_input_dtype(dtype):
    G = torch.randn(8, 16).to(dtype)
    assert gram_newton_schulz(G).dtype == dtype


def test_gram_ns5_internal_matmul_runs_fp32_under_bf16_autocast():
    """G226 defence: gram-NS5 must keep its iteration in fp32 even under
    an ambient bf16 autocast."""
    seen_dtypes: list[torch.dtype] = []
    orig_matmul = torch.Tensor.__matmul__

    def patched(self, other):
        out = orig_matmul(self, other)
        seen_dtypes.append(out.dtype)
        return out

    torch.Tensor.__matmul__ = patched
    try:
        G = torch.randn(8, 64).to(torch.bfloat16)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            _ = gram_newton_schulz(G)
    finally:
        torch.Tensor.__matmul__ = orig_matmul

    # 5 iterations × multiple matmuls per iter, plus the initial X@X^T
    # and final Q@X. At least 8 matmuls total.
    assert len(seen_dtypes) >= 8, (
        f"Expected >=8 matmuls inside gram-NS iteration, saw {len(seen_dtypes)}"
    )
    assert all(d == torch.float32 for d in seen_dtypes), (
        f"At least one matmul ran outside fp32: {seen_dtypes}"
    )


def test_gram_ns5_normalize_is_out_of_place():
    """Load-bearing autograd correctness: the F-norm divide must be
    out-of-place. The upstream library's `X /= ...` mutates a tensor
    that's saved for the norm's backward, breaking autograd's
    version-tracking under torch.compile + AOT. Verify the local impl
    leaves the input's _version untouched."""
    G = torch.randn(8, 16, requires_grad=True)
    v_before = G._version
    Y = gram_newton_schulz(G)
    assert G._version == v_before, (
        f"gram_newton_schulz mutated its input in-place "
        f"(version {v_before} -> {G._version})"
    )
    # And backward must run without raising the in-place error.
    Y.sum().backward()
    assert G.grad is not None


def test_gram_ns5_tiny_input_with_eps_does_not_explode():
    G = torch.randn(8, 16) * 1e-9
    assert torch.isfinite(gram_newton_schulz(G)).all()


@pytest.mark.gpu
def test_gram_ns5_compile_plus_backward():
    """Regression test for the bug that motivated the local impl:
    `torch.compile` + the NMM forward + loss.backward() must not raise
    an in-place autograd error inside gram_newton_schulz. This test
    failed against the upstream library; it must pass against ours."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from config import TitansConfig
    from model.titans_gpt2 import TitansMAGGPT2

    cfg = TitansConfig.gpt2_small(
        n_layer=2, chunk_size=128,
        nmm_block_size=64,
        nmm_state_dtype="bf16",
        nmm_detach_state_between_blocks=True,
        nmm_use_gram_ns5=True,
    )
    m = TitansMAGGPT2(cfg).cuda()
    m = torch.compile(m, mode="default", dynamic=False)
    m.train()
    x = torch.randint(0, cfg.vocab_size, (1, 128)).cuda()
    boundaries = torch.zeros(1, 128, dtype=torch.bool).cuda()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = m(x, None, boundaries)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        x[:, 1:].reshape(-1),
    )
    loss.backward()  # must not raise
    assert torch.isfinite(loss).item()
