import torch
import torch.nn.functional as F

from model.nmm import NMMProjection


def test_output_shape_matches_input():
    proj = NMMProjection(n_embd=32, kernel_size=4)
    x = torch.randn(2, 16, 32)
    y = proj(x)
    assert y.shape == x.shape


def test_call_site_l2_normalization_for_q_and_k():
    """q_hat, k_hat are L2-normalized at the call site (not inside the module)."""
    proj = NMMProjection(n_embd=32, kernel_size=4)
    x = torch.randn(2, 16, 32)
    q_hat = F.normalize(F.silu(proj(x)), dim=-1)
    norms = q_hat.norm(dim=-1)
    # F.normalize with default eps=1e-12 may yield near-zero rows for all-suppressed
    # silu output, but our random init makes that virtually impossible. Assert ~= 1.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_call_site_v_not_l2_normalized():
    """v_t is silu'd but NOT L2-normalized (unconstrained magnitude)."""
    proj = NMMProjection(n_embd=32, kernel_size=4)
    x = torch.randn(2, 16, 32)
    v = F.silu(proj(x))
    norms = v.norm(dim=-1)
    # v can have any positive magnitude; sanity check that it is not unit-normed.
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_no_silu_inside_module():
    """If SiLU were inside NMMProjection, call-site silu(proj(x)) would be silu(silu(x))
    — different from silu(x) for non-zero input. Set linear=I and conv=identity to
    isolate: proj(x) must equal x exactly.
    """
    proj = NMMProjection(n_embd=4, kernel_size=1)
    with torch.no_grad():
        proj.linear.weight.copy_(torch.eye(4))
        proj.conv.conv.weight.fill_(0)
        # depthwise conv weights shape [out=4, in=1, k=1]; set to 1 so output == input
        proj.conv.conv.weight.fill_(1.0)
    x = torch.tensor([[[-1.0, -0.5, 0.5, 1.0]]])
    y = proj(x)
    assert torch.allclose(y, x), f"proj(x) should equal x exactly, got {y}"
    # If module had inner silu, silu(proj(x)) would be silu(silu(x)); check it's silu(x).
    expected_call_site = F.silu(x)
    assert torch.allclose(F.silu(proj(x)), expected_call_site)


def test_submodule_names_for_optimizer_routing():
    """`linear` and `conv` submodule names matter for §4.1 NMM optimizer routing.
    Renaming to anything containing 'norm'/'bias'/'gamma' would misroute to no_decay.
    """
    proj = NMMProjection(n_embd=8, kernel_size=4)
    names = set(dict(proj.named_parameters()).keys())
    assert "linear.weight" in names
    assert "conv.conv.weight" in names
    # No forbidden substrings in any submodule path.
    for name in names:
        for bad in ("norm", "bias", "gamma"):
            assert bad not in name, (
                f"Param name {name!r} contains '{bad}' — would misroute "
                f"to a no_decay group in §4.1."
            )


def test_strict_causality_preserved_through_linear():
    """Linear is pointwise in T; the conv enforces causality. Verify end-to-end."""
    proj = NMMProjection(n_embd=4, kernel_size=4)
    x1 = torch.randn(1, 16, 4)
    x2 = x1.clone()
    x2[0, 8:, :] = torch.randn(8, 4)
    with torch.no_grad():
        y1 = proj(x1)
        y2 = proj(x2)
    assert torch.equal(y1[:, :8, :], y2[:, :8, :])
