"""DeltaProductMemory unit tests — TPTT formulation.

Covers projections (SiLU+L2 on Q/K, V scaling, vector β from CausalAvgPool),
the sequential reference path, the chunkwise WY parallel path with vector
β, multi-head fusion, and the auxiliary-tensor cache.
"""

import math

import pytest
import torch

from model.delta_product import (
    DeltaProductMemory,
    _causal_avg_pool_3,
    _chunkwise_aux_tensors,
)


# -- Module construction --------------------------------------------------


def test_construct_rejects_bad_order():
    with pytest.raises(ValueError, match="order must be >= 1"):
        DeltaProductMemory(n_embd=8, order=0)


def test_construct_rejects_bad_block_size():
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DeltaProductMemory(n_embd=8, order=2, block_size=0)


def test_construct_rejects_bad_n_heads():
    with pytest.raises(ValueError, match="n_heads must be >= 1"):
        DeltaProductMemory(n_embd=8, n_heads=0)


def test_construct_rejects_indivisible_n_embd():
    with pytest.raises(ValueError, match="must be divisible"):
        DeltaProductMemory(n_embd=10, n_heads=3)


def test_param_count_scales_with_order():
    """Each extra order adds one K and one V projection — [H, hd, hd] each.
    β is NOT a learnable parameter under the TPTT recipe (it's computed
    from K via fixed CausalAvgPool + sigmoid)."""
    n_embd = 16
    p1 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=1).parameters())
    p2 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=2).parameters())
    p3 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=3).parameters())
    # n_heads=1, head_dim=16. Per extra order: K [1, 16, 16] + V [1, 16, 16].
    H, hd = 1, n_embd
    extra_per_order = 2 * H * hd * hd
    assert p2 - p1 == extra_per_order
    assert p3 - p2 == extra_per_order


def test_finetune_mode_zero_inits_out_scale():
    """finetune_mode=True must zero out_scale so y_mem = 0 at step 0."""
    m_ft = DeltaProductMemory(n_embd=8, order=2, finetune_mode=True)
    m_scratch = DeltaProductMemory(n_embd=8, order=2, finetune_mode=False)
    assert torch.all(m_ft.out_scale == 0.0)
    assert torch.all(m_scratch.out_scale == 1.0)


def test_finetune_mode_produces_zero_output_at_init():
    """y_mem must be exactly zero at step 0 in finetune_mode (residual
    of pretrained backbone preserved). out_scale=0 zeros y regardless
    of out_proj's init."""
    torch.manual_seed(0)
    m = DeltaProductMemory(n_embd=8, order=2, finetune_mode=True)
    state = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, 8)
    y, _ = m.forward_chunk(x, state, doc_boundaries=None)
    assert torch.allclose(y, torch.zeros_like(y))


# -- init_state -----------------------------------------------------------


def test_init_state_shape_and_zero_singlehead():
    """State is (M, k_buf_list) where M is the recurrent matrix and
    k_buf_list is the per-order CausalAvgPool context buffer."""
    m = DeltaProductMemory(n_embd=8, order=2)
    state = m.init_state(B=3, device="cpu")
    assert isinstance(state, tuple) and len(state) == 2
    M, k_buf = state
    assert M.shape == (3, 1, 8, 8)  # [B, n_heads=1, hd, hd]
    assert torch.all(M == 0.0)
    # k_buf: list of `order` tensors, each [B, 2, n_heads, head_dim].
    assert len(k_buf) == 2  # order=2
    for kb in k_buf:
        assert kb.shape == (3, 2, 1, 8)
        assert torch.all(kb == 0.0)


def test_init_state_shape_multihead():
    m = DeltaProductMemory(n_embd=12, n_heads=3, order=2)
    state = m.init_state(B=2, device="cpu")
    assert len(state) == 2
    M, k_buf = state
    assert M.shape == (2, 3, 4, 4)
    assert torch.all(M == 0.0)
    for kb in k_buf:
        assert kb.shape == (2, 2, 3, 4)


# -- CausalAvgPool helper -------------------------------------------------


def test_causal_avg_pool_3_replicate_at_start():
    """At t=0 and t=1, the pool replicates x[:, 0] so the boundary
    entries equal x[:, 0] (rather than averaging with zeros)."""
    x = torch.tensor([
        [[1.0], [2.0], [3.0], [4.0]],
    ])  # [B=1, T=4, F=1]
    y = _causal_avg_pool_3(x)
    # t=0: (1+1+1)/3 = 1.0
    # t=1: (1+1+2)/3 = 1.333
    # t=2: (1+2+3)/3 = 2.0
    # t=3: (2+3+4)/3 = 3.0
    expected = torch.tensor([[[1.0], [4 / 3.0], [2.0], [3.0]]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_causal_avg_pool_3_at_T_one():
    """T=1: pool returns x[:, 0] unchanged (replicate-pad both prior
    positions to the same value)."""
    x = torch.tensor([[[1.0, 2.0, 3.0]]])  # [1, 1, 3]
    y = _causal_avg_pool_3(x)
    assert torch.allclose(y, x)


# -- Projection invariants (TPTT formulation) -----------------------------


def test_projections_q_and_k_are_silu_l2_normalized():
    """After SiLU+L2, q and k must be unit-norm along the last dim.
    V is NOT normalized (just scaled)."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(2, 5, d)
    state = m.init_state(B=2, device="cpu")
    _, k_buf = state
    q, ks, vs, bs, _ = m._project_kvb(x, k_buf)

    q_norms = q.norm(dim=-1)  # [B, T, H]
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)
    for k in ks:
        k_norms = k.norm(dim=-1)
        assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)


def test_projection_v_scaled_by_inv_sqrt_head_dim():
    """V is scaled by 1/√head_dim — verify magnitude vs an unscaled
    reference projection."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, n_heads=1, order=1, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, k_buf = m.init_state(B=2, device="cpu")
    _, _, vs, _, _ = m._project_kvb(x, k_buf)

    x_h = x.view(2, 5, 1, d)
    v_raw = torch.einsum("bthd,hde->bthe", x_h, m.v_proj_weights[0])
    v_expected = v_raw * (1.0 / math.sqrt(d))
    assert torch.allclose(vs[0], v_expected, atol=1e-6)


def test_beta_is_sigmoid_of_pooled_raw_k():
    """β must equal σ(CausalAvgPool3(k_raw_extended_with_buffer)) —
    vector per component. With a fresh (zero) k_buf, the effective
    pool input is [0, 0, k_raw]; after slicing the leading 2 positions
    off, β matches σ(pool(concat([zeros_2, k_raw]))[2:]).

    Note: with zero buffer, pool(k_raw)[t] for t≥2 matches the original
    no-buffer behavior exactly. For t=0,1 the buffer's zero values
    contribute (different from replicate-padding the chunk's first
    value). This is the price of cross-call pool continuity."""
    torch.manual_seed(0)
    d = 8
    m = DeltaProductMemory(n_embd=d, n_heads=1, order=1, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, k_buf = m.init_state(B=2, device="cpu")
    _, _, _, bs, _ = m._project_kvb(x, k_buf)

    # Reproduce: project K, prepend zero buffer, pool, sigmoid, slice off [:2].
    x_h = x.view(2, 5, 1, d)
    k_raw = torch.einsum("bthd,hde->bthe", x_h, m.k_proj_weights[0])
    k_raw_ext = torch.cat([k_buf[0], k_raw], dim=1)
    beta_full = torch.sigmoid(_causal_avg_pool_3(k_raw_ext))
    beta_expected = beta_full[:, 2:]
    assert torch.allclose(bs[0], beta_expected, atol=1e-6)
    # β is a vector per head, NOT a scalar.
    assert bs[0].shape == (2, 5, 1, d)


def test_project_kvb_updates_k_buffer():
    """The returned k_buf_out is the last 2 raw-K values of the combined
    (buffer + chunk) sequence, ready for the next call."""
    torch.manual_seed(0)
    d, order = 8, 2
    m = DeltaProductMemory(n_embd=d, n_heads=1, order=order, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, k_buf_in = m.init_state(B=2, device="cpu")
    _, _, _, _, k_buf_out = m._project_kvb(x, k_buf_in)

    # For each order, k_buf_out should equal the last 2 raw K values
    # of the current chunk (since the buffer started at zero, T=5 >= 2,
    # the new buffer is just k_raw[:, -2:]).
    x_h = x.view(2, 5, 1, d)
    for i in range(order):
        k_raw_i = torch.einsum("bthd,hde->bthe", x_h, m.k_proj_weights[i])
        assert torch.allclose(k_buf_out[i], k_raw_i[:, -2:], atol=1e-6)


# -- Output: RMSNorm + out_proj + out_scale -------------------------------


def test_output_is_rmsnorm_bounded():
    """After RMSNorm, the row-magnitude is bounded (mean(y²) ≈ 1 before
    out_proj). End-to-end, with random init + scratch mode, the output
    magnitudes shouldn't blow up."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 7, d)
    y, _ = m.forward_chunk(x, s)
    # Final y is out_proj(RMSNorm(y_raw)) * out_scale. With Linear init
    # at kaiming and out_scale=1, magnitude should be O(1) not O(1000).
    assert torch.isfinite(y).all()
    assert y.abs().max() < 100.0, f"y unexpectedly large: {y.abs().max()}"


# -- State threading ------------------------------------------------------


def test_state_continuity_split_chunk_matches_full():
    """forward_chunk(x[:, :T1]) → state → forward_chunk(x[:, T1:])
    must equal forward_chunk(x) outputs and final state."""
    torch.manual_seed(0)
    d = 6
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(2, 7, d)

    s0 = m.init_state(B=2, device="cpu")
    y_full, s_full = m.forward_chunk(x, s0)

    s0b = m.init_state(B=2, device="cpu")
    y1, s_mid = m.forward_chunk(x[:, :3, :], s0b)
    y2, s_after = m.forward_chunk(x[:, 3:, :], s_mid)
    y_split = torch.cat([y1, y2], dim=1)

    assert torch.allclose(y_full, y_split, atol=1e-5)
    assert torch.allclose(s_full[0], s_after[0], atol=1e-5)


def test_step_with_conv_matches_forward_chunk_per_position():
    """step_with_conv called T times must equal forward_chunk on the
    same T-token chunk."""
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(3, 5, d)

    s = m.init_state(B=3, device="cpu")
    y_chunk, s_chunk = m.forward_chunk(x, s)

    s2 = m.init_state(B=3, device="cpu")
    y_steps = []
    for t in range(5):
        y_t, s2 = m.step_with_conv(x[:, t, :], s2)
        y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_chunk, y_step, atol=1e-5)
    assert torch.allclose(s_chunk[0], s2[0], atol=1e-5)


# -- Doc boundaries -------------------------------------------------------


def test_doc_boundaries_reset_M_to_zero():
    """At a doc-boundary position, M must reset to zero BEFORE that
    token's update. Verified via the sequential-vs-chunkwise path's
    end-to-end equivalence with boundaries in
    `test_blockwise_with_doc_boundaries_matches_sequential`.

    Note on what we DON'T test: equality between (full chunk with
    boundary at t=2) and (fresh start on x[:, 2:]). Those produce
    slightly different M because the CausalAvgPool β at positions 2-3
    of the full chunk leaks 2 tokens of pre-boundary k_raw context.
    This is a property of TPTT's pool-based gating — accepted for the
    experiment; the leak is 2 positions wide vs typical scenario
    lengths of 100+.
    """
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(n_embd=d, order=1, finetune_mode=False)

    # With two DIFFERENT pre-boundary inputs but the SAME post-boundary
    # content, the M state after the boundary should be CLOSE (within
    # ~the pool's 2-token leak) regardless of what came before.
    x_post = torch.randn(1, 3, d)
    x1 = torch.cat([torch.randn(1, 2, d), x_post], dim=1)  # different prefix
    x2 = torch.cat([torch.randn(1, 2, d) * 5, x_post], dim=1)  # very different prefix
    db = torch.tensor([[False, False, True, False, False]])

    s1 = m.init_state(B=1, device="cpu")
    _, (M1, _) = m.forward_chunk(x1, s1, doc_boundaries=db)
    s2 = m.init_state(B=1, device="cpu")
    _, (M2, _) = m.forward_chunk(x2, s2, doc_boundaries=db)

    # M's evolved differently in the pre-boundary region but the
    # boundary fully resets M before t=2's writes. The post-boundary
    # M values should agree to within the pool's 2-token leak.
    # Empirically the diff is bounded by ~0.2 at d=4, order=1 with
    # 5x scaled prefix; loose tolerance reflects the leak.
    assert torch.allclose(M1, M2, atol=0.5)


# -- Gradient flow --------------------------------------------------------


def test_gradient_flows_to_all_projections():
    """Backward through forward_chunk must reach every learnable
    parameter (Q, K, V, out_proj, out_scale)."""
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(2, 3, d, requires_grad=True)
    s = m.init_state(B=2, device="cpu")
    y, _ = m.forward_chunk(x, s)
    y.sum().backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().max() > 0, f"zero grad on {name}"


def test_gradient_flows_to_input():
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(2, 3, d, requires_grad=True)
    s = m.init_state(B=2, device="cpu")
    y, _ = m.forward_chunk(x, s)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0


# -- Shape ----------------------------------------------------------------


def test_output_shape_matches_input():
    d = 8
    m = DeltaProductMemory(n_embd=d, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 7, d)
    y, (M_out, _) = m.forward_chunk(x, s)
    assert y.shape == (2, 7, d)
    assert M_out.shape == (2, 1, d, d)


# -- Blockwise (chunkwise WY) parallel path -------------------------------


def _make_pair(
    d=6, n_heads=1, order=2, finetune_mode=False, block_size=8, seed=0,
):
    """Build two DeltaProductMemory instances with identical weights —
    one in sequential mode, one in blockwise mode."""
    torch.manual_seed(seed)
    m_seq = DeltaProductMemory(
        n_embd=d, n_heads=n_heads, order=order,
        finetune_mode=finetune_mode, block_size=1,
    )
    torch.manual_seed(seed)
    m_blk = DeltaProductMemory(
        n_embd=d, n_heads=n_heads, order=order,
        finetune_mode=finetune_mode, block_size=block_size,
    )
    return m_seq, m_blk


def test_blockwise_matches_sequential_no_boundaries_order1():
    """Order=1 chunkwise WY form (with vector β) is bit-equivalent to
    per-token sequential."""
    m_seq, m_blk = _make_pair(d=4, order=1, block_size=8, seed=1)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_no_boundaries_order2():
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=2)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_order3():
    m_seq, m_blk = _make_pair(d=4, order=3, block_size=8, seed=3)
    x = torch.randn(2, 5, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_with_nonzero_initial_state():
    """The chunkwise formula must handle non-zero initial M correctly
    with vector β."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=4)
    x = torch.randn(2, 5, 4)
    s0_seq = m_seq.init_state(B=2, device="cpu")
    _, s_warm = m_seq.forward_chunk(torch.randn(2, 3, 4), s0_seq)
    (M_warm, _) = s_warm
    state_in = (M_warm, _)
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, state_in)
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, state_in)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_boundary_at_position_zero_resets_M():
    """Regression for the chunkwise-vs-sequential divergence at
    doc_boundaries[:, 0]=True (matches sequential path's
    unconditional reset)."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=42)
    x = torch.randn(2, 5, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    _, s_warm = m_seq.forward_chunk(torch.randn(2, 3, 4), s0)
    db = torch.tensor([
        [True,  False, False, False, False],
        [False, False, False, False, False],
    ])
    y_seq, s_seq = m_seq.forward_chunk(x, s_warm, doc_boundaries=db)
    y_blk, s_blk = m_blk.forward_chunk(x, s_warm, doc_boundaries=db)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


def test_blockwise_with_doc_boundaries_matches_sequential():
    """Boundary-aware chunkwise must match sequential when boundaries
    are inside the chunk."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=5)
    x = torch.randn(2, 6, 4)
    db = torch.tensor([
        [False, False, True, False, False, False],
        [False, True,  False, False, True,  False],
    ])
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, s0, doc_boundaries=db)
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, s0, doc_boundaries=db)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_state_continuity_split_chunk():
    _, m_blk = _make_pair(d=4, order=2, block_size=8, seed=6)
    x = torch.randn(2, 7, 4)
    s0 = m_blk.init_state(B=2, device="cpu")
    y_full, s_full = m_blk.forward_chunk(x, s0)

    s0b = m_blk.init_state(B=2, device="cpu")
    y1, s_mid = m_blk.forward_chunk(x[:, :3, :], s0b)
    y2, s_after = m_blk.forward_chunk(x[:, 3:, :], s_mid)
    y_split = torch.cat([y1, y2], dim=1)

    assert torch.allclose(y_full, y_split, atol=1e-5)
    assert torch.allclose(s_full[0], s_after[0], atol=1e-5)


def test_blockwise_gradient_flow():
    _, m_blk = _make_pair(d=4, order=2, block_size=8, seed=7)
    x = torch.randn(2, 5, 4, requires_grad=True)
    s = m_blk.init_state(B=2, device="cpu")
    y, _ = m_blk.forward_chunk(x, s)
    y.sum().backward()
    for name, p in m_blk.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().max() > 0, f"zero grad on {name}"
    assert x.grad is not None
    assert x.grad.abs().max() > 0


# -- Numerical stability at training-realistic scale ----------------------


def test_blockwise_stable_at_training_scale_long_chunk():
    """Regression: at training chunk length (T·N >> 1), the WY
    triangular solve diverges unless keys are L2-normalized. With
    SiLU+L2 on K (TPTT formulation), stability is preserved."""
    torch.manual_seed(0)
    d, order = 64, 2
    T = 1028
    m = DeltaProductMemory(
        n_embd=d, n_heads=1, order=order,
        finetune_mode=False, block_size=64,
    )
    s = m.init_state(B=1, device="cpu")
    x = torch.randn(1, T, d)
    y, (M_out, _) = m.forward_chunk(x, s)
    assert torch.isfinite(y).all()
    assert torch.isfinite(M_out).all()
    assert y.abs().max() < 1e3


# -- Multi-head fusion ----------------------------------------------------


def test_multihead_output_shape():
    d, nh = 12, 3
    m = DeltaProductMemory(n_embd=d, n_heads=nh, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, d)
    y, (M_out, _) = m.forward_chunk(x, s)
    assert y.shape == (2, 5, d)
    assert M_out.shape == (2, nh, d // nh, d // nh)


def test_multihead_step_matches_forward_chunk_per_position():
    torch.manual_seed(0)
    d, nh = 12, 3
    m = DeltaProductMemory(
        n_embd=d, n_heads=nh, order=2, finetune_mode=False,
    )
    x = torch.randn(2, 4, d)
    s = m.init_state(B=2, device="cpu")
    y_chunk, _ = m.forward_chunk(x, s)
    s2 = m.init_state(B=2, device="cpu")
    y_steps = []
    for t in range(4):
        y_t, s2 = m.step_with_conv(x[:, t, :], s2)
        y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)
    assert torch.allclose(y_chunk, y_step, atol=1e-5)


def test_multihead_finetune_mode_zero_output_at_init():
    torch.manual_seed(0)
    m = DeltaProductMemory(
        n_embd=12, n_heads=3, order=2, finetune_mode=True,
    )
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, 12)
    y, _ = m.forward_chunk(x, s)
    assert torch.allclose(y, torch.zeros_like(y))


def test_multihead_blockwise_matches_sequential():
    """Multi-head chunkwise must equal multi-head sequential per token,
    head, and batch element."""
    m_seq, m_blk = _make_pair(
        d=12, n_heads=3, order=2, block_size=8, seed=7,
    )
    x = torch.randn(2, 6, 12)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq, _) = m_seq.forward_chunk(x, s0)
    y_blk, (M_blk, _) = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_multihead_doc_boundaries_apply_to_all_heads():
    """A doc-boundary reset applies to every head's M for the affected
    batch row. Verified by checking that after a row's boundary fires,
    its M dynamics from then on are dominated by post-boundary content
    (the small pool leak across the boundary is tolerated, same
    caveat as `test_doc_boundaries_reset_M_to_zero`)."""
    torch.manual_seed(0)
    d, nh = 8, 2
    m = DeltaProductMemory(
        n_embd=d, n_heads=nh, order=1, finetune_mode=False,
    )
    # Same trick: identical post-boundary content, different prefix.
    x_post = torch.randn(1, 3, d)
    x1 = torch.cat([torch.randn(1, 2, d), x_post], dim=1)
    x2 = torch.cat([torch.randn(1, 2, d) * 5, x_post], dim=1)
    db = torch.tensor([[False, False, True, False, False]])
    s1 = m.init_state(B=1, device="cpu")
    _, (M1, _) = m.forward_chunk(x1, s1, doc_boundaries=db)
    s2 = m.init_state(B=1, device="cpu")
    _, (M2, _) = m.forward_chunk(x2, s2, doc_boundaries=db)
    assert torch.allclose(M1, M2, atol=0.5)


# -- Chunkwise auxiliary-tensor cache -------------------------------------


def test_chunkwise_aux_cache_returns_identical_tensors_on_repeat_call():
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    a = _chunkwise_aux_tensors(device, T=8, N=2, dtype=torch.float32)
    b = _chunkwise_aux_tensors(device, T=8, N=2, dtype=torch.float32)
    for t_a, t_b in zip(a, b):
        assert t_a is t_b


def test_chunkwise_aux_cache_separates_by_T_and_N():
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    dtype = torch.float32
    mask_a, _, real_a = _chunkwise_aux_tensors(device, T=4, N=2, dtype=dtype)
    mask_b, _, real_b = _chunkwise_aux_tensors(device, T=4, N=3, dtype=dtype)
    mask_c, _, real_c = _chunkwise_aux_tensors(device, T=8, N=2, dtype=dtype)
    assert mask_a.shape == (8, 8) and real_a.shape == (4, 8)
    assert mask_b.shape == (12, 12) and real_b.shape == (4, 12)
    assert mask_c.shape == (16, 16) and real_c.shape == (8, 16)
    assert mask_a is not mask_b
    assert mask_a is not mask_c
    assert real_a is not real_b


def test_chunkwise_aux_real_mask_semantics():
    """real_mask[t, s] = 1 iff s < (t+1)·N."""
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    _, _, real_mask = _chunkwise_aux_tensors(
        device, T=3, N=2, dtype=torch.float32,
    )
    expected = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.float32)
    assert torch.allclose(real_mask, expected)


def test_chunkwise_solve_uses_cache_without_breaking_correctness():
    _chunkwise_aux_tensors.cache_clear()
    _, m_blk = _make_pair(d=4, order=2, block_size=8, seed=10)
    s = m_blk.init_state(B=2, device="cpu")
    x = torch.randn(2, 6, 4)

    y1, s1 = m_blk.forward_chunk(x, s)
    info1 = _chunkwise_aux_tensors.cache_info()
    assert info1.currsize >= 1

    s2_in = m_blk.init_state(B=2, device="cpu")
    y2, s2 = m_blk.forward_chunk(x, s2_in)
    info2 = _chunkwise_aux_tensors.cache_info()
    assert info2.hits > info1.hits

    assert torch.allclose(y1, y2)
    assert torch.allclose(s1[0], s2[0])
