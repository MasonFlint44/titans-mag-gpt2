"""DeltaProductMemory unit tests.

Covers the sequential reference path, the chunkwise WY parallel path,
multi-head fusion, and legacy state-dict migration.
"""

import pytest
import torch

from model.delta_product import DeltaProductMemory, _chunkwise_aux_tensors


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
    """Order-N adds (N-1) extra (K, V, β) projection sets per head."""
    n_embd = 16
    p1 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=1).parameters())
    p2 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=2).parameters())
    p3 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=3).parameters())
    # n_heads=1, head_dim=16. Per extra order: K + V + β weight + β bias:
    #   K [1, 16, 16] = 256, V same, β weight [1, 16, 1] = 16, β bias [1, 1] = 1
    extra_per_order = 2 * n_embd * n_embd + n_embd + 1
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
    of pretrained backbone preserved). This is the key invariant for
    safe injection into a pretrained model."""
    torch.manual_seed(0)
    m = DeltaProductMemory(n_embd=8, order=2, finetune_mode=True)
    state = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, 8)
    y, _ = m.forward_chunk(x, state, doc_boundaries=None)
    assert torch.allclose(y, torch.zeros_like(y))


# -- init_state -----------------------------------------------------------


def test_init_state_shape_and_zero_singlehead():
    m = DeltaProductMemory(n_embd=8, order=2)
    state = m.init_state(B=3, device="cpu")
    assert isinstance(state, tuple) and len(state) == 1
    (M,) = state
    # Single-head case: M shape [B, 1, n_embd, n_embd].
    assert M.shape == (3, 1, 8, 8)
    assert torch.all(M == 0.0)


def test_init_state_shape_multihead():
    """Multi-head state is [B, H, head_dim, head_dim]."""
    m = DeltaProductMemory(n_embd=12, n_heads=3, order=2)
    state = m.init_state(B=2, device="cpu")
    (M,) = state
    assert M.shape == (2, 3, 4, 4)  # head_dim = 12 / 3
    assert torch.all(M == 0.0)


# -- Math: order=1 delta rule explicit formula ----------------------------


def test_order1_one_token_matches_delta_rule_formula():
    """For T=1, order=1, M_in=0:
        M_1 = β · v · kᵀ      (since M_0·k = 0)
        y_1 = M_1 · q = β · (kᵀq) · v
    Verify by setting projections to identity (or known) and out_scale=1.
    """
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(
        n_embd=d, n_heads=1, order=1, finetune_mode=False,
    )
    # out_scale per-head [1, d], all ones in scratch mode.
    assert torch.all(m.out_scale == 1.0)

    # Set Q, K, V to identity per-head so the math is direct on input.
    with torch.no_grad():
        m.q_proj_weight[0].copy_(torch.eye(d))
        m.k_proj_weights[0][0].copy_(torch.eye(d))
        m.v_proj_weights[0][0].copy_(torch.eye(d))
        # β weight zeros + bias zero -> σ(0) = 0.5
        m.beta_proj_weights[0].zero_()
        m.beta_proj_biases[0].zero_()

    state = m.init_state(B=1, device="cpu")
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
    y, (M_out,) = m.forward_chunk(x, state)

    # k = v = q = x[0,0] = [1,0,0,0]; β = 0.5
    # M_1 = 0.5 * v * kᵀ -> rank-1 with M_1[0,0]=0.5
    # y = M_1 · q = [0.5, 0, 0, 0]
    expected_y = torch.tensor([[[0.5, 0.0, 0.0, 0.0]]])
    expected_M = torch.zeros(1, 1, d, d)
    expected_M[0, 0, 0, 0] = 0.5

    assert torch.allclose(y, expected_y, atol=1e-6)
    assert torch.allclose(M_out, expected_M, atol=1e-6)


def test_order2_differs_from_order1_with_distinct_beta2():
    """Order=2 with a non-trivial β_2 produces a different M than order=1."""
    torch.manual_seed(0)
    d = 4
    m1 = DeltaProductMemory(n_embd=d, order=1, finetune_mode=False)
    m2 = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)

    # Copy m1's projections into m2's slot-0 so the first sub-step is identical.
    with torch.no_grad():
        m2.q_proj_weight.copy_(m1.q_proj_weight)
        m2.k_proj_weights[0].copy_(m1.k_proj_weights[0])
        m2.v_proj_weights[0].copy_(m1.v_proj_weights[0])
        m2.beta_proj_weights[0].copy_(m1.beta_proj_weights[0])
        m2.beta_proj_biases[0].copy_(m1.beta_proj_biases[0])
        # Slot-1 β bias = +3 so σ(+3) ≈ 0.95 -> very different M.
        m2.beta_proj_biases[1].fill_(3.0)

    state1 = m1.init_state(B=1, device="cpu")
    state2 = m2.init_state(B=1, device="cpu")
    x = torch.randn(1, 4, d)

    y1, (M1,) = m1.forward_chunk(x, state1)
    y2, (M2,) = m2.forward_chunk(x, state2)

    assert not torch.allclose(y1, y2, atol=1e-3)
    assert not torch.allclose(M1, M2, atol=1e-3)


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
    y_step_list = []
    for t in range(5):
        y_t, s2 = m.step_with_conv(x[:, t, :], s2)
        y_step_list.append(y_t)
    y_step = torch.stack(y_step_list, dim=1)

    assert torch.allclose(y_chunk, y_step, atol=1e-5)
    assert torch.allclose(s_chunk[0], s2[0], atol=1e-5)


# -- Doc boundaries -------------------------------------------------------


def test_doc_boundaries_reset_M_to_zero():
    """At a doc-boundary position, M must reset to zero BEFORE that
    token's update. After the boundary token, M should reflect only the
    boundary token's contribution, not any prior history."""
    torch.manual_seed(0)
    d = 4
    m = DeltaProductMemory(n_embd=d, order=1, finetune_mode=False)

    x = torch.randn(1, 5, d)
    doc_boundaries = torch.tensor([[False, False, True, False, False]])

    s = m.init_state(B=1, device="cpu")
    _, (M_with_boundary,) = m.forward_chunk(x, s, doc_boundaries=doc_boundaries)

    s2 = m.init_state(B=1, device="cpu")
    _, (M_post,) = m.forward_chunk(x[:, 2:, :], s2)

    assert torch.allclose(M_with_boundary, M_post, atol=1e-6)


# -- Gradient flow --------------------------------------------------------


def test_gradient_flows_to_all_projections():
    """Backward through forward_chunk must reach every projection
    parameter and produce nonzero gradients."""
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


# -- Shape / dtype --------------------------------------------------------


def test_output_shape_matches_input():
    d = 8
    m = DeltaProductMemory(n_embd=d, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 7, d)
    y, (M_out,) = m.forward_chunk(x, s)
    assert y.shape == (2, 7, d)
    # Single-head -> M_out shape [B, 1, d, d].
    assert M_out.shape == (2, 1, d, d)


def test_keys_are_l2_normalized():
    """K must be L2-normalized along the last dim — required for the
    chunkwise WY solve's numerical stability and matches the standard
    DeltaNet / DeltaProduct formulation in Yang et al. and TPTT."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, order=2, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, ks, _, _ = m._project_kvb(x)
    for k in ks:
        # ks[i] shape: [B, T, H, head_dim]; norm over head_dim.
        norms = k.norm(dim=-1)  # [B, T, H]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"K not L2-normalized: norms={norms}"
        )


# -- Blockwise (chunkwise WY) parallel path -------------------------------


def _make_pair(
    d=6, n_heads=1, order=2, finetune_mode=False, block_size=8, seed=0,
):
    """Build two DeltaProductMemory instances with identical weights —
    one in sequential mode, one in blockwise mode. Used to verify
    blockwise == sequential at the math level."""
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
    """Order=1 chunkwise WY form is bit-equivalent to per-token sequential
    on the same (q, k, v, β) projections (DeltaNet chunkwise correctness)."""
    m_seq, m_blk = _make_pair(d=4, order=1, block_size=8, seed=1)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_no_boundaries_order2():
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=2)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_order3():
    m_seq, m_blk = _make_pair(d=4, order=3, block_size=8, seed=3)
    x = torch.randn(2, 5, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_with_nonzero_initial_state():
    """The chunkwise formula must handle non-zero initial M correctly."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=4)
    x = torch.randn(2, 5, 4)
    s0_seq = m_seq.init_state(B=2, device="cpu")
    s0_blk = m_blk.init_state(B=2, device="cpu")
    _, s_warm_seq = m_seq.forward_chunk(torch.randn(2, 3, 4), s0_seq)
    _, s_warm_blk = m_blk.forward_chunk(torch.randn(2, 3, 4), s0_blk)
    # Use the same warmed state for both (identical weights → identical M).
    (M_warm,) = s_warm_seq
    state_in = (M_warm,)
    y_seq, (M_seq,) = m_seq.forward_chunk(x, state_in)
    y_blk, (M_blk,) = m_blk.forward_chunk(x, state_in)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_boundary_at_position_zero_resets_M():
    """Regression for the chunkwise-vs-sequential divergence at
    doc_boundaries[:, 0]=True. The data loader sets this whenever
    EOT aligns with a chunk boundary; earlier chunkwise code skipped
    the reset for the t_lo=0 segment, leaking the prior chunk's M
    state into the new document. The sequential path resets
    unconditionally — chunkwise must match."""
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
    """Boundary-aware chunkwise must match sequential exactly when
    boundaries are inside the chunk."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=5)
    x = torch.randn(2, 6, 4)
    db = torch.tensor([
        [False, False, True, False, False, False],
        [False, True,  False, False, True,  False],
    ])
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0, doc_boundaries=db)
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0, doc_boundaries=db)
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
    """Regression for the original NaN-loss bug: at training chunk
    length (T·N >> 1), the WY triangular solve diverges unless keys
    are L2-normalized to bound K·Kᵀ. Runs at training chunk shape
    (chunk_size=1024 + persistent_prefix=4 = 1028, head_dim=64,
    order=2 → T·N = 2056) and asserts finite outputs."""
    torch.manual_seed(0)
    d, order = 64, 2
    T = 1028
    m = DeltaProductMemory(
        n_embd=d, n_heads=1, order=order,
        finetune_mode=False, block_size=64,
    )
    s = m.init_state(B=1, device="cpu")
    x = torch.randn(1, T, d)
    y, (M_out,) = m.forward_chunk(x, s)
    assert torch.isfinite(y).all()
    assert torch.isfinite(M_out).all()
    assert y.abs().max() < 1e3, f"y magnitude {y.abs().max()} too large"


# -- Multi-head fusion ----------------------------------------------------


def test_multihead_output_shape():
    """Multi-head forward returns [B, T, n_embd] regardless of n_heads."""
    d, nh = 12, 3
    m = DeltaProductMemory(n_embd=d, n_heads=nh, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, d)
    y, (M_out,) = m.forward_chunk(x, s)
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
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_multihead_doc_boundaries_share_across_heads():
    """doc_boundaries is a single per-batch-position tensor — all heads
    must respect the same boundary positions."""
    torch.manual_seed(0)
    d, nh = 8, 2
    m = DeltaProductMemory(
        n_embd=d, n_heads=nh, order=1, finetune_mode=False,
    )
    x = torch.randn(1, 5, d)
    db = torch.tensor([[False, False, True, False, False]])
    s = m.init_state(B=1, device="cpu")
    _, (M_full,) = m.forward_chunk(x, s, doc_boundaries=db)
    s2 = m.init_state(B=1, device="cpu")
    _, (M_post,) = m.forward_chunk(x[:, 2:, :], s2)
    assert torch.allclose(M_full, M_post, atol=1e-6)


def test_multihead_heads_are_independent():
    """Fused multi-head must produce per-head outputs equal to running
    n_heads independent single-head DeltaProductMemory modules on each
    head's slice of x (with the corresponding stacked-weight slice
    copied to a single-head module). This pins the per-head architecture
    invariant: heads share no parameters, see only their own input slice."""
    torch.manual_seed(0)
    H, hd, order = 3, 4, 2
    d = H * hd
    m_multi = DeltaProductMemory(
        n_embd=d, n_heads=H, order=order, finetune_mode=False,
    )

    # Build n_heads independent single-head modules sharing the multi-
    # head model's per-head slice.
    x = torch.randn(2, 4, d)
    s = m_multi.init_state(B=2, device="cpu")
    y_multi, _ = m_multi.forward_chunk(x, s)

    head_outs = []
    for h in range(H):
        m_single = DeltaProductMemory(
            n_embd=hd, n_heads=1, order=order, finetune_mode=False,
        )
        with torch.no_grad():
            m_single.q_proj_weight[0].copy_(m_multi.q_proj_weight[h])
            m_single.out_scale[0].copy_(m_multi.out_scale[h])
            for i in range(order):
                m_single.k_proj_weights[i][0].copy_(m_multi.k_proj_weights[i][h])
                m_single.v_proj_weights[i][0].copy_(m_multi.v_proj_weights[i][h])
                m_single.beta_proj_weights[i][0].copy_(m_multi.beta_proj_weights[i][h])
                m_single.beta_proj_biases[i][0].copy_(m_multi.beta_proj_biases[i][h])
        x_h = x[..., h * hd : (h + 1) * hd]
        s_h = m_single.init_state(B=2, device="cpu")
        y_h, _ = m_single.forward_chunk(x_h, s_h)
        head_outs.append(y_h)

    y_ref = torch.cat(head_outs, dim=-1)
    assert torch.allclose(y_multi, y_ref, atol=1e-5)


# -- Legacy state-dict migration ------------------------------------------


def test_load_legacy_multihead_per_head_submodule_format():
    """Pre-fusion MultiHeadDeltaProduct stored n_heads independent
    DeltaProductMemory submodules under `heads.{h}.`. The new
    multi-head-native DeltaProductMemory must load those checkpoints
    transparently via the registered pre-hook."""
    torch.manual_seed(0)
    H, hd, order = 3, 4, 2
    n_embd = H * hd

    m = DeltaProductMemory(
        n_embd=n_embd, n_heads=H, order=order, finetune_mode=False,
    )

    # Fabricate an OLD-format state dict by hand.
    legacy = {}
    q_per_head = [torch.randn(hd, hd) for _ in range(H)]
    out_scale_per_head = [torch.randn(hd) for _ in range(H)]
    k_per_head = [[torch.randn(hd, hd) for _ in range(H)] for _ in range(order)]
    v_per_head = [[torch.randn(hd, hd) for _ in range(H)] for _ in range(order)]
    bw_per_head = [[torch.randn(1, hd) for _ in range(H)] for _ in range(order)]
    bb_per_head = [[torch.randn(1) for _ in range(H)] for _ in range(order)]

    for h in range(H):
        legacy[f"heads.{h}.q_proj.weight"] = q_per_head[h]
        legacy[f"heads.{h}.out_scale"] = out_scale_per_head[h]
        for i in range(order):
            legacy[f"heads.{h}.k_projs.{i}.weight"] = k_per_head[i][h]
            legacy[f"heads.{h}.v_projs.{i}.weight"] = v_per_head[i][h]
            legacy[f"heads.{h}.beta_heads.{i}.weight"] = bw_per_head[i][h]
            legacy[f"heads.{h}.beta_heads.{i}.bias"] = bb_per_head[i][h]

    # Load — pre-hook should rewrite the keys.
    missing, unexpected = m.load_state_dict(legacy, strict=False)
    assert missing == [], f"unexpected missing keys: {missing}"
    assert unexpected == [], f"unexpected extra keys: {unexpected}"

    # Verify stacked params received per-head weights.
    for h in range(H):
        assert torch.allclose(m.q_proj_weight[h], q_per_head[h])
        assert torch.allclose(m.out_scale[h], out_scale_per_head[h])
        for i in range(order):
            assert torch.allclose(m.k_proj_weights[i][h], k_per_head[i][h])
            assert torch.allclose(m.v_proj_weights[i][h], v_per_head[i][h])
            # β weight: old [1, hd] -> new [hd, 1] via transpose.
            assert torch.allclose(
                m.beta_proj_weights[i][h], bw_per_head[i][h].t(),
            )
            assert torch.allclose(m.beta_proj_biases[i][h], bb_per_head[i][h])


# -- Chunkwise auxiliary-tensor cache -------------------------------------


def test_chunkwise_aux_cache_returns_identical_tensors_on_repeat_call():
    """`_chunkwise_aux_tensors` is memoized by (device, T, N, dtype). A
    second call with the same key should return the SAME tensor objects
    (identity, not just equality) so we're truly avoiding re-allocation."""
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    a = _chunkwise_aux_tensors(device, T=8, N=2, dtype=torch.float32)
    b = _chunkwise_aux_tensors(device, T=8, N=2, dtype=torch.float32)
    for t_a, t_b in zip(a, b):
        assert t_a is t_b


def test_chunkwise_aux_cache_separates_by_T_and_N():
    """Different (T, N) must produce distinct tensors with the right shapes."""
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    dtype = torch.float32

    mask_a, I_a, real_a = _chunkwise_aux_tensors(device, T=4, N=2, dtype=dtype)
    mask_b, I_b, real_b = _chunkwise_aux_tensors(device, T=4, N=3, dtype=dtype)
    mask_c, I_c, real_c = _chunkwise_aux_tensors(device, T=8, N=2, dtype=dtype)

    # T=4, N=2 -> TN=8
    assert mask_a.shape == (8, 8)
    assert I_a.shape == (8, 8)
    assert real_a.shape == (4, 8)
    # T=4, N=3 -> TN=12
    assert mask_b.shape == (12, 12)
    assert real_b.shape == (4, 12)
    # T=8, N=2 -> TN=16
    assert mask_c.shape == (16, 16)
    assert real_c.shape == (8, 16)

    # And not the same objects.
    assert mask_a is not mask_b
    assert mask_a is not mask_c
    assert real_a is not real_b


def test_chunkwise_aux_real_mask_semantics():
    """`real_mask[t, s] = 1` iff virtual write `s` is in real token `t`'s
    write-window — i.e. s < (t+1)·N. This is what scopes each read to
    its own writes."""
    _chunkwise_aux_tensors.cache_clear()
    device = torch.device("cpu")
    _, _, real_mask = _chunkwise_aux_tensors(
        device, T=3, N=2, dtype=torch.float32,
    )
    # T=3, N=2 -> TN=6. For each real t, virtual positions allowed are 0..(t+1)*N-1.
    expected = torch.tensor([
        [1, 1, 0, 0, 0, 0],  # t=0: s < 2
        [1, 1, 1, 1, 0, 0],  # t=1: s < 4
        [1, 1, 1, 1, 1, 1],  # t=2: s < 6
    ], dtype=torch.float32)
    assert torch.allclose(real_mask, expected)


def test_chunkwise_solve_uses_cache_without_breaking_correctness():
    """End-to-end: warm the cache with one forward, then a second forward
    of the same shape must hit the cache and produce identical results."""
    _chunkwise_aux_tensors.cache_clear()
    _, m_blk = _make_pair(d=4, order=2, block_size=8, seed=10)
    s = m_blk.init_state(B=2, device="cpu")
    x = torch.randn(2, 6, 4)

    # First forward populates the cache.
    y1, s1 = m_blk.forward_chunk(x, s)
    cache_info_after_first = _chunkwise_aux_tensors.cache_info()
    assert cache_info_after_first.currsize >= 1

    # Second forward (same shape) hits the cache.
    s2_in = m_blk.init_state(B=2, device="cpu")
    y2, s2 = m_blk.forward_chunk(x, s2_in)
    cache_info_after_second = _chunkwise_aux_tensors.cache_info()
    assert cache_info_after_second.hits > cache_info_after_first.hits

    # Identical outputs (same weights, same inputs, same cache).
    assert torch.allclose(y1, y2)
    assert torch.allclose(s1[0], s2[0])


def test_load_legacy_singlehead_linear_module_format():
    """Pre-fusion single-head DeltaProductMemory used Linear modules
    (q_proj.weight, k_projs.{i}.weight, etc.). The new layout must
    accept those checkpoints too — needed for any single-head training
    run that predates the multi-head fusion refactor."""
    torch.manual_seed(0)
    n_embd, order = 8, 2
    m = DeltaProductMemory(
        n_embd=n_embd, n_heads=1, order=order, finetune_mode=False,
    )

    legacy = {
        "q_proj.weight": torch.randn(n_embd, n_embd),
        "out_scale": torch.randn(n_embd),
    }
    for i in range(order):
        legacy[f"k_projs.{i}.weight"] = torch.randn(n_embd, n_embd)
        legacy[f"v_projs.{i}.weight"] = torch.randn(n_embd, n_embd)
        legacy[f"beta_heads.{i}.weight"] = torch.randn(1, n_embd)
        legacy[f"beta_heads.{i}.bias"] = torch.randn(1)

    missing, unexpected = m.load_state_dict(legacy, strict=False)
    assert missing == []
    assert unexpected == []

    # Spot-check: stacked params should match wrapped legacy values.
    assert torch.allclose(m.q_proj_weight[0], legacy["q_proj.weight"])
    assert torch.allclose(m.out_scale[0], legacy["out_scale"])
