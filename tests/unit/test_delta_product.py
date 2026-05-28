"""DeltaProductMemory unit tests.

Covers the sequential reference path (block_size=1). Blockwise-vs-
sequential equivalence and MultiHeadDeltaProduct tests live alongside
once those paths land.
"""

import pytest
import torch

from model.delta_product import DeltaProductMemory, MultiHeadDeltaProduct


# -- Module construction --------------------------------------------------


def test_construct_rejects_bad_order():
    with pytest.raises(ValueError, match="order must be >= 1"):
        DeltaProductMemory(n_embd=8, order=0)


def test_construct_rejects_bad_block_size():
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DeltaProductMemory(n_embd=8, order=2, block_size=0)


def test_param_count_scales_with_order():
    """Order-N adds (N-1) extra (K, V, β) projection sets."""
    n_embd = 16
    p1 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=1).parameters())
    p2 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=2).parameters())
    p3 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=3).parameters())
    # Each extra order adds: 1 K (d²) + 1 V (d²) + 1 β head (d+1)
    extra_per_order = 2 * n_embd * n_embd + (n_embd + 1)
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


def test_init_state_shape_and_zero():
    m = DeltaProductMemory(n_embd=8, order=2)
    state = m.init_state(B=3, device="cpu")
    assert isinstance(state, tuple) and len(state) == 1
    (M,) = state
    assert M.shape == (3, 8, 8)
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
    m = DeltaProductMemory(n_embd=d, order=1, finetune_mode=False)
    # Make sure we don't zero out the output.
    assert torch.all(m.out_scale == 1.0)

    # Set Q, K, V to identity so the math is direct on the input.
    with torch.no_grad():
        m.q_proj.weight.copy_(torch.eye(d))
        m.k_projs[0].weight.copy_(torch.eye(d))
        m.v_projs[0].weight.copy_(torch.eye(d))
        # Set β bias so sigmoid(0 + b) is a known value; pick b such that
        # σ(b) = 0.5 (i.e., b = 0). Weight already zeros input contribution.
        m.beta_heads[0].weight.zero_()
        m.beta_heads[0].bias.zero_()

    state = m.init_state(B=1, device="cpu")
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1, 1, 4]
    y, (M_out,) = m.forward_chunk(x, state)

    # k = v = q = x[0,0] = [1,0,0,0]; β = 0.5
    # M_1 = 0.5 * v * kᵀ -> rank-1 with M_1[0,0]=0.5
    # y = M_1 · q = M_1 · [1,0,0,0]ᵀ = [0.5, 0, 0, 0]
    expected_y = torch.tensor([[[0.5, 0.0, 0.0, 0.0]]])
    expected_M = torch.zeros(1, d, d)
    expected_M[0, 0, 0] = 0.5

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
        m2.q_proj.weight.copy_(m1.q_proj.weight)
        m2.k_projs[0].weight.copy_(m1.k_projs[0].weight)
        m2.v_projs[0].weight.copy_(m1.v_projs[0].weight)
        m2.beta_heads[0].weight.copy_(m1.beta_heads[0].weight)
        m2.beta_heads[0].bias.copy_(m1.beta_heads[0].bias)
        # Slot-1 projections random; β_2 head bias = +3 so σ(+3) ≈ 0.95 -> very
        # different M after the second sub-step.
        m2.beta_heads[1].bias.fill_(3.0)

    state1 = m1.init_state(B=1, device="cpu")
    state2 = m2.init_state(B=1, device="cpu")
    x = torch.randn(1, 4, d)

    y1, (M1,) = m1.forward_chunk(x, state1)
    y2, (M2,) = m2.forward_chunk(x, state2)

    # Order-2 must diverge from order-1.
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

    # Forward with the boundary.
    s = m.init_state(B=1, device="cpu")
    _, (M_with_boundary,) = m.forward_chunk(x, s, doc_boundaries=doc_boundaries)

    # Forward only the post-boundary tokens (starting from zero state)
    # — should yield the same M_out as the boundary-respecting run.
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
    assert M_out.shape == (2, d, d)


# -- block_size > 1 raises until blockwise lands --------------------------


# -- Blockwise (chunkwise WY) parallel path -------------------------------


def _make_pair(d=6, order=2, finetune_mode=False, block_size=8, seed=0):
    """Build two DeltaProductMemory instances with identical weights —
    one in sequential mode, one in blockwise mode. Used to verify
    blockwise == sequential at the math level."""
    torch.manual_seed(seed)
    m_seq = DeltaProductMemory(
        n_embd=d, order=order, finetune_mode=finetune_mode, block_size=1,
    )
    torch.manual_seed(seed)
    m_blk = DeltaProductMemory(
        n_embd=d, order=order, finetune_mode=finetune_mode,
        block_size=block_size,
    )
    return m_seq, m_blk


def test_blockwise_matches_sequential_no_boundaries_order1():
    """Order=1 chunkwise WY form is bit-equivalent to per-token sequential
    on the same (q, k, v, β) projections. This is the DeltaNet chunkwise
    correctness test."""
    m_seq, m_blk = _make_pair(d=4, order=1, block_size=8, seed=1)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_no_boundaries_order2():
    """Order=2 (DeltaProduct) chunkwise must match sequential."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=2)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, (M_seq,) = m_seq.forward_chunk(x, s0)
    s0b = m_blk.init_state(B=2, device="cpu")
    y_blk, (M_blk,) = m_blk.forward_chunk(x, s0b)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


def test_blockwise_matches_sequential_order3():
    """Higher orders compose correctly."""
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
    # Warm both modules to a non-zero state.
    s0_seq = m_seq.init_state(B=2, device="cpu")
    s0_blk = m_blk.init_state(B=2, device="cpu")
    _, s_warm_seq = m_seq.forward_chunk(torch.randn(2, 3, 4), s0_seq)
    _, s_warm_blk = m_blk.forward_chunk(torch.randn(2, 3, 4), s0_blk)
    # Now feed the same M_in (use the seq-warmed M for both).
    (M_warm,) = s_warm_seq
    state_in = (M_warm,)
    y_seq, (M_seq,) = m_seq.forward_chunk(x, state_in)
    y_blk, (M_blk,) = m_blk.forward_chunk(x, state_in)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(M_seq, M_blk, atol=1e-5)


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
    """Blockwise forward over two halves of a chunk threaded by state
    must equal blockwise forward over the full chunk."""
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
    """Backward through the chunkwise solver must reach every parameter."""
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


# -- MultiHeadDeltaProduct ------------------------------------------------


def test_multihead_construct_rejects_bad_n_heads():
    with pytest.raises(ValueError, match="n_heads must be >= 1"):
        MultiHeadDeltaProduct(n_embd=8, n_heads=0)


def test_multihead_construct_rejects_indivisible_dim():
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadDeltaProduct(n_embd=10, n_heads=3)


def test_multihead_init_state_returns_per_head_list():
    m = MultiHeadDeltaProduct(n_embd=12, n_heads=3, order=2)
    state = m.init_state(B=2, device="cpu")
    assert isinstance(state, list)
    assert len(state) == 3
    for s in state:
        assert isinstance(s, tuple) and len(s) == 1
        (M_head,) = s
        assert M_head.shape == (2, 4, 4)  # head_dim = 12 / 3


def test_multihead_output_shape():
    d, nh = 12, 3
    m = MultiHeadDeltaProduct(n_embd=d, n_heads=nh, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, d)
    y, ns = m.forward_chunk(x, s)
    assert y.shape == (2, 5, d)
    assert len(ns) == nh


def test_multihead_equals_concat_of_singlehead_calls():
    """MultiHeadDeltaProduct forward must equal concat of per-head
    DeltaProductMemory forwards run on the corresponding head_dim
    slice. Identity test guaranteeing the wrapper is structurally
    correct."""
    torch.manual_seed(0)
    d, nh = 8, 2
    head_dim = d // nh
    mh = MultiHeadDeltaProduct(
        n_embd=d, n_heads=nh, order=2, finetune_mode=False,
    )
    x = torch.randn(2, 4, d)

    # Reference: run each head's underlying DeltaProductMemory directly on
    # the corresponding head_dim slice of x.
    head_outs = []
    head_states_out = []
    for i, head in enumerate(mh.heads):
        x_h = x[..., i * head_dim : (i + 1) * head_dim].contiguous()
        s_h = head.init_state(B=2, device="cpu")
        y_h, ns_h = head.forward_chunk(x_h, s_h)
        head_outs.append(y_h)
        head_states_out.append(ns_h)
    y_ref = torch.cat(head_outs, dim=-1)

    s = mh.init_state(B=2, device="cpu")
    y_mh, ns_mh = mh.forward_chunk(x, s)

    assert torch.allclose(y_mh, y_ref, atol=1e-6)
    for i in range(nh):
        assert torch.allclose(ns_mh[i][0], head_states_out[i][0], atol=1e-6)


def test_multihead_step_matches_forward_chunk_per_position():
    torch.manual_seed(0)
    d, nh = 12, 3
    m = MultiHeadDeltaProduct(
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
    m = MultiHeadDeltaProduct(
        n_embd=12, n_heads=3, order=2, finetune_mode=True,
    )
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, 12)
    y, _ = m.forward_chunk(x, s)
    assert torch.allclose(y, torch.zeros_like(y))


def test_multihead_doc_boundaries_share_across_heads():
    """doc_boundaries is a single per-batch-position tensor — all heads
    must respect the same boundary positions."""
    torch.manual_seed(0)
    d, nh = 8, 2
    m = MultiHeadDeltaProduct(
        n_embd=d, n_heads=nh, order=1, finetune_mode=False,
    )
    x = torch.randn(1, 5, d)
    db = torch.tensor([[False, False, True, False, False]])

    s = m.init_state(B=1, device="cpu")
    _, ns_full = m.forward_chunk(x, s, doc_boundaries=db)

    # Compare to running only the post-boundary tokens from zero state.
    s2 = m.init_state(B=1, device="cpu")
    _, ns_post = m.forward_chunk(x[:, 2:, :], s2)

    for i in range(nh):
        assert torch.allclose(ns_full[i][0], ns_post[i][0], atol=1e-6)
