"""DeltaProductMemory unit tests — TPTT formulation with VirtualTokenExpander.

Covers projections (SiLU+L2 on Q/K, V scaling, vector β from CausalAvgPool),
VirtualTokenExpander (binomial "dt" derivative trick), sequential path,
chunkwise WY parallel path, multi-head fusion, cross-chunk continuity for
both pool and expander buffers, and the auxiliary-tensor cache.
"""

import math

import pytest
import torch

from model.delta_product import (
    DeltaProductMemory,
    _causal_avg_pool_3,
    _chunkwise_aux_tensors,
    _virtual_token_expand,
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


def test_param_count_does_not_scale_with_order():
    """SINGLE Q, K, V projections. Order changes ONLY the VirtualTokenExpander
    kernel (a non-parameter buffer), so the parameter count is invariant
    across orders."""
    n_embd = 16
    p1 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=1).parameters())
    p2 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=2).parameters())
    p3 = sum(p.numel() for p in DeltaProductMemory(n_embd, order=3).parameters())
    assert p1 == p2 == p3, (
        f"param counts should match across orders (got {p1}, {p2}, {p3})"
    )


def test_out_proj_has_bias():
    """TPTT default: `out_proj = Linear(..., bias=True)`."""
    m = DeltaProductMemory(n_embd=8, order=2)
    assert m.out_proj.bias is not None
    assert m.out_proj.bias.shape == (8,)


def test_finetune_mode_zero_inits_out_scale():
    m_ft = DeltaProductMemory(n_embd=8, order=2, finetune_mode=True)
    m_scratch = DeltaProductMemory(n_embd=8, order=2, finetune_mode=False)
    assert torch.all(m_ft.out_scale == 0.0)
    assert torch.all(m_scratch.out_scale == 1.0)


def test_finetune_mode_produces_zero_output_at_init():
    """out_scale=0 zeros y regardless of out_proj's weight + bias."""
    torch.manual_seed(0)
    m = DeltaProductMemory(n_embd=8, order=2, finetune_mode=True)
    state = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, 8)
    y, _ = m.forward_chunk(x, state, doc_boundaries=None)
    assert torch.allclose(y, torch.zeros_like(y))


# -- init_state -----------------------------------------------------------


def test_init_state_is_3tuple_with_1e6_M_and_buffers():
    """State is (M, k_raw_buf, qkvb_buf). M is 1e-6 (TPTT stability fill),
    k_raw_buf is zeros [B, 2, H, hd], qkvb_buf is zeros [B, order-1, H, hd]
    for each of (q, k, v, β) — None at order=1."""
    m = DeltaProductMemory(n_embd=8, order=2)
    state = m.init_state(B=3, device="cpu")
    assert isinstance(state, tuple) and len(state) == 3
    M, k_raw_buf, qkvb_buf = state
    assert M.shape == (3, 1, 8, 8)
    assert torch.allclose(M, torch.full_like(M, 1e-6))
    assert k_raw_buf.shape == (3, 2, 1, 8)
    assert torch.all(k_raw_buf == 0)
    assert isinstance(qkvb_buf, tuple) and len(qkvb_buf) == 4
    for b in qkvb_buf:
        assert b.shape == (3, 1, 1, 8)  # order-1 = 1
        assert torch.all(b == 0)


def test_init_state_order_1_has_no_qkvb_buf():
    """Order=1 is the identity for VirtualTokenExpander — no
    (n-1)-token buffer needed."""
    m = DeltaProductMemory(n_embd=8, order=1)
    M, k_raw_buf, qkvb_buf = m.init_state(B=2, device="cpu")
    assert qkvb_buf is None
    assert k_raw_buf.shape == (2, 2, 1, 8)


def test_init_state_multihead():
    m = DeltaProductMemory(n_embd=12, n_heads=3, order=2)
    M, k_raw_buf, qkvb_buf = m.init_state(B=2, device="cpu")
    assert M.shape == (2, 3, 4, 4)
    assert k_raw_buf.shape == (2, 2, 3, 4)
    for b in qkvb_buf:
        assert b.shape == (2, 1, 3, 4)


# -- CausalAvgPool helper -------------------------------------------------


def test_causal_avg_pool_3_replicate_at_start():
    x = torch.tensor([
        [[1.0], [2.0], [3.0], [4.0]],
    ])
    y = _causal_avg_pool_3(x)
    expected = torch.tensor([[[1.0], [4 / 3.0], [2.0], [3.0]]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_causal_avg_pool_3_at_T_one():
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    y = _causal_avg_pool_3(x)
    assert torch.allclose(y, x)


# -- VirtualTokenExpander -------------------------------------------------


def test_virtual_token_expander_order_2_kernel_values():
    """For n=2, normalized kernel is [0.5, -0.5]. After flip(-1) and
    permute, virtual[s, 0] = x[s] * -0.5 and virtual[s, 1] = x[s-1]
    * 0.5 (zero for s=0 since x[-1] is pad)."""
    n = 2
    coeffs = [(-1) ** k * math.comb(n - 1, k) for k in range(n)]
    kernel = torch.tensor(coeffs, dtype=torch.float32)
    kernel = kernel / kernel.abs().sum()

    x = torch.tensor([
        [[[1.0]], [[2.0]], [[3.0]]],  # [B=1, T=3, H=1, hd=1]
    ])
    out = _virtual_token_expand(x, n, kernel)  # [B, T, n, H, hd]

    # s=0: x_padded[0]=0, x_padded[1]=1. After flip:
    #   virtual[0, 0] = x_padded[1] * kernel[1] = 1 * -0.5 = -0.5
    #   virtual[0, 1] = x_padded[0] * kernel[0] = 0 * 0.5  =  0
    # s=1: virtual[1, 0] = 2 * -0.5 = -1.0, virtual[1, 1] = 1 * 0.5 = 0.5
    # s=2: virtual[2, 0] = 3 * -0.5 = -1.5, virtual[2, 1] = 2 * 0.5 = 1.0
    expected = torch.tensor([
        [
            [[[-0.5]], [[0.0]]],
            [[[-1.0]], [[0.5]]],
            [[[-1.5]], [[1.0]]],
        ],
    ])
    assert torch.allclose(out, expected, atol=1e-6)


def test_virtual_token_expander_order_1_is_identity():
    """For n=1, kernel is [1] (after normalization), and the expander
    just adds a length-1 sub-step dim — no temporal mixing."""
    n = 1
    kernel = torch.tensor([1.0], dtype=torch.float32)
    x = torch.randn(2, 4, 3, 5)  # [B, T, H, hd]
    out = _virtual_token_expand(x, n, kernel)
    assert out.shape == (2, 4, 1, 3, 5)
    assert torch.allclose(out.squeeze(2), x)


# -- Projection invariants (TPTT formulation) -----------------------------


def test_projections_q_and_k_are_silu_l2_normalized_before_expand():
    """After SiLU+L2 (before the expander), q and k are unit-norm. After
    expansion, magnitudes may differ (expander scales by kernel)."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, order=1, finetune_mode=False)
    x = torch.randn(2, 5, d)
    state = m.init_state(B=2, device="cpu")
    _, k_raw_buf, qkvb_buf = state
    q_virt, k_virt, v_virt, beta_virt, _, _ = m._project_kvb(
        x, k_raw_buf, qkvb_buf,
    )
    # At order=1, expander is identity, so q_virt[:, :, 0] == q_normed.
    q = q_virt.squeeze(2)
    k = k_virt.squeeze(2)
    q_norms = q.norm(dim=-1)
    k_norms = k.norm(dim=-1)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)
    assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)


def test_projection_v_scaled_by_inv_sqrt_head_dim():
    """V is scaled by 1/√head_dim — verify magnitude vs an unscaled
    reference projection (at order=1, expander is identity)."""
    torch.manual_seed(0)
    d = 16
    m = DeltaProductMemory(n_embd=d, n_heads=1, order=1, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, k_raw_buf, qkvb_buf = m.init_state(B=2, device="cpu")
    _, _, v_virt, _, _, _ = m._project_kvb(x, k_raw_buf, qkvb_buf)

    x_h = x.view(2, 5, 1, d)
    v_raw = torch.einsum("bthd,hde->bthe", x_h, m.v_proj_weight)
    v_expected = (v_raw * (1.0 / math.sqrt(d))).unsqueeze(2)
    assert torch.allclose(v_virt, v_expected, atol=1e-6)


def test_beta_is_sigmoid_of_pooled_raw_k():
    """β at order=1 (no expansion) matches σ(CausalAvgPool(k_raw_ext))[2:].
    """
    torch.manual_seed(0)
    d = 8
    m = DeltaProductMemory(n_embd=d, n_heads=1, order=1, finetune_mode=False)
    x = torch.randn(2, 5, d)
    _, k_raw_buf, qkvb_buf = m.init_state(B=2, device="cpu")
    _, _, _, beta_virt, _, _ = m._project_kvb(x, k_raw_buf, qkvb_buf)

    x_h = x.view(2, 5, 1, d)
    k_raw = torch.einsum("bthd,hde->bthe", x_h, m.k_proj_weight)
    k_raw_ext = torch.cat([k_raw_buf, k_raw], dim=1)
    beta_full = torch.sigmoid(_causal_avg_pool_3(k_raw_ext))
    beta_expected = beta_full[:, 2:].unsqueeze(2)
    assert torch.allclose(beta_virt, beta_expected, atol=1e-6)


# -- State threading + buffers --------------------------------------------


def test_state_continuity_split_chunk_matches_full():
    """Both the pool buffer (last 2 raw K) and the expander buffer
    (last n-1 post-projection q, k, v, β) thread across forward_chunk
    calls so a split-into-two yields the same outputs and state as a
    single-shot forward."""
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
    """T calls to step_with_conv must equal forward_chunk on the same
    T-token chunk, given identical buffer continuity."""
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


# -- Gradient flow --------------------------------------------------------


def test_gradient_flows_to_all_projections():
    """Backward through forward_chunk must reach every learnable
    parameter (q_proj_weight, k_proj_weight, v_proj_weight, out_proj
    weight + bias, out_scale)."""
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
    y, new_state = m.forward_chunk(x, s)
    assert y.shape == (2, 7, d)
    M_out = new_state[0]
    assert M_out.shape == (2, 1, d, d)


# -- Blockwise (chunkwise WY) parallel path -------------------------------


def _make_pair(
    d=6, n_heads=1, order=2, finetune_mode=False, block_size=8, seed=0,
):
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


def test_blockwise_matches_sequential_order1():
    """Order=1: VirtualTokenExpander is identity. WY chunkwise == sequential."""
    m_seq, m_blk = _make_pair(d=4, order=1, block_size=8, seed=1)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, s_seq = m_seq.forward_chunk(x, s0)
    y_blk, s_blk = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


def test_blockwise_matches_sequential_order2():
    """Order=2: WY chunkwise == sequential with full VirtualTokenExpander."""
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=2)
    x = torch.randn(2, 6, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, s_seq = m_seq.forward_chunk(x, s0)
    y_blk, s_blk = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


def test_blockwise_matches_sequential_order3():
    m_seq, m_blk = _make_pair(d=4, order=3, block_size=8, seed=3)
    x = torch.randn(2, 5, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, s_seq = m_seq.forward_chunk(x, s0)
    y_blk, s_blk = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


def test_blockwise_matches_sequential_with_nonzero_initial_state():
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=4)
    x = torch.randn(2, 5, 4)
    s0 = m_seq.init_state(B=2, device="cpu")
    _, s_warm = m_seq.forward_chunk(torch.randn(2, 3, 4), s0)
    y_seq, s_seq = m_seq.forward_chunk(x, s_warm)
    y_blk, s_blk = m_blk.forward_chunk(x, s_warm)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


def test_blockwise_boundary_at_position_zero_resets_M():
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
    m_seq, m_blk = _make_pair(d=4, order=2, block_size=8, seed=5)
    x = torch.randn(2, 6, 4)
    db = torch.tensor([
        [False, False, True, False, False, False],
        [False, True,  False, False, True,  False],
    ])
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, s_seq = m_seq.forward_chunk(x, s0, doc_boundaries=db)
    y_blk, s_blk = m_blk.forward_chunk(x, s0, doc_boundaries=db)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


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
    """SiLU+L2 on K bounds the chunkwise WY at T·N >> 1."""
    torch.manual_seed(0)
    d, order = 64, 2
    T = 1028
    m = DeltaProductMemory(
        n_embd=d, n_heads=1, order=order,
        finetune_mode=False, block_size=64,
    )
    s = m.init_state(B=1, device="cpu")
    x = torch.randn(1, T, d)
    y, new_state = m.forward_chunk(x, s)
    M_out = new_state[0]
    assert torch.isfinite(y).all()
    assert torch.isfinite(M_out).all()
    assert y.abs().max() < 1e3


# -- Multi-head fusion ----------------------------------------------------


def test_multihead_output_shape():
    d, nh = 12, 3
    m = DeltaProductMemory(n_embd=d, n_heads=nh, order=2)
    s = m.init_state(B=2, device="cpu")
    x = torch.randn(2, 5, d)
    y, new_state = m.forward_chunk(x, s)
    M_out = new_state[0]
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
    m_seq, m_blk = _make_pair(
        d=12, n_heads=3, order=2, block_size=8, seed=7,
    )
    x = torch.randn(2, 6, 12)
    s0 = m_seq.init_state(B=2, device="cpu")
    y_seq, s_seq = m_seq.forward_chunk(x, s0)
    y_blk, s_blk = m_blk.forward_chunk(x, s0)
    assert torch.allclose(y_seq, y_blk, atol=1e-5)
    assert torch.allclose(s_seq[0], s_blk[0], atol=1e-5)


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
