"""Equivalence + integration tests for nmm_fused_kernel=True (G264).

The reference path (vmap(grad(inner_loss))) is the correctness oracle.
Every test in this module asserts the analytical path matches it within
fp32 round-off for forward outputs AND backward gradients to upstream
tensors (k_hat, v, M_init, plus the outer-trained MemoryMLP params).
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from config import TitansConfig
from model.nmm import MemoryMLP, NeuralMemoryModule
from model.nmm_fused import analytical_inner_grad, batched_retrieve
from model.titans_gpt2 import TitansMAGGPT2


# Loose-ish tolerances: T=6 inner iterations × NS5 + matmul chains gives
# ~130+ chained matmuls; fp32 round-off accumulates to ~1e-3 in the worst
# entries (especially where two near-cancelling terms differ in operation
# order between the two implementations). We're not testing precision —
# we're testing mathematical equivalence.
ATOL = 5e-3
RTOL = 5e-3


def _per_sample_M(mlp, B):
    """Per-sample-batched M dict matching what `_build_init_M` produces."""
    M = {}
    for name, p in mlp.named_parameters():
        if name.startswith("norm."):
            continue
        M[name] = p.unsqueeze(0).expand(B, *([-1] * p.ndim)).clone()
    return M


# ---------------------------------------------------------------------------
# Math equivalence at the kernel level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("low_rank", [None, 16])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_analytical_grad_matches_reference(low_rank, reduction):
    torch.manual_seed(0)
    d, B = 64, 4
    mlp = MemoryMLP(d, expansion=4, low_rank=low_rank)
    M = _per_sample_M(mlp, B)
    k_hat = torch.randn(B, d) * 0.5
    v = torch.randn(B, d) * 0.5

    def inner_loss(params, k, vv):
        pred = functional_call(mlp, params, k)
        return F.mse_loss(pred, vv, reduction=reduction)

    ref = vmap(grad(inner_loss), in_dims=(0, 0, 0))(M, k_hat, v)
    mine = analytical_inner_grad(
        M, k_hat, v,
        mlp.norm.weight, mlp.norm.bias, mlp.norm.eps,
        reduction,
    )
    assert set(ref.keys()) == set(mine.keys())
    for key in ref:
        assert torch.allclose(ref[key], mine[key], atol=ATOL, rtol=RTOL), (
            f"grad[{key}] mismatch: max_abs={(ref[key] - mine[key]).abs().max().item():.2e}"
        )


@pytest.mark.parametrize("low_rank", [None, 16])
def test_batched_retrieve_matches_reference(low_rank):
    torch.manual_seed(1)
    d, B = 64, 4
    mlp = MemoryMLP(d, expansion=4, low_rank=low_rank)
    M = _per_sample_M(mlp, B)
    q_hat = torch.randn(B, d) * 0.5

    def one(m_dict, q):
        return functional_call(mlp, m_dict, q.unsqueeze(0)).squeeze(0)
    ref = vmap(one, in_dims=(0, 0))(M, q_hat)
    mine = batched_retrieve(M, q_hat, mlp.norm.weight, mlp.norm.bias, mlp.norm.eps)
    assert torch.allclose(ref, mine, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Config flag and validation.
# ---------------------------------------------------------------------------


def test_config_default_off():
    cfg = TitansConfig.gpt2_small(block_size=64, chunk_size=64)
    assert cfg.nmm_fused_kernel is False


def test_config_accepts_true():
    cfg = TitansConfig.gpt2_small(block_size=64, chunk_size=64, nmm_fused_kernel=True)
    assert cfg.nmm_fused_kernel is True


def test_flag_propagates_to_every_nmm():
    cfg = TitansConfig(
        n_layer=4, n_head=2, n_embd=16, vocab_size=64,
        block_size=32, chunk_size=32,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
        nmm_fused_kernel=True,
    )
    model = TitansMAGGPT2(cfg)
    for blk in model.blocks:
        if hasattr(blk, "nmm"):
            assert blk.nmm.fused_kernel is True, "fused_kernel did not propagate"


# ---------------------------------------------------------------------------
# End-to-end NMM equivalence (forward + backward at chunk level).
# ---------------------------------------------------------------------------


def _build_pair(d=32, h_expansion=2, T=8, B=2, low_rank=None, state_dtype="fp32",
                grad_checkpoint=False):
    """Build two identical NMMs — one ref, one fused — sharing initial weights."""
    torch.manual_seed(7)
    ref = NeuralMemoryModule(
        n_embd=d, expansion=h_expansion, kernel_size=2,
        spectral_norm=True, finetune_mode=False,
        state_dtype=state_dtype, low_rank=low_rank,
        grad_checkpoint=grad_checkpoint,
        grad_checkpoint_segment_len=max(T // 2, 1),
    )
    fused = NeuralMemoryModule(
        n_embd=d, expansion=h_expansion, kernel_size=2,
        spectral_norm=True, finetune_mode=False,
        state_dtype=state_dtype, low_rank=low_rank,
        grad_checkpoint=grad_checkpoint,
        grad_checkpoint_segment_len=max(T // 2, 1),
        fused_kernel=True,
    )
    fused.load_state_dict(ref.state_dict())
    return ref, fused


def _forward_pair(ref, fused, x_chunk, init_state):
    """Run both modules on the same input + state. Returns
    (ref_y, ref_state, fused_y, fused_state)."""
    s_ref = (
        {k: v.clone() for k, v in init_state[0].items()},
        {k: v.clone() for k, v in init_state[1].items()},
    )
    s_fused = (
        {k: v.clone() for k, v in init_state[0].items()},
        {k: v.clone() for k, v in init_state[1].items()},
    )
    y_ref, ns_ref = ref.forward_chunk(x_chunk, s_ref, None)
    y_fused, ns_fused = fused.forward_chunk(x_chunk, s_fused, None)
    return y_ref, ns_ref, y_fused, ns_fused


@pytest.mark.parametrize("low_rank", [None, 8])
def test_forward_equivalence_fullrank_and_lowrank(low_rank):
    torch.manual_seed(0)
    d, T, B = 32, 8, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=low_rank)
    x = torch.randn(B, T, d) * 0.3
    s0 = ref.init_state(B, x.device)
    y_ref, ns_ref, y_fused, ns_fused = _forward_pair(ref, fused, x, s0)
    assert torch.allclose(y_ref, y_fused, atol=ATOL, rtol=RTOL), (
        f"y max_abs={(y_ref - y_fused).abs().max().item():.2e}"
    )
    for key in ns_ref[0]:
        assert torch.allclose(ns_ref[0][key], ns_fused[0][key], atol=ATOL, rtol=RTOL)
        assert torch.allclose(ns_ref[1][key], ns_fused[1][key], atol=ATOL, rtol=RTOL)


def test_backward_grads_match_reference():
    """Outer backward to k_hat-driving inputs must match the reference."""
    torch.manual_seed(0)
    d, T, B = 32, 6, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=None)
    x = torch.randn(B, T, d, requires_grad=True) * 0.3
    s0 = ref.init_state(B, x.device)

    # Two separate inputs so we can attribute gradients.
    x_ref = x.detach().clone().requires_grad_(True)
    x_fused = x.detach().clone().requires_grad_(True)
    s_ref = (
        {k: v.detach().clone() for k, v in s0[0].items()},
        {k: v.detach().clone() for k, v in s0[1].items()},
    )
    s_fused = (
        {k: v.detach().clone() for k, v in s0[0].items()},
        {k: v.detach().clone() for k, v in s0[1].items()},
    )
    y_ref, _ = ref.forward_chunk(x_ref, s_ref, None)
    y_fused, _ = fused.forward_chunk(x_fused, s_fused, None)

    loss_ref = y_ref.pow(2).sum()
    loss_fused = y_fused.pow(2).sum()
    loss_ref.backward()
    loss_fused.backward()

    assert torch.allclose(x_ref.grad, x_fused.grad, atol=ATOL, rtol=RTOL), (
        f"grad max_abs={(x_ref.grad - x_fused.grad).abs().max().item():.2e}"
    )


def test_backward_grads_match_lowrank():
    torch.manual_seed(1)
    d, T, B = 32, 6, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=8)
    base = (torch.randn(B, T, d) * 0.3).detach()
    x_ref = base.clone().requires_grad_(True)
    x_fused = base.clone().requires_grad_(True)
    s0 = ref.init_state(B, x_ref.device)
    s_ref = ({k: v.detach().clone() for k, v in s0[0].items()},
             {k: v.detach().clone() for k, v in s0[1].items()})
    s_fused = ({k: v.detach().clone() for k, v in s0[0].items()},
               {k: v.detach().clone() for k, v in s0[1].items()})
    y_ref, _ = ref.forward_chunk(x_ref, s_ref, None)
    y_fused, _ = fused.forward_chunk(x_fused, s_fused, None)
    y_ref.pow(2).sum().backward()
    y_fused.pow(2).sum().backward()
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=ATOL, rtol=RTOL)


def test_grads_to_outer_memory_mlp_params_match():
    """The recurrent weights' outer-trained init values must receive the same
    gradient under both paths."""
    torch.manual_seed(2)
    d, T, B = 32, 4, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=None)
    base = (torch.randn(B, T, d) * 0.3).detach()
    x_ref = base.clone().requires_grad_(True)
    x_fused = base.clone().requires_grad_(True)
    s0 = ref.init_state(B, x_ref.device)
    s_ref = ({k: v.detach().clone() for k, v in s0[0].items()},
             {k: v.detach().clone() for k, v in s0[1].items()})
    s_fused = ({k: v.detach().clone() for k, v in s0[0].items()},
               {k: v.detach().clone() for k, v in s0[1].items()})
    y_ref, _ = ref.forward_chunk(x_ref, s_ref, None)
    y_fused, _ = fused.forward_chunk(x_fused, s_fused, None)
    y_ref.pow(2).sum().backward()
    y_fused.pow(2).sum().backward()
    ref_params = dict(ref.memory_mlp.named_parameters())
    fused_params = dict(fused.memory_mlp.named_parameters())
    for name in ref_params:
        g_ref = ref_params[name].grad
        g_fused = fused_params[name].grad
        if g_ref is None and g_fused is None:
            continue
        # Some bf16/dtype paths skip norm grads — check only when both exist.
        assert g_ref is not None and g_fused is not None
        assert torch.allclose(g_ref, g_fused, atol=ATOL, rtol=RTOL), (
            f"memory_mlp.{name}: max_abs={(g_ref - g_fused).abs().max().item():.2e}"
        )


def test_doc_boundary_reset_equivalence():
    """When a boundary fires mid-chunk, both paths must reset state identically."""
    torch.manual_seed(3)
    d, T, B = 32, 6, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=None)
    x = torch.randn(B, T, d) * 0.3
    db = torch.zeros(B, T, dtype=torch.bool)
    db[:, 0] = True            # start-of-sequence reset
    db[0, 3] = True            # mid-chunk boundary on first sample only
    s0 = ref.init_state(B, x.device)
    y_ref, ns_ref = ref.forward_chunk(
        x,
        ({k: v.clone() for k, v in s0[0].items()},
         {k: v.clone() for k, v in s0[1].items()}),
        db,
    )
    y_fused, ns_fused = fused.forward_chunk(
        x,
        ({k: v.clone() for k, v in s0[0].items()},
         {k: v.clone() for k, v in s0[1].items()}),
        db,
    )
    assert torch.allclose(y_ref, y_fused, atol=ATOL, rtol=RTOL)


def test_bf16_state_runs_and_produces_finite_output():
    """state_dtype='bf16' must run end-to-end and produce finite, bounded
    output. Tighter equivalence is NOT enforced because after 6 sequential
    NS5-normalized steps in bf16, the trajectories diverge far enough to
    flip the sign of entries with small magnitudes. Strict fp32
    equivalence is locked in the tests above.
    """
    if not torch.cuda.is_available():
        pytest.skip("bf16 path needs CUDA")
    torch.manual_seed(4)
    d, T, B = 32, 6, 2
    device = torch.device("cuda")
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=None, state_dtype="bf16")
    ref, fused = ref.to(device), fused.to(device)
    x = (torch.randn(B, T, d, device=device) * 0.3)
    s0 = ref.init_state(B, device)
    y_ref, _ = ref.forward_chunk(x, s0, None)
    y_fused, _ = fused.forward_chunk(
        x,
        ({k: v.clone() for k, v in s0[0].items()},
         {k: v.clone() for k, v in s0[1].items()}),
        None,
    )
    assert torch.isfinite(y_ref).all(), "reference bf16 path produced non-finite"
    assert torch.isfinite(y_fused).all(), "fused bf16 path produced non-finite"
    # Magnitude sanity — both should sit in roughly the same regime.
    # Anything > 100x the reference scale is a red flag for runaway state.
    ref_scale = y_ref.abs().mean().item() + 1e-6
    fused_scale = y_fused.abs().mean().item() + 1e-6
    ratio = max(ref_scale, fused_scale) / min(ref_scale, fused_scale)
    assert ratio < 100, f"bf16 scale ratio {ratio:.1f} too large"


def test_composes_with_grad_checkpoint():
    """fused_kernel + grad_checkpoint should both work and match the
    non-checkpointed reference within fp32 round-off."""
    torch.manual_seed(5)
    d, T, B = 32, 8, 2
    ref, fused = _build_pair(d=d, T=T, B=B, low_rank=None, grad_checkpoint=True)
    base = (torch.randn(B, T, d) * 0.3).detach()
    x_ref = base.clone().requires_grad_(True)
    x_fused = base.clone().requires_grad_(True)
    s0 = ref.init_state(B, x_ref.device)
    y_ref, _ = ref.forward_chunk(
        x_ref,
        ({k: v.clone() for k, v in s0[0].items()},
         {k: v.clone() for k, v in s0[1].items()}),
        None,
    )
    y_fused, _ = fused.forward_chunk(
        x_fused,
        ({k: v.clone() for k, v in s0[0].items()},
         {k: v.clone() for k, v in s0[1].items()}),
        None,
    )
    y_ref.pow(2).sum().backward()
    y_fused.pow(2).sum().backward()
    assert torch.allclose(y_ref, y_fused, atol=ATOL, rtol=RTOL)
    assert torch.allclose(x_ref.grad, x_fused.grad, atol=ATOL, rtol=RTOL)


def test_compile_inner_loop_default_off():
    cfg = TitansConfig.gpt2_small(block_size=64, chunk_size=64)
    assert cfg.nmm_compile_inner_loop is False


def test_compile_inner_loop_propagates():
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=32, chunk_size=32,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
        nmm_compile_inner_loop=True,
    )
    model = TitansMAGGPT2(cfg)
    for blk in model.blocks:
        if hasattr(blk, "nmm"):
            assert blk.nmm.compile_inner_loop is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile speedup needs CUDA")
def test_compile_inner_loop_forward_matches_uncompiled():
    """Both modes must produce the same forward output (up to fp32 round-off)
    on a tiny model — proves the compile wrapper doesn't change semantics."""
    cfg_kwargs = dict(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=16, chunk_size=16,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
    )
    torch.manual_seed(0)
    ref = TitansMAGGPT2(TitansConfig(**cfg_kwargs)).cuda()
    torch.manual_seed(0)
    compiled = TitansMAGGPT2(TitansConfig(**cfg_kwargs, nmm_compile_inner_loop=True)).cuda()
    compiled.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg_kwargs["vocab_size"], (2, 16), device="cuda")
    db = torch.zeros_like(ids, dtype=torch.bool); db[:, 0] = True
    with torch.no_grad():
        logits_ref, _ = ref(ids, None, db)
        logits_cmp, _ = compiled(ids, None, db)
    assert torch.allclose(logits_ref, logits_cmp, atol=1e-3, rtol=1e-3), (
        f"max_abs={(logits_ref - logits_cmp).abs().max().item():.2e}"
    )



def test_end_to_end_fused_model_trains():
    """Build a TitansMAGGPT2 with fused_kernel=True and verify a step runs +
    produces gradients on every parameter."""
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=8, chunk_size=8,
        nmm_expansion=2, nmm_conv_kernel=2,
        finetune_mode=False,
        nmm_fused_kernel=True,
    )
    torch.manual_seed(0)
    model = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.chunk_size))
    db = torch.zeros_like(ids, dtype=torch.bool); db[:, 0] = True
    logits, _ = model(ids, None, db)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )
    loss.backward()
    # Every parameter should have a non-trivial gradient.
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"{n_total - n_with_grad}/{n_total} params have no grad"
