"""Correctness tests for the analytical inner-gradient kernel.

The analytical kernel in `model/nmm_fused.py` is the only path used by:
- `_run_inner_loop` (sequential `block_size=1` training)
- `_forward_chunk_blockwise` (default training path) via `analytical_chunk_grad`

The earlier `nmm_fused_kernel` toggle is gone — the analytical kernel is
always-on. These tests directly pin the kernel against `vmap(grad(...))`
as the autograd oracle, so the kernel's correctness is independent of
which NMM forward path invokes it.

`per_sample_grad_fn` (the vmap-based path) still exists in
`NeuralMemoryModule` because the decode-time `step` and `step_with_conv`
methods use it at inference time — but it's no longer reachable from
training.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from config import TitansConfig
from model.nmm import MemoryMLP
from model.nmm_fused import analytical_inner_grad, batched_retrieve
from model.titans_gpt2 import TitansMAGGPT2


# Loose-ish tolerances: T=6 inner iterations × NS5 + matmul chains gives
# ~130+ chained matmuls; fp32 round-off accumulates to ~1e-3 in the worst
# entries (especially where two near-cancelling terms differ in operation
# order between the autograd path and the analytical formulae). We're not
# testing precision — we're testing mathematical equivalence.
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
# Kernel correctness: analytical inner gradient matches autograd reference.
# This is the load-bearing safety net — every other NMM training path
# delegates to this kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("low_rank", [None, 16])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_analytical_grad_matches_autograd_reference(low_rank, reduction):
    """Per-token analytical gradient must match `vmap(grad(inner_loss))`
    within fp32 round-off, for both full-rank and low-rank MemoryMLP and
    both reductions. If this regresses, every NMM-driven training run is
    silently using the wrong gradient."""
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
            f"grad[{key}] mismatch: max_abs="
            f"{(ref[key] - mine[key]).abs().max().item():.2e}"
        )


@pytest.mark.parametrize("low_rank", [None, 16])
def test_batched_retrieve_matches_autograd_reference(low_rank):
    """Per-token retrieval must match `vmap(MemoryMLP(...))` — covers
    the analytical forward, not just the backward. Used downstream in
    `_run_inner_loop` to produce y_t."""
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
# bf16 sanity: NMM forward stays numerically bounded under bf16 state.
# Strict equivalence isn't enforced (after 6 sequential NS5-normalized
# bf16 steps the trajectory drifts) — we just verify the path doesn't
# diverge or NaN out.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 path needs CUDA")
def test_bf16_state_runs_and_produces_finite_output():
    """state_dtype='bf16' must run end-to-end and produce finite, bounded
    output via the analytical-kernel inner loop. Smoke guard against
    regressions in bf16-dtype plumbing inside the analytical path."""
    from model.nmm import NeuralMemoryModule
    torch.manual_seed(4)
    d, T, B = 32, 6, 2
    device = torch.device("cuda")
    nmm = NeuralMemoryModule(
        n_embd=d, expansion=2, kernel_size=2,
        spectral_norm=True, finetune_mode=False,
        state_dtype="bf16", low_rank=None,
    ).to(device)
    x = torch.randn(B, T, d, device=device) * 0.3
    s0 = nmm.init_state(B, device)
    y, _ = nmm.forward_chunk(x, s0, None)
    assert torch.isfinite(y).all(), "bf16 NMM forward produced non-finite output"
    # Magnitude sanity — the output should be O(1) given normalized inputs.
    assert y.abs().mean().item() < 10.0, (
        f"bf16 NMM output magnitude {y.abs().mean().item():.2f} is suspiciously "
        f"large — possible runaway state from gradient accumulation error"
    )


# ---------------------------------------------------------------------------
# compile_inner_loop (G264a) — separate config knob, tested here for
# co-location with the NMM-internal training-path tests. Cut in a later
# refactor would move these accordingly.
# ---------------------------------------------------------------------------


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
