"""Behavior test: scan approximation vs true sequential, chunk_size monotonicity.

PLAN.md §6.1 (G232 update) claims the gap is "monotonically decreasing with
shorter chunk_size". This is the part of the claim we CAN verify directly:
at T=1 the gap is exactly zero (only one gradient, no M drift); at larger T
the M drift accumulates and the all-grads-at-M_0 approximation diverges
from the true M_{t-1} gradient.

We do NOT verify the "<5% on trained models" tightness — empirically that
holds only under specific training regimes (the inner-loop gates θ/η/α
moving in a direction that keeps M drift bounded). On synthetic random-
token training at our toy scale, the gap can actually GROW with training
because θ/η/α move away from sigmoid(~0)≈0.5 toward more variable values
that produce larger per-step M drift. That's a real model-dynamics
observation, not a scan implementation bug — see G232.
"""

import pytest
import torch

from config import TitansConfig
from model.nmm import _HAS_ASSOC_SCAN
from model.titans_gpt2 import TitansMAGGPT2


def _rel_error_at_T(model, T: int) -> float:
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, T, model.config.n_embd) * 0.3
    with torch.no_grad():
        y_seq, _ = nmm._forward_chunk_sequential(x, state, None)
        y_scan, _ = nmm._forward_chunk_scan(x, state, None)
    diff = (y_scan - y_seq).norm().item()
    ref = y_seq.norm().item() + 1e-8
    return diff / ref


@pytest.mark.slow
def test_scan_sequential_gap_increases_with_chunk_size():
    """G232 / PLAN.md §6.1: 'monotonically decreasing with shorter chunk_size'.
    Equivalent statement: rel error monotonically INCREASES with longer
    chunk_size. At T=1 the gap is exactly 0 (no M drift); at T>1 it grows.
    """
    if not _HAS_ASSOC_SCAN:
        pytest.skip("associative_scan unavailable on this PyTorch")

    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)

    # At T=1 the approximation is exact (only one gradient, no drift).
    assert _rel_error_at_T(model, T=1) < 1e-5

    # At larger T the gap grows. Sample at a few sizes and verify the
    # error at the larger size dominates the smaller — checked with a
    # multi-seed average to suppress per-seed noise.
    def avg_rel(T: int) -> float:
        accs = []
        for s in range(3):
            torch.manual_seed(s)
            accs.append(_rel_error_at_T(model, T=T))
        return sum(accs) / len(accs)

    err_T2 = avg_rel(2)
    err_T4 = avg_rel(4)
    err_T8 = avg_rel(8)

    assert err_T4 > err_T2, (
        f"rel error did not grow from T=2 to T=4: {err_T2:.3f} -> {err_T4:.3f}"
    )
    assert err_T8 > err_T4, (
        f"rel error did not grow from T=4 to T=8: {err_T4:.3f} -> {err_T8:.3f}"
    )
