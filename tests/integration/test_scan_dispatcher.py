"""Phase 6.2 — associative_scan dispatcher gating + _unwrap + allow_scan_training."""

import pytest
import torch

from config import TitansConfig
from model import _unwrap
from model.nmm import _HAS_ASSOC_SCAN, allow_scan_training
from model.titans_gpt2 import TitansMAGGPT2


def _tiny_model():
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    return cfg, TitansMAGGPT2(cfg)


# ---------------------------------------------------------------------------
# G215 — _associative_scan resolves via documented or private path
# ---------------------------------------------------------------------------

def test_associative_scan_resolution_is_consistent():
    """Either path must yield a callable _associative_scan and True flag."""
    from model.nmm import _associative_scan
    if _HAS_ASSOC_SCAN:
        assert callable(_associative_scan)
    else:
        assert _associative_scan is None


# ---------------------------------------------------------------------------
# G164 — dispatcher gates on torch.is_grad_enabled, not self.training
# ---------------------------------------------------------------------------

def test_dispatcher_uses_sequential_when_grad_enabled():
    """In train mode with autograd on, the scan path must NOT be selected
    (associative_scan lacks autograd). Verified by intercepting both methods."""
    cfg, model = _tiny_model()
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8)

    seq_called = [0]
    scan_called = [0]
    orig_seq = nmm._forward_chunk_sequential
    orig_scan = nmm._forward_chunk_scan

    def count_seq(*a, **kw):
        seq_called[0] += 1
        return orig_seq(*a, **kw)

    def count_scan(*a, **kw):
        scan_called[0] += 1
        return orig_scan(*a, **kw)

    nmm._forward_chunk_sequential = count_seq
    nmm._forward_chunk_scan = count_scan

    # Train mode, grad enabled -> sequential.
    nmm.forward_chunk(x, state, None)
    assert seq_called[0] == 1
    assert scan_called[0] == 0


def test_dispatcher_uses_scan_under_no_grad_when_no_boundaries():
    """Under torch.no_grad with no doc boundaries, the scan path is selected
    (when available)."""
    if not _HAS_ASSOC_SCAN:
        pytest.skip("associative_scan unavailable on this PyTorch")
    cfg, model = _tiny_model()
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8)

    seq_called = [0]
    scan_called = [0]
    orig_seq = nmm._forward_chunk_sequential
    orig_scan = nmm._forward_chunk_scan

    def count_seq(*a, **kw):
        seq_called[0] += 1
        return orig_seq(*a, **kw)

    def count_scan(*a, **kw):
        scan_called[0] += 1
        return orig_scan(*a, **kw)

    nmm._forward_chunk_sequential = count_seq
    nmm._forward_chunk_scan = count_scan

    with torch.no_grad():
        nmm.forward_chunk(x, state, None)
    assert scan_called[0] == 1
    assert seq_called[0] == 0


def test_dispatcher_falls_back_to_sequential_on_doc_boundaries():
    """The scan path can't do mid-chunk state resets — boundaries force sequential."""
    if not _HAS_ASSOC_SCAN:
        pytest.skip("associative_scan unavailable")
    cfg, model = _tiny_model()
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8)
    db = torch.zeros(1, 8, dtype=torch.bool)
    db[0, 3] = True

    scan_called = [0]
    orig_scan = nmm._forward_chunk_scan

    def count_scan(*a, **kw):
        scan_called[0] += 1
        return orig_scan(*a, **kw)

    nmm._forward_chunk_scan = count_scan
    with torch.no_grad():
        nmm.forward_chunk(x, state, db)
    assert scan_called[0] == 0


# ---------------------------------------------------------------------------
# Scan output approximates sequential
# ---------------------------------------------------------------------------

def test_scan_implementation_matches_M0_approx_sequential_exactly():
    """The scan computes "sequential update with all gradients at M_0"
    exactly — this is the scan's contract regardless of how far that
    differs from the true M_{t-1}-gradient sequential. Verify the scan's
    implementation correctness by comparing against a manual M_0-approx
    sequential reference.
    """
    if not _HAS_ASSOC_SCAN:
        pytest.skip("associative_scan unavailable")
    import torch.nn.functional as F
    from model.nmm import _dict_add, _dict_sub, _scale, newton_schulz5

    torch.manual_seed(0)
    cfg, model = _tiny_model()
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 4, 8) * 0.3

    with torch.no_grad():
        y_scan, _ = nmm._forward_chunk_scan(x, state, None)

    # Manual reference: sequential with ALL grads at M_0 (the scan's contract).
    M0, S0 = state
    with torch.no_grad():
        M = {k: v.clone() for k, v in M0.items()}
        S = {k: v.clone() for k, v in S0.items()}
        k_hat = F.normalize(F.silu(nmm.k_proj(x)), dim=-1)
        q_hat = F.normalize(F.silu(nmm.q_proj(x)), dim=-1)
        v_arr = F.silu(nmm.v_proj(x))
        theta = torch.sigmoid(nmm.W_theta(x)).squeeze(-1)
        eta = torch.sigmoid(nmm.W_eta(x)).squeeze(-1)
        alpha = torch.sigmoid(nmm.W_alpha(x)).squeeze(-1)
        y_list = []
        for t in range(x.shape[1]):
            # Capture M_prev BEFORE the update — the scan honors the model
            # config's `retrieval_from_M_prev`; our manual reference must
            # mirror that branch to keep the parity invariant valid.
            M_prev = {k: v.clone() for k, v in M.items()}
            g = nmm.per_sample_grad_fn(M0, k_hat[:, t, :], v_arr[:, t, :])
            g_tilde = {key: newton_schulz5(val) for key, val in g.items()}
            S = _dict_sub(_scale(eta[:, t], S), _scale(theta[:, t], g_tilde))
            M = _dict_add(_scale(1 - alpha[:, t], M), S)
            M_for_retrieval = M_prev if nmm.retrieval_from_M_prev else M
            y_t = nmm.out_scale * nmm._batched_retrieve(
                M_for_retrieval, q_hat[:, t, :],
            )
            y_list.append(y_t)
        y_ref = torch.stack(y_list, dim=1)

    assert torch.allclose(y_scan, y_ref, atol=1e-5), (
        f"scan disagrees with manual M_0-approx-sequential: "
        f"max diff = {(y_scan - y_ref).abs().max().item():.3e}"
    )


def test_scan_output_finite_and_reasonable_magnitude():
    """Behavioural check: the scan must not produce NaN/Inf, and output
    magnitude must stay within the same order as the sequential path.

    The "<5%" target in docs/PLAN.md §6.1 is aspirational for trained models
    (gradients structured, small in magnitude). On a random untrained NMM
    with random inputs, the M_0-vs-M_{t-1} approximation can reach 70-90%
    relative error per chunk — see G232 in docs/GAP_HISTORY.md.
    """
    if not _HAS_ASSOC_SCAN:
        pytest.skip("associative_scan unavailable")
    torch.manual_seed(0)
    cfg, model = _tiny_model()
    nmm = model.blocks[0].nmm
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 4, 8) * 0.3

    with torch.no_grad():
        y_seq, _ = nmm._forward_chunk_sequential(x, state, None)
        y_scan, _ = nmm._forward_chunk_scan(x, state, None)

    assert torch.isfinite(y_scan).all()
    # Same order of magnitude — within 10x.
    seq_norm = y_seq.norm().item()
    scan_norm = y_scan.norm().item()
    assert 0.1 * seq_norm < scan_norm < 10 * seq_norm


# ---------------------------------------------------------------------------
# G180 — allow_scan_training sets flag on every block.nmm
# ---------------------------------------------------------------------------

def test_allow_scan_training_propagates_to_every_nmm():
    cfg, model = _tiny_model()
    # Initially the flag is unset (defaults False via getattr).
    for block in model.blocks:
        assert getattr(block.nmm, "_allow_scan_training", False) is False

    allow_scan_training(model, True)
    for block in model.blocks:
        assert block.nmm._allow_scan_training is True

    allow_scan_training(model, False)
    for block in model.blocks:
        assert block.nmm._allow_scan_training is False


def test_top_level_model_attr_set_does_NOT_enable_scan():
    """G180 footgun: setting model._allow_scan_training is silently ineffective
    (the dispatcher reads it on the NMM, not the model). Verify the bad path
    doesn't accidentally work."""
    cfg, model = _tiny_model()
    model._allow_scan_training = True  # the silently-wrong pattern
    for block in model.blocks:
        # Still False on the NMM — flag never propagated.
        assert getattr(block.nmm, "_allow_scan_training", False) is False


# ---------------------------------------------------------------------------
# G184/G188/G195 — _unwrap
# ---------------------------------------------------------------------------

def test_unwrap_returns_unwrapped_for_bare_module():
    cfg, model = _tiny_model()
    assert _unwrap(model) is model


def test_unwrap_strips_torch_compile_orig_mod():
    """Simulate torch.compile's OptimizedModule by manually attaching ._orig_mod."""
    cfg, model = _tiny_model()

    class FakeCompiled:
        def __init__(self, m):
            self._orig_mod = m

    wrapped = FakeCompiled(model)
    assert _unwrap(wrapped) is model


def test_unwrap_strips_DDP_module_attribute():
    cfg, model = _tiny_model()

    class FakeDDP:
        def __init__(self, m):
            self.module = m

    wrapped = FakeDDP(model)
    assert _unwrap(wrapped) is model


def test_unwrap_strips_stacked_DDP_over_torch_compile():
    """DDP(torch.compile(model)) — needs both layers stripped."""
    cfg, model = _tiny_model()

    class FakeCompiled:
        def __init__(self, m):
            self._orig_mod = m

    class FakeDDP:
        def __init__(self, m):
            self.module = m

    wrapped = FakeDDP(FakeCompiled(model))
    assert _unwrap(wrapped) is model
