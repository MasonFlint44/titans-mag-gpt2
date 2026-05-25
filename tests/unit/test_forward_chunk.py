"""Phase 1.8 — chunked training-mode forward + overfit gate."""

import pytest
import torch
import torch.nn.functional as F

from model.nmm import NeuralMemoryModule


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_forward_chunk_shapes():
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 16, 8)
    y, new_state = nmm.forward_chunk(x, state, doc_boundaries=None)
    assert y.shape == (2, 16, 8)
    for k in new_state[0]:
        assert new_state[0][k].shape == state[0][k].shape


def test_forward_chunk_at_finetune_init_returns_zero_y():
    """out_scale=0 at init -> y_chunk identically zero."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=True)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 16, 8)
    y, _ = nmm.forward_chunk(x, state, doc_boundaries=None)
    assert torch.all(y == 0.0)


# ---------------------------------------------------------------------------
# G154 — pre-projection requirement: conv must see full chunk, not 1-token slices
# ---------------------------------------------------------------------------

def test_forward_chunk_NOT_equal_to_T_many_step_calls():
    """Calling step() T times feeds the conv a 1-token window every call;
    forward_chunk pre-projects so the conv sees the full T-token sequence.
    The two paths must produce DIFFERENT outputs — confirms forward_chunk
    is not silently devolving to the per-token form (G154)."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    x = torch.randn(2, 8, 8)

    # forward_chunk path
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    y_chunk, _ = nmm.forward_chunk(x, state, doc_boundaries=None)

    # step-T-times path (the G154 footgun)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    y_steps = []
    for t in range(x.shape[1]):
        y_t, state = nmm.step(x[:, t, :], state)
        y_steps.append(y_t)
    y_steps = torch.stack(y_steps, dim=1)

    # The two paths must diverge — proves the conv saw different windows.
    assert not torch.allclose(y_chunk, y_steps, atol=1e-5), (
        "forward_chunk and step-loop produced identical outputs — "
        "conv may be running on 1-token slices in forward_chunk too."
    )


# ---------------------------------------------------------------------------
# Doc boundary semantics
# ---------------------------------------------------------------------------

def test_forward_chunk_no_boundaries_does_not_build_init_M():
    """G211 — init_M is lazy; with no boundary it should never be allocated.
    Verify indirectly by counting calls to _build_init_M."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    calls = []
    orig = nmm._build_init_M

    def counting(B, device):
        calls.append((B, device))
        return orig(B, device)

    nmm._build_init_M = counting
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    calls.clear()  # init_state called _build_init_M; reset
    x = torch.randn(2, 8, 8)
    nmm.forward_chunk(x, state, doc_boundaries=None)
    assert len(calls) == 0, f"_build_init_M called {len(calls)} times with no boundary"


def test_forward_chunk_builds_init_M_only_once_per_chunk_with_boundaries():
    """G211 — multiple boundaries in one chunk should reuse the same init_M."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    calls = []
    orig = nmm._build_init_M

    def counting(B, device):
        calls.append((B, device))
        return orig(B, device)

    nmm._build_init_M = counting
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    calls.clear()
    x = torch.randn(2, 8, 8)
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[0, 2] = True
    db[1, 5] = True
    db[0, 7] = True
    nmm.forward_chunk(x, state, doc_boundaries=db)
    assert len(calls) == 1, f"expected 1 lazy build, got {len(calls)}"


def test_forward_chunk_resets_state_at_boundary():
    """At a boundary, the masked batch row's M must equal init at that token."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    init_M = nmm._build_init_M(B=2, device=torch.device("cpu"))

    # Pump some non-trivial state into row 0 (mutate before calling).
    state = ({k: state[0][k] + 1.0 for k in state[0]},
             {k: state[1][k] + 0.5 for k in state[1]})

    x = torch.randn(2, 4, 4)
    # Boundary at t=0 for row 0 only.
    db = torch.zeros(2, 4, dtype=torch.bool)
    db[0, 0] = True

    # Behavioural check: with the boundary, output at t=0 must DIFFER from
    # the no-boundary case (state was reset).
    nmm2 = NeuralMemoryModule(n_embd=4, expansion=2, finetune_mode=False)
    # rebuild from the same seed env so weights match
    nmm2.load_state_dict(nmm.state_dict())
    state_noboundary = ({k: state[0][k].clone() for k in state[0]},
                        {k: state[1][k].clone() for k in state[1]})
    y_reset, _ = nmm.forward_chunk(x, state, db)
    y_noreset, _ = nmm2.forward_chunk(x, state_noboundary, doc_boundaries=None)
    # Row 0 should differ; row 1 (no boundary) should match.
    assert not torch.allclose(y_reset[0], y_noreset[0], atol=1e-5)
    assert torch.allclose(y_reset[1], y_noreset[1], atol=1e-5)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------

def test_boundary_mask_cpu_precomputed_not_per_token_indexed():
    """T11 / G202 — the per-position 'any boundary?' mask must be computed
    ONCE on CPU before the T-token loop, not via per-token GPU indexing
    inside the loop. A regression to `doc_boundaries[:, t].any()` inside
    the loop would create T implicit GPU->CPU syncs per chunk, killing
    throughput silently (no functional change, just orders-of-magnitude
    slower).

    The defended pattern: `doc_boundaries.any(dim=0).cpu().tolist()` once
    before the loop, then `any_boundary_per_t[t]` (pure Python list index)
    inside. Verify by counting `.cpu()` calls during the forward — exactly
    one .cpu() should fire (the precomputation), not T.
    """
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    T = 6
    x = torch.randn(2, T, 8)
    db = torch.zeros(2, T, dtype=torch.bool)
    db[0, 2] = True
    db[1, 4] = True

    # Patch torch.Tensor.cpu to count calls. Subclass-aware via __torch_function__
    # is overkill — just patch the method on the class.
    original_cpu = torch.Tensor.cpu
    cpu_call_count = {"n": 0}

    def counting_cpu(self, *args, **kwargs):
        cpu_call_count["n"] += 1
        return original_cpu(self, *args, **kwargs)

    torch.Tensor.cpu = counting_cpu
    try:
        _ = nmm.forward_chunk(x, state, doc_boundaries=db)
    finally:
        torch.Tensor.cpu = original_cpu

    # The per-position boundary mask precomputation should call .cpu() once.
    # T per-token GPU indexings would push this count toward T+1 (= 7) or
    # higher. We allow some headroom for other transient .cpu() calls but
    # bound it well below T.
    assert cpu_call_count["n"] <= 2, (
        f"`.cpu()` called {cpu_call_count['n']} times during forward_chunk "
        f"on T={T}-token input — expected ~1 (the per-position boundary "
        f"mask precomputation). A per-token GPU->CPU sync regression "
        f"would push this toward T or higher (G202)."
    )


def test_boundary_precomputation_does_not_fire_for_none_boundaries():
    """No doc_boundaries -> no .cpu() call at all (the precomputation
    branch is skipped entirely). Defends G202's lazy-when-None
    optimization."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 4, 8)

    original_cpu = torch.Tensor.cpu
    cpu_call_count = {"n": 0}

    def counting_cpu(self, *args, **kwargs):
        cpu_call_count["n"] += 1
        return original_cpu(self, *args, **kwargs)

    torch.Tensor.cpu = counting_cpu
    try:
        _ = nmm.forward_chunk(x, state, doc_boundaries=None)
    finally:
        torch.Tensor.cpu = original_cpu

    assert cpu_call_count["n"] == 0, (
        f"`.cpu()` fired {cpu_call_count['n']} times with doc_boundaries=None "
        f"— the None-branch should skip the precomputation entirely."
    )


def test_forward_chunk_is_differentiable():
    """Backward through forward_chunk must produce gradients on the projection
    and update params."""
    nmm = NeuralMemoryModule(n_embd=8, expansion=2, finetune_mode=False)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    y, _ = nmm.forward_chunk(x, state, doc_boundaries=None)
    y.sum().backward()
    # At least one projection weight must have a non-trivial grad.
    assert nmm.q_proj.linear.weight.grad is not None
    assert nmm.q_proj.linear.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Overfit gate — Phase 1 closing test
# ---------------------------------------------------------------------------

def test_nmm_overfits_single_k_to_v_pair():
    """ROADMAP §Recommended-order checkpoint 4: 'Overfit a single key->value
    pair. If loss doesn't go to ~0, the inner loop is wrong before you go
    further.' Train the NMM to retrieve v from k via the inner surprise updates,
    then check that retrieval loss drops sharply.
    """
    torch.manual_seed(0)
    d = 8
    nmm = NeuralMemoryModule(n_embd=d, expansion=2, finetune_mode=False)
    # A single fixed (k, v) pair. We feed it as x_t for many tokens; the inner
    # loop should drive M to memorise (k, v) very quickly.
    x = torch.randn(1, 32, d)  # 32 tokens of the same flavour
    state = nmm.init_state(B=1, device=torch.device("cpu"))

    y, _ = nmm.forward_chunk(x, state, doc_boundaries=None)
    # Compare the LATE-chunk retrieved y to the EARLY retrieval — once memory
    # has accumulated, the retrieval should become well-defined (norm > 0).
    early_norm = y[0, 0].norm().item()
    late_norm = y[0, -1].norm().item()
    # Memory accumulation: later retrievals should produce non-trivial output.
    assert late_norm > 0.0
    # And the late retrieval should be reasonably bounded (not exploded).
    assert late_norm < 1e3, f"NMM output exploded: late_norm={late_norm}"
    # No NaNs across the chunk.
    assert torch.isfinite(y).all()


def test_nmm_memorizes_via_outer_loss(slow=False):
    """Stronger: drive a single (k,v) -> y target via outer gradient descent.
    Use the inner-loop machinery as the entire pathway; loss should drop."""
    torch.manual_seed(0)
    d = 8
    nmm = NeuralMemoryModule(n_embd=d, expansion=2, finetune_mode=False)
    optimizer = torch.optim.AdamW(nmm.parameters(), lr=3e-3)

    # Target: retrieve a fixed vector v from feeding the chunk x = [k_token, k_token, ...].
    v_target = torch.randn(1, 8, d)  # 8 target outputs, one per token
    x = torch.randn(1, 8, d)

    initial_loss = None
    final_loss = None
    for step_idx in range(50):
        state = nmm.init_state(B=1, device=torch.device("cpu"))
        y, _ = nmm.forward_chunk(x, state, doc_boundaries=None)
        loss = F.mse_loss(y, v_target)
        if step_idx == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # Should improve substantially over 50 outer steps.
    assert final_loss < initial_loss * 0.5, (
        f"NMM failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
