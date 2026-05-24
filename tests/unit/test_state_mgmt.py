"""Phase 1.9 — reset_state and detach_states."""

import pytest
import torch

from model.nmm import NeuralMemoryModule, detach_states, reset_state


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

def test_reset_state_unmasked_entries_byte_identical():
    """Where mask=False, the original state value must survive bit-perfectly."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    state = nmm.init_state(B=4, device=torch.device("cpu"))
    init_M = nmm._build_init_M(B=4, device=torch.device("cpu"))

    # Mutate state[0] (M) so it differs from init.
    M, S = state
    for k in M:
        M[k] = M[k] + 1.0
        S[k] = S[k] + 0.5
    mutated = (M, S)

    # Reset only batch index 1.
    mask = torch.tensor([False, True, False, False])
    new_state = reset_state(mutated, mask, init_M)
    new_M, new_S = new_state

    # Unmasked rows (0, 2, 3) must be byte-identical to the mutated state.
    for k in new_M:
        for b in (0, 2, 3):
            assert torch.equal(new_M[k][b], mutated[0][k][b])
            assert torch.equal(new_S[k][b], mutated[1][k][b])

    # Masked row (1) must equal init.
    for k in new_M:
        assert torch.equal(new_M[k][1], init_M[k][1])
        assert torch.equal(new_S[k][1], torch.zeros_like(new_S[k][1]))


def test_reset_state_uses_torch_where_not_in_place():
    """Confirm reset is differentiable: in-place mutation would break here."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    M_dict = {"W1.weight": torch.randn(2, 8, 4, requires_grad=True)}
    S_dict = {"W1.weight": torch.randn(2, 8, 4, requires_grad=True)}
    init_M = {"W1.weight": torch.zeros(2, 8, 4)}
    mask = torch.tensor([True, False])
    # In-place index assignment on a leaf with requires_grad=True raises
    # RuntimeError. reset_state must NOT raise — proves it's not in-place.
    new_state = reset_state((M_dict, S_dict), mask, init_M)
    assert isinstance(new_state, tuple)


def test_reset_state_all_true_mask_returns_init_everywhere():
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    init_M = nmm._build_init_M(B=2, device=torch.device("cpu"))
    mask = torch.tensor([True, True])
    new_M, new_S = reset_state(state, mask, init_M)
    for k in new_M:
        assert torch.equal(new_M[k], init_M[k])
        assert torch.all(new_S[k] == 0.0)


def test_reset_state_all_false_mask_is_passthrough():
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    init_M = nmm._build_init_M(B=2, device=torch.device("cpu"))
    mask = torch.tensor([False, False])
    new_M, new_S = reset_state(state, mask, init_M)
    for k in new_M:
        assert torch.equal(new_M[k], state[0][k])
        assert torch.equal(new_S[k], state[1][k])


# ---------------------------------------------------------------------------
# detach_states
# ---------------------------------------------------------------------------

def test_detach_states_passes_None_through():
    """First-step case: nmm_states is None until model.forward seeds it."""
    assert detach_states(None) is None


def test_detach_states_severs_grad_tape():
    """Detached tensors must have requires_grad=False and no grad_fn."""
    M = {"W1.weight": torch.randn(2, 4, requires_grad=True)}
    S = {"W1.weight": torch.randn(2, 4, requires_grad=True)}
    op_M = {k: v * 2.0 for k, v in M.items()}  # gives them a grad_fn
    op_S = {k: v * 2.0 for k, v in S.items()}
    states = [(op_M, op_S)]
    detached = detach_states(states)
    for M_d, S_d in detached:
        for v in M_d.values():
            assert v.requires_grad is False
            assert v.grad_fn is None
        for v in S_d.values():
            assert v.requires_grad is False


def test_detach_states_preserves_values_bitwise():
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    states = [nmm.init_state(B=2, device=torch.device("cpu")) for _ in range(3)]
    detached = detach_states(states)
    assert len(detached) == 3
    for orig, det in zip(states, detached):
        for k in orig[0]:
            assert torch.equal(orig[0][k], det[0][k])
            assert torch.equal(orig[1][k], det[1][k])


def test_detach_states_returns_new_list_not_in_place():
    """Detach must not mutate the input list/dicts."""
    nmm = NeuralMemoryModule(n_embd=4, expansion=2)
    states = [nmm.init_state(B=2, device=torch.device("cpu"))]
    detached = detach_states(states)
    assert detached is not states
    assert detached[0] is not states[0]
    assert detached[0][0] is not states[0][0]  # new dicts
