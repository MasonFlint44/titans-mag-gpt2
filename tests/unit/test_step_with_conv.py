"""NMM step_with_conv parity tests.

Item 6 folded the conv buffer into the per-layer NMM state tuple
`(M, S, conv_buf)`. The decode-time `step_with_conv(x_t, state)` signature
no longer takes a separate conv buffer — it's inside `state`.

Parity invariant: processing T tokens via `forward_chunk` should produce
the same y at position T as processing the first T-1 tokens via
`forward_chunk`, then the Tth token via `step_with_conv` against the
post-warmup state. Same for output state.
"""

import torch

from model.nmm import NeuralMemoryModule


def _nmm(n_embd=8, expansion=2, kernel_size=4, finetune_mode=False):
    return NeuralMemoryModule(
        n_embd=n_embd,
        expansion=expansion,
        kernel_size=kernel_size,
        spectral_norm=True,
        finetune_mode=finetune_mode,
    )


# ---------------------------------------------------------------------------
# State shape
# ---------------------------------------------------------------------------

def test_init_state_includes_conv_buf():
    nmm = _nmm(n_embd=8, kernel_size=4)
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    assert len(state) == 3, f"expected (M, S, conv_buf) triple, got {len(state)}"
    _, _, conv_buf = state
    assert set(conv_buf.keys()) == {"q", "k", "v"}
    for v in conv_buf.values():
        assert v.shape == (2, 3, 8)  # [B, k-1, d]
        assert torch.all(v == 0.0), "fresh conv_buf must be zeros"


def test_init_state_conv_buf_empty_when_kernel_size_one():
    nmm = _nmm(n_embd=8, kernel_size=1)
    _, _, conv_buf = nmm.init_state(B=2, device=torch.device("cpu"))
    for v in conv_buf.values():
        assert v.shape == (2, 0, 8)


# ---------------------------------------------------------------------------
# step_with_conv parity against forward_chunk
# ---------------------------------------------------------------------------

def test_step_with_conv_matches_forward_chunk_at_token_T():
    """Process the prefix via forward_chunk (which threads conv_buf through
    state), then the Tth token via step_with_conv. y_T should equal what
    forward_chunk on the full T tokens would compute at position T."""
    torch.manual_seed(0)
    n_embd, T = 8, 6
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    x_full = torch.randn(B, T, n_embd)

    # Reference: forward_chunk on full length T.
    state0 = nmm.init_state(B, torch.device("cpu"))
    y_chunk, _ = nmm.forward_chunk(x_full, state0, doc_boundaries=None)
    y_ref = y_chunk[:, -1, :]

    # Stepped path: forward_chunk on prefix [0..T-2] (state now carries the
    # right conv_buf inside); then step_with_conv on token T-1.
    x_prefix = x_full[:, :-1, :]
    state = nmm.init_state(B, torch.device("cpu"))
    _, state_after_prefix = nmm.forward_chunk(
        x_prefix, state, doc_boundaries=None,
    )
    x_t = x_full[:, -1, :]
    y_step, _ = nmm.step_with_conv(x_t, state_after_prefix)

    assert torch.allclose(y_step, y_ref, atol=1e-4), (
        f"step_with_conv vs forward_chunk parity broken: max diff = "
        f"{(y_step - y_ref).abs().max().item():.3e}"
    )


def test_step_with_conv_state_matches_forward_chunk_state_at_token_T():
    """Output state parity. After the Tth step, (M, S) should match
    forward_chunk on the full T tokens."""
    torch.manual_seed(0)
    n_embd, T = 8, 6
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    x_full = torch.randn(B, T, n_embd)

    state0 = nmm.init_state(B, torch.device("cpu"))
    _, (M_chunk, S_chunk, _) = nmm.forward_chunk(
        x_full, state0, doc_boundaries=None,
    )

    x_prefix = x_full[:, :-1, :]
    state = nmm.init_state(B, torch.device("cpu"))
    _, state_after_prefix = nmm.forward_chunk(
        x_prefix, state, doc_boundaries=None,
    )
    x_t = x_full[:, -1, :]
    _, (M_step, S_step, _) = nmm.step_with_conv(x_t, state_after_prefix)

    for key in M_chunk:
        assert torch.allclose(M_step[key], M_chunk[key], atol=1e-4), (
            f"M[{key}] mismatch: max diff = "
            f"{(M_step[key] - M_chunk[key]).abs().max().item():.3e}"
        )
        assert torch.allclose(S_step[key], S_chunk[key], atol=1e-4)


def test_step_with_conv_multi_step_parity():
    """Process T tokens via T calls to step_with_conv (carrying state
    forward); outputs must match forward_chunk's per-token y at every
    position."""
    torch.manual_seed(0)
    n_embd, T = 8, 5
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    x = torch.randn(B, T, n_embd)

    state0 = nmm.init_state(B, torch.device("cpu"))
    y_chunk, _ = nmm.forward_chunk(x, state0, doc_boundaries=None)

    state = nmm.init_state(B, torch.device("cpu"))
    y_steps = []
    for t in range(T):
        y_t, state = nmm.step_with_conv(x[:, t, :], state)
        y_steps.append(y_t)
    y_step_stack = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_step_stack, y_chunk, atol=1e-4), (
        f"per-step y parity broken: max diff = "
        f"{(y_step_stack - y_chunk).abs().max().item():.3e}"
    )


def test_step_with_conv_buf_rolls_forward():
    """After a step_with_conv call, the conv_buf inside the new state
    should have shifted left by 1 and appended the just-projected linear
    token at the tail."""
    torch.manual_seed(0)
    n_embd = 8
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    state = nmm.init_state(2, torch.device("cpu"))
    # Seed conv_buf with recognizable values so we can spot the shift.
    M, S, conv_buf = state
    for k in ("q", "k", "v"):
        conv_buf[k] = torch.arange(2 * 3 * n_embd, dtype=torch.float32).view(
            2, 3, n_embd
        ).clone()
    state = (M, S, conv_buf)

    x_t = torch.randn(2, n_embd)
    _, new_state = nmm.step_with_conv(x_t, state)
    _, _, new_buf = new_state

    # The first (k-2 = 2) positions of new_buf should equal positions
    # [1, 2] of the old buf (left-shifted).
    for kk in ("q", "k", "v"):
        assert torch.allclose(new_buf[kk][:, :-1, :], conv_buf[kk][:, 1:, :]), (
            f"conv_buf[{kk}] did not shift left correctly"
        )
