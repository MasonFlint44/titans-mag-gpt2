"""Phase 7.1 — NMM step_with_conv and conv buffer helpers.

The key parity invariant: processing T tokens via forward_chunk should
produce the same y at position T as processing the first T-1 tokens
via forward_chunk, then the Tth token via step_with_conv with a
correctly-seeded conv buffer.

This is what makes the Option-B decode path correct: each decoded token
gets exactly one NMM update with the full conv context, matching what
forward_chunk would compute at that position.
"""

import pytest
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


def _zero_conv_buffer(nmm, B, device):
    """Test helper: construct the zero conv buffer that step_with_conv on
    the very first token reduces to (same compute as the legacy step()).
    Production code never needs this — every decode path uses
    init_conv_buffer_from_prompt — but it's the cleanest way to express
    'fresh state, no warm-up' parity tests.
    """
    k = nmm.k_proj.conv.kernel_size
    zeros = torch.zeros(B, k - 1, nmm.n_embd, device=device)
    return {"q": zeros.clone(), "k": zeros.clone(), "v": zeros.clone()}


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def test_init_conv_buffer_from_prompt_shape():
    nmm = _nmm(n_embd=8, kernel_size=4)
    x = torch.randn(2, 16, 8)  # T >= k-1 = 3
    buf = nmm.init_conv_buffer_from_prompt(x)
    assert set(buf.keys()) == {"q", "k", "v"}
    for v in buf.values():
        assert v.shape == (2, 3, 8)  # [B, k-1, d]


def test_init_conv_buffer_pads_when_prompt_shorter_than_k_minus_1():
    nmm = _nmm(n_embd=8, kernel_size=4)
    x = torch.randn(2, 1, 8)  # T=1 < k-1=3
    buf = nmm.init_conv_buffer_from_prompt(x)
    # First two positions of the buffer are zero-padded.
    for key in ("q", "k", "v"):
        assert torch.all(buf[key][:, :2, :] == 0.0)


# ---------------------------------------------------------------------------
# step_with_conv parity against forward_chunk
# ---------------------------------------------------------------------------

def test_step_with_conv_matches_forward_chunk_at_token_T():
    """The KEY invariant. Process [0..T-1] via forward_chunk + a final
    step_with_conv on token T; verify y_T equals what forward_chunk on
    [0..T] would produce at position T."""
    torch.manual_seed(0)
    n_embd, T = 8, 6
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    state0 = nmm.init_state(B, torch.device("cpu"))
    x_full = torch.randn(B, T, n_embd)

    # Reference: forward_chunk on full length T.
    y_chunk, _ = nmm.forward_chunk(x_full, state0, doc_boundaries=None)
    y_ref = y_chunk[:, -1, :]

    # Stepped path: forward_chunk on prefix [0..T-2]; then step_with_conv on token T-1.
    x_prefix = x_full[:, :-1, :]
    state_after_prefix = nmm.init_state(B, torch.device("cpu"))
    _, state_after_prefix = nmm.forward_chunk(
        x_prefix, state_after_prefix, doc_boundaries=None
    )
    conv_buf = nmm.init_conv_buffer_from_prompt(x_prefix)
    x_t = x_full[:, -1, :]  # [B, d]

    y_step, _, _ = nmm.step_with_conv(x_t, state_after_prefix, conv_buf)

    assert torch.allclose(y_step, y_ref, atol=1e-4), (
        f"step_with_conv vs forward_chunk parity broken: max diff = "
        f"{(y_step - y_ref).abs().max().item():.3e}"
    )


def test_step_with_conv_state_matches_forward_chunk_state_at_token_T():
    """Same as above but for the OUTPUT STATE, not just y."""
    torch.manual_seed(0)
    n_embd, T = 8, 6
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    state0 = nmm.init_state(B, torch.device("cpu"))
    x_full = torch.randn(B, T, n_embd)

    _, (M_chunk, S_chunk) = nmm.forward_chunk(x_full, state0, doc_boundaries=None)

    x_prefix = x_full[:, :-1, :]
    state_after_prefix = nmm.init_state(B, torch.device("cpu"))
    _, state_after_prefix = nmm.forward_chunk(
        x_prefix, state_after_prefix, doc_boundaries=None
    )
    conv_buf = nmm.init_conv_buffer_from_prompt(x_prefix)
    x_t = x_full[:, -1, :]
    _, (M_step, S_step), _ = nmm.step_with_conv(x_t, state_after_prefix, conv_buf)

    for key in M_chunk:
        assert torch.allclose(M_step[key], M_chunk[key], atol=1e-4)
        assert torch.allclose(S_step[key], S_chunk[key], atol=1e-4)


def test_step_with_conv_buffer_evolves_correctly():
    """After step_with_conv, the new buffer must equal what
    init_conv_buffer_from_prompt would produce for the extended prompt."""
    torch.manual_seed(0)
    n_embd, T = 8, 6
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    state0 = nmm.init_state(1, torch.device("cpu"))
    x_full = torch.randn(1, T, n_embd)

    x_prefix = x_full[:, :-1, :]
    state_after_prefix = nmm.init_state(1, torch.device("cpu"))
    _, state_after_prefix = nmm.forward_chunk(
        x_prefix, state_after_prefix, doc_boundaries=None
    )
    buf_prefix = nmm.init_conv_buffer_from_prompt(x_prefix)
    x_t = x_full[:, -1, :]
    _, _, buf_after_step = nmm.step_with_conv(x_t, state_after_prefix, buf_prefix)

    # Reference: re-init buffer on the FULL prompt (prefix + Tth token).
    buf_full = nmm.init_conv_buffer_from_prompt(x_full)
    for key in ("q", "k", "v"):
        assert torch.allclose(buf_after_step[key], buf_full[key], atol=1e-6), (
            f"buffer key={key} drift: max diff = "
            f"{(buf_after_step[key] - buf_full[key]).abs().max().item():.3e}"
        )


def test_step_with_conv_multi_step_parity():
    """Process T tokens via T calls to step_with_conv; outputs must match
    forward_chunk's per-token y for every position."""
    torch.manual_seed(0)
    n_embd, T = 8, 5
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    B = 1
    x = torch.randn(B, T, n_embd)

    # Reference
    state0 = nmm.init_state(B, torch.device("cpu"))
    y_chunk, _ = nmm.forward_chunk(x, state0, doc_boundaries=None)  # [B, T, d]

    # Stepped
    state = nmm.init_state(B, torch.device("cpu"))
    conv_buf = _zero_conv_buffer(nmm, B, torch.device("cpu"))
    y_steps = []
    for t in range(T):
        y_t, state, conv_buf = nmm.step_with_conv(x[:, t, :], state, conv_buf)
        y_steps.append(y_t)
    y_step_stack = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_step_stack, y_chunk, atol=1e-4), (
        f"per-step y parity broken: max diff per t = "
        f"{(y_step_stack - y_chunk).abs().max(dim=-1).values.max(dim=0).values}"
    )


# ---------------------------------------------------------------------------
# step_with_conv with zero buffer reduces to step()
# ---------------------------------------------------------------------------

def test_step_with_zero_buffer_matches_legacy_step():
    """An empty buffer (zeros) should reproduce the original step()'s T=1
    zero-padded conv behavior on the first token."""
    torch.manual_seed(0)
    n_embd = 8
    nmm = _nmm(n_embd=n_embd, kernel_size=4)
    state = nmm.init_state(2, torch.device("cpu"))
    x_t = torch.randn(2, n_embd)

    y_legacy, state_legacy = nmm.step(x_t, state)

    state2 = nmm.init_state(2, torch.device("cpu"))
    buf = _zero_conv_buffer(nmm, 2, torch.device("cpu"))
    y_conv, state_conv, _ = nmm.step_with_conv(x_t, state2, buf)

    assert torch.allclose(y_legacy, y_conv, atol=1e-6)
    for k in state_legacy[0]:
        assert torch.allclose(state_legacy[0][k], state_conv[0][k], atol=1e-6)
        assert torch.allclose(state_legacy[1][k], state_conv[1][k], atol=1e-6)
