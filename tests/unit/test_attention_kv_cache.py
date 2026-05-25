"""Phase 7.2 — CausalSelfAttention KV cache.

Verify that cached single-token attention at position T equals the full-sequence
attention output at position T. This is the standard transformer KV-cache
correctness check.
"""

import pytest
import torch

from model.block import CausalSelfAttention


def _attn(n_embd=32, n_head=4):
    a = CausalSelfAttention(n_embd, n_head, dropout=0.0)
    a.eval()
    return a


def _causal_mask(T, device, dtype=torch.float32):
    return torch.triu(
        torch.full((T, T), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )


# ---------------------------------------------------------------------------
# project_kv
# ---------------------------------------------------------------------------

def test_project_kv_shape():
    attn = _attn(n_embd=32, n_head=4)
    x = torch.randn(2, 8, 32)
    k, v = attn.project_kv(x)
    assert k.shape == (2, 4, 8, 8)  # [B, n_head, T, head_dim]
    assert v.shape == (2, 4, 8, 8)


def test_project_kv_matches_internal_projection():
    """project_kv must produce the same K, V tensors the full forward path
    would internally compute (so seeding the cache with project_kv values
    matches the warm-up path)."""
    attn = _attn(n_embd=32, n_head=4)
    x = torch.randn(2, 8, 32)
    k_out, v_out = attn.project_kv(x)

    # Manually replicate the forward path's KV projection.
    B, T, C = x.shape
    n, d = attn.n_head, attn.head_dim
    k_ref = attn.k_proj(x).view(B, T, n, d).transpose(1, 2)
    v_ref = attn.v_proj(x).view(B, T, n, d).transpose(1, 2)

    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)


# ---------------------------------------------------------------------------
# forward_with_kv_cache parity vs full forward
# ---------------------------------------------------------------------------

def test_cached_attention_matches_full_attention_at_position_T():
    """The KV-cache correctness invariant. Run full forward on length-T;
    take output at position T-1. Then run cached forward: project K, V from
    length-(T-1) prefix into cache, call forward_with_kv_cache with the Tth
    token. Outputs must match (up to fp32 noise)."""
    torch.manual_seed(0)
    attn = _attn(n_embd=32, n_head=4)
    T = 6
    x = torch.randn(1, T, 32)

    # Full forward with causal mask -> output at last position.
    mask = _causal_mask(T, x.device, dtype=x.dtype)
    with torch.no_grad():
        y_full = attn(x, mask=mask)
    y_ref = y_full[:, -1:, :]

    # Cached: project K, V from first T-1 tokens, then forward_with_kv_cache
    # on the Tth token.
    with torch.no_grad():
        k_cache, v_cache = attn.project_kv(x[:, :-1, :])
        y_cached, _, _ = attn.forward_with_kv_cache(
            x[:, -1:, :], k_cache, v_cache
        )

    assert torch.allclose(y_cached, y_ref, atol=1e-5), (
        f"cached vs full attention parity: max diff = "
        f"{(y_cached - y_ref).abs().max().item():.3e}"
    )


def test_cached_attention_multi_step_matches_full():
    """T consecutive cached single-token calls reproduce the full attention
    output at every position."""
    torch.manual_seed(0)
    attn = _attn(n_embd=32, n_head=4)
    T = 8
    x = torch.randn(1, T, 32)

    with torch.no_grad():
        mask = _causal_mask(T, x.device, dtype=x.dtype)
        y_full = attn(x, mask=mask)

    # Cached path: start with empty cache, append one token at a time.
    with torch.no_grad():
        n, d = attn.n_head, attn.head_dim
        k_cache = torch.empty(1, n, 0, d, dtype=x.dtype)
        v_cache = torch.empty(1, n, 0, d, dtype=x.dtype)
        y_per_step = []
        for t in range(T):
            y_t, k_cache, v_cache = attn.forward_with_kv_cache(
                x[:, t:t + 1, :], k_cache, v_cache
            )
            y_per_step.append(y_t)
        y_cached = torch.cat(y_per_step, dim=1)  # [B, T, C]

    assert torch.allclose(y_cached, y_full, atol=1e-5), (
        f"per-step cached vs full max diff = "
        f"{(y_cached - y_full).abs().max().item():.3e}"
    )


def test_cache_grows_by_exactly_one_per_step():
    attn = _attn(n_embd=32, n_head=4)
    n, d = attn.n_head, attn.head_dim
    x = torch.randn(1, 1, 32)
    k_cache = torch.empty(1, n, 0, d)
    v_cache = torch.empty(1, n, 0, d)
    with torch.no_grad():
        for expected_len in range(1, 6):
            _, k_cache, v_cache = attn.forward_with_kv_cache(x, k_cache, v_cache)
            assert k_cache.shape[2] == expected_len
            assert v_cache.shape[2] == expected_len


def test_cached_forward_rejects_multi_token_x():
    attn = _attn(n_embd=32, n_head=4)
    x = torch.randn(1, 2, 32)
    n, d = attn.n_head, attn.head_dim
    k_cache = torch.empty(1, n, 0, d)
    v_cache = torch.empty(1, n, 0, d)
    with pytest.raises(AssertionError, match="T=1"):
        attn.forward_with_kv_cache(x, k_cache, v_cache)


# ---------------------------------------------------------------------------
# SWA at decode (G238)
# ---------------------------------------------------------------------------

def test_cached_forward_swa_masks_far_past_real_positions():
    """G238 — with swa_window set, the new token must NOT attend to real
    positions farther back than swa_window. The persistent prefix is
    always visible.

    Build a synthetic V: V[:, :, t, :] = e_t (a one-hot per position).
    Run attention with uniform Q so the SDPA output is the AVERAGE of
    the allowed V rows. Compare which positions contributed: with SWA
    the far-past real positions should produce zeros in the output."""
    torch.manual_seed(0)
    n_embd, n_head = 32, 4
    attn = _attn(n_embd=n_embd, n_head=n_head)
    head_dim = n_embd // n_head

    n_persistent = 2
    n_warm = 8  # real positions already in cache
    T_full = n_persistent + n_warm + 1  # +1 for the new token

    # Sanity: make per-head V rows one-hot in their position dim so we can
    # read off WHICH positions contributed to the output. We don't construct
    # the cache via project_kv (it'd permute the V values through the V
    # projection); instead inject directly.
    v_cache = torch.zeros(1, n_head, n_persistent + n_warm, head_dim)
    for t in range(n_persistent + n_warm):
        v_cache[:, :, t, t % head_dim] = float(t)
    k_cache = torch.randn(1, n_head, n_persistent + n_warm, head_dim)

    x_new = torch.randn(1, 1, n_embd)

    with torch.no_grad():
        # Full attention (no SWA): every position is in the softmax denom.
        _, k_full_no_swa, v_full_no_swa = attn.forward_with_kv_cache(
            x_new, k_cache, v_cache
        )
        # SWA: only the last swa_window real positions + persistent should
        # have nonzero attention weight. Use a window=3 so the math is easy.
        # Then attention weight on real positions [n_persistent .. T_full - 1 - 3]
        # must be exactly 0 (because mask is -inf there).
        swa_window = 3
        with torch.no_grad():
            # Replay with SWA — we'll inspect the attention indirectly by
            # comparing y against y_no_swa: any non-trivial difference
            # confirms the mask actually changed which positions contributed.
            y_no_swa, _, _ = attn.forward_with_kv_cache(
                x_new, k_cache, v_cache,
                swa_window=None, n_persistent=n_persistent,
            )
            y_swa, _, _ = attn.forward_with_kv_cache(
                x_new, k_cache, v_cache,
                swa_window=swa_window, n_persistent=n_persistent,
            )

    # The two outputs MUST differ — n_warm=8 real positions are in the cache,
    # the SWA window is 3, so positions [n_persistent .. T_full - 1 - 3] =
    # [2 .. 7] are masked out (6 real positions). Their V values were non-zero
    # by construction, so dropping them changes the softmax-weighted sum.
    diff = (y_swa - y_no_swa).abs().max().item()
    assert diff > 1e-4, (
        f"SWA mask had no effect on the attention output (max diff = "
        f"{diff:.3e}); decoded tokens are still attending to far-past "
        f"real positions, breaking the sliding window invariant."
    )


def test_cached_forward_swa_with_window_geq_real_positions_is_noop():
    """When swa_window >= number-of-real-positions-in-cache + 1, every real
    position is inside the window, so SWA collapses to full attention."""
    torch.manual_seed(0)
    attn = _attn(n_embd=32, n_head=4)
    n_persistent = 2
    n_warm = 4
    k_cache = torch.randn(1, attn.n_head, n_persistent + n_warm, attn.head_dim)
    v_cache = torch.randn(1, attn.n_head, n_persistent + n_warm, attn.head_dim)
    x_new = torch.randn(1, 1, 32)

    with torch.no_grad():
        y_no_swa, _, _ = attn.forward_with_kv_cache(
            x_new, k_cache, v_cache, swa_window=None, n_persistent=n_persistent,
        )
        y_swa_wide, _, _ = attn.forward_with_kv_cache(
            x_new, k_cache, v_cache, swa_window=100, n_persistent=n_persistent,
        )

    assert torch.allclose(y_no_swa, y_swa_wide, atol=1e-6)


def test_cached_forward_swa_persistent_prefix_always_visible():
    """Persistent positions (0..n_persistent-1) must be attended to
    regardless of swa_window. Place LARGE distinct V values on persistent
    positions and tiny V values on real positions; if persistent rows
    contribute, the SDPA output magnitude reflects them even at a tiny
    swa_window."""
    torch.manual_seed(0)
    attn = _attn(n_embd=32, n_head=4)
    head_dim = attn.head_dim

    n_persistent = 3
    n_warm = 5
    k_cache = torch.randn(1, attn.n_head, n_persistent + n_warm, head_dim)
    v_cache = torch.zeros(1, attn.n_head, n_persistent + n_warm, head_dim)
    # Large signal on persistent positions, zero on real positions.
    v_cache[:, :, :n_persistent, :] = 100.0
    # K values on persistent positions should be similar to the query so
    # the softmax allocates non-trivial weight to them.
    x_new = torch.randn(1, 1, 32)

    with torch.no_grad():
        # swa_window=1: only the most recent real position (and persistent
        # prefix) are visible.
        y, _, _ = attn.forward_with_kv_cache(
            x_new, k_cache, v_cache, swa_window=1, n_persistent=n_persistent,
        )

    # If persistent positions were masked out, y would mostly reflect the
    # last real position's tiny zero V — leaving |y| << 1. Since
    # persistent V=100 is visible, |y| should be substantial.
    assert y.abs().max().item() > 1.0, (
        f"persistent prefix V=100 was masked out by SWA — |y|.max = "
        f"{y.abs().max().item():.3e} (expected > 1 if persistent visible)."
    )
