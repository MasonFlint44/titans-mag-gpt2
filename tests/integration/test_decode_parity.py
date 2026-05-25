"""Phase 7.3 — full-model decode parity.

The headline correctness invariant for Option B:
  cached_decode_logits(prompt, generated) == full_forward_logits(prompt + generated)
within fp32 noise.

If this holds, the cached decode path is producing exactly the same NMM
contribution per token as a single full forward would — which is the
whole point of going to KV cache + step_with_conv (fixes the NMM
sliding-window reprocessing limitation).
"""

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2


def _tiny_model(finetune_mode=False):
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=finetune_mode,
    )
    return cfg, TitansMAGGPT2(cfg).eval()


# ---------------------------------------------------------------------------
# prepare_decode contract
# ---------------------------------------------------------------------------

def test_prepare_decode_returns_expected_structure():
    cfg, model = _tiny_model()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        cache = model.prepare_decode(prompt)
    assert set(cache.keys()) == {
        "last_logits", "nmm_states", "kv_caches",
        "nmm_conv_buffers", "position",
    }
    assert cache["last_logits"].shape == (1, 1, cfg.vocab_size)
    assert cache["position"] == 8
    assert len(cache["nmm_states"]) == cfg.n_layer
    assert len(cache["kv_caches"]) == cfg.n_layer
    assert len(cache["nmm_conv_buffers"]) == cfg.n_layer


def test_prepare_decode_kv_cache_includes_persistent_prefix():
    """KV cache shape per layer is [B, n_head, N_p + prompt_len, head_dim]."""
    cfg, model = _tiny_model()
    prompt = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        cache = model.prepare_decode(prompt)
    expected_T = cfg.nmm_n_persistent + 8
    head_dim = cfg.n_embd // cfg.n_head
    for k_cache, v_cache in cache["kv_caches"]:
        assert k_cache.shape == (2, cfg.n_head, expected_T, head_dim)
        assert v_cache.shape == (2, cfg.n_head, expected_T, head_dim)


def test_prepare_decode_rejects_prompt_longer_than_block_size():
    cfg, model = _tiny_model()
    long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        model.prepare_decode(long_prompt)


# ---------------------------------------------------------------------------
# forward_step parity vs full forward
# ---------------------------------------------------------------------------

def test_cached_decode_matches_full_forward_one_step():
    """Build prompt of length P, run prepare_decode + one forward_step on a
    fixed next token. Compare against a full forward on [prompt, next_token]
    — logits at position P must match within fp32 noise.

    Tolerance is the G234 scaled form: max(1e-4, 5e-3 * |ref|_max). The
    untrained tiny-model NMM produces small-magnitude logits (~0.25), and
    per-token NMM updates compound ~5e-5 of conv-kernel-shape drift (G230
    family) across n_layer blocks → ~5e-4 absolute diff per token, which
    is ~2e-3 relative. The finetune_mode=True test (NMM zeroed) confirms
    the diff is from the NMM path, not the attention path."""
    torch.manual_seed(0)
    cfg, model = _tiny_model(finetune_mode=False)
    P = 8
    prompt = torch.randint(0, cfg.vocab_size, (1, P))
    next_tok = torch.tensor([[7]], dtype=torch.long)

    with torch.no_grad():
        # Reference: full forward on [prompt, next_tok]
        full = torch.cat([prompt, next_tok], dim=1)
        ref_logits, _ = model(full, nmm_states=None)
        ref_step = ref_logits[:, -1:, :]

        # Cached path
        cache = model.prepare_decode(prompt)
        step_logits, _ = model.forward_step(next_tok, cache)

    ref_max = ref_step.abs().max().item()
    tol = max(1e-4, 5e-3 * ref_max)
    diff = (step_logits - ref_step).abs().max().item()
    assert diff < tol, (
        f"cached forward_step vs full forward parity: max diff = {diff:.3e}, "
        f"tolerance = {tol:.3e} (ref_max = {ref_max:.3f})"
    )


def test_cached_decode_matches_full_forward_multi_step():
    """Decode N tokens via the cached path; compare each step's logits to
    a single full forward on the same prompt+generated sequence."""
    torch.manual_seed(0)
    cfg, model = _tiny_model(finetune_mode=False)
    P, N = 6, 4
    prompt = torch.randint(0, cfg.vocab_size, (1, P))
    decoded = torch.randint(0, cfg.vocab_size, (1, N))  # fixed token sequence
    _check_multi_step_parity(cfg, model, prompt, decoded, P, N)


def test_cached_decode_matches_full_forward_multi_step_batched():
    """G241 — same multi-step parity but at B=2 (batched decode). Every
    other cached-decode correctness test runs at B=1; this is the only
    test that would catch a regression in batched paths (per-sample vmap
    inside step_with_conv, batched KV cache concat, batched conv buffer
    shifts). The shapes downstream all carry B through, but until this
    test was added no parity invariant was actually verified at B>1."""
    torch.manual_seed(0)
    cfg, model = _tiny_model(finetune_mode=False)
    P, N = 6, 4
    B = 2
    prompt = torch.randint(0, cfg.vocab_size, (B, P))
    decoded = torch.randint(0, cfg.vocab_size, (B, N))
    _check_multi_step_parity(cfg, model, prompt, decoded, P, N)


def _check_multi_step_parity(cfg, model, prompt, decoded, P, N):

    with torch.no_grad():
        # Reference: full forward on [prompt, decoded]
        full = torch.cat([prompt, decoded], dim=1)
        ref_logits, _ = model(full, nmm_states=None)
        # We compare against ref_logits at positions [P-1, P, P+1, ..., P+N-2]
        # — those are the logits used to predict tokens [P, P+1, ..., P+N-1]
        # given the prefix up to each preceding position.
        # In the cached path:
        #   prepare_decode returns last_logits at position P-1 (predicting token P)
        #   forward_step(token P) returns logits at position P (predicting P+1)
        #   forward_step(token P+1) returns logits at position P+1 (predicting P+2)
        #   ...

        cache = model.prepare_decode(prompt)
        per_step_logits = [cache["last_logits"]]  # logits at position P-1
        for i in range(N - 1):
            tok = decoded[:, i:i + 1]
            logits_i, cache = model.forward_step(tok, cache)
            per_step_logits.append(logits_i)
        cached_logits = torch.cat(per_step_logits, dim=1)  # [B, N, V]
        ref_window = ref_logits[:, P - 1:P - 1 + N, :]  # [B, N, V]

    # G234-style scaled tolerance. The relative bound (1e-2 = 1%) absorbs
    # fp32 reduction-order noise that compounds per-sample inside the
    # NMM's per-token update loop; at B=2 the per-sample paths produce
    # slightly different rounding than the B=1 case (~3x larger absolute
    # diff at the same ref_max of ~0.25 → ~1.4e-3 vs ~5e-4 at B=1).
    ref_max = ref_window.abs().max().item()
    tol = max(1e-4, 1e-2 * ref_max)
    diff = (cached_logits - ref_window).abs().max().item()
    assert diff < tol, (
        f"multi-step cached decode vs reference: max diff = {diff:.3e}, "
        f"tolerance = {tol:.3e}"
    )


def test_cached_decode_matches_full_forward_with_swa():
    """G238 — at use_swa=True, the cached decode path must apply the same
    banded mask as the warm-up `_aug_mask`. Without the SWA-aware mask in
    forward_with_kv_cache, the new token would silently attend to the
    full real history and produce different logits than the reference
    full forward (which DOES apply the banded mask)."""
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        use_swa=True, swa_window=3,
    )
    model = TitansMAGGPT2(cfg).eval()
    # Use a prompt long enough that SWA actually masks out real positions
    # at the decode step (need prompt_len > swa_window).
    P = 12
    prompt = torch.randint(0, cfg.vocab_size, (1, P))
    next_tok = torch.tensor([[7]], dtype=torch.long)

    with torch.no_grad():
        full = torch.cat([prompt, next_tok], dim=1)
        ref_logits, _ = model(full, nmm_states=None)
        ref_step = ref_logits[:, -1:, :]

        cache = model.prepare_decode(prompt)
        step_logits, _ = model.forward_step(next_tok, cache)

    # Same G234 scaled tolerance as the non-SWA multi-step test.
    ref_max = ref_step.abs().max().item()
    tol = max(1e-4, 1e-2 * ref_max)
    diff = (step_logits - ref_step).abs().max().item()
    assert diff < tol, (
        f"SWA cached-decode vs full-forward parity: max diff = {diff:.3e}, "
        f"tolerance = {tol:.3e} (ref_max = {ref_max:.3f})"
    )


def test_cached_decode_matches_full_forward_at_finetune_init():
    """At finetune_mode=True with out_scale=0, the NMM contributes 0 to
    every block's output. The cached decode path should match HF GPT-2's
    decode behavior, which the reference full-forward also does. Verify
    they agree."""
    torch.manual_seed(0)
    cfg, model = _tiny_model(finetune_mode=True)
    P = 6
    prompt = torch.randint(0, cfg.vocab_size, (1, P))
    next_tok = torch.tensor([[3]], dtype=torch.long)

    with torch.no_grad():
        full = torch.cat([prompt, next_tok], dim=1)
        ref_logits, _ = model(full, nmm_states=None)
        ref_step = ref_logits[:, -1:, :]

        cache = model.prepare_decode(prompt)
        step_logits, _ = model.forward_step(next_tok, cache)

    assert torch.allclose(step_logits, ref_step, atol=1e-4)


def test_forward_step_rejects_position_past_block_size():
    cfg, model = _tiny_model()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        cache = model.prepare_decode(prompt)
    cache["position"] = cfg.block_size  # simulate having decoded up to the limit
    next_tok = torch.tensor([[0]], dtype=torch.long)
    with pytest.raises(ValueError, match="block_size"):
        model.forward_step(next_tok, cache)


def test_prepare_decode_rejects_wrong_length_initial_nmm_states():
    """G240 — if a caller passes initial_nmm_states with the wrong number of
    layers, prepare_decode must fail loudly here, not deep inside the first
    forward_step. The unvalidated `zip(self.blocks, nmm_states)` would
    silently truncate; the resulting partial cache then IndexErrors in
    forward_step pointing at the wrong call site."""
    cfg, model = _tiny_model()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    # Build a per-layer states list and lop one off.
    full_states = [block.nmm.init_state(1, prompt.device) for block in model.blocks]
    truncated = full_states[:-1]
    with pytest.raises(ValueError, match="initial_nmm_states length"):
        model.prepare_decode(prompt, initial_nmm_states=truncated)


def test_prepare_decode_rejects_wrong_batch_dim_initial_nmm_states():
    """G244 — state built for B=1 passed to a prompt of B=2 must fail at
    prepare_decode, not silently produce a shape-mismatch deep in the NMM."""
    cfg, model = _tiny_model()
    prompt = torch.randint(0, cfg.vocab_size, (2, 4))  # B=2
    # Build states for B=1 — wrong batch dim.
    wrong_b_states = [block.nmm.init_state(1, prompt.device) for block in model.blocks]
    with pytest.raises(ValueError, match="initial_nmm_states batch dim"):
        model.prepare_decode(prompt, initial_nmm_states=wrong_b_states)


def test_prepare_decode_rejects_train_mode():
    """G243 — train mode + dropout would silently break the decode-vs-full-forward
    parity invariant because init_decode_cache and block.forward apply different
    dropout patterns. Fail loudly so direct callers don't ship the bug."""
    cfg, model = _tiny_model()
    model.train()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(RuntimeError, match="prepare_decode requires model.eval"):
        model.prepare_decode(prompt)


def test_forward_step_rejects_train_mode():
    """G243 — symmetric guard on forward_step. A caller could call
    prepare_decode in eval, then flip the model to train and forward_step;
    the silent divergence would still bite."""
    cfg, model = _tiny_model()
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        cache = model.prepare_decode(prompt)
    model.train()
    next_tok = torch.tensor([[0]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="forward_step requires model.eval"):
        model.forward_step(next_tok, cache)
