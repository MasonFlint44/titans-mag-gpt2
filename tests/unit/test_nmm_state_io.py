"""Tests for model.state_io — persistent NMM state across generate calls.

Covers:
- Round-trip: save state, load it, tensor values match.
- Config-mismatch rejection: state saved with config A fails to load into B.
- Missing file: FileNotFoundError so caller can decide to start fresh.
- Atomic write: no `.tmp` leftover on success.
- Format version: future versions rejected (we can't read them).
- `prepare_decode_chunked` honors `initial_nmm_states` (separate file's
  responsibility but covered here too because it's the integration point).
- `generate_with_state` returns NMM state matching `forward()`'s convention.
"""

import os
from pathlib import Path

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from model.state_io import (
    StateConfigMismatch, load_nmm_state, save_nmm_state, _fingerprint,
)


def _tiny_cfg(**overrides):
    """Smallest config that still has all the structural variety we care
    about (12 layers would be wasteful for unit tests)."""
    base = dict(
        n_layer=2, n_head=2, n_embd=8, vocab_size=16,
        block_size=32, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
    )
    base.update(overrides)
    return TitansConfig(**base)


def _build_model_and_state(cfg, B=2, T=16, device=None):
    """Build a TitansMAGGPT2 from cfg and run one forward to get a realistic
    nmm_states list (mix of dicts and tensors with non-trivial values, not
    just init-zero placeholders)."""
    if device is None:
        device = torch.device("cpu")
    model = TitansMAGGPT2(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    with torch.no_grad():
        _, nmm_states = model(idx)
    return model, nmm_states


# ---------------------------------------------------------------------------
# Round-trip + config-mismatch
# ---------------------------------------------------------------------------

def test_save_load_round_trip_preserves_state_tensors(tmp_path):
    cfg = _tiny_cfg()
    _, nmm_states = _build_model_and_state(cfg)
    path = tmp_path / "session.pt"

    save_nmm_state(path, nmm_states, cfg)
    loaded = load_nmm_state(path, cfg, device=torch.device("cpu"))

    assert len(loaded) == len(nmm_states)
    for layer_a, layer_b in zip(nmm_states, loaded):
        # Single-head NMM: (M_dict, S_dict).
        M_a, S_a, _ = layer_a
        M_b, S_b, _ = layer_b
        assert set(M_a.keys()) == set(M_b.keys())
        for k in M_a:
            assert torch.equal(M_a[k], M_b[k]), f"M[{k}] diverged across round-trip"
        # S may be a dict (momentum_order=1) or a tuple of dicts (>1); the
        # tiny config above uses the default 1.
        for k in S_a:
            assert torch.equal(S_a[k], S_b[k]), f"S[{k}] diverged"


def test_load_into_mismatched_config_raises(tmp_path):
    cfg_a = _tiny_cfg(n_embd=8)
    cfg_b = _tiny_cfg(n_embd=16)  # different state shape
    _, states_a = _build_model_and_state(cfg_a)
    path = tmp_path / "session.pt"
    save_nmm_state(path, states_a, cfg_a)
    with pytest.raises(StateConfigMismatch, match="n_embd"):
        load_nmm_state(path, cfg_b, device=torch.device("cpu"))


def test_load_into_mismatched_layer_count_raises(tmp_path):
    cfg_a = _tiny_cfg(n_layer=2)
    cfg_b = _tiny_cfg(n_layer=4)
    _, states_a = _build_model_and_state(cfg_a)
    path = tmp_path / "session.pt"
    save_nmm_state(path, states_a, cfg_a)
    with pytest.raises(StateConfigMismatch, match="n_layer"):
        load_nmm_state(path, cfg_b, device=torch.device("cpu"))


def test_load_into_mismatched_low_rank_raises(tmp_path):
    cfg_a = _tiny_cfg(nmm_low_rank=None)  # full-rank: 3 state keys
    cfg_b = _tiny_cfg(nmm_low_rank=4)     # low-rank: 6 state keys
    _, states_a = _build_model_and_state(cfg_a)
    path = tmp_path / "session.pt"
    save_nmm_state(path, states_a, cfg_a)
    with pytest.raises(StateConfigMismatch, match="nmm_low_rank"):
        load_nmm_state(path, cfg_b, device=torch.device("cpu"))


def test_fingerprint_captures_shape_affecting_fields():
    """Defensive: if someone adds a new shape-affecting NMM config field,
    they should also add it to `_fingerprint` — this test reminds them by
    pinning the current key set."""
    cfg = _tiny_cfg()
    expected = {
        "n_layer", "n_embd", "nmm_n_persistent", "nmm_expansion",
        "nmm_low_rank", "nmm_n_heads", "nmm_momentum_order",
        "nmm_layer_indices", "nmm_state_dtype",
        # conv_buf is part of state now (item 6); the conv kernel size
        # determines its shape, so a k mismatch must error loudly at load.
        "nmm_conv_kernel",
    }
    assert set(_fingerprint(cfg).keys()) == expected


# ---------------------------------------------------------------------------
# Missing file + atomic write
# ---------------------------------------------------------------------------

def test_load_missing_file_raises_FileNotFoundError(tmp_path):
    """Caller (`generate.py --nmm-state-file`) catches this to decide
    'start fresh on first call'. Must be the exact FileNotFoundError so
    that `os.path.is_file()` + try/except both work cleanly."""
    with pytest.raises(FileNotFoundError):
        load_nmm_state(tmp_path / "does_not_exist.pt",
                       _tiny_cfg(), device=torch.device("cpu"))


def test_save_leaves_no_tmp_file_on_success(tmp_path):
    """Atomic write uses a sibling `.tmp` file + rename. On success the
    tmp file must NOT remain behind (would accumulate junk over many calls)."""
    cfg = _tiny_cfg()
    _, states = _build_model_and_state(cfg)
    path = tmp_path / "session.pt"
    save_nmm_state(path, states, cfg)

    # No .tmp file should remain in the directory.
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == [], f"unexpected .tmp leftovers: {tmps}"
    assert path.is_file()


def test_save_creates_parent_dir(tmp_path):
    """Caller may pass a path under a directory that doesn't exist yet
    (e.g., a fresh session directory). Make it loud-or-clean: we choose
    clean (mkdir -p) since this matches `save_checkpoint_rotating`."""
    cfg = _tiny_cfg()
    _, states = _build_model_and_state(cfg)
    nested = tmp_path / "sessions" / "user42" / "state.pt"
    save_nmm_state(nested, states, cfg)
    assert nested.is_file()


def test_save_overwrites_existing_file(tmp_path):
    """The whole point of the rotating design — repeated saves overwrite
    the same path. Verify the second save's contents replace the first's."""
    cfg = _tiny_cfg()
    model, states_first = _build_model_and_state(cfg)
    path = tmp_path / "session.pt"

    save_nmm_state(path, states_first, cfg)

    # Run a few more forwards to evolve the state into something different,
    # save the new state, then verify the loaded state matches the new one
    # (and would NOT match the first).
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        for _ in range(3):
            _, states_first = model(idx, states_first)
    save_nmm_state(path, states_first, cfg)

    loaded = load_nmm_state(path, cfg, device=torch.device("cpu"))
    # The second save's states should be in the file now.
    M_loaded, _, _ = loaded[0]
    M_expected, _, _ = states_first[0]
    for k in M_loaded:
        assert torch.equal(M_loaded[k], M_expected[k])


# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

def test_load_rejects_future_format_version(tmp_path):
    """If a newer build wrote a state file with format_version > what we
    know, we can't decode it safely (the schema may have moved). Fail loud
    with a clear hint instead of attempting to read whatever the saved
    dict happened to contain."""
    cfg = _tiny_cfg()
    _, states = _build_model_and_state(cfg)
    path = tmp_path / "future.pt"
    # Use torch.save directly so we can fake a future version.
    torch.save({
        "format_version": 999,
        "fingerprint": _fingerprint(cfg),
        "nmm_states": states,
    }, path)
    with pytest.raises(ValueError, match="format_version"):
        load_nmm_state(path, cfg, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Integration with prepare_decode_chunked / generate_with_state
# ---------------------------------------------------------------------------

def test_prepare_decode_chunked_honors_initial_nmm_states():
    """Passing `initial_nmm_states` to prepare_decode_chunked should
    actually use those — verify by checking the cache's returned
    `nmm_states` start non-zero (would be zero S + init M from
    memory_mlp.W*.weight if state was discarded)."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg).eval()
    # Build a deliberately non-init state.
    custom_states = []
    for block in model.blocks:
        M, S, conv_buf = block.nmm.init_state(B=1, device=torch.device("cpu"))
        # Perturb M so it's distinguishable from init.
        for k in M:
            M[k] = M[k] + torch.full_like(M[k], 7.0)
        custom_states.append((M, S, conv_buf))

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    cache = model.prepare_decode_chunked(prompt, initial_nmm_states=custom_states)
    # After the prompt runs, M will have evolved (NMM updates per token), but
    # the perturbation should still influence values — they should NOT match
    # what you'd get with init state. Compare against a fresh-init run.
    cache_fresh = model.prepare_decode_chunked(prompt)

    M_custom, _, _ = cache["nmm_states"][0]
    M_fresh, _, _ = cache_fresh["nmm_states"][0]
    diffs = [(M_custom[k] - M_fresh[k]).abs().max().item() for k in M_custom]
    assert max(diffs) > 1e-4, (
        f"initial_nmm_states did not propagate through prepare_decode_chunked — "
        f"final state matches the fresh-init version (max diff {max(diffs):.2e})"
    )


def _real_vocab_cfg(**overrides):
    """Tiny model with GPT-2's real vocab — required for `generate_with_state`
    tests because `Tokenizer.encode` produces real BPE ids."""
    base = dict(
        n_layer=2, n_head=2, n_embd=8, vocab_size=50257,
        block_size=32, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
    )
    base.update(overrides)
    return TitansConfig(**base)


def test_generate_with_state_returns_state_with_correct_structure():
    """`generate_with_state` must return a 2-tuple (text, state) where state
    is the same shape as `forward()`'s output (one (M, S) per layer)."""
    from cli.generate import generate_with_state
    cfg = _real_vocab_cfg()
    model = TitansMAGGPT2(cfg).eval()
    text, state = generate_with_state(
        model, "hi", max_new_tokens=2, top_k=1, temperature=0,
    )
    assert isinstance(text, str)
    assert isinstance(state, list)
    assert len(state) == cfg.n_layer
    # Each layer (under default single-head) is (M_dict, S_dict_or_tuple).
    for layer_state in state:
        M, _S, _ = layer_state
        assert isinstance(M, dict)
        assert all(isinstance(v, torch.Tensor) for v in M.values())


def test_generate_with_state_session_continuity():
    """End-to-end smoke: thread NMM state from one generate call into the
    next. Both turns must run cleanly without error; deeper state-propagation
    proof is in test_prepare_decode_chunked_honors_initial_nmm_states."""
    from cli.generate import generate_with_state
    cfg = _real_vocab_cfg()
    torch.manual_seed(0)
    model = TitansMAGGPT2(cfg).eval()

    _, state_after_turn1 = generate_with_state(
        model, "alpha beta", max_new_tokens=4, top_k=1, temperature=0,
    )
    text_continued, _ = generate_with_state(
        model, "gamma delta",
        initial_nmm_states=state_after_turn1,
        max_new_tokens=4, top_k=1, temperature=0,
    )
    text_fresh, _ = generate_with_state(
        model, "gamma delta",
        initial_nmm_states=None,
        max_new_tokens=4, top_k=1, temperature=0,
    )
    assert isinstance(text_continued, str)
    assert isinstance(text_fresh, str)
