"""Phase 5.3 — needle_in_haystack runs without error (untrained model;
real recall accuracy is a behaviour test that needs a trained checkpoint).

Also locks in that the harness uses Option B (cached decode), NOT the
old broken sliding-window NMM-reprocess pattern. The call-counting test
below would fail loudly if anyone reintroduced the old pattern, since
it would call `model.forward()` repeatedly inside the decode loop.

needle_in_haystack_sweep tests at the bottom — the batched sweep
harness that aggregates per-position and per-secret recall stats.
"""

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from evaluation import needle_in_haystack, needle_in_haystack_sweep
from model.titans_gpt2 import TitansMAGGPT2


def test_needle_in_haystack_runs_to_completion():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=128, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    tok = Tokenizer()
    # Untrained model — recall is essentially random; we only verify the
    # harness runs and returns a bool.
    result = needle_in_haystack(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="the quick brown fox jumps over the lazy dog. " * 4,
        block_size=64,
    )
    assert isinstance(result, bool)


def test_needle_in_haystack_uses_cached_decode_path():
    """Regression guard for the Phase 7 architecture fix in needle_in_haystack.

    The decode loop must use `model.forward_step`, NOT `model.forward`, on
    the per-token path. A reintroduction of the old sliding-window
    NMM-reprocess pattern would re-feed the trailing block_size window
    through forward() at every decoded token, compounding NMM state
    updates ~block_size× per sampled token.

    Concretely:
      - model.forward() should fire ONLY for prompt-prefix chunks when
        the prompt is longer than block_size (zero times for short prompts
        that fit in a single prepare_decode).
      - model.prepare_decode() should fire exactly once (the warm-up).
      - model.forward_step() may fire up to (max_new - 1) times during the
        per-token loop.
    """
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=128, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    tok = Tokenizer()

    fwd_calls = {"n": 0}
    prep_calls = {"n": 0}
    step_calls = {"n": 0}
    orig_fwd = model.forward
    orig_prep = model.prepare_decode
    orig_step = model.forward_step

    def counting_fwd(*a, **kw):
        fwd_calls["n"] += 1
        return orig_fwd(*a, **kw)

    def counting_prep(*a, **kw):
        prep_calls["n"] += 1
        return orig_prep(*a, **kw)

    def counting_step(*a, **kw):
        step_calls["n"] += 1
        return orig_step(*a, **kw)

    model.forward = counting_fwd
    model.prepare_decode = counting_prep
    model.forward_step = counting_step

    _ = needle_in_haystack(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="the quick brown fox jumps over the lazy dog. " * 4,
        block_size=64,
    )

    # Short haystack: no chunked-warm-up needed -> 0 forward() calls.
    assert fwd_calls["n"] == 0, (
        f"needle_in_haystack should NOT call model.forward() for prompts "
        f"<= block_size — got {fwd_calls['n']} calls, suggesting the broken "
        f"sliding-window NMM-reprocess pattern was reintroduced."
    )
    assert prep_calls["n"] == 1, (
        f"prepare_decode should fire exactly once; got {prep_calls['n']}"
    )
    assert step_calls["n"] >= 1, (
        f"forward_step should fire at least once during decode; got "
        f"{step_calls['n']}"
    )


def test_needle_in_haystack_long_prompt_uses_cached_decode_path():
    """long-prompt branch of needle_in_haystack must also use the
    cached pipeline (chunked-warm-up via forward() for the prefix +
    prepare_decode on the tail, with NO per-token forward() inside the
    decode loop).

    The short-prompt regression test (above) does not exercise the long-
    prompt branch and so wouldn't catch a sliding-window NMM-reprocess
    bug accidentally reintroduced into the > block_size code path.

    Expected call pattern for prompt_len > block_size:
      - model.forward()        : ceil((prompt_len - block_size) / block_size)
                                 — one call per warm-up chunk before the tail
      - model.prepare_decode() : 1                   (warm-up on the tail)
      - model.forward_step()   : 0                   (max_new = 1 at the boundary;
                                                       generate / needle skip
                                                       forward_step on the last iter)
    """
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=64, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    tok = Tokenizer()

    # Force the long-prompt branch: this haystack is hundreds of tokens; with
    # block_size=64 the chunked-warm-up loop will fire several times.
    long_haystack = (
        "the quick brown fox jumps over the lazy dog. " * 30
    )
    # Confirm we're actually in the long-prompt regime once the needle and
    # probe are spliced in (matches needle_in_haystack's construction).
    full_built = (
        long_haystack[: len(long_haystack) // 2] + " The secret password is X. "
        + long_haystack[len(long_haystack) // 2 :] + " The secret password is"
    )
    full_len = len(tok.encode(full_built))
    assert full_len > cfg.block_size, (
        f"test misconfigured: full prompt is {full_len} tokens but "
        f"block_size is {cfg.block_size}"
    )

    fwd_calls = {"n": 0}
    prep_calls = {"n": 0}
    step_calls = {"n": 0}
    orig_fwd = model.forward
    orig_prep = model.prepare_decode
    orig_step = model.forward_step

    def counting_fwd(*a, **kw):
        fwd_calls["n"] += 1
        return orig_fwd(*a, **kw)

    def counting_prep(*a, **kw):
        prep_calls["n"] += 1
        return orig_prep(*a, **kw)

    def counting_step(*a, **kw):
        step_calls["n"] += 1
        return orig_step(*a, **kw)

    model.forward = counting_fwd
    model.prepare_decode = counting_prep
    model.forward_step = counting_step

    _ = needle_in_haystack(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack=long_haystack,
        block_size=cfg.block_size,
    )

    # Warm-up chunks: at least one (the test was set up so the prompt is
    # long enough to trigger the long-prompt branch). Cap at a sane upper
    # bound to catch "every token re-feeds the window" regressions
    # (which would balloon this count past 10 trivially).
    assert fwd_calls["n"] >= 1, (
        f"long-prompt branch should call model.forward() at least once for "
        f"the chunked warm-up; got {fwd_calls['n']}"
    )
    assert fwd_calls["n"] <= (full_len // cfg.block_size) + 1, (
        f"model.forward() fired {fwd_calls['n']} times for a {full_len}-token "
        f"prompt at block_size={cfg.block_size}: more than one per warm-up "
        f"chunk, suggesting the sliding-window pattern leaked into the long-"
        f"prompt branch."
    )
    assert prep_calls["n"] == 1, (
        f"prepare_decode should fire exactly once; got {prep_calls['n']}"
    )
    assert step_calls["n"] == 0, (
        f"long-prompt branch caps decode at 1 token (sampled from "
        f"last_logits, no forward_step needed); got {step_calls['n']} "
        f"forward_step calls — likely the boundary cap was lifted."
    )


# ---------------------------------------------------------------------------
# needle_in_haystack_sweep
# ---------------------------------------------------------------------------

def _sweep_model():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=128, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    return cfg, TitansMAGGPT2(cfg)


def test_sweep_default_grid_returns_expected_structure():
    """default sweep with n_positions=9, n_secrets=5 should run 45
    pairs and return a result dict with the documented keys."""
    cfg, model = _sweep_model()
    tok = Tokenizer()
    result = needle_in_haystack_sweep(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="the quick brown fox jumps over the lazy dog. " * 3,
        block_size=64,
    )
    # Documented keys.
    expected_keys = {
        "recall", "n_pairs", "n_matched", "per_position",
        "per_secret", "details", "insert_fractions", "secrets",
    }
    assert set(result.keys()) == expected_keys
    # Default grid: 9 positions x 5 secrets = 45 pairs.
    assert result["n_pairs"] == 45
    assert len(result["insert_fractions"]) == 9
    assert len(result["secrets"]) == 5
    assert len(result["details"]) == 45
    # Recall is a float in [0, 1].
    assert isinstance(result["recall"], float)
    assert 0.0 <= result["recall"] <= 1.0
    # n_matched and recall are consistent.
    assert result["n_matched"] == sum(1 for _, _, m in result["details"] if m)
    assert abs(result["recall"] - result["n_matched"] / result["n_pairs"]) < 1e-9


def test_sweep_per_position_per_secret_aggregation_correct():
    """Per-position recall must equal mean of details with that position;
    per-secret recall must equal mean of details with that secret."""
    cfg, model = _sweep_model()
    tok = Tokenizer()
    result = needle_in_haystack_sweep(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="hello world. " * 5,
        n_positions=3,
        n_secrets=2,
        block_size=64,
    )
    assert result["n_pairs"] == 6

    # Per-position aggregation
    for f in result["insert_fractions"]:
        matches = [m for ff, _, m in result["details"] if ff == f]
        expected = sum(matches) / len(matches)
        assert abs(result["per_position"][f] - expected) < 1e-9

    # Per-secret aggregation
    for s in result["secrets"]:
        matches = [m for _, ss, m in result["details"] if ss == s]
        expected = sum(matches) / len(matches)
        assert abs(result["per_secret"][s] - expected) < 1e-9


def test_sweep_explicit_secrets_and_positions_override_defaults():
    """When the caller supplies `secrets` and `insert_fractions`, those are
    used directly — `n_secrets` / `n_positions` are ignored."""
    cfg, model = _sweep_model()
    tok = Tokenizer()
    result = needle_in_haystack_sweep(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="hello world. " * 5,
        insert_fractions=[0.25, 0.75],
        secrets=["FOO1", "BAR2", "BAZ3"],
        n_positions=99,  # ignored
        n_secrets=99,    # ignored
        block_size=64,
    )
    assert result["insert_fractions"] == [0.25, 0.75]
    assert result["secrets"] == ["FOO1", "BAR2", "BAZ3"]
    assert result["n_pairs"] == 6


def test_sweep_seed_determinism():
    """Same seed -> same default secrets across calls."""
    cfg, model = _sweep_model()
    tok = Tokenizer()
    r1 = needle_in_haystack_sweep(
        model=model, tokenizer=tok, device=torch.device("cpu"),
        haystack="hello. " * 5, n_positions=2, n_secrets=3,
        seed=42, block_size=64,
    )
    r2 = needle_in_haystack_sweep(
        model=model, tokenizer=tok, device=torch.device("cpu"),
        haystack="hello. " * 5, n_positions=2, n_secrets=3,
        seed=42, block_size=64,
    )
    assert r1["secrets"] == r2["secrets"]


def test_sweep_different_seed_produces_different_secrets():
    cfg, model = _sweep_model()
    tok = Tokenizer()
    r1 = needle_in_haystack_sweep(
        model=model, tokenizer=tok, device=torch.device("cpu"),
        haystack="hello. " * 5, n_positions=2, n_secrets=3,
        seed=0, block_size=64,
    )
    r2 = needle_in_haystack_sweep(
        model=model, tokenizer=tok, device=torch.device("cpu"),
        haystack="hello. " * 5, n_positions=2, n_secrets=3,
        seed=1, block_size=64,
    )
    assert r1["secrets"] != r2["secrets"]


def test_sweep_one_secret_one_position_runs_one_pair():
    """Degenerate grid — n_positions=1, n_secrets=1 — should run exactly 1
    pair; recall is either 0.0 or 1.0."""
    cfg, model = _sweep_model()
    tok = Tokenizer()
    result = needle_in_haystack_sweep(
        model=model, tokenizer=tok, device=torch.device("cpu"),
        haystack="hello. " * 5, n_positions=1, n_secrets=1,
        block_size=64,
    )
    assert result["n_pairs"] == 1
    assert result["recall"] in (0.0, 1.0)
