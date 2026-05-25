"""Phase 5.3 — needle_in_haystack runs without error (untrained model;
real recall accuracy is a behaviour test that needs a trained checkpoint).

Also locks in that the harness uses Option B (cached decode), NOT the
old broken sliding-window NMM-reprocess pattern. The call-counting test
below would fail loudly if anyone reintroduced the old pattern, since
it would call `model.forward()` repeatedly inside the decode loop.
"""

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from eval import needle_in_haystack
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
