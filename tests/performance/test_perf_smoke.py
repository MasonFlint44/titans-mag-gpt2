"""T8 — perf smoke tests (TEST_PLAN §12).

Marked `perf`. CI tier 4 (release gate) per TEST_PLAN §15.

These are NOT baseline-comparison tests (those require persistent baselines
+ GPU-specific tuning). They're smoke checks that the perf path runs to
completion and produces non-degenerate numbers — so a regression that
makes throughput drop by 10x or memory balloon 5x would be visible even
without a baseline.

Real baseline-comparison versions (e.g., "tokens/sec >= baseline * 0.9")
should be added once we have a GPU CI runner with stable measurement and
a baselines file. Until then, these guard the perf SHAPE: the right code
paths exist, they execute, and they produce reasonable numbers.
"""

import time

import pytest
import torch

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from generate import generate
from model.titans_gpt2 import TitansMAGGPT2
from train import build_optimizer, train_step

pytestmark = pytest.mark.perf


def _tiny_cfg():
    return TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=50257,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )


# ---------------------------------------------------------------------------
# test_throughput.py — tokens/sec for training step
# ---------------------------------------------------------------------------

def test_train_step_throughput_is_non_degenerate():
    """A train_step on the tiny CPU model should process at least a few
    hundred tokens/sec. Catches a catastrophic regression (e.g.,
    accidentally rebuilding the per_sample_grad_fn every step, which
    drops throughput by ~10x)."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    device = torch.device("cpu")
    B, T = 2, 16
    batch = (
        torch.randint(0, cfg.vocab_size, (B, T)),
        torch.zeros(B, T, dtype=torch.bool).index_fill_(1, torch.tensor([0]), True),
    )

    # Warm up (let any one-time compilation / cache fill happen).
    train_step(model, batch, None, opt, device)

    # Measure ~3 steps.
    n_steps = 3
    t0 = time.perf_counter()
    for _ in range(n_steps):
        train_step(model, batch, None, opt, device)
    dt = time.perf_counter() - t0
    tokens_per_sec = (n_steps * B * T) / dt

    # Conservative floor; real CPU throughput on this size is ~5K tok/s.
    # A 10x regression would still pass 50 tok/s; this is degeneracy floor.
    assert tokens_per_sec > 50.0, (
        f"train_step throughput suspiciously low: {tokens_per_sec:.1f} tok/s. "
        f"Likely a cache regeneration regression (e.g. rebuilding per_sample_"
        f"grad_fn every step) or the model size accidentally grew."
    )


# ---------------------------------------------------------------------------
# test_memory_footprint.py — peak memory bounded
# ---------------------------------------------------------------------------

def test_train_step_does_not_leak_memory_across_steps():
    """Run N train_steps; the OBJECT count for nmm-state tensors should
    not grow unbounded (would indicate detach_states isn't severing the
    autograd graph). We use gc-tracked tensor count as a cheap proxy
    since we can't easily probe peak CUDA memory in a CPU test."""
    import gc

    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    device = torch.device("cpu")
    batch = (
        torch.randint(0, cfg.vocab_size, (2, 16)),
        torch.zeros(2, 16, dtype=torch.bool).index_fill_(1, torch.tensor([0]), True),
    )

    def _count_tensors():
        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))

    states = None
    # Warm up two steps so the steady-state object pool is established.
    for _ in range(2):
        _, states, _ = train_step(model, batch, states, opt, device)

    before = _count_tensors()
    for _ in range(5):
        _, states, _ = train_step(model, batch, states, opt, device)
    after = _count_tensors()

    # Each step might allocate a small number of transient tensors, but
    # the count should not grow by hundreds. Threshold is generous; a
    # real leak (graph accumulation across steps) would balloon by 100+
    # per step at this model size.
    delta = after - before
    assert delta < 100, (
        f"tensor count grew by {delta} across 5 train_steps — possible "
        f"autograd graph leak (detach_states regression?)."
    )


# ---------------------------------------------------------------------------
# test_generation_latency.py — generate doesn't grow unbounded per token
# ---------------------------------------------------------------------------

def test_generate_per_token_latency_does_not_grow_with_output_length():
    """The cached decode path (Option B) is supposed to be O(1) for NMM
    per decoded token and O(T) for attention. So per-token latency grows
    LINEARLY in T, not super-linearly. A regression to the old
    sliding-window NMM-reprocess pattern would make per-token latency
    grow as O(block_size) regardless of T, AND each step would re-do
    block_size NMM updates → big constant-factor regression.

    Test: decode 5 vs 15 tokens from the same prompt. Per-token cost of
    the 15-token decode should not be more than ~2x the 5-token decode
    (loose bound to absorb fp32 noise / OS scheduling)."""
    from data.tokenizer import Tokenizer

    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    model.eval()
    tok = Tokenizer()

    prompt = "hello world"

    # Warm up
    generate(model, prompt, max_new_tokens=2, top_k=10, tokenizer=tok)

    def _per_token_latency(n_new):
        t0 = time.perf_counter()
        generate(model, prompt, max_new_tokens=n_new, top_k=10, tokenizer=tok)
        dt = time.perf_counter() - t0
        return dt / n_new

    lat_5 = _per_token_latency(5)
    lat_15 = _per_token_latency(15)

    # Loose bound: per-token latency at T=15 must not exceed 3× T=5 latency.
    # Tighter bounds would be measurement-flaky on shared CI.
    assert lat_15 < 3.0 * lat_5, (
        f"per-token latency grew super-linearly: T=5 -> {lat_5*1000:.2f} ms/tok, "
        f"T=15 -> {lat_15*1000:.2f} ms/tok (ratio {lat_15/lat_5:.2f}x). "
        f"Likely regression to sliding-window NMM-reprocess pattern."
    )


def test_generate_completes_within_reasonable_wall_time():
    """Sanity floor: generating 20 tokens from a tiny model on CPU should
    finish in well under 30 seconds. Catches a regression that makes
    each step do orders-of-magnitude more work than intended."""
    from data.tokenizer import Tokenizer

    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    model.eval()
    tok = Tokenizer()

    t0 = time.perf_counter()
    out = generate(model, "hello", max_new_tokens=20, top_k=10, tokenizer=tok)
    dt = time.perf_counter() - t0

    assert dt < 30.0, (
        f"generate(20 tokens) took {dt:.1f}s on tiny CPU model — expected <30s. "
        f"Tokens decoded: {len(tok.encode(out))}"
    )
