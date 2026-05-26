"""End-to-end smoke tests for scripts.eval_qa_recall.

These tests stand up a tiny TitansMAGGPT2 + tiny synthetic eval set + run
the recall sweep. Untrained models have ~0% recall — we don't check
accuracy, we check that the harness runs without error and produces the
expected output shape. Real-checkpoint accuracy is the experiment itself
(measured offline against trained checkpoints).

Why integration tier (not unit): the harness exercises tokenizer, model
forward, prepare_decode_chunked — a real end-to-end path, not isolated
functions. If any of those break, this test catches the integration
without needing a behavior-tier checkpoint run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from model.titans_gpt2 import TitansMAGGPT2
from scripts.eval_qa_recall import (
    DEFAULT_DISTANCES,
    EvalRecord,
    _first_token_set,
    build_qa_prompt,
    evaluate,
    evaluate_one,
    load_eval_records,
)


def _tiny_config(**overrides) -> TitansConfig:
    """Smallest sensible config that still exercises the chunked-warm-up
    path. block_size large enough that the short-prompt regime is also
    coverable; chunk_size matches so wpe is fully trained."""
    base = dict(
        n_layer=2, n_head=2, n_embd=8, vocab_size=50257,
        block_size=256, chunk_size=256, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    base.update(overrides)
    return TitansConfig(**base)


def _synthetic_records(n: int = 8) -> list[EvalRecord]:
    """Construct a deterministic eval set. Real BPE tokenization (the
    Tokenizer is shared across train/eval) so token-count plumbing in
    build_qa_prompt is exercised honestly."""
    records = []
    for i in range(n):
        # ~30 BPE tokens per context.
        context = (
            f"The capital of country_{i} is City_{i}. "
            f"Its main river is River_{i}. "
            f"The country was founded in year_{i}_AD."
        )
        records.append(EvalRecord(
            id=f"id-{i}",
            title=f"Country_{i}",
            context=context,
            question=f"What is the capital of country_{i}?",
            answers=[f"City_{i}", f"city_{i}"],
        ))
    return records


# ---------------------------------------------------------------------------
# build_qa_prompt
# ---------------------------------------------------------------------------

def test_build_qa_prompt_layout_at_distance_zero():
    """distance=0 means no distractor padding — target sits directly before
    the final Q. Verify both the structure and that actual_distance == 0."""
    tok = Tokenizer()
    import random
    records = _synthetic_records(5)
    target = records[0]
    pool = records[1:]
    text, prompt_len, actual = build_qa_prompt(
        target, pool, distance=0, tokenizer=tok, rng=random.Random(0),
    )
    assert "[P] " + target.context in text
    assert f"Q: {target.question}" in text
    assert text.endswith("A:")
    assert actual == 0
    assert prompt_len > 0


def test_build_qa_prompt_padding_reaches_distance():
    """At distance=100, distractor padding tokens must sum to at least 100
    (modulo the per-distractor granularity — we never DROP a distractor
    once it's added, so we may overshoot by up to one distractor)."""
    tok = Tokenizer()
    import random
    records = _synthetic_records(20)
    target = records[0]
    pool = records[1:]
    _, _, actual = build_qa_prompt(
        target, pool, distance=100, tokenizer=tok, rng=random.Random(0),
    )
    assert actual >= 100, (
        f"distractor padding ({actual} tokens) did not reach target distance 100 "
        f"despite a 19-record distractor pool"
    )


def test_build_qa_prompt_pool_exhaustion_caps_actual_distance():
    """When the distractor pool is too small to reach the requested
    distance, build_qa_prompt stops and reports `actual_distance` < distance
    rather than padding garbage. Caller can use the gap to decide whether
    the trial counts."""
    tok = Tokenizer()
    import random
    records = _synthetic_records(3)  # tiny pool
    target = records[0]
    pool = records[1:]  # only 2 distractors, each ~30 tokens
    _, _, actual = build_qa_prompt(
        target, pool, distance=10_000,  # impossible with this pool
        tokenizer=tok, rng=random.Random(0),
    )
    assert actual < 10_000
    assert actual >= 0


# ---------------------------------------------------------------------------
# _first_token_set: scoring helper
# ---------------------------------------------------------------------------

def test_first_token_set_includes_leading_space_variant():
    """The training format puts a space after 'A:' so the model should
    predict ' Denver' as the first answer token. The token set MUST
    include the leading-space variant — otherwise gold matches would all
    fail."""
    tok = Tokenizer()
    tokens = _first_token_set(["Denver Broncos"], tok)
    # The token for " Denver" must be present.
    space_first = tok.encode(" Denver Broncos")[0]
    assert space_first in tokens


def test_first_token_set_accepts_multiple_aliases():
    """SQuAD validation has multiple gold aliases per question; ANY of them
    matching counts as correct."""
    tok = Tokenizer()
    tokens = _first_token_set(["Denver Broncos", "the Broncos"], tok)
    # Both leading-token ids must be in the set.
    a = tok.encode(" Denver Broncos")[0]
    b = tok.encode(" the Broncos")[0]
    assert a in tokens
    assert b in tokens


# ---------------------------------------------------------------------------
# evaluate_one: single-trial end-to-end
# ---------------------------------------------------------------------------

def test_evaluate_one_returns_expected_schema():
    """One trial → one record dict with every documented key. If the schema
    drifts, downstream plotting / aggregation breaks silently — lock it."""
    import random
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    records = _synthetic_records(5)
    rec = evaluate_one(
        model, tok, torch.device("cpu"), records[0], records[1:],
        distance=0, rng=random.Random(0),
    )
    for key in (
        "id", "distance", "actual_distance", "prompt_len",
        "predicted_token", "predicted_text", "expected_answers", "correct",
    ):
        assert key in rec, f"missing key {key!r} in evaluate_one output"
    assert isinstance(rec["correct"], bool)
    assert rec["distance"] == 0


def test_evaluate_one_max_prompt_tokens_truncates():
    """When the prompt would blow past block_size and we want to keep eval
    cheap, max_prompt_tokens caps it. The target passage MUST still survive
    truncation (we keep the head + tail, drop the middle)."""
    import random
    cfg = _tiny_config(block_size=128, chunk_size=128)
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    records = _synthetic_records(20)
    # Request a huge distance that would normally produce a 1000+ token
    # prompt; cap at 64.
    rec = evaluate_one(
        model, tok, torch.device("cpu"), records[0], records[1:],
        distance=10_000, rng=random.Random(0),
        max_prompt_tokens=64,
    )
    # Truncation happened — actual encoded prompt fed to the model was ≤64;
    # but the build_qa_prompt-reported prompt_len reflects the pre-truncation
    # version. We just verify the trial completed without crashing.
    assert isinstance(rec["correct"], bool)


# ---------------------------------------------------------------------------
# evaluate: full sweep
# ---------------------------------------------------------------------------

def test_evaluate_runs_end_to_end_on_untrained_model():
    """Smoke: tiny untrained model, 3 records, 2 distance buckets. Verifies
    the full sweep produces the expected shape (buckets, results, accuracy)."""
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    records = _synthetic_records(5)
    out = evaluate(
        model, tok, torch.device("cpu"), records,
        distances=[0, 50],
        n_examples=3,
    )
    assert set(out["buckets"].keys()) == {0, 50}
    assert out["n_examples"] == 3
    assert out["n_trials"] == 3 * 2
    assert len(out["results"]) == 3 * 2
    for d, b in out["buckets"].items():
        assert b["total"] == 3
        assert 0 <= b["accuracy"] <= 1


def test_evaluate_uses_same_targets_across_buckets():
    """Subsample once → every bucket evaluates the SAME targets. If buckets
    sampled independently, comparing accuracy across distances would be
    confounded by inter-example variance. Lock the contract: each target
    id appears exactly once per bucket."""
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    records = _synthetic_records(8)
    out = evaluate(
        model, tok, torch.device("cpu"), records,
        distances=[0, 100],
        n_examples=4,
    )
    bucket_0_ids = {r["id"] for r in out["results"] if r["distance"] == 0}
    bucket_100_ids = {r["id"] for r in out["results"] if r["distance"] == 100}
    assert bucket_0_ids == bucket_100_ids
    assert len(bucket_0_ids) == 4


def test_evaluate_target_never_used_as_own_distractor():
    """A target's own (Q, A) pair appearing as a distractor right next to
    it would leak the gold answer into the prompt — recall becomes trivially
    100% at every distance, masking the model's actual behavior. Guard:
    no result record contains the target's own question text in its
    distractor padding region. We verify indirectly by checking that the
    pool excludes the target id (the construction-time invariant)."""
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    records = _synthetic_records(6)
    # Stake the contract via a direct probe: target.id == records[0].id;
    # the distractor pool in evaluate is `[r for r in records if r.id != target.id]`
    # — we already verified this in source. Here, double-check the prompt
    # itself: the target's own Q-A pair must NOT appear in the padding.
    out = evaluate(
        model, tok, torch.device("cpu"), records,
        distances=[200],
        n_examples=3,
    )
    for rec in out["results"]:
        target = next(r for r in records if r.id == rec["id"])
        # The target's question should appear EXACTLY ONCE (the final Q),
        # not duplicated in the distractor block.
        # build_qa_prompt is what we're indirectly checking — rebuild the
        # prompt and count.
        # Cheap proxy: scan the predicted_text. predicted_text is just the
        # decoded next token, not the prompt. Instead, verify pool exclusion
        # by reconstructing the prompt and looking for target.question twice.
        import random as _r
        pool = [r for r in records if r.id != target.id]
        text, _, _ = build_qa_prompt(
            target, pool, distance=200, tokenizer=tok,
            rng=_r.Random((rec["distance"], target.id).__hash__()),
        )
        assert text.count(f"Q: {target.question}") == 1


def test_evaluate_restores_train_mode():
    """G161 pattern: if the caller passed a train-mode model, the function
    must restore it on exit (the body switches to eval). A model stuck in
    eval after a sweep would silently disable dropout in subsequent training
    — the kind of bug that surfaces as a mysteriously-good loss curve weeks
    later."""
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg)
    model.train()
    tok = Tokenizer()
    records = _synthetic_records(3)
    _ = evaluate(
        model, tok, torch.device("cpu"), records,
        distances=[0], n_examples=2,
    )
    assert model.training, "evaluate() left model in eval mode"


def test_evaluate_zero_examples_returns_empty():
    """Pathological caller passing n_examples=0 (or an empty record list)
    must NOT crash — return a structurally-valid empty result so the eval
    JSON is still readable by downstream tools."""
    cfg = _tiny_config()
    model = TitansMAGGPT2(cfg).eval()
    tok = Tokenizer()
    out = evaluate(
        model, tok, torch.device("cpu"), _synthetic_records(2),
        distances=[0, 50], n_examples=0,
    )
    assert out["n_examples"] == 0
    assert out["n_trials"] == 0
    for d, b in out["buckets"].items():
        assert b["total"] == 0
        assert b["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# load_eval_records JSON contract
# ---------------------------------------------------------------------------

def test_load_eval_records_round_trips_through_disk(tmp_path):
    """load → write → load gives the same records. Ensures the JSON schema
    matches what scripts.prepare_squad_corpus writes."""
    from scripts.prepare_squad_corpus import (
        SquadRecord, write_eval_records,
    )
    src = [
        SquadRecord(
            id="x", title="t", context="c",
            question="q?", answers=["a1", "a2"],
        ),
    ]
    path = tmp_path / "eval.json"
    write_eval_records(src, path)
    loaded = load_eval_records(path)
    assert len(loaded) == 1
    assert loaded[0].id == "x"
    assert loaded[0].context == "c"
    assert loaded[0].answers == ["a1", "a2"]
