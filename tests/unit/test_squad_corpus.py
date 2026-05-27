"""Tests for scripts.prepare_squad_corpus.

These tests construct SquadRecord lists by hand — they never touch the real
HuggingFace dataset cache. The dataset loader (`load_squad_train` /
`load_squad_eval`) is a thin wrapper around `datasets.load_dataset` and
isn't worth re-testing here; the contracts we care about are the scenario
construction and serialization logic.
"""

import json
from pathlib import Path

import pytest

from data.tokenizer import Tokenizer
from scripts.prepare_squad_corpus import (
    COMMON_FIRST_TOKENS,
    EOT_LITERAL,
    SquadRecord,
    _format_passage_block,
    _format_qa_block,
    build_recall_scenario,
    build_recall_scenarios,
    filter_eval_by_first_token,
    group_by_title,
    has_high_entropy_first_token,
    read_eot_separated_documents,
    write_eval_records,
    write_train_corpus,
)


def _record(i: int, title: str | None = None, n_words: int = 20) -> SquadRecord:
    """Synthetic SquadRecord. `title` defaults to a per-record unique value
    so default-config tests get topic-disjoint records for free; callers
    that want title collisions pass an explicit shared title."""
    context = " ".join(f"word{i}_{j}" for j in range(n_words))
    return SquadRecord(
        id=f"id-{i}",
        title=title if title is not None else f"title-{i}",
        context=context,
        question=f"What is word{i}_5?",
        answers=[f"word{i}_5"],
    )


# ---------------------------------------------------------------------------
# Passage / Q&A block formatting
# ---------------------------------------------------------------------------

def test_format_passage_block_emits_only_passage():
    """Passage block must contain ONLY the `[P] ...` line — no Q/A. In the
    new training scenario shape, distractors are passage-only; the single
    Q/A goes at the end. If passage-block leaked Q/A content, scenarios
    would have spurious extra questions during training."""
    r = _record(0, n_words=5)
    text = _format_passage_block(r)
    assert text.startswith("[P] word0_0 word0_1 word0_2 word0_3 word0_4")
    assert text.endswith("\n")
    assert "Q:" not in text
    assert "A:" not in text


def test_format_qa_block_emits_question_and_first_answer_only():
    """Q/A block must contain the question and the FIRST listed answer.
    Validation records may have multiple aliases; we use only the first
    so the training-time next-token target is deterministic."""
    r = SquadRecord(
        id="x", title="t", context="ctx",
        question="q?", answers=["primary", "alias_b", "alias_c"],
    )
    text = _format_qa_block(r)
    assert "Q: q?" in text
    assert "A: primary" in text
    assert "alias_b" not in text
    assert "alias_c" not in text


def test_format_qa_block_rejects_empty_answers():
    """Records without answers can't generate a Q/A block — fail loud
    rather than emit `A: ` and silently teach the model to predict
    nothing."""
    r = SquadRecord(id="x", title="t", context="c", question="q?", answers=[])
    with pytest.raises(ValueError, match="no answers"):
        _format_qa_block(r)


def test_format_qa_block_has_no_trailing_newline():
    """The Q/A block sits at scenario end and is followed by `<|endoftext|>`
    (added by write_train_corpus's separator). A trailing newline here
    would inject extra whitespace between answer and EOT, drifting the
    train distribution from the eval prompt that ends with `A:`."""
    r = _record(0, n_words=3)
    text = _format_qa_block(r)
    assert not text.endswith("\n")


# ---------------------------------------------------------------------------
# group_by_title
# ---------------------------------------------------------------------------

def test_group_by_title_collects_records_per_title():
    """Multi-record-per-title is the common SQuAD pattern (many questions
    per Wikipedia article). Topic-disjoint sampling depends on this
    grouping working correctly."""
    records = [
        _record(0, title="Alpha"),
        _record(1, title="Alpha"),
        _record(2, title="Beta"),
        _record(3, title="Gamma"),
    ]
    groups = group_by_title(records)
    assert set(groups.keys()) == {"Alpha", "Beta", "Gamma"}
    assert len(groups["Alpha"]) == 2
    assert len(groups["Beta"]) == 1
    assert len(groups["Gamma"]) == 1


# ---------------------------------------------------------------------------
# build_recall_scenario / build_recall_scenarios
# ---------------------------------------------------------------------------

def test_scenario_has_passage_count_plus_exactly_one_qa():
    """Each scenario must contain `n_passages` passage blocks and exactly
    ONE Q/A pair. Multiple Q/A pairs would deviate from the eval prompt
    shape (which has one final question)."""
    import random
    records = [_record(i) for i in range(10)]
    groups = group_by_title(records)
    scenario = build_recall_scenario(groups, n_passages=5, rng=random.Random(0))
    assert scenario.count("[P] ") == 5
    assert scenario.count("Q: ") == 1
    assert scenario.count("A: ") == 1


def test_scenario_target_position_varies_across_seeds():
    """The 'answered' passage must be uniformly random in [0, n_passages),
    not fixed at the start. Without this the model would learn a
    positional shortcut ('the first passage is the target') that wouldn't
    transfer to the eval prompt's content-based recall."""
    import random
    # Generate 50 scenarios and check that the answered question varies
    # across multiple records (not always pointing to position 0).
    records = [_record(i) for i in range(20)]
    groups = group_by_title(records)
    asked_indices: set[int] = set()
    for seed in range(50):
        rng = random.Random(seed)
        scenario = build_recall_scenario(groups, n_passages=5, rng=rng)
        # Identify which record the Q/A is about by checking which
        # record's question text appears in the scenario.
        for idx, r in enumerate(records):
            if f"Q: {r.question}" in scenario:
                # Find this record's position in the scenario.
                passages = scenario.split("[P] ")[1:]  # split off leading "[P] "
                for pos, p in enumerate(passages):
                    if r.context in p:
                        asked_indices.add(pos)
                        break
                break
    # Across 50 scenarios, the target should land at multiple positions.
    assert len(asked_indices) >= 3, (
        f"target position only varied across {len(asked_indices)} unique "
        f"positions in 50 scenarios — positional bias risk"
    )


def test_scenario_passages_are_topic_disjoint():
    """No two passages within a scenario should share a Wikipedia title.
    The eval also filters distractors by title; training must match."""
    import random
    # 20 records, 4 per title × 5 titles
    records = []
    for t in range(5):
        for i in range(4):
            records.append(_record(t * 4 + i, title=f"shared_title_{t}"))
    groups = group_by_title(records)
    rng = random.Random(0)
    scenario = build_recall_scenario(groups, n_passages=5, rng=rng)
    # Each of the 5 passages should be from a different title.
    titles_seen = set()
    for r in records:
        if r.context in scenario:
            titles_seen.add(r.title)
    # Even though some records share a title, only one record per title
    # should appear → titles_seen size should equal n_passages (5).
    assert len(titles_seen) == 5


def test_build_recall_scenario_raises_when_too_few_titles():
    """Asking for more topic-disjoint passages than there are unique
    titles is a configuration error; fail loud rather than silently
    repeating titles."""
    import random
    records = [_record(i, title="only_one") for i in range(5)]
    groups = group_by_title(records)
    with pytest.raises(ValueError, match="unique titles"):
        build_recall_scenario(groups, n_passages=3, rng=random.Random(0))


def test_build_recall_scenarios_count_matches_request():
    records = [_record(i) for i in range(25)]
    scenarios = build_recall_scenarios(
        records, n_scenarios=7, min_passages=2, max_passages=5, seed=0,
    )
    assert len(scenarios) == 7


def test_build_recall_scenarios_distance_distribution_spans_range():
    """Across many scenarios, passage count must span [min, max]
    so training sees a wide range of recall distances. Without this,
    eval at large distances would be testing OOD generalization."""
    records = [_record(i) for i in range(50)]
    scenarios = build_recall_scenarios(
        records, n_scenarios=200, min_passages=2, max_passages=15, seed=0,
    )
    passage_counts = [s.count("[P] ") for s in scenarios]
    assert min(passage_counts) == 2
    assert max(passage_counts) == 15


def test_build_recall_scenarios_rejects_invalid_passage_bounds():
    records = [_record(i) for i in range(20)]
    with pytest.raises(ValueError, match="min_passages"):
        build_recall_scenarios(records, n_scenarios=1, min_passages=0)
    with pytest.raises(ValueError, match="max_passages"):
        build_recall_scenarios(
            records, n_scenarios=1, min_passages=5, max_passages=3,
        )


# ---------------------------------------------------------------------------
# First-token entropy filter
# ---------------------------------------------------------------------------

def test_common_first_tokens_contains_articles_and_prepositions():
    """Spot-check the blacklist — these are the obvious LM-predictable
    answer-starting words we don't want to count toward recall."""
    assert " The" in COMMON_FIRST_TOKENS
    assert " the" in COMMON_FIRST_TOKENS
    assert " A" in COMMON_FIRST_TOKENS
    assert " a" in COMMON_FIRST_TOKENS
    assert " is" in COMMON_FIRST_TOKENS
    assert " in" not in COMMON_FIRST_TOKENS or " In" in COMMON_FIRST_TOKENS


def test_has_high_entropy_first_token_drops_common_starts():
    tok = Tokenizer()
    low = SquadRecord(
        id="low", title="t", context="c", question="q?",
        answers=["The Beatles"],
    )
    assert not has_high_entropy_first_token(low, tok)


def test_has_high_entropy_first_token_keeps_proper_nouns():
    tok = Tokenizer()
    high = SquadRecord(
        id="high", title="t", context="c", question="q?",
        answers=["Beyoncé"],
    )
    assert has_high_entropy_first_token(high, tok)


def test_has_high_entropy_first_token_handles_empty_answers():
    """Records without answers can't have a first token — must return
    False so they don't silently pass through the filter."""
    tok = Tokenizer()
    empty = SquadRecord(
        id="empty", title="t", context="c", question="q?", answers=[],
    )
    assert not has_high_entropy_first_token(empty, tok)


def test_filter_eval_by_first_token_drops_low_entropy_records():
    tok = Tokenizer()
    records = [
        SquadRecord("a", "t", "c", "q?", ["The first answer"]),
        SquadRecord("b", "t", "c", "q?", ["Beyoncé"]),
        SquadRecord("c", "t", "c", "q?", ["the second answer"]),
        SquadRecord("d", "t", "c", "q?", ["1492"]),
    ]
    kept = filter_eval_by_first_token(records, tok)
    kept_ids = {r.id for r in kept}
    # "The" / "the" should be filtered; proper nouns and numbers kept.
    assert "a" not in kept_ids
    assert "c" not in kept_ids
    assert "b" in kept_ids
    assert "d" in kept_ids


# ---------------------------------------------------------------------------
# Serialization round-trip + EOT-separated read path
# ---------------------------------------------------------------------------

def test_write_train_corpus_round_trips_through_split(tmp_path):
    """`write_train_corpus` + `read_eot_separated_documents` is the
    contract for the per-scenario EOT-reset loader path. Whatever write
    puts down, read must recover (modulo leading/trailing whitespace
    stripped by read)."""
    scenarios = [
        "[P] passage A\nQ: q?\nA: a",
        "[P] passage B\nQ: q2?\nA: a2",
    ]
    path = tmp_path / "corpus.txt"
    write_train_corpus(scenarios, path)

    recovered = read_eot_separated_documents(path)
    assert len(recovered) == len(scenarios)
    for orig, got in zip(scenarios, recovered):
        assert orig.strip() == got.strip()


def test_write_train_corpus_inserts_eot_literal(tmp_path):
    """The EOT marker must appear between scenarios so the corpus loader
    correctly identifies document boundaries (which trigger the NMM
    state reset). Without this, training is back in the unbounded-
    accumulation regime fixed by 3cbb4f8."""
    scenarios = ["one", "two", "three"]
    path = tmp_path / "corpus.txt"
    write_train_corpus(scenarios, path)
    text = path.read_text(encoding="utf-8")
    assert text.count(EOT_LITERAL) == len(scenarios) - 1


def test_write_train_corpus_creates_parent_dir(tmp_path):
    """Same mkdir-p contract as save_nmm_state / save_checkpoint_rotating."""
    nested = tmp_path / "deep" / "nested" / "corpus.txt"
    write_train_corpus(["only one"], nested)
    assert nested.is_file()


# ---------------------------------------------------------------------------
# Eval JSON write
# ---------------------------------------------------------------------------

def test_write_eval_records_serializes_all_fields(tmp_path):
    """Eval consumer (scripts.eval_qa_recall) needs every field present —
    missing answers or context would silently make examples unscorable."""
    records = [
        SquadRecord(
            id="rid-1", title="article", context="ctx text",
            question="q?", answers=["a1", "alias-of-a1"],
        ),
    ]
    path = tmp_path / "eval.json"
    n = write_eval_records(records, path)
    assert n == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0] == {
        "id": "rid-1",
        "title": "article",
        "context": "ctx text",
        "question": "q?",
        "answers": ["a1", "alias-of-a1"],
    }


def test_write_eval_records_preserves_answer_aliases(tmp_path):
    """SQuAD validation has multiple answer aliases per question. They
    MUST all reach the eval JSON so the scorer can match any of them."""
    r = SquadRecord(
        id="x", title="t", context="c", question="q?",
        answers=["primary", "alias_b", "alias_c"],
    )
    path = tmp_path / "eval.json"
    write_eval_records([r], path)
    out = json.loads(path.read_text())
    assert out[0]["answers"] == ["primary", "alias_b", "alias_c"]


def test_squad_record_from_hf_dedupes_aliases():
    """HF SQuAD val rows often list each annotator's answer separately,
    so the same string appears multiple times. We dedupe at SquadRecord
    construction so the eval scorer doesn't double-weight common
    answers."""
    row = {
        "id": "x", "title": "t", "context": "c", "question": "q?",
        "answers": {
            "text": ["Denver Broncos", "Denver Broncos", "Denver Broncos"],
            "answer_start": [0, 0, 0],
        },
    }
    r = SquadRecord.from_hf(row)
    assert r.answers == ["Denver Broncos"]


def test_squad_record_from_hf_preserves_alias_order():
    """Dedup must be order-preserving — the FIRST listed answer is
    treated as canonical by `_format_qa_block` (used as the training
    target). If dedup reordered, training would target a non-canonical
    alias."""
    row = {
        "id": "x", "title": "t", "context": "c", "question": "q?",
        "answers": {
            "text": ["primary", "alias_b", "primary", "alias_c"],
            "answer_start": [0, 0, 0, 0],
        },
    }
    r = SquadRecord.from_hf(row)
    assert r.answers == ["primary", "alias_b", "alias_c"]
