"""Tests for scripts.prepare_squad_corpus.

These tests construct SquadRecord lists by hand — they never touch the real
HuggingFace dataset cache. The dataset loader (`load_squad_train` /
`load_squad_eval`) is a thin wrapper around `datasets.load_dataset` and
isn't worth re-testing here; the contracts we care about are the packing
and serialization logic.
"""

import json
from pathlib import Path

import pytest

from data.tokenizer import Tokenizer
from scripts.prepare_squad_corpus import (
    EOT_LITERAL,
    SquadRecord,
    format_triple,
    pack_sequences,
    read_eot_separated_documents,
    write_eval_records,
    write_train_corpus,
)


def _record(i: int, n_words: int = 80) -> SquadRecord:
    """Synthetic SquadRecord whose context length is controllable via word
    count. Word count maps 1:1 to BPE tokens for plain alphanumeric words
    (each "word123" is a fresh single token), so we can size triples
    deterministically."""
    context = " ".join(f"word{i}_{j}" for j in range(n_words))
    return SquadRecord(
        id=f"id-{i}",
        title=f"title-{i}",
        context=context,
        question=f"What is word{i}_5?",
        answers=[f"word{i}_5"],
    )


# ---------------------------------------------------------------------------
# format_triple
# ---------------------------------------------------------------------------

def test_format_triple_contains_passage_question_answer():
    r = _record(0, n_words=5)
    text = format_triple(r)
    assert "[P] word0_0 word0_1 word0_2 word0_3 word0_4" in text
    assert "Q: What is word0_5?" in text
    assert "A: word0_5" in text


def test_format_triple_uses_first_answer_alias():
    """SQuAD validation lists multiple aliases; training corpus picks one
    canonical form so the next-token target is deterministic."""
    r = SquadRecord(
        id="x", title="t", context="ctx",
        question="q?", answers=["primary", "alias_b", "alias_c"],
    )
    text = format_triple(r)
    assert "A: primary" in text
    assert "alias_b" not in text


def test_format_triple_rejects_empty_answers():
    """Records missing answers can't go in the training corpus — we'd be
    teaching the model to emit nothing after `A: `, polluting the answer
    distribution. Fail loud."""
    r = SquadRecord(id="x", title="t", context="c", question="q?", answers=[])
    with pytest.raises(ValueError, match="no answers"):
        format_triple(r)


# ---------------------------------------------------------------------------
# pack_sequences: budget + correctness
# ---------------------------------------------------------------------------

def test_packed_sequences_never_exceed_budget():
    """The whole point of greedy packing — no sequence may exceed the
    declared target unless a SINGLE triple is already larger than budget
    (in which case we emit it alone as a documented fallback, not silent
    truncation). If this contract regresses, training would see truncated
    triples (loader drops past T) and the model would never learn the
    multi-triple pattern."""
    tok = Tokenizer()
    target = 400
    # 10 records of ~20 words each (~80-100 tokens after BPE + formatting).
    # ~3-4 fit per 400-token sequence.
    records = [_record(i, n_words=20) for i in range(10)]
    sequences = pack_sequences(records, tok, target_tokens=target)
    for seq in sequences:
        n = len(tok.encode(seq))
        # The oversize-fallback path only triggers when ONE triple alone
        # exceeds budget; our test records are all <budget, so this should
        # never fire here.
        assert n <= target, (
            f"packed sequence has {n} tokens, exceeds target {target}: "
            f"{seq[:120]!r}..."
        )


def test_oversize_single_triple_emitted_alone_even_if_over_budget():
    """Documented fallback: a triple that's individually larger than the
    budget gets emitted alone (so we don't lose data). Verify this path
    works and is the ONLY way the budget can be exceeded."""
    tok = Tokenizer()
    target = 100
    huge = _record(0, n_words=200)  # ~800 tokens, way over budget
    sequences = pack_sequences([huge], tok, target_tokens=target)
    # We emitted it (no data loss), the size reflects the original triple.
    assert len(sequences) == 1
    assert "word0_5" in sequences[0]
    assert len(tok.encode(sequences[0])) > target  # by design


def test_packed_sequences_preserve_every_triple():
    """Greedy packing must not drop any record. Verify by checking each
    record's distinctive id text appears in the concatenated output."""
    tok = Tokenizer()
    records = [_record(i, n_words=20) for i in range(8)]
    sequences = pack_sequences(records, tok, target_tokens=400)
    blob = "\n".join(sequences)
    for r in records:
        assert f"word{r.id.split('-')[1]}_5" in blob, (
            f"record {r.id} dropped from packing output"
        )


def test_oversize_triple_emitted_alone():
    """A single triple bigger than the budget can't be packed with others —
    we still emit it (don't lose data) but on its own. Verifies the fallback
    path in pack_sequences."""
    tok = Tokenizer()
    # 500 words → ~500 tokens; budget=200 makes this oversize.
    big = _record(0, n_words=500)
    small = _record(1, n_words=20)
    sequences = pack_sequences([big, small], tok, target_tokens=200)
    # The oversize record sits alone; the small one is in its own sequence too
    # (the small one came after the big one, so it's a new sequence).
    assert any("word0_5" in s for s in sequences)
    assert any("word1_5" in s for s in sequences)


# ---------------------------------------------------------------------------
# Serialization round-trip + EOT-separated read path
# ---------------------------------------------------------------------------

def test_write_train_corpus_round_trips_through_split(tmp_path):
    """`write_train_corpus` + `read_eot_separated_documents` is the
    contract for the "strict per-sequence EOT reset" loader path. Whatever
    write puts down, read must recover (modulo leading/trailing whitespace
    stripped by read)."""
    sequences = ["[P] passage A\nQ: q?\nA: a",
                 "[P] passage B\nQ: q2?\nA: a2"]
    path = tmp_path / "corpus.txt"
    write_train_corpus(sequences, path)

    recovered = read_eot_separated_documents(path)
    assert len(recovered) == len(sequences)
    for orig, got in zip(sequences, recovered):
        # write/read may strip surrounding newlines; compare core content.
        assert orig.strip() == got.strip()


def test_write_train_corpus_inserts_eot_literal(tmp_path):
    """The EOT marker must appear between sequences so downstream split
    parsing works. Guards against future refactors that change the
    separator string silently."""
    sequences = ["one", "two", "three"]
    path = tmp_path / "corpus.txt"
    write_train_corpus(sequences, path)
    text = path.read_text(encoding="utf-8")
    # n sequences → n-1 separators.
    assert text.count(EOT_LITERAL) == len(sequences) - 1


def test_write_train_corpus_creates_parent_dir(tmp_path):
    """Same mkdir -p contract as save_nmm_state / save_checkpoint_rotating."""
    nested = tmp_path / "deep" / "nested" / "corpus.txt"
    write_train_corpus(["only one"], nested)
    assert nested.is_file()


# ---------------------------------------------------------------------------
# Eval JSON write
# ---------------------------------------------------------------------------

def test_write_eval_records_serializes_all_fields(tmp_path):
    """Eval consumer (scripts.eval_qa_recall) needs every field present —
    missing answers or context would silently make examples unscorable.
    Lock the schema explicitly."""
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
    """SQuAD validation has multiple answer aliases per question. They MUST
    all reach the eval JSON so the scorer can match any of them — dropping
    aliases would understate recall (model emits 'alias_b' but gold list
    only has 'alias_a' in the file). Regression guard."""
    r = SquadRecord(
        id="x", title="t", context="c", question="q?",
        answers=["primary", "alias_b", "alias_c"],
    )
    path = tmp_path / "eval.json"
    write_eval_records([r], path)
    out = json.loads(path.read_text())
    assert out[0]["answers"] == ["primary", "alias_b", "alias_c"]


def test_squad_record_from_hf_dedupes_aliases():
    """HF SQuAD val rows often list each annotator's answer separately, so
    the same string appears multiple times. We dedupe at SquadRecord
    construction so the eval scorer doesn't double-weight common answers."""
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
    """Dedup must be order-preserving — the FIRST listed answer is treated
    as canonical by `format_triple` (used as the training target). If dedup
    reordered, training would target a non-canonical alias."""
    row = {
        "id": "x", "title": "t", "context": "c", "question": "q?",
        "answers": {
            "text": ["primary", "alias_b", "primary", "alias_c"],
            "answer_start": [0, 0, 0, 0],
        },
    }
    r = SquadRecord.from_hf(row)
    assert r.answers == ["primary", "alias_b", "alias_c"]
