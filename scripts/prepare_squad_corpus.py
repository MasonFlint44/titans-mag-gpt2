"""Pack SQuAD into TITANS training sequences and a held-out eval set.

Each "packed sequence" interleaves several (passage, Q, A) triples so the model
sees the long-range pattern at training time:

    [P] passage_1
    Q: question_1
    A: answer_1

    [P] passage_2
    Q: question_2
    A: answer_2

    ... (until token budget exhausted)

We target T ≤ 1024 per packed sequence with a small slack (32 tokens) to absorb
tokenization variance. SQuAD passages average ~150 BPE tokens and Q+A average
~25, so each packed sequence holds 4-6 triples.

Outputs:

  --train-out  PATH/squad_train.txt
      Plain UTF-8 text. Each packed sequence is one paragraph (the within-
      sequence formatting already contains blank lines); paragraphs are
      separated by an EOT marker `<|endoftext|>\\n\\n`. `finetune.py` reads
      the file as one document by default, which means cross-sequence NMM
      state carries over — for this experiment that's acceptable (residual
      state from prior topics decays inside the within-sequence training
      signal). If you want strict per-sequence EOT resets, swap the text-
      file read for `read_eot_separated_documents(path)` defined in this
      module (returns the list-of-sequences ready for `encode_corpus`).

  --eval-out   PATH/squad_eval.json
      JSON list of held-out (id, context, question, answers) records pulled
      from the SQuAD validation split. `scripts.eval_qa_recall` consumes
      this directly — it doesn't need the training-side packing.

This script reads from the local Hugging Face datasets cache. It will fall
through to a network download if the dataset isn't cached yet; we keep the
default download_mode so cached runs are fast and the first run still works.

CLI:
    python -m scripts.prepare_squad_corpus \\
        --out-dir corpora/squad \\
        --target-tokens 992 \\
        --max-eval-records 2000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from data.tokenizer import Tokenizer


# Sentinel between packed sequences. Tiktoken's `encode_single_token` resolves
# this to id 50256 only when `allowed_special` is passed — plain `encode()` BPEs
# the literal text. We use the literal anyway so a hypothetical strict-EOT
# loader can split on it without re-tokenizing.
EOT_LITERAL = "<|endoftext|>"


@dataclass
class SquadRecord:
    """A single (context, question, answers) triple. `answers` is a list of
    accepted gold-answer strings (SQuAD validation has multiple aliases per
    question; training has one)."""
    id: str
    title: str
    context: str
    question: str
    answers: list[str]

    @classmethod
    def from_hf(cls, row: dict) -> "SquadRecord":
        """Construct from a Hugging Face SQuAD row. Deduplicates answer aliases
        because SQuAD validation lists each alias once per annotator and we
        only need unique strings for substring scoring."""
        answers = row["answers"]["text"]
        # Order-preserving dedup so the canonical answer (annotator 1) stays
        # first — useful when callers want a single "primary" answer.
        seen = set()
        unique = []
        for a in answers:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return cls(
            id=row["id"],
            title=row["title"],
            context=row["context"],
            question=row["question"],
            answers=unique,
        )


def format_triple(record: SquadRecord) -> str:
    """Format one (passage, Q, A) triple. The training corpus uses the FIRST
    listed answer as the gold completion (SQuAD train always lists exactly
    one answer per question; validation may have multiple, but we don't pack
    validation into the training corpus).

    Layout choice: explicit `[P]` / `Q:` / `A:` markers because (a) they give
    the model a clear cue for "next come Q/A pairs that reference the marked
    passage", and (b) the same markers are reused at eval time when we build
    the recall prompt — keeping training and eval syntactically aligned
    avoids spurious distribution-shift loss.
    """
    if not record.answers:
        # Validation has multi-alias lists; training always has at least one.
        # If a caller hands us a malformed record, skip it loudly.
        raise ValueError(
            f"SquadRecord {record.id!r} has no answers — cannot format for "
            f"training. Drop the record upstream or fill answers manually."
        )
    return (
        f"[P] {record.context}\n"
        f"Q: {record.question}\n"
        f"A: {record.answers[0]}\n"
    )


def pack_sequences(
    records: Sequence[SquadRecord],
    tokenizer: Tokenizer,
    target_tokens: int = 992,
) -> list[str]:
    """Greedily pack triples into sequences each ≤ `target_tokens` tokens.

    Why greedy + target=992 (not strict T=1024): tokenizer count is over the
    triple text only — actual training-time positions include the EOT marker
    between sequences and any residual context from prior sequences in the
    ParallelStreamLoader stride. 32 tokens of slack absorbs that without
    forcing per-sequence trimming.

    Triples that individually exceed `target_tokens` are emitted as their
    own sequence anyway (so we don't lose data); they'll be truncated by
    the loader, but that's their problem — SQuAD passages above 1000 BPE
    tokens are vanishingly rare (< 0.1%).
    """
    sequences: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for record in records:
        text = format_triple(record)
        n_tokens = len(tokenizer.encode(text))
        # Triples larger than the budget go in alone — emit any pending
        # sequence first, then the oversize triple, then start fresh.
        if n_tokens > target_tokens:
            if current:
                sequences.append("\n".join(current))
                current = []
                current_tokens = 0
            sequences.append(text)
            continue
        if current_tokens + n_tokens > target_tokens and current:
            sequences.append("\n".join(current))
            current = [text]
            current_tokens = n_tokens
        else:
            current.append(text)
            current_tokens += n_tokens
    if current:
        sequences.append("\n".join(current))
    return sequences


def write_train_corpus(sequences: Iterable[str], path: Path) -> int:
    """Write packed sequences separated by `<|endoftext|>` literal markers.

    The EOT literal between sequences makes it trivial for a downstream
    loader to split on the marker if it wants per-sequence document
    boundaries. The default `finetune.py` path treats the whole file as
    one document (state threads across sequences); both are valid.

    Returns the number of characters written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = f"\n{EOT_LITERAL}\n\n"
    text = sep.join(sequences)
    path.write_text(text, encoding="utf-8")
    return len(text)


def write_eval_records(records: Sequence[SquadRecord], path: Path) -> int:
    """Write held-out eval records as JSON. Returns the number of records
    written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": r.id,
            "title": r.title,
            "context": r.context,
            "question": r.question,
            "answers": r.answers,
        }
        for r in records
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return len(payload)


def read_eot_separated_documents(path: Path) -> list[str]:
    """Inverse of `write_train_corpus`: split on the EOT literal so the
    returned list can be passed to `Tokenizer.encode_corpus` for proper
    EOT-id insertion between sequences (NMM resets per sequence at training
    time). Strips empty leading/trailing fragments produced by the join."""
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split(EOT_LITERAL)
    return [p.strip("\n") for p in parts if p.strip()]


def load_squad_train(seed: int = 0) -> list[SquadRecord]:
    """Load the SQuAD training split, shuffle deterministically, return as
    SquadRecord list. Cached download — first run pays a one-time fetch."""
    from datasets import load_dataset  # local import: optional at module load
    ds = load_dataset("rajpurkar/squad", split="train")
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [SquadRecord.from_hf(r) for r in rows]


def load_squad_eval(
    n_records: int | None = None, seed: int = 0,
) -> list[SquadRecord]:
    """Load the SQuAD validation split and subsample. Validation is the
    "held-out" set in SQuAD's own definition — disjoint from train by
    construction."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    rows = list(ds)
    rng = random.Random(seed + 1)  # different seed than train shuffle
    rng.shuffle(rows)
    if n_records is not None:
        rows = rows[:n_records]
    return [SquadRecord.from_hf(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack SQuAD into TITANS training sequences + eval JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("corpora/squad"),
        help="Directory for squad_train.txt and squad_eval.json. Created if "
             "missing. Default: corpora/squad",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=992,
        help="Target token budget per packed training sequence. T=1024 model "
             "context minus 32-token slack. Default: 992",
    )
    parser.add_argument(
        "--max-eval-records",
        type=int,
        default=2500,
        help="Cap on eval set size — 500 examples × 5 distance buckets needs "
             "2500 records minimum (each example evaluated once per bucket). "
             "Default: 2500",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for train shuffle + eval subsample. Default: 0",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer()

    print(f"[squad] loading train split...", flush=True)
    train_records = load_squad_train(seed=args.seed)
    print(f"[squad] loaded {len(train_records)} train records", flush=True)

    print(f"[squad] packing into ≤{args.target_tokens}-token sequences...",
          flush=True)
    sequences = pack_sequences(
        train_records, tokenizer, target_tokens=args.target_tokens,
    )
    print(f"[squad] packed {len(sequences)} sequences "
          f"({len(train_records)/len(sequences):.1f} triples/seq avg)",
          flush=True)

    train_path = args.out_dir / "squad_train.txt"
    n_chars = write_train_corpus(sequences, train_path)
    print(f"[squad] wrote {train_path} ({n_chars / 1e6:.1f} MB)", flush=True)

    print(f"[squad] loading validation split (cap {args.max_eval_records})...",
          flush=True)
    eval_records = load_squad_eval(
        n_records=args.max_eval_records, seed=args.seed,
    )

    eval_path = args.out_dir / "squad_eval.json"
    n_records = write_eval_records(eval_records, eval_path)
    print(f"[squad] wrote {eval_path} ({n_records} records)", flush=True)


if __name__ == "__main__":
    main()
