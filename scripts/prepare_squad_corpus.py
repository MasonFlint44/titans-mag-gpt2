"""Pack SQuAD into TITANS recall-scenario training data + held-out eval set.

Each training scenario mirrors the eval prompt shape exactly:

    [P] passage_1
    [P] passage_2
    ...
    [P] passage_N
    Q: question_about_passage_K
    A: answer_K<|endoftext|>

The N passages are topically disjoint (one per Wikipedia title) so the
model can't answer via topic-matching shortcuts. The "answered" passage
position K is uniformly random in [0, N-1] — sometimes the target is
first, sometimes middle, sometimes last — forcing the model to learn
content-based retrieval rather than positional shortcuts.

Scenario length is randomized by `min_passages..max_passages`, which
gives a uniform-ish distribution of recall distances at training time —
matching the eval bucket range so we're not testing OOD extrapolation.

Train and eval splits inherit SQuAD's own train/validation disjointness;
eval records additionally get filtered to those whose first answer
token is high-entropy (not a common LM word like " The" or " a"), so
first-token-argmax scoring measures recall, not LM-pattern guessing.

Outputs:

  --train-out  PATH/squad_train.txt
      Plain UTF-8 text. Each scenario is one document. Documents are
      `<|endoftext|>`-separated. The corpus loader splits on this marker
      (fixed by 3cbb4f8) so `encode_corpus` injects EOT-id 50256 between
      scenarios and the NMM resets per-scenario.

  --eval-out   PATH/squad_eval.json
      JSON list of held-out (id, title, context, question, answers)
      records. `scripts.eval_qa_recall` consumes this directly.

CLI:
    python -m scripts.prepare_squad_corpus \\
        --out-dir corpora/squad \\
        --n-scenarios 50000 \\
        --max-eval-records 2500 \\
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


# Sentinel between scenarios in the corpus file. The corpus loader
# (`read_eot_separated_documents`) splits on this and `encode_corpus`
# then injects EOT-id 50256 between documents — what the dataloader's
# `(streams == eot_id)` mask checks to reset NMM state.
EOT_LITERAL = "<|endoftext|>"


# Common low-entropy first BPE tokens that vanilla GPT-2 can predict from
# generic LM patterns without actually retrieving the target passage. Eval
# records whose first-answer token (with leading space, as it appears
# after `A: `) is in this set get filtered out — keeps the first-token-
# argmax metric measuring recall, not LM-bias.
#
# Tuned to include the most common answer-leading words in SQuAD; expand
# if a particular evaluation still shows vanilla doing suspiciously well
# at d=0 via LM-pattern guessing.
COMMON_FIRST_TOKENS = frozenset({
    " The", " the", " A", " a", " An", " an",
    " In", " On", " At", " To", " By", " For", " With", " From", " Of",
    " is", " was", " are", " were", " has", " have", " had", " be", " been",
    " of", " and", " or", " but", " not", " also", " only",
    " he", " she", " it", " they", " his", " her", " its", " their",
    " one", " two", " three", " many", " most", " some", " all",
})


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


def _format_passage_block(record: SquadRecord) -> str:
    """Render the passage portion of a scenario. Trailing newline so
    adjacent passage blocks read as separate paragraphs."""
    return f"[P] {record.context}\n"


def _format_qa_block(record: SquadRecord) -> str:
    """Render the Q/A portion of a scenario. No trailing newline — the
    scenario's EOT separator handles termination, and at eval time the
    same prompt prefix ends at `A:` with the model's next-token sample
    completing the answer."""
    if not record.answers:
        raise ValueError(
            f"SquadRecord {record.id!r} has no answers — cannot format Q/A "
            f"block. Drop the record upstream."
        )
    return f"Q: {record.question}\nA: {record.answers[0]}"


def group_by_title(records: Sequence[SquadRecord]) -> dict[str, list[SquadRecord]]:
    """Group SQuAD records by their `title` (Wikipedia article name). Used
    to enforce topic-disjoint passage sampling: each scenario draws at most
    one record per title, so distractors can't accidentally share subject
    matter with the target.
    """
    groups: dict[str, list[SquadRecord]] = {}
    for r in records:
        groups.setdefault(r.title, []).append(r)
    return groups


def has_high_entropy_first_token(
    record: SquadRecord, tokenizer: Tokenizer,
) -> bool:
    """True if the record's first answer's first BPE token (with leading
    space, as it appears after `A: ` in the prompt) is NOT a common
    LM-predictable word. Used by `filter_eval_by_first_token` to keep the
    eval set focused on records where first-token-argmax actually measures
    recall."""
    if not record.answers:
        return False
    first_ids = tokenizer.encode(f" {record.answers[0]}")
    if not first_ids:
        return False
    first_text = tokenizer.decode([first_ids[0]])
    return first_text not in COMMON_FIRST_TOKENS


def filter_eval_by_first_token(
    records: Sequence[SquadRecord], tokenizer: Tokenizer,
) -> list[SquadRecord]:
    """Drop records whose first answer token is in the common-LM blacklist.
    Typically halves the eval set; ensures the surviving records measure
    the architectural claim (cross-distance recall) without metric noise
    from LM-pattern guessing."""
    return [r for r in records if has_high_entropy_first_token(r, tokenizer)]


def build_recall_scenario(
    title_groups: dict[str, list[SquadRecord]],
    n_passages: int,
    rng: random.Random,
) -> str:
    """Build one recall scenario: `n_passages` topic-disjoint passages,
    one Q/A pair about a uniformly-random one of them. Returns the
    scenario text — caller adds the trailing `<|endoftext|>` separator.

    Raises ValueError if title_groups has fewer than n_passages titles.
    """
    titles = list(title_groups.keys())
    if len(titles) < n_passages:
        raise ValueError(
            f"Need {n_passages} unique titles for a scenario; "
            f"title_groups has {len(titles)}."
        )
    selected_titles = rng.sample(titles, n_passages)
    passages = [rng.choice(title_groups[t]) for t in selected_titles]
    # Uniformly random target position — breaks positional shortcuts the
    # model could otherwise exploit (e.g., "the first passage is the
    # target"). At eval, target is always at position 0; if training
    # also always put target at 0, the model could win on position alone.
    target_idx = rng.randint(0, n_passages - 1)
    target = passages[target_idx]
    passages_block = "".join(_format_passage_block(r) for r in passages)
    qa_block = _format_qa_block(target)
    return passages_block + qa_block


def build_recall_scenarios(
    records: Sequence[SquadRecord],
    n_scenarios: int,
    *,
    min_passages: int = 2,
    max_passages: int = 20,
    seed: int = 0,
) -> list[str]:
    """Build `n_scenarios` recall scenarios from `records`. Each scenario's
    passage count is uniformly random in [min_passages, max_passages] —
    this gives a wide spread of recall distances at training time so the
    model sees the recall task across the same range it'll be evaluated
    at.

    With `min_passages=2, max_passages=20` and SQuAD's ~150-token average
    passage length, scenarios span ~300 to ~3000 tokens — covering the
    eval buckets [0, 256, 512, 1024, 2048, 3072].
    """
    if min_passages < 1:
        raise ValueError(f"min_passages must be >=1 (got {min_passages})")
    if max_passages < min_passages:
        raise ValueError(
            f"max_passages ({max_passages}) must be >= "
            f"min_passages ({min_passages})"
        )
    title_groups = group_by_title(records)
    if len(title_groups) < max_passages:
        raise ValueError(
            f"Need at least {max_passages} unique titles to support "
            f"max-passages scenarios; SQuAD-train has "
            f"{len(title_groups)} after grouping."
        )
    rng = random.Random(seed)
    scenarios: list[str] = []
    for _ in range(n_scenarios):
        n_passages = rng.randint(min_passages, max_passages)
        scenarios.append(build_recall_scenario(title_groups, n_passages, rng))
    return scenarios


def write_train_corpus(scenarios: Iterable[str], path: Path) -> int:
    """Write EOT-literal-separated scenarios. Returns the byte count
    written. Each scenario becomes one document at corpus-load time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = f"\n{EOT_LITERAL}\n"
    text = sep.join(scenarios)
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
    EOT-id insertion between scenarios. Strips empty leading/trailing
    fragments produced by the join."""
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
        description=(
            "Pack SQuAD into TITANS recall-scenario training corpus + "
            "held-out eval JSON."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("corpora/squad"),
        help="Output directory for squad_train.txt and squad_eval.json. "
             "Default: corpora/squad",
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=50_000,
        help="Number of training scenarios. Each is one EOT-bounded "
             "document. Default: 50000.",
    )
    parser.add_argument(
        "--min-passages", type=int, default=2,
        help="Min number of passages per scenario. Default: 2.",
    )
    parser.add_argument(
        "--max-passages", type=int, default=20,
        help="Max passages per scenario. Default: 20 (≈3000 tokens at "
             "150 tokens/passage, covers the d=3072 eval bucket).",
    )
    parser.add_argument(
        "--max-eval-records", type=int, default=5000,
        help="Cap on raw eval set size BEFORE first-token filtering. "
             "Filtering typically drops ~50%, so 5000 raw → ~2500 usable. "
             "Default: 5000.",
    )
    parser.add_argument(
        "--no-filter-eval-first-token",
        action="store_true",
        help="Disable the high-entropy first-token filter on eval records. "
             "Default: filter ON.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for train shuffle + scenario generation + eval "
             "subsample. Default: 0.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer()

    print("[squad] loading train split...", flush=True)
    train_records = load_squad_train(seed=args.seed)
    print(f"[squad] loaded {len(train_records)} train records",
          flush=True)

    print(f"[squad] building {args.n_scenarios} recall scenarios "
          f"(passages {args.min_passages}-{args.max_passages})...",
          flush=True)
    scenarios = build_recall_scenarios(
        train_records,
        n_scenarios=args.n_scenarios,
        min_passages=args.min_passages,
        max_passages=args.max_passages,
        seed=args.seed,
    )

    train_path = args.out_dir / "squad_train.txt"
    n_chars = write_train_corpus(scenarios, train_path)
    print(f"[squad] wrote {train_path} ({n_chars / 1e6:.1f} MB, "
          f"{len(scenarios)} scenarios)", flush=True)

    print(f"[squad] loading validation split (cap "
          f"{args.max_eval_records})...", flush=True)
    eval_records = load_squad_eval(
        n_records=args.max_eval_records, seed=args.seed,
    )
    print(f"[squad] loaded {len(eval_records)} raw eval records",
          flush=True)

    if not args.no_filter_eval_first_token:
        print("[squad] filtering eval records by first-token entropy...",
              flush=True)
        before = len(eval_records)
        eval_records = filter_eval_by_first_token(eval_records, tokenizer)
        print(f"[squad] kept {len(eval_records)}/{before} eval records "
              f"after first-token filter", flush=True)

    eval_path = args.out_dir / "squad_eval.json"
    n_records = write_eval_records(eval_records, eval_path)
    print(f"[squad] wrote {eval_path} ({n_records} records)",
          flush=True)


if __name__ == "__main__":
    main()
