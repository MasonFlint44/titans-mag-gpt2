"""Needle-in-haystack corpus generator for TITANS recall evaluation.

Each training example tests the model's ability to encode and retrieve a
specific high-entropy token sequence at a controlled distance from the
question. Format:

    The secret code is XK-7281.
    [passage text padding totaling ≈ `distance` tokens]
    Q: What is the secret code?
    A: XK-7281<|endoftext|>

The needle (e.g., "XK-7281") is a random 2-letter + 4-digit alphanumeric
code — chosen for high entropy (26² × 10⁴ ≈ 6.7M unique codes, so no
training collision at our scale) and predictable tokenization (the first
BPE token after `A: ` is unambiguous and not LM-guessable, so first-
token-argmax scoring directly measures recall).

Padding comes from SQuAD passage `context` fields — natural English text
that's in-distribution for fine-tuned GPT-2. We deliberately exclude
SQuAD's Q/A formatting so the padding contains no competing question/
answer patterns that could interfere with the needle question.

Training distance distribution: uniform random in [0, --max-distance]
per example. This ensures the model sees the recall task at every
distance the eval will probe — no out-of-distribution generalization.

Examples are `<|endoftext|>`-separated in the output stream so the
corpus loader (fixed by 3cbb4f8) injects EOT-id 50256 between them,
giving the NMM a per-example reset signal.

Train and eval needle sets are disjoint by construction — the eval set
is generated first and its needles are passed as an exclude list when
generating training needles.

CLI:
    python -m scripts.prepare_needle_corpus \\
        --out-dir corpora/needle \\
        --n-train 50000 \\
        --n-eval 2500 \\
        --max-distance 3072 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from data.tokenizer import Tokenizer
from scripts.prepare_squad_corpus import (
    EOT_LITERAL,
    load_squad_eval,
    load_squad_train,
)


# Needle template. Two uppercase letters + dash + four digits.
# Entropy: 26² × 10⁴ ≈ 6.7M unique codes — collision-free at 50K + 2500 needles.
# Format chosen for unambiguous first BPE token after `A: ` (e.g., ` XK`).
NEEDLE_LETTERS = string.ascii_uppercase
NEEDLE_DIGITS = string.digits

NEEDLE_PHRASE_TEMPLATE = "The secret code is {needle}."
QUESTION_TEMPLATE = "\nQ: What is the secret code?\nA:"


@dataclass
class NeedleRecord:
    """One eval scenario: id + the needle. The eval driver builds the
    actual prompt at eval time using the saved padding pool."""
    id: str
    needle: str

    @classmethod
    def from_dict(cls, d: dict) -> "NeedleRecord":
        return cls(id=d["id"], needle=d["needle"])


NEEDLE_ALNUM = string.ascii_letters + string.digits  # 62 chars

NEEDLE_FORMATS = ("alpha", "alnum20")


def generate_needle(rng: random.Random, format: str = "alpha") -> str:
    """Random needle in the requested format.

    Formats:
      - "alpha" (default, backward-compatible): 2-letter + 4-digit code
        like 'XK-7281'. ~461 unique first BPE tokens but distribution
        heavily concentrated on common capital letters (' X', ' Y', etc.).
        Marginal-output strategy gets ~4-5% accuracy at first-token
        argmax.

      - "alnum20": 20-character random alphanumeric like 'k3F9pZ2x7vQ8aB5cD1eR'.
        ~812 unique first BPE tokens with a flatter distribution. Makes
        the marginal-output strategy structurally weaker at first-token
        scoring, since no single common prefix covers >2% of needles.
        Used in conjunction with --needle-contrastive-loss-weight to
        attack the marginal-output failure mode from both data and
        loss directions.
    """
    if format == "alpha":
        letters = "".join(rng.choices(NEEDLE_LETTERS, k=2))
        digits = "".join(rng.choices(NEEDLE_DIGITS, k=4))
        return f"{letters}-{digits}"
    if format == "alnum20":
        return "".join(rng.choices(NEEDLE_ALNUM, k=20))
    raise ValueError(
        f"Unknown needle format {format!r}. Supported: {NEEDLE_FORMATS}."
    )


def generate_unique_needles(
    n: int,
    rng: random.Random,
    exclude: set[str] | None = None,
    format: str = "alpha",
) -> list[str]:
    """Generate `n` distinct needles, avoiding any in `exclude` (used to
    keep train/eval sets disjoint). Uses a simple draw-and-reject loop —
    at ~52K needles drawn from ~6.7M-space (alpha) or 62^20-space (alnum20)
    the collision rate is essentially zero, so rejection sampling
    terminates immediately."""
    exclude = set(exclude or ())
    seen = set(exclude)
    out: list[str] = []
    # Cap the outer loop so an exhausted-space pathology fails loud
    # rather than spinning forever; in practice we're nowhere near
    # space exhaustion.
    max_draws = max(n * 100, 1000)
    draws = 0
    while len(out) < n and draws < max_draws:
        needle = generate_needle(rng, format=format)
        draws += 1
        if needle in seen:
            continue
        seen.add(needle)
        out.append(needle)
    if len(out) < n:
        raise RuntimeError(
            f"Could not generate {n} unique needles after {draws} draws "
            f"({len(out)} produced). Increase the needle space or reduce n."
        )
    return out


def precompute_padding_pool(
    passages: Sequence[str], tokenizer: Tokenizer,
) -> list[tuple[str, int]]:
    """Tokenize each padding passage once and cache the length so the
    `build_padding` inner loop can sample without re-encoding. This drops
    train-corpus generation time from ~15 minutes to ~30 seconds at the
    50K-example scale."""
    return [(p, len(tokenizer.encode(p))) for p in passages]


def build_padding(
    padding_pool: Sequence[tuple[str, int]],
    target_tokens: int,
    rng: random.Random,
) -> tuple[str, int]:
    """Concatenate random padding passages until token count meets
    `target_tokens`. Returns (joined_text, actual_token_count). Actual
    count may slightly exceed `target_tokens` because we don't truncate
    mid-passage (avoids cutting through a BPE token at the boundary).

    With target_tokens=0 returns ("", 0) — needle adjacent to question.

    Both `build_training_example` and the eval driver's prompt builder
    call THIS function so the padding shape is identical between train
    and eval distributions. Mismatches here would silently bias the
    eval against the training distribution."""
    if target_tokens <= 0:
        return "", 0
    pieces: list[str] = []
    so_far = 0
    while so_far < target_tokens:
        text, n = rng.choice(padding_pool)
        pieces.append(text)
        so_far += n
    return " ".join(pieces), so_far


def build_training_example(
    needle: str,
    padding_pool: Sequence[tuple[str, int]],
    distance: int,
    rng: random.Random,
) -> str:
    """Build one training scenario including the gold answer (so the
    LM-loss gradient covers the answer tokens). The trailing
    `<|endoftext|>` is added by `write_train_corpus`, not here."""
    needle_phrase = NEEDLE_PHRASE_TEMPLATE.format(needle=needle)
    padding, _ = build_padding(padding_pool, distance, rng)
    answer = f" {needle}"  # leading space — matches eval prompt's `A:` follow-on
    return f"{needle_phrase}\n{padding}{QUESTION_TEMPLATE}{answer}"


def generate_train_corpus(
    needles: Sequence[str],
    padding_pool: Sequence[tuple[str, int]],
    max_distance: int,
    seed: int = 0,
) -> list[str]:
    """Build one training scenario per needle. Distance is uniform random
    in [0, max_distance] per example, drawn from a per-needle local RNG so
    individual examples are independently reproducible."""
    examples: list[str] = []
    for needle in needles:
        # Per-example RNG keyed by needle for stable regeneration without
        # depending on iteration order.
        local_rng = random.Random((seed, "train", needle).__hash__())
        distance = local_rng.randint(0, max_distance)
        examples.append(build_training_example(
            needle, padding_pool, distance, local_rng,
        ))
    return examples


def generate_eval_records(
    n_records: int, rng: random.Random, format: str = "alpha",
) -> list[NeedleRecord]:
    """Generate held-out eval records. Each record is just an id + needle;
    padding is sampled at eval time from the saved pool (so the same
    record can be evaluated across multiple distance buckets)."""
    needles = generate_unique_needles(n_records, rng, format=format)
    return [
        NeedleRecord(id=f"needle_{i:05d}", needle=needle)
        for i, needle in enumerate(needles)
    ]


def write_train_corpus(examples: Sequence[str], path: Path) -> int:
    """Write EOT-literal-separated training examples. The corpus loader
    splits on `<|endoftext|>` and `encode_corpus` then injects EOT-id 50256
    between them — the signal the NMM dataloader uses to reset state.
    Returns the byte count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = f"\n{EOT_LITERAL}\n"
    text = sep.join(examples)
    path.write_text(text, encoding="utf-8")
    return len(text)


def write_eval_records(records: Sequence[NeedleRecord], path: Path) -> int:
    """Write eval records as JSON: list of {id, needle}. Returns count
    written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"id": r.id, "needle": r.needle} for r in records]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return len(payload)


def write_padding_pool(passages: Sequence[str], path: Path) -> int:
    """Write the held-out padding-passage list (eval-time use). Returns
    count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(passages)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return len(payload)


def load_padding_pool(path: Path) -> list[str]:
    """Inverse of write_padding_pool."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_eval_records(path: Path) -> list[NeedleRecord]:
    """Load eval records from JSON (mirrors load_eval_records in
    eval_qa_recall.py for symmetry)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [NeedleRecord.from_dict(r) for r in payload]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate needle-in-haystack train/eval data.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("corpora/needle"),
        help="Output directory (created if missing). Default: corpora/needle.",
    )
    parser.add_argument(
        "--n-train", type=int, default=50_000,
        help="Number of training examples. Default: 50000.",
    )
    parser.add_argument(
        "--n-eval", type=int, default=2500,
        help="Number of held-out eval needles. Default: 2500 (500 examples × "
             "5+ distance buckets).",
    )
    parser.add_argument(
        "--max-distance", type=int, default=3072,
        help="Maximum needle-to-question distance in tokens. Per-example "
             "distance is uniform random in [0, this]. Default: 3072 (covers "
             "the eval buckets and a margin past block_size=1024).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for needle generation, distance draw, padding sampling. "
             "Default: 0.",
    )
    parser.add_argument(
        "--needle-format", choices=list(NEEDLE_FORMATS), default="alpha",
        help="Needle string format. 'alpha' (default) is XX-NNNN (2 letters + "
             "4 digits) — backward-compatible. 'alnum20' is a 20-character "
             "random alphanumeric string, giving ~812 unique first BPE tokens "
             "with a flatter distribution than alpha (~461 unique, heavily "
             "concentrated on common capital-letter starts). alnum20 is "
             "designed to make the marginal-output failure mode "
             "structurally weaker — combine with the "
             "--needle-contrastive-loss-weight finetune flag for the full "
             "anti-marginal-output recipe.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer()
    rng = random.Random(args.seed)

    # Padding pools: train pool from SQuAD train passages (deduped by
    # context); eval pool from SQuAD validation passages. Disjoint by
    # SQuAD's own train/validation split, so the eval test never reuses
    # passages the model saw embedded in training prompts.
    print("[needle] loading SQuAD train as training padding pool...",
          flush=True)
    squad_train = load_squad_train(seed=args.seed)
    train_padding_text = sorted({r.context for r in squad_train})
    print(f"[needle] {len(train_padding_text)} unique training-padding "
          f"passages", flush=True)

    print("[needle] loading SQuAD validation as eval padding pool...",
          flush=True)
    squad_val = load_squad_eval(n_records=None, seed=args.seed)
    eval_padding = sorted({r.context for r in squad_val})
    print(f"[needle] {len(eval_padding)} unique eval-padding passages",
          flush=True)

    # Precompute token lengths on the train padding pool so the inner
    # build_padding loop is just a dict lookup, not a tokenizer call.
    print("[needle] precomputing training-padding token lengths...",
          flush=True)
    train_padding = precompute_padding_pool(train_padding_text, tokenizer)

    # Generate eval needles FIRST so we can exclude them when generating
    # train needles — keeps train/eval sets disjoint at the needle level.
    print(f"[needle] generating {args.n_eval} eval needles "
          f"(format={args.needle_format})...", flush=True)
    eval_records = generate_eval_records(
        args.n_eval, rng, format=args.needle_format,
    )
    eval_needle_set = {r.needle for r in eval_records}

    print(f"[needle] generating {args.n_train} training needles...",
          flush=True)
    train_needles = generate_unique_needles(
        args.n_train, rng, exclude=eval_needle_set,
        format=args.needle_format,
    )

    print(f"[needle] building {args.n_train} training scenarios...",
          flush=True)
    examples = generate_train_corpus(
        train_needles, train_padding, args.max_distance, seed=args.seed,
    )

    # Write out everything.
    train_path = args.out_dir / "needle_train.txt"
    n_chars = write_train_corpus(examples, train_path)
    print(f"[needle] wrote {train_path} ({n_chars / 1e6:.1f} MB)", flush=True)

    eval_path = args.out_dir / "needle_eval.json"
    n_records = write_eval_records(eval_records, eval_path)
    print(f"[needle] wrote {eval_path} ({n_records} records)", flush=True)

    padding_path = args.out_dir / "needle_eval_padding.json"
    n_pad = write_padding_pool(eval_padding, padding_path)
    print(f"[needle] wrote {padding_path} ({n_pad} passages)", flush=True)


if __name__ == "__main__":
    main()
