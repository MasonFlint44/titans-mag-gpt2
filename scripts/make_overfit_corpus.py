"""Build a single-needle overfit corpus for capacity-diagnostic experiments.

The test: take ONE needle example with a controlled cross-chunk distance,
repeat it N times with EOT separators, train on that corpus. Whether the
model can memorize this single (needle, answer) pair tells us if the
architecture has the expressive capacity to do cross-chunk recall at all.

Usage:
    python -m scripts.make_overfit_corpus \\
        --out corpora/overfit_test/single_needle.txt \\
        --eval-padding corpora/needle/needle_eval_padding.json \\
        --needle XX-9999 --distance 1500 --n-copies 20 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.tokenizer import Tokenizer
from scripts.prepare_squad_corpus import EOT_LITERAL

NEEDLE_PHRASE_TEMPLATE = "The secret code is {needle}."
QUESTION_TEMPLATE = "\nQ: What is the secret code?\nA:"


def build_padding_to_distance(
    pool: list[str], target_tokens: int, tok: Tokenizer, rng: random.Random,
) -> str:
    """Concatenate padding passages until we hit `target_tokens` BPE tokens.

    Pool is a list of plain strings (per needle_eval_padding.json format).
    We tokenize each candidate and greedy-fill keeping the running count
    under target.
    """
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    chosen: list[str] = []
    running_tokens = 0
    for idx in indices:
        text_i = pool[idx]
        n = len(tok.encode(text_i))
        if running_tokens + n > target_tokens:
            continue
        chosen.append(text_i)
        running_tokens += n
        if running_tokens >= target_tokens - 100:
            break
    return " ".join(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--eval-padding",
        required=True,
        help="Path to needle_eval_padding.json (or any precomputed pool).",
    )
    parser.add_argument(
        "--needles",
        nargs="+",
        default=["XJ-9871"],
        help="One or more needles. When >1 supplied, examples alternate "
             "between them — constant-output is no longer a winning strategy "
             "for the optimizer, so it has to use the memory pathway.",
    )
    parser.add_argument("--distance", type=int, default=1500)
    parser.add_argument(
        "--n-copies",
        type=int,
        default=20,
        help="How many times to repeat each needle. With K needles, total "
             "examples = K * n_copies. Need enough to span multiple chunks; "
             "at distance=1500, each example is ~1530 tokens.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tok = Tokenizer()

    with open(args.eval_padding) as f:
        pool = json.load(f)

    # One example per needle. We reuse the SAME padding for all needles
    # so the only thing varying between examples is the needle and the
    # answer. This makes "use the memory pathway to look up the needle"
    # the unique low-loss solution: constant output can't predict the
    # right answer when answers differ across examples.
    padding_text = build_padding_to_distance(pool, args.distance, tok, rng)
    examples = []
    for needle in args.needles:
        needle_phrase = NEEDLE_PHRASE_TEMPLATE.format(needle=needle)
        answer = f" {needle}"
        examples.append(
            f"{needle_phrase}\n{padding_text}{QUESTION_TEMPLATE}{answer}"
        )

    actual_distance = (
        len(tok.encode(examples[0]))
        - len(tok.encode(NEEDLE_PHRASE_TEMPLATE.format(needle=args.needles[0])))
        - len(tok.encode(QUESTION_TEMPLATE))
        - len(tok.encode(f" {args.needles[0]}"))
    )

    # Round-robin interleave: alternate needles so the model can't memorize
    # a "first N examples use needle A" pattern. Each needle appears
    # `n_copies` times, interleaved with the others.
    interleaved = []
    for copy_i in range(args.n_copies):
        for ex in examples:
            interleaved.append(ex)

    sep = f"\n{EOT_LITERAL}\n"
    corpus = sep.join(interleaved)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(corpus, encoding="utf-8")

    full_tokens = tok.encode(corpus)
    print(f"[overfit] needles={args.needles} target_distance={args.distance}")
    print(f"[overfit] actual_distance_of_one_example={actual_distance}")
    print(f"[overfit] tokens_per_example≈{len(tok.encode(examples[0]))}")
    print(f"[overfit] n_copies_per_needle={args.n_copies}")
    print(f"[overfit] total_examples={len(interleaved)}")
    print(f"[overfit] total_tokens={len(full_tokens)}")
    print(f"[overfit] chunks at chunk_size=1024: {len(full_tokens) // 1024}")
    print(f"[overfit] wrote {args.out}")


if __name__ == "__main__":
    main()
