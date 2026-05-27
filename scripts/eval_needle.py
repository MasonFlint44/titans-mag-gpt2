"""Needle-in-haystack evaluation: measure needle recall vs. context distance.

Given a checkpoint, a needle-eval JSON (from
`scripts.prepare_needle_corpus`), and the held-out padding pool, this
script builds prompts of the form:

    The secret code is XK-7281.
    [passage text padding totaling ≈ `distance` tokens]
    Q: What is the secret code?
    A:

with padding sized so the needle phrase sits exactly `distance` tokens
before the final `Q:`. We greedily decode one token from
`cache["last_logits"]` and check whether it matches the first BPE token
of the needle (after the leading space — the same token the training
target sees right after `A: `).

Why this beats SQuAD QA-recall as an architecture test:
  - The "thing to remember" is high-entropy and a single BPE token, so
    first-token-argmax measures recall directly (no LM-pattern noise).
  - Padding contains no Q/A patterns to compete with the target
    question — every (Q, A) signal in the prompt is the needle's.
  - Training data (from prepare_needle_corpus) covers the same distance
    range we evaluate at, so we're not testing OOD generalization.

Past `block_size=1024`, vanilla GPT-2 has no architectural mechanism to
carry information across chunks — its accuracy MUST collapse. If TITANS
shows any signal above chance past block_size, that's evidence the NMM
is doing what it's designed for.

CLI:
    python -m scripts.eval_needle \\
        --checkpoint ckpts/titans/latest.pt \\
        --eval-data corpora/needle/needle_eval.json \\
        --padding-pool corpora/needle/needle_eval_padding.json \\
        --out results/needle_titans.json \\
        --n-examples 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from scripts.prepare_needle_corpus import (
    NEEDLE_PHRASE_TEMPLATE,
    NeedleRecord,
    QUESTION_TEMPLATE,
    build_padding,
    load_eval_records,
    load_padding_pool,
    precompute_padding_pool,
)


# Same buckets as eval_qa_recall — keeps the two charts directly comparable
# on the same axes.
#   0..512:   comfortably within attention's reach — both models should score
#             high (the needle is right there).
#   1024:     at block_size; attention barely reaches.
#   2048,3072: well beyond attention — only NMM recall can help.
DEFAULT_DISTANCES = [0, 256, 512, 1024, 2048, 3072]


def _needle_first_token(needle: str, tokenizer: Tokenizer) -> int:
    """First BPE token of the needle as it appears after `A: ` (with
    leading space). The training target also has a leading space, so
    train and eval scoring agree on the same token id."""
    ids = tokenizer.encode(f" {needle}")
    if not ids:
        raise ValueError(f"Needle {needle!r} tokenized to empty list.")
    return ids[0]


def build_needle_prompt(
    needle: str,
    padding_pool: Sequence[tuple[str, int]],
    distance: int,
    tokenizer: Tokenizer,
    rng: random.Random,
) -> tuple[str, int, int]:
    """Build an eval prompt (without the answer): needle phrase + ≈distance
    tokens of padding + question + `A:`. Returns
    (prompt_text, prompt_len, actual_distance).

    Calls the same `build_padding` helper as `build_training_example` so
    the eval prompt shape matches the training distribution exactly —
    given identical RNG state, eval_prompt + f' {needle}' == training_example.
    Pinned by `test_eval_prompt_followed_by_needle_matches_training_format`."""
    needle_phrase = NEEDLE_PHRASE_TEMPLATE.format(needle=needle)
    padding_text, actual_distance = build_padding(padding_pool, distance, rng)
    full = f"{needle_phrase}\n{padding_text}{QUESTION_TEMPLATE}"
    prompt_len = len(tokenizer.encode(full))
    return full, prompt_len, actual_distance


@torch.no_grad()
def evaluate_one(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    target: NeedleRecord,
    padding_pool: Sequence[tuple[str, int]],
    distance: int,
    rng: random.Random,
) -> dict:
    """Score one (needle, distance) pair. Returns a record dict for the
    eval JSON. eval-mode contract assumed by the surrounding driver."""
    prompt_text, prompt_len, actual_distance = build_needle_prompt(
        target.needle, padding_pool, distance, tokenizer, rng,
    )
    ids_t = torch.tensor(
        tokenizer.encode(prompt_text), dtype=torch.long, device=device,
    ).unsqueeze(0)

    cache = model.prepare_decode_chunked(ids_t)
    pred_token = int(cache["last_logits"].squeeze(1).argmax(-1).item())
    pred_text = tokenizer.decode([pred_token])

    expected_token = _needle_first_token(target.needle, tokenizer)
    correct = pred_token == expected_token

    return {
        "id": target.id,
        "needle": target.needle,
        "distance": distance,
        "actual_distance": actual_distance,
        "prompt_len": prompt_len,
        "predicted_token": pred_token,
        "predicted_text": pred_text,
        "expected_token": expected_token,
        "correct": correct,
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    records: Sequence[NeedleRecord],
    padding_pool_raw: Sequence[str],
    distances: Sequence[int] = DEFAULT_DISTANCES,
    n_examples: int | None = None,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    """Run the full sweep across distance buckets. Each (record, distance)
    pair counts as one trial.

    Same record evaluated across all distance buckets — controls for
    inter-needle variance so per-bucket curves are directly comparable.
    Per-(record, distance) RNG seeded by (seed, record.id, distance) so
    distractor padding is reproducible and decoupled across buckets.

    `padding_pool_raw`: list of passage strings. Tokenized once up front
    and passed as (text, n_tokens) tuples to the per-trial prompt builder
    — avoids re-tokenizing the same passages for every trial."""
    was_training = model.training
    model.eval()
    try:
        rng = random.Random(seed)

        if n_examples is not None and n_examples < len(records):
            sample = rng.sample(list(records), n_examples)
        else:
            sample = list(records)

        # Tokenize the padding pool once (~1500 passages × <1ms each = a
        # one-time cost of seconds, not minutes-per-trial).
        padding_pool = precompute_padding_pool(padding_pool_raw, tokenizer)

        results: list[dict] = []
        buckets: dict[int, dict] = {
            d: {"correct": 0, "total": 0} for d in distances
        }
        n_trials = len(sample) * len(distances)
        t0 = time.time()

        for target in sample:
            for distance in distances:
                local_rng = random.Random(
                    (seed, target.id, distance).__hash__(),
                )
                rec = evaluate_one(
                    model, tokenizer, device, target,
                    padding_pool, distance, local_rng,
                )
                results.append(rec)
                bucket = buckets[distance]
                bucket["total"] += 1
                if rec["correct"]:
                    bucket["correct"] += 1

                if progress:
                    done = len(results)
                    if done % max(1, n_trials // 20) == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta_s = (n_trials - done) / rate if rate > 0 else 0
                        print(
                            f"[eval] {done}/{n_trials} trials "
                            f"({100 * done // n_trials}%) — "
                            f"{rate:.1f} trials/s, "
                            f"eta {eta_s/60:.1f} min",
                            file=sys.stderr, flush=True,
                        )

        for d in distances:
            tot = buckets[d]["total"]
            buckets[d]["accuracy"] = (
                buckets[d]["correct"] / tot if tot else 0.0
            )

        return {
            "n_examples": len(sample),
            "n_trials": n_trials,
            "distances": list(distances),
            "buckets": buckets,
            "results": results,
        }
    finally:
        if was_training:
            model.train()


def _load_model(checkpoint_path: Path, device: torch.device):
    """Same loader pattern as eval_qa_recall._load_model — rebuilds the
    model from the checkpoint's saved config and loads weights via
    `_unwrap` (strips torch.compile/DDP prefixes)."""
    from model.titans_gpt2 import TitansMAGGPT2
    from model import _unwrap
    from train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path, device=device)
    if "config" not in ckpt:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} lacks a 'config' key. "
            f"Pass a checkpoint saved by `save_checkpoint` from train.py."
        )
    config = TitansConfig.from_dict(ckpt["config"])
    model = TitansMAGGPT2(config).to(device)
    state = ckpt.get("state_dict", ckpt.get("model"))
    if state is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has neither 'state_dict' nor "
            f"'model' keys."
        )
    model.load_state_dict(_unwrap(state))
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a training checkpoint (step_*.pt or latest.pt).",
    )
    parser.add_argument(
        "--eval-data", type=Path, required=True,
        help="Path to needle_eval.json from scripts.prepare_needle_corpus.",
    )
    parser.add_argument(
        "--padding-pool", type=Path, required=True,
        help="Path to needle_eval_padding.json — held-out passages used "
             "as padding at eval time.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output JSON path (per-bucket accuracy + per-trial records).",
    )
    parser.add_argument(
        "--n-examples", type=int, default=500,
        help="Records to evaluate (each scored across every distance "
             "bucket). Default: 500.",
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=DEFAULT_DISTANCES,
        help=f"Distance buckets in tokens. Default: {DEFAULT_DISTANCES}.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed. Default: 0.",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Print periodic progress lines to stderr (every ~5%% of trials).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()

    print(f"[eval] loading checkpoint {args.checkpoint}...",
          file=sys.stderr, flush=True)
    model, config = _load_model(args.checkpoint, device)
    print(f"[eval] model: n_layer={config.n_layer} n_embd={config.n_embd} "
          f"block_size={config.block_size} "
          f"vanilla={'yes' if config.nmm_layer_indices == [] else 'no'}",
          file=sys.stderr, flush=True)

    print(f"[eval] loading eval data {args.eval_data}...",
          file=sys.stderr, flush=True)
    records = load_eval_records(args.eval_data)
    print(f"[eval] {len(records)} records loaded", file=sys.stderr, flush=True)

    print(f"[eval] loading padding pool {args.padding_pool}...",
          file=sys.stderr, flush=True)
    padding_pool_raw = load_padding_pool(args.padding_pool)
    print(f"[eval] {len(padding_pool_raw)} padding passages",
          file=sys.stderr, flush=True)

    results = evaluate(
        model, tokenizer, device, records, padding_pool_raw,
        distances=args.distances,
        n_examples=args.n_examples,
        seed=args.seed,
        progress=args.progress,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_data": str(args.eval_data),
        "padding_pool": str(args.padding_pool),
        "config": {
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vanilla_gpt2": config.nmm_layer_indices == [],
        },
        "seed": args.seed,
        **results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"[eval] wrote {args.out}", file=sys.stderr, flush=True)
    for d, b in results["buckets"].items():
        print(f"[eval]   distance={d:>5d}  "
              f"accuracy={b['accuracy']:.3f}  "
              f"({b['correct']}/{b['total']})",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
