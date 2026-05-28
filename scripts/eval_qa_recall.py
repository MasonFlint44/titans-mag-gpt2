"""QA-recall evaluation: measure answer recall vs. context distance.

Given a trained checkpoint and the SQuAD eval JSON (from
`scripts.prepare_squad_corpus`), this script builds prompts of the form:

    [P] {target_passage}
    [P] {distractor_1}
    Q: {distractor_q1}
    A: {distractor_a1}
    [P] {distractor_2}
    ...
    Q: {target_question}
    A:

with distractor padding sized so that the *target* passage sits exactly
`distance` tokens before the final question. We then greedily decode one
token from `cache["last_logits"]` and check whether that token matches the
first BPE token of any gold answer alias.

Why single-token / first-token scoring (rather than multi-token greedy +
substring match):

  - At long distances (target_passage_tokens + distance > block_size), the
    prompt exceeds block_size and `prepare_decode_chunked` only lets us
    sample ONE token before forward_step would wpe-OOB. To keep the
    scoring metric consistent across distance buckets — and the experiment
    chart interpretable — we use the same first-token-argmax everywhere.
  - First-token argmax is also a strict recall test: the model must
    commit to the gold answer's leading token as its top-1 prediction
    right after `A:`. No partial-credit drift through the rest of the
    answer hides a wrong recall.

Distance buckets span both the within-attention regime (0..block_size) and
the beyond-attention regime (>block_size) so the chart shows where vanilla
GPT-2 collapses and where TITANS' NMM holds.

CLI:
    python -m scripts.eval_qa_recall \\
        --checkpoint ckpts/titans/latest.pt \\
        --eval-data corpora/squad/squad_eval.json \\
        --out results/titans.json \\
        --n-examples 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from config import TitansConfig
from data.tokenizer import Tokenizer


# Distance buckets (tokens between target passage end and the final question).
# Chosen so the chart spans both regimes:
#   0..512: comfortably within attention's reach — sanity baseline (both
#     models should score high modulo locality bias).
#   1024:   right at block_size; attention only barely covers target.
#   2048, 3072: well beyond attention — only NMM recall can help.
DEFAULT_DISTANCES = [0, 256, 512, 1024, 2048, 3072]


@dataclass
class EvalRecord:
    """Mirror of the JSON record produced by scripts.prepare_squad_corpus."""
    id: str
    title: str
    context: str
    question: str
    answers: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "EvalRecord":
        return cls(
            id=d["id"], title=d["title"], context=d["context"],
            question=d["question"], answers=list(d["answers"]),
        )


def load_eval_records(path: Path) -> list[EvalRecord]:
    """Load the held-out eval JSON written by scripts.prepare_squad_corpus."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalRecord.from_dict(r) for r in payload]


def _format_distractor(d: EvalRecord) -> str:
    """Format a distractor as a bare passage block (NO Q/A pair). Matches
    the training scenario shape produced by `prepare_squad_corpus.build_
    recall_scenarios`: passages clustered at the start, exactly one Q/A
    at the end. Including distractor Q/A here would create a different
    distribution from training and could leak answer-pattern signal."""
    return f"[P] {d.context}\n"


def build_qa_prompt(
    target: EvalRecord,
    distractor_pool: Sequence[EvalRecord],
    distance: int,
    tokenizer: Tokenizer,
    rng: random.Random,
) -> tuple[str, int, int]:
    """Build a prompt where the target passage ends `distance` tokens before
    the final question.

    Layout (in token order, matching training scenario shape):
        [P] {target_passage}
        [P] {distractor_1_passage}
        [P] {distractor_2_passage}
        ...
        Q: {target_question}
        A:

    Distractors are appended AFTER the target until the byte-pair token gap
    between target's end and the final `Q:` reaches at least `distance`.
    Distractors are drawn from `distractor_pool` (which must exclude the
    target AND any record sharing the target's title — caller filters
    before passing) without replacement; if the pool is exhausted before
    reaching `distance`, we stop and return whatever we packed.

    Returns:
      prompt_text:     the full string to encode and feed the model.
      prompt_len:      total BPE token count of `prompt_text`.
      actual_distance: tokens between target_passage end and the final
                       `Q:` start (≤ `distance` if pool exhausted).
    """
    target_block = f"[P] {target.context}\n"
    # Q-prompt has NO leading newline — the trailing `\n` from the last
    # passage block provides the separator. Adding one here would double
    # the newline and diverge from training scenarios.
    q_prompt = f"Q: {target.question}\nA:"

    # Build distractor padding by drawing without replacement until the
    # accumulated padding token count meets the target distance. Shuffle
    # the pool so each target sees a different distractor order — avoids
    # any per-position memorization at eval time.
    shuffled = list(distractor_pool)
    rng.shuffle(shuffled)

    padding_chunks: list[str] = []
    padding_tokens = 0
    for d in shuffled:
        if padding_tokens >= distance:
            break
        chunk = _format_distractor(d)
        n = len(tokenizer.encode(chunk))
        padding_chunks.append(chunk)
        padding_tokens += n

    padding_text = "".join(padding_chunks)
    full = target_block + padding_text + q_prompt
    prompt_len = len(tokenizer.encode(full))
    return full, prompt_len, padding_tokens


def _first_token_set(answers: Sequence[str], tokenizer: Tokenizer) -> set[int]:
    """Token-id set the model is allowed to emit to score "correct".

    Considers two leading-context variants per gold alias:
      - leading space ("A: Denver"): matches our training format where
        the answer appears right after `A: ` — tokenizer emits ` Denver`
        as the first answer token.
      - no leading space ("ADenver"): defensive fallback for BPE quirks
        where the leading space might fold differently.

    Token-id set, not list, so callers can do a fast `pred in set` check.
    """
    tokens: set[int] = set()
    for ans in answers:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + ans)
            if ids:
                tokens.add(ids[0])
    return tokens


@torch.no_grad()
def evaluate_one(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    target: EvalRecord,
    distractor_pool: Sequence[EvalRecord],
    distance: int,
    rng: random.Random,
    max_prompt_tokens: int | None = None,
) -> dict:
    """Score one (target, distance) pair. Returns a record dict suitable
    for the eval JSON output.

    `max_prompt_tokens`: optional hard cap. Prompts longer than this are
    truncated from the LEFT (preserving target_passage + nearby distractors;
    dropping the oldest distractor padding). Used by tests to keep smoke
    runs fast — production runs leave it None and let `prepare_decode_chunked`
    handle long prompts via chunked warm-up.

    Mode is captured-and-restored by the surrounding `evaluate` driver via
    try/finally (pattern).
    """
    prompt_text, prompt_len, actual_distance = build_qa_prompt(
        target, distractor_pool, distance, tokenizer, rng,
    )

    ids = tokenizer.encode(prompt_text)
    if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:
        # Keep the FIRST target_passage tokens (so target stays in chunked
        # warm-up) and the LAST tokens up to the cap. This preserves the
        # experimental contract: target is N tokens before the question.
        # Approximate: drop tokens from the middle (distractor padding).
        # Specifically, keep target_passage + final tail of length
        # (max_prompt_tokens - len(target_passage)).
        target_block_ids = tokenizer.encode(f"[P] {target.context}\n")
        tail_budget = max_prompt_tokens - len(target_block_ids)
        if tail_budget > 0:
            ids = target_block_ids + ids[-tail_budget:]
        else:
            ids = ids[-max_prompt_tokens:]

    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    cache = model.prepare_decode_chunked(ids_t)
    pred_token = int(cache["last_logits"].squeeze(1).argmax(-1).item())
    pred_text = tokenizer.decode([pred_token])

    valid_tokens = _first_token_set(target.answers, tokenizer)
    correct = pred_token in valid_tokens

    return {
        "id": target.id,
        "distance": distance,
        "actual_distance": actual_distance,
        "prompt_len": prompt_len,
        "predicted_token": pred_token,
        "predicted_text": pred_text,
        "expected_answers": target.answers,
        "correct": correct,
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    records: Sequence[EvalRecord],
    distances: Sequence[int] = DEFAULT_DISTANCES,
    n_examples: int | None = None,
    seed: int = 0,
    max_prompt_tokens: int | None = None,
    progress: bool = False,
) -> dict:
    """Run the full sweep across distance buckets and return aggregated results.

    Each (record, distance) pair counts as one trial. With `n_examples=500`
    and 6 distance buckets, that's 3000 trials per checkpoint. Eval is fast
    (one forward, one argmax per trial) — typically <10 minutes on consumer
    GPU.

    Same record is reused across distance buckets (different padding each
    time): controls for inter-example variance, so the per-bucket curves
    are comparable. The distractor pool draws from records that AREN'T the
    current target (no leak — a target can't be its own distractor).
    """
    was_training = model.training
    model.eval()
    try:
        rng = random.Random(seed)

        if n_examples is not None and n_examples < len(records):
            # Subsample once up front, then use the same subsample for every
            # distance bucket. Keeps the per-bucket scores comparable.
            sample = rng.sample(list(records), n_examples)
        else:
            sample = list(records)

        results = []
        buckets: dict[int, dict] = {
            d: {"correct": 0, "total": 0} for d in distances
        }
        n_trials = len(sample) * len(distances)
        t0 = time.time()

        for i, target in enumerate(sample):
            # Distractor pool excludes the target AND any record sharing
            # the target's Wikipedia title. Without the title filter, a
            # distractor about the same article (e.g., another Q/A pair
            # from the "Beyoncé" passage set) could let the model answer
            # via topic-matching rather than cross-passage recall —
            # exactly the failure mode the architecture test is supposed
            # to expose. Matches the topic-disjoint sampling used in
            # training scenarios.
            pool = [
                r for r in records
                if r.id != target.id and r.title != target.title
            ]

            for distance in distances:
                # Per-distance rng makes the distractor shuffle reproducible
                # without coupling distances together — same target gets
                # different distractor orders per bucket.
                local_rng = random.Random((seed, target.id, distance).__hash__())
                rec = evaluate_one(
                    model, tokenizer, device, target, pool, distance,
                    local_rng, max_prompt_tokens=max_prompt_tokens,
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
                        eta = (n_trials - done) / rate if rate > 0 else 0
                        print(
                            f"[eval] {done}/{n_trials} trials "
                            f"({100*done/n_trials:.0f}%) — "
                            f"{rate:.1f} trials/s, eta {eta/60:.1f} min",
                            file=sys.stderr, flush=True,
                        )

        for d, b in buckets.items():
            b["accuracy"] = b["correct"] / b["total"] if b["total"] else 0.0

        return {
            "distances": list(distances),
            "n_examples": len(sample),
            "n_trials": len(results),
            "buckets": buckets,
            "results": results,
        }
    finally:
        if was_training:
            model.train()


def _load_model(checkpoint_path: Path, device: torch.device):
    """Rebuild the model from a training checkpoint.

    Checkpoints store `config` as `dataclasses.asdict(config)` (see
    `train.save_checkpoint`). We use `TitansConfig.from_dict` so saved
    checkpoints from older schema versions (with knobs we've since
    removed) still load cleanly — `from_dict` silently drops keys in
    `_REMOVED_CONFIG_KEYS` and raises loud on truly unknown keys.
    """
    from model.titans_gpt2 import TitansMAGGPT2
    from model import _unwrap
    from cli.train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path, device=device)
    if "config" not in ckpt:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} lacks a 'config' key. "
            f"Pass a checkpoint saved by `save_checkpoint` from cli.train.py."
        )
    config = TitansConfig.from_dict(ckpt["config"])
    model = TitansMAGGPT2(config).to(device)
    # train.py stores under "state_dict" (//path); generate.py
    # used "model" in an earlier draft — accept either for forward compat.
    state = ckpt.get("state_dict", ckpt.get("model"))
    if state is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has neither 'state_dict' nor "
            f"'model' keys. Pass a checkpoint saved by train.py."
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
        help="Path to squad_eval.json from scripts.prepare_squad_corpus.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output JSON path (per-bucket accuracy + per-trial records).",
    )
    parser.add_argument(
        "--n-examples", type=int, default=500,
        help="Targets to evaluate (each scored across every distance bucket). "
             "Default: 500 → ~3000 trials at 6 buckets, ±2%% CI per bucket.",
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=DEFAULT_DISTANCES,
        help=f"Distance buckets in tokens. Default: {DEFAULT_DISTANCES}.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for example subsample + distractor shuffling.",
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

    results = evaluate(
        model, tokenizer, device, records,
        distances=args.distances,
        n_examples=args.n_examples,
        seed=args.seed,
        progress=args.progress,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_data": str(args.eval_data),
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
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"[eval] wrote {args.out}", file=sys.stderr, flush=True)
    for d, b in results["buckets"].items():
        print(f"[eval]   distance={d:>5d}  "
              f"accuracy={b['accuracy']:.3f}  "
              f"({b['correct']}/{b['total']})",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
