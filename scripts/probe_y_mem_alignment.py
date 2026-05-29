"""Diagnostic: does the memory pathway's y_mem at the question position
align with the correct-answer token at long distance?

This is the same probe we ran on the NMM diagnostic series. For each
of N random needle scenarios at the configured distance:

  1. Build the prompt: `needle_phrase\\n<padding>question`.
  2. Run `prepare_decode_chunked` — same path as `scripts.eval_needle`.
  3. Capture y_mem at the LAST position of the last chunk (just before
     the answer would be generated) from a target memory-bearing block.
  4. Project that y_mem through `ln_f + tied wte^T` to get a vocab
     prediction.
  5. Compare against the correct first-answer token's embedding via:
       a. Cosine similarity (memory's directional alignment with the right token)
       b. Logit-rank of the correct token under that projection
       c. Top-5 tokens the memory would predict

If the memory pathway has learned to retrieve, cos(y_mem, e_correct)
should be meaningfully positive (>> 0) and the correct token should be
near the top of the memory-projected vocab. If memory writes input-
dependent noise without retrieval structure, both stay at noise floor.

Usage:
    python -m scripts.probe_y_mem_alignment \\
        --checkpoint ckpts/needle_delta_product/latest.pt \\
        --eval-data corpora/needle/needle_eval.json \\
        --padding-pool corpora/needle/needle_eval_padding.json \\
        --distance 2048 --n 20
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from cli.train import install_y_mem_capture, load_checkpoint
from data.tokenizer import Tokenizer
from model import _unwrap
from scripts.eval_needle import (
    NeedleRecord,
    _load_model,
    _needle_first_token,
    build_needle_prompt,
    load_eval_records,
    load_padding_pool,
    precompute_padding_pool,
)


@torch.no_grad()
def probe_one(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    target: NeedleRecord,
    padding_pool,
    distance: int,
    rng: random.Random,
    capture: dict,
) -> dict:
    """Run one trial and return cos / rank / top-5 against the correct
    first-answer token. Same prompt/forward path as `evaluate_one`."""
    prompt_text, prompt_len, actual_distance = build_needle_prompt(
        target.needle, padding_pool, distance, tokenizer, rng,
    )
    ids_t = torch.tensor(
        tokenizer.encode(prompt_text), dtype=torch.long, device=device,
    ).unsqueeze(0)

    # Forward — the hook captures y_mem from the target NMM block's
    # forward_chunk on every chunk; we want the LAST chunk's last
    # position, which is what cache["last_logits"] also reads from.
    capture.pop("y_mem", None)
    cache = model.prepare_decode_chunked(ids_t)
    pred_token = int(cache["last_logits"].squeeze(1).argmax(-1).item())
    expected_token = _needle_first_token(target.needle, tokenizer)

    real = _unwrap(model)
    y_mem_chunk = capture["y_mem"]  # [B, T_aug, n_embd] from the last chunk
    # Strip the persistent_mem prefix and take the last real-token position.
    N_p = real.config.nmm_n_persistent
    if N_p > 0:
        y_mem_real = y_mem_chunk[:, N_p:, :]
    else:
        y_mem_real = y_mem_chunk
    y_mem_last = y_mem_real[:, -1, :]  # [B, n_embd], last position only

    # Project through ln_f + tied LM head — the read pathway that
    # `compute_aux_retrieval_loss` uses.
    aux_logits = real.ln_f(y_mem_last) @ real.wte.weight.T  # [B, vocab]
    aux_logits = aux_logits.squeeze(0).float()  # [vocab]

    # Cosine of (ln_f-normed y_mem) with the correct token's embedding.
    y_normed = real.ln_f(y_mem_last).squeeze(0).float()  # [n_embd]
    e_correct = real.wte.weight[expected_token].float()  # [n_embd]
    cos_correct = torch.nn.functional.cosine_similarity(
        y_normed, e_correct, dim=0,
    ).item()

    # Rank of correct token under the memory-projected logits.
    sorted_idx = torch.argsort(aux_logits, descending=True)
    rank_correct = int((sorted_idx == expected_token).nonzero().item())

    # Top-5 memory predictions.
    top5_idx = sorted_idx[:5].tolist()
    top5_text = [tokenizer.decode([i]) for i in top5_idx]

    return {
        "id": target.id,
        "needle": target.needle,
        "distance": distance,
        "actual_distance": actual_distance,
        "pred_token": pred_token,
        "expected_token": expected_token,
        "lm_correct": pred_token == expected_token,
        "cos_y_mem_vs_correct": cos_correct,
        "mem_rank_of_correct": rank_correct,
        "mem_top5_tokens": top5_idx,
        "mem_top5_text": top5_text,
        "expected_text": tokenizer.decode([expected_token]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--padding-pool", type=Path, required=True)
    parser.add_argument("--distance", type=int, default=2048)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-layer", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()

    print(f"[probe] loading checkpoint {args.checkpoint}...",
          file=sys.stderr, flush=True)
    model, config = _load_model(args.checkpoint, device)
    print(f"[probe] model: n_layer={config.n_layer} n_embd={config.n_embd} "
          f"memory_type={getattr(config, 'memory_type', 'nmm')}",
          file=sys.stderr, flush=True)

    print(f"[probe] loading eval data...", file=sys.stderr, flush=True)
    records = load_eval_records(args.eval_data)
    padding_pool_raw = load_padding_pool(args.padding_pool)
    padding_pool = precompute_padding_pool(padding_pool_raw, tokenizer)

    rng = random.Random(args.seed)
    chosen = rng.sample(records, k=args.n)

    capture, uninstall = install_y_mem_capture(
        model, target_layer=args.target_layer,
    )
    print(f"[probe] hooked y_mem capture on block {capture['target_layer']}; "
          f"running {args.n} trials at distance={args.distance}",
          file=sys.stderr, flush=True)

    results = []
    for i, target in enumerate(chosen):
        rec = probe_one(
            model, tokenizer, device, target, padding_pool,
            args.distance, rng, capture,
        )
        results.append(rec)
        if (i + 1) % max(1, args.n // 10) == 0:
            print(f"[probe] {i+1}/{args.n}", file=sys.stderr, flush=True)
    uninstall()

    # Aggregate.
    cos_vals = [r["cos_y_mem_vs_correct"] for r in results]
    ranks = [r["mem_rank_of_correct"] for r in results]
    lm_correct = [r["lm_correct"] for r in results]

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    print(f"\n=== y_mem alignment probe @ distance={args.distance} (n={args.n}) ===")
    print(f"target block: {capture['target_layer']}")
    print()
    print(f"LM accuracy on this slice: {sum(lm_correct)}/{args.n} = "
          f"{sum(lm_correct) / args.n:.2%}")
    print()
    print(f"cos(y_mem, e_correct):")
    print(f"  mean   {sum(cos_vals) / len(cos_vals):+.4f}")
    print(f"  median {median(cos_vals):+.4f}")
    print(f"  min    {min(cos_vals):+.4f}")
    print(f"  max    {max(cos_vals):+.4f}")
    pos = sum(1 for c in cos_vals if c > 0.05)
    print(f"  > 0.05 (meaningfully positive): {pos}/{args.n}")
    print()
    print(f"Rank of correct token under memory-projection (vocab=50257):")
    print(f"  mean   {sum(ranks) / len(ranks):.0f}")
    print(f"  median {median(ranks)}")
    print(f"  min    {min(ranks)}")
    print(f"  max    {max(ranks)}")
    in_top1 = sum(1 for r in ranks if r == 0)
    in_top5 = sum(1 for r in ranks if r < 5)
    in_top100 = sum(1 for r in ranks if r < 100)
    print(f"  rank=0 (top-1):   {in_top1}/{args.n}")
    print(f"  rank<5 (top-5):   {in_top5}/{args.n}")
    print(f"  rank<100:         {in_top100}/{args.n}")
    print()
    print("Per-trial sample (first 5):")
    for r in results[:5]:
        print(f"  needle={r['needle']}  expect={r['expected_text']!r}  "
              f"pred={tokenizer.decode([r['pred_token']])!r}  "
              f"cos={r['cos_y_mem_vs_correct']:+.4f}  "
              f"mem_rank={r['mem_rank_of_correct']}  "
              f"mem_top5={r['mem_top5_text']}")


if __name__ == "__main__":
    main()
