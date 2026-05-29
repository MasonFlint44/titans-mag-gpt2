"""Single-example overfit evaluation.

Companion to `scripts.make_overfit_corpus`. Loads the trained checkpoint
and runs greedy decode on the same needle prompt the model was trained
on. Reports whether the model memorized the cross-chunk recall.

The metric is direct: does `argmax(logits[answer_position])` equal the
first BPE token of " {needle}"? If yes, the architecture's expressive
capacity is sufficient to do cross-chunk retrieval. If no, then no
training budget / data distribution will fix it — the architecture
itself can't traverse the chunk boundary at this scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cli.train import load_checkpoint
from data.tokenizer import Tokenizer
from scripts.make_overfit_corpus import (
    NEEDLE_PHRASE_TEMPLATE, QUESTION_TEMPLATE,
    build_padding_to_distance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval-padding",
        required=True,
        help="Same padding pool used to build the train corpus.",
    )
    parser.add_argument("--needle", default="XJ-9871")
    parser.add_argument("--distance", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    import json
    import random

    rng = random.Random(args.seed)
    tok = Tokenizer()
    with open(args.eval_padding) as f:
        pool = json.load(f)

    needle_phrase = NEEDLE_PHRASE_TEMPLATE.format(needle=args.needle)
    padding_text = build_padding_to_distance(pool, args.distance, tok, rng)
    prompt = f"{needle_phrase}\n{padding_text}{QUESTION_TEMPLATE}"
    prompt_tokens = tok.encode(prompt)
    print(f"[eval] prompt token length: {len(prompt_tokens)}")

    # Load model.
    print(f"[eval] loading {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from config import TitansConfig
    from model.titans_gpt2 import TitansMAGGPT2
    from model import _unwrap
    ckpt = load_checkpoint(args.checkpoint, device=device)
    cfg = TitansConfig.from_dict(ckpt["config"])
    model = TitansMAGGPT2(cfg).to(device)
    state = ckpt.get("state_dict", ckpt.get("model"))
    model.load_state_dict(_unwrap(state))
    model.eval()

    # Run chunked forward to set up cache, then read last_logits.
    prompt_t = torch.tensor(
        prompt_tokens, dtype=torch.long, device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        cache = model.prepare_decode_chunked(prompt_t)
    logits = cache["last_logits"][:, 0, :]  # [1, vocab]

    # Target: first BPE token of " {needle}" (with leading space, since the
    # prompt ends with "A:" and the model is predicting what comes next —
    # the training target was " {needle}" with the same leading space).
    target_ids = tok.encode(f" {args.needle}")
    expected_token_id = target_ids[0]
    expected_token_text = tok.decode([expected_token_id])

    pred_token_id = int(logits.argmax(dim=-1).item())
    pred_token_text = tok.decode([pred_token_id])

    # Top-k for diagnostic context.
    top_k = torch.topk(logits[0], k=args.top_k)
    top_k_ids = top_k.indices.tolist()
    top_k_probs = torch.softmax(logits[0], dim=-1)[top_k_ids].tolist()
    top_k_tokens = [tok.decode([i]) for i in top_k_ids]

    # Where does the expected token rank?
    sorted_logits = logits[0].argsort(descending=True)
    expected_rank = (sorted_logits == expected_token_id).nonzero().item()
    expected_prob = torch.softmax(logits[0], dim=-1)[expected_token_id].item()

    print()
    print("=" * 60)
    print(f"needle:                {args.needle}")
    print(f"distance:              {args.distance}")
    print(f"expected next token:   {expected_token_text!r} (id {expected_token_id})")
    print(f"predicted next token:  {pred_token_text!r} (id {pred_token_id})")
    print(f"match:                 {pred_token_id == expected_token_id}")
    print()
    print(f"expected token rank:   {expected_rank}/{logits.size(-1)}")
    print(f"expected token prob:   {expected_prob:.6f}")
    print()
    print(f"top-{args.top_k} predictions:")
    for tid, txt, prob in zip(top_k_ids, top_k_tokens, top_k_probs):
        marker = " *" if tid == expected_token_id else "  "
        print(f"  {marker}  {txt!r:30s}  prob={prob:.6f}  (id {tid})")
    print("=" * 60)


if __name__ == "__main__":
    main()
