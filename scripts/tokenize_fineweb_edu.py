"""Stream FineWeb-Edu sample-10BT, tokenize with GPT-2 BPE, write a uint16
binary token stream with EOT separators between documents.

The output is the dense LongTensor that ParallelStreamLoader expects, but
serialized as raw little-endian uint16 to save disk (~4 GB for 1.5B tokens
vs ~12 GB if we stored as int64). At training time `cli/train.py` memmaps
the file and converts to LongTensor in one shot.

Format on disk:
    raw uint16 little-endian, one token per pair of bytes, no header.
    Length = file_size / 2 tokens. EOT (50256) is inserted between
    consecutive documents (same convention as `Tokenizer.encode_corpus`).

This matches nanoGPT's data layout, which is the de-facto standard for
GPT-2-class from-scratch training and trivially round-trips.

Streaming usage:
    The HuggingFace `datasets` library streams FineWeb-Edu without
    downloading the full ~25 GB sample-10BT shard archive. We pull
    examples one at a time, tokenize, and append to the output. Memory
    footprint stays bounded regardless of total token count.

Resumability:
    The script reads the existing output file's size before starting and
    skips that many examples (approximately — counted by accumulated
    token total). For exact resume use `--no-resume` and a clean output
    path. Good enough for normal interrupt-and-restart cycles.

CLI:
    uv run python -m scripts.tokenize_fineweb_edu \\
        --output corpora/fineweb_edu_1p5b.bin \\
        --max-tokens 1500000000 \\
        --dataset-name HuggingFaceFW/fineweb-edu \\
        --dataset-config sample-10BT
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from data.tokenizer import Tokenizer

# Empirical: the HuggingFace `datasets` streaming iterator's __exit__ /
# generator close hangs for an indefinite time after we break out of the
# main loop — observed on the FineWeb-Edu sample-10BT smoke run, process
# stays alive with no further file writes for many seconds. Once our work
# is done (target token count reached, file closed and flushed by the
# `with open(...)` __exit__), there's nothing in the iterator we need to
# clean up. Using `os._exit(0)` bypasses Python's normal shutdown,
# skipping the streaming-dataset teardown. We do NOT want sys.exit() —
# that still runs the iterator's __exit__ via gc/atexit.
_FINAL_EXIT_CODE = 0


def _format_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True,
        help="Path to the output uint16 binary token stream.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1_500_000_000,
        help="Stop after writing this many tokens. Default 1.5B — enough "
             "headroom for a ~10-day NMM-from-scratch run at gpt2_small.",
    )
    parser.add_argument(
        "--dataset-name", default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace datasets repo id.",
    )
    parser.add_argument(
        "--dataset-config", default="sample-10BT",
        help="Subset within the dataset. sample-10BT is FineWeb-Edu's "
             "10B-token deterministic sample — the canonical small-scale "
             "training set.",
    )
    parser.add_argument(
        "--split", default="train",
        help="Split to stream. Default 'train' (the only one in sample-10BT).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Truncate the output file before starting. Default behavior is "
             "to append to an existing file (assumes prior interrupted run).",
    )
    parser.add_argument(
        "--log-every", type=int, default=10000,
        help="Print throughput every N documents. Default 10000.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume vs fresh: count tokens already on disk.
    existing_tokens = 0
    open_mode = "wb"
    if not args.no_resume and out_path.exists():
        existing_size = out_path.stat().st_size
        if existing_size % 2 != 0:
            raise SystemExit(
                f"Existing output {out_path} has odd byte size "
                f"({existing_size}) — not a clean uint16 stream. Delete "
                f"or pass --no-resume to overwrite."
            )
        existing_tokens = existing_size // 2
        if existing_tokens >= args.max_tokens:
            print(
                f"[tokenize] Output already has {_format_count(existing_tokens)} "
                f"tokens (>= --max-tokens {_format_count(args.max_tokens)}). "
                f"Nothing to do.",
                file=sys.stderr,
            )
            return
        open_mode = "ab"
        print(
            f"[tokenize] Resuming — found {_format_count(existing_tokens)} "
            f"tokens already in {out_path}; appending up to "
            f"{_format_count(args.max_tokens)} total.",
            file=sys.stderr,
        )

    # Streaming dataset. Lazy import so the script's --help works without
    # `datasets` installed.
    from datasets import load_dataset

    ds = load_dataset(
        args.dataset_name, name=args.dataset_config,
        split=args.split, streaming=True,
    )
    tok = Tokenizer()
    eot = tok.eot_token

    # On a resumed run we have to skip already-consumed documents. The
    # streaming dataset is deterministic per-shard, so we can advance the
    # iterator to roughly the right place. We approximate by skipping
    # `existing_tokens / avg_doc_len` docs — close enough for resume.
    # Subtle: documents already partially written are fine; the binary
    # stream is just a flat sequence of tokens.
    # In practice we just count from where we are.
    skip_docs = 0
    if existing_tokens > 0:
        # Rough average: FineWeb-Edu docs average ~500 tokens. The exact
        # number doesn't matter — under-skipping just means we re-tokenize
        # some text; over-skipping loses some. We err on the side of
        # under-skipping (so no token loss; the actual `total_tokens`
        # accounting below stops at --max-tokens anyway).
        skip_docs = max(0, int(existing_tokens / 600) - 100)
        if skip_docs > 0:
            print(
                f"[tokenize] Advancing iterator past ~{skip_docs} docs to "
                f"approximate the resume point.",
                file=sys.stderr,
            )

    total_tokens = existing_tokens
    docs_processed = 0
    t0 = time.time()
    t_last_log = t0

    with open(out_path, open_mode, buffering=4 * 1024 * 1024) as f:
        for i, example in enumerate(ds):
            if i < skip_docs:
                continue
            text = example.get("text", "")
            if not text:
                continue
            ids = tok.encode(text)
            ids.append(eot)
            # Validate that all ids fit in uint16 (GPT-2 vocab is 50257,
            # uint16 max is 65535 — comfortable margin).
            arr = np.array(ids, dtype=np.uint16)
            if arr.max() >= 65536:
                raise RuntimeError(
                    f"Token id {arr.max()} exceeds uint16 max — tokenizer "
                    f"vocab does not fit in uint16. Adjust dtype."
                )
            f.write(arr.tobytes())
            total_tokens += len(ids)
            docs_processed += 1

            if docs_processed % args.log_every == 0:
                t_now = time.time()
                tok_per_sec = (
                    (total_tokens - existing_tokens) / (t_now - t0)
                )
                pct = 100.0 * total_tokens / args.max_tokens
                eta_sec = (args.max_tokens - total_tokens) / max(
                    tok_per_sec, 1,
                )
                eta_min = eta_sec / 60
                print(
                    f"[tokenize] {_format_count(total_tokens)} tokens "
                    f"({pct:.1f}%)  {docs_processed} docs  "
                    f"{tok_per_sec / 1e6:.2f} Mtok/s  "
                    f"eta {eta_min:.1f} min",
                    file=sys.stderr,
                )
                t_last_log = t_now

            if total_tokens >= args.max_tokens:
                break

    print(
        f"[tokenize] Done. {_format_count(total_tokens)} tokens, "
        f"{docs_processed} docs, "
        f"{(time.time() - t0) / 60:.1f} min wall-clock. "
        f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB).",
        file=sys.stderr,
    )
    sys.stderr.flush()
    # Bypass the HF streaming iterator's teardown — see comment at top.
    os._exit(_FINAL_EXIT_CODE)


if __name__ == "__main__":
    main()
