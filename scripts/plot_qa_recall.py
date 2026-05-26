"""Plot recall accuracy vs. context distance for one or more eval runs.

Input: one or more JSON files produced by scripts.eval_qa_recall.
Output: a single PNG showing accuracy-vs-distance curves with bootstrapped
95% confidence intervals.

CLI:
    python -m scripts.plot_qa_recall \\
        --input results/vanilla.json results/titans.json \\
        --label "Vanilla GPT-2" "TITANS" \\
        --out docs/figures/qa_recall.png

If --label is omitted, the script falls back to the input file stems
(`vanilla`, `titans`) as labels. The vertical line at x=block_size is
drawn automatically when every input agrees on block_size — that's the
visual marker for "attention runs out here; only NMM recall can hold
past this point."
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence


def bootstrap_ci(
    trials: Sequence[bool],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Nonparametric bootstrap CI for the proportion of True in `trials`.

    For per-distance accuracy with ~500 trials, the closed-form Wilson CI
    would also work, but bootstrap is easier to extend (e.g., to median
    decode-time later) and the cost is trivial at this trial count. seed
    keeps the chart reproducible across runs.
    """
    if not trials:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(trials)
    means = []
    for _ in range(n_boot):
        sample = [trials[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_per_distance(run: dict) -> dict[int, dict]:
    """Re-aggregate per-distance trials from a run's `results` list so we
    can compute bootstrap CIs. The pre-aggregated `buckets` field has the
    means but not the per-trial booleans we need for resampling."""
    by_distance: dict[int, list[bool]] = {}
    for r in run["results"]:
        by_distance.setdefault(r["distance"], []).append(bool(r["correct"]))
    out = {}
    for d, trials in by_distance.items():
        lo, hi = bootstrap_ci(trials)
        out[d] = {
            "n": len(trials),
            "accuracy": sum(trials) / len(trials),
            "ci_lo": lo,
            "ci_hi": hi,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True,
        help="One or more eval JSON files (output of scripts.eval_qa_recall).",
    )
    parser.add_argument(
        "--label", type=str, nargs="*", default=None,
        help="One label per --input file (in the same order). If omitted, "
             "input file stems are used.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output PNG path. Parent dirs are created if missing.",
    )
    parser.add_argument(
        "--title", type=str, default="QA recall vs. context distance",
        help="Plot title.",
    )
    parser.add_argument(
        "--xlim-max", type=int, default=None,
        help="Override the x-axis upper limit (default: max distance across "
             "all inputs).",
    )
    args = parser.parse_args()

    # Defer matplotlib import so importing this module (e.g., for testing
    # the bootstrap helper) doesn't require a plot stack.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib not installed. Run: pip install matplotlib"
        )

    if args.label is None:
        labels = [p.stem for p in args.input]
    elif len(args.label) != len(args.input):
        raise SystemExit(
            f"--label count ({len(args.label)}) must match --input count "
            f"({len(args.input)})."
        )
    else:
        labels = args.label

    runs = [load_run(p) for p in args.input]
    aggregated = [aggregate_per_distance(r) for r in runs]

    # Detect a shared block_size for the attention boundary line.
    block_sizes = {r.get("config", {}).get("block_size") for r in runs}
    block_sizes.discard(None)
    shared_block_size = (
        block_sizes.pop() if len(block_sizes) == 1 else None
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, agg in zip(labels, aggregated):
        distances = sorted(agg.keys())
        accs = [agg[d]["accuracy"] for d in distances]
        lo = [agg[d]["ci_lo"] for d in distances]
        hi = [agg[d]["ci_hi"] for d in distances]
        ax.plot(distances, accs, marker="o", label=label)
        ax.fill_between(distances, lo, hi, alpha=0.2)

    if shared_block_size is not None:
        ax.axvline(
            shared_block_size, color="gray", linestyle="--", alpha=0.6,
            label=f"block_size = {shared_block_size}",
        )

    ax.set_xlabel("Distance from answer to question (tokens)")
    ax.set_ylabel("First-token recall accuracy")
    ax.set_title(args.title)
    ax.set_ylim(-0.02, 1.02)
    if args.xlim_max is not None:
        ax.set_xlim(left=0, right=args.xlim_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"[plot] wrote {args.out}", file=sys.stderr, flush=True)
    # Also print the table to stderr so logs capture the numbers without
    # opening the PNG.
    for label, agg in zip(labels, aggregated):
        print(f"[plot] {label}:", file=sys.stderr)
        for d in sorted(agg.keys()):
            r = agg[d]
            print(
                f"  d={d:>5d}  acc={r['accuracy']:.3f}  "
                f"95%CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
