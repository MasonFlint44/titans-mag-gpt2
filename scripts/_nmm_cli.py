"""Shared CLI helper: NMM perf knobs for train.py / scripts/finetune.py.

Registers `--nmm-*` flags on an `argparse.ArgumentParser` and converts the
parsed namespace into a kwargs dict ready to splat into a `TitansConfig`
factory. Centralised here so both entry points stay in sync.

Scope: the knobs that change performance/memory in practice. Ablation
flags (lookahead_value, per_param_lr_modulation, momentum_order,
per_head_learned_params, softclamp_max, per_token_ns5, n_heads) are NOT
exposed — they're research/ablation territory; edit `config.py` directly
if you need them.
"""

import argparse
from typing import Optional


def _csv_int_list(s: str) -> list:
    """Parse a comma-separated int list for --nmm-layer-indices.

    "0,3,6,9" -> [0, 3, 6, 9]. Empty string raises (use "" to mean
    "None" is ambiguous; users should just omit the flag instead).
    """
    items = [p.strip() for p in s.split(",") if p.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected at least one integer (got empty list); omit the flag "
            "to leave NMM on every block."
        )
    try:
        return [int(p) for p in items]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated ints (e.g. '0,3,6,9'); got {s!r}"
        ) from e


def add_nmm_args(parser: argparse.ArgumentParser) -> None:
    """Register `--nmm-*` perf flags on `parser`. Defaults preserve the
    existing factory defaults (None means "don't override")."""
    group = parser.add_argument_group(
        "NMM perf/memory knobs",
        "Override TitansConfig defaults from the command line. Omit a flag "
        "to keep the factory default. See docs/CONFIG_REFERENCE.md for the "
        "full memory/throughput characterization.",
    )
    group.add_argument(
        "--nmm-block-size",
        type=int,
        default=None,
        metavar="N",
        help="Blockwise NMM aggregation. 1=paper-strict per-token (default); "
             ">=16 engages tensor cores via batched matmul and replaces the "
             "per-token autograd graph with a per-block one (~T/block_size "
             "smaller). 64 is the recommended starting point.",
    )
    group.add_argument(
        "--nmm-state-dtype",
        choices=["fp32", "bf16", "int8"],
        default=None,
        help="Storage dtype for the recurrent (M, S). bf16 halves it, int8 "
             "(blockwise-only) quarters it. NS5 still runs in fp32 internally.",
    )
    group.add_argument(
        "--nmm-low-rank",
        type=int,
        default=None,
        metavar="R",
        help="Factor MemoryMLP weights as A @ B with intermediate rank R. "
             "Per-step state drops ~10x at r=64. Loses some capacity vs "
             "full-rank; measure loss curves before relying on it.",
    )
    group.add_argument(
        "--nmm-expansion",
        type=int,
        default=None,
        metavar="N",
        help="MemoryMLP hidden-dim multiplier. Default 4 (paper). Setting 1 "
             "makes the three weight matrices square [d, d] and quarters the "
             "per-step NMM state (paper ablation; minor capacity loss).",
    )
    group.add_argument(
        "--nmm-layer-indices",
        type=_csv_int_list,
        default=None,
        metavar="I,J,K",
        help="Comma-separated block indices that get NMM; others become plain "
             "GPT-2 blocks (attn + MLP only). E.g. '0,3,6,9' for 4-of-12. "
             "Linear reduction in NMM-related compute and memory. Omit to "
             "keep NMM on every block (paper-faithful).",
    )
    group.add_argument(
        "--nmm-detach-state-between-blocks",
        action="store_true",
        help="Truncated BPTT at block boundaries (requires --nmm-block-size > 1). "
             "Backward graph spans one block instead of the full chunk; outer "
             "params only learn from gradients within a single block. Cuts "
             "peak transient memory ~proportional to T/block_size.",
    )
    group.add_argument(
        "--nmm-compile-inner-loop",
        action="store_true",
        help="Wrap the per-token inner loop in torch.compile(mode='default'). "
             "Inductor fuses adjacent ops into batched Triton kernels — "
             "dominant ~1.7x speedup. Pays a 30-60s warm-up on the first "
             "training step.",
    )
    group.add_argument(
        "--nmm-compile-ns5",
        action="store_true",
        help="Fused NS5 via torch.compile. No effect when "
             "--nmm-compile-inner-loop is set (the inner-loop compile already "
             "traces NS5 transitively). Useful for the sequential / per-token "
             "NS5 paths.",
    )
    group.add_argument(
        "--nmm-ns5-steps",
        type=int,
        default=None,
        metavar="N",
        help="Newton-Schulz iteration count. Default 5 (Muon coefficients "
             "tuned for this fixed point). Lower is faster but the spectral "
             "norm of NS5(g) drifts away from 1, scaling every memory update. "
             "Measured at gpt2_small: steps=4 ~16%% faster + ~12%% LR drift; "
             "steps=3 ~33%% faster + ~20%% LR drift. Validate convergence on "
             "your data before lowering.",
    )
    group.add_argument(
        "--vanilla-gpt2",
        action="store_true",
        help="Vanilla GPT-2 control mode: every block becomes a plain GPT-2 "
             "block (attn + MLP only, no NMM, no persistent prefix, no MAG "
             "gate). Implemented as nmm_layer_indices=[]. Mutually exclusive "
             "with --nmm-layer-indices: use this flag to mean 'no NMM at all', "
             "not the empty subset. Useful as a control condition for "
             "experiments measuring the NMM's contribution.",
    )
    group.add_argument(
        "--nmm-use-gram-ns5",
        action="store_true",
        help="Replace stock Newton-Schulz with Tri Dao's Gram-Newton-Schulz "
             "(Dao-AILab/gram-newton-schulz). Standard NS5 does 2T "
             "rectangular matmuls; Gram-NS5 does 2 rectangular + T cheap "
             "n×n Gram-matrix iterations. Empirically measured 1.17-3.07× "
             "speedup at gpt2_small dims on consumer Blackwell (RTX 5070 Ti). "
             "Requires `pip install gram-newton-schulz` and PyTorch 2.7+ / "
             "CUDA 12.9+ on a Hopper/Blackwell GPU. Overrides "
             "--nmm-ns5-steps and --nmm-compile-ns5.",
    )


def nmm_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Convert parsed args to a TitansConfig kwargs dict. Only includes
    fields the user explicitly set (i.e. not None / not False).

    Raises:
        argparse.ArgumentTypeError if --vanilla-gpt2 is combined with
        --nmm-layer-indices. The two flags both write `nmm_layer_indices`
        and silently dropping one would mask a config bug.
    """
    kwargs = {}
    if args.nmm_block_size is not None:
        kwargs["nmm_block_size"] = args.nmm_block_size
    if args.nmm_state_dtype is not None:
        kwargs["nmm_state_dtype"] = args.nmm_state_dtype
    if args.nmm_low_rank is not None:
        kwargs["nmm_low_rank"] = args.nmm_low_rank
    if args.nmm_expansion is not None:
        kwargs["nmm_expansion"] = args.nmm_expansion
    if getattr(args, "vanilla_gpt2", False) and args.nmm_layer_indices is not None:
        raise argparse.ArgumentTypeError(
            "--vanilla-gpt2 and --nmm-layer-indices are mutually exclusive: "
            "both write nmm_layer_indices. Use --vanilla-gpt2 alone for the "
            "no-NMM control, or --nmm-layer-indices alone for a subset of "
            "blocks with NMM."
        )
    if getattr(args, "vanilla_gpt2", False):
        # Empty list = every block is a PlainGPT2Block — no NMM, no persistent
        # prefix, no MAG gate. The config validator accepts [] (no items to
        # validate); the model treats nmm_idx_set = set() correctly.
        kwargs["nmm_layer_indices"] = []
    elif args.nmm_layer_indices is not None:
        kwargs["nmm_layer_indices"] = args.nmm_layer_indices
    if args.nmm_detach_state_between_blocks:
        kwargs["nmm_detach_state_between_blocks"] = True
    if args.nmm_compile_inner_loop:
        kwargs["nmm_compile_inner_loop"] = True
    if args.nmm_compile_ns5:
        kwargs["nmm_compile_ns5"] = True
    if args.nmm_ns5_steps is not None:
        kwargs["nmm_ns5_steps"] = args.nmm_ns5_steps
    if args.nmm_use_gram_ns5:
        kwargs["nmm_use_gram_ns5"] = True
    return kwargs
