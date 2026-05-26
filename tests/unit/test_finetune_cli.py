"""Tests for scripts.finetune CLI flag plumbing.

The finetune entry point is a thin wrapper around train.run_training; we
don't run training in these tests. We just verify the CLI parser exposes
every flag the consumer-GPU recipe documented in README.md depends on,
with the right defaults — flag drift between README and parser would
break copy-pasted commands the user actually runs.
"""

import pytest

from scripts.finetune import build_parser


def _parse(*argv):
    """Helper: parse argv with `--data x` already added (it's required)."""
    return build_parser().parse_args(["--data", "x", *argv])


# ---------------------------------------------------------------------------
# Existence + defaults
# ---------------------------------------------------------------------------

def test_compile_model_flag_exists_and_defaults_false():
    """G277. The flag must EXIST on finetune.py (the README recipe sends it
    here, not to train.py). Default False so omitting the flag preserves
    pre-G277 behavior."""
    args = _parse()
    assert args.compile_model is False


def test_optim8bit_flag_exists_and_defaults_false():
    """G278. bitsandbytes is an OPTIONAL extra — the default must be False
    so users without bitsandbytes installed can still fine-tune."""
    args = _parse()
    assert args.optim8bit is False


def test_compile_model_round_trip():
    args = _parse("--compile-model")
    assert args.compile_model is True


def test_optim8bit_round_trip():
    args = _parse("--optim8bit")
    assert args.optim8bit is True


# ---------------------------------------------------------------------------
# Composition with existing NMM flags + vanilla-gpt2
# ---------------------------------------------------------------------------

def test_full_titans_recipe_parses():
    """End-to-end: the exact TITANS recipe documented in README.md +
    docs/QA_RECALL_PLAN.md must parse cleanly. Acts as a regression guard
    against any future flag rename."""
    args = _parse(
        "--size", "small",
        "--chunk-size", "1024",
        "--batch-size", "1", "--grad-accum", "16",
        "--nmm-block-size", "64",
        "--nmm-state-dtype", "bf16",
        "--nmm-detach-state-between-blocks",
        "--nmm-use-gram-ns5",
        "--compile-model",
        "--optim8bit",
        "--max-steps", "5000",
    )
    assert args.compile_model is True
    assert args.optim8bit is True
    assert args.nmm_block_size == 64
    assert args.nmm_state_dtype == "bf16"
    assert args.nmm_detach_state_between_blocks is True
    assert args.nmm_use_gram_ns5 is True
    assert args.batch_size == 1
    assert args.grad_accum == 16
    assert args.max_steps == 5000
    assert args.chunk_size == 1024


def test_vanilla_gpt2_recipe_parses():
    """The vanilla GPT-2 control recipe (also from QA_RECALL_PLAN.md) must
    parse cleanly with --vanilla-gpt2 alongside the other perf flags."""
    args = _parse(
        "--size", "small",
        "--chunk-size", "1024",
        "--batch-size", "1", "--grad-accum", "16",
        "--vanilla-gpt2",
        "--compile-model",
        "--optim8bit",
        "--max-steps", "5000",
    )
    assert args.vanilla_gpt2 is True
    assert args.compile_model is True
    assert args.optim8bit is True
    assert args.nmm_layer_indices is None  # vanilla flag handles conversion


def test_data_is_required():
    """Hard requirement — finetune.py needs a corpus path. The parser must
    enforce this so missing-data invocations fail loud at CLI parse time,
    not deep inside encode_corpus with a confusing FileNotFoundError."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
