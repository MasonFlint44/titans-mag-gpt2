"""Tests for cli.finetune CLI flag plumbing.

The finetune entry point is a thin wrapper around train.run_training; we
don't run training in these tests. We just verify the CLI parser exposes
every flag the consumer-GPU recipe documented in README.md depends on,
with the right defaults — flag drift between README and parser would
break copy-pasted commands the user actually runs.
"""

import pytest

from cli.finetune import build_parser


def _parse(*argv):
    """Helper: parse argv with `--data x` already added (it's required)."""
    return build_parser().parse_args(["--data", "x", *argv])


# ---------------------------------------------------------------------------
# Existence + defaults
# ---------------------------------------------------------------------------

def test_compile_model_flag_exists_and_defaults_false():
    """. The flag must EXIST on finetune.py (the README recipe sends it
    here, not to train.py). Default False so omitting the flag preserves
    pre-behavior."""
    args = _parse()
    assert args.compile_model is False


def test_optim8bit_flag_exists_and_defaults_false():
    """. bitsandbytes is an OPTIONAL extra — the default must be False
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
    """End-to-end: the exact TITANS recipe documented in README.md must
    parse cleanly. Acts as a regression guard against any future flag
    rename."""
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
    """The vanilla GPT-2 control recipe must parse cleanly with
    --vanilla-gpt2 alongside the other perf flags."""
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


# ---------------------------------------------------------------------------
# --resume-from
# ---------------------------------------------------------------------------

def test_resume_from_flag_exists_and_defaults_none():
    """No resume requested → args.resume_from is None, training proceeds
    from a fresh load_pretrained. The flag must be opt-in."""
    args = _parse()
    assert args.resume_from is None


def test_resume_from_round_trip():
    args = _parse("--resume-from", "ckpts/titans/latest.pt")
    assert args.resume_from == "ckpts/titans/latest.pt"


def test_resume_compatible_with_recipe_flags():
    """Resume + the documented consumer-GPU recipe must parse cleanly —
    the user keeping the same training scaffolding (--max-steps, --grad-accum,
    --compile-model, --optim8bit) is the common case for "extend a run by N
    more steps."""
    args = _parse(
        "--resume-from", "ckpts/titans/latest.pt",
        "--max-steps", "10000",
        "--save-dir", "ckpts/titans",
        "--batch-size", "1", "--grad-accum", "16",
        "--compile-model", "--optim8bit",
    )
    assert args.resume_from == "ckpts/titans/latest.pt"
    assert args.max_steps == 10000
    assert args.compile_model is True
    assert args.optim8bit is True


def test_config_size_label_helper_round_trips():
    """`config_size_label` powers the resume-mode "ignored --size" warning.
    Must correctly identify each preset by its (n_layer, n_embd) signature.
    Lives in train.py (shared between train.py and cli/finetune.py)."""
    from cli.train import config_size_label
    from config import TitansConfig
    assert config_size_label(TitansConfig.gpt2_small()) == "small"
    assert config_size_label(TitansConfig.gpt2_medium()) == "medium"
    assert config_size_label(TitansConfig.gpt2_large()) == "large"
    assert config_size_label(TitansConfig.gpt2_xl()) == "xl"


def test_config_size_label_handles_custom_dims():
    """For custom configs not matching any factory, return 'custom' rather
    than guessing — the warning would otherwise be misleading."""
    from cli.train import config_size_label
    from config import TitansConfig
    custom = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=32, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    assert config_size_label(custom) == "custom"


# ---------------------------------------------------------------------------
# TPTT-inspired training-regime flags: --freeze-backbone, --nmm-gate-ramp-*
# ---------------------------------------------------------------------------

def test_freeze_backbone_flag_exists_and_defaults_false():
    """Default must be False so existing recipes (full fine-tune) are
    unchanged. Opt-in only — flipping the default would silently change
    every downstream user's training behavior."""
    args = _parse()
    assert args.freeze_backbone is False


def test_freeze_backbone_round_trip():
    args = _parse("--freeze-backbone")
    assert args.freeze_backbone is True


def test_nmm_gate_ramp_steps_defaults_to_zero():
    """Default 0 means no ramping — preserves prior training behavior for
    recipes that don't pass the flag."""
    args = _parse()
    assert args.nmm_gate_ramp_steps == 0


def test_nmm_gate_ramp_target_defaults_to_small_positive():
    """Default target should be small (~0.1) — close to what the optimizer
    empirically finds when out_scale is free. A large default would over-
    open the memory gate and degrade short-distance attention performance."""
    args = _parse()
    assert 0.0 < args.nmm_gate_ramp_target <= 0.5


def test_nmm_gate_ramp_flags_round_trip():
    args = _parse(
        "--nmm-gate-ramp-steps", "100",
        "--nmm-gate-ramp-target", "0.2",
    )
    assert args.nmm_gate_ramp_steps == 100
    assert args.nmm_gate_ramp_target == pytest.approx(0.2)


def test_freeze_backbone_and_gate_ramp_compose_in_recipe():
    """The intended use case is to set BOTH flags together (TPTT-style:
    freeze backbone + ramp gate). Verify they parse without conflict."""
    args = _parse(
        "--freeze-backbone",
        "--nmm-gate-ramp-steps", "100",
        "--nmm-gate-ramp-target", "0.1",
        "--nmm-use-gram-ns5",
        "--compile-model",
        "--optim8bit",
    )
    assert args.freeze_backbone is True
    assert args.nmm_gate_ramp_steps == 100
    assert args.nmm_gate_ramp_target == pytest.approx(0.1)
    assert args.nmm_use_gram_ns5 is True


def test_freeze_embeddings_flag_exists_and_defaults_false():
    """The softer freeze must be opt-in — default keeps the existing
    full-fine-tune behavior so legacy recipes are unchanged."""
    args = _parse()
    assert args.freeze_embeddings is False


def test_freeze_embeddings_round_trip():
    args = _parse("--freeze-embeddings")
    assert args.freeze_embeddings is True
    assert args.freeze_backbone is False  # mutex with freeze-backbone


def test_freeze_backbone_and_freeze_embeddings_are_mutually_exclusive():
    """Passing both freeze modes is contradictory and the parser should
    reject it loudly rather than silently picking one."""
    with pytest.raises(SystemExit):
        _parse("--freeze-backbone", "--freeze-embeddings")


def test_freeze_embeddings_composes_with_gate_ramp():
    """The intended recommended recipe: --freeze-embeddings +
    --nmm-gate-ramp-*. Verify they parse together."""
    args = _parse(
        "--freeze-embeddings",
        "--nmm-gate-ramp-steps", "100",
        "--nmm-gate-ramp-target", "0.1",
    )
    assert args.freeze_embeddings is True
    assert args.freeze_backbone is False
    assert args.nmm_gate_ramp_steps == 100


def test_nmm_aux_loss_weight_flag_exists_and_defaults_to_zero():
    """Default 0.0 means the aux loss is disabled — preserves
    backward-compatible training behavior for recipes that don't pass
    the flag. Opt-in only."""
    args = _parse()
    assert args.nmm_aux_loss_weight == 0.0


def test_nmm_aux_loss_weight_round_trip():
    args = _parse("--nmm-aux-loss-weight", "0.25")
    assert args.nmm_aux_loss_weight == pytest.approx(0.25)


def test_nmm_aux_loss_weight_composes_with_freeze_and_ramp():
    """The full recommended recipe: --freeze-embeddings +
    --nmm-gate-ramp-* + --nmm-aux-loss-weight. Verify they all parse
    together without conflict."""
    args = _parse(
        "--freeze-embeddings",
        "--nmm-gate-ramp-steps", "100",
        "--nmm-gate-ramp-target", "0.1",
        "--nmm-aux-loss-weight", "0.5",
        "--nmm-use-gram-ns5",
        "--compile-model",
        "--optim8bit",
    )
    assert args.freeze_embeddings is True
    assert args.nmm_gate_ramp_steps == 100
    assert args.nmm_aux_loss_weight == pytest.approx(0.5)
    assert args.nmm_use_gram_ns5 is True
    assert args.compile_model is True
    assert args.optim8bit is True


# ---------------------------------------------------------------------------
# DeltaProduct memory selection flags
# ---------------------------------------------------------------------------


def test_memory_type_default_is_none_unset():
    """`--memory-type` default is None so that omitting the flag preserves
    the TitansConfig default ('nmm') — i.e. existing recipes unchanged."""
    args = _parse()
    assert args.memory_type is None


def test_delta_flags_default_is_none_unset():
    args = _parse()
    assert args.delta_order is None
    assert args.delta_n_heads is None
    assert args.delta_block_size is None


def test_memory_type_round_trip_delta_product():
    args = _parse(
        "--memory-type", "delta_product",
        "--delta-order", "2",
        "--delta-n-heads", "4",
        "--delta-block-size", "32",
    )
    assert args.memory_type == "delta_product"
    assert args.delta_order == 2
    assert args.delta_n_heads == 4
    assert args.delta_block_size == 32


def test_memory_type_round_trip_nmm():
    args = _parse("--memory-type", "nmm")
    assert args.memory_type == "nmm"


def test_memory_type_rejects_unknown_value():
    """Choices restrict to {nmm, delta_product}; argparse rejects others."""
    with pytest.raises(SystemExit):
        _parse("--memory-type", "elephant")


def test_delta_order_without_memory_type_raises_via_kwargs():
    """--delta-* flags are only meaningful with --memory-type delta_product.
    The cross-flag validator runs at kwargs-extraction time so config-build
    fails fast rather than silently no-op'ing the flag."""
    import argparse as _ap
    from cli.nmm_cli import nmm_kwargs_from_args
    args = _parse("--delta-order", "2")
    with pytest.raises(_ap.ArgumentTypeError, match="delta_product"):
        nmm_kwargs_from_args(args)


def test_delta_n_heads_without_memory_type_raises_via_kwargs():
    import argparse as _ap
    from cli.nmm_cli import nmm_kwargs_from_args
    args = _parse("--delta-n-heads", "4")
    with pytest.raises(_ap.ArgumentTypeError, match="delta_product"):
        nmm_kwargs_from_args(args)


def test_delta_block_size_without_memory_type_raises_via_kwargs():
    import argparse as _ap
    from cli.nmm_cli import nmm_kwargs_from_args
    args = _parse("--delta-block-size", "32")
    with pytest.raises(_ap.ArgumentTypeError, match="delta_product"):
        nmm_kwargs_from_args(args)


def test_delta_kwargs_extraction_includes_only_set_fields():
    """nmm_kwargs_from_args only emits keys for flags the user explicitly
    set — keeps the TitansConfig defaults authoritative for everything else."""
    from cli.nmm_cli import nmm_kwargs_from_args
    args = _parse("--memory-type", "delta_product", "--delta-order", "2")
    kwargs = nmm_kwargs_from_args(args)
    assert kwargs["memory_type"] == "delta_product"
    assert kwargs["delta_order"] == 2
    # delta_n_heads and delta_block_size NOT set on CLI -> not in kwargs.
    assert "delta_n_heads" not in kwargs
    assert "delta_block_size" not in kwargs


def test_delta_product_composes_with_freeze_and_compile():
    """The DeltaProduct training recipe should compose cleanly with the
    existing freeze/ramp/compile flags."""
    args = _parse(
        "--memory-type", "delta_product",
        "--delta-order", "2",
        "--delta-n-heads", "12",
        "--delta-block-size", "64",
        "--freeze-embeddings",
        "--nmm-gate-ramp-steps", "100",
        "--nmm-gate-ramp-target", "0.1",
        "--compile-model",
        "--optim8bit",
    )
    assert args.memory_type == "delta_product"
    assert args.delta_order == 2
    assert args.delta_n_heads == 12
    assert args.delta_block_size == 64
    assert args.freeze_embeddings is True
    assert args.compile_model is True
    assert args.optim8bit is True
