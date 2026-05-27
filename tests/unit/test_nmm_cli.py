"""Tests for the shared NMM CLI helper in scripts/_nmm_cli.py.

Covers:
- Defaults: omitting all flags leaves the kwargs dict empty (factory defaults
  preserved).
- Each flag round-trips into the right kwargs key.
- --nmm-layer-indices parses CSV, rejects bad input.
- The kwargs dict splats cleanly into TitansConfig (no unknown keys, no
  validation errors at typical settings).
"""

import argparse

import pytest

from config import TitansConfig
from scripts._nmm_cli import add_nmm_args, nmm_kwargs_from_args


def _parse(*argv):
    p = argparse.ArgumentParser()
    add_nmm_args(p)
    return p.parse_args(list(argv))


def test_defaults_produce_empty_kwargs():
    """No flags passed -> empty kwargs dict so the factory default for every
    field stays untouched. This is the contract callers rely on."""
    args = _parse()
    assert nmm_kwargs_from_args(args) == {}


def test_block_size_round_trip():
    args = _parse("--nmm-block-size", "64")
    assert nmm_kwargs_from_args(args) == {"nmm_block_size": 64}


def test_state_dtype_round_trip():
    args = _parse("--nmm-state-dtype", "bf16")
    assert nmm_kwargs_from_args(args) == {"nmm_state_dtype": "bf16"}


def test_state_dtype_rejects_unknown_choice():
    """argparse should reject fp16 itself — we don't want the user to find
    out at TitansConfig construction time."""
    with pytest.raises(SystemExit):
        _parse("--nmm-state-dtype", "fp16")


def test_low_rank_round_trip():
    args = _parse("--nmm-low-rank", "64")
    assert nmm_kwargs_from_args(args) == {"nmm_low_rank": 64}


def test_expansion_round_trip():
    args = _parse("--nmm-expansion", "1")
    assert nmm_kwargs_from_args(args) == {"nmm_expansion": 1}


def test_layer_indices_parses_csv():
    args = _parse("--nmm-layer-indices", "0,3,6,9")
    assert nmm_kwargs_from_args(args) == {"nmm_layer_indices": [0, 3, 6, 9]}


def test_layer_indices_parses_single_index():
    args = _parse("--nmm-layer-indices", "5")
    assert nmm_kwargs_from_args(args) == {"nmm_layer_indices": [5]}


def test_layer_indices_rejects_non_int():
    with pytest.raises(SystemExit):
        _parse("--nmm-layer-indices", "0,foo,6")


def test_layer_indices_rejects_empty_string():
    """Empty list would silently mean "no NMM blocks" which is almost
    certainly not what the user wanted; force them to omit the flag."""
    with pytest.raises(SystemExit):
        _parse("--nmm-layer-indices", "")


def test_boolean_flags_default_false():
    """The `action='store_true'` flags must default to False, and only
    appear in the kwargs dict when explicitly set (the contract that lets
    TitansConfig defaults stand)."""
    args = _parse()
    kwargs = nmm_kwargs_from_args(args)
    for k in (
        "nmm_detach_state_between_blocks",
    ):
        assert k not in kwargs


def test_detach_state_round_trip():
    args = _parse("--nmm-detach-state-between-blocks", "--nmm-block-size", "16")
    kwargs = nmm_kwargs_from_args(args)
    assert kwargs == {
        "nmm_detach_state_between_blocks": True,
        "nmm_block_size": 16,
    }


def test_ns5_steps_round_trip():
    args = _parse("--nmm-ns5-steps", "3")
    assert nmm_kwargs_from_args(args) == {"nmm_ns5_steps": 3}


def test_ns5_steps_omitted_stays_default():
    """Omitting --nmm-ns5-steps must leave the field at the paper default
    of 5 (not 0 or some sentinel)."""
    args = _parse()
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_ns5_steps == 5


def test_use_gram_ns5_round_trip():
    args = _parse("--nmm-use-gram-ns5")
    assert nmm_kwargs_from_args(args) == {"nmm_use_gram_ns5": True}


def test_use_gram_ns5_omitted_stays_default():
    args = _parse()
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_use_gram_ns5 is False


def test_use_cans_round_trip():
    args = _parse("--nmm-use-cans")
    assert nmm_kwargs_from_args(args) == {"nmm_use_cans": True}


def test_use_cans_omitted_stays_default():
    args = _parse()
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_use_cans is False


def test_full_recipe_round_trip():
    """The recommended T=1024 consumer-GPU recipe must produce a kwargs dict
    that splats cleanly into TitansConfig.gpt2_small. Uses the blockwise
    path so no sequential-only knobs apply."""
    args = _parse(
        "--nmm-block-size", "64",
        "--nmm-state-dtype", "bf16",
        "--nmm-low-rank", "64",
        "--nmm-detach-state-between-blocks",
    )
    kwargs = nmm_kwargs_from_args(args)
    assert kwargs == {
        "nmm_block_size": 64,
        "nmm_state_dtype": "bf16",
        "nmm_low_rank": 64,
        "nmm_detach_state_between_blocks": True,
    }
    # Construct the config — catches any rename/typo that would only surface
    # at first training run.
    cfg = TitansConfig.gpt2_small(
        chunk_size=1024, block_size=1024, **kwargs,
    )
    assert cfg.nmm_block_size == 64
    assert cfg.nmm_state_dtype == "bf16"
    assert cfg.nmm_low_rank == 64
    assert cfg.nmm_detach_state_between_blocks is True


def test_low_rank_omitted_stays_none():
    """Omitting --nmm-low-rank must leave the field None (full-rank
    paper-faithful default), not 0 or some sentinel."""
    args = _parse("--nmm-block-size", "64")
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_low_rank is None


def test_recipe_without_low_rank_constructs():
    """The documented consumer-GPU recipe: T=1024 + block=64 + detach,
    no low_rank. Full-rank with truncated BPTT."""
    args = _parse(
        "--nmm-block-size", "64",
        "--nmm-state-dtype", "bf16",
        "--nmm-detach-state-between-blocks",
    )
    cfg = TitansConfig.gpt2_small(
        chunk_size=1024, block_size=1024, **nmm_kwargs_from_args(args),
    )
    assert cfg.nmm_low_rank is None
    assert cfg.nmm_block_size == 64
    assert cfg.nmm_detach_state_between_blocks is True


# ---------------------------------------------------------------------------
# --vanilla-gpt2 control mode
# ---------------------------------------------------------------------------

def test_vanilla_gpt2_sets_empty_layer_indices():
    """Vanilla mode = every block is a PlainGPT2Block (no NMM). The wire
    format is nmm_layer_indices=[] — the model sees `nmm_idx_set = set()`
    and constructs all-plain blocks. Lock this contract so a refactor
    doesn't switch to e.g. an `is_vanilla` flag that the model has to learn
    to honor separately."""
    args = _parse("--vanilla-gpt2")
    assert nmm_kwargs_from_args(args) == {"nmm_layer_indices": []}


def test_vanilla_gpt2_conflict_with_layer_indices():
    """Both flags write nmm_layer_indices — silently picking one would mask
    a config bug. Fail loud when both are present."""
    import argparse as _ap
    args = _parse("--vanilla-gpt2", "--nmm-layer-indices", "0,3")
    with pytest.raises(_ap.ArgumentTypeError, match="mutually exclusive"):
        nmm_kwargs_from_args(args)


def test_vanilla_gpt2_builds_all_plain_blocks():
    """End-to-end smoke: feed the flag through the same pipeline finetune.py
    uses, and verify the resulting TitansMAGGPT2 has zero NMM blocks. If
    this regresses, the "control condition" experiment would silently train
    a model with NMM and produce a meaningless baseline."""
    from model.titans_gpt2 import TitansMAGGPT2
    from model.block import PlainGPT2Block

    args = _parse("--vanilla-gpt2")
    cfg = TitansConfig.gpt2_small(
        chunk_size=128, block_size=128, **nmm_kwargs_from_args(args),
    )
    model = TitansMAGGPT2(cfg)
    assert cfg.nmm_layer_indices == []
    assert all(isinstance(b, PlainGPT2Block) for b in model.blocks)
    assert all(not has for has in model._block_has_nmm)


def test_vanilla_gpt2_omitted_leaves_default():
    """Without the flag, nmm_layer_indices must stay None (every block has
    NMM — paper-faithful default). No silent activation of vanilla mode."""
    args = _parse()
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_layer_indices is None
