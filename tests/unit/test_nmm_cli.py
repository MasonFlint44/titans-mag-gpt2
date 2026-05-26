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
    """The four `action='store_true'` flags must default to False, and only
    appear in the kwargs dict when explicitly set (the contract that lets
    TitansConfig defaults stand)."""
    args = _parse()
    kwargs = nmm_kwargs_from_args(args)
    for k in (
        "nmm_detach_state_between_blocks",
        "nmm_compile_inner_loop",
        "nmm_fused_kernel",
        "nmm_compile_ns5",
    ):
        assert k not in kwargs


def test_detach_state_round_trip():
    args = _parse("--nmm-detach-state-between-blocks", "--nmm-block-size", "16")
    kwargs = nmm_kwargs_from_args(args)
    assert kwargs == {
        "nmm_detach_state_between_blocks": True,
        "nmm_block_size": 16,
    }


def test_compile_inner_loop_round_trip():
    args = _parse("--nmm-compile-inner-loop")
    assert nmm_kwargs_from_args(args) == {"nmm_compile_inner_loop": True}


def test_fused_kernel_round_trip():
    args = _parse("--nmm-fused-kernel")
    assert nmm_kwargs_from_args(args) == {"nmm_fused_kernel": True}


def test_compile_ns5_round_trip():
    args = _parse("--nmm-compile-ns5")
    assert nmm_kwargs_from_args(args) == {"nmm_compile_ns5": True}


def test_full_recipe_round_trip():
    """The recommended T=1024 consumer-GPU recipe must produce a kwargs dict
    that splats cleanly into TitansConfig.gpt2_small."""
    args = _parse(
        "--nmm-block-size", "64",
        "--nmm-state-dtype", "bf16",
        "--nmm-low-rank", "64",
        "--nmm-compile-inner-loop",
        "--nmm-fused-kernel",
    )
    kwargs = nmm_kwargs_from_args(args)
    assert kwargs == {
        "nmm_block_size": 64,
        "nmm_state_dtype": "bf16",
        "nmm_low_rank": 64,
        "nmm_compile_inner_loop": True,
        "nmm_fused_kernel": True,
    }
    # Construct the config — catches any rename/typo that would only surface
    # at first training run.
    cfg = TitansConfig.gpt2_small(
        chunk_size=1024, block_size=1024, **kwargs,
    )
    assert cfg.nmm_block_size == 64
    assert cfg.nmm_state_dtype == "bf16"
    assert cfg.nmm_low_rank == 64
    assert cfg.nmm_compile_inner_loop is True
    assert cfg.nmm_fused_kernel is True


def test_low_rank_omitted_stays_none():
    """Omitting --nmm-low-rank must leave the field None (full-rank
    paper-faithful default), not 0 or some sentinel."""
    args = _parse("--nmm-block-size", "64")
    cfg = TitansConfig.gpt2_small(**nmm_kwargs_from_args(args))
    assert cfg.nmm_low_rank is None


def test_recipe_without_low_rank_constructs():
    """The user-requested 'T=1024 without low_rank' recipe."""
    args = _parse(
        "--nmm-block-size", "64",
        "--nmm-state-dtype", "bf16",
        "--nmm-compile-inner-loop",
        "--nmm-fused-kernel",
    )
    cfg = TitansConfig.gpt2_small(
        chunk_size=1024, block_size=1024, **nmm_kwargs_from_args(args),
    )
    assert cfg.nmm_low_rank is None
    assert cfg.nmm_block_size == 64
