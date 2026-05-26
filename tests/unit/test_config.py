import subprocess
import sys
import warnings

import pytest

from config import TitansConfig


# ---------------------------------------------------------------------------
# Factory dimensions (G143, G150)
# ---------------------------------------------------------------------------

def test_gpt2_small_factory_dims():
    cfg = TitansConfig.gpt2_small()
    assert cfg.n_embd == 768
    assert cfg.n_head == 12
    assert cfg.n_layer == 12


def test_gpt2_medium_factory_dims():
    cfg = TitansConfig.gpt2_medium()
    assert cfg.n_embd == 1024
    assert cfg.n_head == 16
    assert cfg.n_layer == 24


def test_gpt2_large_factory_dims():
    cfg = TitansConfig.gpt2_large()
    assert cfg.n_embd == 1280
    assert cfg.n_head == 20
    assert cfg.n_layer == 36


def test_gpt2_xl_factory_dims():
    cfg = TitansConfig.gpt2_xl()
    assert cfg.n_embd == 1600
    assert cfg.n_head == 25
    assert cfg.n_layer == 48


# ---------------------------------------------------------------------------
# Factory accepts overrides (G150)
# ---------------------------------------------------------------------------

def test_factory_accepts_dropout_override():
    cfg = TitansConfig.gpt2_small(dropout=0.1)
    assert cfg.dropout == 0.1
    # Other defaults intact.
    assert cfg.n_embd == 768


def test_factory_accepts_dim_override():
    # The **{**defaults, **overrides} pattern must allow overriding the
    # backbone dims the factory itself sets — earlier (fixed kwargs +
    # **overrides) code raised TypeError: multiple values for keyword argument.
    cfg = TitansConfig.gpt2_small(n_embd=768, n_head=12)
    assert cfg.n_embd == 768


def test_factory_accepts_chunk_size_override():
    cfg = TitansConfig.gpt2_small(chunk_size=256)
    assert cfg.chunk_size == 256


# ---------------------------------------------------------------------------
# chunk_size > block_size rejected (G190, G206)
# ---------------------------------------------------------------------------

def test_chunk_size_exceeds_block_size_rejected():
    # G206: must be ValueError, NOT AssertionError. A test using
    # pytest.raises(AssertionError) would silently pass-the-wrong-way once
    # the assert→raise migration happened.
    with pytest.raises(ValueError, match="chunk_size"):
        TitansConfig(chunk_size=2048, block_size=1024)


def test_chunk_size_equal_to_block_size_accepted():
    cfg = TitansConfig(chunk_size=1024, block_size=1024)
    assert cfg.chunk_size == 1024


# ---------------------------------------------------------------------------
# n_embd not divisible by n_head rejected at config time (G223)
# ---------------------------------------------------------------------------

def test_n_embd_not_divisible_by_n_head_rejected():
    with pytest.raises(ValueError, match="divisible by n_head"):
        TitansConfig.gpt2_small(n_head=10)


def test_n_embd_divisibility_error_quotes_values():
    # Error should be informative, not generic "invalid".
    with pytest.raises(ValueError) as excinfo:
        TitansConfig(n_embd=768, n_head=10)
    msg = str(excinfo.value)
    assert "768" in msg
    assert "10" in msg


# ---------------------------------------------------------------------------
# SWA window validation (G166)
# ---------------------------------------------------------------------------

def test_swa_zero_window_rejected():
    with pytest.raises(ValueError, match="swa_window"):
        TitansConfig(use_swa=True, swa_window=0)


def test_swa_negative_window_rejected():
    with pytest.raises(ValueError, match="swa_window"):
        TitansConfig(use_swa=True, swa_window=-1)


def test_swa_disabled_window_value_irrelevant():
    # When use_swa=False, swa_window value is unused; do not reject.
    cfg = TitansConfig(use_swa=False, swa_window=0)
    assert cfg.swa_window == 0


# ---------------------------------------------------------------------------
# NMM field validation
# ---------------------------------------------------------------------------

def test_nmm_n_persistent_negative_rejected():
    with pytest.raises(ValueError, match="nmm_n_persistent"):
        TitansConfig(nmm_n_persistent=-1)


def test_nmm_n_persistent_zero_accepted():
    cfg = TitansConfig(nmm_n_persistent=0)
    assert cfg.nmm_n_persistent == 0


def test_nmm_expansion_zero_rejected():
    with pytest.raises(ValueError, match="nmm_expansion"):
        TitansConfig(nmm_expansion=0)


def test_nmm_expansion_negative_rejected():
    with pytest.raises(ValueError, match="nmm_expansion"):
        TitansConfig(nmm_expansion=-1)


# ---------------------------------------------------------------------------
# G163 — from-scratch + chunk_size < block_size warns; finetune does not
# ---------------------------------------------------------------------------

def test_from_scratch_short_chunk_warns():
    with pytest.warns(UserWarning, match="finetune_mode=False"):
        TitansConfig(finetune_mode=False, chunk_size=512, block_size=1024)


def test_finetune_short_chunk_does_not_warn():
    # finetune path overwrites wpe with HF's trained table — no risk.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning -> test failure
        TitansConfig(finetune_mode=True, chunk_size=512, block_size=1024)


def test_from_scratch_equal_chunk_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TitansConfig(finetune_mode=False, chunk_size=1024, block_size=1024)


# ---------------------------------------------------------------------------
# Sequential-path flags warn when set with blockwise path (no-op there).
# Bench confirmed: nmm_compile_inner_loop / nmm_fused_kernel have zero
# effect on the blockwise path, and compile_inner_loop costs ~20s of
# warm-up. Warn so users don't pay that cost for nothing.
# ---------------------------------------------------------------------------

def test_compile_inner_loop_with_blockwise_warns():
    with pytest.warns(UserWarning, match="sequential path"):
        TitansConfig(nmm_block_size=64, nmm_compile_inner_loop=True)


def test_fused_kernel_with_blockwise_warns():
    with pytest.warns(UserWarning, match="sequential path"):
        TitansConfig(nmm_block_size=64, nmm_fused_kernel=True)


def test_compile_inner_loop_with_sequential_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TitansConfig(nmm_block_size=1, nmm_compile_inner_loop=True)


def test_fused_kernel_with_sequential_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TitansConfig(nmm_block_size=1, nmm_fused_kernel=True)


# ---------------------------------------------------------------------------
# nmm_ns5_steps — opt-in speed/quality knob, default 5
# ---------------------------------------------------------------------------

def test_ns5_steps_default_is_5():
    cfg = TitansConfig()
    assert cfg.nmm_ns5_steps == 5


def test_ns5_steps_accepts_valid_range():
    for n in (1, 3, 4, 5, 10):
        cfg = TitansConfig(nmm_ns5_steps=n)
        assert cfg.nmm_ns5_steps == n


def test_ns5_steps_rejects_zero():
    with pytest.raises(ValueError, match="nmm_ns5_steps"):
        TitansConfig(nmm_ns5_steps=0)


def test_ns5_steps_rejects_negative():
    with pytest.raises(ValueError, match="nmm_ns5_steps"):
        TitansConfig(nmm_ns5_steps=-1)


def test_ns5_steps_rejects_excessive():
    """10 is the sanity upper bound — past that, the iteration is wasting
    compute since convergence is exponential."""
    with pytest.raises(ValueError, match="nmm_ns5_steps"):
        TitansConfig(nmm_ns5_steps=11)


def test_ns5_steps_rejects_non_int():
    with pytest.raises(ValueError, match="nmm_ns5_steps"):
        TitansConfig(nmm_ns5_steps=3.5)


def test_ns5_steps_propagates_to_nmm():
    """The config field must actually flow through to NeuralMemoryModule."""
    from model.titans_gpt2 import TitansMAGGPT2
    cfg = TitansConfig.gpt2_small(
        n_layer=1, nmm_block_size=64,
        nmm_detach_state_between_blocks=True,
        nmm_ns5_steps=3,
    )
    m = TitansMAGGPT2(cfg)
    assert m.blocks[0].nmm.ns5_steps == 3


# ---------------------------------------------------------------------------
# nmm_use_gram_ns5 — Tri Dao's Gram-Newton-Schulz drop-in
# ---------------------------------------------------------------------------

def test_use_gram_ns5_default_is_false():
    """Default must be off so existing configs are unaffected and we
    don't add a runtime import failure for users who haven't installed
    the optional dep."""
    cfg = TitansConfig()
    assert cfg.nmm_use_gram_ns5 is False


def test_use_gram_ns5_can_be_set_at_config_time():
    """Config construction must succeed even when the gram-newton-schulz
    package isn't imported. The import is lazy (happens at NMM
    construction time when the flag is True)."""
    cfg = TitansConfig(nmm_use_gram_ns5=True)
    assert cfg.nmm_use_gram_ns5 is True


def test_use_gram_ns5_warns_when_overridden_knobs_set():
    """If user sets both gram_ns5 AND nmm_ns5_steps != 5, warn — the
    ns5_steps value is ignored because Gram-NS5 has its own per-iter
    coefficient table."""
    with pytest.warns(UserWarning, match="nmm_use_gram_ns5=True overrides"):
        TitansConfig(nmm_use_gram_ns5=True, nmm_ns5_steps=4)


def test_use_gram_ns5_warns_when_compile_ns5_set():
    with pytest.warns(UserWarning, match="nmm_use_gram_ns5=True overrides"):
        TitansConfig(nmm_use_gram_ns5=True, nmm_compile_ns5=True)


def test_use_gram_ns5_no_warn_with_default_knobs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TitansConfig(nmm_use_gram_ns5=True)  # default ns5_steps=5, compile_ns5=False


# ---------------------------------------------------------------------------
# Validation survives `python -O` (G190)
# ---------------------------------------------------------------------------

def test_validation_fires_under_python_O():
    # If __post_init__ used `assert`, -O would strip it and the bad config
    # would build silently. We use `raise ValueError`; verify under -O.
    result = subprocess.run(
        [
            sys.executable, "-O", "-c",
            "from config import TitansConfig; "
            "TitansConfig(chunk_size=2048, block_size=1024)",
        ],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit under -O; got 0.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ValueError" in result.stderr


# ---------------------------------------------------------------------------
# T9 — fuzzed invalid-config rejection (TEST_PLAN §13 test_invalid_configs.py)
# ---------------------------------------------------------------------------

import random
import pytest


def _invalid_config_kwargs(rng):
    """Generate ONE invalid TitansConfig kwarg combination. Each branch
    violates a distinct __post_init__ check; the function chooses one at
    random so the fuzz test covers different rejection paths."""
    choice = rng.choice(["chunk_gt_block", "div", "n_persistent_neg",
                         "expansion_lt1", "swa_window"])
    if choice == "chunk_gt_block":
        return {
            "chunk_size": rng.randint(1025, 2048),
            "block_size": rng.randint(64, 1024),
            "n_layer": 1, "n_head": 2, "n_embd": 16,
            "vocab_size": 32, "nmm_expansion": 2,
        }, "chunk_size"
    if choice == "div":
        # n_embd not divisible by n_head.
        return {
            "n_layer": 1, "n_head": 3, "n_embd": 16,  # 16 % 3 != 0
            "vocab_size": 32, "block_size": 64, "chunk_size": 8,
            "nmm_expansion": 2,
        }, "divisible"
    if choice == "n_persistent_neg":
        return {
            "n_layer": 1, "n_head": 2, "n_embd": 16, "vocab_size": 32,
            "block_size": 64, "chunk_size": 8, "nmm_expansion": 2,
            "nmm_n_persistent": -rng.randint(1, 100),
        }, "nmm_n_persistent"
    if choice == "expansion_lt1":
        return {
            "n_layer": 1, "n_head": 2, "n_embd": 16, "vocab_size": 32,
            "block_size": 64, "chunk_size": 8,
            "nmm_expansion": rng.choice([0, -1, -5]),
        }, "nmm_expansion"
    if choice == "swa_window":
        return {
            "n_layer": 1, "n_head": 2, "n_embd": 16, "vocab_size": 32,
            "block_size": 64, "chunk_size": 8, "nmm_expansion": 2,
            "use_swa": True, "swa_window": rng.choice([0, -1, -10]),
        }, "swa_window"


@pytest.mark.parametrize("seed", list(range(40)))
def test_fuzzed_invalid_config_raises_value_error_with_informative_message(seed):
    """T9 — 40 random invalid-config combinations across 5 rejection paths.
    Each must raise ValueError, and the message must name the offending
    field so a user can fix it without reading the source."""
    rng = random.Random(seed)
    kwargs, expected_substring = _invalid_config_kwargs(rng)
    with pytest.raises(ValueError) as excinfo:
        TitansConfig(**kwargs)
    msg = str(excinfo.value).lower()
    assert expected_substring.lower() in msg, (
        f"ValueError message {msg!r} does not mention the offending field "
        f"{expected_substring!r}; kwargs={kwargs}"
    )


def test_nmm_depth_default_is_2():
    """Paper-default L_M; the only value the implementation supports today."""
    cfg = TitansConfig.gpt2_small()
    assert cfg.nmm_depth == 2


def test_nmm_depth_rejects_other_values():
    """L_M != 2 raises ValueError — wiring it up requires changes to the
    analytical-gradient + Triton kernels that we have not made."""
    import pytest
    for bad in (1, 3, 4, 0, -1):
        with pytest.raises(ValueError, match=r"nmm_depth=.*not supported"):
            TitansConfig.gpt2_small(nmm_depth=bad)
