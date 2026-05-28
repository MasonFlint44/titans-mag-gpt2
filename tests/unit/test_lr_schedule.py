"""Phase 4.3 — LR schedule (warmup + cosine)."""

import math

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import (
    BASE_LR_GPT2,
    BASE_LR_NMM,
    apply_lr,
    base_lrs_from_constants,
    build_optimizer,
    get_lr_multiplier,
)


# ---------------------------------------------------------------------------
# get_lr_multiplier shape
# ---------------------------------------------------------------------------

def test_warmup_starts_at_zero():
    assert get_lr_multiplier(0, warmup_steps=1000, max_steps=10000) == 0.0


def test_warmup_reaches_one_at_end():
    assert get_lr_multiplier(1000, warmup_steps=1000, max_steps=10000) == 1.0 \
        or abs(get_lr_multiplier(1000, warmup_steps=1000, max_steps=10000) - 1.0) < 1e-9


def test_warmup_is_linear():
    lr_500 = get_lr_multiplier(500, warmup_steps=1000, max_steps=10000)
    assert abs(lr_500 - 0.5) < 1e-9


def test_post_max_steps_floor_at_min_ratio():
    assert get_lr_multiplier(20000, warmup_steps=1000, max_steps=10000, min_ratio=0.1) == 0.1


def test_cosine_decays_smoothly():
    """Right after warmup -> ~1.0; halfway through cosine -> midway."""
    lr_warmup_end = get_lr_multiplier(1000, warmup_steps=1000, max_steps=11000, min_ratio=0.0)
    lr_half_cosine = get_lr_multiplier(6000, warmup_steps=1000, max_steps=11000, min_ratio=0.0)
    lr_end = get_lr_multiplier(11000, warmup_steps=1000, max_steps=11000, min_ratio=0.0)
    assert abs(lr_warmup_end - 1.0) < 1e-9
    # Halfway through the 10k-step cosine, value is 0.5*(1+cos(pi/2)) = 0.5.
    assert abs(lr_half_cosine - 0.5) < 1e-6
    assert lr_end == 0.0


# ---------------------------------------------------------------------------
# degenerate max_steps <= warmup_steps rejected
# ---------------------------------------------------------------------------

def test_max_steps_equal_warmup_rejected():
    with pytest.raises(ValueError, match="max_steps"):
        get_lr_multiplier(0, warmup_steps=1000, max_steps=1000)


def test_max_steps_less_than_warmup_rejected():
    with pytest.raises(ValueError, match="max_steps"):
        get_lr_multiplier(0, warmup_steps=2000, max_steps=1000)


# ---------------------------------------------------------------------------
# apply_lr scales ALL 4 groups, preserves 1:1:3:3 ratio
# ---------------------------------------------------------------------------

def _tiny_optimizer():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    return build_optimizer(model)


def test_apply_lr_scales_all_four_groups_proportionally():
    opt = _tiny_optimizer()
    base_lrs = base_lrs_from_constants()
    lr_mul = apply_lr(opt, base_lrs, step=500, warmup_steps=1000, max_steps=10000)
    # Half warmup -> lr_mul = 0.5
    assert abs(lr_mul - 0.5) < 1e-9
    for g, base in zip(opt.param_groups, base_lrs):
        assert abs(g["lr"] - base * 0.5) < 1e-9


def test_apply_lr_preserves_3x_nmm_ratio():
    """the 1:1:3:3 group LR ratio must persist after apply_lr."""
    opt = _tiny_optimizer()
    base_lrs = base_lrs_from_constants()
    apply_lr(opt, base_lrs, step=750, warmup_steps=1000, max_steps=10000)
    gpt2_lr = opt.param_groups[0]["lr"]
    nmm_lr = opt.param_groups[2]["lr"]
    assert abs(nmm_lr / gpt2_lr - 3.0) < 1e-9


def test_apply_lr_respects_user_max_and_warmup_steps():
    """max_steps/warmup_steps are positional — caller can't accidentally
    inherit the 1k/100k defaults when running a 200-step overfit."""
    opt = _tiny_optimizer()
    base_lrs = base_lrs_from_constants()
    # 200-step training, warmup at 50.
    apply_lr(opt, base_lrs, step=25, warmup_steps=50, max_steps=200)
    # At step 25 of 50 warmup, lr_mul = 0.5.
    for g, base in zip(opt.param_groups, base_lrs):
        assert abs(g["lr"] - base * 0.5) < 1e-9


# ---------------------------------------------------------------------------
# base_lrs from constants, not from optimizer.param_groups
# ---------------------------------------------------------------------------

def test_base_lrs_from_constants_does_not_read_optimizer_state():
    """base_lrs derived from CODE constants survives a deflated
    `optimizer.param_groups[i]['lr']` after load_state_dict. Simulate the
    deflation by manually setting low LRs on the groups, then re-deriving
    base_lrs from constants — must still produce the peak values."""
    opt = _tiny_optimizer()
    # Simulate mid-cosine deflation.
    for g in opt.param_groups:
        g["lr"] = 1e-6
    base_lrs = base_lrs_from_constants()
    assert base_lrs == [BASE_LR_GPT2, BASE_LR_GPT2, BASE_LR_NMM, BASE_LR_NMM]
    # apply_lr at full ramp returns to the original peaks, NOT 1e-6 * 1 = 1e-6.
    apply_lr(opt, base_lrs, step=10000, warmup_steps=1000, max_steps=10001)
    assert abs(opt.param_groups[0]["lr"] - BASE_LR_GPT2 * 0.1) < 1e-9
    assert abs(opt.param_groups[2]["lr"] - BASE_LR_NMM * 0.1) < 1e-9
