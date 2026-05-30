"""Phased-training tests: `--freeze-attention-steps N`.

The two interventions that have to compose correctly:
  (a) `collect_attention_proj_params` enumerates exactly the q/k/v/proj
      weight + bias tensors across every transformer block (handling
      both plain `nn.Linear` and `LoRALinear` wrapping).
  (b) `run_training`, when `freeze_attention_steps > 0`, sets those
      params to `requires_grad=False` at start, then flips them back to
      True at step N. The memory pathway and MLP stay trainable
      throughout.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import (
    build_optimizer,
    collect_attention_proj_params,
    run_training,
)


def _tiny_cfg(**overrides):
    base = dict(
        n_layer=2, n_head=4, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    base.update(overrides)
    return TitansConfig(**base)


def _stream(n=320, vocab=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g)


# ---------------------------------------------------------------------------
# collect_attention_proj_params
# ---------------------------------------------------------------------------


def test_collect_returns_q_k_v_proj_per_block():
    """Each block contributes 4 projections × 2 params each (weight +
    bias) = 8 tensors. With n_layer=2 the function should return 16."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    params = collect_attention_proj_params(model)
    # 2 layers × 4 projections × (weight + bias) = 16
    assert len(params) == 16


def test_collect_returns_params_that_belong_to_attn():
    """Every returned tensor must be either the weight or bias of one
    of the attention projections — defends against accidental capture
    of MLP/LN/embedding params."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    collected = set(map(id, collect_attention_proj_params(model)))

    expected = set()
    for block in model.blocks:
        attn = block.attn
        for name in ("q_proj", "k_proj", "v_proj", "proj"):
            proj = getattr(attn, name)
            base = getattr(proj, "linear", proj)
            expected.add(id(base.weight))
            if base.bias is not None:
                expected.add(id(base.bias))

    assert collected == expected


def test_collect_handles_lora_wrapping():
    """When LoRA is enabled, q/k/v/proj are wrapped in `LoRALinear`. The
    collector should reach through `.linear` and grab the base weight/
    bias — not the LoRA A/B matrices (those have their own freeze
    semantics)."""
    cfg = _tiny_cfg(lora_rank=4, lora_alpha=8.0, lora_dropout=0.0)
    model = TitansMAGGPT2(cfg)
    params = collect_attention_proj_params(model)
    # Still 16 (4 projs × 2 layers × {weight, bias}); LoRA A/B are NOT
    # included here.
    assert len(params) == 16
    # None of them should be a lora_A or lora_B tensor.
    for p in params:
        # Identify by walking named_parameters and comparing.
        pass  # The assertion structure below covers this.

    # Build a name → param map. Verify that exactly the expected names
    # appear, and that no lora_A/lora_B is among them.
    name_to_param = {n: pp for n, pp in model.named_parameters()}
    ids = set(map(id, params))
    for name, pp in name_to_param.items():
        if id(pp) in ids:
            assert ".linear.weight" in name or ".linear.bias" in name, (
                f"unexpected param in attention freeze set: {name}"
            )


# ---------------------------------------------------------------------------
# run_training honors the freeze schedule
# ---------------------------------------------------------------------------


def test_run_training_signature_includes_freeze_attention_steps():
    sig = inspect.signature(run_training)
    assert "freeze_attention_steps" in sig.parameters


def test_run_training_freezes_attention_params_before_phase_1():
    """At step 0, with freeze_attention_steps > 0, the attention
    projections should be frozen (requires_grad=False). Memory pathway
    + MLP should remain trainable."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    loader = ParallelStreamLoader(
        _stream(), batch_size=4, chunk_size=4, eot_id=50256,
    )

    # Run training with freeze active. After step 0 inits, the attention
    # params should be frozen. We run a single step and inspect.
    run_training(
        model=model, optimizer=opt, loader=loader,
        device=torch.device("cpu"),
        max_steps=1, warmup_steps=0, accum_steps=1,
        log_every=100, save_every=None, show_progress=False,
        freeze_attention_steps=2,
    )
    # After step 0 — we've completed 1 step. freeze_attention_steps=2
    # means freeze is still active (release fires when `step == N`, and
    # we ran step 0 and 1 isn't done yet); attention should still be
    # frozen.
    attn_params = collect_attention_proj_params(model)
    for p in attn_params:
        assert p.requires_grad is False, (
            "attention projection should be frozen during phase 1"
        )


def test_run_training_unfreezes_attention_at_step_N():
    """After max_steps reaches freeze_attention_steps, the unfreeze
    branch should have fired and all attention params should have
    requires_grad=True."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    loader = ParallelStreamLoader(
        _stream(), batch_size=4, chunk_size=4, eot_id=50256,
    )

    run_training(
        model=model, optimizer=opt, loader=loader,
        device=torch.device("cpu"),
        max_steps=3, warmup_steps=0, accum_steps=1,
        log_every=100, save_every=None, show_progress=False,
        freeze_attention_steps=1,
    )

    # After running 3 steps (>= freeze_attention_steps=1), attention
    # should be released.
    attn_params = collect_attention_proj_params(model)
    for p in attn_params:
        assert p.requires_grad is True, (
            "attention projection should be unfrozen after phase 1 release"
        )


def test_run_training_keeps_memory_and_mlp_trainable_during_freeze():
    """The freeze should ONLY touch attention; memory pathway, MLP,
    LayerNorms must stay trainable so they can absorb the gradient
    signal in phase 1."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    loader = ParallelStreamLoader(
        _stream(), batch_size=4, chunk_size=4, eot_id=50256,
    )

    run_training(
        model=model, optimizer=opt, loader=loader,
        device=torch.device("cpu"),
        max_steps=1, warmup_steps=0, accum_steps=1,
        log_every=100, save_every=None, show_progress=False,
        freeze_attention_steps=5,
    )

    attn_param_ids = set(map(id, collect_attention_proj_params(model)))
    for name, p in model.named_parameters():
        if id(p) in attn_param_ids:
            assert p.requires_grad is False
        else:
            # Non-attention param: should be trainable (modulo any
            # unrelated freeze flags, which this test doesn't use).
            assert p.requires_grad is True, (
                f"non-attention param wrongly frozen: {name}"
            )


def test_run_training_with_freeze_steps_0_is_a_noop():
    """The default `freeze_attention_steps=0` must leave attention
    fully trainable from step 0 — preserves backward compatibility
    with every existing training recipe."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    opt = build_optimizer(model)
    loader = ParallelStreamLoader(
        _stream(), batch_size=4, chunk_size=4, eot_id=50256,
    )

    run_training(
        model=model, optimizer=opt, loader=loader,
        device=torch.device("cpu"),
        max_steps=2, warmup_steps=0, accum_steps=1,
        log_every=100, save_every=None, show_progress=False,
        # freeze_attention_steps=0 (default)
    )

    for p in collect_attention_proj_params(model):
        assert p.requires_grad is True


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_finetune_cli_exposes_freeze_attention_steps_flag():
    from cli.finetune import build_parser
    parser = build_parser()
    assert "--freeze-attention-steps" in parser.format_help()
