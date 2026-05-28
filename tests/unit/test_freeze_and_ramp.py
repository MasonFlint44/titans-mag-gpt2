"""Tests for the TPTT-inspired training-regime helpers in cli.train:

- `freeze_backbone(model)` — sets requires_grad=False on backbone params,
  leaves NMM / MAG-gate / out_scale / persistent_mem trainable.
- `collect_out_scale_params(model)` — returns every per-block out_scale
  Parameter, cached at run_training startup for the ramp inner loop.
- `gate_ramp_value(step, ramp_steps, target)` — linear ramp schedule.

These helpers replicate (in spirit) TPTT's LoRA-only + LiZACallback
training regime, the motivation being that fully fine-tuning a 125M-param
backbone alongside a fresh memory module dilutes the gradient signal that
should concentrate on the memory mechanism.
"""

from __future__ import annotations

import pytest

import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import (
    EMBEDDING_SUBSTRINGS,
    FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS,
    collect_out_scale_params,
    freeze_backbone,
    freeze_embeddings_only,
    gate_ramp_value,
)


def _tiny_model(**cfg_overrides) -> TitansMAGGPT2:
    """Minimal TitansMAGGPT2 that exercises every memory-path param class
    (NMM internals, gamma_mem, out_scale, persistent_mem) at gpt2_small-
    shaped dimensions but tiny depth/seq for speed."""
    cfg = TitansConfig.gpt2_small(
        finetune_mode=True, chunk_size=64, block_size=64, **cfg_overrides,
    )
    return TitansMAGGPT2(cfg)


# ---------------------------------------------------------------------------
# freeze_backbone
# ---------------------------------------------------------------------------

def test_freeze_backbone_returns_partition_counts():
    """Returns (frozen, trainable) tensor counts. The sum must equal the
    total parameter-tensor count — every param is in exactly one bucket."""
    model = _tiny_model()
    total = sum(1 for _ in model.parameters())
    n_frozen, n_train = freeze_backbone(model)
    assert n_frozen + n_train == total


def test_freeze_backbone_keeps_memory_path_trainable():
    """Every param matching one of the memory-path substrings must remain
    trainable. If this regresses, the freeze accidentally turns off the
    NMM's own training — defeating the whole point of the flag."""
    model = _tiny_model()
    freeze_backbone(model)
    for name, p in model.named_parameters():
        if any(s in name for s in FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS):
            assert p.requires_grad, (
                f"memory-path param {name!r} was frozen — "
                f"freeze_backbone should have kept it trainable"
            )


def test_freeze_backbone_freezes_non_memory_params():
    """Every param NOT matching a memory-path substring must be frozen.
    Without this we'd still be training the backbone (defeating the
    TPTT-style gradient-concentration goal)."""
    model = _tiny_model()
    freeze_backbone(model)
    for name, p in model.named_parameters():
        if not any(s in name for s in FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS):
            assert not p.requires_grad, (
                f"backbone param {name!r} stayed trainable after "
                f"freeze_backbone — must be frozen"
            )


def test_freeze_backbone_idempotent():
    """Calling freeze_backbone twice must produce the same result. This
    protects against subtle state mutation if the function is called both
    in main() and accidentally re-invoked from run_training."""
    model = _tiny_model()
    a_frozen, a_train = freeze_backbone(model)
    b_frozen, b_train = freeze_backbone(model)
    assert (a_frozen, a_train) == (b_frozen, b_train)


def test_freeze_backbone_freezes_attention_and_mlp_explicitly():
    """Spot-check the most important backbone params: attention Q/K/V/O
    projections and the per-block MLP. If these accidentally stayed
    trainable, the gradient would still be split across the bulk of the
    backbone and the whole point of freezing is lost."""
    model = _tiny_model()
    freeze_backbone(model)
    backbone_patterns = (".attn.", ".mlp.", "wte", "wpe", "ln_f")
    for name, p in model.named_parameters():
        if any(s in name for s in backbone_patterns):
            assert not p.requires_grad, (
                f"expected backbone param {name!r} to be frozen, "
                f"but requires_grad=True"
            )


def test_freeze_backbone_keeps_out_scale_trainable():
    """out_scale is the memory-path scalar that the gate ramp will
    overwrite. Even though it's tiny (768 params/block), it must remain
    in `requires_grad=True` so the optimizer can take over after the ramp.
    Regression check on the substring set."""
    model = _tiny_model()
    freeze_backbone(model)
    found_out_scale = False
    for name, p in model.named_parameters():
        if "out_scale" in name:
            found_out_scale = True
            assert p.requires_grad, (
                f"out_scale param {name!r} got frozen — gate ramp would "
                f"then have no parameter to release back to the optimizer"
            )
    assert found_out_scale, "no out_scale params found in model — fixture broken"


# ---------------------------------------------------------------------------
# freeze_embeddings_only
# ---------------------------------------------------------------------------

def test_freeze_embeddings_only_freezes_wte_wpe_ln_f():
    """The whole point of the softer freeze: ONLY the input/output
    representation params are immobilized. Transformer blocks stay free
    to adapt to the NMM-augmented residual stream."""
    model = _tiny_model()
    freeze_embeddings_only(model)
    for name, p in model.named_parameters():
        if any(s in name for s in EMBEDDING_SUBSTRINGS):
            assert not p.requires_grad, (
                f"embedding param {name!r} stayed trainable — "
                f"freeze_embeddings_only must freeze wte/wpe/ln_f"
            )


def test_freeze_embeddings_only_keeps_transformer_blocks_trainable():
    """Attention + MLP MUST remain trainable so they can learn to attend
    to NMM-modulated tokens. If this regresses, the softer freeze
    degenerates to the catastrophic full-freeze behavior."""
    model = _tiny_model()
    freeze_embeddings_only(model)
    found_attn = False
    found_mlp = False
    for name, p in model.named_parameters():
        if ".attn." in name and "ln_f" not in name:
            found_attn = True
            assert p.requires_grad, (
                f"attention param {name!r} got frozen — defeats the point "
                f"of the embedding-only freeze"
            )
        if ".mlp." in name:
            found_mlp = True
            assert p.requires_grad, (
                f"MLP param {name!r} got frozen — defeats the point of "
                f"the embedding-only freeze"
            )
    assert found_attn, "no attention params found in model — fixture broken"
    assert found_mlp, "no MLP params found in model — fixture broken"


def test_freeze_embeddings_only_keeps_memory_path_trainable():
    """Memory path stays trainable under the softer freeze, same as
    under --freeze-backbone."""
    model = _tiny_model()
    freeze_embeddings_only(model)
    for name, p in model.named_parameters():
        if any(s in name for s in FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS):
            assert p.requires_grad, (
                f"memory-path param {name!r} got frozen — must stay "
                f"trainable for the NMM to learn"
            )


def test_freeze_embeddings_only_keeps_more_trainable_than_freeze_backbone():
    """Sanity check the relationship between the two freeze modes:
    embedding-only freeze MUST leave strictly more params trainable than
    full backbone freeze. If they produce the same count, one of the
    helpers is wrong."""
    model_a = _tiny_model()
    model_b = _tiny_model()
    _, n_train_emb = freeze_embeddings_only(model_a)
    _, n_train_full = freeze_backbone(model_b)
    assert n_train_emb > n_train_full, (
        f"freeze_embeddings_only left {n_train_emb} trainable, "
        f"freeze_backbone left {n_train_full}; softer freeze should "
        f"leave strictly more trainable"
    )


def test_freeze_embeddings_only_returns_partition_counts():
    """Returned (frozen, trainable) counts sum to the total parameter
    tensor count — no double-counting or omissions."""
    model = _tiny_model()
    total = sum(1 for _ in model.parameters())
    n_frozen, n_train = freeze_embeddings_only(model)
    assert n_frozen + n_train == total


# ---------------------------------------------------------------------------
# collect_out_scale_params
# ---------------------------------------------------------------------------

def test_collect_out_scale_params_returns_one_per_block():
    """gpt2_small has 12 NMM blocks → 12 out_scale parameters. If a future
    refactor accidentally moves out_scale or renames it, this test catches
    the ramp-loop silently iterating over an empty list."""
    model = _tiny_model()
    params = collect_out_scale_params(model)
    assert len(params) == 12


def test_collect_out_scale_params_returns_tensors_not_names():
    """Returned objects must be Parameters that the ramp loop can write
    into via `.data.fill_(...)`. Returning names instead would crash at
    runtime."""
    model = _tiny_model()
    params = collect_out_scale_params(model)
    for p in params:
        assert isinstance(p, torch.nn.Parameter)
        # The ramp body writes `p.data.fill_(val)`; verify .data exists.
        assert hasattr(p, "data")


def test_collect_out_scale_params_returns_per_channel_vectors():
    """out_scale shape is [n_embd] per block — a per-channel scaling
    vector. If a refactor accidentally changed it to a scalar or matrix,
    `data.fill_(val)` would still work but produce different broadcast
    semantics in the forward pass."""
    model = _tiny_model()
    params = collect_out_scale_params(model)
    for p in params:
        assert p.shape == (model.config.n_embd,), (
            f"expected out_scale shape ({model.config.n_embd},), got "
            f"{tuple(p.shape)}"
        )


# ---------------------------------------------------------------------------
# gate_ramp_value
# ---------------------------------------------------------------------------

def test_gate_ramp_first_step_is_one_over_ramp_steps_fraction():
    """Step 0 should be ramp_target / ramp_steps — i.e. ~0 but not
    literally 0. A literal 0 here would mean the model trains for an
    entire step with the memory effectively off, defeating the gentle-
    open intent of the ramp."""
    val = gate_ramp_value(step=0, ramp_steps=100, target=0.5)
    assert val == pytest.approx(0.5 / 100)


def test_gate_ramp_last_in_ramp_step_reaches_target():
    """Step (ramp_steps - 1) is the LAST step still under ramp control —
    it must see the full target so the model trains at least once at the
    target before the optimizer takes over at step == ramp_steps."""
    val = gate_ramp_value(step=99, ramp_steps=100, target=0.5)
    assert val == pytest.approx(0.5)


def test_gate_ramp_post_ramp_returns_target():
    """For step >= ramp_steps the function returns target. Callers stop
    invoking it after the ramp, but for safety this should be defined."""
    assert gate_ramp_value(100, 100, 0.5) == pytest.approx(0.5)
    assert gate_ramp_value(1000, 100, 0.5) == pytest.approx(0.5)


def test_gate_ramp_is_monotonic_increasing():
    """The whole point is a SMOOTH OPEN. Any non-monotonicity in the
    schedule would cause oscillation in the memory contribution and
    confuse the model during early training."""
    vals = [gate_ramp_value(s, 100, 0.5) for s in range(100)]
    for a, b in zip(vals, vals[1:]):
        assert a <= b, (
            f"ramp schedule must be monotonic non-decreasing; got "
            f"{a} then {b}"
        )


def test_gate_ramp_raises_on_nonpositive_ramp_steps():
    """ramp_steps <= 0 is a configuration error — the caller should
    short-circuit with `if ramp_steps > 0`. Failing loud is preferable
    to silent divide-by-zero or returning target immediately (which
    would skip the ramp without flagging the bug)."""
    with pytest.raises(ValueError, match="ramp_steps must be > 0"):
        gate_ramp_value(step=0, ramp_steps=0, target=0.5)
    with pytest.raises(ValueError, match="ramp_steps must be > 0"):
        gate_ramp_value(step=0, ramp_steps=-1, target=0.5)


def test_gate_ramp_target_zero_returns_zero_throughout():
    """Target 0 is a degenerate but valid configuration — the ramp does
    nothing and out_scale stays at 0 throughout. Verifies the linear
    scaling is `target * fraction`, not just `fraction` (which would
    ignore the target)."""
    for step in (0, 50, 99):
        assert gate_ramp_value(step, 100, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Integration: freeze + ramp work together
# ---------------------------------------------------------------------------

def test_optimizer_sees_only_trainable_params_under_freeze():
    """Build the 4-group optimizer AFTER freezing — every Adam moment
    should correspond to a trainable param. The optimizer's param-group
    walk in `build_optimizer` skips `requires_grad=False` tensors, so the
    total optimizer param count must equal the post-freeze trainable
    count."""
    from cli.train import build_optimizer
    model = _tiny_model()
    n_frozen, n_train = freeze_backbone(model)
    optim = build_optimizer(model, use_8bit=False)
    n_in_groups = sum(len(g["params"]) for g in optim.param_groups)
    assert n_in_groups == n_train, (
        f"optimizer holds {n_in_groups} tensors but {n_train} are "
        f"trainable; freeze + build_optimizer should agree"
    )


def test_ramp_overwrite_pattern_does_not_break_optimizer_construction():
    """Simulate what `run_training` does at startup: freeze backbone,
    set out_scale.requires_grad=False, build optimizer. The optimizer
    should NOT include out_scale params in its groups — only trainable
    params should be there. After the ramp ends, run_training flips
    out_scale.requires_grad=True; the optimizer.step() call will skip
    out_scale until its first .grad arrives (standard PyTorch behavior).
    """
    from cli.train import build_optimizer
    model = _tiny_model()
    freeze_backbone(model)
    out_scale_ps = collect_out_scale_params(model)
    for p in out_scale_ps:
        p.requires_grad = False
    optim = build_optimizer(model, use_8bit=False)
    n_in_groups = sum(len(g["params"]) for g in optim.param_groups)
    # out_scale tensors should NOT be in optimizer groups since they're
    # frozen during the ramp.
    out_scale_in_optim = sum(
        1
        for g in optim.param_groups
        for p in g["params"]
        if any(p is q for q in out_scale_ps)
    )
    assert out_scale_in_optim == 0, (
        f"out_scale params should be excluded from optimizer groups "
        f"during the ramp; found {out_scale_in_optim} in groups"
    )
