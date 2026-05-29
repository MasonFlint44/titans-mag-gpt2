"""LoRALinear unit tests.

Covers:
  - At init, LoRA contribution is exactly zero (lora_B=0); LoRALinear output
    matches its underlying base nn.Linear.
  - When `freeze_base=True`, base weight/bias have requires_grad=False;
    lora_A, lora_B stay trainable.
  - Backward delivers gradient to lora_A and lora_B (but not to base).
  - The scaling factor is alpha/rank, applied to the LoRA path.
"""
import math

import pytest
import torch
import torch.nn as nn

from model.lora import LoRALinear


def test_init_lora_B_zero_so_layer_equals_base_at_step_0():
    """B is zero-init by design. Total output = base + scaling · B·A·x =
    base + scaling · 0 = base. Verifies the "drop-in" property: a model
    rebuilt with LoRA wrapping produces identical logits to one without,
    BEFORE any training."""
    torch.manual_seed(0)
    layer = LoRALinear(8, 16, bias=True, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(2, 5, 8)
    # The base nn.Linear's behavior:
    base_out = layer.linear(x)
    out = layer(x)
    assert torch.allclose(out, base_out, atol=1e-6)


def test_freeze_base_true_freezes_base_weight_and_bias():
    layer = LoRALinear(8, 16, bias=True, rank=4, freeze_base=True)
    assert layer.linear.weight.requires_grad is False
    assert layer.linear.bias.requires_grad is False
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True


def test_freeze_base_false_keeps_base_trainable():
    layer = LoRALinear(8, 16, bias=True, rank=4, freeze_base=False)
    assert layer.linear.weight.requires_grad is True
    assert layer.linear.bias.requires_grad is True


def test_scaling_factor_equals_alpha_over_rank():
    """The LoRA contribution is `scaling · B·A·x` where scaling=alpha/rank.
    We force lora_B to a known non-zero value and verify the output picks up
    exactly the expected scaling factor."""
    torch.manual_seed(0)
    layer = LoRALinear(4, 6, bias=False, rank=2, alpha=10.0, dropout=0.0)
    assert pytest.approx(layer.scaling) == 5.0  # alpha/rank = 10/2

    # Set lora_B to a known non-zero. lora_A keeps its Kaiming init.
    with torch.no_grad():
        layer.lora_B.fill_(1.0)
    # Also zero the base weight so the entire output IS the LoRA path.
    with torch.no_grad():
        layer.linear.weight.zero_()

    x = torch.randn(1, 1, 4)
    out = layer(x)
    # Hand-computed: scaling · B·A·x with B = all ones, base = 0
    # F.linear(x, A) = x @ A.T; then F.linear(_, B) = _ @ B.T
    expected = 5.0 * (x @ layer.lora_A.T) @ layer.lora_B.T
    assert torch.allclose(out, expected, atol=1e-5)


def test_backward_delivers_grad_to_lora_params_only_when_base_frozen():
    """With freeze_base=True, after backward only lora_A and lora_B have
    non-None grads on the LoRA-wrapped layer. base weight/bias must have
    grad=None (skipped entirely from the graph)."""
    layer = LoRALinear(4, 4, bias=True, rank=2, freeze_base=True)
    x = torch.randn(3, 4)
    y = layer(x).sum()
    y.backward()
    assert layer.linear.weight.grad is None
    assert layer.linear.bias.grad is None
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_rank_zero_rejected():
    """rank<=0 is meaningless — there's no adapter. The CLI's `lora_rank=0`
    path uses plain nn.Linear instead, so this constructor only sees rank>=1."""
    with pytest.raises(ValueError, match="rank must be > 0"):
        LoRALinear(4, 4, bias=True, rank=0)


def test_forward_shape_correctness():
    """Output shape matches (..., out_features) regardless of leading dims."""
    layer = LoRALinear(8, 16, bias=True, rank=4)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 16)


def test_kaiming_init_for_lora_A_nonzero():
    """lora_A is initialized Kaiming-uniform, so it's non-zero. lora_B is
    explicitly zero. Together: LoRA contribution at step 0 is zero (because
    B=0), but post-init A has structure ready to be projected through B
    once B starts moving."""
    torch.manual_seed(0)
    layer = LoRALinear(8, 16, rank=4)
    assert layer.lora_A.abs().sum() > 0
    assert layer.lora_B.abs().sum() == 0


# ---------------------------------------------------------------------------
# Integration with the model + config
# ---------------------------------------------------------------------------


def test_attention_module_uses_lora_when_config_rank_gt_0():
    """Building CausalSelfAttention with lora_rank>0 wraps q/k/v/proj in
    LoRALinear. With lora_rank=0, attention uses plain nn.Linear."""
    from model.block import CausalSelfAttention

    # No LoRA: plain Linear.
    attn = CausalSelfAttention(n_embd=16, n_head=4, lora_rank=0)
    assert isinstance(attn.q_proj, nn.Linear)
    assert not isinstance(attn.q_proj, LoRALinear)

    # With LoRA: all four projections are wrapped.
    attn_lora = CausalSelfAttention(n_embd=16, n_head=4, lora_rank=8)
    assert isinstance(attn_lora.q_proj, LoRALinear)
    assert isinstance(attn_lora.k_proj, LoRALinear)
    assert isinstance(attn_lora.v_proj, LoRALinear)
    assert isinstance(attn_lora.proj, LoRALinear)


def test_model_with_lora_at_init_matches_no_lora_logits():
    """Building a TitansMAGGPT2 with LoRA wrapping must produce identical
    forward output to the same model without LoRA, BEFORE any training.
    LoRA's zero-init of lora_B guarantees this — defends against a
    regression where the wrapping accidentally introduces a non-zero
    contribution at step 0 (which would silently break parity-with-HF
    tests).

    Approach: build both models, then copy the no-LoRA's full state_dict
    into the LoRA model's matching keys (LoRA-specific keys stay at their
    zero/Kaiming init). Same backbone behavior, plus lora_B=0 contribution.
    """
    from config import TitansConfig
    from model.titans_gpt2 import TitansMAGGPT2

    cfg = TitansConfig(
        n_layer=2, n_head=4, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=64, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=True,
    )
    torch.manual_seed(42)
    m_no_lora = TitansMAGGPT2(cfg)

    cfg_lora = TitansConfig(
        n_layer=2, n_head=4, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=64, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=True,
        lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
    )
    torch.manual_seed(42)
    m_lora = TitansMAGGPT2(cfg_lora)

    # Copy every overlapping param. The attention projections renamed
    # `q_proj.weight` -> `q_proj.linear.weight` under LoRA, so they don't
    # appear by name in m_lora; we handle them separately. All other
    # named params (LN, MLP, NMM, embeddings) share names across the two.
    src_state = m_no_lora.state_dict()
    dst_state = m_lora.state_dict()
    with torch.no_grad():
        for k, v in src_state.items():
            if k in dst_state and dst_state[k].shape == v.shape:
                dst_state[k].copy_(v)
        # Attention projections: src has e.g. "blocks.0.attn.q_proj.weight",
        # dst has "blocks.0.attn.q_proj.linear.weight". Map them.
        for k, v in src_state.items():
            if ".attn." in k and any(
                f".{name}.weight" in k or f".{name}.bias" in k
                for name in ("q_proj", "k_proj", "v_proj", "proj")
            ):
                # Insert ".linear" before the final ".weight"/".bias"
                if k.endswith(".weight"):
                    dst_key = k[:-len(".weight")] + ".linear.weight"
                else:
                    dst_key = k[:-len(".bias")] + ".linear.bias"
                if dst_key in dst_state:
                    dst_state[dst_key].copy_(v)

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    m_no_lora.eval()
    m_lora.eval()
    with torch.no_grad():
        out_nl, _ = m_no_lora(idx)
        out_l, _ = m_lora(idx)

    assert torch.allclose(out_nl, out_l, atol=1e-5), (
        f"LoRA wrapping changed logits at init "
        f"(max diff {(out_nl - out_l).abs().max().item()}); "
        f"this breaks the drop-in compatibility property."
    )


def test_lora_params_routed_to_no_decay_group():
    """LoRA's A/B matrices should land in the no-decay group (decay is
    inappropriate for LoRA adapters by standard convention)."""
    from cli.train import _is_no_decay
    assert _is_no_decay("blocks.0.attn.q_proj.lora_A")
    assert _is_no_decay("blocks.0.attn.q_proj.lora_B")


def test_config_validates_lora_dropout_range():
    """dropout must be in [0, 1). Reject 1.0 (no information passes) and
    negative values."""
    from config import TitansConfig
    with pytest.raises(ValueError, match="lora_dropout"):
        TitansConfig(lora_rank=8, lora_dropout=1.0)
    with pytest.raises(ValueError, match="lora_dropout"):
        TitansConfig(lora_rank=8, lora_dropout=-0.1)


def test_config_validates_lora_alpha_positive():
    """alpha must be positive when LoRA is enabled."""
    from config import TitansConfig
    with pytest.raises(ValueError, match="lora_alpha"):
        TitansConfig(lora_rank=8, lora_alpha=0.0)
    with pytest.raises(ValueError, match="lora_alpha"):
        TitansConfig(lora_rank=8, lora_alpha=-1.0)


def test_config_lora_rank_zero_disables_lora():
    """rank=0 (default) means no LoRA — model uses plain Linear in
    attention. Validation should accept rank=0 without complaints about
    alpha or dropout."""
    from config import TitansConfig
    # Should not raise even with weird alpha/dropout — they're ignored.
    cfg = TitansConfig(lora_rank=0, lora_alpha=-99.0, lora_dropout=2.0)
    assert cfg.lora_rank == 0
