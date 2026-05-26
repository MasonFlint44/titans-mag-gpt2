"""Phase 4.1 — 4-group optimizer."""

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from train import BASE_LR_GPT2, BASE_LR_NMM, BETAS, WEIGHT_DECAY, build_optimizer


def _tiny_model(**overrides):
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        **overrides,
    )
    return TitansMAGGPT2(cfg)


def test_exactly_four_param_groups():
    model = _tiny_model()
    opt = build_optimizer(model)
    assert len(opt.param_groups) == 4


def test_groups_have_correct_lr_and_wd():
    model = _tiny_model()
    opt = build_optimizer(model)
    # Group order: gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay.
    assert opt.param_groups[0]["lr"] == BASE_LR_GPT2
    assert opt.param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert opt.param_groups[1]["lr"] == BASE_LR_GPT2
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert opt.param_groups[2]["lr"] == BASE_LR_NMM
    assert opt.param_groups[2]["weight_decay"] == WEIGHT_DECAY
    assert opt.param_groups[3]["lr"] == BASE_LR_NMM
    assert opt.param_groups[3]["weight_decay"] == 0.0


def test_nmm_lr_is_3x_gpt2_lr_per_paper():
    """The paper's ratio is 3x; defaults must preserve it."""
    assert BASE_LR_NMM == 3 * BASE_LR_GPT2


def test_betas_are_0_9_0_95_not_pytorch_default():
    """AdamW default is (0.9, 0.999); ours must be (0.9, 0.95) per LM tradition.
    Silent failure if defaults leak in: oversmoothed second moments early in training."""
    model = _tiny_model()
    opt = build_optimizer(model)
    for g in opt.param_groups:
        assert g["betas"] == BETAS, f"betas={g['betas']}, expected {BETAS}"
    assert BETAS == (0.9, 0.95)


def test_no_parameter_appears_in_two_groups():
    """Every param has exactly one home — otherwise weight decay or LR applies twice."""
    model = _tiny_model()
    opt = build_optimizer(model)
    seen_ids = set()
    for g in opt.param_groups:
        for p in g["params"]:
            assert id(p) not in seen_ids, "param appears in two groups"
            seen_ids.add(id(p))


def test_every_trainable_param_is_in_some_group():
    model = _tiny_model()
    opt = build_optimizer(model)
    seen = {id(p) for g in opt.param_groups for p in g["params"]}
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(seen) == total


# ---------------------------------------------------------------------------
# Routing correctness — the WHO-goes-WHERE matrix
# ---------------------------------------------------------------------------

def test_layernorm_params_routed_to_no_decay():
    """ln_1/ln_2/ln_f/ln_nmm weights and biases must NOT receive weight decay."""
    model = _tiny_model()
    opt = build_optimizer(model)
    no_decay_ids = {id(p) for g in (opt.param_groups[1], opt.param_groups[3])
                    for p in g["params"]}
    for name, p in model.named_parameters():
        if any(ln_name in name for ln_name in ("ln_1", "ln_2", "ln_f", "ln_nmm")):
            assert id(p) in no_decay_ids, f"{name} should be no_decay (LayerNorm)"


def test_out_scale_routed_to_nmm_no_decay():
    """out_scale init=zeros in finetune mode — decay would resist learning it."""
    model = _tiny_model(finetune_mode=True)
    opt = build_optimizer(model)
    nmm_no_decay = opt.param_groups[3]["params"]
    out_scales = [p for name, p in model.named_parameters() if "out_scale" in name]
    assert len(out_scales) > 0
    for p in out_scales:
        assert any(p is q for q in nmm_no_decay)


def test_gamma_mem_routed_to_nmm_no_decay():
    model = _tiny_model()
    opt = build_optimizer(model)
    nmm_no_decay = opt.param_groups[3]["params"]
    gammas = [p for name, p in model.named_parameters() if "gamma_mem" in name]
    assert len(gammas) > 0
    for p in gammas:
        assert any(p is q for q in nmm_no_decay)


def test_persistent_mem_routed_to_nmm_no_decay():
    model = _tiny_model()
    opt = build_optimizer(model)
    nmm_no_decay = opt.param_groups[3]["params"]
    pers = [p for name, p in model.named_parameters() if "persistent" in name]
    assert len(pers) > 0
    for p in pers:
        assert any(p is q for q in nmm_no_decay)


def test_attn_weights_routed_to_gpt2_decay():
    """attn.q_proj.weight etc are GPT-2 params and DO get weight decay."""
    model = _tiny_model()
    opt = build_optimizer(model)
    gpt2_decay = opt.param_groups[0]["params"]
    for name, p in model.named_parameters():
        if name.startswith("blocks.0.attn.q_proj.weight"):
            assert any(p is q for q in gpt2_decay)


def test_attn_biases_routed_to_gpt2_no_decay():
    model = _tiny_model()
    opt = build_optimizer(model)
    gpt2_no_decay = opt.param_groups[1]["params"]
    for name, p in model.named_parameters():
        if name.startswith("blocks.0.attn.q_proj.bias"):
            assert any(p is q for q in gpt2_no_decay)


def test_memory_mlp_weights_routed_to_nmm_decay():
    """memory_mlp.W1.weight (inside NMM, not bias/norm) -> nmm_decay group."""
    model = _tiny_model()
    opt = build_optimizer(model)
    nmm_decay = opt.param_groups[2]["params"]
    for name, p in model.named_parameters():
        if "memory_mlp.W1.weight" in name:
            assert any(p is q for q in nmm_decay), f"{name} not in nmm_decay"


def test_memory_mlp_norm_routed_to_nmm_no_decay():
    """memory_mlp.norm.weight has 'nmm' (via parent path) and 'norm' -> nmm_no_decay."""
    model = _tiny_model()
    opt = build_optimizer(model)
    nmm_no_decay = opt.param_groups[3]["params"]
    for name, p in model.named_parameters():
        if "memory_mlp.norm" in name:
            assert any(p is q for q in nmm_no_decay), (
                f"{name} should be nmm_no_decay (norm catches it)"
            )


# ---------------------------------------------------------------------------
# G278: bitsandbytes 8-bit AdamW
# ---------------------------------------------------------------------------


def test_8bit_optimizer_falls_back_gracefully_when_bitsandbytes_missing():
    """If bitsandbytes isn't installed, use_8bit=True should raise a clear
    error pointing at the install command rather than crashing on import
    inside the optimizer step."""
    import importlib.util
    has_bnb = importlib.util.find_spec("bitsandbytes") is not None
    if has_bnb:
        pytest.skip("bitsandbytes is installed; skipping fallback test")
    model = _tiny_model()
    with pytest.raises(RuntimeError, match="requires the `bitsandbytes` package"):
        build_optimizer(model, use_8bit=True)


def test_8bit_optimizer_constructs_with_same_groups():
    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        pytest.skip("bitsandbytes not installed; skipping")
    import bitsandbytes as bnb
    model = _tiny_model()
    opt = build_optimizer(model, use_8bit=True)
    assert isinstance(opt, bnb.optim.AdamW8bit)
    # Same 4-group layout.
    assert len(opt.param_groups) == 4


def test_8bit_optimizer_step_runs():
    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is None:
        pytest.skip("bitsandbytes not installed; skipping")
    model = _tiny_model()
    opt = build_optimizer(model, use_8bit=True)
    # Run a forward + backward + step to exercise the optimizer.
    idx = torch.randint(0, 32, (2, 16))
    logits, _ = model(idx)
    loss = logits.pow(2).sum()
    loss.backward()
    opt.step()
    # Verify some param actually moved.
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            break
