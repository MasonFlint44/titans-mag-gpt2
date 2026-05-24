"""Phase 4.3 — checkpoint save/load round-trip + compute_nmm_norm helper."""

import dataclasses
import os
import tempfile

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from train import (
    BASE_LR_GPT2,
    BASE_LR_NMM,
    base_lrs_from_constants,
    build_optimizer,
    compute_nmm_norm,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def _tiny_setup(finetune_mode=False):
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=finetune_mode,
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)
    return cfg, model, optimizer


def _fake_batch(cfg, B=2, T=4):
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    db = torch.zeros(B, T, dtype=torch.bool)
    db[:, 0] = True
    return (idx, db)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_state_dict():
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=42, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert ckpt["step"] == 42
        assert set(ckpt["state_dict"].keys()) == set(model.state_dict().keys())
        for k in ckpt["state_dict"]:
            assert torch.equal(ckpt["state_dict"][k], model.state_dict()[k])


def test_save_load_round_trip_preserves_config():
    cfg, model, opt = _tiny_setup(finetune_mode=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        # Round-trip via dict -> TitansConfig must reconstruct equivalent config.
        cfg2 = TitansConfig(**ckpt["config"])
        assert dataclasses.asdict(cfg) == dataclasses.asdict(cfg2)


def test_save_load_optimizer_state_populated_after_step():
    """G153: after a real .step(), Adam moments exp_avg / exp_avg_sq exist."""
    cfg, model, opt = _tiny_setup()
    batch = _fake_batch(cfg)
    train_step(model, batch, None, opt, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=1, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert "optimizer" in ckpt
        # The state dict's 'state' must be non-empty (at least one param has Adam moments).
        opt_state = ckpt["optimizer"]["state"]
        assert len(opt_state) > 0, "Adam moments not saved"


def test_load_works_under_weights_only_false():
    """G168: torch.load(weights_only=True) (the PyTorch 2.6+ default) would
    reject our nested optimizer state on some version combos. load_checkpoint
    passes weights_only=False explicitly — verify it returns the dict."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert isinstance(ckpt, dict)


# ---------------------------------------------------------------------------
# G219 — HF-init checkpoint with no 'optimizer' key
# ---------------------------------------------------------------------------

def test_resume_with_no_optimizer_key_does_not_raise():
    """G219: scripts/load_pretrained.py emits {state_dict, config, step}
    without 'optimizer'. The resume path must tolerate the missing key.
    Simulate the path: load checkpoint, build fresh optimizer, only call
    load_state_dict if the key is present."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hf_init.pt")
        # Save like load_pretrained does — no optimizer key.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": dataclasses.asdict(cfg),
                "step": 0,
            },
            path,
        )
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        # Caller-side pattern from G219:
        cfg2 = TitansConfig(**ckpt["config"])
        model2 = TitansMAGGPT2(cfg2)
        model2.load_state_dict(ckpt["state_dict"])
        opt2 = build_optimizer(model2)
        if "optimizer" in ckpt:
            opt2.load_state_dict(ckpt["optimizer"])
        # No exception — the missing-optimizer branch is allowed.
        assert opt2 is not None


# ---------------------------------------------------------------------------
# G221 — resume sequence ends in train()
# ---------------------------------------------------------------------------

def test_resume_ends_with_model_train_mode():
    """G221: explicit model.train() at end of resume protects against any
    intervening eval-mode code (smoke perplexity, sample generation).
    The resume sequence is caller-owned, but verify the pattern works."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        cfg2 = TitansConfig(**ckpt["config"])
        model2 = TitansMAGGPT2(cfg2)
        model2.load_state_dict(ckpt["state_dict"])
        model2.eval()  # simulate a smoke-test eval that came before final train()
        model2.train()  # the G221 defensive call
        assert model2.training


# ---------------------------------------------------------------------------
# G172 — compute_nmm_norm
# ---------------------------------------------------------------------------

def test_compute_nmm_norm_returns_None_when_states_is_None():
    assert compute_nmm_norm(None) is None


def test_compute_nmm_norm_returns_one_float_per_layer():
    cfg, model, opt = _tiny_setup()
    states = [block.nmm.init_state(2, torch.device("cpu")) for block in model.blocks]
    norms = compute_nmm_norm(states)
    assert isinstance(norms, list)
    assert len(norms) == cfg.n_layer
    for n in norms:
        assert isinstance(n, float)
        assert n > 0  # at init the Xavier-uniform weights have non-zero Frobenius


def test_compute_nmm_norm_increases_when_M_is_larger():
    """Sanity: doubling all entries of M should ~double the reported norm."""
    cfg, model, opt = _tiny_setup()
    states = [block.nmm.init_state(2, torch.device("cpu")) for block in model.blocks]
    norms1 = compute_nmm_norm(states)
    # Scale all M entries by 2.0.
    for M, S in states:
        for k in M:
            M[k] = M[k] * 2.0
    norms2 = compute_nmm_norm(states)
    for n1, n2 in zip(norms1, norms2):
        assert abs(n2 / n1 - 2.0) < 1e-5
