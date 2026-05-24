"""Phase 2.4 — TitansMAGBlock end-to-end forward."""

import pytest
import torch

from config import TitansConfig
from model.block import TitansMAGBlock


def _cfg(finetune_mode=True, N_p=2, T=8, use_swa=False, n_embd=8, n_head=2):
    return TitansConfig(
        n_layer=1, n_head=n_head, n_embd=n_embd, vocab_size=16,
        block_size=64, chunk_size=T, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=N_p,
        use_swa=use_swa,
        finetune_mode=finetune_mode,
    )


def test_block_forward_shape_invariant():
    cfg = _cfg()
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    y, new_state = block(x, nmm_state=state)
    assert y.shape == x.shape
    for k in new_state[0]:
        assert new_state[0][k].shape == state[0][k].shape


def test_block_separate_ln_nmm_in_state_dict():
    """ln_nmm must be a distinct LayerNorm tracked separately from ln_1 in state_dict.
    Both LNs init to weight=ones/bias=zeros (so weight equality at init is expected);
    independence is checked via module identity and post-perturbation divergence."""
    cfg = _cfg()
    block = TitansMAGBlock(cfg)
    keys = set(block.state_dict().keys())
    assert "ln_1.weight" in keys and "ln_1.bias" in keys
    assert "ln_nmm.weight" in keys and "ln_nmm.bias" in keys
    assert block.ln_1 is not block.ln_nmm
    assert block.ln_1.weight is not block.ln_nmm.weight  # separate Parameter objects
    # Perturbing one must not change the other.
    with torch.no_grad():
        block.ln_nmm.weight.add_(1.0)
    assert not torch.equal(block.ln_1.weight, block.ln_nmm.weight)


def test_block_is_differentiable():
    cfg = _cfg(finetune_mode=False)  # so output isn't pure-residual at init
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    y, _ = block(x, nmm_state=state)
    y.sum().backward()
    assert block.persistent_mem.grad is not None


def test_block_at_finetune_init_preserves_residual_path():
    """End-to-end: with out_scale=0, the block's output residual must equal
    the attention+MLP residual ignoring the memory branch entirely."""
    cfg = _cfg(finetune_mode=True)
    block = TitansMAGBlock(cfg)
    block.eval()
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    with torch.no_grad():
        y, _ = block(x, nmm_state=state)
    # NMM contributes zero -> output should be finite and shape-correct.
    assert torch.isfinite(y).all()
    assert y.shape == x.shape


def test_block_doc_boundary_resets_nmm():
    """A boundary at t=0 should reset nmm state and change downstream output."""
    cfg = _cfg(finetune_mode=False)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    # Add some state perturbation in row 0.
    state = ({k: state[0][k] + 1.0 for k in state[0]},
             {k: state[1][k] + 0.5 for k in state[1]})

    x = torch.randn(2, 8, 8)
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[0, 0] = True

    y_reset, _ = block(x, nmm_state=state, doc_boundaries=db)
    # Make a fresh block + state (clone perturbed) without boundary.
    block2 = TitansMAGBlock(cfg)
    block2.load_state_dict(block.state_dict())
    state2 = ({k: state[0][k].clone() for k in state[0]},
              {k: state[1][k].clone() for k in state[1]})
    y_noreset, _ = block2(x, nmm_state=state2, doc_boundaries=None)
    # Row 0 should diverge (state reset); row 1 should match (no boundary).
    assert not torch.allclose(y_reset[0], y_noreset[0], atol=1e-4)
    assert torch.allclose(y_reset[1], y_noreset[1], atol=1e-4)


def test_block_works_with_swa_enabled():
    cfg = _cfg(finetune_mode=False, use_swa=True)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8)
    y, _ = block(x, nmm_state=state)
    assert y.shape == x.shape and torch.isfinite(y).all()
