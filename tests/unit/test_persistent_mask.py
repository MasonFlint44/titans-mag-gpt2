"""Phase 2.1 / 2.4 — persistent token augmented mask."""

import torch

from config import TitansConfig
from model.block import TitansMAGBlock


def _tiny_block(finetune_mode=True, N_p=4, T=8, use_swa=False, swa_window=4):
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=T, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=N_p,
        use_swa=use_swa, swa_window=swa_window,
        finetune_mode=finetune_mode,
    )
    return TitansMAGBlock(cfg)


def test_aug_mask_shape():
    block = _tiny_block(N_p=4)
    mask = block._aug_mask(T=8)
    assert mask.shape == (12, 12)


def test_persistent_to_persistent_block_is_open():
    block = _tiny_block(N_p=4)
    mask = block._aug_mask(T=8)
    # Top-left N_p x N_p must be zeros (full attention among persistent tokens).
    assert torch.all(mask[:4, :4] == 0.0)


def test_persistent_to_real_block_is_minus_inf():
    """Persistent tokens MUST NOT attend to real tokens — top-right block."""
    block = _tiny_block(N_p=4)
    mask = block._aug_mask(T=8)
    assert torch.all(mask[:4, 4:] == float("-inf"))


def test_real_to_persistent_block_is_open():
    """Real tokens always see all persistent tokens — bottom-left block."""
    block = _tiny_block(N_p=4)
    mask = block._aug_mask(T=8)
    assert torch.all(mask[4:, :4] == 0.0)


def test_real_to_real_block_is_upper_triangular_causal():
    block = _tiny_block(N_p=4)
    mask = block._aug_mask(T=8)
    real = mask[4:, 4:]
    # Lower triangle + diagonal = 0; strict upper triangle = -inf.
    for i in range(8):
        for j in range(8):
            if j <= i:
                assert real[i, j] == 0.0, f"({i},{j}) should be open"
            else:
                assert real[i, j] == float("-inf"), f"({i},{j}) should be masked"


def test_swa_banded_mask_attends_only_to_window():
    """G136: with use_swa=True, real token i attends to real j in (i-W, i]."""
    W = 3
    block = _tiny_block(N_p=2, T=8, use_swa=True, swa_window=W)
    mask = block._aug_mask(T=8)
    real = mask[2:, 2:]
    for i in range(8):
        for j in range(8):
            if i - W < j <= i:
                assert real[i, j] == 0.0, f"({i},{j}) should be open (within window)"
            else:
                assert real[i, j] == float("-inf"), f"({i},{j}) should be masked"


def test_aug_mask_dtype_matches_passed_dtype():
    block = _tiny_block()
    mask = block._aug_mask(T=8, dtype=torch.bfloat16)
    assert mask.dtype == torch.bfloat16
