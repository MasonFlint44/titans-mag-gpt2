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


# ---------------------------------------------------------------------------
# T14 — softmax-NaN guard: no row is entirely -inf (TEST_PLAN §4 spec)
# ---------------------------------------------------------------------------

def test_aug_mask_no_row_is_fully_inf_standard_causal():
    """T14 — every row of the augmented mask must have at least one
    finite (= 0.0) entry. A fully-`-inf` row makes
    `softmax(-inf, -inf, ...)` produce NaN, killing the loss with no
    other signal. Confirm for the default (non-SWA) mask:
      - persistent rows always see ALL persistent columns -> open
      - real row i always sees itself -> at least 1 open column
    """
    block = _tiny_block(N_p=3, T=8, use_swa=False)
    mask = block._aug_mask(T=8)
    for i in range(mask.shape[0]):
        finite_count = (mask[i] == 0.0).sum().item()
        assert finite_count > 0, (
            f"row {i} of aug_mask is fully -inf — softmax would NaN. "
            f"Standard causal: persistent rows attend to all persistent "
            f"(N_p={3} open columns); real row i attends to itself + all "
            f"persistent (at least 1 + N_p = 4 open columns)."
        )


def test_aug_mask_no_row_is_fully_inf_with_swa_at_edge_window():
    """T14 — same invariant under SWA at the smallest valid window
    (swa_window=1). Every real row must still see itself (causal includes
    diagonal) plus all persistent columns. A swa_window=1 setup is the
    tightest possible window and the most likely to silently produce
    fully-`-inf` rows if the mask logic is wrong."""
    block = _tiny_block(N_p=2, T=6, use_swa=True, swa_window=1)
    mask = block._aug_mask(T=6)
    for i in range(mask.shape[0]):
        finite_count = (mask[i] == 0.0).sum().item()
        assert finite_count > 0, (
            f"row {i} of aug_mask is fully -inf under swa_window=1 — "
            f"softmax would NaN. Each real row should see itself + "
            f"all N_p={2} persistent."
        )


def test_aug_mask_softmax_produces_no_nan_in_attention_forward():
    """T14 — end-to-end: feed _aug_mask through scaled_dot_product_attention
    on a real Q, K, V; verify the output is all finite. This is the
    behavioral version of the structural test above — it catches any
    mask layout that makes some attention head's softmax NaN under
    realistic random inputs (e.g., a dtype-related -inf-cancellation
    bug in the additive mask)."""
    import torch.nn.functional as F

    block = _tiny_block(N_p=4, T=8, use_swa=True, swa_window=2)
    mask = block._aug_mask(T=8)
    N = mask.shape[0]
    n_head, head_dim = 2, 4
    q = torch.randn(1, n_head, N, head_dim)
    k = torch.randn(1, n_head, N, head_dim)
    v = torch.randn(1, n_head, N, head_dim)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.isfinite(y).all(), (
        "scaled_dot_product_attention with _aug_mask produced NaN — "
        "some row of the mask was fully -inf."
    )
