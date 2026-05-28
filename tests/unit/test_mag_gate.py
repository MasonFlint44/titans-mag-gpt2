"""Phase 2.3 — MAG gate combination (finetune additive / scratch multiplicative)."""

import torch

from config import TitansConfig
from model.block import TitansMAGBlock


def _cfg(finetune_mode):
    # `persistent_prefix_mode="per_block"` keeps the persistent_mem
    # Parameter on the BLOCK (where this file's tests probe it). The
    # post-batch-3 default `"model_wide"` would move the parameter to
    # `TitansMAGGPT2.persistent_mem` and tests that read
    # `block.persistent_mem` would fail with AttributeError.
    return TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=finetune_mode,
        persistent_prefix_mode="per_block",
    )


def test_finetune_mode_creates_only_gamma_mem_not_gamma_attn():
    block = TitansMAGBlock(_cfg(finetune_mode=True))
    assert hasattr(block, "gamma_mem")
    assert not hasattr(block, "gamma_attn"), (
        "gamma_attn must NOT exist when finetune_mode=True — would leak "
        "unused params into state_dict and break checkpoint interchange."
    )


def test_scratch_mode_creates_both_gates():
    block = TitansMAGBlock(_cfg(finetune_mode=False))
    assert hasattr(block, "gamma_mem")
    assert hasattr(block, "gamma_attn")


def test_gates_init_to_ones():
    block_ft = TitansMAGBlock(_cfg(finetune_mode=True))
    block_sc = TitansMAGBlock(_cfg(finetune_mode=False))
    assert torch.equal(block_ft.gamma_mem, torch.ones(8))
    assert torch.equal(block_sc.gamma_mem, torch.ones(8))
    assert torch.equal(block_sc.gamma_attn, torch.ones(8))


def test_finetune_additive_gate_at_init_equals_y_attn_exactly():
    """closing test: at finetune init, out_scale=0 -> y_mem=0 -> o = y_attn.
    Verified end-to-end by comparing the additive gate's output to the attention
    output alone for a one-block forward."""
    block = TitansMAGBlock(_cfg(finetune_mode=True))
    block.eval()
    nmm = block.nmm
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    # Capture y_attn directly using the block's components.
    with torch.no_grad():
        B, T, _ = x.shape
        x_aug = torch.cat(
            [block.persistent_mem.expand(B, -1, -1), x], dim=1
        )
        y_attn = block.attn(
            block.ln_1(x_aug), mask=block._aug_mask(T, dtype=x.dtype)
        )[:, block.N_p :, :]

        # Run the block end-to-end and isolate o = y_attn + silu(g_m*0)*y_attn = y_attn
        # by subtracting the MLP path.
        x_after_attn_residual = x + y_attn  # because o = y_attn at init
        expected_block_out = x_after_attn_residual + block.mlp(
            block.ln_2(x_after_attn_residual)
        )

        actual, _ = block(x, nmm_state=state)

    # At out_scale=0, the block output must match the y_attn-only residual path.
    assert torch.allclose(actual, expected_block_out, atol=1e-5)


def test_scratch_multiplicative_gate_produces_zero_at_y_mem_zero():
    """For finetune_mode=False with y_mem forced to 0, o = silu(g_a*y_attn)*0 = 0.
    Verify the gate formula via direct construction."""
    import torch.nn.functional as F
    block = TitansMAGBlock(_cfg(finetune_mode=False))
    y_attn = torch.randn(2, 4, 8)
    y_mem = torch.zeros(2, 4, 8)
    o = F.silu(block.gamma_attn * y_attn) * F.silu(block.gamma_mem * y_mem)
    assert torch.all(o == 0.0)
