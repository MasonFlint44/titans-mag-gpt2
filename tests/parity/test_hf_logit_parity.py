"""Phase 2.6 HARD GATE — TitansMAGGPT2 with NMM zeroed and N_p=0 must produce
logits identical to HF GPT-2 to <1e-4 max diff. If this fails, downstream
training is meaningless."""

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained


@pytest.fixture(scope="module")
def hf_gpt2_small():
    """Cached HF GPT-2 small (loaded once per module)."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


@pytest.fixture(scope="module")
def parity_config():
    # N_p=0 so x_aug=x exactly; out_scale=0 (finetune_mode=True) so y_mem=0
    # exactly; both together collapse the block to the standard GPT-2 path.
    return TitansConfig.gpt2_small(nmm_n_persistent=0)


@pytest.fixture(scope="module")
def loaded_titans(parity_config):
    model = TitansMAGGPT2(parity_config)
    load_pretrained(model, parity_config)
    model.eval()
    return model


def _hf_logits(hf_model, idx):
    """Run HF GPT-2 in eval mode and return logits [B, T, vocab]."""
    hf_model.eval()
    with torch.no_grad():
        out = hf_model(idx, output_hidden_states=False)
    return out.logits


def _titans_logits(model, idx):
    with torch.no_grad():
        logits, _ = model(idx, nmm_states=None)
    return logits


@pytest.mark.slow
def test_logit_parity_max_diff_lt_1e_4(hf_gpt2_small, loaded_titans):
    """HARD GATE: max |logit_ours - logit_HF| < 1e-4 on a fixed-seed batch.

    Small (B, T) for CPU speed — NMM forward at d=768 is ~250ms per token-
    per-layer (per_sample_grad + NS5). Parity at a small shape implies
    parity at any shape (it's pure arithmetic equivalence under
    out_scale=0 + N_p=0), so the small input is the right test, not a
    weakened one."""
    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (1, 4))
    ours = _titans_logits(loaded_titans, idx)
    theirs = _hf_logits(hf_gpt2_small, idx)
    diff = (ours - theirs).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff < 1e-4, (
        f"HARD GATE FAILED: max logit diff = {max_diff:.3e} "
        f"(mean = {mean_diff:.3e}). Check Conv1D transpose direction, "
        f"wte tied head, ln_f application, out_scale zero, N_p=0."
    )


@pytest.mark.slow
def test_logit_parity_across_two_shapes(hf_gpt2_small, loaded_titans):
    """Two small shapes to catch shape-dependent bugs without paying for the
    NMM forward at large T."""
    for B, T in [(1, 2), (2, 4)]:
        torch.manual_seed(B * 100 + T)
        idx = torch.randint(0, 50257, (B, T))
        ours = _titans_logits(loaded_titans, idx)
        theirs = _hf_logits(hf_gpt2_small, idx)
        max_diff = (ours - theirs).abs().max().item()
        assert max_diff < 1e-4, (
            f"parity failed at (B={B}, T={T}): max_diff = {max_diff:.3e}"
        )


@pytest.mark.slow
def test_titans_lm_head_weights_match_hf(loaded_titans, hf_gpt2_small):
    """wte must be a byte-perfect copy of HF's wte (tied head is wte.weight.T)."""
    assert torch.equal(
        loaded_titans.wte.weight, hf_gpt2_small.transformer.wte.weight
    )


@pytest.mark.slow
def test_titans_wpe_matches_hf(loaded_titans, hf_gpt2_small):
    assert torch.equal(
        loaded_titans.wpe.weight, hf_gpt2_small.transformer.wpe.weight
    )


@pytest.mark.slow
def test_titans_ln_f_matches_hf(loaded_titans, hf_gpt2_small):
    assert torch.equal(
        loaded_titans.ln_f.weight, hf_gpt2_small.transformer.ln_f.weight
    )
    assert torch.equal(
        loaded_titans.ln_f.bias, hf_gpt2_small.transformer.ln_f.bias
    )


def test_load_pretrained_rejects_unsupported_n_embd():
    """Sizes not in {768, 1024, 1280, 1600} must raise with a clear message.
    Cheap — fails before any HF download attempt."""
    cfg = TitansConfig(n_embd=512, n_head=8, n_layer=6, vocab_size=50257,
                       block_size=64, chunk_size=64, nmm_n_persistent=0)
    model = TitansMAGGPT2(cfg)
    with pytest.raises(ValueError, match="No HF GPT-2 checkpoint"):
        load_pretrained(model, cfg)


@pytest.mark.slow
def test_load_pretrained_rejects_n_layer_override():
    """Defensive: TitansConfig.gpt2_small(n_layer=14) makes HF zip-truncate
    silently; must raise instead. Needs the HF load to compare layer counts."""
    cfg = TitansConfig.gpt2_small(n_layer=14)
    model = TitansMAGGPT2(cfg)
    with pytest.raises(ValueError, match="n_layer"):
        load_pretrained(model, cfg)
