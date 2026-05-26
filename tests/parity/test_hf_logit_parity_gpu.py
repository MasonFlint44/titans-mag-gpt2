"""GPU-tier HF parity at realistic (B, T) sizes.

CPU parity tests (test_hf_logit_parity.py) run at (B=1, T=4) because the
NMM forward at gpt2_small dims is ~250ms per token-per-layer on CPU
(G231). On GPU the same forward is milliseconds, so we can exercise
realistic batch and sequence shapes.

Same invariant as the CPU tests: with N_p=0 and out_scale=0, our model's
logits equal HF GPT-2's to <1e-4 max diff. Size-independence of the
arithmetic guarantee is the point.
"""

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _disable_tf32():
    """Parity vs HF GPT-2 is verified at full fp32 precision.

    Why: conftest.py enables TF32 (`set_float32_matmul_precision("high")`)
    to match training-time behavior. TF32 trades fp32 precision for
    tensor-core speed — across 12 layers of fp32 matmuls in the gpt2_small
    backbone, that accumulates to ~1.8e-4 relative error in the logits,
    blowing past this test's `2e-6 * logit_max` tolerance. The tolerance
    is intentional: it's "bit-level fp32 noise," not "TF32 noise," because
    the invariant we're locking is that our model performs the same fp32
    arithmetic as HF's, not a looser numerical-equivalence claim.
    """
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


@pytest.fixture(scope="module")
def hf_gpt2_small():
    _gpu_or_skip()
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to("cuda")


@pytest.fixture(scope="module")
def loaded_titans():
    _gpu_or_skip()
    cfg = TitansConfig.gpt2_small(nmm_n_persistent=0)
    model = TitansMAGGPT2(cfg).to("cuda")
    load_pretrained(model, cfg)
    model.eval()
    return model


# The default forward_chunk dispatcher routes to SCAN under @torch.no_grad,
# and scan's batched all_grads + per-T Newton-Schulz blow past 16 GB on
# realistic shapes (B=4, T=32 hits OOM via NS's [T, B, h, h] matmul, which
# is ~T*B*38 MB per layer; at (4, 32) that's ~4.8 GB just for one matmul).
# Forcing the SEQUENTIAL path (per-token loop, reuses M/S in-place) keeps
# memory bounded. We trigger it via doc_boundaries with a True flag at
# t=0 — a no-op observationally because the state at t=0 IS the init
# state already, but the dispatcher sees any-True boundaries and falls
# back to sequential.
def _force_sequential_boundaries(B, T, device):
    db = torch.zeros(B, T, dtype=torch.bool, device=device)
    db[:, 0] = True
    return db


@pytest.mark.parametrize("B,T", [(1, 4), (4, 32), (2, 64), (1, 128), (4, 128)])
def test_logit_parity_at_realistic_size(B, T, hf_gpt2_small, loaded_titans):
    """Logit max-diff <= max(1e-4, 2e-6 * |logits|_max) across realistic
    sizes. The CPU HARD GATE used a flat <1e-4; at larger T the logits
    grow in magnitude (~100 at T=128 with HF GPT-2), so an absolute
    1e-4 bound becomes tighter than fp32 precision allows. The scaled
    form keeps the small-input cases at the original tight bound while
    accepting the proportionally-larger drift at large sizes (which is
    still ~1.6e-6 relative — bit-level fp32 noise)."""
    torch.cuda.empty_cache()
    torch.manual_seed(B * 1000 + T)
    idx = torch.randint(0, 50257, (B, T), device="cuda")
    db = _force_sequential_boundaries(B, T, idx.device)

    with torch.no_grad():
        ours_logits, _ = loaded_titans(idx, nmm_states=None, doc_boundaries=db)
        theirs_out = hf_gpt2_small(idx)
        theirs_logits = theirs_out.logits

    diff = (ours_logits - theirs_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    logit_max = theirs_logits.abs().max().item()
    tolerance = max(1e-4, 2e-6 * logit_max)

    del idx, db, ours_logits, theirs_logits, theirs_out
    torch.cuda.empty_cache()

    assert max_diff < tolerance, (
        f"parity at (B={B}, T={T}) FAILED: max diff = {max_diff:.3e} "
        f"(mean = {mean_diff:.3e}, logit_max = {logit_max:.2f}, "
        f"tolerance = {tolerance:.3e})"
    )
