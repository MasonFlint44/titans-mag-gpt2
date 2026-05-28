"""GPU-tier: torch.compile + _unwrap save/load round-trip.

The fake-wrapper tests in test_scan_dispatcher.py exercise the _unwrap
logic but not the real torch.compile path. Without _unwrap, a compiled
model's state_dict has every key prefixed with `_orig_mod.`, and loading
it into an uncompiled model fails with "Missing key(s) / Unexpected
key(s)". save_checkpoint uses _unwrap; this test verifies the round-trip
on a real OptimizedModule.
"""

import os
import tempfile

import pytest
import torch

from config import TitansConfig
from model import _unwrap
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import build_optimizer, load_checkpoint, save_checkpoint


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def _tiny_model():
    _gpu_or_skip()
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=32, vocab_size=64,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg).cuda()
    return cfg, model


def test_compiled_model_state_dict_has_orig_mod_prefix():
    """Sanity: torch.compile DOES add the _orig_mod. prefix (confirms the
 hazard is real on this PyTorch version)."""
    cfg, model = _tiny_model()
    compiled = torch.compile(model)
    keys = list(compiled.state_dict().keys())
    assert any(k.startswith("_orig_mod.") for k in keys), (
        f"torch.compile did NOT add _orig_mod. prefix on this version — "
        f"keys: {keys[:5]}"
    )


def test_unwrap_strips_orig_mod_from_real_compiled_model():
    """_unwrap on a real torch.compile output returns the inner
    model whose state_dict has NO _orig_mod. prefix."""
    cfg, model = _tiny_model()
    compiled = torch.compile(model)
    unwrapped = _unwrap(compiled)
    keys = list(unwrapped.state_dict().keys())
    assert not any(k.startswith("_orig_mod.") for k in keys), (
        f"_unwrap left _orig_mod. prefix on real torch.compile output: "
        f"sample keys: {keys[:3]}"
    )


def test_save_load_roundtrip_with_real_torch_compile():
    """End-to-end: compile, save via save_checkpoint (which uses _unwrap),
    rebuild uncompiled model, load — must succeed without strict=False."""
    cfg, model = _tiny_model()
    compiled = torch.compile(model)
    optimizer = build_optimizer(_unwrap(compiled))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, compiled, optimizer, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cuda"))

        # Resume into an UNCOMPILED model.
        cfg2 = TitansConfig(**ckpt["config"])
        model2 = TitansMAGGPT2(cfg2).cuda()
        # strict=True (default) — must succeed.
        model2.load_state_dict(ckpt["state_dict"])

    # Sanity: model2's params match unwrapped(compiled)'s params byte-for-byte.
    for (n1, p1), (n2, p2) in zip(
        _unwrap(compiled).named_parameters(),
        model2.named_parameters(),
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param {n1} differs after round-trip"
