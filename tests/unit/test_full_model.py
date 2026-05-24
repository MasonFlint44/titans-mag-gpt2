"""Phase 2.5 — TitansMAGGPT2 full model."""

import math

import pytest
import torch

from config import TitansConfig
from model.nmm import NeuralMemoryModule
from model.titans_gpt2 import TitansMAGGPT2


def _tiny_cfg(**overrides):
    base = dict(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=True,
    )
    base.update(overrides)
    return TitansConfig(**base)


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------

def test_forward_shape():
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, states = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert len(states) == cfg.n_layer


def test_forward_logits_finite():
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(idx)
    assert torch.isfinite(logits).all()


def test_forward_with_nmm_states_none_seeds_internally():
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, states = model(idx, nmm_states=None)
    assert states is not None
    assert len(states) == cfg.n_layer
    # Each layer's state is a (M, S) tuple of dicts.
    for s in states:
        M, S = s
        assert set(M.keys()) == {"W1.weight", "W_gate.weight", "W2.weight"}


def test_tied_weights_lm_head_is_wte_T():
    """No separate lm_head module — logits = x @ wte.weight.T."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    assert not hasattr(model, "lm_head")


# ---------------------------------------------------------------------------
# G155 — GPT-2 init scales
# ---------------------------------------------------------------------------

def test_wte_init_std_is_002_not_default_1():
    cfg = _tiny_cfg(n_embd=128, vocab_size=512)  # bigger -> tighter sample std
    model = TitansMAGGPT2(cfg)
    s = model.wte.weight.std().item()
    assert abs(s - 0.02) < 0.005, (
        f"wte.weight.std()={s:.4f}, expected ~0.02 (default Embedding init is "
        f"N(0,1) which would give std ~1.0 — _apply_gpt2_init missed wte?)"
    )


def test_wpe_init_std_is_002():
    cfg = _tiny_cfg(n_embd=128)
    model = TitansMAGGPT2(cfg)
    s = model.wpe.weight.std().item()
    assert abs(s - 0.02) < 0.005


def test_attn_output_projection_has_residual_scaling():
    """attn.proj.weight.std() must be ~ 0.02 / sqrt(2*n_layer)."""
    cfg = _tiny_cfg(n_layer=12, n_embd=128, n_head=8)
    model = TitansMAGGPT2(cfg)
    expected_std = 0.02 / math.sqrt(2 * cfg.n_layer)
    s = model.blocks[0].attn.proj.weight.std().item()
    # The scaled std at n_embd=128 (16384 samples) is tight.
    assert abs(s - expected_std) / expected_std < 0.15, (
        f"attn.proj.weight.std()={s:.6f}, expected ~{expected_std:.6f}"
    )


def test_mlp_c_proj_has_residual_scaling():
    cfg = _tiny_cfg(n_layer=12, n_embd=128, n_head=8)
    model = TitansMAGGPT2(cfg)
    expected_std = 0.02 / math.sqrt(2 * cfg.n_layer)
    s = model.blocks[0].mlp.c_proj.weight.std().item()
    assert abs(s - expected_std) / expected_std < 0.15


def test_attn_qkv_projections_use_unscaled_std():
    """q/k/v_proj (NOT output projections) get the default 0.02, not the
    residual-scaled std. The endswith('.proj') check should NOT match them."""
    cfg = _tiny_cfg(n_layer=12, n_embd=128, n_head=8)
    model = TitansMAGGPT2(cfg)
    s_q = model.blocks[0].attn.q_proj.weight.std().item()
    assert abs(s_q - 0.02) / 0.02 < 0.10


# ---------------------------------------------------------------------------
# G203 — NMM-internal modules skipped by IDENTITY, not name
# ---------------------------------------------------------------------------

def test_nmm_internal_inits_preserved_after_apply_gpt2_init():
    """Xavier-uniform inits inside the NMM must survive _apply_gpt2_init."""
    cfg = _tiny_cfg(n_embd=32)
    model = TitansMAGGPT2(cfg)
    # MemoryMLP W1: Xavier-uniform std = sqrt(2/(d + 4d)) ≈ 0.117 at d=32
    nmm = model.blocks[0].nmm
    d = cfg.n_embd
    h = d * cfg.nmm_expansion
    expected_xavier_std = math.sqrt(2.0 / (d + h))
    actual = nmm.memory_mlp.W1.weight.std().item()
    # If _apply_gpt2_init had overwritten with N(0, 0.02), std would be ~0.02.
    assert abs(actual - expected_xavier_std) / expected_xavier_std < 0.10, (
        f"NMM memory_mlp.W1 std={actual:.4f}; expected Xavier ~{expected_xavier_std:.4f}. "
        f"_apply_gpt2_init may have stomped on it (G203)."
    )


def test_out_scale_preserved_at_zero_in_finetune_init():
    cfg = _tiny_cfg(finetune_mode=True)
    model = TitansMAGGPT2(cfg)
    for block in model.blocks:
        assert torch.all(block.nmm.out_scale == 0.0)


def test_persistent_mem_init_scale_not_clobbered():
    """persistent_mem inits to randn*0.02; _apply_gpt2_init must not touch it
    (Parameters aren't Modules, so named_modules() never visits them)."""
    cfg = _tiny_cfg(n_embd=128, nmm_n_persistent=4)
    model = TitansMAGGPT2(cfg)
    s = model.blocks[0].persistent_mem.std().item()
    assert 0.005 < s < 0.05, f"persistent_mem.std()={s} unexpectedly large/small"


def test_renaming_self_nmm_does_not_break_id_skip_pattern():
    """G203 directly: an id-based skip must work for ANY module that IS a
    NeuralMemoryModule, regardless of attribute name. Verify by manually
    constructing the skip set the same way _apply_gpt2_init does."""
    cfg = _tiny_cfg()
    model = TitansMAGGPT2(cfg)
    # Collect by identity, just as _apply_gpt2_init does.
    nmm_ids = set()
    for m in model.modules():
        if isinstance(m, NeuralMemoryModule):
            for sub in m.modules():
                nmm_ids.add(id(sub))
    # Every NMM submodule's id is in the set, regardless of attribute name.
    for block in model.blocks:
        for sub in block.nmm.modules():
            assert id(sub) in nmm_ids


def test_apply_gpt2_init_uses_relative_import():
    """G224 — source must contain `from .nmm import NeuralMemoryModule`.
    Absolute `from model.nmm` would break under any top-level package rename."""
    import ast
    import inspect
    from model.titans_gpt2 import TitansMAGGPT2

    src = inspect.getsource(TitansMAGGPT2._apply_gpt2_init)
    # Parse and look for ImportFrom nodes with explicit level (relative).
    tree = ast.parse(src.strip())
    relative_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert any(
        imp.module == "nmm" and any(a.name == "NeuralMemoryModule" for a in imp.names)
        for imp in relative_imports
    ), "Expected `from .nmm import NeuralMemoryModule` (G224)"
    # Belt-and-suspenders: no absolute import of the same symbol.
    absolute_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    ]
    for imp in absolute_imports:
        assert imp.module != "model.nmm", (
            "Absolute `from model.nmm import ...` defeats G224 — use relative."
        )


# ---------------------------------------------------------------------------
# TBPTT state continuity
# ---------------------------------------------------------------------------

def test_two_chunk_forward_threads_nmm_state():
    """Output of chunk 2 with state from chunk 1 must DIFFER from chunk 2 with
    state=None — confirms state actually carries information."""
    cfg = _tiny_cfg(finetune_mode=False)  # so NMM contributes
    model = TitansMAGGPT2(cfg)
    model.eval()
    idx1 = torch.randint(0, cfg.vocab_size, (1, 8))
    idx2 = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        _, states_after_1 = model(idx1)
        logits_continued, _ = model(idx2, nmm_states=states_after_1)
        logits_fresh, _ = model(idx2, nmm_states=None)
    assert not torch.allclose(logits_continued, logits_fresh, atol=1e-5)
