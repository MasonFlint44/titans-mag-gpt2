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
        M, S, _ = s
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
    """persistent_mem inits to randn*0.02; _apply_gpt2_init must not touch
    it (Parameters aren't Modules, so named_modules() never visits them).

    Verifies the invariant in BOTH modes — model_wide puts persistent_mem
    on the model itself, per_block puts it on each TitansMAGBlock."""
    for mode, get_param in [
        ("model_wide", lambda m: m.persistent_mem),
        ("per_block",  lambda m: m.blocks[0].persistent_mem),
    ]:
        cfg = _tiny_cfg(
            n_embd=128, nmm_n_persistent=4, persistent_prefix_mode=mode,
        )
        model = TitansMAGGPT2(cfg)
        s = get_param(model).std().item()
        assert 0.005 < s < 0.05, (
            f"mode={mode}: persistent_mem.std()={s} unexpectedly large/small"
        )


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


# ---------------------------------------------------------------------------
# T3 — doc_boundaries=all_true reset equivalence (TEST_PLAN §8 test_model_forward.py)
#
# The strict "doc_boundaries=all_true => per-position output == per-token
# fresh-state forwards" interpretation doesn't hold at the LOGITS level —
# attention sees different context in chunked vs single-token forwards
# even when NMM state is perfectly reset. So we test the NMM STATE
# equivalence instead, which IS the actual invariant reset_state defends:
# after T tokens with all-boundaries-true, the returned (M, S) reflects
# exactly "one update step from init_M on the last token", because reset
# fires before each token's update including the last.
# ---------------------------------------------------------------------------

def test_doc_boundaries_all_true_isolates_positions_from_earlier_input_changes():
    """T3 — the OBSERVABLE consequence of "state resets every position":
    with doc_boundaries=all_true, the NMM state at position T-1 depends
    only on the position-T-1 input — earlier-token changes don't propagate
    through the (reset) state.

    Swap test: change idx[0]. With all-true boundaries, the returned state
    at the end of the chunk should be UNCHANGED (since position 0's update
    is reset away before position 1's update fires, etc.). With no
    boundaries (default first-token-only), changing idx[0] propagates
    through the state and changes the final state.

    Use nmm_conv_kernel=1 so the conv has no temporal mixing — changing
    idx[0] only affects position 0's projection, not later positions'
    projections. (With kernel > 1, idx[0]'s linear projection would still
    be in the conv buffer at positions 1..k-1, contaminating their k_hat/
    v through the conv even with state-reset.)
    """
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        nmm_conv_kernel=1,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    model.eval()
    T = 3
    idx_a = torch.randint(0, cfg.vocab_size, (1, T))
    idx_b = idx_a.clone()
    idx_b[0, 0] = (idx_a[0, 0] + 1) % cfg.vocab_size  # change ONLY position 0

    db_all_true = torch.ones(1, T, dtype=torch.bool)

    with torch.no_grad():
        _, states_a = model(idx_a, nmm_states=None, doc_boundaries=db_all_true)
        _, states_b = model(idx_b, nmm_states=None, doc_boundaries=db_all_true)

    # Returned NMM state for idx_a and idx_b must match (position 0's
    # contribution was reset away before position 1's update). Note: this
    # holds only for the NMM STATE — attention still sees both versions
    # of position 0 differently, so the model's LOGITS would differ.
    for (M_a, S_a, _), (M_b, S_b, _) in zip(states_a, states_b):
        for key in M_a:
            md = (M_a[key] - M_b[key]).abs().max().item()
            sd = (S_a[key] - S_b[key]).abs().max().item()
            assert md < 1e-5, (
                f"M[{key}] differed across position-0 swap: {md:.3e} — "
                f"earlier-input change propagated through the reset state"
            )
            assert sd < 1e-5, f"S[{key}] differed: {sd:.3e}"


def test_doc_boundaries_no_reset_path_DOES_propagate_position_zero_changes():
    """Counterpart sanity check: WITHOUT resets, a position-0 swap MUST
    propagate to the final state. If this fails, the swap test above is
    passing vacuously."""
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        nmm_conv_kernel=1,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    model.eval()
    T = 3
    idx_a = torch.randint(0, cfg.vocab_size, (1, T))
    idx_b = idx_a.clone()
    idx_b[0, 0] = (idx_a[0, 0] + 1) % cfg.vocab_size

    db_first_only = torch.zeros(1, T, dtype=torch.bool)
    db_first_only[:, 0] = True  # standard boundary at start of stream

    with torch.no_grad():
        _, states_a = model(idx_a, nmm_states=None, doc_boundaries=db_first_only)
        _, states_b = model(idx_b, nmm_states=None, doc_boundaries=db_first_only)

    # SOMETHING must differ.
    differs = False
    for (M_a, _, _), (M_b, _, _) in zip(states_a, states_b):
        for key in M_a:
            if (M_a[key] - M_b[key]).abs().max().item() > 1e-6:
                differs = True
                break
        if differs:
            break
    assert differs, (
        "M state was identical across position-0 swap WITHOUT resets — "
        "either propagation is broken or the test isn't exercising it."
    )


def test_doc_boundaries_all_true_state_differs_from_no_reset_path():
    """Sanity check: the same T-token forward WITHOUT resets produces a
    DIFFERENT (M, S) than the all-reset path. Otherwise the reset test
    above could pass trivially if state never changed."""
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        nmm_conv_kernel=1,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    model.eval()
    T = 3
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    db_all_true = torch.ones(1, T, dtype=torch.bool)
    db_none = torch.zeros(1, T, dtype=torch.bool)
    db_none[:, 0] = True  # mandatory first-position boundary

    with torch.no_grad():
        _, states_reset = model(idx, nmm_states=None, doc_boundaries=db_all_true)
        _, states_no_reset = model(idx, nmm_states=None, doc_boundaries=db_none)

    # At least one M entry must differ — otherwise resets had no effect.
    differs = False
    for (M_r, _, _), (M_n, _, _) in zip(states_reset, states_no_reset):
        for key in M_r:
            if (M_r[key] - M_n[key]).abs().max().item() > 1e-6:
                differs = True
                break
        if differs:
            break
    assert differs, (
        "M state was identical with vs without per-token resets; either "
        "reset_state is a no-op or the test inputs aren't exercising it."
    )


# ---------------------------------------------------------------------------
# T6 — Multi-block stack gradient flow (TEST_PLAN §8 test_block_forward.py)
# ---------------------------------------------------------------------------

def test_multi_block_stack_forward_and_gradient_flow_to_every_param():
    """T6 — 3-block model: forward + backward; every named parameter must
    have a non-None grad. Catches dead-branch / detached-tensor regressions
    that would silently freeze part of the model during training.

    Single-block tests in test_block.py cover one block; this exercises
    the stack-wise gradient flow (output of block i goes to block i+1's
    input, residual stream survives, ln_f is in the path, lm_head ties
    back to wte, etc.).
    """
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=3, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, _ = model(idx, nmm_states=None, doc_boundaries=None)
    assert logits.shape == (2, 4, cfg.vocab_size)

    # Cross-entropy loss against next-token targets.
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        idx[:, 1:].reshape(-1),
    )
    loss.backward()

    # Every named parameter must have a grad.
    no_grad_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad_params.append(name)
    assert not no_grad_params, (
        f"{len(no_grad_params)} params have grad=None after backward through "
        f"3-block stack: {no_grad_params[:5]}{'...' if len(no_grad_params) > 5 else ''}"
    )
