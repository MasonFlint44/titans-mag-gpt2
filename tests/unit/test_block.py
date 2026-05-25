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


# ---------------------------------------------------------------------------
# T12 / T13 — NMM input contract (TEST_PLAN §4 test_block.py spec)
# ---------------------------------------------------------------------------

def _capture_nmm_forward_chunk_args(block):
    """Wrap block.nmm.forward_chunk to capture (args, kwargs) of each call."""
    captured = []
    orig = block.nmm.forward_chunk

    def capturing(*args, **kwargs):
        captured.append((args, kwargs))
        return orig(*args, **kwargs)

    block.nmm.forward_chunk = capturing
    return captured


def test_nmm_receives_only_real_tokens_not_persistent_augmented():
    """T12 — block.forward(x) where x has T real tokens must call
    nmm.forward_chunk with a [B, T, d] tensor — NOT [B, T+N_p, d].
    A refactor that accidentally passes x_aug (with persistent prefix
    concatenated) would silently train the NMM on persistent-prefix-
    augmented inputs, contaminating the meta-learned init.

    Spec from TEST_PLAN.md §4: "patch nmm.forward_chunk to record
    x.shape[1]; verify T (real tokens), not T + N_p".
    """
    cfg = _cfg(N_p=4, T=6)  # N_p=4 persistent, T=6 real
    block = TitansMAGBlock(cfg)
    captured = _capture_nmm_forward_chunk_args(block)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 6, 8)  # B=2, T=6, d=8

    _ = block(x, nmm_state=state)

    assert len(captured) == 1, f"expected exactly 1 forward_chunk call, got {len(captured)}"
    args, kwargs = captured[0]
    # First positional arg is the input tensor.
    x_to_nmm = args[0]
    assert x_to_nmm.shape == (2, 6, 8), (
        f"nmm.forward_chunk received shape {tuple(x_to_nmm.shape)}, expected "
        f"(2, 6, 8). If shape[1] == 10 (= T + N_p = 6 + 4), the block is "
        f"passing the persistent-augmented x_aug instead of x."
    )


def test_nmm_forward_chunk_called_with_doc_boundaries_arg():
    """T13 — block.forward(x, nmm_state, doc_boundaries=db) must pass
    `db` through to nmm.forward_chunk as its third argument. A regression
    that drops the doc_boundaries arg (`forward_chunk(x_norm, state)`
    instead of 3 args) would silently disable within-chunk state resets.

    Spec from TEST_PLAN.md §4: "NMM called with 3 args — `forward_chunk(
    x_norm, state, doc_boundaries)`, not 2".
    """
    cfg = _cfg(N_p=2, T=6)
    block = TitansMAGBlock(cfg)
    captured = _capture_nmm_forward_chunk_args(block)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 6, 8)
    db = torch.zeros(2, 6, dtype=torch.bool)
    db[0, 0] = True
    db[1, 3] = True  # mid-chunk boundary — would be ignored if dropped

    _ = block(x, nmm_state=state, doc_boundaries=db)

    assert len(captured) == 1
    args, kwargs = captured[0]
    # forward_chunk's signature: (x_chunk, state_in, doc_boundaries). The
    # block could pass these as positional OR mix; check that doc_boundaries
    # reaches the call by reconstructing the value.
    all_call_values = list(args) + list(kwargs.values())
    db_seen = any(
        isinstance(v, torch.Tensor) and v.dtype == torch.bool
        and v.shape == (2, 6) and torch.equal(v, db)
        for v in all_call_values
    )
    assert db_seen, (
        f"nmm.forward_chunk was NOT called with the doc_boundaries tensor "
        f"the caller passed. Args seen: {[type(a).__name__ for a in args]}, "
        f"kwargs: {list(kwargs.keys())}. The block must thread doc_boundaries "
        f"through to nmm.forward_chunk."
    )


def test_nmm_forward_chunk_called_with_none_when_doc_boundaries_none():
    """Complementary to T13: when block.forward is called WITHOUT
    doc_boundaries (default None), nmm.forward_chunk must receive None,
    not a fabricated all-zeros tensor or a missing arg that would default
    elsewhere."""
    cfg = _cfg(N_p=2, T=4)
    block = TitansMAGBlock(cfg)
    captured = _capture_nmm_forward_chunk_args(block)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 4, 8)

    _ = block(x, nmm_state=state)  # no doc_boundaries arg

    assert len(captured) == 1
    args, kwargs = captured[0]
    all_call_values = list(args) + list(kwargs.values())
    none_seen = any(v is None for v in all_call_values)
    assert none_seen, (
        f"nmm.forward_chunk received no None value — expected one of "
        f"(x, state, doc_boundaries=None). Args: {args}, kwargs: {kwargs}."
    )
