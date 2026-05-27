"""Tests for the 3 paper-strict ablation flags (G254).

Three independent config flags expose deliberate paper/lucidrains
divergences as runtime-flippable behavior:

  - `retrieval_from_M_prev`: paper Eq. 15 — retrieve from M_{t-1} (read-
    then-write) instead of the lucidrains-default M_t (write-then-read).
  - `feed_persistent_to_nmm`: paper Eq. 28 — pass `x̃` (persistent-
    augmented) through the NMM instead of real tokens only.
  - `nmm_n_heads`: lucidrains enhancement — N parallel NMMs each on
    `head_dim = n_embd // n_heads`. NOT in the paper proper.

Each flag has tests for:
  1. Default OFF preserves current behavior (existing-test parity).
  2. Flipping the flag changes outputs in the documented way.
  3. The flag survives a full forward+backward without NaN/shape errors.
"""

import pytest
import torch

from config import TitansConfig
from model.block import TitansMAGBlock
from model.nmm import MultiHeadNMM, NeuralMemoryModule, detach_states
from model.titans_gpt2 import TitansMAGGPT2


# ===========================================================================
# retrieval_from_M_prev (paper Eq. 15)
# ===========================================================================


def _nmm(retrieval_from_M_prev=False, n_embd=8):
    return NeuralMemoryModule(
        n_embd=n_embd,
        expansion=2,
        kernel_size=4,
        spectral_norm=True,
        finetune_mode=False,
        retrieval_from_M_prev=retrieval_from_M_prev,
    )


def test_retrieval_from_M_prev_default_is_True():
    """Post-G254-default-flip: default config prefers paper Eq. 15 (read-
    then-write). Flip to False for lucidrains-flavored write-then-read."""
    cfg = TitansConfig()
    assert cfg.retrieval_from_M_prev is True


def test_step_retrieval_from_M_prev_differs_from_default():
    """step_with_conv() with retrieval_from_M_prev=True must produce a
    different y_t than the default (write-then-read). With the same input,
    weights, and state, the only difference is the retrieval source
    (M_prev vs M_t)."""
    torch.manual_seed(0)
    nmm_default = _nmm(retrieval_from_M_prev=False)
    nmm_paper = _nmm(retrieval_from_M_prev=True)
    nmm_paper.load_state_dict(nmm_default.state_dict())

    state = nmm_default.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)

    # Item 6: conv_buf lives inside state now; init_state seeds zeros,
    # equivalent to the deleted step()'s implicit zero-pad behavior.
    y_default, _ = nmm_default.step_with_conv(x_t, state)
    y_paper, _ = nmm_paper.step_with_conv(x_t, state)

    # Outputs MUST differ — different retrieval source means different y.
    diff = (y_default - y_paper).abs().max().item()
    assert diff > 1e-6, (
        f"retrieval_from_M_prev had no effect on step_with_conv(): max "
        f"diff = {diff:.3e}. The flag should swap M_t for M_prev in retrieval."
    )


def test_step_retrieval_from_M_prev_state_update_identical():
    """The flag only changes the RETRIEVAL — the returned (M, S) state
    must be bit-identical between True and False, because the surprise-
    update math is the same."""
    torch.manual_seed(0)
    nmm_default = _nmm(retrieval_from_M_prev=False)
    nmm_paper = _nmm(retrieval_from_M_prev=True)
    nmm_paper.load_state_dict(nmm_default.state_dict())

    state = nmm_default.init_state(B=2, device=torch.device("cpu"))
    x_t = torch.randn(2, 8)

    _, (M_d, S_d, _) = nmm_default.step_with_conv(x_t, state)
    _, (M_p, S_p, _) = nmm_paper.step_with_conv(x_t, state)

    for k in M_d:
        assert torch.equal(M_d[k], M_p[k]), f"M[{k}] differs across flag values"
        assert torch.equal(S_d[k], S_p[k]), f"S[{k}] differs across flag values"


def test_forward_chunk_retrieval_from_M_prev_first_token_uses_init_M():
    """In a chunk forward with retrieval_from_M_prev=True, the FIRST token's
    y_t is retrieved from M_0 (init_M), not M_1. Verify by comparing
    against the y produced by running just the first token via the
    paper-strict path."""
    torch.manual_seed(0)
    nmm = _nmm(retrieval_from_M_prev=True)
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 3, 8)
    y_chunk, _ = nmm.forward_chunk(x, state, doc_boundaries=None)
    # First token's y must equal out_scale * MLP(init_M, q̂_0).
    # Reconstruct: take init_M (which we have via state[0]) and retrieve.
    import torch.nn.functional as F

    M_init = state[0]
    q_raw = nmm.q_proj(x[:, :1, :]).squeeze(1)  # [1, d]
    q_hat = F.normalize(F.silu(q_raw), dim=-1)
    y_first_expected = nmm.out_scale * nmm._batched_retrieve(M_init, q_hat)
    assert torch.allclose(y_chunk[:, 0, :], y_first_expected, atol=1e-5), (
        "first-token y with retrieval_from_M_prev=True did not equal "
        "out_scale * MLP(init_M, q̂_0) as paper Eq. 15 requires."
    )


def test_forward_chunk_default_first_token_uses_M_1_not_init():
    """Symmetric check for default behavior: first-token y is from M_1
    (post-update), NOT init_M."""
    torch.manual_seed(0)
    nmm = _nmm(retrieval_from_M_prev=False)
    state = nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 3, 8)
    y_chunk, _ = nmm.forward_chunk(x, state, doc_boundaries=None)

    import torch.nn.functional as F
    M_init = state[0]
    q_raw = nmm.q_proj(x[:, :1, :]).squeeze(1)
    q_hat = F.normalize(F.silu(q_raw), dim=-1)
    y_init_baseline = nmm.out_scale * nmm._batched_retrieve(M_init, q_hat)
    # The default's y MUST differ from M_init-retrieval (since it uses M_1).
    assert not torch.allclose(y_chunk[:, 0, :], y_init_baseline, atol=1e-4)


# ===========================================================================
# feed_persistent_to_nmm (paper Eq. 28)
# ===========================================================================


def _cfg_feed_persistent(feed):
    return TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        feed_persistent_to_nmm=feed,
    )


def test_feed_persistent_to_nmm_default_is_True():
    """Post-G254-default-flip: default config prefers paper Eq. 28 (NMM
    sees x̃, the persistent-augmented input). Flip to False for lucidrains-
    flavored real-tokens-only NMM input."""
    cfg = TitansConfig()
    assert cfg.feed_persistent_to_nmm is True


def test_block_feed_persistent_flag_propagates_to_attribute():
    cfg = _cfg_feed_persistent(feed=True)
    block = TitansMAGBlock(cfg)
    assert block.feed_persistent_to_nmm is True


def test_block_feed_persistent_changes_output_shape_invariant():
    """Output shape must stay [B, T, d] regardless of flag — the
    persistent-prefix prefix is sliced off before residual."""
    cfg = _cfg_feed_persistent(feed=True)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    y, _ = block(x, nmm_state=state)
    assert y.shape == x.shape


def test_block_feed_persistent_changes_nmm_output():
    """With feed_persistent_to_nmm=True, the NMM sees a longer input
    (N_p+T) and produces a different y_mem for the real positions than
    the default (which feeds only T real tokens). Verify outputs differ."""
    torch.manual_seed(0)
    cfg_default = _cfg_feed_persistent(feed=False)
    cfg_paper = _cfg_feed_persistent(feed=True)
    block_default = TitansMAGBlock(cfg_default)
    block_paper = TitansMAGBlock(cfg_paper)
    block_paper.load_state_dict(block_default.state_dict())

    state_d = block_default.nmm.init_state(B=1, device=torch.device("cpu"))
    state_p = block_paper.nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8)
    block_default.eval(); block_paper.eval()
    with torch.no_grad():
        y_d, _ = block_default(x, nmm_state=state_d)
        y_p, _ = block_paper(x, nmm_state=state_p)

    # Both produce same-shape outputs but the NMM saw different inputs.
    diff = (y_d - y_p).abs().max().item()
    assert diff > 1e-5, (
        f"feed_persistent_to_nmm had no effect on block output: "
        f"max diff = {diff:.3e}"
    )


def test_block_feed_persistent_handles_doc_boundaries():
    """When doc_boundaries is passed, the flag-True path must augment it
    with a False prefix (persistent positions never trigger doc resets).
    Verify the forward runs without crashing and produces finite output."""
    cfg = _cfg_feed_persistent(feed=True)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8)
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[0, 0] = True
    db[1, 4] = True  # mid-chunk boundary
    y, _ = block(x, nmm_state=state, doc_boundaries=db)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ===========================================================================
# nmm_n_heads (lucidrains enhancement, NOT paper-strict)
# ===========================================================================


def _cfg_n_heads(n_heads, n_embd=8):
    return TitansConfig(
        n_layer=1, n_head=2, n_embd=n_embd, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        nmm_n_heads=n_heads,
        finetune_mode=False,
    )


def test_nmm_n_heads_default_is_1():
    cfg = TitansConfig()
    assert cfg.nmm_n_heads == 1


def test_nmm_n_heads_negative_rejected():
    with pytest.raises(ValueError, match="nmm_n_heads"):
        TitansConfig(nmm_n_heads=0)
    with pytest.raises(ValueError, match="nmm_n_heads"):
        TitansConfig(nmm_n_heads=-1)


def test_nmm_n_heads_not_dividing_n_embd_rejected():
    # n_head=2 divides 8; nmm_n_heads=3 doesn't.
    with pytest.raises(ValueError, match="divisible by nmm_n_heads"):
        TitansConfig(n_embd=8, n_head=2, nmm_n_heads=3)


def test_block_n_heads_1_uses_single_head_NeuralMemoryModule():
    """nmm_n_heads=1 keeps the single-head path (no wrapper) — confirms
    no MultiHeadNMM overhead for the default case."""
    cfg = _cfg_n_heads(1)
    block = TitansMAGBlock(cfg)
    assert isinstance(block.nmm, NeuralMemoryModule)
    assert not isinstance(block.nmm, MultiHeadNMM)


def test_block_n_heads_gt_1_uses_MultiHeadNMM():
    cfg = _cfg_n_heads(2, n_embd=8)
    block = TitansMAGBlock(cfg)
    assert isinstance(block.nmm, MultiHeadNMM)
    assert block.nmm.n_heads == 2
    assert block.nmm.head_dim == 4


def test_multi_head_nmm_init_state_returns_list_of_per_head_states():
    cfg = _cfg_n_heads(2)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    assert isinstance(state, list)
    assert len(state) == 2
    # Each per-head state is a (M, S) tuple of dicts.
    for M, S, _ in state:
        assert isinstance(M, dict)
        assert isinstance(S, dict)
        # M's W1 shape: [B, head_dim*expansion, head_dim] = [2, 8, 4] for head_dim=4, expansion=2
        assert M["W1.weight"].shape == (2, 8, 4)


def test_multi_head_block_forward_shape_invariant():
    cfg = _cfg_n_heads(2, n_embd=16)
    cfg2 = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        nmm_n_heads=2,
        finetune_mode=False,
    )
    block = TitansMAGBlock(cfg2)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 16)
    y, _ = block(x, nmm_state=state)
    assert y.shape == x.shape


def test_multi_head_block_is_differentiable():
    """Backward through a multi-head block must produce gradients on every
    head's NMM params + the attn/mlp params."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        nmm_n_heads=4,  # head_dim = 4
        finetune_mode=False,
    )
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    x = torch.randn(1, 4, 16)
    y, _ = block(x, nmm_state=state)
    y.sum().backward()
    # Every head's k_proj must have a grad.
    for i, head in enumerate(block.nmm.heads):
        assert head.k_proj.linear.weight.grad is not None, (
            f"head {i} k_proj.linear has no gradient"
        )
        assert head.k_proj.linear.weight.grad.abs().sum() > 0


def test_multi_head_state_threads_across_calls():
    """Pass state from one forward into the next; outputs must reflect
    the carried multi-head state (each head independently accumulates)."""
    torch.manual_seed(0)
    cfg = _cfg_n_heads(2, n_embd=8)
    block = TitansMAGBlock(cfg)
    block.eval()
    x1 = torch.randn(1, 4, 8)
    x2 = torch.randn(1, 4, 8)
    init_state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    with torch.no_grad():
        _, state_after_1 = block(x1, nmm_state=init_state)
        y_continued, _ = block(x2, nmm_state=state_after_1)
        fresh_state = block.nmm.init_state(B=1, device=torch.device("cpu"))
        y_fresh, _ = block(x2, nmm_state=fresh_state)
    diff = (y_continued - y_fresh).abs().max().item()
    assert diff > 1e-5, (
        "carrying multi-head state had no effect — heads aren't actually "
        "accumulating across calls"
    )


def test_multi_head_full_model_forward_shape_and_grad():
    """End-to-end smoke: multi-head model.forward returns correctly-shaped
    logits and per-layer multi-head states; gradient flows through."""
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        nmm_n_heads=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, states = model(idx, nmm_states=None)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert len(states) == cfg.n_layer
    # Each per-layer state is a list of per-head (M, S) tuples.
    for layer_state in states:
        assert isinstance(layer_state, list)
        assert len(layer_state) == cfg.nmm_n_heads

    # Gradient flow
    import torch.nn.functional as F
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        idx[:, 1:].reshape(-1),
    )
    loss.backward()
    # Pick one per-head NMM param and verify grad exists.
    p = model.blocks[0].nmm.heads[0].k_proj.linear.weight
    assert p.grad is not None and p.grad.abs().sum() > 0


# ===========================================================================
# detach_states / compute_nmm_norm work for multi-head states
# ===========================================================================


def test_detach_states_handles_multi_head_nested_structure():
    """detach_states must recurse into per-head lists when nmm_n_heads > 1."""
    cfg = _cfg_n_heads(2)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    # Wrap in a per-layer list to match top-level nmm_states shape.
    nmm_states = [state]
    detached = detach_states(nmm_states)
    assert detached is not None
    assert len(detached) == 1
    assert isinstance(detached[0], list)  # multi-head: per-layer is a list
    assert len(detached[0]) == 2
    for M, S, _ in detached[0]:
        for v in M.values():
            assert v.requires_grad is False
        for v in S.values():
            assert v.requires_grad is False


def test_detach_states_single_head_unchanged():
    """Sanity: single-head path still produces flat (M, S) tuples."""
    cfg = _cfg_n_heads(1)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    nmm_states = [state]
    detached = detach_states(nmm_states)
    assert len(detached) == 1
    # Single-head: per-layer is a (M, S) tuple, not a list.
    assert isinstance(detached[0], tuple)
    M, S, _ = detached[0]
    assert isinstance(M, dict) and isinstance(S, dict)


def test_compute_nmm_norm_handles_multi_head_nested_structure():
    """compute_nmm_norm returns one float per layer regardless of head count
    — multi-head averages across heads."""
    from train import compute_nmm_norm

    cfg = _cfg_n_heads(2)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=1, device=torch.device("cpu"))
    nmm_states = [state]
    norms = compute_nmm_norm(nmm_states)
    assert len(norms) == 1
    assert isinstance(norms[0], float)
    assert norms[0] > 0  # init from Xavier => non-trivial Frobenius norm


def test_compute_nmm_norm_returns_none_on_none_multi_head_safe():
    from train import compute_nmm_norm
    assert compute_nmm_norm(None) is None


# ===========================================================================
# All three flags combined (paper-strict ablation set)
# ===========================================================================


def test_all_three_flags_together_does_not_crash():
    """Smoke: set all three flags simultaneously and run a full forward
    + backward. No NaN, correct shapes, gradients flow."""
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=64,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        nmm_n_heads=2,
        retrieval_from_M_prev=True,
        feed_persistent_to_nmm=True,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, states = model(idx, nmm_states=None)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    import torch.nn.functional as F
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        idx[:, 1:].reshape(-1),
    )
    loss.backward()
    # NMM params have grads on every head.
    for block in model.blocks:
        for head in block.nmm.heads:
            assert head.k_proj.linear.weight.grad is not None
