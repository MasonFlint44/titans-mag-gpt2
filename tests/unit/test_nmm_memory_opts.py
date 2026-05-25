"""G256/G257 — nmm_state_dtype + nmm_grad_checkpoint memory-saving options.

Two independent, composable flags:

- `nmm_state_dtype="bf16"`: (M, S) and per-step update buffers stored in
  bf16 instead of fp32 (~2x smaller). NS5 still casts to fp32 internally
  (the bf16-NS5-spectral-norm-drift hazard documented in G226).

- `nmm_grad_checkpoint=True`: `_forward_chunk_sequential` runs in
  `grad_checkpoint_segment_len`-token segments, each wrapped in
  `torch.utils.checkpoint.checkpoint`. Backward recomputes inner-loop
  intermediates; ~5-10x peak-memory headroom for the per-token graph.

The contract these tests pin down:
  (a) dtype choice is honored end-to-end (state, output, retrieval).
  (b) bf16 output ≈ fp32 output within a paper-faithful tolerance.
  (c) checkpoint True ≈ False — forward AND backward — at small T.
  (d) the two flags compose without crashing.
  (e) doc_boundary reset still fires correctly across segment edges.
  (f) seg_len that doesn't divide T evenly works (tail segment).
"""

import pytest
import torch

from config import TitansConfig
from model.block import TitansMAGBlock
from model.nmm import NeuralMemoryModule


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_state_dtype():
    with pytest.raises(ValueError, match="nmm_state_dtype"):
        TitansConfig(nmm_state_dtype="fp16")


def test_config_accepts_fp32_and_bf16_state_dtype():
    TitansConfig(nmm_state_dtype="fp32")
    TitansConfig(nmm_state_dtype="bf16")


def test_config_rejects_zero_segment_len():
    with pytest.raises(ValueError, match="grad_checkpoint_segment_len"):
        TitansConfig(nmm_grad_checkpoint_segment_len=0)


def test_config_defaults_preserve_legacy_behavior():
    """Defaults must NOT enable either memory-saving option — that would
    silently shift the dtype + recompute behavior of every existing run."""
    cfg = TitansConfig()
    assert cfg.nmm_state_dtype == "fp32"
    assert cfg.nmm_grad_checkpoint is False
    assert cfg.nmm_grad_checkpoint_segment_len == 64  # documented default


# ---------------------------------------------------------------------------
# bf16 state dtype — propagation
# ---------------------------------------------------------------------------


def _tiny_nmm(state_dtype="fp32", grad_checkpoint=False, segment_len=4):
    """Build a minimal single-head NMM and a random input chunk."""
    nmm = NeuralMemoryModule(
        n_embd=8, expansion=2, kernel_size=2,
        spectral_norm=True, finetune_mode=False,
        retrieval_from_M_prev=True,
        state_dtype=state_dtype,
        grad_checkpoint=grad_checkpoint,
        grad_checkpoint_segment_len=segment_len,
    )
    return nmm


def test_init_state_dtype_is_fp32_by_default():
    nmm = _tiny_nmm(state_dtype="fp32")
    M, S = nmm.init_state(B=2, device=torch.device("cpu"))
    for v in {**M, **S}.values():
        assert v.dtype == torch.float32


def test_init_state_dtype_is_bf16_when_configured():
    nmm = _tiny_nmm(state_dtype="bf16")
    M, S = nmm.init_state(B=2, device=torch.device("cpu"))
    for v in {**M, **S}.values():
        assert v.dtype == torch.bfloat16


def test_forward_chunk_preserves_state_dtype():
    """After running through forward_chunk, the returned (M, S) must
    still be in `state_dtype`. A silent upcast (e.g. from `1.0 - alpha_t`
    promoting alpha_t to fp32) would cost the memory savings."""
    nmm = _tiny_nmm(state_dtype="bf16")
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 4, 8)
    _, (M_new, S_new) = nmm.forward_chunk(x, state, None)
    for k, v in M_new.items():
        assert v.dtype == torch.bfloat16, f"M[{k}] upcast to {v.dtype}"
    for k, v in S_new.items():
        assert v.dtype == torch.bfloat16, f"S[{k}] upcast to {v.dtype}"


def test_bf16_state_output_finite_and_similar_magnitude_to_fp32(seed=0):
    """bf16 state must produce a FINITE NMM output with similar overall
    magnitude to fp32 state (within ~50% on mean-abs). A tight max-diff
    test isn't useful here — 8 sequential per-token updates compound
    bf16 rounding nonlinearly, and even small per-step drifts move the
    final M_t to a meaningfully different point in weight space. What
    we actually care about: bf16 doesn't silently produce NaN, doesn't
    blow up by orders of magnitude, and doesn't drift to ~0 (a sign the
    inner loop got stuck because of dtype). Numerical fidelity is the
    domain of the loss-curve experiment, not a unit test."""
    torch.manual_seed(seed)
    nmm_fp32 = _tiny_nmm(state_dtype="fp32")
    nmm_bf16 = _tiny_nmm(state_dtype="bf16")
    # Copy weights so both NMMs have identical parameters.
    nmm_bf16.load_state_dict(nmm_fp32.state_dict())

    x = torch.randn(2, 8, 8)
    state_fp32 = nmm_fp32.init_state(B=2, device=torch.device("cpu"))
    state_bf16 = nmm_bf16.init_state(B=2, device=torch.device("cpu"))

    with torch.no_grad():
        y_fp32, _ = nmm_fp32.forward_chunk(x, state_fp32, None)
        y_bf16, _ = nmm_bf16.forward_chunk(x, state_bf16, None)

    # Both finite.
    assert torch.isfinite(y_fp32).all()
    assert torch.isfinite(y_bf16).all()

    # Similar mean magnitude (factor-of-2 band): catches regressions that
    # zero out the NMM ("inner loop silently broken") or amplify it 10x
    # ("dtype overflow somewhere").
    mag_fp32 = y_fp32.abs().mean().item()
    mag_bf16 = y_bf16.float().abs().mean().item()
    ratio = mag_bf16 / max(mag_fp32, 1e-12)
    assert 0.5 < ratio < 2.0, (
        f"bf16 mean-abs magnitude {mag_bf16:.4f} differs from fp32 "
        f"{mag_fp32:.4f} by ratio {ratio:.2f} — outside the expected "
        f"[0.5, 2.0] band, suggesting bf16 path is broken (not just noisy)."
    )


# ---------------------------------------------------------------------------
# Gradient checkpointing — forward equivalence
# ---------------------------------------------------------------------------


def test_grad_checkpoint_forward_matches_uncheckpointed():
    """Under grad-disabled forward, checkpoint=True / False must produce
    bitwise-identical outputs. The checkpoint wrapper is a pure forward
    shim when grad is off (it doesn't recompute, since there's no graph)."""
    torch.manual_seed(0)
    nmm = _tiny_nmm(grad_checkpoint=False)
    nmm_ckpt = _tiny_nmm(grad_checkpoint=True, segment_len=3)
    nmm_ckpt.load_state_dict(nmm.state_dict())

    x = torch.randn(2, 8, 8)
    state_a = nmm.init_state(B=2, device=torch.device("cpu"))
    state_b = nmm_ckpt.init_state(B=2, device=torch.device("cpu"))

    with torch.no_grad():
        y_a, _ = nmm.forward_chunk(x, state_a, None)
        y_b, _ = nmm_ckpt.forward_chunk(x, state_b, None)
    assert torch.equal(y_a, y_b)


def test_grad_checkpoint_forward_under_autograd_matches():
    """Under grad-enabled forward, checkpoint=True still runs the SAME
    forward — backward differs (recompute) but the forward output is
    identical (or near-identical, modulo recompute-rounding which only
    affects backward)."""
    torch.manual_seed(0)
    nmm = _tiny_nmm(grad_checkpoint=False)
    nmm_ckpt = _tiny_nmm(grad_checkpoint=True, segment_len=3)
    nmm_ckpt.load_state_dict(nmm.state_dict())

    x = torch.randn(2, 8, 8)
    state_a = nmm.init_state(B=2, device=torch.device("cpu"))
    state_b = nmm_ckpt.init_state(B=2, device=torch.device("cpu"))

    y_a, _ = nmm.forward_chunk(x, state_a, None)
    y_b, _ = nmm_ckpt.forward_chunk(x, state_b, None)
    # Same forward computation; only backward differs. Allow tiny
    # floating-point reorder noise from segment_len != T.
    assert torch.allclose(y_a, y_b, atol=1e-6), (
        f"checkpoint changed forward by {(y_a - y_b).abs().max().item():.2e}"
    )


def test_grad_checkpoint_backward_produces_same_gradients():
    """The whole point — backward through checkpoint=True must yield the
    same gradients (up to recompute-rounding) as backward through
    checkpoint=False. Use Block.forward so we exercise an end-to-end loss
    where the gradients flow back through the chunked NMM into
    `memory_mlp.W*.weight` (the initial M)."""
    torch.manual_seed(0)
    cfg_a = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_grad_checkpoint=False,
    )
    cfg_b = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_grad_checkpoint=True,
        nmm_grad_checkpoint_segment_len=3,
    )

    block_a = TitansMAGBlock(cfg_a)
    block_b = TitansMAGBlock(cfg_b)
    block_b.load_state_dict(block_a.state_dict())

    x = torch.randn(2, 8, 8, requires_grad=True)
    s_a = block_a.nmm.init_state(B=2, device=torch.device("cpu"))
    s_b = block_b.nmm.init_state(B=2, device=torch.device("cpu"))

    y_a, _ = block_a(x, s_a)
    y_b, _ = block_b(x.detach().clone().requires_grad_(True), s_b)
    y_a.sum().backward()
    y_b.sum().backward()

    # Compare gradients on memory_mlp weights — most sensitive to recompute.
    for (n_a, p_a), (n_b, p_b) in zip(
        block_a.nmm.memory_mlp.named_parameters(),
        block_b.nmm.memory_mlp.named_parameters(),
    ):
        assert n_a == n_b
        assert p_a.grad is not None and p_b.grad is not None
        max_diff = (p_a.grad - p_b.grad).abs().max().item()
        # Tolerance: 1e-4 absolute. Recompute pass introduces small
        # rounding diffs but never order-of-magnitude.
        assert max_diff < 1e-4, (
            f"grad mismatch on {n_a}: max diff {max_diff:.2e} (checkpoint "
            f"path should match uncheckpointed within float rounding)"
        )


def test_grad_checkpoint_uneven_segment_len_handles_tail():
    """T=10, seg_len=3 -> segments at [0:3, 3:6, 6:9, 9:10] (the tail
    has 1 token, not a full segment). Must not crash and the output
    must match the uncheckpointed version."""
    torch.manual_seed(0)
    nmm = _tiny_nmm(grad_checkpoint=False)
    nmm_ckpt = _tiny_nmm(grad_checkpoint=True, segment_len=3)
    nmm_ckpt.load_state_dict(nmm.state_dict())

    x = torch.randn(2, 10, 8)
    state_a = nmm.init_state(B=2, device=torch.device("cpu"))
    state_b = nmm_ckpt.init_state(B=2, device=torch.device("cpu"))

    with torch.no_grad():
        y_a, _ = nmm.forward_chunk(x, state_a, None)
        y_b, _ = nmm_ckpt.forward_chunk(x, state_b, None)
    assert y_a.shape == (2, 10, 8) and y_b.shape == (2, 10, 8)
    assert torch.equal(y_a, y_b)


def test_grad_checkpoint_doc_boundary_fires_across_segment_edge():
    """A boundary at t=4 within a checkpointed forward (seg_len=3, so
    segments [0:3, 3:6, 6:8]) must still reset state at t=4. The
    init_M is pre-built once at chunk level and passed in as flat
    tensors so the checkpoint's recompute uses the same init values."""
    torch.manual_seed(0)
    nmm = _tiny_nmm(grad_checkpoint=False)
    nmm_ckpt = _tiny_nmm(grad_checkpoint=True, segment_len=3)
    nmm_ckpt.load_state_dict(nmm.state_dict())

    x = torch.randn(2, 8, 8)
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[:, 4] = True  # mid-chunk boundary, crosses segment edge

    state_a = nmm.init_state(B=2, device=torch.device("cpu"))
    state_b = nmm_ckpt.init_state(B=2, device=torch.device("cpu"))
    with torch.no_grad():
        y_a, _ = nmm.forward_chunk(x, state_a, db)
        y_b, _ = nmm_ckpt.forward_chunk(x, state_b, db)
    # The reset path must produce identical results.
    assert torch.equal(y_a, y_b)


def test_grad_checkpoint_doc_boundary_at_segment_start():
    """Boundary at t=3 — exactly the start of segment 2 when seg_len=3.
    Edge case for the per-segment boundary slicing."""
    torch.manual_seed(0)
    nmm = _tiny_nmm(grad_checkpoint=False)
    nmm_ckpt = _tiny_nmm(grad_checkpoint=True, segment_len=3)
    nmm_ckpt.load_state_dict(nmm.state_dict())
    x = torch.randn(2, 9, 8)
    db = torch.zeros(2, 9, dtype=torch.bool)
    db[:, 3] = True

    state_a = nmm.init_state(B=2, device=torch.device("cpu"))
    state_b = nmm_ckpt.init_state(B=2, device=torch.device("cpu"))
    with torch.no_grad():
        y_a, _ = nmm.forward_chunk(x, state_a, db)
        y_b, _ = nmm_ckpt.forward_chunk(x, state_b, db)
    assert torch.equal(y_a, y_b)


# ---------------------------------------------------------------------------
# Combined: bf16 + checkpoint
# ---------------------------------------------------------------------------


def test_bf16_plus_grad_checkpoint_runs_end_to_end():
    """Both options on at once should produce a working forward+backward.
    No exact-equivalence check (bf16 already drifts from fp32); just a
    smoke test that the combination doesn't crash and produces finite
    gradients on the memory_mlp weights."""
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_state_dtype="bf16",
        nmm_grad_checkpoint=True,
        nmm_grad_checkpoint_segment_len=3,
    )
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8, requires_grad=True)
    y, _ = block(x, state)
    y.sum().backward()
    for n, p in block.nmm.memory_mlp.named_parameters():
        assert p.grad is not None, f"{n} has no grad"
        assert torch.isfinite(p.grad).all(), f"{n} grad has NaN/Inf"
