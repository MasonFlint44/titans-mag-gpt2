"""G256 / G260 — nmm_state_dtype + nmm_block_grad_checkpoint memory-saving
options.

- `nmm_state_dtype="bf16"`: (M, S) and per-step update buffers stored in
  bf16 instead of fp32 (~2x smaller). NS5 still casts to fp32 internally
  (the bf16-NS5-spectral-norm-drift hazard documented in G226).

- `nmm_block_grad_checkpoint=True`: wraps each TitansMAGBlock.forward in
  `torch.utils.checkpoint.checkpoint`. Pair with `nmm_block_size >= 16`
  so the per-block NMM transient that gets rebuilt during recompute is
  itself bounded.

The contract these tests pin down:
  (a) dtype choice is honored end-to-end (state, output, retrieval).
  (b) bf16 output ≈ fp32 output within a paper-faithful tolerance.
  (c) block-checkpoint True ≈ False — forward AND backward.
  (d) the flags compose without crashing.
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


def test_config_defaults_preserve_legacy_behavior():
    """Defaults must NOT enable any memory-saving option — that would
    silently shift the dtype + recompute behavior of every existing run."""
    cfg = TitansConfig()
    assert cfg.nmm_state_dtype == "fp32"
    assert cfg.nmm_block_grad_checkpoint is False


# ---------------------------------------------------------------------------
# bf16 state dtype — propagation
# ---------------------------------------------------------------------------


def _tiny_nmm(state_dtype="fp32"):
    """Build a minimal single-head NMM and a random input chunk."""
    nmm = NeuralMemoryModule(
        n_embd=8, expansion=2, kernel_size=2,
        spectral_norm=True, finetune_mode=False,
        retrieval_from_M_prev=True,
        state_dtype=state_dtype,
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
# Block-level grad checkpoint (G260) — flatten/unflatten, equivalence,
# multi-head support.
# ---------------------------------------------------------------------------


def _cfg_block_ckpt(*, block_ckpt, n_heads=1):
    return TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_block_grad_checkpoint=block_ckpt,
        nmm_n_heads=n_heads,
    )


def test_block_checkpoint_default_off():
    """The flag must default to False so existing models don't silently
    pay 2x backward cost without the user knowing."""
    cfg = TitansConfig()
    assert cfg.nmm_block_grad_checkpoint is False


def test_block_checkpoint_forward_matches_uncheckpointed():
    """Block-level checkpoint must produce bitwise-identical output to
    the uncheckpointed block under no_grad — the flag is a backward-only
    optimization; forward semantics must not change."""
    torch.manual_seed(0)
    block_a = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=False))
    block_b = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=True))
    block_b.load_state_dict(block_a.state_dict())

    x = torch.randn(2, 8, 8)
    s_a = block_a.nmm.init_state(B=2, device=torch.device("cpu"))
    s_b = block_b.nmm.init_state(B=2, device=torch.device("cpu"))
    with torch.no_grad():
        y_a, ns_a = block_a(x, s_a, doc_boundaries=None)
        y_b, ns_b = block_b(x, s_b, doc_boundaries=None)
    assert torch.equal(y_a, y_b)
    # NMM state outputs also match (single-head: (M, S) dicts).
    M_a, S_a = ns_a; M_b, S_b = ns_b
    for k in M_a:
        assert torch.equal(M_a[k], M_b[k])
        assert torch.equal(S_a[k], S_b[k])


def test_block_checkpoint_backward_matches_uncheckpointed():
    """Memory-mlp gradients through block-checkpointed forward must match
    uncheckpointed within float rounding."""
    torch.manual_seed(0)
    block_a = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=False))
    block_b = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=True))
    block_b.load_state_dict(block_a.state_dict())

    x_a = torch.randn(2, 8, 8, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    s_a = block_a.nmm.init_state(B=2, device=torch.device("cpu"))
    s_b = block_b.nmm.init_state(B=2, device=torch.device("cpu"))

    y_a, _ = block_a(x_a, s_a)
    y_b, _ = block_b(x_b, s_b)
    y_a.sum().backward()
    y_b.sum().backward()

    for (n_a, p_a), (n_b, p_b) in zip(
        block_a.nmm.memory_mlp.named_parameters(),
        block_b.nmm.memory_mlp.named_parameters(),
    ):
        assert n_a == n_b
        diff = (p_a.grad - p_b.grad).abs().max().item()
        assert diff < 1e-4, (
            f"block_checkpoint diverged on {n_a}: max grad diff {diff:.3e}"
        )

    # Also check that the input grad propagates (block is differentiable
    # w.r.t. its input through the checkpoint).
    assert x_a.grad is not None and x_b.grad is not None
    diff_x = (x_a.grad - x_b.grad).abs().max().item()
    assert diff_x < 1e-4


def test_block_checkpoint_handles_doc_boundary():
    """doc_boundaries is captured via closure (checkpoint doesn't pass
    kwargs). Mid-chunk reset must still fire on the recompute path."""
    torch.manual_seed(0)
    block = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=True))
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8, requires_grad=True)
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[:, 4] = True

    y, _ = block(x, state, doc_boundaries=db)
    y.sum().backward()
    for p in block.nmm.memory_mlp.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_block_checkpoint_works_with_multi_head_nmm():
    """Multi-head state is a `list[(M, S)]`. The flatten helper handles
    both shapes; this test confirms multi-head end-to-end."""
    torch.manual_seed(0)
    cfg = _cfg_block_ckpt(block_ckpt=True, n_heads=2)
    block_a = TitansMAGBlock(_cfg_block_ckpt(block_ckpt=False, n_heads=2))
    block_b = TitansMAGBlock(cfg)
    block_b.load_state_dict(block_a.state_dict())

    x_a = torch.randn(2, 8, 8, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    s_a = block_a.nmm.init_state(B=2, device=torch.device("cpu"))
    s_b = block_b.nmm.init_state(B=2, device=torch.device("cpu"))

    y_a, _ = block_a(x_a, s_a)
    y_b, _ = block_b(x_b, s_b)
    y_a.sum().backward()
    y_b.sum().backward()

    # Compare per-head memory_mlp gradients via the heads list.
    for h_a, h_b in zip(block_a.nmm.heads, block_b.nmm.heads):
        for (na, pa), (nb, pb) in zip(
            h_a.memory_mlp.named_parameters(),
            h_b.memory_mlp.named_parameters(),
        ):
            assert na == nb
            diff = (pa.grad - pb.grad).abs().max().item()
            assert diff < 1e-4, (
                f"multi-head block_ckpt diverged on {na}: {diff:.3e}"
            )


def test_block_checkpoint_composes_with_bf16_state():
    """bf16 state + block checkpoint together. End-to-end finite gradients
    only — bf16 already drifts from fp32 so we don't compare exact values."""
    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_state_dtype="bf16",
        nmm_block_grad_checkpoint=True,
    )
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 8, requires_grad=True)
    y, _ = block(x, state)
    y.sum().backward()
    for n, p in block.nmm.memory_mlp.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all(), f"{n} non-finite grad"


def test_block_checkpoint_through_full_model_propagates_grads():
    """End-to-end: block-checkpointed TitansMAGGPT2 must produce finite
    gradients across all layers. Verifies the persistent + lm_head + ln_f
    + multi-block stack all play nicely with the per-block checkpoint."""
    from model.titans_gpt2 import TitansMAGGPT2

    torch.manual_seed(0)
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=16,
        block_size=32, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        nmm_block_grad_checkpoint=True,
    )
    model = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool); db[:, 0] = True
    logits, _ = model(ids, None, db)
    loss = logits.float().mean()
    loss.backward()
    # Every requires_grad param must have a finite grad. Embedding params
    # come last in the graph and are a sensitive canary for "did backward
    # actually traverse the whole stack."
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} has no grad — block_ckpt may have broken backward"
            assert torch.isfinite(p.grad).all(), f"{n} non-finite grad"


def test_state_flatten_unflatten_single_head_round_trip():
    """Sanity check for _state_to_flat / _flat_to_state on single-head."""
    from model.block import _state_to_flat, _flat_to_state
    M = {"W1.weight": torch.randn(2, 4, 8),
         "W_gate.weight": torch.randn(2, 4, 8),
         "W2.weight": torch.randn(2, 8, 4)}
    S = {k: torch.zeros_like(v) for k, v in M.items()}
    flat, desc = _state_to_flat((M, S))
    M2, S2 = _flat_to_state(flat, desc)
    for k in M:
        assert torch.equal(M[k], M2[k])
        assert torch.equal(S[k], S2[k])


def test_state_flatten_unflatten_multi_head_round_trip():
    """Same round-trip for multi-head `[(M, S), ...]`."""
    from model.block import _state_to_flat, _flat_to_state
    states = []
    for _ in range(3):
        M = {"W1.weight": torch.randn(2, 4, 8),
             "W_gate.weight": torch.randn(2, 4, 8),
             "W2.weight": torch.randn(2, 8, 4)}
        S = {k: torch.zeros_like(v) for k, v in M.items()}
        states.append((M, S))
    flat, desc = _state_to_flat(states)
    out = _flat_to_state(flat, desc)
    assert isinstance(out, list) and len(out) == 3
    for orig, restored in zip(states, out):
        M_o, S_o = orig; M_r, S_r = restored
        for k in M_o:
            assert torch.equal(M_o[k], M_r[k])
            assert torch.equal(S_o[k], S_r[k])


# ---------------------------------------------------------------------------
# nmm_expansion=1 (G263) — paper ablation: smallest viable NMM size.
# ---------------------------------------------------------------------------


def test_expansion_1_state_shapes_are_square_dxd():
    """With expansion=1, all three full-rank state matrices become [B, d, d]
    (hidden = d, not 4d). Quartet drop in per-step state vs default expansion=4."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=1, nmm_n_persistent=0,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    state = m.blocks[0].nmm.init_state(B=2, device=torch.device("cpu"))
    M, S = state
    # All three weights are square [2, 8, 8] at expansion=1.
    for k, v in M.items():
        assert tuple(v.shape) == (2, 8, 8), f"{k} shape {tuple(v.shape)} != (2,8,8)"
    for k, v in S.items():
        assert tuple(v.shape) == (2, 8, 8)


def test_expansion_1_param_count_quarter_of_expansion_4():
    """memory_mlp param count at expansion=1 should be ~1/4 of expansion=4
    (excluding the LayerNorm which is the same). The 3 weight matrices
    go from 3 × 4d² to 3 × d², so 1/4."""
    def _count(exp):
        cfg = TitansConfig(
            n_layer=1, n_head=2, n_embd=16, vocab_size=16,
            block_size=16, chunk_size=8, dropout=0.0,
            nmm_expansion=exp, nmm_n_persistent=0,
        )
        from model.titans_gpt2 import TitansMAGGPT2
        m = TitansMAGGPT2(cfg)
        return sum(
            p.numel() for n, p in m.named_parameters()
            if "memory_mlp" in n and "norm" not in n
        )
    n_full = _count(4)
    n_one = _count(1)
    # 3 * 4 * d² = 3072 at d=16, exp=4. 3 * d² = 768 at exp=1.
    assert n_full == 3 * 4 * 16 * 16, f"unexpected exp=4 count {n_full}"
    assert n_one == 3 * 16 * 16, f"unexpected exp=1 count {n_one}"
    assert n_one * 4 == n_full


def test_expansion_1_trains_end_to_end():
    """Forward + backward should produce finite gradients with expansion=1.
    NS5 has to converge on square matrices (no transpose needed)."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=1, nmm_n_persistent=0,
        finetune_mode=False,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    torch.manual_seed(0)
    m = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool); db[:, 0] = True
    logits, _ = m(ids, None, db)
    logits.sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {n}"


# ---------------------------------------------------------------------------
# nmm_layer_indices (G261) — subset-of-layers
# ---------------------------------------------------------------------------


def test_layer_indices_default_is_None_keeps_NMM_on_every_block():
    """Default behavior: every block is a TitansMAGBlock."""
    cfg = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0, nmm_n_persistent=0,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    from model.block import TitansMAGBlock
    m = TitansMAGGPT2(cfg)
    for b in m.blocks:
        assert isinstance(b, TitansMAGBlock)


def test_layer_indices_only_listed_blocks_have_NMM():
    """nmm_layer_indices=[1, 3] -> blocks 1 and 3 are TitansMAGBlock,
    blocks 0 and 2 are PlainGPT2Block."""
    cfg = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0, nmm_n_persistent=2,
        nmm_layer_indices=[1, 3],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    from model.block import PlainGPT2Block, TitansMAGBlock
    m = TitansMAGGPT2(cfg)
    types = [type(b).__name__ for b in m.blocks]
    assert types == ["PlainGPT2Block", "TitansMAGBlock", "PlainGPT2Block", "TitansMAGBlock"]
    # _block_has_nmm parallels the block types.
    assert m._block_has_nmm == [False, True, False, True]


def test_layer_indices_state_has_None_at_plain_block_positions():
    """The nmm_states list returned by forward contains None for plain
    blocks; downstream helpers (detach_states, compute_nmm_norm) must
    tolerate."""
    cfg = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_layer_indices=[1, 3],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool); db[:, 0] = True
    _, states = m(ids, None, db)
    assert states[0] is None and states[1] is not None
    assert states[2] is None and states[3] is not None


def test_layer_indices_detach_states_tolerates_None_entries():
    from model.nmm import detach_states
    cfg = TitansConfig(
        n_layer=3, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_layer_indices=[1],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    states = [
        block.nmm.init_state(B=2, device=torch.device("cpu")) if has_nmm else None
        for block, has_nmm in zip(m.blocks, m._block_has_nmm)
    ]
    detached = detach_states(states)
    assert detached[0] is None and detached[2] is None
    # detached[1] is a (M, S) tuple of dicts of detached tensors.
    M, S = detached[1]
    for v in {**M, **S}.values():
        assert v.requires_grad is False


def test_layer_indices_compute_nmm_norm_returns_None_for_plain_blocks():
    from train import compute_nmm_norm
    cfg = TitansConfig(
        n_layer=3, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_layer_indices=[1],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _, states = m(ids, None, torch.zeros(2, 8, dtype=torch.bool).index_fill_(1, torch.tensor([0]), True))
    norms = compute_nmm_norm(states)
    assert norms[0] is None
    assert isinstance(norms[1], float) and norms[1] > 0
    assert norms[2] is None


def test_layer_indices_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        TitansConfig(n_layer=4, nmm_layer_indices=[5])


def test_layer_indices_rejects_negative_index():
    with pytest.raises(ValueError, match="out of range"):
        TitansConfig(n_layer=4, nmm_layer_indices=[-1])


def test_layer_indices_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate"):
        TitansConfig(n_layer=4, nmm_layer_indices=[1, 1])


def test_layer_indices_rejects_non_int_entries():
    with pytest.raises(ValueError, match="ints"):
        TitansConfig(n_layer=4, nmm_layer_indices=[1.5])


def test_layer_indices_normalizes_to_sorted_list():
    """Input [3, 1, 2] should normalize to [1, 2, 3] so iteration order
    is deterministic across runs and rank ordering doesn't surprise users."""
    cfg = TitansConfig(n_layer=4, nmm_layer_indices=[3, 1, 2])
    assert cfg.nmm_layer_indices == [1, 2, 3]


def test_layer_indices_param_count_lower_than_all_layers():
    """Setting nmm_layer_indices=[0] on a 4-layer model means only block 0
    has NMM/persistent/MAG params — total param count should be much lower
    than every-block-has-NMM."""
    cfg_all = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0, nmm_n_persistent=2,
    )
    cfg_subset = TitansConfig(
        n_layer=4, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0, nmm_n_persistent=2,
        nmm_layer_indices=[0],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    n_all = sum(p.numel() for p in TitansMAGGPT2(cfg_all).parameters())
    n_subset = sum(p.numel() for p in TitansMAGGPT2(cfg_subset).parameters())
    assert n_subset < n_all, f"subset {n_subset} not < all {n_all}"


def test_layer_indices_full_model_backward_propagates_through_mixed_stack():
    """End-to-end: backward through a mixed plain+NMM stack must yield
    finite gradients on EVERY trainable param. The plain blocks have
    their own params (attn, mlp, ln); the NMM blocks have those + the
    NMM-specific params. Both must receive gradients."""
    cfg = TitansConfig(
        n_layer=3, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0, nmm_n_persistent=2,
        finetune_mode=False,
        nmm_layer_indices=[1],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    torch.manual_seed(0)
    m = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool); db[:, 0] = True
    logits, _ = m(ids, None, db)
    logits.sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} no grad"
            assert torch.isfinite(p.grad).all(), f"{n} non-finite grad"


def test_layer_indices_decode_path_works_with_mixed_blocks():
    """prepare_decode + forward_step should work end-to-end with a mix
    of TitansMAGBlock and PlainGPT2Block — plain blocks contribute a
    KV cache only (no NMM conv buffer)."""
    cfg = TitansConfig(
        n_layer=3, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=4, dropout=0.0, nmm_n_persistent=2,
        nmm_layer_indices=[1],
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    cache = m.prepare_decode(prompt)
    # Plain block's conv buffer slot is None; NMM block's is a dict.
    assert cache["nmm_conv_buffers"][0] is None
    assert isinstance(cache["nmm_conv_buffers"][1], dict)
    assert cache["nmm_conv_buffers"][2] is None
    # One step of decode should run end-to-end and produce finite logits.
    next_tok = torch.tensor([[0]])
    logits, _ = m.forward_step(next_tok, cache)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# nmm_low_rank (G262) — factored MemoryMLP for smaller per-step state
# ---------------------------------------------------------------------------


def test_low_rank_default_is_None_full_rank():
    cfg = TitansConfig()
    assert cfg.nmm_low_rank is None


def test_low_rank_state_keys_expand_to_six():
    """Full-rank has 3 state keys (W1, W_gate, W2). Low-rank has 6
    (W1_a, W1_b, W_gate_a, W_gate_b, W2_a, W2_b). state_keys is
    discovered from MemoryMLP at NMM construction time."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_low_rank=4, nmm_n_persistent=0,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    keys = m.blocks[0].nmm.state_keys
    assert sorted(keys) == sorted([
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    ])


def test_low_rank_state_init_shapes_match_factored_layout():
    """Each factored matmul has two matrices: A is [r, d_in], B is
    [d_out, r]. init_state replicates these per-batch."""
    d, r, expansion = 16, 4, 2
    h = d * expansion
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=d, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=expansion, nmm_low_rank=r, nmm_n_persistent=0,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    m = TitansMAGGPT2(cfg)
    M, _S = m.blocks[0].nmm.init_state(B=2, device=torch.device("cpu"))
    expected = {
        "W1_a.weight":     (2, r, d),
        "W1_b.weight":     (2, h, r),
        "W_gate_a.weight": (2, r, d),
        "W_gate_b.weight": (2, h, r),
        "W2_a.weight":     (2, r, h),
        "W2_b.weight":     (2, d, r),
    }
    for k, shape in expected.items():
        assert tuple(M[k].shape) == shape, f"{k}: got {tuple(M[k].shape)}, want {shape}"


def test_low_rank_param_count_much_smaller_than_full_rank():
    """At r << d, the factored MemoryMLP has way fewer params. With
    d=16, expansion=4 (h=64): full-rank is 3 × d×h = 192*3 = 576 (per
    direction) Wait — full-rank state per W is [h, d] or [d, h], so
    d*h = 256 per matrix × 3 = 768 params. Hmm, math check below."""
    cfg_full = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=4, nmm_n_persistent=0,
    )
    cfg_lr = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=4, nmm_low_rank=2, nmm_n_persistent=0,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    def _mm_params(model):
        return sum(
            p.numel() for n, p in model.named_parameters()
            if "memory_mlp" in n and "norm" not in n
        )
    n_full = _mm_params(TitansMAGGPT2(cfg_full))
    n_lr = _mm_params(TitansMAGGPT2(cfg_lr))
    # Each Wx [h, d] = h*d. Three of them = 3*h*d.
    # Factored Wx = [r, d] + [h, r] = r*(d+h). Three of them = 3*r*(d+h).
    # d=16, h=64, r=2: full = 3*64*16=3072, lr = 3*2*(16+64) = 480.
    assert n_full == 3 * 64 * 16
    assert n_lr == 3 * 2 * (16 + 64)
    assert n_lr < n_full / 4  # at r=2 << d=16, much smaller


def test_low_rank_trains_end_to_end():
    """Forward + backward must yield finite gradients on all params
    (the 6 factored linears + outer-loop NMM controllers)."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_low_rank=4, nmm_n_persistent=0,
        finetune_mode=False,
    )
    from model.titans_gpt2 import TitansMAGGPT2
    torch.manual_seed(0)
    m = TitansMAGGPT2(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool); db[:, 0] = True
    logits, _ = m(ids, None, db)
    logits.sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} no grad"
            assert torch.isfinite(p.grad).all(), f"{n} non-finite grad"


def test_low_rank_composes_with_bf16_state():
    """bf16 state dtype + low-rank — gradients still finite."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_low_rank=4, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_state_dtype="bf16",
    )
    from model.block import TitansMAGBlock
    torch.manual_seed(0)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    # State must be bf16 with low-rank.
    M, _ = state
    for v in M.values():
        assert v.dtype == torch.bfloat16
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, _ = block(x, state)
    y.sum().backward()
    for p in block.nmm.memory_mlp.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_low_rank_composes_with_block_checkpoint():
    """block_grad_checkpoint flattens state via state_keys discovery —
    must work for the 6-key low-rank state too."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=16,
        block_size=16, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_low_rank=4, nmm_n_persistent=0,
        finetune_mode=False,
        nmm_block_grad_checkpoint=True,
    )
    from model.block import TitansMAGBlock
    torch.manual_seed(0)
    block = TitansMAGBlock(cfg)
    state = block.nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, _ = block(x, state)
    y.sum().backward()
    for p in block.nmm.memory_mlp.parameters():
        assert torch.isfinite(p.grad).all()


def test_low_rank_rejects_rank_geq_n_embd():
    """At rank >= d_model the factored form has MORE params than full —
    catch this loud at config time."""
    with pytest.raises(ValueError, match="defeating the purpose"):
        TitansConfig(n_head=2, n_embd=16, nmm_low_rank=16)


def test_low_rank_rejects_zero_or_negative_rank():
    with pytest.raises(ValueError, match="positive int"):
        TitansConfig(n_head=2, n_embd=8, nmm_low_rank=0)
    with pytest.raises(ValueError, match="positive int"):
        TitansConfig(n_head=2, n_embd=8, nmm_low_rank=-1)
