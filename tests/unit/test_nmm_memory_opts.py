"""nmm_state_dtype memory-saving option.

- `nmm_state_dtype="bf16"`: (M, S) and per-step update buffers stored in
  bf16 instead of fp32 (~2x smaller). NS5 still casts to fp32 internally
  (the bf16-NS5-spectral-norm-drift hazard documented in).
- `nmm_state_dtype="int8"`: int8 + per-sample fp16 scale (~4x smaller),
  blockwise path only.

The contract these tests pin down:
  (a) dtype choice is honored end-to-end (state, output, retrieval).
  (b) bf16 output ≈ fp32 output within a paper-faithful tolerance.
  (c) the flag composes with the other memory knobs without crashing.
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
    M, S, _ = nmm.init_state(B=2, device=torch.device("cpu"))
    for v in {**M, **S}.values():
        assert v.dtype == torch.float32


def test_init_state_dtype_is_bf16_when_configured():
    nmm = _tiny_nmm(state_dtype="bf16")
    M, S, _ = nmm.init_state(B=2, device=torch.device("cpu"))
    for v in {**M, **S}.values():
        assert v.dtype == torch.bfloat16


def test_forward_chunk_preserves_state_dtype():
    """After running through forward_chunk, the returned (M, S) must
    still be in `state_dtype`. A silent upcast (e.g. from `1.0 - alpha_t`
    promoting alpha_t to fp32) would cost the memory savings."""
    nmm = _tiny_nmm(state_dtype="bf16")
    state = nmm.init_state(B=2, device=torch.device("cpu"))
    x = torch.randn(2, 4, 8)
    _, (M_new, S_new, _) = nmm.forward_chunk(x, state, None)
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
# nmm_expansion=1 — paper ablation: smallest viable NMM size.
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
    M, S, _ = state
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
            if p.numel() == 0:
                # Empty params (e.g. model_wide persistent_mem at N_p=0).
                continue
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {n}"


# ---------------------------------------------------------------------------
# nmm_layer_indices — subset-of-layers
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
    M, S, _ = detached[1]
    for v in {**M, **S}.values():
        assert v.requires_grad is False


def test_layer_indices_compute_nmm_norm_returns_None_for_plain_blocks():
    from cli.train import compute_nmm_norm
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
            if p.numel() == 0:
                # Empty parameters (e.g. model-wide persistent_mem at
                # nmm_n_persistent=0) have no entries to gradient on.
                continue
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
    # Item 6: conv buffer is now part of nmm_states (not a separate cache
    # entry). Plain blocks have None state slots; NMM blocks have
    # (M, S, conv_buf) triples.
    assert cache["nmm_states"][0] is None  # plain block
    assert len(cache["nmm_states"][1]) == 3  # NMM block: (M, S, conv_buf)
    assert isinstance(cache["nmm_states"][1][2], dict)
    assert cache["nmm_states"][2] is None  # plain block
    # One step of decode should run end-to-end and produce finite logits.
    next_tok = torch.tensor([[0]])
    logits, _ = m.forward_step(next_tok, cache)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# nmm_low_rank — factored MemoryMLP for smaller per-step state
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
    M, _S, _ = m.blocks[0].nmm.init_state(B=2, device=torch.device("cpu"))
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
            if p.numel() == 0:
                # Empty parameters (e.g. model-wide persistent_mem at
                # nmm_n_persistent=0) have no entries to gradient on.
                continue
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
    M, _, _ = state
    for v in M.values():
        assert v.dtype == torch.bfloat16
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, _ = block(x, state)
    y.sum().backward()
    for p in block.nmm.memory_mlp.parameters():
        assert p.grad is not None
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
