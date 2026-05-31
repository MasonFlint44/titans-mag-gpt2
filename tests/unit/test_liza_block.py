"""TitansLizaBlock + MemoryAsGate tests.

Covers the parallel softmax + DeltaProduct linear-attention topology
that mirrors TPTT's published recipe — distinct from the original
TitansMAGBlock topology where memory only multiplicatively gates
attention's output.
"""

import pytest
import torch

from config import TitansConfig
from model.block import MemoryAsGate, TitansLizaBlock


# ---------------------------------------------------------------------------
# MemoryAsGate
# ---------------------------------------------------------------------------


def test_mag_gate_zero_init_under_finetune_mode():
    """finetune_mode=True zero-inits the gate so output = o_base exactly
    at step 0 (preserves pretrained softmax-attention behavior)."""
    mag = MemoryAsGate(hidden_dim=8, finetune_mode=True)
    assert torch.all(mag.gate == 0.0)


def test_mag_gate_half_init_under_from_scratch_mode():
    """finetune_mode=False inits gate at 0.5 (rough equal mix)."""
    mag = MemoryAsGate(hidden_dim=8, finetune_mode=False)
    assert torch.allclose(mag.gate, torch.full((8,), 0.5))


def test_mag_forward_is_additive_with_per_channel_gate():
    """o = o_base + gate · o_lin; gate broadcasts across (B, T) and
    multiplies each channel of o_lin independently."""
    mag = MemoryAsGate(hidden_dim=4, finetune_mode=False)
    # Set distinct per-channel gate values.
    with torch.no_grad():
        mag.gate.copy_(torch.tensor([1.0, 0.0, 2.0, -0.5]))
    o_base = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])  # [B=1, T=1, d=4]
    o_lin = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    out = mag(o_lin, o_base)
    # Expected: o_base[i] + gate[i] · o_lin[i]
    expected = torch.tensor([[[11.0, 20.0, 32.0, 39.5]]])
    assert torch.allclose(out, expected)


def test_mag_at_zero_gate_output_equals_base():
    """With gate = 0, output must equal o_base exactly regardless of o_lin."""
    mag = MemoryAsGate(hidden_dim=4, finetune_mode=True)
    o_base = torch.randn(2, 5, 4)
    o_lin = torch.randn(2, 5, 4)
    out = mag(o_lin, o_base)
    assert torch.allclose(out, o_base)


# ---------------------------------------------------------------------------
# TitansLizaBlock construction & validation
# ---------------------------------------------------------------------------


def _liza_config(**overrides):
    """Standard LiZA-friendly TitansConfig for tests."""
    defaults = dict(
        n_layer=2, n_head=4, n_embd=16,
        memory_type="delta_product",
        memory_topology="liza",
        delta_order=2, delta_n_heads=4, delta_block_size=64,
        finetune_mode=True,
        chunk_size=64, block_size=64,
    )
    defaults.update(overrides)
    return TitansConfig(**defaults)


def test_liza_config_auto_overrides_nmm_n_persistent_to_zero():
    """memory_topology='liza' implies no persistent prefix. The config's
    __post_init__ silently sets nmm_n_persistent=0 even if the user
    passed a non-zero value."""
    cfg = TitansConfig(
        n_layer=2, n_head=4, n_embd=16, nmm_n_persistent=4,
        memory_type="delta_product", memory_topology="liza",
        chunk_size=64, block_size=64, finetune_mode=True,
    )
    assert cfg.nmm_n_persistent == 0


def test_liza_block_supports_nmm_memory_type():
    """LiZA was originally authored as TPTT's DeltaProduct-specific
    topology, but the block's combination math (additive MaG of
    `y_attn + gate · y_lin`) only requires the memory module to produce
    a `[B, T, n_embd]` output via `forward_chunk(...)`. NMM satisfies
    that contract, so the combination is allowed. This is the data-side
    half of the anti-marginal-output experiment recipe — we keep NMM's
    surprise-driven inner-loop update but lose MAG's multiplicative
    bottleneck that gates memory's contribution by attention's output."""
    cfg = TitansConfig(
        n_layer=2, n_head=4, n_embd=16,
        memory_type="nmm", memory_topology="liza",
        nmm_n_persistent=4,  # nonzero — should be auto-overridden to 0
        nmm_expansion=2,
        chunk_size=64, block_size=64,
        finetune_mode=True,
    )
    # `__post_init__` should silently zero `nmm_n_persistent` for LiZA,
    # matching the existing TPTT-style auto-override behavior.
    assert cfg.nmm_n_persistent == 0


def test_liza_block_supports_nmm_construction_and_forward():
    """End-to-end: building a LiZA block with NMM memory must produce a
    finite, correctly-shaped forward output. Defends against any latent
    coupling between LiZA's path and DeltaProduct's specific API
    surface (e.g., expecting `init_state()` to return a 3-tuple)."""
    import torch
    from model.titans_gpt2 import TitansMAGGPT2
    cfg = TitansConfig(
        n_layer=2, n_head=4, n_embd=16, vocab_size=32,
        memory_type="nmm", memory_topology="liza",
        nmm_expansion=2,
        chunk_size=64, block_size=64,
        finetune_mode=True,
    )
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, 32, (1, 8))
    out, _ = model(idx)
    assert out.shape == (1, 8, 32)
    assert torch.isfinite(out).all()


def test_liza_block_nmm_backward_propagates():
    """Gradient through the LiZA + NMM stack must reach the NMM's own
    learnable parameters (W1/W2/W_gate, W_alpha, W_eta, W_theta, conv
    weights). Without backward, training does nothing — defends against
    an accidental detach somewhere in the LiZA combination math.

    We use `finetune_mode=False` here so the MaG gate starts non-zero
    (the gate is zero-init under finetune_mode, which intentionally
    drops y_mem's contribution to zero at step 0 to preserve HF GPT-2
    behavior; that path is tested separately). With a non-zero gate,
    y_mem's contribution to the residual is non-zero from step 0 and
    gradients flow into the NMM as expected.
    """
    import torch
    from model.titans_gpt2 import TitansMAGGPT2
    cfg = TitansConfig(
        n_layer=2, n_head=4, n_embd=16, vocab_size=32,
        memory_type="nmm", memory_topology="liza",
        nmm_expansion=2,
        chunk_size=64, block_size=64,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    idx = torch.randint(0, 32, (1, 8))
    out, _ = model(idx)
    out.sum().backward()
    has_grad = False
    for n, p in model.named_parameters():
        if "nmm" in n and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, (
        "No NMM parameter received a non-zero gradient — backward isn't "
        "reaching the memory pathway"
    )


def test_liza_block_construction_with_valid_config():
    cfg = _liza_config()
    block = TitansLizaBlock(cfg)
    assert hasattr(block, "ln_1")
    assert hasattr(block, "attn")
    assert hasattr(block, "nmm")  # polymorphic with TitansMAGBlock
    assert hasattr(block, "mag")
    assert hasattr(block, "ln_2")
    assert hasattr(block, "mlp")


# ---------------------------------------------------------------------------
# Forward + invariants
# ---------------------------------------------------------------------------


def test_liza_block_forward_shape():
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=False)
    block = TitansLizaBlock(cfg)
    B, T, d = 2, 6, cfg.n_embd
    x = torch.randn(B, T, d)
    state = block.nmm.init_state(B, "cpu")
    y, new_state = block(x, state, doc_boundaries=None)
    assert y.shape == (B, T, d)
    # State threading: new_state is the DeltaProduct's 3-tuple shape.
    assert isinstance(new_state, tuple) and len(new_state) == 3


def test_liza_block_finetune_mode_reproduces_softmax_only_at_init():
    """At step 0 under finetune_mode, MaG's gate is zero so the
    linear-attention path contributes nothing to the residual.
    Equivalent block flow:
        o = softmax_attn(LN(x))                              (y_lin · 0 = 0)
        x = x + o
        x = x + MLP(LN(x))
    No M-state updates affect the output."""
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=True)
    block = TitansLizaBlock(cfg)

    B, T, d = 2, 6, cfg.n_embd
    x = torch.randn(B, T, d)
    state = block.nmm.init_state(B, "cpu")

    y, _ = block(x, state, doc_boundaries=None)

    # Reproduce the softmax-only path by hand: ln_1, attn (causal), residual, ln_2, mlp.
    x_norm = block.ln_1(x)
    mask = block._causal_mask(T, x.device, x.dtype)
    y_attn = block.attn(x_norm, mask=mask)
    expected = x + y_attn
    expected = expected + block.mlp(block.ln_2(expected))

    assert torch.allclose(y, expected, atol=1e-6)


def test_liza_block_with_nonzero_gate_diverges_from_softmax_only():
    """Sanity: once the MaG gate is non-zero, the LiZA block's output
    must differ from the softmax-only baseline (otherwise the linear
    pathway is doing nothing)."""
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=False)  # gate inits at 0.5
    block = TitansLizaBlock(cfg)

    B, T, d = 2, 6, cfg.n_embd
    x = torch.randn(B, T, d)
    state = block.nmm.init_state(B, "cpu")

    y, _ = block(x, state, doc_boundaries=None)

    # Softmax-only baseline.
    x_norm = block.ln_1(x)
    mask = block._causal_mask(T, x.device, x.dtype)
    y_attn = block.attn(x_norm, mask=mask)
    softmax_only = x + y_attn
    softmax_only = softmax_only + block.mlp(block.ln_2(softmax_only))

    assert not torch.allclose(y, softmax_only, atol=1e-3)


def test_liza_block_gradient_flows_to_all_params():
    """Backward through the block must reach every learnable parameter
    (softmax attn projections, DeltaProduct memory projections, MaG gate,
    MLP weights, LN affine params)."""
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=False)
    block = TitansLizaBlock(cfg)
    B, T, d = 2, 5, cfg.n_embd
    x = torch.randn(B, T, d, requires_grad=True)
    state = block.nmm.init_state(B, "cpu")
    y, _ = block(x, state, doc_boundaries=None)
    y.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().max() > 0, f"zero grad on {name}"


def test_liza_block_doc_boundaries_threaded_through_delta_product():
    """A boundary at position t should reset the linear-attention M
    state for that batch row. End-to-end test: a chunk with a mid-stream
    boundary should give different output than the same chunk without
    the boundary (since the memory pathway evolves differently)."""
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=False)
    block = TitansLizaBlock(cfg)
    B, T, d = 1, 6, cfg.n_embd
    x = torch.randn(B, T, d)
    state = block.nmm.init_state(B, "cpu")

    db_none = None
    db_mid = torch.tensor([[False, False, False, True, False, False]])

    y_none, _ = block(x, state, doc_boundaries=db_none)
    y_with, _ = block(x, state, doc_boundaries=db_mid)
    assert not torch.allclose(y_none, y_with, atol=1e-4)


def test_liza_block_finetune_zero_output_invariant_across_chunks():
    """Under finetune_mode, the linear-attention contribution stays
    zero across multiple forward calls (state evolves but MaG gate
    stays zero). End-to-end softmax-only behavior is preserved
    chunk-by-chunk."""
    torch.manual_seed(0)
    cfg = _liza_config(finetune_mode=True)
    block = TitansLizaBlock(cfg)

    B, T, d = 1, 4, cfg.n_embd
    x1 = torch.randn(B, T, d)
    x2 = torch.randn(B, T, d)
    state = block.nmm.init_state(B, "cpu")

    # First chunk: should equal softmax-only on x1.
    y1, state = block(x1, state)
    expected1 = block.ln_1(x1)
    mask = block._causal_mask(T, x1.device, x1.dtype)
    y_attn1 = block.attn(expected1, mask=mask)
    softmax_only_1 = x1 + y_attn1
    softmax_only_1 = softmax_only_1 + block.mlp(block.ln_2(softmax_only_1))
    assert torch.allclose(y1, softmax_only_1, atol=1e-6)

    # Second chunk: also softmax-only on x2 (state-evolution doesn't
    # leak into the output because gate=0).
    y2, _ = block(x2, state)
    expected2 = block.ln_1(x2)
    y_attn2 = block.attn(expected2, mask=mask)
    softmax_only_2 = x2 + y_attn2
    softmax_only_2 = softmax_only_2 + block.mlp(block.ln_2(softmax_only_2))
    assert torch.allclose(y2, softmax_only_2, atol=1e-6)
