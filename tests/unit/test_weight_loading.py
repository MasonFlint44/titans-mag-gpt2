"""HF GPT-2 weight transfer correctness — Phase 2.6 / `scripts.load_pretrained`.

(HF model name derived from n_embd; previously named test
in regression matrix didn't exist). Also covers Conv1D-transpose
correctness and all-params-loaded invariants from docs/TEST_PLAN.md §4.

These tests do NOT download from HuggingFace — they exercise the static
derivation logic and (where actual weight copy is needed) construct a
mock HF model in-process with matching dims. The end-to-end parity test
that DOES download lives in tests/parity/test_hf_logit_parity.py (slow).
"""

import pytest
import torch
import torch.nn as nn

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import _HF_GPT2_NAMES, load_pretrained


# ---------------------------------------------------------------------------
# HF model name derived from n_embd
# ---------------------------------------------------------------------------

def test_hf_model_name_from_n_embd_table():
    """_HF_GPT2_NAMES maps every supported n_embd to the correct HF Hub id."""
    assert _HF_GPT2_NAMES[768] == "openai-community/gpt2"
    assert _HF_GPT2_NAMES[1024] == "openai-community/gpt2-medium"
    assert _HF_GPT2_NAMES[1280] == "openai-community/gpt2-large"
    assert _HF_GPT2_NAMES[1600] == "openai-community/gpt2-xl"


def test_load_pretrained_rejects_unsupported_n_embd():
    """An n_embd not in _HF_GPT2_NAMES must raise BEFORE any download — caller
    catches this at validation time, not 30 seconds into an HF fetch."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=64,  # not in {768, 1024, 1280, 1600}
        vocab_size=50257, block_size=64, chunk_size=64, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
    )
    model = TitansMAGGPT2(cfg)
    with pytest.raises(ValueError, match="No HF GPT-2 checkpoint maps to n_embd"):
        load_pretrained(model, cfg)


def test_load_pretrained_factory_overrides_route_to_correct_hf_name():
    """gpt2_small/medium/large/xl factories yield n_embd values whose HF name
    derivation matches the spec. (Doesn't actually load — just verifies that
    config.n_embd is in _HF_GPT2_NAMES and the right entry.)"""
    assert TitansConfig.gpt2_small().n_embd == 768
    assert _HF_GPT2_NAMES[TitansConfig.gpt2_small().n_embd].endswith("/gpt2")
    assert TitansConfig.gpt2_medium().n_embd == 1024
    assert _HF_GPT2_NAMES[TitansConfig.gpt2_medium().n_embd].endswith("-medium")
    assert TitansConfig.gpt2_large().n_embd == 1280
    assert _HF_GPT2_NAMES[TitansConfig.gpt2_large().n_embd].endswith("-large")
    assert TitansConfig.gpt2_xl().n_embd == 1600
    assert _HF_GPT2_NAMES[TitansConfig.gpt2_xl().n_embd].endswith("-xl")


# ---------------------------------------------------------------------------
# Conv1D transpose correctness + all-params-loaded
#
# We construct a MOCK hf-like model with HF's [in, out] Conv1D-shaped weight
# layout, monkey-patch transformers.AutoModelForCausalLM.from_pretrained to
# return it, and verify load_pretrained transposes correctly.
# ---------------------------------------------------------------------------

class _FakeConv1D(nn.Module):
    """HF GPT-2's Conv1D layer: weight is [in_features, out_features]
    (transpose of nn.Linear's [out, in]). Bias is [out_features]."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))


class _FakeHFAttn(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # HF GPT-2 fuses Q, K, V into a single [n_embd, 3*n_embd] matrix.
        self.c_attn = _FakeConv1D(n_embd, 3 * n_embd)
        self.c_proj = _FakeConv1D(n_embd, n_embd)


class _FakeHFMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = _FakeConv1D(n_embd, 4 * n_embd)
        self.c_proj = _FakeConv1D(4 * n_embd, n_embd)


class _FakeHFBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = _FakeHFAttn(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = _FakeHFMLP(n_embd)


class _FakeHFTransformer(nn.Module):
    def __init__(self, n_layer, n_embd, vocab_size, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList([_FakeHFBlock(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)


class _FakeHFConfig:
    def __init__(self, n_layer, n_head):
        self.n_layer = n_layer
        self.n_head = n_head


class _FakeHFModel(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()
        self.transformer = _FakeHFTransformer(n_layer, n_embd, vocab_size, block_size)
        self.config = _FakeHFConfig(n_layer, n_head)


@pytest.fixture
def patched_hf(monkeypatch):
    """Patch AutoModelForCausalLM.from_pretrained to return our fake model
    so load_pretrained runs without touching the network."""
    fakes = {}  # captured fake model for later inspection

    def _fake_from_pretrained(name, **kwargs):
        # Decide config from name. Tests construct configs with n_embd=64
        # and n_layer=1, n_head=2 to keep the test fast — we register that
        # in _HF_GPT2_NAMES via monkeypatch for the test.
        cfg = fakes["cfg"]
        m = _FakeHFModel(
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
            vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        )
        fakes["model"] = m
        return m

    monkeypatch.setattr(
        "scripts.load_pretrained.AutoModelForCausalLM.from_pretrained",
        _fake_from_pretrained,
    )
    # Also register our tiny n_embd in _HF_GPT2_NAMES so load_pretrained
    # accepts it as a "supported" variant for the test.
    monkeypatch.setitem(_HF_GPT2_NAMES, 64, "fake-test-model")
    return fakes


def _tiny_cfg():
    return TitansConfig(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=128, block_size=16, chunk_size=16, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=0,
    )


def test_load_pretrained_transposes_c_attn_into_q_k_v_correctly(patched_hf):
    """HF's fused c_attn.weight is [n_embd, 3*n_embd]. Our q/k/v_proj are
    nn.Linear's [n_embd, n_embd] each (transposed). load_pretrained must
    chunk the c_attn weight along dim=1 then transpose each chunk."""
    cfg = _tiny_cfg()
    patched_hf["cfg"] = cfg
    model = TitansMAGGPT2(cfg)

    load_pretrained(model, cfg)
    hf = patched_hf["model"]

    for our_block, hf_block in zip(model.blocks, hf.transformer.h):
        # HF c_attn chunked along dim=1 gives W_q, W_k, W_v each [n_embd, n_embd].
        # Our attn.q_proj.weight is [n_embd, n_embd] but transposed (Linear convention).
        W_q, W_k, W_v = hf_block.attn.c_attn.weight.chunk(3, dim=1)
        b_q, b_k, b_v = hf_block.attn.c_attn.bias.chunk(3, dim=0)
        assert torch.equal(our_block.attn.q_proj.weight, W_q.T), \
            "q_proj.weight must be c_attn's Q-chunk transposed"
        assert torch.equal(our_block.attn.k_proj.weight, W_k.T)
        assert torch.equal(our_block.attn.v_proj.weight, W_v.T)
        assert torch.equal(our_block.attn.q_proj.bias, b_q)
        assert torch.equal(our_block.attn.k_proj.bias, b_k)
        assert torch.equal(our_block.attn.v_proj.bias, b_v)


def test_load_pretrained_transposes_c_proj_correctly(patched_hf):
    """attn.c_proj and mlp.c_proj are HF Conv1Ds — weight needs transpose."""
    cfg = _tiny_cfg()
    patched_hf["cfg"] = cfg
    model = TitansMAGGPT2(cfg)
    load_pretrained(model, cfg)
    hf = patched_hf["model"]

    for our_block, hf_block in zip(model.blocks, hf.transformer.h):
        assert torch.equal(
            our_block.attn.proj.weight, hf_block.attn.c_proj.weight.T,
        )
        assert torch.equal(our_block.attn.proj.bias, hf_block.attn.c_proj.bias)
        assert torch.equal(
            our_block.mlp.c_fc.weight, hf_block.mlp.c_fc.weight.T,
        )
        assert torch.equal(our_block.mlp.c_fc.bias, hf_block.mlp.c_fc.bias)
        assert torch.equal(
            our_block.mlp.c_proj.weight, hf_block.mlp.c_proj.weight.T,
        )
        assert torch.equal(our_block.mlp.c_proj.bias, hf_block.mlp.c_proj.bias)


def test_load_pretrained_copies_layernorms_without_transpose(patched_hf):
    """LayerNorm weight/bias layouts match between HF and us — no transpose."""
    cfg = _tiny_cfg()
    patched_hf["cfg"] = cfg
    model = TitansMAGGPT2(cfg)
    load_pretrained(model, cfg)
    hf = patched_hf["model"]

    for our_block, hf_block in zip(model.blocks, hf.transformer.h):
        assert torch.equal(our_block.ln_1.weight, hf_block.ln_1.weight)
        assert torch.equal(our_block.ln_1.bias, hf_block.ln_1.bias)
        assert torch.equal(our_block.ln_2.weight, hf_block.ln_2.weight)
        assert torch.equal(our_block.ln_2.bias, hf_block.ln_2.bias)
    assert torch.equal(model.ln_f.weight, hf.transformer.ln_f.weight)
    assert torch.equal(model.ln_f.bias, hf.transformer.ln_f.bias)


def test_load_pretrained_copies_embeddings(patched_hf):
    """wte and wpe are nn.Embeddings on both sides — direct copy."""
    cfg = _tiny_cfg()
    patched_hf["cfg"] = cfg
    model = TitansMAGGPT2(cfg)
    load_pretrained(model, cfg)
    hf = patched_hf["model"]
    assert torch.equal(model.wte.weight, hf.transformer.wte.weight)
    assert torch.equal(model.wpe.weight, hf.transformer.wpe.weight)


def test_load_pretrained_does_not_touch_nmm_params(patched_hf):
    """NMM params (memory_mlp.W*.weight, W_theta/eta/alpha, out_scale,
    persistent_mem, gamma_mem, ln_nmm) must NOT be overwritten by
    load_pretrained — they keep their own (Xavier/zero/randn-0.02) inits."""
    cfg = _tiny_cfg()
    patched_hf["cfg"] = cfg
    model = TitansMAGGPT2(cfg)

    # Snapshot every NMM/persistent/gate param before load.
    before = {}
    for name, p in model.named_parameters():
        if any(s in name for s in ("nmm", "persistent_mem", "gamma", "ln_nmm")):
            before[name] = p.detach().clone()

    load_pretrained(model, cfg)

    # All those params must be unchanged.
    for name, p in model.named_parameters():
        if name in before:
            assert torch.equal(p, before[name]), (
                f"{name} was modified by load_pretrained — NMM/gate/persistent "
                f"params should keep their own inits."
            )


def test_load_pretrained_n_layer_mismatch_raises(patched_hf):
    """A config with n_layer not matching the HF model raises clearly."""
    cfg = _tiny_cfg()
    cfg_mismatch = TitansConfig(
        n_layer=cfg.n_layer + 1,  # mismatch
        n_head=cfg.n_head, n_embd=cfg.n_embd,
        vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        chunk_size=cfg.chunk_size, dropout=cfg.dropout,
        nmm_expansion=cfg.nmm_expansion,
        nmm_n_persistent=cfg.nmm_n_persistent,
    )
    patched_hf["cfg"] = cfg  # the FAKE has cfg.n_layer
    model = TitansMAGGPT2(cfg_mismatch)
    with pytest.raises(ValueError, match="n_layer"):
        load_pretrained(model, cfg_mismatch)


def test_load_pretrained_n_head_mismatch_raises(patched_hf):
    """Head-count mismatch is incompatible with the QKV split — raise loudly."""
    cfg = _tiny_cfg()
    cfg_mismatch = TitansConfig(
        n_layer=cfg.n_layer,
        n_head=4,  # mismatch (HF will say 2 per _tiny_cfg)
        n_embd=cfg.n_embd,
        vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        chunk_size=cfg.chunk_size, dropout=cfg.dropout,
        nmm_expansion=cfg.nmm_expansion,
        nmm_n_persistent=cfg.nmm_n_persistent,
    )
    patched_hf["cfg"] = cfg  # the FAKE has cfg.n_head=2
    model = TitansMAGGPT2(cfg_mismatch)
    with pytest.raises(ValueError, match="n_head"):
        load_pretrained(model, cfg_mismatch)
