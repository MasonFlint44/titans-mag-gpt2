"""Phase 5.1 — generate(), and Phase 5.2 — perplexity() (unit-level)."""

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from eval import perplexity
from generate import generate
from model.titans_gpt2 import TitansMAGGPT2


def _tiny_model_real_vocab():
    """Tiny model using the real GPT-2 vocab — needed because generate uses
    Tokenizer.encode/decode and the model's lm_head must match vocab_size."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=32, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    return cfg, TitansMAGGPT2(cfg)


# ---------------------------------------------------------------------------
# G161 — generate restores model.training
# ---------------------------------------------------------------------------

def test_generate_restores_training_mode_when_called_in_train_mode():
    cfg, model = _tiny_model_real_vocab()
    model.train()
    tok = Tokenizer()
    _ = generate(model, "hello", max_new_tokens=2, top_k=10, tokenizer=tok)
    assert model.training is True


def test_generate_keeps_eval_mode_if_caller_was_in_eval():
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    _ = generate(model, "hello", max_new_tokens=2, top_k=10, tokenizer=tok)
    assert model.training is False


# ---------------------------------------------------------------------------
# G173 — sampling order: temperature, top_k, softmax
# ---------------------------------------------------------------------------

def test_temperature_zero_is_deterministic_argmax():
    """With temperature=0 the function takes argmax — same prompt -> same token."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    out1 = generate(model, "hello", max_new_tokens=5, temperature=0, top_k=None, tokenizer=tok)
    out2 = generate(model, "hello", max_new_tokens=5, temperature=0, top_k=None, tokenizer=tok)
    assert out1 == out2


def test_top_k_actually_filters_to_top_k_tokens():
    """With top_k=1 the sampled token must equal the argmax — same as temp=0."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    torch.manual_seed(0)
    out_topk1 = generate(model, "hello", max_new_tokens=3, temperature=1.0, top_k=1, tokenizer=tok)
    out_argmax = generate(model, "hello", max_new_tokens=3, temperature=0, top_k=None, tokenizer=tok)
    assert out_topk1 == out_argmax


# ---------------------------------------------------------------------------
# G208 — tokenizer reuse
# ---------------------------------------------------------------------------

def test_generate_accepts_caller_supplied_tokenizer():
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    # Should not raise and should use the same eot id.
    out = generate(model, "hi", max_new_tokens=3, top_k=10, tokenizer=tok)
    assert isinstance(out, str)


def test_generate_constructs_default_tokenizer_when_none_passed():
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    out = generate(model, "hi", max_new_tokens=3, top_k=10)
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# G176 — prompts > block_size are chunked, not truncated
# ---------------------------------------------------------------------------

def test_generate_chunks_long_prompts_through_NMM():
    """G176 — for prompt_len > block_size, the full prompt is processed
    (no [-block_size:] truncation). The new cached-decode generate splits
    this into:
      - model.forward() calls for the prefix [0 .. prompt_len - block_size]
      - model.prepare_decode() once for the final block_size tail
    Verify both call counts so the total covers the full prompt."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()

    long_prompt = "the quick brown fox jumps over the lazy dog " * 50
    prompt_ids_len = len(tok.encode(long_prompt))
    assert prompt_ids_len > 2 * cfg.block_size

    fwd_calls = {"n": 0}
    prep_calls = {"n": 0}
    orig_forward = model.forward
    orig_prep = model.prepare_decode

    def counting_forward(*args, **kwargs):
        fwd_calls["n"] += 1
        return orig_forward(*args, **kwargs)

    def counting_prep(*args, **kwargs):
        prep_calls["n"] += 1
        return orig_prep(*args, **kwargs)

    model.forward = counting_forward
    model.prepare_decode = counting_prep

    _ = generate(model, long_prompt, max_new_tokens=1, top_k=10, tokenizer=tok)

    # Total chunks covering the prompt = ceil(prompt_len / block_size).
    # The final chunk goes through prepare_decode (always 1 call); the
    # earlier ones go through forward() (one per block_size).
    expected_total = (prompt_ids_len + cfg.block_size - 1) // cfg.block_size
    actual_total = fwd_calls["n"] + prep_calls["n"]
    assert prep_calls["n"] == 1, (
        f"prepare_decode should fire once for the warm-up; got {prep_calls['n']}"
    )
    assert actual_total >= expected_total, (
        f"only {actual_total} prompt-chunk passes (forward={fwd_calls['n']} + "
        f"prepare_decode={prep_calls['n']}); expected >= {expected_total} "
        f"for prompt_len={prompt_ids_len}, block_size={cfg.block_size}"
    )


# ---------------------------------------------------------------------------
# Sanity: generate produces a string
# ---------------------------------------------------------------------------

def test_generate_returns_string():
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    out = generate(model, "hello", max_new_tokens=5, top_k=10, tokenizer=tok)
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Perplexity contract + G161 mode restore
# ---------------------------------------------------------------------------

def test_perplexity_returns_positive_float():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    # Synthetic loader: one batch.
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[:, 0] = True
    loader = [(idx, db)]
    ppl = perplexity(model, loader, torch.device("cpu"))
    assert isinstance(ppl, float)
    assert ppl > 0


def test_perplexity_restores_training_mode():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    db = torch.zeros(2, 8, dtype=torch.bool)
    db[:, 0] = True
    perplexity(model, [(idx, db)], torch.device("cpu"))
    assert model.training is True
