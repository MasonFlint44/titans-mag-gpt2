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
    """For prompt_len > block_size, the warm-up loop must run multiple times
    (one per chunk). Truncating to [-block_size:] would drop everything before
    the tail. Verify the call pattern via a forward-call counter."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()

    # Build a prompt that tokenizes to >2 * block_size = >64 tokens.
    long_prompt = "the quick brown fox jumps over the lazy dog " * 50  # ~450 chars
    prompt_ids_len = len(tok.encode(long_prompt))
    assert prompt_ids_len > 2 * cfg.block_size  # confirm test setup

    call_count = {"n": 0}
    orig_forward = model.forward

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return orig_forward(*args, **kwargs)

    model.forward = counting

    # max_new_tokens=1 so the bulk of model() calls come from warm-up.
    _ = generate(model, long_prompt, max_new_tokens=1, top_k=10, tokenizer=tok)
    # Warm-up alone needs ceil(prompt_ids_len / block_size) calls.
    expected_warmup_calls = (prompt_ids_len + cfg.block_size - 1) // cfg.block_size
    # +1 for the post-sample model() call inside the generation loop.
    assert call_count["n"] >= expected_warmup_calls, (
        f"only {call_count['n']} forward calls; expected >= "
        f"{expected_warmup_calls} chunked-warm-up calls for "
        f"prompt_len={prompt_ids_len}, block_size={cfg.block_size}"
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
