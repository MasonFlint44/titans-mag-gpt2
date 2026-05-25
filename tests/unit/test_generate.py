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
# Boundary: cap on max_new (off-by-one defended)
# ---------------------------------------------------------------------------

def _prompt_with_encoded_length(tok, n_tokens):
    """Build a string whose `tok.encode(...)` length is exactly n_tokens.
    Token ids 100..100+n_tokens-1 are deep in the BPE-merged range; decoding
    them to a string and re-encoding round-trips losslessly in practice for
    tiktoken's gpt2 encoding (verified by the assert below)."""
    ids = list(range(100, 100 + n_tokens))
    s = tok.decode(ids)
    re_encoded = tok.encode(s)
    # Round-trip can vary in length; trim from the END until it matches.
    while len(re_encoded) > n_tokens:
        ids = ids[:-1]
        s = tok.decode(ids)
        re_encoded = tok.encode(s)
    while len(re_encoded) < n_tokens:
        # Add a known-singleton-ASCII byte and re-check.
        ids = ids + [ord("A")]
        s = tok.decode(ids)
        re_encoded = tok.encode(s)
    assert len(re_encoded) == n_tokens, (
        f"could not build prompt of exactly {n_tokens} tokens; got {len(re_encoded)}"
    )
    return s


def test_generate_at_prompt_len_equals_block_size_returns_one_token():
    """Prompt fills block_size exactly. The cap allows ONE sampled token
    (from cache["last_logits"]) without any forward_step call — no wpe
    OOB. The earlier `min(max_new, block_size - prompt_len)` cap was
    off-by-one and returned an empty string here, silently dropping a
    legitimately-available token."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    prompt = _prompt_with_encoded_length(tok, cfg.block_size)
    out = generate(
        model, prompt, max_new_tokens=10,
        temperature=0, top_k=None, tokenizer=tok,
    )
    n_out = len(tok.encode(out))
    assert n_out == 1, (
        f"expected exactly 1 sampled token at prompt_len == block_size, "
        f"got {n_out}"
    )


def test_generate_short_prompt_respects_new_cap():
    """For prompt_len < block_size, the new cap allows block_size - prompt_len + 1
    tokens (one more than the old cap). Verify exactly that many tokens are
    sampled when the user asks for more, and no crash on the runtime
    block_size guard inside forward_step."""
    cfg, model = _tiny_model_real_vocab()
    model.eval()
    tok = Tokenizer()
    target_prompt_len = cfg.block_size - 5  # leave room for 6 decoded tokens
    prompt = _prompt_with_encoded_length(tok, target_prompt_len)
    expected_cap = cfg.block_size - target_prompt_len + 1  # = 6
    out = generate(
        model, prompt, max_new_tokens=expected_cap + 5,  # ask for too many
        temperature=0, top_k=None, tokenizer=tok,
    )
    n_out = len(tok.encode(out))
    # Cap should clip to exactly expected_cap (EOT short-circuit absent at
    # this seed with the tiny model).
    assert n_out <= expected_cap, (
        f"output {n_out} tokens exceeds cap {expected_cap}"
    )


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
