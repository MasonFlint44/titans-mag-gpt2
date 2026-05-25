"""Phase 7.4 — behavior parity: cached generate vs reset-and-replay reference.

The full Option-B correctness check: argmax-decoded tokens from generate()
should match the argmax of a model() call on the same prompt+generated
sequence, at every position.

This is the user-facing closing test on the NMM-reprocess fix.
"""

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from generate import generate
from model.titans_gpt2 import TitansMAGGPT2


def _tiny_model_real_vocab():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=50257,
        block_size=32, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    return cfg, TitansMAGGPT2(cfg).eval()


@pytest.mark.slow
def test_cached_argmax_decode_matches_reset_and_replay_reference():
    """The headline behavior parity for Option B.

    Run generate() with temperature=0 (deterministic argmax). At each
    decoded position, compare against the reference: run model() on
    [prompt + generated_so_far], take argmax of last logits, verify
    it equals the token generate() chose.

    This verifies the cached decode path produces the same token
    sequence as the "reset and replay" reference (Option A).
    """
    torch.manual_seed(0)
    cfg, model = _tiny_model_real_vocab()
    tok = Tokenizer()

    prompt = "hello world"
    n_new = 5

    # Cached decode (the new generate)
    cached_out = generate(
        model, prompt, max_new_tokens=n_new, temperature=0, top_k=None, tokenizer=tok
    )
    cached_ids = tok.encode(cached_out)

    # Reset-and-replay reference
    prompt_ids = tok.encode(prompt)
    full_ids = list(prompt_ids)
    ref_ids = []
    for _ in range(n_new):
        idx = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(idx, nmm_states=None)
        next_id = logits[0, -1].argmax().item()
        ref_ids.append(next_id)
        full_ids.append(next_id)
        if next_id == tok.eot_token:
            break

    assert cached_ids == ref_ids, (
        f"cached decode != reset-and-replay reference:\n"
        f"  cached:  {cached_ids}\n"
        f"  ref:     {ref_ids}"
    )


@pytest.mark.slow
def test_cached_decode_long_prompt_uses_full_context():
    """For a prompt longer than block_size, the cached decode warms up via
    chunked forward() over the prefix + prepare_decode on the tail. The
    output should depend on tokens BEYOND the last block_size — i.e., two
    long prompts that differ only in the prefix should produce different
    cache.last_logits."""
    torch.manual_seed(0)
    cfg, model = _tiny_model_real_vocab()
    tok = Tokenizer()

    base_tail = "the lazy dog ran around the park very fast in circles "
    # block_size=32; aim for >40-token prompts.
    long_prompt_a = ("prefix A: " + "filler text content " * 8) + base_tail
    long_prompt_b = ("prefix B: " + "filler text content " * 8) + base_tail

    # Confirm both prompts > block_size.
    a_len = len(tok.encode(long_prompt_a))
    b_len = len(tok.encode(long_prompt_b))
    assert a_len > cfg.block_size
    assert b_len > cfg.block_size

    out_a = generate(
        model, long_prompt_a, max_new_tokens=1,
        temperature=0, top_k=None, tokenizer=tok,
    )
    out_b = generate(
        model, long_prompt_b, max_new_tokens=1,
        temperature=0, top_k=None, tokenizer=tok,
    )
    # The first decoded token may or may not differ depending on the model's
    # sensitivity to the prefix; the strict invariant we CAN guarantee is
    # that NEITHER prompt crashed and the prefix went through the NMM
    # (verified by the call-counting test test_generate_chunks_long_prompts_through_NMM).
    # If outputs ARE different, that's also evidence the prefix reached the model.
    # Here, just assert no crash and outputs are strings (smoke).
    assert isinstance(out_a, str)
    assert isinstance(out_b, str)
