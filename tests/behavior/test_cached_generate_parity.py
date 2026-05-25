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
    output MUST depend on tokens beyond the last block_size — i.e., two
    long prompts that differ only in the prefix should produce different
    cache["last_logits"].

    Asserting at the LOGITS level (not the argmax token) is the strong
    form of the test: the prefix-only delta would otherwise have to vanish
    through every block's NMM update to leave logits identical. At
    untrained random init that's vanishingly unlikely if the prefix
    actually reached the NMM. Argmax token equality is not a useful
    invariant — at untrained init the same token can win both vocab
    distributions even when they differ by ~0.01.
    """
    torch.manual_seed(0)
    cfg, model = _tiny_model_real_vocab()
    tok = Tokenizer()

    base_tail = "the lazy dog ran around the park very fast in circles "
    # block_size=32; aim for >40-token prompts whose LAST block_size tokens
    # are identical so any logit difference comes from the prefix.
    prefix_a = "prefix A is short. "
    prefix_b = "prefix B differs substantially from prefix A in content. "
    long_prompt_a = (prefix_a + "filler text content " * 8) + base_tail
    long_prompt_b = (prefix_b + "filler text content " * 8) + base_tail

    # Confirm both prompts > block_size.
    a_ids = tok.encode(long_prompt_a)
    b_ids = tok.encode(long_prompt_b)
    assert len(a_ids) > cfg.block_size
    assert len(b_ids) > cfg.block_size

    # Run the same chunked-warm-up + prepare_decode path that generate uses,
    # and compare last_logits directly.
    def _last_logits_for(ids_list):
        torch.manual_seed(0)  # model is deterministic; only random source
        ids = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
        prompt_len = ids.size(1)
        tail_start = prompt_len - cfg.block_size
        nmm_states = None
        with torch.no_grad():
            for start in range(0, tail_start, cfg.block_size):
                end = min(start + cfg.block_size, tail_start)
                _, nmm_states = model(ids[:, start:end], nmm_states, None)
            tail = ids[:, tail_start:]
            cache = model.prepare_decode(tail, initial_nmm_states=nmm_states)
        return cache["last_logits"].squeeze()  # [V]

    logits_a = _last_logits_for(a_ids)
    logits_b = _last_logits_for(b_ids)

    diff = (logits_a - logits_b).abs().max().item()
    assert diff > 1e-3, (
        f"prefix difference produced ≈identical last_logits (max diff = "
        f"{diff:.3e}); the prefix did not reach the NMM through the "
        f"chunked-warm-up path"
    )
