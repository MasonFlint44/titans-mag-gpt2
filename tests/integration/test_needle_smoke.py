"""Phase 5.3 — needle_in_haystack runs without error (untrained model;
real recall accuracy is a behaviour test that needs a trained checkpoint)."""

import pytest
import torch

from config import TitansConfig
from data.tokenizer import Tokenizer
from eval import needle_in_haystack
from model.titans_gpt2 import TitansMAGGPT2


def test_needle_in_haystack_runs_to_completion():
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=50257,
        block_size=128, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )
    model = TitansMAGGPT2(cfg)
    tok = Tokenizer()
    # Untrained model — recall is essentially random; we only verify the
    # harness runs and returns a bool.
    result = needle_in_haystack(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        haystack="the quick brown fox jumps over the lazy dog. " * 4,
        block_size=64,
    )
    assert isinstance(result, bool)
