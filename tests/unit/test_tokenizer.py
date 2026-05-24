"""Phase 3.1 — Tokenizer."""

import torch

from data.tokenizer import Tokenizer


def test_eot_token_is_50256():
    tok = Tokenizer()
    assert tok.eot_token == 50256


def test_roundtrip_ascii():
    tok = Tokenizer()
    for s in ["hello world", "the quick brown fox", "12345 -=+", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_encode_returns_list_of_ints():
    tok = Tokenizer()
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_corpus_appends_eot_between_and_after_docs():
    tok = Tokenizer()
    out = tok.encode_corpus(["a", "b"])
    assert out.dtype == torch.long
    assert out.ndim == 1
    # Last token must be EOT.
    assert out[-1].item() == tok.eot_token
    # Two docs -> exactly two EOTs (one after each).
    assert (out == tok.eot_token).sum().item() == 2


def test_encode_corpus_literal_endoftext_is_BPE_not_special_id():
    """G152 — literal <|endoftext|> in source text must encode as BPE characters,
    NOT as the special EOT id 50256. Otherwise document boundaries get fabricated
    inside user content."""
    tok = Tokenizer()
    out = tok.encode_corpus(["a<|endoftext|>b"])
    # Only ONE EOT in the output: the one appended at end of doc.
    # The literal <|endoftext|> in the middle becomes ordinary BPE tokens.
    assert (out == tok.eot_token).sum().item() == 1
    assert out[-1].item() == tok.eot_token


def test_encode_corpus_empty_doc_list_returns_empty_tensor():
    tok = Tokenizer()
    out = tok.encode_corpus([])
    assert out.shape == (0,)


def test_encode_corpus_single_empty_doc_is_just_eot():
    tok = Tokenizer()
    out = tok.encode_corpus([""])
    assert out.tolist() == [tok.eot_token]
