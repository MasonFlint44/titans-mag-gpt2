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
    """literal <|endoftext|> in source text must encode as BPE characters,
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


# ---------------------------------------------------------------------------
# warn when a file handle is passed directly
# ---------------------------------------------------------------------------

def test_encode_corpus_warns_on_file_handle():
    """iterating a file yields lines, not documents. Without the warning,
    `with open(path) as f: tok.encode_corpus(f)` silently inserts an EOT
    after every line, firing reset_state per line and killing the NMM's
    long-range memory with no signal."""
    import io
    import pytest

    tok = Tokenizer()
    fake_file = io.StringIO("line one\nline two\n")
    with pytest.warns(UserWarning, match="file handle"):
        out = tok.encode_corpus(fake_file)
    # Behavior still works (we warn but proceed); two lines -> two EOTs.
    assert (out == tok.eot_token).sum().item() == 2


def test_encode_corpus_does_not_warn_on_list_of_documents():
    import warnings as _w

    tok = Tokenizer()
    with _w.catch_warnings():
        _w.simplefilter("error")  # any UserWarning -> test failure
        tok.encode_corpus(["doc one", "doc two"])


def test_encode_corpus_does_not_warn_on_generator_of_strings():
    """Generators are the recommended pattern (HF streaming datasets);
    must not trigger the file-handle warning."""
    import warnings as _w

    tok = Tokenizer()
    docs = (f"doc {i}" for i in range(3))
    with _w.catch_warnings():
        _w.simplefilter("error")
        out = tok.encode_corpus(docs)
    assert (out == tok.eot_token).sum().item() == 3


# ---------------------------------------------------------------------------
# T10 — whole-file-as-one-document positive case (TEST_PLAN §5 test_tokenizer.py)
# ---------------------------------------------------------------------------

def test_encode_corpus_whole_file_pattern_produces_one_document(tmp_path):
    """The warning recommends `encode_corpus([f.read()])` as the
    correct whole-file-as-one-document pattern. The negative test
    `test_encode_corpus_warns_on_file_handle` covers the WRONG case;
    this is the POSITIVE invariant: when wrapped in a list around
    f.read(), the file's full content is treated as exactly ONE document
    (a single trailing EOT, no per-line boundaries)."""
    path = tmp_path / "corpus.txt"
    path.write_text(
        "first line of the document\n"
        "second line\n"
        "third\n"
    )
    tok = Tokenizer()
    with open(path, "r", encoding="utf-8") as f:
        ids = tok.encode_corpus([f.read()])

    # Exactly one EOT (the one appended after the single document).
    assert (ids == tok.eot_token).sum().item() == 1, (
        f"whole-file pattern produced {(ids == tok.eot_token).sum().item()} "
        f"EOTs; expected 1 (one document)."
    )
    # The EOT is at the END (not embedded mid-stream).
    assert ids[-1].item() == tok.eot_token

    # Content survives the round-trip: decode without the trailing EOT
    # should match the source file's text (modulo BPE normalization).
    decoded = tok.decode(ids[:-1].tolist())
    assert "first line" in decoded
    assert "second line" in decoded
    assert "third" in decoded
