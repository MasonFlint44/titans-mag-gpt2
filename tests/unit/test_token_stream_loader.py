"""load_token_stream binary + text dispatch tests.

The from-scratch TITANS recipe pre-tokenizes FineWeb-Edu to a uint16 binary
once (via `scripts.tokenize_fineweb_edu`) and then memmaps it on every
training restart. `load_token_stream` is the dispatch point: it picks
between the binary fast-path and the legacy text-tokenize path based on
file extension.

These tests pin the round-trip equivalence (a text corpus tokenized via
the legacy path equals the same content written as binary then loaded),
the dtype contract, and the error path on a malformed binary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data.tokenizer import Tokenizer, load_token_stream


def _make_text_corpus(tmp_path: Path) -> Path:
    """A two-document corpus with the literal `<|endoftext|>` separator
    that `read_eot_separated_documents` looks for."""
    text = "the cat sat\n<|endoftext|>\nthe dog ran"
    p = tmp_path / "tiny.txt"
    p.write_text(text)
    return p


def _make_binary_for_same_corpus(text_path: Path, bin_path: Path) -> None:
    """Tokenize the text corpus to a uint16 binary, mimicking what
    `scripts.tokenize_fineweb_edu` does for FineWeb-Edu (split into docs
    + tokenize + insert EOT between)."""
    from scripts.prepare_squad_corpus import read_eot_separated_documents
    docs = read_eot_separated_documents(text_path)
    tok = Tokenizer()
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d))
        ids.append(tok.eot_token)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)


def test_text_path_returns_long_tensor(tmp_path):
    p = _make_text_corpus(tmp_path)
    out = load_token_stream(p)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.ndim == 1
    # The two documents + 2 EOT separators (one after each doc per
    # encode_corpus convention).
    tok = Tokenizer()
    assert (out == tok.eot_token).sum().item() == 2


def test_binary_path_returns_long_tensor(tmp_path):
    text_p = _make_text_corpus(tmp_path)
    bin_p = tmp_path / "tiny.bin"
    _make_binary_for_same_corpus(text_p, bin_p)
    out = load_token_stream(bin_p)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.ndim == 1


def test_binary_and_text_paths_produce_identical_streams(tmp_path):
    """Round-trip: tokenize the same content via text path AND via the
    binary path; the resulting LongTensors must be bit-identical.
    Defends against off-by-one EOT bugs in either path."""
    text_p = _make_text_corpus(tmp_path)
    bin_p = tmp_path / "tiny.bin"
    _make_binary_for_same_corpus(text_p, bin_p)

    text_out = load_token_stream(text_p)
    bin_out = load_token_stream(bin_p)
    assert torch.equal(text_out, bin_out), (
        f"binary load drifted from text load — "
        f"text[:10]={text_out[:10].tolist()}, "
        f"bin[:10]={bin_out[:10].tolist()}"
    )


def test_binary_with_odd_byte_count_raises_clear_error(tmp_path):
    """uint16 = 2 bytes per token. A file with an odd byte count cannot
    be a clean stream and is almost certainly a write-truncation. We
    detect and raise at load time rather than silently returning a
    corrupted tensor."""
    bin_p = tmp_path / "broken.bin"
    bin_p.write_bytes(b"\x00\x01\x02")  # 3 bytes — odd
    with pytest.raises(ValueError, match="odd byte size"):
        load_token_stream(bin_p)


def test_binary_path_does_not_require_text_file_companion(tmp_path):
    """Once you have a .bin, you should never need the original text
    file again. Sanity-check that loading from `*.bin` works without any
    accompanying text corpus on disk."""
    bin_p = tmp_path / "lonely.bin"
    arr = np.array([1, 2, 3, 4, 50256, 7, 8, 50256], dtype=np.uint16)
    arr.tofile(bin_p)
    out = load_token_stream(bin_p)
    assert out.tolist() == [1, 2, 3, 4, 50256, 7, 8, 50256]


def test_binary_path_token_ids_match_uint16_values_exactly(tmp_path):
    """No off-by-one or endianness drift: bytes 0..1 of the file become
    `out[0]`, bytes 2..3 become `out[1]`, etc. Little-endian uint16."""
    bin_p = tmp_path / "lonely.bin"
    # Token id 0x1234 → bytes [0x34, 0x12] little-endian.
    bin_p.write_bytes(b"\x34\x12\x00\x80")  # ids: [0x1234, 0x8000]
    out = load_token_stream(bin_p)
    assert out.tolist() == [0x1234, 0x8000]


def test_binary_path_compatible_with_parallel_stream_loader(tmp_path):
    """End-to-end: a binary-loaded token stream feeds into the same
    ParallelStreamLoader that the text path uses, with the same yielded
    shapes. This is what the from-scratch training command will actually
    consume."""
    from data.dataloader import ParallelStreamLoader
    bin_p = tmp_path / "stream.bin"
    # 64 tokens, half are non-EOT, half are EOT — enough for
    # ParallelStreamLoader to yield several chunks.
    arr = np.array(
        [i % 50000 for i in range(64)], dtype=np.uint16,
    )
    arr.tofile(bin_p)
    stream = load_token_stream(bin_p)
    loader = ParallelStreamLoader(
        stream, batch_size=2, chunk_size=8, eot_id=50256,
    )
    batches = list(loader)
    assert len(batches) > 0
    for input_ids, doc_boundaries in batches:
        assert input_ids.shape == (2, 8)
        assert doc_boundaries.shape == (2, 8)
        assert input_ids.dtype == torch.long
