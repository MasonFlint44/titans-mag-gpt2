"""tiktoken GPT-2 adapter."""

import io
import warnings
from pathlib import Path

import numpy as np
import tiktoken
import torch


class Tokenizer:
    """Thin wrapper around tiktoken's `gpt2` encoding.

    encode_corpus joins documents with the EOT *id* (not the literal text
    `<|endoftext|>`) because re-encoding the literal as BPE produces ordinary
    character tokens, not the special EOT id 50256 — silently leaving
    ParallelStreamLoader's `(streams == eot_id)` mask all False and
    cross-document NMM state leaking corpus-wide.
    """

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # Portable across tiktoken versions: encode_single_token is stable;
        # the `eot_token` attribute is only present in newer (~0.7+) releases.
        self.eot_token = self.enc.encode_single_token("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        # disallowed_special=() allows arbitrary text (including the literal
        # `<|endoftext|>`) to encode as ordinary BPE tokens without an error.
        # Special-token handling lives in encode_corpus, not here.
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids) -> str:
        return self.enc.decode(ids)

    def encode_corpus(self, documents) -> torch.Tensor:
        """Tokenize each document and join with the EOT id. Returns a 1-D LongTensor.

        WARNING: iterating a file handle yields LINES, not documents. The
        natural-looking `with open(path) as f: encode_corpus(f)` silently
        treats every newline as a document boundary, which fires reset_state
        in the NMM at every line and disables long-range memory with no
        error. To opt into that behavior intentionally, wrap the file in
        a list comprehension (`encode_corpus([line for line in f])`) which
        makes the intent explicit. Otherwise use `encode_corpus([f.read()])`
        for whole-file-as-one-document, or `re.split(...)` for paragraph
        documents.
        """
        if isinstance(documents, io.IOBase):
            warnings.warn(
                "encode_corpus received a file handle directly. Iterating a "
                "file yields LINES, so every newline becomes a document "
                "boundary — the NMM's long-range memory is silently disabled "
                "by per-line resets. Use encode_corpus([f.read()]) for "
                "whole-file-as-one-document, or split the text into logical "
                "documents first. (See in docs/archive/GAP_HISTORY.md.)",
                UserWarning,
                stacklevel=2,
            )
        ids: list[int] = []
        for doc in documents:
            ids.extend(self.enc.encode(doc, disallowed_special=()))
            ids.append(self.eot_token)
        return torch.tensor(ids, dtype=torch.long)


def load_token_stream(path: str | Path) -> torch.Tensor:
    """Load a token stream from disk, dispatching on file extension.

    Two formats supported:
      - `*.bin`: raw little-endian uint16 binary written by
        `scripts.tokenize_fineweb_edu` (or any nanoGPT-compatible
        producer). Read via numpy and cast to LongTensor.
      - everything else (`*.txt`, no extension, etc.): text corpus.
        Split on literal `<|endoftext|>` markers (via
        `scripts.prepare_squad_corpus.read_eot_separated_documents`)
        and tokenize through `Tokenizer.encode_corpus`.

    The binary path skips the per-startup re-tokenization cost — at
    ~1.5B tokens, tokenization is a multi-hour job; pre-tokenizing once
    to a binary blob saves that on every resume / retry.

    Returns a 1-D `torch.long` tensor of token ids, ready to hand to
    `ParallelStreamLoader`.
    """
    path = Path(path)
    if path.suffix == ".bin":
        size_bytes = path.stat().st_size
        if size_bytes % 2 != 0:
            raise ValueError(
                f"{path} has odd byte size ({size_bytes}); not a clean "
                f"uint16 stream. Re-run scripts.tokenize_fineweb_edu."
            )
        # np.memmap + .astype(int64) does NOT load lazily on the .astype
        # call — it materializes the converted array. So memory peak is
        # 1× uint16 (file size) + 1× int64 (4× file size) momentarily,
        # then the uint16 backing memmap can be dropped. For 1.5B tokens
        # that's ~3GB + ~12GB transient. Workable on a 32GB+ box.
        memmapped = np.memmap(path, dtype=np.uint16, mode="r")
        # Convert to int64 (the dtype nn.Embedding expects). torch.from_numpy
        # on a uint16 array isn't supported; go through int64 via numpy.
        as_int64 = np.asarray(memmapped, dtype=np.int64)
        return torch.from_numpy(as_int64)
    # Text path — the existing recipe.
    from scripts.prepare_squad_corpus import read_eot_separated_documents
    tok = Tokenizer()
    return tok.encode_corpus(read_eot_separated_documents(path))
