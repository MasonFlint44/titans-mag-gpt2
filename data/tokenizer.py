"""tiktoken GPT-2 adapter."""

import io
import warnings

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
