"""Phase 3.3 — ParallelStreamLoader."""

import pytest
import torch

from data.dataloader import ParallelStreamLoader


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_yields_correct_shape_and_dtype():
    stream = torch.arange(400, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    for idx, db in loader:
        assert idx.shape == (4, 10)
        assert db.shape == (4, 10)
        assert idx.dtype == torch.long
        assert db.dtype == torch.bool


def test_num_chunks_matches_len():
    # 4 streams * 10 chunks * 10 tokens = 400 tokens
    stream = torch.arange(400, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    assert len(loader) == 10
    chunks = list(loader)
    assert len(chunks) == 10


def test_drops_trailing_remainder():
    # 405 tokens; B=4, T=10 -> can fit 400 -> 10 chunks; 5 tokens dropped.
    stream = torch.arange(405, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    assert len(loader) == 10


# ---------------------------------------------------------------------------
# Position-i continuity — the defining property
# ---------------------------------------------------------------------------

def test_position_i_streams_are_contiguous_across_batches():
    """Position i at batch N's last token must equal position i at batch N+1's
    first token's predecessor. With token_stream = arange(N), each sub-stream
    is a contiguous slice — verify directly."""
    stream = torch.arange(400, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    chunks = list(loader)
    for b in range(4):
        # Concatenate position b across all chunks; should be a contiguous slice.
        concat = torch.cat([c[0][b] for c in chunks])
        # The original stream is [0..399]; the b-th sub-stream after reshape
        # is rows of view(4, 100): row 0 = [0..99], row 1 = [100..199], etc.
        expected = torch.arange(b * 100, (b + 1) * 100)
        assert torch.equal(concat, expected), (
            f"sub-stream {b} broken contiguity: got first few = "
            f"{concat[:5].tolist()}, expected {expected[:5].tolist()}"
        )


def test_each_substream_starts_where_previous_chunk_ended():
    """End-of-batch-N for sub-stream b equals start-of-batch-N+1 for sub-stream b
    minus one token gap (since they're consecutive tokens in the stream)."""
    stream = torch.arange(400, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    chunks = list(loader)
    for k in range(len(chunks) - 1):
        last = chunks[k][0][:, -1]   # [B]
        first_next = chunks[k + 1][0][:, 0]  # [B]
        # Consecutive tokens differ by exactly 1 in arange.
        assert torch.equal(first_next, last + 1), (
            f"batch {k}->{k+1}: position-i continuity broken. "
            f"last={last.tolist()}, first_next={first_next.tolist()}"
        )


# ---------------------------------------------------------------------------
# Doc boundaries
# ---------------------------------------------------------------------------

def test_boundary_marks_position_zero_of_every_substream():
    stream = torch.arange(400, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)
    first_batch = next(iter(loader))
    db = first_batch[1]
    assert torch.all(db[:, 0])


def test_boundary_marks_token_AFTER_eot():
    """boundaries[t] = True iff streams[t-1] == eot. Place EOT explicitly."""
    EOT = 999
    # Stream: [1, EOT, 2, 3, EOT, 4, ...] — one EOT in each row's content.
    raw = torch.tensor([1, EOT, 2, 3, EOT, 4, 5, EOT], dtype=torch.long)
    loader = ParallelStreamLoader(raw, batch_size=1, chunk_size=8, eot_id=EOT)
    idx, db = next(iter(loader))
    # streams[0, t-1] == EOT at t in {2, 5} -> boundaries True there.
    # Position 0 is also True (segment start).
    expected = torch.tensor([[True, False, True, False, False, True, False, False]])
    assert torch.equal(db, expected)


def test_no_boundary_when_no_eot_in_segment():
    stream = torch.arange(100, dtype=torch.long)
    loader = ParallelStreamLoader(stream, batch_size=2, chunk_size=10, eot_id=50256)
    chunks = list(loader)
    # Position 0 of batch 0 is True for both sub-streams (segment start);
    # nothing else should be True since no EOT in [0..99].
    db0 = chunks[0][1]
    assert torch.all(db0[:, 0])
    assert not torch.any(db0[:, 1:])
    for c in chunks[1:]:
        assert not torch.any(c[1])  # no boundaries anywhere else


# ---------------------------------------------------------------------------
# DDP sharding
# ---------------------------------------------------------------------------

def test_ddp_rank_partition_gives_disjoint_segments():
    """rank=0 and rank=1 with world_size=2 must see different tokens, with
    rank 1's first token being rank 0's last + 1 (contiguous partition)."""
    stream = torch.arange(800, dtype=torch.long)
    loader_r0 = ParallelStreamLoader(
        stream, batch_size=4, chunk_size=10, eot_id=50256, rank=0, world_size=2
    )
    loader_r1 = ParallelStreamLoader(
        stream, batch_size=4, chunk_size=10, eot_id=50256, rank=1, world_size=2
    )
    assert len(loader_r0) == len(loader_r1)
    chunks_r0 = list(loader_r0)
    chunks_r1 = list(loader_r1)
    # Both ranks must see disjoint slices.
    tokens_r0 = set(int(t) for c in chunks_r0 for t in c[0].flatten().tolist())
    tokens_r1 = set(int(t) for c in chunks_r1 for t in c[0].flatten().tolist())
    assert tokens_r0.isdisjoint(tokens_r1)
    # Together they cover ~the whole stream (modulo the rounding-down trim).
    assert max(tokens_r0) < min(tokens_r1)  # r0 segment precedes r1's


def test_ddp_each_rank_has_same_num_chunks():
    """requirement for synchronous DDP: every rank's loader yields the
    same number of batches (otherwise hang at the all-reduce barrier)."""
    stream = torch.arange(1000, dtype=torch.long)
    counts = []
    for r in range(4):
        loader = ParallelStreamLoader(
            stream, batch_size=2, chunk_size=10, eot_id=50256, rank=r, world_size=4
        )
        counts.append(len(loader))
    assert len(set(counts)) == 1, f"per-rank batch counts diverge: {counts}"


def test_ddp_per_rank_segment_start_marked_as_boundary():
    """Every rank's batch 0 position 0 must be True — no cross-rank NMM continuity."""
    stream = torch.arange(800, dtype=torch.long)
    for r in range(2):
        loader = ParallelStreamLoader(
            stream, batch_size=4, chunk_size=10, eot_id=50256, rank=r, world_size=2
        )
        first = next(iter(loader))
        assert torch.all(first[1][:, 0])


def test_too_small_corpus_raises():
    """Tiny corpus that can't fill even one chunk -> ValueError, not silent zero-chunk loader."""
    stream = torch.arange(5, dtype=torch.long)
    with pytest.raises(ValueError, match="too small"):
        ParallelStreamLoader(stream, batch_size=4, chunk_size=10, eot_id=50256)


# ---------------------------------------------------------------------------
# Integration: works with Tokenizer.encode_corpus output
# ---------------------------------------------------------------------------

def test_integration_with_tokenizer():
    from data.tokenizer import Tokenizer
    tok = Tokenizer()
    docs = ["the quick brown fox", "jumps over the lazy dog", "another short doc"]
    stream = tok.encode_corpus(docs)
    loader = ParallelStreamLoader(
        stream, batch_size=1, chunk_size=4, eot_id=tok.eot_token
    )
    chunks = list(loader)
    assert len(chunks) > 0
    # Position 0 of first chunk is start-of-segment -> True boundary.
    assert chunks[0][1][0, 0].item() is True
