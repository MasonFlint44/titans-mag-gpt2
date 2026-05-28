"""ParallelStreamLoader: TBPTT-aware batching with cross-batch sub-stream continuity."""

import torch
import torch.distributed as dist


class ParallelStreamLoader:
    """Yields (idx_BT, doc_boundaries_BT) such that position-i is contiguous
    across consecutive batches in the same document stream.

    The naive `DataLoader(shuffle=False, batch_size=B)` collates chunks
    [0..B-1] into batch 0, [B..2B-1] into batch 1, etc. The carried
    `nmm_states[i]` then jumps over B-1 chunks between batches —
    cross-document state corruption with no error.

    Correct: reshape the token stream into B parallel sub-streams of equal
    length; each batch is one chunk from each sub-stream. Position-i at
    batch N+1 directly continues position-i at batch N.

    DDP: each rank reads a CONTIGUOUS segment of the corpus so
    aggregate batch size scales linearly with world_size. Per-rank
    boundaries[:, 0] is True (segment start has no cross-rank predecessor).
    """

    def __init__(
        self,
        token_stream: torch.Tensor,
        batch_size: int,
        chunk_size: int,
        eot_id: int,
        rank: int = None,
        world_size: int = None,
    ):
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Per-rank segment, rounded down to a multiple of (B * chunk_size) so
        # every rank has the same number of batches — required for synchronous
        # DDP (mismatched batch counts hang at the all-reduce barrier).
        N_total = len(token_stream) // world_size
        N_per_rank = (N_total // (batch_size * chunk_size)) * batch_size * chunk_size
        if N_per_rank == 0:
            raise ValueError(
                f"Corpus too small: {len(token_stream)} tokens for "
                f"world_size={world_size} ranks × batch_size={batch_size} × "
                f"chunk_size={chunk_size}. Need at least "
                f"{world_size * batch_size * chunk_size} tokens."
            )
        seg_start = rank * N_per_rank
        seg_end = seg_start + N_per_rank

        self.streams = token_stream[seg_start:seg_end].view(batch_size, -1)
        self.B, self.S = self.streams.shape
        self.chunk_size = chunk_size
        self.num_chunks = self.S // chunk_size
        self.eot_id = eot_id

        # boundaries[:, t] is True iff streams[:, t-1] was EOT
        # (i.e., t is the first token of a new document). Position 0 of every
        # rank's segment is flagged True — no cross-rank or pre-corpus
        # NMM-state continuity to preserve.
        eot_mask = self.streams == eot_id
        boundaries = torch.zeros_like(self.streams, dtype=torch.bool)
        boundaries[:, 1:] = eot_mask[:, :-1]
        boundaries[:, 0] = True
        self.boundaries = boundaries

    def __iter__(self):
        for c in range(self.num_chunks):
            s = slice(c * self.chunk_size, (c + 1) * self.chunk_size)
            yield self.streams[:, s].contiguous(), self.boundaries[:, s].contiguous()

    def __len__(self):
        return self.num_chunks
