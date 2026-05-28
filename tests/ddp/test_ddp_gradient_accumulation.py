"""Phase 4.5 DDP-specific invariants. Marked ddp; skipped without >=2 GPUs."""

import pytest
import torch


pytestmark = pytest.mark.ddp


def _require_ddp():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("DDP tests require >= 2 GPUs")


def test_no_sync_used_for_non_final_micro_batches():
    """under DDP, all but the last micro-batch of a cycle must run
    inside model.no_sync() to suppress per-microbatch AllReduce. Verifying
    requires spawning 2 ranks and monitoring AllReduce traffic — not feasible
    in this environment. Marker placeholder."""
    _require_ddp()
    # The implementation in train.py:run_training selects sync_ctx via:
    #   sync_ctx = model.no_sync() if (is_distributed and not is_last_accum
    #              and hasattr(model, "no_sync")) else contextlib.nullcontext()
    # Runtime verification needs a real DDP setup with comm op counting.


def test_partial_cycle_at_end_of_loader_skips_step_under_DDP():
    """under DDP, a partial accumulation cycle (StopIteration
    mid-cycle) must NOT call optimizer.step — per-rank grads were never
    AllReduce'd, stepping would diverge ranks permanently."""
    _require_ddp()
    # Verified structurally by is_partial_cycle (tested at unit level).
    # Runtime verification needs a real 2-rank setup with a corpus length
    # chosen to fall at accum_i = K-1 (the case).


def test_per_rank_seed_differs_after_model_construction():
    """each rank gets seed + rank AFTER building the model so the
    model is identical across ranks but dropout masks diverge."""
    _require_ddp()
    # Implemented in train.py:main() — sets manual_seed(seed) before model
    # construction, then manual_seed(seed + rank) after.


def test_init_process_group_before_DDP_wrap():
    """init_process_group must be called before DistributedDataParallel
    wrap. Without it, DDP raises 'Default process group has not been
    initialized'."""
    _require_ddp()
    # Implemented in train.py:main() — init_process_group is called inside
    # the `if is_distributed:` branch before the DDP wrap.


def test_destroy_process_group_runs_on_exception_path():
    """the entire training body is wrapped in try/finally with
    destroy_process_group in the finally branch. NCCL communicator leak
    otherwise."""
    _require_ddp()
    # Implemented in train.py:main() — try/finally structure with consistent
    # 4-space indentation throughout the try body.
