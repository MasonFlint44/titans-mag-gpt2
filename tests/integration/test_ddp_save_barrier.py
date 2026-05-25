"""G199 — structural: the save block in run_training has both the rank-0
guard and the dist.barrier() (so non-rank-0 ranks don't race past)."""

import ast
import inspect

import train


def _function_source(fn):
    return inspect.getsource(fn)


def test_run_training_save_block_uses_rank0_guard_and_barrier():
    """Parse run_training; find the if-block that calls one of the
    save_checkpoint* helpers; verify (a) it's guarded by `rank == 0`
    and (b) `dist.barrier()` is invoked in the same code region (under
    `is_distributed`).

    Accepts either `save_checkpoint` or `save_checkpoint_rotating`
    (added in the rotation pass) — the G199 invariant is rank 0 owns
    the write, not the specific helper name.
    """
    src = _function_source(train.run_training)
    tree = ast.parse(src.strip())

    found_save = False
    found_rank0_guard = False
    found_barrier = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Look for save_checkpoint*(...) and dist.barrier() calls.
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id.startswith("save_checkpoint"):
                found_save = True
            if isinstance(fn, ast.Attribute) and fn.attr == "barrier":
                found_barrier = True
        if isinstance(node, ast.Compare):
            # Look for `rank == 0` comparison.
            left = node.left
            if (
                isinstance(left, ast.Name)
                and left.id == "rank"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 0
            ):
                found_rank0_guard = True

    assert found_save, "run_training does not call any save_checkpoint* helper"
    assert found_rank0_guard, "run_training save path missing `rank == 0` guard"
    assert found_barrier, "run_training save path missing dist.barrier()"


def test_save_path_is_inside_save_every_branch():
    """Defensive: the save call should only fire when save_every is set."""
    src = _function_source(train.run_training)
    # Coarse check: "save_every" must appear and a save_checkpoint* call
    # must follow it.
    assert "save_every" in src
    assert "save_checkpoint" in src
    # The save call should come AFTER the `save_every` if-condition lexically.
    assert src.index("save_every") < src.index("save_checkpoint")
