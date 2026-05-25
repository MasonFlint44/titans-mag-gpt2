"""G225 / G227 — structural checks on train.py:main() body.

G225: training loop must be wrapped in try/finally with destroy_process_group
in the finally clause.

G227: the try body must use CONSISTENT 4-space indentation step (not the
2-2-4 pattern that the minimal-edit G225 fix originally produced).
"""

import ast
import inspect
import re

import train


def _main_source() -> str:
    return inspect.getsource(train.main)


# ---------------------------------------------------------------------------
# G225 — try/finally with destroy_process_group
# ---------------------------------------------------------------------------

def test_main_wraps_training_in_try_finally():
    src = _main_source()
    tree = ast.parse(src.strip())

    try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert len(try_nodes) >= 1, "train.main() has no try/finally block"

    # At least one try must have a non-empty finalbody (the finally clause).
    has_finally = any(t.finalbody for t in try_nodes)
    assert has_finally, "train.main() has no finally clause (G225)"


def test_destroy_process_group_called_in_finally():
    src = _main_source()
    tree = ast.parse(src.strip())

    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            # Recursively walk the finally body for a destroy_process_group call.
            for child in ast.walk(ast.Module(body=node.finalbody, type_ignores=[])):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "destroy_process_group"
                ):
                    return
    raise AssertionError("destroy_process_group not found in try/finally (G225)")


# ---------------------------------------------------------------------------
# G227 — consistent 4-space indentation throughout the try body
# ---------------------------------------------------------------------------

def test_try_body_uses_consistent_4_space_indentation():
    """Walk the lines INSIDE the try body of train.main() and verify the
    minimum-indent step between nesting levels is exactly 4 spaces (not 2).

    The original G225 minimal edit produced 2-2-4 spacing which parses but
    is a copy-paste hazard and a PEP-8 violation.
    """
    src = _main_source()
    lines = src.splitlines()

    # Find the `try:` line and the matching `finally:` line at the same indent.
    try_indent = None
    try_line = None
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("try:"):
            try_indent = len(ln) - len(stripped)
            try_line = i
            break
    assert try_line is not None, "no `try:` line found in train.main()"

    # Find the finally at the same indent level.
    finally_line = None
    for j in range(try_line + 1, len(lines)):
        ln = lines[j]
        stripped = ln.lstrip()
        if not stripped:
            continue
        ind = len(ln) - len(stripped)
        if ind == try_indent and stripped.startswith("finally:"):
            finally_line = j
            break
    assert finally_line is not None, "no matching `finally:` found"

    # All non-blank non-comment lines inside the try body must have indents
    # that are multiples of 4 spaces relative to try_indent.
    body_lines = [
        (j, ln) for j, ln in enumerate(lines[try_line + 1:finally_line], start=try_line + 1)
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    for j, ln in body_lines:
        rel_indent = len(ln) - len(ln.lstrip()) - try_indent
        assert rel_indent >= 4, (
            f"line {j} indent {rel_indent} smaller than try-body baseline (4)"
        )
        assert rel_indent % 4 == 0, (
            f"line {j} relative indent {rel_indent} is not a multiple of 4: {ln!r}"
        )


# ---------------------------------------------------------------------------
# T7 — DDP setup/cleanup structural assertions (G201, G204 via AST)
# ---------------------------------------------------------------------------

def _find_call_lines(tree, callable_name):
    """Return line numbers of calls matching `callable_name` (last attr or
    bare function name)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == callable_name:
                out.append(node.lineno)
            elif isinstance(func, ast.Name) and func.id == callable_name:
                out.append(node.lineno)
    return sorted(out)


def test_init_process_group_called_before_ddp_wrap():
    """G201 — `init_process_group()` must run BEFORE the model is wrapped
    in DDP. Otherwise DDP construction has no communicator and fails.

    AST check: the earliest `init_process_group` line precedes the earliest
    `DDP(` call line.
    """
    src = _main_source()
    tree = ast.parse(src.strip())

    init_lines = _find_call_lines(tree, "init_process_group")
    assert init_lines, "train.main() does not call init_process_group"

    # DDP() is a Name call (imported alias `DDP`).
    ddp_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DDP"
    ]
    assert ddp_lines, "train.main() does not call DDP() to wrap the model"

    assert min(init_lines) < min(ddp_lines), (
        f"init_process_group line {min(init_lines)} comes AFTER DDP() line "
        f"{min(ddp_lines)} — DDP construction has no communicator (G201)."
    )


def test_model_to_device_called_before_ddp_wrap():
    """G201 — model.to(device) must run BEFORE DDP(model, ...). DDP needs
    the model on the right device to bind its communicator correctly.

    AST check: there's a `.to(...)` call on `model` before any `DDP(` call.
    """
    src = _main_source()
    tree = ast.parse(src.strip())

    # Find `model = TitansMAGGPT2(...).to(device)` or similar chained `.to`.
    to_call_lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
        ):
            to_call_lines.append(node.lineno)
    assert to_call_lines, "train.main() does not call `.to(device)` anywhere"

    ddp_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DDP"
    ]
    assert ddp_lines, "train.main() does not call DDP()"
    assert min(to_call_lines) < min(ddp_lines), (
        f".to() line {min(to_call_lines)} comes AFTER DDP() line "
        f"{min(ddp_lines)} (G201)."
    )


def test_ddp_wrap_before_optimizer_construction():
    """G201 — optimizer.params should come from the DDP-wrapped model so
    `model.no_sync()` and DDP's gradient hooks fire on the same param set
    the optimizer steps. Build order: model → .to() → DDP(model) →
    build_optimizer(model).

    AST check: earliest DDP(...) line precedes earliest build_optimizer(...) line.
    """
    src = _main_source()
    tree = ast.parse(src.strip())

    ddp_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DDP"
    ]
    build_opt_lines = _find_call_lines(tree, "build_optimizer")
    assert ddp_lines and build_opt_lines, (
        "expected both DDP() and build_optimizer() in train.main()"
    )
    assert min(ddp_lines) < min(build_opt_lines), (
        f"DDP() at line {min(ddp_lines)} runs AFTER build_optimizer() at "
        f"line {min(build_opt_lines)} — optimizer would hold pre-wrap "
        f"params (G201)."
    )


def test_per_rank_seed_set_after_model_construction():
    """G204 — per-rank seed (`args.seed + rank`) must be applied AFTER
    model construction. If applied before, all ranks build the model with
    the same seed but ALSO get the same dropout masks during training,
    silently shrinking the effective batch to single-rank.

    AST check: there are two `torch.manual_seed` calls in train.main();
    the second one (the per-rank one, using `args.seed + rank`) comes
    AFTER the TitansMAGGPT2 construction call.
    """
    src = _main_source()
    tree = ast.parse(src.strip())

    seed_calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
    ]
    assert len(seed_calls) >= 2, (
        f"expected >= 2 torch.manual_seed() calls in train.main() "
        f"(one for base seed, one per-rank); got {len(seed_calls)}"
    )

    # Find TitansMAGGPT2(...) construction line.
    model_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TitansMAGGPT2"
    ]
    assert model_lines, "TitansMAGGPT2() not constructed in train.main()"

    model_line = min(model_lines)
    # The LAST manual_seed call must be after the model construction.
    last_seed = max(seed_calls)
    assert last_seed > model_line, (
        f"last manual_seed at line {last_seed} comes BEFORE model "
        f"construction at line {model_line} — per-rank divergence "
        f"(G204) won't fire."
    )


def test_init_destroy_process_group_pair_present():
    """G201 — `init_process_group` and `destroy_process_group` must both
    appear in train.main(). G225's try/finally test covers that destroy
    is in the finally clause; this test asserts the basic pair exists at
    all (catches a future refactor that drops one without removing the other).
    """
    src = _main_source()
    tree = ast.parse(src.strip())
    init_lines = _find_call_lines(tree, "init_process_group")
    destroy_lines = _find_call_lines(tree, "destroy_process_group")
    assert init_lines, "init_process_group not found in train.main() (G201)"
    assert destroy_lines, "destroy_process_group not found in train.main() (G201)"
