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
