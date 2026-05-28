"""Resource leak guards: file handles must be opened in `with` blocks.

the natural `documents = (line for line in open('corpus.txt'))`
pattern leaks the file handle until generator GC at program exit. In a
long-running workflow that spawns many tokenizer jobs the handle count
hits ulimit. Defense: every script that reads a corpus must open the
file inside a `with` block.

Structural test — parses the entry-point scripts and checks that every
`open()` call appears as the context expression of a `with` statement.
"""

import ast
import inspect

import cli.finetune as finetune_mod
import cli.train as train_mod


def _open_calls_in_with(source: str) -> tuple[int, int]:
    """Count (open-in-with, total-open) calls in `source`."""
    tree = ast.parse(source)
    total = 0
    in_with = 0
    open_call_ids = set()

    # First pass: gather every `open(...)` call's id().
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ):
            total += 1
            open_call_ids.add(id(node))

    # Second pass: walk every `with` statement's context_expr and check
    # whether it (or any nested Call inside it) is one of the open calls.
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                for sub in ast.walk(ctx):
                    if id(sub) in open_call_ids:
                        in_with += 1

    return in_with, total


def test_finetune_opens_corpus_in_with_block():
    src = inspect.getsource(finetune_mod)
    in_with, total = _open_calls_in_with(src)
    assert total > 0, "cli/finetune.py has no open() — corpus not read?"
    assert in_with == total, (
        f"{total - in_with} of {total} open() calls in cli/finetune.py "
        f"are NOT inside `with` blocks — file-handle leak hazard."
    )


def test_train_main_opens_corpus_in_with_block():
    src = inspect.getsource(train_mod.main)
    in_with, total = _open_calls_in_with(src)
    assert total > 0, "train.main() has no open() — corpus not read?"
    assert in_with == total, (
        f"{total - in_with} of {total} open() calls in train.main() are NOT "
        f"inside `with` blocks — file-handle leak hazard."
    )
