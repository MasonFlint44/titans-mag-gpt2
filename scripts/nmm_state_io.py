"""Persistent NMM state I/O for `generate.py`.

A `nmm_states` list (per-layer `(M, S)` tuples, possibly with `None` entries
for layers without NMM and possibly list-of-tuples for multi-head) is the
"session memory" that lets the model carry context across separate generation
calls. This module saves/loads that state atomically with a config fingerprint
so a state file built for one model can't silently load into a mismatched one.

The conv buffer is INTENTIONALLY not persisted — it's rebuilt from the first
k-1 tokens of the next prompt by `init_decode_cache`, which is the right
behavior for "this state is from a previous session, the conv window for the
new prompt starts fresh."
"""

import os
import tempfile
from pathlib import Path

import torch


# Format version: bump when the on-disk schema changes incompatibly. The load
# path checks this against the loaded file's `format_version` and rejects
# files from a future version (we can't know how to read them safely).
_FORMAT_VERSION = 1


def _fingerprint(config) -> dict:
    """Subset of TitansConfig fields that determine NMM state shape.

    Two configs with identical fingerprints produce structurally compatible
    state files. Fields that don't affect state shape (chunk_size, dropout,
    use_swa, training-only knobs, finetune_mode, etc.) are intentionally
    excluded so a state file from a fine-tune can be loaded for generation
    with a different chunk_size, for example.
    """
    return {
        "n_layer": config.n_layer,
        "n_embd": config.n_embd,
        "nmm_n_persistent": config.nmm_n_persistent,  # affects state INIT (memory_mlp weights)
        "nmm_expansion": config.nmm_expansion,
        "nmm_low_rank": config.nmm_low_rank,
        "nmm_n_heads": config.nmm_n_heads,
        "nmm_momentum_order": config.nmm_momentum_order,
        "nmm_layer_indices": (
            None if config.nmm_layer_indices is None
            else sorted(config.nmm_layer_indices)
        ),
        "nmm_state_dtype": config.nmm_state_dtype,
    }


class StateConfigMismatch(ValueError):
    """Loaded NMM state was saved for a model with different shape-affecting
    config. The state would either silently corrupt forward (wrong shapes
    promoted via broadcasting) or fail deep in the per-block loop with a
    confusing shape error. We fail fast at load with a clear diff instead.
    """


def save_nmm_state(path, nmm_states, config) -> None:
    """Atomically write `nmm_states` to `path` with a config fingerprint.

    Atomicity: write to `path.tmp` then `os.rename(tmp, path)`. POSIX rename
    is atomic on the same filesystem — prevents corruption if the process
    dies mid-write. Same pattern as `save_checkpoint_rotating`.

    `nmm_states` is the per-layer list returned by `forward()` /
    `prepare_decode()` / the cache after a `forward_step` loop. Entries may
    be `None` (non-NMM layers under `nmm_layer_indices`) or list-of-tuples
    (multi-head NMM). torch.save handles all three structures natively.

    Tensors are CPU'd before save so the file is portable across machines
    and we don't pin VRAM holding a reference.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_cpu(obj):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu")
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            cls = type(obj)
            return cls(_to_cpu(x) for x in obj)
        return obj

    payload = {
        "format_version": _FORMAT_VERSION,
        "fingerprint": _fingerprint(config),
        "nmm_states": _to_cpu(nmm_states),
    }
    # Atomic write: tmp file in the same dir (rename is atomic only within
    # one filesystem) + rename. Use tempfile.NamedTemporaryFile for the name
    # but write via torch.save so the tensor metadata is preserved.
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    os.close(tmp_fd)  # we'll reopen via torch.save
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX, atomic on Windows ≥ Vista
    except BaseException:
        # Clean up the partial tmp file on any failure path.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def load_nmm_state(path, config, device) -> list:
    """Load NMM state from `path`, validate it matches `config`, move to `device`.

    Raises:
        FileNotFoundError if `path` doesn't exist (caller decides whether
            to initialize fresh).
        StateConfigMismatch if the saved fingerprint doesn't match this
            model's config — see the exception docstring for why this has
            to be loud.
        ValueError if the file is from a future format_version.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    # weights_only=False: state can contain non-tensor structure (None
    # entries for plain blocks, tuple-of-dicts for multi-head). Same
    # rationale as `load_checkpoint` in train.py (G168).
    payload = torch.load(path, map_location="cpu", weights_only=False)

    fmt = payload.get("format_version")
    if fmt is None or fmt > _FORMAT_VERSION:
        raise ValueError(
            f"NMM state file at {path} has format_version={fmt!r}, but this "
            f"build only knows how to read versions <= {_FORMAT_VERSION}. "
            f"Upgrade your codebase or delete the file to start a fresh session."
        )

    saved_fp = payload.get("fingerprint", {})
    current_fp = _fingerprint(config)
    if saved_fp != current_fp:
        diffs = {
            k: (saved_fp.get(k), current_fp.get(k))
            for k in set(saved_fp) | set(current_fp)
            if saved_fp.get(k) != current_fp.get(k)
        }
        raise StateConfigMismatch(
            f"NMM state at {path} was saved for a different model shape. "
            f"Mismatched fields (saved → current): {diffs}. The state would "
            f"silently produce wrong outputs (broadcasting) or fail with a "
            f"cryptic shape error inside the first block. Save a fresh state "
            f"with the current config, or load with the matching config."
        )

    def _to_device(obj):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            cls = type(obj)
            return cls(_to_device(x) for x in obj)
        return obj

    return _to_device(payload["nmm_states"])
