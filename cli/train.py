"""Consolidated training driver (fine-tune and from-scratch entry points)."""

import dataclasses
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm


# --- Resume-flow shared constants and helpers ----------------------------------
# These are used by both `train.py` (from-scratch / DDP) and
# `cli/finetune.py` (single-GPU finetune) so the two entry points have
# matching behavior on `--resume-from`. Adding a new backend-only NMM flag?
# Append to RESUME_OVERRIDABLE_BACKEND_FLAGS so both paths honor it.

# NMM CLI flags that can be overridden when resuming a checkpoint. These
# select the NS5 implementation backend without changing any parameter
# shape or graph topology — safe to flip mid-run when one backend OOMs
# or breaks under a newer torch.
RESUME_OVERRIDABLE_BACKEND_FLAGS = frozenset({
    "nmm_use_gram_ns5",
    "nmm_use_cans",
    "nmm_ns5_steps",
})

# Scaffolding flags persisted into the checkpoint at save time. On resume,
# we compare the user's CLI values against these and warn on disagreement
# — catches "forgot --grad-accum 16, now I'm OOMing" before training
# starts. The user still controls the final values (extending --max-steps
# is a legitimate, common case), the warning is purely informational.
SAVED_TRAINING_ARGS = (
    "batch_size",
    "grad_accum",
    "warmup_steps",
    "max_steps",
    "save_every",
)


# graph-break suppression for the NMM's `bool(doc_boundaries.any())`
# scalar read at `model/nmm.py:_forward_chunk_blockwise` (and the matching
# sequential-path call). Without this, `torch.compile` warns at first
# encounter and SPLITS the compiled forward at every NMM block boundary,
# losing some of the fusion the --compile-model flag is meant to deliver.
#
# Setting `capture_scalar_outputs = True` tells dynamo to include the
# scalar sync (item() / bool()) inside the captured graph instead of
# bailing out. The Python-level branch downstream (`if any_boundary: ...`)
# would still cause specialization, but in our SQuAD training the bool
# value is overwhelmingly False (no doc boundary mid-chunk), so dynamo
# caches that branch's graph and reuses it for the vast majority of
# steps.
#
# Set at module import — runs before any `torch.compile(...)` call in
# downstream entry points (cli/finetune.py wraps the model AFTER
# `from cli.train import ...`). Idempotent if set again elsewhere.
torch._dynamo.config.capture_scalar_outputs = True


# ---------------------------------------------------------------------------
# 4.1 — 4-group optimizer
# ---------------------------------------------------------------------------

# Code-level constants so apply_lr's base_lrs cannot drift on resume.
BASE_LR_GPT2 = 3e-4
BASE_LR_NMM = 9e-4  # 3x GPT-2 per paper
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)  # NOT PyTorch default (0.9, 0.999)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0

# Substring set for no-decay routing. Catches:
#   bias       -- bias params on Linear/Conv
#   ln, norm   -- LayerNorm and ResidualNorm params
#   out_scale  -- magnitude gate (init=0; decay would resist learning)
#   gamma      -- gamma_mem / gamma_attn (init=1; decay shrinks memory branch)
#   persistent -- persistent_mem (learned prefix; decay reduces capacity)
NO_DECAY_SUBSTRINGS = ("bias", "ln", "norm", "out_scale", "gamma", "persistent")

# Substring set for NMM routing. Paper-strict: only the NMM module's own
# parameters get the 3× learning rate. `gamma_mem` / `gamma_attn` and
# `persistent_mem` (block-level under `persistent_prefix_mode="per_block"`,
# model-level under `"model_wide"`) are not specified by the paper as
# fast-LR; they route to the backbone (1×) group. `ln_nmm` is the NMM
# pre-norm and still routes via the leading "nmm" hit on its full path.
NMM_SUBSTRINGS = ("nmm",)


# Memory-path substrings used by `--freeze-backbone`. ANY param whose name
# contains one of these is left trainable; everything else is frozen.
# Wider than NMM_SUBSTRINGS because under freeze we want to train every
# component that participates in the memory pathway:
#   nmm        -- NMM internals + ln_nmm
#   gamma      -- MAG gate scalars (gamma_mem, gamma_attn)
#   out_scale  -- block-level NMM output scaling
#   persistent -- persistent_mem prefix (both model_wide and per_block modes)
# TPTT-inspired (https://github.com/fabienfrfr/tptt): freezing the backbone
# concentrates the fine-tune gradient on the new memory mechanism. Without
# this, gradient is split across ~125M backbone params and the memory path
# starves.
FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS = (
    "nmm", "gamma", "out_scale", "persistent",
)


def _is_no_decay(name: str) -> bool:
    return any(nd in name for nd in NO_DECAY_SUBSTRINGS)


def _is_nmm(name: str) -> bool:
    return any(k in name for k in NMM_SUBSTRINGS)


def _is_memory_path_param(name: str) -> bool:
    return any(k in name for k in FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS)


def freeze_backbone(model: nn.Module) -> tuple[int, int]:
    """Set `requires_grad=False` on every backbone param. Memory-path
    params (`nmm`, `gamma`, `out_scale`, `persistent`) stay trainable.

    Call BEFORE `build_optimizer` — the optimizer's param-group walk
    skips `requires_grad=False` tensors automatically.

    Returns (frozen_count, trainable_count) for a sanity log line."""
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        if _is_memory_path_param(name):
            p.requires_grad = True
            trainable += 1
        else:
            p.requires_grad = False
            frozen += 1
    return frozen, trainable


# Substrings identifying the input/output representation layer — the
# pieces that define the model's vocabulary and absolute-position
# semantics. Freezing these is conservative (they're data-hungry to
# fine-tune well, and small training corpora can distort them) while
# keeping the transformer blocks free to adapt their attention/MLP
# weights for the NMM-augmented residual stream.
EMBEDDING_SUBSTRINGS = ("wte", "wpe", "ln_f")


def _is_embedding_param(name: str) -> bool:
    return any(s in name for s in EMBEDDING_SUBSTRINGS)


def freeze_embeddings_only(model: nn.Module) -> tuple[int, int]:
    """Softer freeze than `freeze_backbone`: only the input/output
    representation params (`wte`, `wpe`, `ln_f`) are frozen. Transformer
    blocks (attention + MLP + block LayerNorms) stay trainable so they
    can adapt to the NMM-augmented residual stream — specifically, so
    attention can LEARN to attend to NMM-modulated tokens, which a
    fully frozen backbone cannot.

    Use this when full `--freeze-backbone` is too restrictive (it broke
    short-distance recall in our needle-in-haystack run because attention
    couldn't compensate for the injected NMM signal) but full fine-tune
    leaves the gradient signal too diluted.

    Memory-path params stay trainable regardless (they're not embeddings).
    Returns (frozen_count, trainable_count)."""
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        if _is_embedding_param(name):
            p.requires_grad = False
            frozen += 1
        else:
            p.requires_grad = True
            trainable += 1
    return frozen, trainable


def collect_out_scale_params(model: nn.Module) -> list:
    """Return every `out_scale` Parameter in the model (one per NMM block,
    typically). The gate-ramp schedule writes into these tensors directly
    each step during the ramp phase, so we cache the list once instead of
    walking `named_parameters()` per step."""
    out = []
    for name, p in model.named_parameters():
        if "out_scale" in name:
            out.append(p)
    return out


def gate_ramp_value(step: int, ramp_steps: int, target: float) -> float:
    """Linear ramp schedule value at training `step`. The schedule reaches
    `target` exactly at `step == ramp_steps - 1` (the LAST in-ramp step),
    so the model has been trained at the full target value for at least
    one optimizer cycle before requires_grad is re-enabled at
    `step == ramp_steps`.

    For step >= ramp_steps the function returns `target` for completeness,
    but in practice callers stop overwriting out_scale at that point and
    let the optimizer take over.

    Raises ValueError if `ramp_steps <= 0` — caller should guard with
    `if ramp_steps > 0` before invoking.
    """
    if ramp_steps <= 0:
        raise ValueError(
            f"ramp_steps must be > 0 (got {ramp_steps}); guard the call site "
            f"with `if ramp_steps > 0`."
        )
    if step >= ramp_steps:
        return target
    fraction = (step + 1) / ramp_steps
    return target * fraction


def build_optimizer(
    model: nn.Module,
    lr_gpt2: float = BASE_LR_GPT2,
    lr_nmm: float = BASE_LR_NMM,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple = BETAS,
    eps: float = ADAM_EPS,
    use_8bit: bool = False,
):
    """Build the 4-group AdamW (optional 8-bit variant).

    Groups: (gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay).
    NMM groups get a higher LR (paper uses 3x); no_decay groups have wd=0.

    Every param must land in exactly one group. Verified by checking sum-of-
    group-sizes equals the model's total param count.

    When `use_8bit=True`, the returned optimizer is `bnb.optim.AdamW8bit`
    instead of `torch.optim.AdamW`. Optimizer state (m, v moments)
    quantizes to 8-bit per-block — roughly 4x smaller than fp32. The
    fp32 master weights are unaffected. Compatible with the existing
    4-group layout. Requires `bitsandbytes` installed; falls back with a
    clear error if missing.
    """
    gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_dc = _is_no_decay(name)
        if _is_nmm(name):
            (nmm_no_decay if no_dc else nmm_decay).append(p)
        else:
            (gpt2_no_decay if no_dc else gpt2_decay).append(p)

    groups = [
        {"params": gpt2_decay, "lr": lr_gpt2, "weight_decay": weight_decay},
        {"params": gpt2_no_decay, "lr": lr_gpt2, "weight_decay": 0.0},
        {"params": nmm_decay, "lr": lr_nmm, "weight_decay": weight_decay},
        {"params": nmm_no_decay, "lr": lr_nmm, "weight_decay": 0.0},
    ]

    total_in_groups = sum(len(g["params"]) for g in groups)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    if total_in_groups != total_params:
        raise ValueError(
            f"Optimizer groups cover {total_in_groups} params but the model has "
            f"{total_params} trainable params — either a param is missing or "
            f"is double-counted across groups."
        )

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(
                "use_8bit=True requires the `bitsandbytes` package. Install "
                "with `pip install bitsandbytes` (or `uv pip install "
                "bitsandbytes`)."
            ) from e
        # bnb.optim.AdamW8bit takes the same betas/eps/groups API as
        # torch.optim.AdamW — drop-in replacement. The 8-bit quantization
        # applies block-wise to m, v moments; the fp32 master weights are
        # unaffected, so per-step numerics differ only by the moment-
        # quantization noise (empirically <1% loss-curve drift in
        # bitsandbytes' own benchmarks).
        return bnb.optim.AdamW8bit(groups, betas=betas, eps=eps)

    return AdamW(groups, betas=betas, eps=eps)


# ---------------------------------------------------------------------------
# 4.2 — TBPTT train_step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    batch: tuple,
    nmm_states,
    optimizer: AdamW,
    device: torch.device,
    autocast_dtype: torch.dtype = None,
) -> tuple:
    """One TBPTT step: forward (optionally in autocast), backward/clip/step in fp32.

    Returns (loss, new_nmm_states, grad_norm) where loss/grad_norm are floats.
    On NaN/Inf grad-norm: zero grads, RETURN None for nmm_states so the
    caller's next call re-initializes. Returning the existing
    state would propagate a NaN-tainted M through the next forward and
    livelock until the next document boundary.

    `autocast_dtype=None` (the default) runs forward in fp32 — required for
    CPU tests. On CUDA, the entry points pass `torch.bfloat16` to enable the
    bf16-autocast forward + fp32 backward pattern. Mixing CPU autocast
    with our torch.func.grad inner loop produces a mixed-dtype backward graph
    that fails with "expected BFloat16 but found Float" — autocast on CPU
    has narrower op coverage than CUDA, so we don't enable it there by default.

    Mixed precision: backward + clip + step always run outside autocast in
    fp32. clip_grad_norm_ inside autocast computes the norm in low
    precision, defeating gradient clipping.

    Note: detach_states is imported lazily to avoid a circular import at
    module load.
    """
    from model.nmm import detach_states

    input_ids, doc_boundaries = batch
    input_ids = input_ids.to(device, non_blocking=True)
    doc_boundaries = doc_boundaries.to(device, non_blocking=True)
    nmm_states = detach_states(nmm_states)

    if autocast_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
    else:
        logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    if not torch.isfinite(grad_norm):
        # NaN-skip: don't apply the bad gradients; reset NMM state so the
        # next call re-inits (otherwise a NaN-tainted M livelocks).
        optimizer.zero_grad(set_to_none=True)
        return loss.item(), None, grad_norm.item()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item(), nmm_states, grad_norm.item()


# ---------------------------------------------------------------------------
# 4.3 — LR schedule, NMM-norm logging, checkpoint helpers
# ---------------------------------------------------------------------------

def get_lr_multiplier(
    step: int,
    warmup_steps: int = 1000,
    max_steps: int = 100_000,
    min_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to `min_ratio * peak`.

    Returns a scalar in [min_ratio, 1.0]. Caller multiplies each
    param_group's base LR by this.

    Validates max_steps > warmup_steps. max_steps == warmup_steps
    yields a degenerate zero-length cosine; max_steps < warmup_steps lets
    the warmup branch fire past max_steps and never decay. Both are silent
    miscalibrations — raise loudly instead.
    """
    if max_steps <= warmup_steps:
        raise ValueError(
            f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps}). "
            f"For a warmup-only schedule with no decay, set min_ratio=1.0 and "
            f"max_steps just beyond your intended training duration."
        )
    if step < warmup_steps:
        return step / warmup_steps
    if step >= max_steps:
        return min_ratio
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def apply_lr(
    optimizer: AdamW,
    base_lrs: list,
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_ratio: float = 0.1,
) -> float:
    """Scale every param_group's LR by the schedule multiplier. Preserves the
    1:1:3:3 gpt2/nmm LR ratio across groups — naive single-group or
    uniform-clobber updates would either skip groups or destroy the ratio.

    `base_lrs` MUST come from code-level constants, never from
    `optimizer.param_groups[i]['lr']` after `load_state_dict` — that path
    captures the mid-cosine deflated value and compounds the deflation
    every resume.

    `warmup_steps` and `max_steps` are required positional args (no
    defaults) so the caller cannot silently inherit the 1k/100k schedule
    when running a 200-step overfit.
    """
    lr_mul = get_lr_multiplier(
        step,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_ratio=min_ratio,
    )
    for g, base_lr in zip(optimizer.param_groups, base_lrs):
        g["lr"] = base_lr * lr_mul
    return lr_mul


def base_lrs_from_constants() -> list:
    """The canonical base_lrs derivation: from code constants, matching the
    4-group order in build_optimizer (gpt2_decay, gpt2_no_decay, nmm_decay,
    nmm_no_decay). Resume-safe by construction."""
    return [BASE_LR_GPT2, BASE_LR_GPT2, BASE_LR_NMM, BASE_LR_NMM]


def _layer_norm_M(layer_state):
    """Compute ||M||_F (batch-mean) for a single per-layer state. Handles
    three shapes:
      - None: returns None. Plain (non-NMM) blocks have None state slots
        when `nmm_layer_indices` is set.
      - single-head: `(M, S, conv_buf)` tuple — item 6.
      - multi-head: `[(M_h, S_h, conv_buf_h), ...]` list — returns
        per-head mean.

    `_qs` int8 scale companions in M are filtered out so they don't
    contribute to the norm. `conv_buf` is the rolling cross-chunk conv
    state; ignored for this metric.
    """
    if layer_state is None:
        return None
    if isinstance(layer_state, list):
        return sum(_layer_norm_M(s) for s in layer_state) / len(layer_state)
    M = layer_state[0]
    sq_sum = sum(
        (v.float() ** 2).sum(dim=tuple(range(1, v.ndim)))
        for k, v in M.items() if not k.endswith("_qs")
    )
    return sq_sum.sqrt().mean().detach().item()


def compute_nmm_norm(nmm_states) -> list:
    """Per-layer ||M||_F (treating W1/W_gate/W2 as one block), averaged across
    batch. Returns None if states is None (first-step case).

    Multi-head safe: when `nmm_n_heads > 1`, the per-layer state is a
    list of per-head `(M, S)` tuples; we report the mean of per-head norms
    per layer (so the returned list has length n_layer regardless of head
    count — convenient for log parsers).

    Subset-of-layers safe: plain (non-NMM) blocks contribute a None
    entry at their position rather than skewing the average."""
    if nmm_states is None:
        return None
    return [_layer_norm_M(s) for s in nmm_states]


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    path,
    model: nn.Module,
    optimizer: AdamW,
    step: int,
    config,
    training_args: dict | None = None,
) -> None:
    """Save model state_dict + optimizer state + step + config to a single file.

    config is required (not optional): finetune_mode controls block structure
    (gamma_attn presence, out_scale init), so resume needs it to rebuild the
    same model topology. Without saving it, the resume code can't tell which
    structure to construct, producing state_dict mismatch errors or silently-
    wrong inits.

    `training_args` (optional) records the scaffolding CLI flags
    (batch_size, grad_accum, warmup_steps, max_steps, save_every) the run
    was using at save time. On resume, the loader compares these against
    the current CLI values and warns on disagreement — prevents silent OOM
    when the user forgets to re-pass --grad-accum / --batch-size.

    NMM states are intentionally NOT saved — they're per-sequence
    accumulators, not model state. Resume re-initializes from
    memory_mlp.W*.weight.
    """
    #//— _unwrap strips torch.compile and DDP/FSDP prefixes so
    # the saved state_dict is portable across wrapping choices on resume.
    from model import _unwrap

    payload = {
        "state_dict": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),
    }
    if training_args is not None:
        payload["training_args"] = dict(training_args)
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> dict:
    """Load checkpoint dict. weights_only=False is required: PyTorch
    2.6+ flipped the default to True and would reject our nested optimizer
    state on some version combos.

    Returns the raw dict; caller rebuilds the model from `ckpt['config']`,
    then `model.load_state_dict(ckpt['state_dict'])`, then optimizer-
    construct + (optional) `optimizer.load_state_dict(ckpt['optimizer'])`.
    Missing 'optimizer' key (HF-init checkpoint from load_pretrained) is
    handled by the caller.
    """
    return torch.load(path, map_location=device, weights_only=False)


def config_size_label(config) -> str:
    """Map a config's backbone dims back to the closest factory name for
    use in --resume-from's architecture-mismatch warning."""
    if config.n_embd == 768 and config.n_layer == 12:
        return "small"
    if config.n_embd == 1024 and config.n_layer == 24:
        return "medium"
    if config.n_embd == 1280 and config.n_layer == 36:
        return "large"
    if config.n_embd == 1600 and config.n_layer == 48:
        return "xl"
    return "custom"


def training_args_from_namespace(args) -> dict:
    """Pluck the scaffolding flags listed in SAVED_TRAINING_ARGS out of an
    argparse Namespace into a plain dict (for persistence into checkpoints).
    Missing attributes are silently skipped so this stays usable from
    notebooks / tests that don't go through the full CLI."""
    return {k: getattr(args, k) for k in SAVED_TRAINING_ARGS if hasattr(args, k)}


def apply_resume_overrides_and_warn(
    config, args, nmm_kwargs: dict, *, log_prefix: str, rank: int = 0,
) -> None:
    """On --resume-from, apply NMM backend overrides to `config` (mutates in
    place) and warn on rank 0 about any architecture-affecting CLI flags
    that are being ignored. Shared between train.py and finetune.py so the
    two entry points behave identically.

    `nmm_kwargs` is the dict returned by `nmm_kwargs_from_args(args)` —
    only flags the user explicitly set. `--size` and `--chunk-size` are
    handled separately because they don't come through that helper."""
    overrides = {
        k: v for k, v in nmm_kwargs.items()
        if k in RESUME_OVERRIDABLE_BACKEND_FLAGS
    }
    for k, v in overrides.items():
        setattr(config, k, v)
    ignored = {
        k: v for k, v in nmm_kwargs.items()
        if k not in RESUME_OVERRIDABLE_BACKEND_FLAGS
    }
    # --size default is "small"; only flag if the user passed a value that
    # actually disagrees with the checkpoint.
    ckpt_size = config_size_label(config)
    if args.size != ckpt_size and args.size != "small":
        ignored["size"] = args.size
    # --chunk-size: default is 1024 (matches the consumer-GPU recipe). Only
    # warn if a non-default value disagrees with the checkpoint.
    if args.chunk_size != config.chunk_size and args.chunk_size != 1024:
        ignored["chunk_size"] = args.chunk_size

    if rank != 0:
        return
    if overrides:
        print(
            f"{log_prefix} --resume-from: applying non-architecture "
            f"overrides ({sorted(overrides.keys())}) to the saved config.",
            file=sys.stderr,
        )
    if ignored:
        print(
            f"{log_prefix} --resume-from: ignoring architecture-affecting "
            f"CLI flags ({sorted(ignored.keys())}) — checkpoint's saved "
            f"config is authoritative. Drop them from the command line "
            f"to silence this warning.",
            file=sys.stderr,
        )


def warn_training_arg_drift(
    ckpt: dict, args, *, log_prefix: str, rank: int = 0,
) -> None:
    """If the checkpoint persisted `training_args`, compare them against the
    current CLI values and warn on disagreement. No-op for older checkpoints
    that predate this field. Emits on rank 0 only."""
    if rank != 0:
        return
    saved = ckpt.get("training_args")
    if not saved:
        return
    current = training_args_from_namespace(args)
    diffs = []
    for k in SAVED_TRAINING_ARGS:
        if k in saved and k in current and saved[k] != current[k]:
            diffs.append((k, saved[k], current[k]))
    if not diffs:
        return
    body = ", ".join(f"{k}: {old} -> {new}" for k, old, new in diffs)
    print(
        f"{log_prefix} --resume-from: scaffolding flags differ from "
        f"checkpoint ({body}). Continuing with the CLI values; re-pass the "
        f"checkpoint's values explicitly to silence this warning.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Checkpoint rotation
# ---------------------------------------------------------------------------

# Step checkpoints are named `step_{N:07d}.pt`. The fixed width keeps lex
# order == numeric order so callers can `sorted(dir.glob("step_*.pt"))` if
# they want to skip our helpers. 7 digits comfortably covers any realistic
# training run (up to 10M steps).
_STEP_CKPT_RE = re.compile(r"^step_(\d{7})\.pt$")
LATEST_CKPT_NAME = "latest.pt"


def list_step_checkpoints(save_dir) -> list:
    """Return `[(step, Path)]` for all rotated checkpoints in `save_dir`,
    sorted by step ascending. Files that don't match `step_NNNNNNN.pt`
    (including `latest.pt` and any user-placed files) are ignored.
    """
    p = Path(save_dir)
    if not p.is_dir():
        return []
    out = []
    for entry in p.iterdir():
        m = _STEP_CKPT_RE.match(entry.name)
        if m and entry.is_file():
            out.append((int(m.group(1)), entry))
    out.sort(key=lambda t: t[0])
    return out


def prune_old_checkpoints(save_dir, keep_last_n) -> list:
    """Delete oldest `step_*.pt` files until at most `keep_last_n` remain.

    Returns the list of `Path`s that were deleted (useful for tests and
    logging). `keep_last_n=None` or `<= 0` means unbounded — nothing is
    pruned. `latest.pt` and any non-matching files are never touched.
    """
    if keep_last_n is None or keep_last_n <= 0:
        return []
    existing = list_step_checkpoints(save_dir)
    if len(existing) <= keep_last_n:
        return []
    to_delete = existing[: len(existing) - keep_last_n]
    deleted = []
    for _step, path in to_delete:
        try:
            path.unlink()
            deleted.append(path)
        except FileNotFoundError:
            # Another process beat us to it; not an error worth crashing for.
            pass
    return deleted


def save_checkpoint_rotating(
    save_dir,
    model: nn.Module,
    optimizer: AdamW,
    step: int,
    config,
    keep_last_n: int = 3,
    training_args: dict | None = None,
) -> Path:
    """Save a step checkpoint into `save_dir` and prune oldest beyond `keep_last_n`.

    Writes two files:
      - `step_{N:07d}.pt` — the rotated, immutable per-step checkpoint
      - `latest.pt`       — a COPY (not symlink) of the most recent step,
                            so consumers don't need to know the step number
                            and the file works on Windows

    Why copy and not symlink: symlinks are cross-filesystem-fragile on some
    Linux setups and outright unsupported on Windows. A copy costs one
    extra fsync per save (negligible compared to the model write) and Just
    Works everywhere.

    `keep_last_n=None` or `<= 0` keeps every checkpoint (no pruning).
    Default `3` is a reasonable balance — enough to recover from a bad
    save or an OOM-truncated last write, small enough to bound disk.

    Returns the `Path` to the rotated step file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_path = save_dir / f"step_{step:07d}.pt"
    save_checkpoint(
        step_path, model, optimizer, step, config,
        training_args=training_args,
    )

    # Mirror to latest.pt. Atomic replace (write to .tmp then rename) so a
    # mid-save crash never leaves latest.pt half-written.
    latest_path = save_dir / LATEST_CKPT_NAME
    tmp_path = save_dir / f"{LATEST_CKPT_NAME}.tmp"
    shutil.copyfile(step_path, tmp_path)
    os.replace(tmp_path, latest_path)

    prune_old_checkpoints(save_dir, keep_last_n)
    return step_path


# ---------------------------------------------------------------------------
# 4.5 — Training loop helpers
# ---------------------------------------------------------------------------

def is_partial_cycle(batch, accum_i: int) -> bool:
    """the correct partial-cycle skip condition.

    A partial cycle is when the loader's StopIteration fires mid-accumulation
    (i.e., before the final micro-batch of an accumulation cycle that would
    have triggered the all-reduce). The trip wire is `batch is None` (loader
    exhausted) AND `accum_i > 0` (we already ran at least one micro-batch in
    this cycle, so accumulator grads are non-zero — we'd be tempted to
    optimizer.step() on them, but that would diverge across ranks because
    different ranks may have different numbers of partial micro-batches
    depending on stream length).

    The naive `accum_i < ACCUM_STEPS - 1` check is silently off-by-one: at
    iter K-1, StopIteration fires after running micro-batch K-1 (the last
    one in the cycle), so accum_i == K-1 == ACCUM_STEPS - 1 -> check is
    False -> the cycle is treated as complete -> step fires (correct). But
    when StopIteration fires at iter K (cycle finished, fetching the NEXT
    cycle's first batch), accum_i == 0 -> naive check is True
    (0 < K-1) -> we erroneously skip what was actually a complete cycle. The
    `(batch is None) and (accum_i > 0)` form has no off-by-one.
    """
    return (batch is None) and (accum_i > 0)


def run_training(
    model: nn.Module,
    optimizer: AdamW,
    loader,
    device: torch.device,
    max_steps: int,
    warmup_steps: int,
    accum_steps: int = 1,
    log_every: int = 50,
    save_every: int = None,
    save_dir: str = None,
    keep_last_n: int = 3,
    config=None,
    autocast_dtype: torch.dtype = None,
    rank: int = 0,
    is_distributed: bool = False,
    show_progress: bool = True,
    start_step: int = 0,
    batch_size: int | None = None,
    gate_ramp_steps: int = 0,
    gate_ramp_target: float = 0.0,
) -> None:
    """Top-level training loop covering both Phase 4.4 (fine-tune) and 4.5
    (from-scratch). One optimizer.step per accumulation cycle of `accum_steps`
    micro-batches.

    DDP: caller wraps `model` with DDP and passes `is_distributed=True`. All
    micro-batches except the last in a cycle run inside `model.no_sync()` to
    suppress per-microbatch all-reduce. Partial cycle at corpus end
    is detected via's `(batch is None) and (accum_i > 0)` check and
    skipped to avoid rank divergence — the per-rank `.grad` buffers were
    never AllReduce'd. On single GPU, partial cycles step normally.

    NaN-skip: if accumulated grad_norm is non-finite, zero grads AND reset
    nmm_states to None so the next cycle's first micro-batch hits
    model.forward's None branch and re-inits — otherwise NaN-tainted M
    livelocks until the next document boundary.

    NCCL teardown (try/finally with destroy_process_group) lives
    in the entry-point script that wraps this call, not here — keeps this
    function single-purpose and reusable from notebooks / tests.

    Training continues until `step >= max_steps`. When the loader exhausts
    mid-training (max_steps > batches-per-epoch), the iterator is rebuilt
    and iteration continues — matching docs/archive/PLAN.md §4.5's
    `for epoch in range(N_EPOCHS):` structure. Without this, small corpora
    silently early-stop after one pass and the user sees max_steps not
    reached with no error.

    On a partial cycle at corpus end under DDP, we discard the cycle's
    accumulated grads and end training — re-iterating mid-cycle would
    re-process the same micro-batches across ranks asymmetrically and
    drift them.

    Checkpointing: when both `save_every` and `save_dir` are set, every
    `save_every` completed cycles (and on the final step) write a rotated
    checkpoint via `save_checkpoint_rotating` into `save_dir`. Files are
    named `step_{N:07d}.pt` plus a `latest.pt` copy of the most recent;
    only the most recent `keep_last_n` step files are retained (default 3).
    Pass `keep_last_n=None` to disable pruning.

    Progress: when `show_progress=True` and `rank == 0`, a `tqdm` bar with
    per-step `loss`, `grad_norm`, `lr`, `tok/s`, and mean `nmm` norm is
    drawn to stderr. `log_every` periodic dumps are emitted via
    `tqdm.write` so they don't tear the bar.

    Resume: pass `start_step > 0` to pick up where a previous run left off.
    The caller is responsible for restoring model + optimizer state from a
    checkpoint BEFORE calling run_training (see `load_checkpoint` and the
    --resume-from CLI flag on train.py / cli/finetune.py). The
    `start_step` value sets the initial step counter so the LR schedule
    resumes at the right point in the warmup-then-cosine trajectory; the
    progress bar starts at `start_step / max_steps` rather than 0. Note
    that the dataloader iterator is rebuilt from scratch on every call —
    so a resumed run re-reads the corpus from the beginning, which is
    fine for multi-epoch training where the loader gets recycled anyway
    but means we don't preserve exact intra-epoch data position.
    """
    import contextlib

    if start_step < 0:
        raise ValueError(
            f"start_step must be >= 0 (got {start_step!r}); use 0 for a fresh "
            f"run or a positive value matching a saved checkpoint's step."
        )
    if start_step >= max_steps:
        raise ValueError(
            f"start_step ({start_step}) >= max_steps ({max_steps}) — there's "
            f"nothing to train. Increase max_steps past the checkpoint's step "
            f"to continue training."
        )

    # Snapshot the scaffolding flags so save_checkpoint can persist them
    # alongside `config`. Resume-time drift detection compares these
    # against the resumed CLI invocation. batch_size is the one value
    # that doesn't already live in `config`; the rest mirror this
    # function's parameters.
    training_args = {
        "grad_accum": accum_steps,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "save_every": save_every,
    }
    if batch_size is not None:
        training_args["batch_size"] = batch_size

    base_lrs = base_lrs_from_constants()
    model.train()
    nmm_states = None

    # Progress bar — only on rank 0, never under explicit show_progress=False.
    # `initial=start_step` makes the bar render with the resumed progress
    # rather than restarting from 0%.
    bar = tqdm(
        total=max_steps,
        initial=start_step,
        desc="train",
        unit="step",
        disable=not (show_progress and rank == 0),
        dynamic_ncols=True,
        leave=True,
    )

    step = start_step
    micro_batches = iter(loader)
    tokens_since_last_step = 0
    step_t0 = time.perf_counter()
    last_lr_mul = 0.0

    # Gate-ramp setup: cache the out_scale params once, freeze them during
    # the ramp so the optimizer doesn't fight the schedule, and re-enable
    # gradients exactly at step==gate_ramp_steps. Inspired by TPTT's
    # LiZACallback (initial_weight → final_weight over transition_step
    # steps): forcing the memory gate open on a fixed schedule prevents
    # the model from passively leaving the NMM contribution near zero
    # during early fine-tuning, where the LM-loss signal alone is too
    # diffuse to open it.
    out_scale_params = (
        collect_out_scale_params(model) if gate_ramp_steps > 0 else []
    )
    if out_scale_params and rank == 0:
        print(
            f"[run_training] gate ramping: {len(out_scale_params)} out_scale "
            f"params will be held to a linear schedule "
            f"0 -> {gate_ramp_target} over the first {gate_ramp_steps} steps, "
            f"then released to the optimizer.",
            file=sys.stderr,
        )
    if out_scale_params:
        for p in out_scale_params:
            p.requires_grad = False

    try:
        while step < max_steps:
            # Gate-ramp: hold out_scale to the schedule during the ramp.
            # `gate_ramp_value` returns target * (step+1)/ramp_steps so the
            # LAST in-ramp step (step == ramp_steps - 1) sees the full
            # target value. At step == gate_ramp_steps we hand control
            # back to the optimizer.
            if out_scale_params:
                if step < gate_ramp_steps:
                    target_val = gate_ramp_value(
                        step, gate_ramp_steps, gate_ramp_target,
                    )
                    for p in out_scale_params:
                        p.data.fill_(target_val)
                elif step == gate_ramp_steps:
                    # Ramp just ended — let the optimizer take over.
                    for p in out_scale_params:
                        p.requires_grad = True
                    if rank == 0:
                        tqdm.write(
                            f"[run_training] gate ramp complete at step "
                            f"{step}; out_scale params released to optimizer."
                        )

            cycle_ran_any_microbatch = False
            cycle_completed = True

            for accum_i in range(accum_steps):
                try:
                    batch = next(micro_batches)
                except StopIteration:
                    batch = None

                if batch is None and accum_i == 0:
                    # Loader exhausted exactly at cycle boundary. If we still
                    # have steps left, restart the iterator (next epoch); reset
                    # nmm_states since the new pass through the corpus is a
                    # fresh context. Returning here would be the silent-early-
                    # stop bug.
                    micro_batches = iter(loader)
                    nmm_states = None
                    try:
                        batch = next(micro_batches)
                    except StopIteration:
                        # Empty loader — nothing to do; stop.
                        return

                if is_partial_cycle(batch, accum_i):
                    #/: under DDP, this cycle's micro-batches ran with
                    # no_sync; per-rank .grad never AllReduce'd. Stepping would
                    # diverge ranks permanently. Discard + stop.
                    if is_distributed:
                        optimizer.zero_grad(set_to_none=True)
                        nmm_states = None  # match NaN-skip semantics
                        return
                    # Single-GPU: partial is safe (no AllReduce). Treat as complete.
                    cycle_completed = False
                    break

                input_ids, doc_boundaries = batch
                input_ids = input_ids.to(device, non_blocking=True)
                doc_boundaries = doc_boundaries.to(device, non_blocking=True)
                nmm_states = _detach_states(nmm_states)
                tokens_since_last_step += input_ids.numel()

                is_last_accum = (accum_i == accum_steps - 1)
                sync_ctx = (
                    model.no_sync()
                    if (is_distributed and not is_last_accum
                        and hasattr(model, "no_sync"))
                    else contextlib.nullcontext()
                )

                with sync_ctx:
                    if autocast_dtype is not None:
                        with torch.autocast(
                            device_type=device.type, dtype=autocast_dtype
                        ):
                            logits, nmm_states = model(
                                input_ids, nmm_states, doc_boundaries
                            )
                            loss = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.size(-1)),
                                input_ids[:, 1:].reshape(-1),
                            ) / accum_steps
                    else:
                        logits, nmm_states = model(
                            input_ids, nmm_states, doc_boundaries
                        )
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            input_ids[:, 1:].reshape(-1),
                        ) / accum_steps
                    loss.backward()
                cycle_ran_any_microbatch = True

            if not cycle_ran_any_microbatch:
                break

            # One optimizer.step per completed accumulation cycle.
            last_lr_mul = apply_lr(
                optimizer, base_lrs, step,
                warmup_steps=warmup_steps, max_steps=max_steps,
            )
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                # match the train_step NaN-skip pattern.
                nmm_states = None
            optimizer.zero_grad(set_to_none=True)

            # Per-step metrics. `loss * accum_steps` undoes the per-microbatch
            # division so reported loss is comparable across accum_steps choices.
            step_dt = max(time.perf_counter() - step_t0, 1e-9)
            tok_per_sec = tokens_since_last_step / step_dt
            loss_full = loss.item() * accum_steps
            grad_norm_v = grad_norm.item()
            nmm_norms = compute_nmm_norm(nmm_states)
            # `compute_nmm_norm` returns None when nmm_states is None (,
            # first-step case) AND emits per-layer Nones for plain blocks
            #. Under --vanilla-gpt2 every block is plain, so nmm_norms
            # is a list of Nones — truthy as a list, but `sum([None, ...])`
            # raises TypeError. Filter Nones BEFORE summing so the vanilla
            # control run (and any subset-NMM config) logs cleanly.
            valid_norms = (
                [n for n in nmm_norms if n is not None] if nmm_norms else []
            )
            nmm_mean = (
                sum(valid_norms) / len(valid_norms)
                if valid_norms else float("nan")
            )
            cur_lr = optimizer.param_groups[0]["lr"]  # gpt2_decay group

            if rank == 0:
                bar.update(1)
                bar.set_postfix(
                    loss=f"{loss_full:.3f}",
                    gn=f"{grad_norm_v:.2f}",
                    lr=f"{cur_lr:.2e}",
                    toks_s=f"{tok_per_sec:.0f}",
                    nmm=f"{nmm_mean:.2f}" if valid_norms else "nan",
                    refresh=False,
                )
                if step % log_every == 0:
                    # tqdm.write goes around the bar instead of through it.
                    tqdm.write(
                        f"step={step} loss={loss_full:.4f} "
                        f"grad_norm={grad_norm_v:.4f} lr={cur_lr:.3e} "
                        f"tok/s={tok_per_sec:.0f} "
                        f"nmm_norms={nmm_norms}"
                    )

            # Reset per-step accumulators.
            tokens_since_last_step = 0
            step_t0 = time.perf_counter()

            # Periodic save + final-step save. `step > 0` skips the spurious
            # save_every=N firing at step 0. The final-step save guarantees
            # the last training state is on disk even when max_steps isn't a
            # multiple of save_every.
            is_final_step = (step + 1 >= max_steps)
            save_due = (
                save_every is not None
                and save_dir is not None
                and (
                    (step > 0 and step % save_every == 0)
                    or (is_final_step and config is not None)
                )
            )
            if save_due:
                # rank 0 owns the write; all ranks barrier afterwards so
                # the non-rank-0 processes don't race into the next iteration
                # while rank 0 is still flushing to disk. Without the barrier,
                # rank 0 falls behind on the next all-reduce and the timeout
                # eventually fires on large checkpoints (1.5B-XL ~ 6 GB).
                if rank == 0:
                    saved_path = save_checkpoint_rotating(
                        save_dir, model, optimizer, step, config,
                        keep_last_n=keep_last_n,
                        training_args=training_args,
                    )
                    if show_progress:
                        tqdm.write(f"  saved {saved_path}")
                if is_distributed:
                    import torch.distributed as dist
                    dist.barrier()

            step += 1
            if not cycle_completed:
                # Single-GPU partial cycle: the loader exhausted mid-cycle.
                # We've already stepped on the partial gradients (safe on
                # single-GPU; DDP returned earlier at line ~587). For
                # max_steps > one_epoch we want training to continue into
                # the next epoch — the clean-boundary branch above will
                # rebuild `micro_batches` on the next outer iteration. Resetting
                # nmm_states here matches that branch's semantics (a fresh
                # pass through the corpus is a fresh context). The original
                # `break` here truncated training to one epoch silently —
                # exactly the "silently early-stop" failure mode this
                # function's docstring warns against. (Bug observed on
                # vanilla GPT-2 SQuAD training: max_steps=5000 stopped at
                # step ~998 with no checkpoint written, because save_every
                # hadn't fired yet.)
                if is_distributed:
                    # DDP can't safely restart mid-cycle — the asymmetric
                    # per-rank partial micro-batch count would diverge ranks
                    # on the next iter. (Belt-and-suspenders: we already
                    # returned in the DDP branch above.)
                    break
                micro_batches = iter(loader)
                nmm_states = None
    finally:
        bar.close()


def _detach_states(states):
    """Local alias for detach_states to avoid the top-level circular import."""
    from model.nmm import detach_states
    return detach_states(states)


# ---------------------------------------------------------------------------
# 4.5 — From-scratch entry point (CLI)
# ---------------------------------------------------------------------------

def main():
    """From-scratch training entry. Distributed via torchrun (sets LOCAL_RANK)."""
    import argparse
    import os

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from config import TitansConfig
    from data.dataloader import ParallelStreamLoader
    from data.tokenizer import Tokenizer
    from model.titans_gpt2 import TitansMAGGPT2

    # enable TF32 for fp32 matmul. Most of the model runs under
    # bf16 autocast (attention, MLP, NMM Q/K/V projections); the remaining
    # fp32 matmuls live in Newton-Schulz 5 (which opts out of autocast for
    # the fixed-point guarantee) and the analytical-grad LayerNorm
    # internals. TF32 keeps the iteration variable in fp32 and only
    # truncates matmul inputs from 23-bit to 10-bit mantissa — much less
    # aggressive than bf16 throughout, and preserves NS5's spectral-norm
    # convergence to ~1. ~5-15% step-time win depending on path. This is
    # a process-global setting; placed here so it covers training and
    # composes with any later eval / generate calls in the same process.
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--size", default="small",
                        choices=["small", "medium", "large", "xl"])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--save-dir",
        default="ckpts/from_scratch",
        help="Directory for rotated checkpoints (step_NNNNNNN.pt + latest.pt).",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help="Retain the most recent N step checkpoints; older ones are deleted. "
             "Pass 0 or a negative value to disable pruning.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap the full model in torch.compile after construction. Traces "
             "the entire forward (embedding + N transformer blocks + LN + LM "
             "head) into one Inductor graph per shape. Composes with "
             "nmm_compile_inner_loop (the inner compile is taken first; the "
             "outer compile then traces around it). Adds 1-3 minutes of "
             "warm-up compile time on the first training step. Should give "
             "10-20%% throughput on top of the inner-loop compile alone.",
    )
    parser.add_argument(
        "--optim8bit",
        action="store_true",
        help="Use bitsandbytes' 8-bit AdamW for optimizer state. Cuts "
             "optimizer memory ~4x (8-byte fp32 moments -> 2-byte 8-bit "
             "moments). Requires `bitsandbytes` package; install via "
             "`pip install bitsandbytes`. Trained quality is empirically "
             "close to fp32 AdamW; small drift possible at long horizons.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a saved checkpoint (step_*.pt or latest.pt). "
             "Restores model weights, optimizer state, and step counter so the "
             "LR schedule resumes at the right position in warmup-then-cosine. "
             "Architecture-affecting flags (--size, --chunk-size, --nmm-*) are "
             "IGNORED in resume mode — the checkpoint's saved config is "
             "authoritative. Training scaffolding (--max-steps, --save-dir, "
             "--save-every, --warmup-steps, --grad-accum) remains user-"
             "controlled. DDP-aware: each rank loads from the same checkpoint.",
    )
    from cli.nmm_cli import add_nmm_args, nmm_kwargs_from_args
    add_nmm_args(parser)
    args = parser.parse_args()

    # Device selection — LOCAL_RANK aware under torchrun.
    is_distributed = "LOCAL_RANK" in os.environ and torch.cuda.is_available()
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        rank, world_size = 0, 1
    else:
        device = torch.device("cpu")
        rank, world_size = 0, 1

    try:
        # Resume vs fresh. Resume: load config + state from checkpoint and
        # apply backend overrides via the shared helper so train.py and
        # cli/finetune.py have matching behavior. Architecture-affecting
        # CLI flags are ignored with a rank-0 warning; backend-only flags
        # listed in RESUME_OVERRIDABLE_BACKEND_FLAGS are applied.
        if args.resume_from is not None:
            ckpt = load_checkpoint(args.resume_from, device=device)
            if "config" not in ckpt:
                raise SystemExit(
                    f"Checkpoint {args.resume_from} lacks a 'config' key."
                )
            config = TitansConfig.from_dict(ckpt["config"])
            apply_resume_overrides_and_warn(
                config, args, nmm_kwargs_from_args(args),
                log_prefix="[train]", rank=rank,
            )
            warn_training_arg_drift(
                ckpt, args, log_prefix="[train]", rank=rank,
            )
        else:
            # Config BEFORE loader (loader reads chunk_size) —.
            factory = {
                "small": TitansConfig.gpt2_small,
                "medium": TitansConfig.gpt2_medium,
                "large": TitansConfig.gpt2_large,
                "xl": TitansConfig.gpt2_xl,
            }[args.size]
            config = factory(
                finetune_mode=False,
                chunk_size=args.chunk_size,
                block_size=args.chunk_size,  # match so all wpe positions train
                **nmm_kwargs_from_args(args),
            )

        # Seed BEFORE model so all ranks build identical params, then re-seed
        # PER RANK so dropout masks diverge.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = TitansMAGGPT2(config).to(device)

        # Per-rank seed AFTER model construction.
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)

        # full-forward torch.compile. Wrap BEFORE DDP so the
        # compiled graph sees per-rank model without the DDP comm hooks
        # threaded in. The `_unwrap` helper already strips both
        # `_orig_mod.` (compile) and `module.` (DDP) prefixes so
        # checkpoint save/load survives. mode="default" — same rationale
        # as the inner-loop compile: dynamic shapes from per-
        # chunk dict rebuilds make "reduce-overhead" + cudagraphs unsafe.
        if args.compile_model:
            model = torch.compile(model, mode="default", dynamic=False)

        if is_distributed:
            model = DDP(model, device_ids=[local_rank])

        # Load checkpoint weights AFTER compile + DDP wrap (matching the
        # established save-side _unwrap pattern: _unwrap on save strips the
        # prefixes, so we load into the inner module here too). Optimizer
        # construction must follow because bnb 8-bit AdamW reads the
        # current param tensors at __init__.
        if args.resume_from is not None:
            from model import _unwrap
            state = ckpt.get("state_dict", ckpt.get("model"))
            if state is None:
                raise SystemExit(
                    f"Checkpoint {args.resume_from} has neither 'state_dict' "
                    f"nor 'model' keys."
                )
            _unwrap(model).load_state_dict(_unwrap(state))

        optimizer = build_optimizer(model, use_8bit=args.optim8bit)

        if args.resume_from is not None:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            # The saved `step` is the value of the counter at the save call
            # site, BEFORE the `step += 1` increment — so `step=N` records
            # model state after N+1 completed cycles. Resume must add 1 to
            # avoid re-doing the saved cycle.
            start_step = int(ckpt.get("step", -1)) + 1
            saved_step_for_log = ckpt.get("step")
            # Drop the checkpoint dict — its state_dict and optimizer-state
            # tensors sit on GPU otherwise (~3 GiB at gpt2_small). Holding
            # them past load time would double-up with the freshly-instantiated
            # `model` + `optimizer` and push the first compile pass over the
            # VRAM budget. `load_state_dict` copied (not aliased) the data we
            # need, so this is safe.
            del ckpt, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if rank == 0:
                print(
                    f"[train] resumed from {args.resume_from} "
                    f"(saved at step {saved_step_for_log}, "
                    f"continuing from cycle {start_step})",
                    flush=True,
                )
        else:
            start_step = 0

        # Tokenize corpus. `read_eot_separated_documents` splits the file on
        # any literal `<|endoftext|>` markers so `encode_corpus` can append
        # the EOT *id* (50256) between them — the signal the dataloader uses
        # to set `doc_boundaries[i]=True` and reset the NMM state. Without
        # the split step, the literals BPE-tokenize as 7 ordinary tokens
        # and the reset never fires, leaving the NMM in unbounded-
        # accumulation mode across the whole corpus. For files with no
        # markers the helper returns a single-element list (matches the
        # previous whole-file-as-one-document behavior).
        from scripts.prepare_squad_corpus import read_eot_separated_documents
        tok = Tokenizer()
        token_stream = tok.encode_corpus(read_eot_separated_documents(args.data))

        loader = ParallelStreamLoader(
            token_stream,
            batch_size=args.batch_size,
            chunk_size=config.chunk_size,
            eot_id=tok.eot_token,
            rank=rank,
            world_size=world_size,
        )

        autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

        run_training(
            model=model,
            optimizer=optimizer,
            loader=loader,
            device=device,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            accum_steps=args.grad_accum,
            log_every=args.log_every,
            save_every=args.save_every,
            save_dir=args.save_dir,
            keep_last_n=args.keep_last_n,
            config=config,
            autocast_dtype=autocast_dtype,
            rank=rank,
            is_distributed=is_distributed,
            start_step=start_step,
            batch_size=args.batch_size,
        )
    finally:
        #/— NCCL cleanup on exception path. Consistent 4-space indent.
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
