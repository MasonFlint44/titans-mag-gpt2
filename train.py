"""Consolidated training driver (fine-tune and from-scratch entry points)."""

import dataclasses
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 4.1 — 4-group optimizer
# ---------------------------------------------------------------------------

# Code-level constants so apply_lr's base_lrs cannot drift on resume (G162).
BASE_LR_GPT2 = 3e-4
BASE_LR_NMM = 9e-4  # 3x GPT-2 per paper
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)  # NOT PyTorch default (0.9, 0.999); G153
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0

# Substring set for no-decay routing. Catches:
#   bias       -- bias params on Linear/Conv
#   ln, norm   -- LayerNorm and ResidualNorm params
#   out_scale  -- magnitude gate (init=0; decay would resist learning)
#   gamma      -- gamma_mem / gamma_attn (init=1; decay shrinks memory branch)
#   persistent -- persistent_mem (learned prefix; decay reduces capacity)
NO_DECAY_SUBSTRINGS = ("bias", "ln", "norm", "out_scale", "gamma", "persistent")

# Substring set for NMM routing. Catches every param that should get the NMM LR.
NMM_SUBSTRINGS = ("nmm", "gamma", "persistent", "ln_nmm")


def _is_no_decay(name: str) -> bool:
    return any(nd in name for nd in NO_DECAY_SUBSTRINGS)


def _is_nmm(name: str) -> bool:
    return any(k in name for k in NMM_SUBSTRINGS)


def build_optimizer(
    model: nn.Module,
    lr_gpt2: float = BASE_LR_GPT2,
    lr_nmm: float = BASE_LR_NMM,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple = BETAS,
    eps: float = ADAM_EPS,
    use_8bit: bool = False,
):
    """Build the 4-group AdamW (G278: optional 8-bit variant).

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
    caller's next call re-initializes (G158, G213). Returning the existing
    state would propagate a NaN-tainted M through the next forward and
    livelock until the next document boundary.

    `autocast_dtype=None` (the default) runs forward in fp32 — required for
    CPU tests. On CUDA, the entry points pass `torch.bfloat16` to enable the
    bf16-autocast forward + fp32 backward pattern (G159). Mixing CPU autocast
    with our torch.func.grad inner loop produces a mixed-dtype backward graph
    that fails with "expected BFloat16 but found Float" — autocast on CPU
    has narrower op coverage than CUDA, so we don't enable it there by default.

    Mixed precision: backward + clip + step always run outside autocast in
    fp32 (G159). clip_grad_norm_ inside autocast computes the norm in low
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

    Validates max_steps > warmup_steps (G197). max_steps == warmup_steps
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
    1:1:3:3 gpt2/nmm LR ratio across groups (G157) — naive single-group or
    uniform-clobber updates would either skip groups or destroy the ratio.

    `base_lrs` MUST come from code-level constants (G162), never from
    `optimizer.param_groups[i]['lr']` after `load_state_dict` — that path
    captures the mid-cosine deflated value and compounds the deflation
    every resume.

    `warmup_steps` and `max_steps` are required positional args (no
    defaults) so the caller cannot silently inherit the 1k/100k schedule
    when running a 200-step overfit (G175).
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
    nmm_no_decay). Resume-safe by construction (G162)."""
    return [BASE_LR_GPT2, BASE_LR_GPT2, BASE_LR_NMM, BASE_LR_NMM]


def _layer_norm_M(layer_state):
    """Compute ||M||_F (batch-mean) for a single per-layer state. Handles
    three shapes:
      - None: returns None. Plain (non-NMM) blocks have None state slots
        when `nmm_layer_indices` is set (G261).
      - single-head: `(M, S)` tuple of dicts.
      - multi-head: `[(M_h, S_h), ...]` list (G254) — returns per-head mean.
    """
    if layer_state is None:
        return None
    if isinstance(layer_state, list):
        return sum(_layer_norm_M(s) for s in layer_state) / len(layer_state)
    M, _S = layer_state
    sq_sum = sum((v.float() ** 2).sum(dim=(-2, -1)) for v in M.values())  # [B]
    return sq_sum.sqrt().mean().detach().item()


def compute_nmm_norm(nmm_states) -> list:
    """Per-layer ||M||_F (treating W1/W_gate/W2 as one block), averaged across
    batch. Returns None if states is None (first-step case, G172).

    Multi-head safe (G254): when `nmm_n_heads > 1`, the per-layer state is a
    list of per-head `(M, S)` tuples; we report the mean of per-head norms
    per layer (so the returned list has length n_layer regardless of head
    count — convenient for log parsers).

    Subset-of-layers safe (G261): plain (non-NMM) blocks contribute a None
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
) -> None:
    """Save model state_dict + optimizer state + step + config to a single file.

    config is required (not optional): finetune_mode controls block structure
    (gamma_attn presence, out_scale init), so resume needs it to rebuild the
    same model topology. Without saving it, the resume code can't tell which
    structure to construct, producing state_dict mismatch errors or silently-
    wrong inits.

    NMM states are intentionally NOT saved — they're per-sequence
    accumulators, not model state. Resume re-initializes from
    memory_mlp.W*.weight.
    """
    # G184/G186/G195 — _unwrap strips torch.compile and DDP/FSDP prefixes so
    # the saved state_dict is portable across wrapping choices on resume.
    from model import _unwrap

    torch.save(
        {
            "state_dict": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
        },
        path,
    )


def load_checkpoint(path, device: torch.device) -> dict:
    """Load checkpoint dict. weights_only=False is required (G168): PyTorch
    2.6+ flipped the default to True and would reject our nested optimizer
    state on some version combos.

    Returns the raw dict; caller rebuilds the model from `ckpt['config']`,
    then `model.load_state_dict(ckpt['state_dict'])`, then optimizer-
    construct + (optional) `optimizer.load_state_dict(ckpt['optimizer'])`.
    Missing 'optimizer' key (HF-init checkpoint from load_pretrained) is
    handled by the caller (G219).
    """
    return torch.load(path, map_location=device, weights_only=False)


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
    save_checkpoint(step_path, model, optimizer, step, config)

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
    """G222: the correct partial-cycle skip condition.

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
) -> None:
    """Top-level training loop covering both Phase 4.4 (fine-tune) and 4.5
    (from-scratch). One optimizer.step per accumulation cycle of `accum_steps`
    micro-batches.

    DDP: caller wraps `model` with DDP and passes `is_distributed=True`. All
    micro-batches except the last in a cycle run inside `model.no_sync()` to
    suppress per-microbatch all-reduce (G200). Partial cycle at corpus end
    is detected via G222's `(batch is None) and (accum_i > 0)` check and
    skipped to avoid rank divergence — the per-rank `.grad` buffers were
    never AllReduce'd. On single GPU, partial cycles step normally.

    NaN-skip: if accumulated grad_norm is non-finite, zero grads AND reset
    nmm_states to None (G217) so the next cycle's first micro-batch hits
    model.forward's None branch and re-inits — otherwise NaN-tainted M
    livelocks until the next document boundary.

    NCCL teardown (try/finally with destroy_process_group, G225/G227) lives
    in the entry-point script that wraps this call, not here — keeps this
    function single-purpose and reusable from notebooks / tests.

    Training continues until `step >= max_steps`. When the loader exhausts
    mid-training (max_steps > batches-per-epoch), the iterator is rebuilt
    and iteration continues — matching docs/PLAN.md §4.5's
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
    """
    import contextlib

    base_lrs = base_lrs_from_constants()
    model.train()
    nmm_states = None

    # Progress bar — only on rank 0, never under explicit show_progress=False.
    bar = tqdm(
        total=max_steps,
        desc="train",
        unit="step",
        disable=not (show_progress and rank == 0),
        dynamic_ncols=True,
        leave=True,
    )

    step = 0
    micro_batches = iter(loader)
    tokens_since_last_step = 0
    step_t0 = time.perf_counter()
    last_lr_mul = 0.0
    try:
        while step < max_steps:
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
                    # G214/G222: under DDP, this cycle's micro-batches ran with
                    # no_sync; per-rank .grad never AllReduce'd. Stepping would
                    # diverge ranks permanently. Discard + stop.
                    if is_distributed:
                        optimizer.zero_grad(set_to_none=True)
                        nmm_states = None  # G217 — match NaN-skip semantics
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
                # G217 — match the train_step NaN-skip pattern.
                nmm_states = None
            optimizer.zero_grad(set_to_none=True)

            # Per-step metrics. `loss * accum_steps` undoes the per-microbatch
            # division so reported loss is comparable across accum_steps choices.
            step_dt = max(time.perf_counter() - step_t0, 1e-9)
            tok_per_sec = tokens_since_last_step / step_dt
            loss_full = loss.item() * accum_steps
            grad_norm_v = grad_norm.item()
            nmm_norms = compute_nmm_norm(nmm_states)
            # `compute_nmm_norm` returns None when nmm_states is None (G172,
            # first-step case) AND emits per-layer Nones for plain blocks
            # (G261). Under --vanilla-gpt2 every block is plain, so nmm_norms
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
                # G199: rank 0 owns the write; all ranks barrier afterwards so
                # the non-rank-0 processes don't race into the next iteration
                # while rank 0 is still flushing to disk. Without the barrier,
                # rank 0 falls behind on the next all-reduce and the timeout
                # eventually fires on large checkpoints (1.5B-XL ~ 6 GB).
                if rank == 0:
                    saved_path = save_checkpoint_rotating(
                        save_dir, model, optimizer, step, config,
                        keep_last_n=keep_last_n,
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

    # G280: enable TF32 for fp32 matmul. Most of the model runs under
    # bf16 autocast (attention, MLP, NMM Q/K/V projections); the remaining
    # fp32 matmuls live in Newton-Schulz 5 (which opts out of autocast for
    # the G226 fixed-point guarantee) and the analytical-grad LayerNorm
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
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
        help="Use bitsandbytes' 8-bit AdamW for optimizer state (G278). Cuts "
             "optimizer memory ~4x (8-byte fp32 moments -> 2-byte 8-bit "
             "moments). Requires `bitsandbytes` package; install via "
             "`pip install bitsandbytes`. Trained quality is empirically "
             "close to fp32 AdamW; small drift possible at long horizons.",
    )
    from scripts._nmm_cli import add_nmm_args, nmm_kwargs_from_args
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
        # Config BEFORE loader (loader reads chunk_size) — G205.
        factory = {
            "small": TitansConfig.gpt2_small,
            "medium": TitansConfig.gpt2_medium,
            "large": TitansConfig.gpt2_large,
            "xl": TitansConfig.gpt2_xl,
        }[args.size]
        config = factory(
            finetune_mode=False,
            chunk_size=args.chunk_size,
            block_size=args.chunk_size,  # match so all wpe positions train (G163)
            **nmm_kwargs_from_args(args),
        )

        # Seed BEFORE model so all ranks build identical params, then re-seed
        # PER RANK so dropout masks diverge (G204).
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = TitansMAGGPT2(config).to(device)

        # Per-rank seed AFTER model construction.
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)

        # G277 — full-forward torch.compile. Wrap BEFORE DDP so the
        # compiled graph sees per-rank model without the DDP comm hooks
        # threaded in. The `_unwrap` helper already strips both
        # `_orig_mod.` (compile) and `module.` (DDP) prefixes so
        # checkpoint save/load survives. mode="default" — same rationale
        # as the inner-loop compile (G264a): dynamic shapes from per-
        # chunk dict rebuilds make "reduce-overhead" + cudagraphs unsafe.
        if args.compile_model:
            model = torch.compile(model, mode="default", dynamic=False)

        if is_distributed:
            model = DDP(model, device_ids=[local_rank])

        optimizer = build_optimizer(model, use_8bit=args.optim8bit)

        tok = Tokenizer()
        with open(args.data, "r", encoding="utf-8") as f:
            token_stream = tok.encode_corpus([f.read()])

        loader = ParallelStreamLoader(
            token_stream,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
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
        )
    finally:
        # G225/G227 — NCCL cleanup on exception path. Consistent 4-space indent.
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
