"""Consolidated training driver (fine-tune and from-scratch entry points)."""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


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
) -> AdamW:
    """Build the 4-group AdamW.

    Groups: (gpt2_decay, gpt2_no_decay, nmm_decay, nmm_no_decay).
    NMM groups get a higher LR (paper uses 3x); no_decay groups have wd=0.

    Every param must land in exactly one group. Verified by checking sum-of-
    group-sizes equals the model's total param count.
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


def compute_nmm_norm(nmm_states) -> list:
    """Per-layer ||M||_F (treating W1/W_gate/W2 as one block), averaged across
    batch. Returns None if states is None (first-step case, G172)."""
    if nmm_states is None:
        return None
    out = []
    for M, _S in nmm_states:
        sq_sum = sum((v.float() ** 2).sum(dim=(-2, -1)) for v in M.values())  # [B]
        out.append(sq_sum.sqrt().mean().detach().item())
    return out


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
    torch.save(
        {
            "state_dict": model.state_dict(),
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
