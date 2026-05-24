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
    save_path: str = None,
    config=None,
    autocast_dtype: torch.dtype = None,
    rank: int = 0,
    is_distributed: bool = False,
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
    """
    import contextlib

    base_lrs = base_lrs_from_constants()
    model.train()
    nmm_states = None

    step = 0
    micro_batches = iter(loader)
    while step < max_steps:
        cycle_ran_any_microbatch = False
        cycle_completed = True

        for accum_i in range(accum_steps):
            try:
                batch = next(micro_batches)
            except StopIteration:
                batch = None

            if batch is None and accum_i == 0:
                # Loader exhausted exactly at cycle boundary; clean stop.
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
        apply_lr(
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

        if rank == 0 and step % log_every == 0:
            nmm_norms = compute_nmm_norm(nmm_states)
            print(
                f"step={step} loss={loss.item() * accum_steps:.4f} "
                f"grad_norm={grad_norm.item():.4f} nmm_norms={nmm_norms}"
            )

        if (
            rank == 0
            and save_every is not None
            and save_path is not None
            and step > 0
            and step % save_every == 0
        ):
            save_checkpoint(save_path, model, optimizer, step, config)

        step += 1
        if not cycle_completed:
            # Single-GPU partial cycle exhausted the loader; stop.
            break


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
    parser.add_argument("--save-path", default="ckpts/from_scratch.pt")
    parser.add_argument("--seed", type=int, default=42)
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

        if is_distributed:
            model = DDP(model, device_ids=[local_rank])

        optimizer = build_optimizer(model)

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
            save_path=args.save_path,
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
