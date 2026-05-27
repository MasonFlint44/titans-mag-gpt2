"""Profile the TITANS fine-tune recipe with torch.profiler.

Runs a small number of optimizer steps under the same configuration as the
recommended consumer-GPU recipe, captures a per-op CUDA-time summary, and
optionally writes a Chrome trace.

Usage:
    uv run python -m scripts.profile_finetune \\
        --data corpora/squad/squad_train.txt \\
        --out profiles/titans

The schedule is:
  wait=1   — let the first optimizer step run uninstrumented
  warmup=1 — absorb the torch.compile / inductor first-call cost
  active=1 — profile this one step  (default; pass --n-active-steps N for more)

Outputs:
  profiles/titans/key_averages_cuda.txt  — sorted by CUDA self time
  profiles/titans/key_averages_cpu.txt   — sorted by CPU self time
  profiles/titans/trace.json             — only when --chrome-trace is passed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.profiler import (
    profile, schedule, ProfilerActivity, tensorboard_trace_handler,
)

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from data.tokenizer import Tokenizer
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained
from train import build_optimizer


# Same recipe as docs/QA_RECALL_PLAN.md / docs/RUNBOOK.md consumer-GPU defaults.
RECIPE_KWARGS = dict(
    chunk_size=1024,
    nmm_block_size=64,
    nmm_state_dtype="bf16",
    nmm_detach_state_between_blocks=True,
    nmm_use_cans=True,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", type=Path, required=True,
        help="Path to the training corpus (same file you'd pass to "
             "scripts/finetune.py).",
    )
    p.add_argument(
        "--out", type=Path, default=Path("profiles/titans"),
        help="Output directory for trace.json + summary files.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Micro-batch size (recipe default: 1).",
    )
    p.add_argument(
        "--grad-accum", type=int, default=16,
        help="Gradient accumulation steps per optimizer step (recipe: 16).",
    )
    p.add_argument(
        "--compile-model", action="store_true", default=True,
        help="Wrap the model in torch.compile (recipe default: True). Pass "
             "--no-compile-model to disable.",
    )
    p.add_argument("--no-compile-model", dest="compile_model",
                   action="store_false")
    p.add_argument(
        "--optim8bit", action="store_true", default=True,
        help="Use bnb.AdamW8bit (recipe default: True).",
    )
    p.add_argument("--no-optim8bit", dest="optim8bit", action="store_false")
    p.add_argument(
        "--n-active-steps", type=int, default=1,
        help="Number of optimizer steps to profile (default: 1). Increase "
             "for better kernel averages at the cost of more RAM.",
    )
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="Steps absorbed by warmup (torch.compile traces "
                        "here). Default 1.")
    p.add_argument("--wait-steps", type=int, default=1,
                   help="Steps run before the profiler arms (default 1).")
    p.add_argument(
        "--chrome-trace", action="store_true", default=False,
        help="Export a chrome://tracing JSON trace (can be very large; "
             "disabled by default to avoid OOM).",
    )
    p.add_argument(
        "--record-shapes", action="store_true", default=False,
        help="Record tensor shapes in the trace (adds RAM; off by default).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (matches finetune.py default).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Profiler requires CUDA — this is a GPU benchmark.")

    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16

    print(f"[profile] building model + optimizer (recipe={RECIPE_KWARGS})",
          file=sys.stderr)
    config = TitansConfig.gpt2_small(
        finetune_mode=True,
        **RECIPE_KWARGS,
    )
    model = TitansMAGGPT2(config).to(device)
    load_pretrained(model, config)
    if args.compile_model:
        model = torch.compile(model, mode="default", dynamic=False)
    optimizer = build_optimizer(model, use_8bit=args.optim8bit)

    tok = Tokenizer()
    with open(args.data, "r", encoding="utf-8") as f:
        token_stream = tok.encode_corpus([f.read()])
    loader = ParallelStreamLoader(
        token_stream,
        batch_size=args.batch_size,
        chunk_size=config.chunk_size,
        eot_id=tok.eot_token,
    )

    total_steps = args.wait_steps + args.warmup_steps + args.n_active_steps
    print(
        f"[profile] total optimizer steps to run: {total_steps} "
        f"(wait={args.wait_steps}, warmup={args.warmup_steps}, "
        f"active={args.n_active_steps})", file=sys.stderr,
    )

    # Inline a tight version of run_training so we can call prof.step() at
    # each optimizer-step boundary. We don't need save logic, LR scheduling,
    # or checkpoint rotation for the profile — just the hot path.
    import torch.nn.functional as F
    from train import GRAD_CLIP, BASE_LR_GPT2, BASE_LR_NMM, apply_lr
    from train import base_lrs_from_constants

    base_lrs = base_lrs_from_constants()
    model.train()
    nmm_states = None
    micro_batches = iter(loader)

    prof_sched = schedule(
        wait=args.wait_steps,
        warmup=args.warmup_steps,
        active=args.n_active_steps,
        repeat=1,
    )

    def _on_ready(prof_obj):
        if args.chrome_trace:
            trace_path = args.out / "trace.json"
            prof_obj.export_chrome_trace(str(trace_path))
            print(f"[profile] wrote {trace_path}", file=sys.stderr)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=_on_ready,
        record_shapes=args.record_shapes,
        profile_memory=False,  # True buffers every alloc event in RAM → OOM
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        for step in range(total_steps):
            cycle_t0 = time.perf_counter()
            for accum_i in range(args.grad_accum):
                try:
                    batch = next(micro_batches)
                except StopIteration:
                    micro_batches = iter(loader)
                    nmm_states = None
                    batch = next(micro_batches)
                input_ids, doc_boundaries = batch
                input_ids = input_ids.to(device, non_blocking=True)
                doc_boundaries = doc_boundaries.to(device, non_blocking=True)
                from model.nmm import detach_states
                nmm_states = detach_states(nmm_states) if nmm_states is not None else None

                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits, nmm_states = model(input_ids, nmm_states, doc_boundaries)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                    ) / args.grad_accum
                loss.backward()

            apply_lr(
                optimizer, base_lrs, step,
                warmup_steps=args.warmup_steps, max_steps=total_steps + 100,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                nmm_states = None
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            dt = time.perf_counter() - cycle_t0
            phase = (
                "wait" if step < args.wait_steps else
                "warmup" if step < args.wait_steps + args.warmup_steps else
                "active"
            )
            print(
                f"[profile] step {step}/{total_steps-1} "
                f"phase={phase} loss={loss.item() * args.grad_accum:.3f} "
                f"gn={grad_norm.item():.2f} dt={dt:.2f}s",
                file=sys.stderr, flush=True,
            )
            prof.step()

        total_dt = time.perf_counter() - t0
        print(f"[profile] total time: {total_dt:.1f}s", file=sys.stderr)

    # Summary tables. prof.key_averages() returns aggregated per-op stats
    # across the active window only.
    print(f"[profile] writing summary tables to {args.out}/", file=sys.stderr)
    key_avg = prof.key_averages()
    by_cuda = key_avg.table(sort_by="self_cuda_time_total", row_limit=40)
    by_cpu = key_avg.table(sort_by="self_cpu_time_total", row_limit=40)

    (args.out / "key_averages_cuda.txt").write_text(by_cuda)
    (args.out / "key_averages_cpu.txt").write_text(by_cpu)

    # Brief stdout summary for the user.
    print("\n[profile] === Top 15 ops by CUDA self time ===", file=sys.stderr)
    print(key_avg.table(sort_by="self_cuda_time_total", row_limit=15),
          file=sys.stderr)


if __name__ == "__main__":
    main()
