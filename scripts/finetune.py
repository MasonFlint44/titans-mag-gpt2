"""Fine-tune entry point: load pretrained GPT-2, splice in NMM, run train loop.

Phase 4.4: builds a TitansMAGGPT2 with finetune_mode=True from a TitansConfig
factory, calls load_pretrained to overwrite backbone weights with HF GPT-2,
then runs the standard training loop. NMM contributes from y_mem=0 at step 0
(out_scale init zeros) so the model produces vanilla GPT-2 logits before
training starts.
"""

import argparse

import torch

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from data.tokenizer import Tokenizer
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained
from train import build_optimizer, run_training


_FACTORY = {
    "small": TitansConfig.gpt2_small,
    "medium": TitansConfig.gpt2_medium,
    "large": TitansConfig.gpt2_large,
    "xl": TitansConfig.gpt2_xl,
}


def build_finetune_config(size: str, **overrides):
    """Configs default to finetune_mode=True; callers may override anything."""
    return _FACTORY[size](finetune_mode=True, **overrides)


def build_parser() -> argparse.ArgumentParser:
    """Construct the finetune CLI parser. Extracted from main() so tests
    can inspect flags + defaults without invoking the training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=list(_FACTORY))
    parser.add_argument("--data", required=True, help="path to a text corpus")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--save-dir",
        default="ckpts/finetune",
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
    # Mirror train.py G277 / G278 so the consumer-GPU recipe documented in
    # README.md actually works against finetune.py (not only train.py).
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap the full model in torch.compile after construction (G277). "
             "Traces the entire forward (embedding + N transformer blocks + LN "
             "+ LM head) into one Inductor graph per shape. Composes with "
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
             "`pip install bitsandbytes` (or `uv sync --extra optim8bit`). "
             "Trained quality is empirically close to fp32 AdamW; small drift "
             "possible at long horizons.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a saved checkpoint (step_*.pt or latest.pt). "
             "Restores model weights, optimizer state (including 8-bit AdamW "
             "moments), and step counter. Architecture-affecting CLI flags "
             "(--size, --chunk-size, --nmm-*) are IGNORED in resume mode — the "
             "checkpoint's saved config is authoritative. Training scaffolding "
             "flags (--max-steps, --save-dir, --save-every, --warmup-steps, "
             "--grad-accum) remain user-controlled so you can extend a run, "
             "redirect saves, etc.",
    )
    from scripts._nmm_cli import add_nmm_args
    add_nmm_args(parser)
    return parser


def main():
    import sys
    from pathlib import Path
    from scripts._nmm_cli import nmm_kwargs_from_args
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    # Resume vs fresh: in resume mode, the checkpoint's saved config is
    # authoritative. Architecture-affecting CLI flags are ignored — pinning
    # them at the checkpoint's values prevents the user from accidentally
    # changing the model shape mid-run (which would silently corrupt
    # `optimizer.load_state_dict`).
    if args.resume_from is not None:
        from train import load_checkpoint
        from model import _unwrap
        ckpt = load_checkpoint(args.resume_from, device=device)
        if "config" not in ckpt:
            raise SystemExit(
                f"Checkpoint {args.resume_from} lacks a 'config' key; cannot "
                f"reconstruct the model architecture. Use a checkpoint saved "
                f"by save_checkpoint from train.py."
            )
        config = TitansConfig.from_dict(ckpt["config"])
        # Warn loudly if architecture-affecting flags were passed — silently
        # ignoring them would mask a configuration bug.
        nmm_kwargs = nmm_kwargs_from_args(args)
        # Backend toggles that don't change parameter shapes / graph
        # topology — safe to override on resume.  These flags select the
        # NS5 *implementation* (which polynomial/coefficients/kernels)
        # without changing any saved weight or optimizer state.  Useful
        # when a chosen backend turns out to OOM or break under newer
        # library versions and we want to fall back without re-training.
        _RESUME_OVERRIDABLE = {
            "nmm_use_gram_ns5",
            "nmm_use_cans",
            "nmm_ns5_steps",
        }
        overrides = {k: v for k, v in nmm_kwargs.items() if k in _RESUME_OVERRIDABLE}
        for k, v in overrides.items():
            setattr(config, k, v)
        ignored = {k: v for k, v in nmm_kwargs.items() if k not in _RESUME_OVERRIDABLE}
        # `--size` defaults to "small" but might mismatch the checkpoint;
        # only flag it if it actually disagrees.
        ckpt_size = _config_size_label(config)
        if args.size != ckpt_size and args.size != "small":
            ignored["size"] = args.size
        # chunk_size in the checkpoint config IS the authoritative chunk
        # size; warn if the user tried to change it.
        if args.chunk_size != config.chunk_size and args.chunk_size != 512:
            ignored["chunk_size"] = args.chunk_size
        if overrides:
            print(
                f"[finetune] --resume-from: applying non-architecture "
                f"overrides ({sorted(overrides.keys())}) to the saved config.",
                file=sys.stderr,
            )
        if ignored:
            print(
                f"[finetune] --resume-from: ignoring architecture-affecting "
                f"CLI flags ({sorted(ignored.keys())}) — checkpoint's saved "
                f"config is authoritative. Drop them from the command line "
                f"to silence this warning.",
                file=sys.stderr,
            )
        model = TitansMAGGPT2(config).to(device)
        # Skip load_pretrained — the checkpoint already has trained weights.
        # G277 — wrap with torch.compile BEFORE load_state_dict so the
        # `_orig_mod.` prefix is in place before we load (or alternatively,
        # use _unwrap on the saved dict). We do the latter — matches how
        # other consumers (eval_qa_recall, generate.py) handle this.
        if args.compile_model:
            model = torch.compile(model, mode="default", dynamic=False)
        state = ckpt.get("state_dict", ckpt.get("model"))
        if state is None:
            raise SystemExit(
                f"Checkpoint {args.resume_from} has neither 'state_dict' nor "
                f"'model' keys. Use a checkpoint saved by save_checkpoint."
            )
        # `_unwrap` strips any `_orig_mod.` (torch.compile) or `module.` (DDP)
        # prefixes the checkpoint may have. The freshly-wrapped model above
        # uses `_orig_mod.` again if compile_model is set, but
        # `model.load_state_dict` accepts the unprefixed keys via standard
        # PyTorch resolution.
        if args.compile_model:
            _unwrap(model).load_state_dict(_unwrap(state))
        else:
            model.load_state_dict(_unwrap(state))
        optimizer = build_optimizer(model, use_8bit=args.optim8bit)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(
                f"[finetune] --resume-from: checkpoint has no 'optimizer' key, "
                f"continuing with fresh optimizer state (m, v moments reset "
                f"to zero). Loss curve may briefly spike before the optimizer "
                f"re-equilibrates.",
                file=sys.stderr,
            )
        # The saved `step` field is the value of the step counter at the
        # save call site, which lives BETWEEN `optimizer.step()` and the
        # `step += 1` increment. So a checkpoint with `step=N` was actually
        # saved AFTER `N+1` completed cycles. To resume without re-doing
        # the most recent cycle, set `start_step = N + 1`.
        start_step = int(ckpt.get("step", -1)) + 1
        print(
            f"[finetune] resumed from {args.resume_from} "
            f"(saved at step {ckpt.get('step')}, "
            f"continuing from cycle {start_step})",
            file=sys.stderr,
        )
    else:
        config = build_finetune_config(
            args.size,
            chunk_size=args.chunk_size,
            **nmm_kwargs_from_args(args),
        )
        model = TitansMAGGPT2(config).to(device)
        load_pretrained(model, config)

        # G277 — full-forward torch.compile. Wrap AFTER load_pretrained (so
        # the compile sees post-pretrained-init weights, not the random init)
        # and BEFORE build_optimizer (so param groups are derived from the
        # unwrapped model's named_parameters() — _unwrap strips both
        # `_orig_mod.` compile prefixes and any DDP `module.` prefixes at
        # save/load time).
        if args.compile_model:
            model = torch.compile(model, mode="default", dynamic=False)

        optimizer = build_optimizer(model, use_8bit=args.optim8bit)
        start_step = 0

    # Tokenize corpus (whole-file-as-one-document; users can swap in an HF
    # streaming reader for FineWebEdu-scale runs).
    tok = Tokenizer()
    with open(args.data, "r", encoding="utf-8") as f:
        token_stream = tok.encode_corpus([f.read()])

    loader = ParallelStreamLoader(
        token_stream,
        batch_size=args.batch_size,
        chunk_size=config.chunk_size,
        eot_id=tok.eot_token,
    )

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
        start_step=start_step,
    )


def _config_size_label(config: TitansConfig) -> str:
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


if __name__ == "__main__":
    main()
