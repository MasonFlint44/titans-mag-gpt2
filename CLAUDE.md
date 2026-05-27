# Working in this repo

A from-scratch implementation of TITANS MAG on top of GPT-2. The architecture
is documented in depth elsewhere — this file is for orienting *you* (the agent)
on conventions, commands, and traps when making changes.

## Read these first

- `SPEC.md` — authoritative description of what the code does. If code and SPEC
  disagree, that's a bug worth flagging.
- `docs/ARCHITECTURE.md` — equations and block diagram.
- `docs/CONFIG_REFERENCE.md` — every CLI / config knob, its range, defaults,
  and speed/memory tradeoff.
- `docs/RUNBOOK.md` — symptom → cause table for training failures.
- `docs/GAP_HISTORY.md` — chronological audit log of bugs found (gap IDs
  like `G226`, `G277` are referenced from code comments).

When the user mentions a `Gxxx` identifier, look it up in `GAP_HISTORY.md`.
Code comments cite these — keep that convention when adding new safeguards.

## Layout

| Path | Role |
|---|---|
| `model/nmm.py` | Neural Memory Module — inner gradient + NS5 normalization. The most complex file in the repo (~1900 lines). Three NS5 variants live here: stock, CANS-stationary, and gram-iteration (`gram_newton_schulz`). |
| `model/block.py` | MAG block: attention + NMM + learnable gate. |
| `model/titans_gpt2.py` | Top-level model wrapping HF GPT-2 backbone + MAG blocks. |
| `config.py` | `TitansConfig` dataclass — factory methods `gpt2_small/medium/large/xl`. Validation lives here too. |
| `train.py` | Multi-GPU training loop (DDP) + `run_training`, `build_optimizer`, `save_checkpoint`, `load_checkpoint`. |
| `scripts/finetune.py` | Single-GPU finetune entry point — wraps `train.py`'s loop with HF pretrained loading. |
| `scripts/_nmm_cli.py` | Shared `--nmm-*` argparse definitions for both `train.py` and `finetune.py`. |
| `data/` | Tokenizer (tiktoken GPT-2), streaming dataloader, doc-boundary tracking. |
| `tests/{unit,integration,parity,ddp,performance,behavior,failure_modes}/` | See `docs/TEST_PLAN.md` for tier definitions. |

## Commands

Always run inside the uv-managed venv. Don't activate it manually — use `uv run`.

```bash
uv sync                            # install / update deps from uv.lock
uv sync --extra optim8bit          # adds bitsandbytes (required for --optim8bit)

uv run pytest tests/unit/          # fast inner loop
uv run pytest -m "not slow and not gpu"   # local dev tier
uv run pytest tests/parity/        # HF GPT-2 logit/perplexity parity
uv run pytest tests/ddp/ -m ddp    # needs 2+ GPUs + torchrun
```

Pytest markers: `slow`, `gpu`, `slow_gpu`, `ddp`, `compile`, `perf`. See
`pyproject.toml` for definitions.

## The consumer-GPU finetune recipe

`finetune.py` and `train.py` share defaults that fit a 16 GiB card:
`--chunk-size 1024 --batch-size 1 --grad-accum 16 --max-steps 5000
--warmup-steps 500`. The full canonical command also passes a few `--nmm-*`
backend toggles and `--compile-model --optim8bit`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/finetune.py \
    --size small --data corpus.txt \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks \
    --nmm-use-gram-ns5 \
    --compile-model --optim8bit
```

For multi-GPU from-scratch runs on a bigger box you'll want to bump
`--batch-size`, drop `--grad-accum`, and raise `--max-steps`. The shared
defaults are sized for the consumer-GPU path; the bigger-box path is an
override.

## Resume flow gotchas

- Architecture-affecting flags (`--size`, `--chunk-size`, `--nmm-*` that change
  shapes) are **ignored** on `--resume-from` with a rank-0 warning. The
  checkpoint's saved config is authoritative; changing shape would silently
  break `optimizer.load_state_dict`.
- Backend-only flags listed in `train.RESUME_OVERRIDABLE_BACKEND_FLAGS`
  *can* be overridden on resume — currently `nmm_use_gram_ns5`, `nmm_use_cans`,
  `nmm_ns5_steps`. These swap the NS5 implementation without touching any
  saved weight. Add new backend-only NMM toggles to that frozenset; both
  `train.py` and `scripts/finetune.py` import from it.
- Scaffolding flags (`--max-steps`, `--save-dir`, `--grad-accum`,
  `--batch-size`, warmup, save-every) are **not** overridden — the user
  controls them. They're persisted in the checkpoint under `training_args`
  (see `SAVED_TRAINING_ARGS` in `train.py`), and on resume the loader warns
  if the CLI value disagrees with what was saved. Defaults match the
  documented recipe so a bare `--resume-from` is a no-warning resume.
- `train.py` and `scripts/finetune.py` share the resume helpers
  (`apply_resume_overrides_and_warn`, `warn_training_arg_drift`,
  `config_size_label`, `training_args_from_namespace`) so the two entry
  points are behaviorally identical on resume. Don't fork them.

## NS5 variants

Three implementations of the Newton-Schulz spectral normalization of the inner
NMM gradient, selected by mutually exclusive CLI flags:

| Variant | Flag | When to pick it |
|---|---|---|
| Stock NS5 | (default) | Paper-faithful; fp32 invariant (G226). |
| CANS-stationary | `--nmm-use-cans` | 3 steps, fp32, Chebyshev coefficients. Better orthogonalization, slower. |
| Gram-iteration | `--nmm-use-gram-ns5` | POLAR_EXPRESS coefficients + reset at iter 2. fp16 inner loop with `torch.baddbmm` fusion. Fastest of the three. |

All three live in `model/nmm.py`. The gram-iteration variant was reimplemented
locally (replacing an external dep) because the upstream library used an
in-place divide that broke AOT autograd under `torch.compile`. Don't add the
external `gram_ns5` package back as a dependency.

## Conventions

- Comments tag gap-driven safeguards by ID: `# G226 — fp32 NS5 invariant`.
  Keep this format when adding new ones; cross-reference in `GAP_HISTORY.md`.
- `_unwrap(state_dict)` in `model/__init__.py` strips both `_orig_mod.`
  (torch.compile) and `module.` (DDP) prefixes. Always use it when loading
  checkpoints across compiled / DDP / plain runs.
- Inner-loop NS5 must run in fp32 *or* the dedicated fp16 path (gram only).
  Letting bf16 autocast leak into NS5 silently produces NaN loss after a few
  steps (`docs/RUNBOOK.md §NaN loss`).
- `finetune_mode=True` (default for `scripts/finetune.py`) uses an additive
  gate with `out_scale=0` so initial logits match HF GPT-2 exactly. `train.py`
  hard-codes `finetune_mode=False` for from-scratch with the paper's
  multiplicative gate.
- Checkpoints persist via `save_checkpoint` in `train.py`; the file format
  includes `config`, `state_dict`, `optimizer`, `step`. The saved `step` is
  the value *before* the post-step increment — `start_step = step + 1` on
  resume.

## Things that are not bugs

- `out_scale=0` at finetune init is deliberate (G279). Don't "fix" the zero.
- The NMM keeps updating during `generate.py` even under `torch.no_grad()` —
  `torch.func.grad` is independent of the no-grad context. This is the whole
  TITANS premise.
- `pyproject.toml` has no `[gram_ns5]` extra anymore — the external library
  was removed in favor of the local reimplementation. Don't re-add it.
