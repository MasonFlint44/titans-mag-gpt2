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

`docs/archive/` holds the original implementation plan and gap-history audit
log from the bootstrap phase. Read-only history — don't update or extend it.

## Layout

| Path | Role |
|---|---|
| `model/nmm.py` | Neural Memory Module — inner gradient + NS5 normalization. The most complex file in the repo (~1900 lines). Three NS5 variants live here: stock, CANS-stationary, and gram-iteration (`gram_newton_schulz`). |
| `model/block.py` | MAG block: attention + NMM + learnable gate. |
| `model/titans_gpt2.py` | Top-level model wrapping HF GPT-2 backbone + MAG blocks. |
| `model/state_io.py` | NMM state save/load for persistent generate sessions. |
| `config.py` | `TitansConfig` dataclass — factory methods `gpt2_small/medium/large/xl`. Validation lives here too. |
| `evaluation.py` | Perplexity + needle-in-haystack helpers (importable library). |
| `cli/train.py` | Multi-GPU training loop (DDP) + `run_training`, `build_optimizer`, `save_checkpoint`, `load_checkpoint`. |
| `cli/finetune.py` | Single-GPU finetune entry point — wraps `cli/train.py`'s loop with HF pretrained loading. |
| `cli/generate.py` | Cached autoregressive sampling CLI + interactive REPL. |
| `cli/nmm_cli.py` | Shared `--nmm-*` argparse definitions for both `cli/train.py` and `cli/finetune.py`. |
| `data/` | Tokenizer (tiktoken GPT-2), streaming dataloader, doc-boundary tracking. |
| `scripts/` | Experiment scripts: corpus prep, eval CLIs, benchmarks, profiling. NOT library code. |
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

> **Tests are slow while training is running.** Both share the local
> GPU; expect pytest wall times to balloon (and possibly timeout) until
> the training process exits. Either wait for training to finish, or
> run a small focused subset (e.g. `uv run pytest tests/unit/test_optimizer.py`)
> that doesn't need CUDA.

Pytest markers: `slow`, `gpu`, `slow_gpu`, `ddp`, `compile`, `perf`. See
`pyproject.toml` for definitions.

## The consumer-GPU finetune recipe

`cli/finetune.py` and `cli/train.py` share defaults that fit a 16 GiB card:
`--chunk-size 1024 --batch-size 1 --grad-accum 16 --max-steps 5000
--warmup-steps 500`. The full canonical command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m cli.finetune \
    --size small --data corpus.txt \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks \
    --nmm-use-gram-ns5 \
    --freeze-backbone \
    --nmm-gate-ramp-steps 100 \
    --nmm-gate-ramp-target 0.1 \
    --compile-model --optim8bit
```

The TPTT-inspired flags
([fabienfrfr/tptt](https://github.com/fabienfrfr/tptt)) at the bottom
are the difference from a plain LM fine-tune:

- `--freeze-backbone` — freeze every param outside the memory pathway
  (anything not matching one of `cli.train.FREEZE_BACKBONE_KEEP_TRAINABLE_SUBSTRINGS`
  = `("nmm", "gamma", "out_scale", "persistent")`). Halves the trainable
  parameter count at gpt2_small (251M → 128M). The backbone's pretrained
  representations are preserved exactly; the NMM has a stable target to
  integrate with instead of chasing a moving backbone.

- `--nmm-gate-ramp-steps N` + `--nmm-gate-ramp-target X` — during the
  first N training steps, hold every per-block `out_scale` to a linear
  ramp from `~0 → X`, with `requires_grad=False` so the optimizer doesn't
  fight the schedule. At step N, the optimizer takes over. Forces the
  memory gate open on a fixed schedule rather than relying on LM loss
  alone to slowly open it (TPTT's LiZACallback pattern).

Why these matter: without them, the LM-loss gradient is diluted across
~125M backbone params and the NMM's `out_scale` is left to passively
self-bootstrap from 0. A diagnostic run on needle-in-haystack found
this produced only ~0.45 logits of needle-dependence past block_size
— ~100× too weak to overcome the LM prior. The TPTT recipe is the
recommended way to give the memory mechanism a chance to actually
learn cross-chunk recall.

The argparse defaults leave both flags OFF (`--freeze-backbone` not
set, `--nmm-gate-ramp-steps=0`) so legacy scripts that didn't pass
them get full-fine-tune behavior unchanged. The canonical recipe
above is the new recommendation.

For multi-GPU from-scratch runs on a bigger box you'll want to bump
`--batch-size`, drop `--grad-accum`, raise `--max-steps`, and probably
drop `--freeze-backbone` (more data + more capacity makes the
gradient-dilution concern less acute).

## Resume flow gotchas

- Architecture-affecting flags (`--size`, `--chunk-size`, `--nmm-*` that change
  shapes) are **ignored** on `--resume-from` with a rank-0 warning. The
  checkpoint's saved config is authoritative; changing shape would silently
  break `optimizer.load_state_dict`.
- Backend-only flags listed in `cli.train.RESUME_OVERRIDABLE_BACKEND_FLAGS`
  *can* be overridden on resume — currently `nmm_use_gram_ns5`, `nmm_use_cans`,
  `nmm_ns5_steps`. These swap the NS5 implementation without touching any
  saved weight. Add new backend-only NMM toggles to that frozenset; both
  `cli/train.py` and `cli/finetune.py` import from it.
- Scaffolding flags (`--max-steps`, `--save-dir`, `--grad-accum`,
  `--batch-size`, warmup, save-every) are **not** overridden — the user
  controls them. They're persisted in the checkpoint under `training_args`
  (see `SAVED_TRAINING_ARGS` in `cli/train.py`), and on resume the loader
  warns if the CLI value disagrees with what was saved. Defaults match the
  documented recipe so a bare `--resume-from` is a no-warning resume.
- `cli/train.py` and `cli/finetune.py` share the resume helpers
  (`apply_resume_overrides_and_warn`, `warn_training_arg_drift`,
  `config_size_label`, `training_args_from_namespace`) so the two entry
  points are behaviorally identical on resume. Don't fork them.
- `TitansConfig.from_dict` (used by the resume loader) silently drops keys
  listed in `config._REMOVED_CONFIG_KEYS` so checkpoints from older schemas
  still load. When you delete a config field, append the name to that
  frozenset rather than breaking old checkpoints.

## NS5 variants

Three implementations of the Newton-Schulz spectral normalization of the inner
NMM gradient, selected by mutually exclusive CLI flags:

| Variant | Flag | When to pick it |
|---|---|---|
| Stock NS5 | (default) | Paper-faithful; fp32 invariant. |
| CANS-stationary | `--nmm-use-cans` | 3 steps, fp32, Chebyshev coefficients. Better orthogonalization, slower. |
| Gram-iteration | `--nmm-use-gram-ns5` | POLAR_EXPRESS coefficients + reset at iter 2. fp16 inner loop with `torch.baddbmm` fusion. Fastest of the three. |

All three live in `model/nmm.py`. The gram-iteration variant was reimplemented
locally (replacing an external dep) because the upstream library used an
in-place divide that broke AOT autograd under `torch.compile`. Don't add the
external `gram_ns5` package back as a dependency.

## Conventions

- **Validation uses `raise ValueError`, never `assert`.** `python -O` strips
  asserts, which would let invalid configs ship silently. Same for any
  precondition that has to fire in production. The pattern is everywhere
  in `config.py::__post_init__`; mirror it.
- `_unwrap(state_dict)` in `model/__init__.py` strips both `_orig_mod.`
  (torch.compile) and `module.` (DDP) prefixes. Always use it when loading
  checkpoints across compiled / DDP / plain runs.
- Inner-loop NS5 must run in fp32 *or* the dedicated fp16 path (gram only).
  Letting bf16 autocast leak into NS5 silently produces NaN loss after a few
  steps (`docs/RUNBOOK.md §NaN loss`).
- `finetune_mode=True` (default for `cli/finetune.py`) uses an additive
  gate with `out_scale=0` so initial logits match HF GPT-2 exactly.
  `cli/train.py` hard-codes `finetune_mode=False` for from-scratch with
  the paper's multiplicative gate.
- Checkpoints persist via `save_checkpoint` in `cli/train.py`; the file
  format includes `config`, `state_dict`, `optimizer`, `step`. The saved
  `step` is the value *before* the post-step increment — `start_step =
  step + 1` on resume.
- Run logs go to `logs/`. The directory is gitignored except for
  `.gitkeep`; don't write `.log` files anywhere else.

## Per-layer NMM state shape

A per-layer entry in `nmm_states` can be any of these — every code path
that walks state must handle all four:

- `None` — plain (non-NMM) block, when the index is not in `nmm_layer_indices`.
- `(M, S, conv_buf)` — single-head, `nmm_momentum_order=1`. `M` and `S` are
  dicts keyed by parameter name (`W1`, `W_gate`, `W2` for full-rank; six
  factors for `nmm_low_rank`).
- `(M, S_tuple, conv_buf)` — single-head, `nmm_momentum_order>1`. `S_tuple`
  is a tuple of N dicts.
- `[(M_h, S_h, conv_buf_h), ...]` — multi-head (`nmm_n_heads>1`). One entry
  per head.

`detach_states`, `compute_nmm_norm` (in `cli/train.py`), and the
decode-cache plumbing in `model/titans_gpt2.py` recurse over these
shapes. Mirror the same recursion in any new helper.

## Paper-strict vs lucidrains defaults

The defaults prefer the paper for the two documented divergences:

- `retrieval_from_M_prev=True` — paper Eq. 15 (read-then-write). False
  recovers the lucidrains-flavored write-then-read.
- `feed_persistent_to_nmm=True` — paper Eq. 28 (`M(x̃)`). False feeds
  only real tokens to the NMM.
- `nmm_n_heads=1` — paper is single-head. `>1` opts into the
  lucidrains-style `MultiHeadNMM` wrapper.

`persistent_prefix_mode="model_wide"` (default) prepends a single learned
prefix at the model level; every block — including plain non-NMM blocks
— sees it and applies a block-structured attention mask. `"per_block"`
gives each NMM block its own prefix with no shared model-wide prepend.

## Where new flags go

- **Config field** → `config.py::TitansConfig` dataclass + validation in
  `__post_init__`.
- **NMM backend toggle** (CLI knob, no shape change) → register in
  `cli/nmm_cli.py::add_nmm_args`. If it's resume-overridable, add to
  `RESUME_OVERRIDABLE_BACKEND_FLAGS` in `cli/train.py`.
- **Training-scaffolding flag** (persisted across resume) → add to
  `SAVED_TRAINING_ARGS` in `cli/train.py`.

## Tests

- `tests/conftest.py` seeds `torch.manual_seed(0)` per-test (autouse
  fixture) and sets `set_float32_matmul_precision("high")` globally.
  Don't override either unless the test specifically needs it.
- Tier picking: `unit/` for module invariants, `integration/` for
  cross-component flow, `parity/` for HF GPT-2 numerical match,
  `behavior/` for emergent properties (memorization, convergence),
  `ddp/` for multi-GPU, `performance/` for timing/leak gates,
  `failure_modes/` for error paths.

## Things that are not bugs

- `out_scale=0` at finetune init is deliberate. Don't "fix" the zero.
- The NMM keeps updating during `cli/generate.py` even under
  `torch.no_grad()` — `torch.func.grad` is independent of the no-grad
  context. This is the whole TITANS premise.
- `pyproject.toml` has no `[gram_ns5]` extra anymore — the external
  library was removed in favor of the local reimplementation. Don't
  re-add it.
- `docs/archive/` references `Gxxx` audit IDs throughout. Those tags
  have been scrubbed from the live code and docs; the archive is the
  only place they still live. Don't reintroduce the convention.
