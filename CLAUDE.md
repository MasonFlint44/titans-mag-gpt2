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
    --freeze-embeddings \
    --nmm-gate-ramp-steps 100 \
    --nmm-gate-ramp-target 0.1 \
    --compile-model --optim8bit
```

The training-regime flags at the bottom — `--freeze-embeddings` and
`--nmm-gate-ramp-*` — were added after needle-in-haystack diagnostics
showed that vanilla fine-tuning produces a too-weak NMM signal
(~0.45 logits of needle-dependence past block_size, ~100× too small to
flip top-1 predictions). They're inspired by TPTT
([fabienfrfr/tptt](https://github.com/fabienfrfr/tptt)).

There's also a `--nmm-aux-loss-weight` flag that adds direct
supervision on `y_mem`. It's preserved as a diagnostic tool (defaults
to 0.0, disabled) but is **NOT** in the default recipe — see the flag
description below for what it tested and why it doesn't help at our
scale.

### `--freeze-embeddings`

Freezes only the input/output representation params: `wte`, `wpe`,
`ln_f`. Transformer blocks (attention, MLP, block LayerNorms) stay
trainable so they can adapt to the NMM-augmented residual stream —
specifically, so attention learns to attend to NMM-modulated tokens.

There's also a `--freeze-backbone` flag (mutually exclusive with this
one) that freezes everything except the memory pathway. **Don't use it.**
A diagnostic run showed it's too aggressive: with attention frozen, the
model can't compensate for the NMM signal being injected into the
residual stream, and short-distance recall collapses to near-zero. The
flag is preserved for completeness but the help text discourages it.

### `--nmm-gate-ramp-steps N` + `--nmm-gate-ramp-target X`

During the first N steps, hold every per-block `out_scale` to a linear
ramp `0 → X`, with `requires_grad=False` so the optimizer doesn't fight
the schedule. At step N, the optimizer takes over. Forces the memory
gate open on a fixed schedule instead of relying on LM loss alone to
slowly discover that the NMM is worth using.

### `--nmm-aux-loss-weight α` (diagnostic-only, not in default recipe)

When `α > 0`, install a hook on the **last NMM block's `forward_chunk`**
that captures its raw (pre-gate) `y_mem` each step. After the main
forward, project that y_mem through `ln_f` + tied wte LM head and
cross-entropy it against the same next-token labels as the main LM
loss; add `α * aux_loss` to the total before backprop.

Implementation helpers in `cli.train`:
- `install_y_mem_capture(model, target_layer=-1) -> (capture, uninstall)`
- `compute_aux_retrieval_loss(y_mem, input_ids, model)`

The capture **must be installed before `torch.compile`** so the patched
method is part of the traced graph; `cli/finetune.py`'s
`_install_aux_capture_if_enabled` does this at the right point in both
the resume and fresh paths. Default `α = 0.0` (disabled).

**Why this flag exists and why it's not in the default recipe:**

Needle-in-haystack diagnostics showed `y_mem` at the answer position
has cosine alignment ≈ 0 with the correct-answer embedding at long
distance — the NMM's `k_proj` and `q_proj` aren't aligned and M ends
up holding input-dependent noise rather than retrievable structure.
This flag was added to test whether direct supervision could fix that:
by training `y_mem` to predict next tokens, we hoped to pressure the
NMM read pathway into producing outputs that correctly predict the
answer.

When we tested with `α=0.5` for 1000 steps on the needle corpus, the
result was instructive:

- Short-distance accuracy saturated at 100% (vs ~99% without aux loss)
- Long-distance accuracy stayed at chance (~4%)
- **The diff-under-swap magnitude at long distance _decreased_** (from
  ~2 logits to ~0.4) and `y_mem` alignment stayed at noise floor

The interpretation: the optimizer, given direct supervision and no
retrievable structure to extract from M, correctly chose to *suppress*
input-dependent noise at long-distance positions rather than
manufacture signal. This is the right behavior for a calibrated
predictor — but it means the aux loss can't actually create
retrievable structure that the surprise-driven update rule isn't
already producing. The result indicates the failure is structural to
the update rule + scale + pretrained-backbone combination, not
something more training signal can fix.

The flag stays in the codebase as a diagnostic tool for future
experiments (e.g., if testing a different memory update rule).

### Defaults

The argparse defaults leave both freeze flags OFF,
`--nmm-gate-ramp-steps=0`, and `--nmm-aux-loss-weight=0.0`, so legacy
scripts that didn't pass them get full-fine-tune-with-passive-gate
behavior unchanged. The canonical recipe above is the new recommendation;
the previous full-fine-tune recipe is still supported but doesn't appear
to learn cross-chunk recall well.

For multi-GPU from-scratch runs on a bigger box you'll want to bump
`--batch-size`, drop `--grad-accum`, raise `--max-steps`, and probably
drop the freeze flags (more data + more capacity makes the
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

## DeltaProduct memory (alternative to NMM)

`config.memory_type` selects the fast-weight mechanism that slots into
each MAG block. Default `"nmm"` keeps the paper-strict surprise-driven
inner-loop gradient update from Behrouz et al. (Titans). Setting
`"delta_product"` swaps it for the closed-form delta-rule update of
Siems et al. (DeltaProduct, ICLR 2025 / arxiv 2502.10297), which
generalizes Yang et al.'s DeltaNet (NeurIPS 2024) via a configurable
`order` parameter.

| Order | Mechanism | Notes |
|---|---|---|
| 1 | DeltaNet | One rank-1 delta update per token. |
| 2 | DeltaProduct order-2 | Matches Titans expressivity per TPTT (arxiv 2506.17671). Default. |
| N≥2 | DeltaProduct order-N | More state-tracking capacity; cost scales linearly in N. |

**Why this exists alongside the NMM:** the gradient-based NMM appears not
to adapt well from a pretrained backbone at our scale — needle
diagnostics showed M is input-dependent but `y_mem` at the answer
position is uncorrelated with the right answer token at long distance,
suggesting the surprise-driven update doesn't naturally produce
retrievable key-value structure unless the backbone is co-evolved with
it (paper-style from-scratch training). DeltaProduct's update rule
forms key-value associations by construction (`M ← M + β·(v − M·k)·kᵀ`),
which sidesteps that adaptation problem. The TPTT library / paper
formalizes exactly this pattern as the production pretrained-adaptation
recipe — same MAG-style integration, persistent prefix, and gating as
the NMM, with the inner mechanism swapped.

### Flags

```
--memory-type {nmm,delta_product}   # selector; default "nmm"
--delta-order N                      # 1 = DeltaNet, 2 = DeltaProduct (default 2)
--delta-n-heads N                    # heads per block; default 1, prefer n_head for prod
--delta-block-size N                 # 1 = sequential reference, >1 = chunkwise parallel
```

The `--delta-*` flags only meaningful with `--memory-type delta_product`;
the CLI raises `argparse.ArgumentTypeError` if you set them without
selecting delta_product (saves you from a typo silently no-op'ing).

### Sequential vs blockwise (chunkwise WY)

`model/delta_product.py` provides two equivalent forward paths:

- **Sequential** (`delta_block_size=1`): paper-strict per-token recurrence.
  Reference correctness path. The tests pin chunkwise == sequential at
  any order, with non-zero initial state, and with document boundaries.
- **Blockwise** (`delta_block_size>1`): closed-form chunkwise WY-form
  solve. Bit-equivalent to sequential, but replaces the T-step Python
  loop with one triangular solve + a few batched matmuls. Training-time
  speed path. Doc-boundary aware: splits the chunk at boundary positions
  and runs one WY solve per segment.

State is a 1-tuple `(M,)` with `M ∈ R^(B, d_head, d_head)` per head —
no momentum stack, no conv buffer. State serialization (`model/state_io.py`)
recognizes `memory_type` in the fingerprint so a checkpoint from one
mechanism can't silently load into the other.

### Multi-head (`delta_n_heads > 1`)

`MultiHeadDeltaProduct` wraps N parallel single-head `DeltaProductMemory`
instances on `head_dim = n_embd / n_heads`. For training, prefer
`delta_n_heads = n_head` (matches attention) so M per head is
`head_dim × head_dim` — dramatically smaller state vs single-head
`n_embd × n_embd`. Default 1 = simplest case, but the production recipe
should match attention's head count.

### Compatibility with existing training-regime flags

The freeze, gate-ramp, and aux-loss helpers in `cli/train.py` are
mechanism-agnostic — they key off the `nmm` attribute name on the block
(which is the polymorphic handle for either NMM or DeltaProductMemory),
`out_scale`, `gamma`, and `persistent`, all of which exist on both
modules. So `--freeze-embeddings`, `--nmm-gate-ramp-*`, and even
`--nmm-aux-loss-weight` compose with `--memory-type delta_product`
unchanged.

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
