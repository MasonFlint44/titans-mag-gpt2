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

For tasks that need cross-chunk retrieval (needle-in-haystack at
distances > chunk_size), add `--bptt-window K` with K large enough to
span your longest training example (e.g. K=4 for the needle corpus's
3072-token max distance at chunk_size=1024). See the section below.

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

### `--bptt-window K` (cross-chunk gradient flow)

The training loop's default (`K=1`) is classical truncated BPTT:
recurrent state is detached at every chunk boundary, so the gradient
from chunk `t`'s loss cannot reach memory-write projections that fired
in chunk `t-1` or earlier. Under TBPTT, the memory pathway only gets
end-task supervision *within* a single chunk — fine for streaming
language-model training, but the wrong recipe for tasks that demand
cross-chunk retrieval (e.g. needle-in-haystack at distances larger
than `chunk_size`).

Setting `--bptt-window K` keeps the autograd graph alive across `K`
consecutive chunks. The loss at chunk `K-1` then backprops through
`M`'s recurrence into the Q/K/V/β projections that wrote into `M`
during chunks `0..K-2` — the path required for the memory pathway to
learn that "this key now will be looked up later." Per-optimizer-step
chunks become `accum_steps * bptt_window`. Memory cost grows linearly
in `K` (the full forward graph for `K` chunks is held until the single
backward at window end).

**Why this matters for the needle corpus:** training examples in
`corpora/needle/` carry distances drawn uniformly from `[0, 3072]`
tokens. At `chunk_size=1024`, examples longer than ~900 tokens span
multiple chunks, so the answer position's loss can never reach the
needle's write under K=1 — leaving the memory pathway unsupervised at
long distance. K=4 spans the longest example, restoring the gradient
path. Five separate architectures (NMM, pre-TPTT DeltaProduct,
TPTT-MAG, TPTT-LiZA, aux-loss NMM) all hit the same ~chance accuracy
cliff at exactly `d == chunk_size` under K=1; this flag is the fix.

`bptt_window` is in `SAVED_TRAINING_ARGS`, so resume drift detection
warns if you resume with a different K than was saved.

### Contrastive needle loss (`--needle-contrastive-loss-weight`) + `--needle-format alnum20`

Two interventions targeting the **marginal-output failure mode** we
diagnosed across the 8 prior DeltaProduct experiments: the optimizer
reliably converges to "output the empirical distribution over plausible
answer tokens" rather than learning retrieval, because the marginal-
output basin is mechanically easier to find than the retrieval basin
under LM cross-entropy alone. The single-needle and 4-needle overfit
diagnostics confirmed this: even when forced into a setting where
constant output was provably suboptimal, the model output the same
fixed probability distribution at every answer position regardless of
needle content.

**Data side: `--needle-format alnum20` in `scripts.prepare_needle_corpus`.**
Replaces the default `XX-NNNN` needles (~461 unique first BPE tokens,
heavily concentrated on common capital letters) with random 20-char
alphanumeric strings (~812 unique first BPE tokens with a flatter
distribution). The marginal-output strategy's per-prompt accuracy is
bounded by `1/n_unique_first_tokens`; widening the distribution
mechanically lowers the ceiling.

**Loss side: `--needle-contrastive-loss-weight λ` (default 0).** Adds a
top-K hard-negative contrastive term at positions whose label is the
needle's first answer token. We detect these positions by matching the
`"A:"` marker (GPT-2 BPE token sequence `(32, 25)`) at indices
`[t-1, t]`. The contrastive term at each answer position is
`F.cross_entropy([correct_logit, top_k_wrong_logits], target=0)` — an
InfoNCE softmax over the correct answer concatenated with the K most
confident wrong predictions. The model can't satisfy this by hedging
across plausible needle tokens; it has to commit to the specific
correct token, which can only be done by reading the prompt's needle.

`--needle-contrastive-top-k` (default 10) controls how many of the
model's currently-most-confident wrong predictions act as negatives;
higher K = stronger pressure but also more compute and more risk of
suppressing rare-but-plausible tokens.

**Recipe usage:**

```bash
# Generate the alnum20 corpus once
uv run python -m scripts.prepare_needle_corpus \
    --out-dir corpora/needle_alnum20 \
    --needle-format alnum20 \
    --max-distance 3072

# Finetune with the contrastive loss enabled
uv run python -m cli.finetune \
    --size small --data corpora/needle_alnum20/needle_train.txt \
    --memory-type delta_product --memory-topology liza \
    --delta-order 2 --delta-n-heads 12 --delta-block-size 64 \
    --freeze-embeddings \
    --nmm-gate-ramp-steps 100 --nmm-gate-ramp-target 0.1 \
    --bptt-window 2 \
    --needle-contrastive-loss-weight 0.5 \
    --needle-contrastive-top-k 10 \
    --max-steps 1000 --warmup-steps 100 \
    --compile-model --optim8bit \
    --save-dir ckpts/needle_anti_marginal/
```

**Two caveats and one tuning tip:**

- The contrastive loss only fires at chunks that contain at least one
  answer position (every full needle example contributes one). At
  `batch_size=1, chunk_size=1024` with average example length ~1500
  tokens, this is roughly half of chunks. Increasing `batch_size` (if
  VRAM allows) gives more answer positions per backward.
- Implementation lives in `cli/train.py::compute_contrastive_needle_loss`;
  the marker tokens `NEEDLE_ANSWER_MARKER_TOKENS = (32, 25)` are
  hard-coded for the GPT-2 tokenizer. If you swap tokenizers, regenerate
  this constant (`tok.encode("A:")`).
- Start with `λ = 0.5`. Too high (`λ > 2`) tends to destabilize early
  training because the contrastive signal is high-magnitude at random
  init; too low (`λ < 0.1`) won't meaningfully alter the optimization
  landscape.

### Phased training (`--freeze-attention-steps N`)

Targets a specific failure observed in the anti-marginal-output NMM
finetune (commit history May 29): even when the marginal-output trap is
broken by `--needle-contrastive-loss-weight`, the pretrained model
satisfies the retrieval constraint via softmax attention rather than
recruiting the memory pathway. The eval cleanly showed 95% accuracy at
d=768 (within attention window) but 2.8% at d=1024+ (past attention
window) — attention was doing all the work; y_mem was anti-aligned
(cos = −0.088) with the correct answer.

The mechanism: pretrained attention is already so competent at retrieval
that the optimizer never has any pressure to develop the memory pathway.
Phased training removes that pressure release: for the first N steps the
attention q/k/v/proj projections are frozen (`requires_grad=False`), so
the optimizer cannot tweak attention. The memory pathway + MLP +
LayerNorms must absorb the gradient signal. At step N the freeze
releases and normal training resumes.

The hypothesis is that by the time attention unfreezes, the memory
pathway has already learned to do retrieval (because contrastive loss
forced *something* to satisfy the retrieval constraint and attention
was off-limits), and that capability survives the transition. Whether
it survives is the empirical open question.

**Recipe usage:**

```bash
uv run python -m cli.finetune \
    --size small --data corpora/needle_alnum20_d900/needle_train.txt \
    --freeze-embeddings \
    --nmm-gate-ramp-steps 100 --nmm-gate-ramp-target 0.1 \
    --bptt-window 1 \
    --needle-contrastive-loss-weight 0.5 \
    --needle-contrastive-top-k 10 \
    --freeze-attention-steps 500 \
    --max-steps 1000 --warmup-steps 100 \
    --nmm-block-size 64 --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks --nmm-use-gram-ns5 \
    --compile-model --optim8bit
```

This is the consumer-GPU NMM finetune recipe plus alnum20 + contrastive
+ a 500-step attention freeze. Total 1000 steps splits as 500 phase 1
(memory only) + 500 phase 2 (joint).

**What gets frozen vs trained in phase 1:**

| Component | Phase 1 (steps 0..N-1) | Phase 2 (steps N..end) |
|---|---|---|
| Attention q/k/v/proj | frozen (requires_grad=False) | trainable |
| Attention LayerNorms (ln_1) | trainable | trainable |
| MLP (c_fc, c_proj) | trainable | trainable |
| MLP LayerNorm (ln_2) | trainable | trainable |
| Memory pathway (NMM/DeltaProduct) | trainable | trainable |
| Gate / out_scale | gate-ramp-controlled | trainable |
| Embeddings (wte/wpe/ln_f) | frozen by `--freeze-embeddings` | frozen |

**Important caveats:**

- This is structurally different from `--freeze-backbone`, which froze
  *everything* except the memory pathway and broke short-distance recall
  in the earlier diagnostic. Here MLP and LayerNorms stay trainable, so
  the model retains the ability to integrate residual-stream signals.
- The optimizer (built before training starts) still has the attention
  projections in its parameter groups; their AdamW moments simply don't
  update while `requires_grad=False`. When the freeze releases, moments
  initialize lazily on first gradient — there's no "cold start" for the
  optimizer state.

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

## TITANS-paper from-scratch recipe

For replicating the TITANS paper's published NIAH/BABILong results
(Behrouz et al. 2024, arxiv 2501.00663) rather than the TPTT
pretrained-adaptation recipe. Two changes from the consumer-GPU finetune
recipe above: (a) the memory module is `nmm` not `delta_product` (paper's
actual mechanism — the closed-form delta rule was TPTT's
pretrained-adaptation workaround, not what the paper validated), and
(b) training runs through `cli/train.py` rather than `cli/finetune.py`
(no pretrained backbone load, no `--freeze-embeddings`, no gate ramp).

### Tokenize the corpus once

The paper uses FineWeb-Edu (the 760M model gets 30B tokens; the
170M/340M/400M models get 15B tokens). At consumer-GPU scale we target
~1.5B tokens; tokenize once to a uint16 binary so the multi-hour
tokenize step isn't paid on every training restart:

```bash
uv run python -m scripts.tokenize_fineweb_edu \
    --output corpora/fineweb_edu_1p5b.bin \
    --max-tokens 1500000000
```

The script streams `HuggingFaceFW/fineweb-edu` `sample-10BT` from HF,
tokenizes with GPT-2 BPE, and writes raw little-endian uint16 (~3 GB for
1.5B tokens). Resumable — re-running picks up where the previous run
stopped, modulo a few approximate document re-tokenizations.

`cli/train.py` and `cli/finetune.py` both detect the `.bin` extension at
load and memmap the file (`data.tokenizer.load_token_stream`); text
inputs still flow through the legacy
`read_eot_separated_documents → encode_corpus` path unchanged.

### Training command (paper-strict NMM + MAG)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m cli.train \
    --size small \
    --data corpora/fineweb_edu_1p5b.bin \
    --chunk-size 1024 \
    --batch-size 1 --grad-accum 16 \
    --bptt-window 1 \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks \
    --nmm-use-gram-ns5 \
    --max-steps 50000 \
    --warmup-steps 2000 \
    --log-every 100 \
    --save-every 5000 \
    --save-dir ckpts/titans_scratch/ \
    --compile-model --optim8bit
```

`cli/train.py` hard-codes `finetune_mode=False` for from-scratch, which
selects the paper's multiplicative MAG gate (`o = y_attn + SiLU(γ·y_mem)·y_attn`,
γ init = 1) rather than the additive zero-init gate the finetune recipe
uses. No `--freeze-embeddings`, no `--nmm-gate-ramp-*` flags — the
memory pathway is active and load-bearing from step 0.

### What each flag matches in the paper

| Our flag | Paper (§5.1) | Notes |
|---|---|---|
| `nmm` (default `memory_type`) | Neural Memory Module | Paper's surprise-driven inner-loop gradient update |
| `mag` (default `memory_topology`) | Memory-as-Gate | Paper's multiplicative-gate topology (Table 5 row +Attn (MAG)) |
| `nmm_depth=2`, `nmm_expansion=4` | `L_M=2`, full-rank MemoryMLP | Deep memory — drops 11.4 NIAH points if replaced with linear |
| `nmm_conv_kernel=4` | "1D depthwise-separable conv after Q/K/V" | §4.4; drops 6.4 NIAH points if removed |
| `nmm_momentum_order=1`, `W_alpha` decay | momentum + weight decay | Each drops ~10 NIAH points if removed |
| `nmm_n_persistent=4` | persistent memory | Drops 4.2 NIAH points if removed |
| `nmm_n_heads=1` | single-head | Paper-strict |
| `--bptt-window 1` | training length 4K, single-chunk forward | Paper uses no cross-chunk BPTT; the model extrapolates at inference |
| `--chunk-size 1024` | training length 4K | **forced by GPT-2 wpe ceiling**; paper trains at 4K |
| `--batch-size 1 --grad-accum 16` ≈ 16K tok/step | batch 0.5M tok/step | **forced by VRAM**; we're at ~1/32 of paper's batch |
| LR 3e-4 (our default) | LR 4e-4 | Paper's 4e-4 is calibrated for 0.5M-tok batches; our default is safer at our batch size |
| `--max-steps 50000` × 16K tok = 800M tok | 15B-30B tok | **forced by time budget**; we're at ~1/20-1/40 of paper's tokens |

### What this experiment tests

Whether the TITANS architecture acquires *any* cross-chunk retrieval
signal at gpt2_small scale within a ~10-day NMM training budget. The
paper's headline 92-98% NIAH accuracy came from 15B-30B tokens at
170M-760M params; we're testing the lower bound. A clean positive result
at d > 1024 (the chunk boundary) would replicate the paper's
extrapolation claim at smaller scale; a negative result at ~800M tokens
suggests the paper's numbers are more scale-dependent than the paper
acknowledges.

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
which sidesteps that adaptation problem. The TPTT paper formalizes
exactly this pattern as the production pretrained-adaptation recipe.

**Formulation:** our implementation follows TPTT's specific design
choices verbatim. From TPTT Section 3.1 and the source at
`fabienfrfr/tptt/src/tptt/modeling_tptt.py`:

- **Single Q, K, V projections.** No N independent per-order projections —
  there's one Q, one K, one V projection per layer regardless of `order`.
  The N virtual writes per real token come from the VirtualTokenExpander.
- **Projections (eq. 4)**: `q_normed = L2_normalize(SiLU(q_raw))`,
  `k_normed = L2_normalize(SiLU(k_raw))`, `v_scaled = v_raw / √head_dim`.
  Q and K get SiLU + L2 normalize; V is just scaled by the standard
  attention factor.
- **β gating (eq. 5)**: `β = σ(CausalAvgPool3(k_raw))` — a vector per
  head, per token, per dim. The pool is a fixed-weight kernel-3
  causal moving average (NOT learnable; weights `[1/3, 1/3, 1/3]`).
  β is computed from the RAW K projection (before SiLU+L2).
- **VirtualTokenExpander ("dt" derivative trick)**: each (q, k, v, β)
  is expanded to N virtual tokens via a fixed binomial-coefficient
  convolution. For order N, the normalized kernel is
  `deriv[k] = (-1)^k · C(N-1, k) / sum(|coeffs|)`. After
  flip-and-permute, `virtual[s, k] = x_padded[s + (N-1-k)] · deriv[N-1-k]`,
  so sub-step 0 carries the current token and sub-step N-1 carries
  the oldest. The N virtual tokens form the T·N virtual sequence the
  WY solve operates on.
- **Per-token update with vector β over virtual sequence**:
  `M ← M + (β⊙v − M·(β⊙k))·kᵀ` per virtual write. The K on the right
  is un-gated.
- **Reads at virtual position N·t + (N−1)**: TPTT reads at all virtual
  positions internally but takes only the last sub-step's output for
  each real token (`output[..., -1, :]` at line 1407 of modeling_tptt.py).
  We do the equivalent: read at each real token using
  `virtual_q[t, N−1]`, the last sub-step's expanded query.
- **Output (eq. 6)**: `y = RMSNorm(y_raw) · out_proj(bias=True) · out_scale`.
  Manual RMSNorm (no learnable scale), Linear `out_proj(n_embd, n_embd,
  bias=True)`, then per-channel `out_scale` gain. The `out_scale` is
  our addition to preserve the gate-ramp logic in `cli/train.py` and
  the finetune-mode "y = 0 at step 0" invariant.

**What we don't replicate from TPTT**:
- The parallel linear-attention integration (LiZA). TPTT runs linear
  attention in parallel with softmax attention combined via MaG. We
  apply DeltaProduct as the memory path under the original Titans MAG
  formulation (memory output gates attention output). The user
  explicitly chose to keep our attention topology unchanged.
- The alpha gate. TPTT's default `alpha_gate = "c"` makes alpha =
  constant ones, which is a no-op in the WY math. We don't carry an
  alpha parameter at all; equivalent under default config.
- Grouped-query attention (`num_key_value_heads ≠ num_heads`). Not
  needed at gpt2_small scale.
- Initial state fill of 1e-6 → we now match this for "stability if
  unlinear activation" per TPTT.

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

State is a 3-tuple `(M, k_raw_buf, qkvb_buf)`:
- `M ∈ R^(B, n_heads, head_dim, head_dim)` — recurrent state, initialized
  to 1e-6 (TPTT's "stability if unlinear activation" fill).
- `k_raw_buf ∈ R^(B, 2, n_heads, head_dim)` — last 2 raw K projections
  for the CausalAvgPool's 3-token window.
- `qkvb_buf` — tuple of 4 tensors (q, k, v, β), each
  `R^(B, order-1, n_heads, head_dim)`, for the VirtualTokenExpander's
  (order-1)-token continuity. `None` at order=1 (expander is identity).

Both buffers are threaded across `forward_chunk` calls so the pool and
the expander see continuous context (without these, splitting a stream
into multiple chunks would produce different β AND different virtual
tokens at chunk heads than a single-shot forward). State serialization
(`model/state_io.py`) recognizes `memory_type` in the fingerprint so a
checkpoint from one mechanism can't silently load into the other.

**Doc-boundary caveat**: the pool's 2-token memory and the expander's
(order-1)-token memory leak across doc boundaries within a chunk. M
correctly resets at boundaries, but β and the virtual tokens at the
boundary position and the next few positions still see pre-boundary
context. For our use case (needle scenarios with ~100+ token contexts),
the leak is a small effect.

### Multi-head (`delta_n_heads > 1`)

`DeltaProductMemory` is multi-head native — `n_heads` is a ctor arg, no
wrapper class involved. Per-head architecture is unchanged from the
single-head case (each head has its own [head_dim × head_dim] M, Q, K,
V, β projections; heads share no parameters and see only their own
slice of `x`), but parameters are stored as stacked tensors of shape
`[n_heads, head_dim, *]` so projections fuse into one einsum call per
role and the WY solve runs with batch dim = B × n_heads. This avoids
the Python-loop-over-heads overhead in the old per-head-module design.

For training, prefer `delta_n_heads = n_head` (matches attention) so M
per head is `head_dim × head_dim` — dramatically smaller state vs
single-head `n_embd × n_embd`. Default 1 = simplest case, but the
production recipe should match attention's head count.

**Checkpoint compatibility note**: the TPTT formulation is architecturally
incompatible with prior DeltaProduct checkpoints (β was a learnable
Linear in the old recipe; it's now derived from K via fixed
CausalAvgPool). The legacy state-dict migration hook was removed —
TPTT-recipe runs must train from pretrained-GPT-2 weights, not from
an older DeltaProduct checkpoint.

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
