# TITANS MAG + GPT-2

A from-scratch PyTorch implementation of the **Memory as a Gate (MAG)** variant
of TITANS (["Titans: Learning to Memorize at Test Time"](https://arxiv.org/abs/2501.00663),
Sun et al. 2025) built on top of GPT-2.

Each transformer block is augmented with a **Neural Memory Module (NMM)** whose
weights update online during the forward pass via surprise-driven gradient
descent. The memory output is combined with attention via a learned gate, blending
long-range associative memory with local attention. The NMM keeps learning at
test time — that's the whole point.

> **Not affiliated with the paper authors.** Reference implementation for research
> and study. Do **not** use `titans-pytorch` as a runtime dependency; this repo
> is an independent implementation.

## Quick links

| Doc | What it is |
|---|---|
| [`SPEC.md`](SPEC.md) | **Authoritative implementation spec** — what the code actually does |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design decisions, equations, block diagram |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phase-by-phase implementation guide (start here) |
| [`docs/PLAN.md`](docs/PLAN.md) | Full code sketches and every gap-driven safeguard |
| [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) | Unit / integration / parity / DDP test plan |
| [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) | Every config knob with range and defaults |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | What to do when training breaks |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | TITANS terminology |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Ablation plan and success criteria |
| [`docs/GAP_HISTORY.md`](docs/GAP_HISTORY.md) | Audit log (background reading) |
| [`diagrams/`](diagrams/) | Mermaid diagrams (architecture, sequences, lifecycle, DDP) |

## What you get

- **MAG block**: attention + NMM combined via a learnable gate. `finetune_mode=True`
  (default) uses an additive gate so a freshly loaded pretrained GPT-2 produces
  *identical* logits at init.
- **NMM**: SiLU-GLU gated MLP (depth `L_M=2`, hidden `4·d_model`) with per-token
  data-dependent learning rate, momentum decay, and forgetting rate.
- **Newton-Schulz 5-step spectral normalization** of the inner gradient, in fp32
  with autocast disabled (otherwise bf16 silently undoes the cast).
- **Persistent memory tokens** (`N_p=4`) prepended to each block's input.
- **Paper-strict defaults**: `retrieval_from_M_prev=True` (Eq. 15, read-then-write)
  and `feed_persistent_to_nmm=True` (Eq. 28). Flip to `False` for lucidrains-flavored
  ablations.
- **TBPTT** with chunked forward + state detach between chunks.
- **DDP**: 4-group optimizer, gradient accumulation via `model.no_sync()`,
  try/finally NCCL teardown.
- **HF GPT-2 weight loading**: identical logits to HF GPT-2 when NMM is zeroed
  (max diff < 1e-4).
- **Cached decoding** (`prepare_decode` + `forward_step`): KV cache for attention,
  conv buffer for the NMM, so each decoded token gets exactly one NMM update with
  full k-token conv context.
- **Optional fast-inference path** via `torch.associative_scan` (~10× speedup,
  <5% relative error).

## Install

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Install
uv first ([instructions](https://docs.astral.sh/uv/getting-started/installation/)),
then:

```bash
git clone https://github.com/MasonFlint44/titans-mag-gpt2.git
cd titans-mag-gpt2
uv sync                     # creates .venv, installs locked deps + dev tools
```

`uv sync` materializes `.venv` from `uv.lock` (PyTorch, transformers,
tiktoken, datasets, numpy, pytest). Run commands inside the env with
`uv run …` (no manual `source .venv/bin/activate` needed).

Requirements pinned by `uv.lock`: Python ≥ 3.12, PyTorch ≥ 2.3 (for
`torch.func.grad` + `vmap` in the inner gradient; ≥ 2.8 enables the
optional Phase 6 associative scan).

## Fine-tune GPT-2 with the NMM

```bash
uv run python scripts/finetune.py \
    --size small \
    --data /path/to/corpus.txt \
    --chunk-size 512 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-steps 5000
```

This loads pretrained `openai-community/gpt2`, splices in the NMM with
`out_scale=0` (so initial logits exactly match HF), and starts training. At
step 0 perplexity should equal vanilla GPT-2; from there it decreases as the
memory contribution ramps up.

## Train from scratch (multi-GPU)

```bash
uv run torchrun --nproc_per_node=4 train.py \
    --size small \
    --data /path/to/corpus.txt \
    --chunk-size 1024 \
    --batch-size 8 \
    --grad-accum 2 \
    --max-steps 100000
```

`train.py` hard-codes `finetune_mode=False` so the paper's pure-multiplicative
MAG gate is used (`o = silu(γ_a·y_attn) · silu(γ_m·y_mem)`) and `out_scale`
initializes to ones. `block_size` is set equal to `chunk_size` so every wpe row
sees training (G163).

## Generate

```bash
uv run python generate.py \
    --checkpoint ckpts/step_5000.pt \
    --prompt "The capital of France is" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-k 40
```

The NMM keeps updating during generation (test-time learning) — `torch.func.grad`
runs independently of `torch.no_grad()`. This is the whole TITANS premise: the
memory adapts to whatever you're feeding it right now.

> See [`diagrams/inference_sequence.mmd`](diagrams/inference_sequence.mmd) for
> the full inference flow including the conv-window mitigation.

## Test

```bash
uv run pytest tests/                              # all
uv run pytest tests/unit/                         # fast, every commit
uv run pytest -m "not slow and not gpu"           # local dev loop
uv run pytest tests/parity/                       # HF GPT-2 logit/perplexity parity
uv run pytest tests/ddp/ -m ddp                   # spawns 2-rank ranges
```

CI tiers in [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) §15.

## Pointers if something is off

| Symptom | Likely cause | See |
|---|---|---|
| Loss NaN after a few steps | Inner-loop NS5 leaking bf16 | docs/RUNBOOK.md §NaN loss |
| Logits drift from HF GPT-2 at init | Conv1D transpose or `out_scale ≠ 0` | docs/RUNBOOK.md §Logit parity |
| DDP hang during accumulation | Missing `model.no_sync()` or G222 off-by-one | docs/RUNBOOK.md §DDP hang |
| LR shrinking every resume | `base_lrs` read from `param_groups` instead of constants (G162) | docs/RUNBOOK.md §LR deflation |
| Generation gibberish beyond `block_size` | Conv window not maintained at T=1 step | docs/RUNBOOK.md §Long-context drift |

## Citation

If you use this implementation in research, cite the original paper:

```bibtex
@article{sun2025titans,
  title  = {Titans: Learning to Memorize at Test Time},
  author = {Sun, Aniket and others},
  year   = {2025},
  eprint = {2501.00663},
  archivePrefix = {arXiv}
}
```

## License

See `LICENSE`. The reference implementation `lucidrains/titans-pytorch` is MIT-licensed;
HF model weights (`openai-community/gpt2`) carry their own license — check before
distributing fine-tuned checkpoints.
