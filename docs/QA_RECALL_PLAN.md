# QA Recall Experiment — Plan

## Goal

One chart: **first-token recall accuracy vs. context distance**. TITANS
holds, vanilla GPT-2 collapses past attention's reach.

## Success criteria

- Pipeline runs end-to-end: both training conditions complete, both eval
  sweeps produce numbers.
- **Vanilla GPT-2 drops sharply past `block_size`** (expected; if it
  doesn't, something's wrong with the eval).
- **TITANS curve stays measurably higher** at distances beyond
  `block_size` (the research claim).
- Reproducible: prepared corpus, eval set, and checkpoint hashes pinned
  in the repo (corpora regenerated from cached SQuAD; checkpoints stored
  outside git).

## Phases

### Phase 1 — Data preparation (~4 h, no GPU) — **DONE**

- `scripts/prepare_squad_corpus.py`: load cached `rajpurkar/squad`, pack
  5–6 (passage, Q, A) triples per T=1024 sequence with `[P] passage Q: …
  A: …` formatting.
- Output: `corpora/squad/squad_train.txt` + `corpora/squad/squad_eval.json`.
- Train: shuffled SQuAD train split, packed greedily into ≤992-token
  sequences, separated by `<|endoftext|>` literal markers.
- Eval: 2500 records from SQuAD validation, with all answer aliases.
- Tests: `tests/unit/test_squad_corpus.py` — packing budget, round-trip
  serialization, alias dedup, oversize-triple fallback.

### Phase 2 — Vanilla GPT-2 control mode (~1 h) — **DONE**

- `scripts/_nmm_cli.py`: `--vanilla-gpt2` flag → `nmm_layer_indices=[]`,
  mutually exclusive with `--nmm-layer-indices`.
- Result: every block becomes a `PlainGPT2Block` (attn + MLP only).
- Tests: `tests/unit/test_nmm_cli.py` — flag round-trip, conflict
  detection, end-to-end model construction check.

### Phase 3 — Eval infrastructure (~3 h) — **DONE**

- `scripts/eval_qa_recall.py`: builds prompts with target passage at
  controlled distance from the final question, runs the cached decode
  path, scores first-token argmax.
- `scripts/plot_qa_recall.py`: bootstrapped CI, accuracy-vs-distance
  curve, automatic block_size guideline.
- Default distance buckets: `[0, 256, 512, 1024, 2048, 3072]` — spans
  within-attention and beyond-attention regimes.
- Tests: `tests/integration/test_qa_recall_smoke.py` +
  `tests/unit/test_plot_qa_recall.py`.

### Phase 4 — Training (~23 h GPU, mostly unattended)

**Vanilla GPT-2 control (~1–2 h, run first to validate the pipeline):**

```bash
uv run python scripts/finetune.py \
    --size small \
    --data corpora/squad/squad_train.txt \
    --chunk-size 1024 \
    --batch-size 1 --grad-accum 16 \
    --vanilla-gpt2 \
    --compile-model --optim8bit \
    --max-steps 5000 \
    --save-dir ckpts/vanilla
```

**TITANS (~21 h, run overnight):**

```bash
uv run python scripts/finetune.py \
    --size small \
    --data corpora/squad/squad_train.txt \
    --chunk-size 1024 \
    --batch-size 1 --grad-accum 16 \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks \
    --nmm-use-cans \
    --compile-model --optim8bit \
    --max-steps 5000 \
    --save-dir ckpts/titans
```

Sanity at step 1000: TITANS `out_scale` > 0, NMM norms growing, loss
descending. Vanilla loss should also descend; if it doesn't, the data
pipeline is broken.

### Phase 5 — Eval + plot + writeup (~3 h)

```bash
uv run python -m scripts.eval_qa_recall \
    --checkpoint ckpts/vanilla/latest.pt \
    --eval-data corpora/squad/squad_eval.json \
    --out results/vanilla.json \
    --n-examples 500

uv run python -m scripts.eval_qa_recall \
    --checkpoint ckpts/titans/latest.pt \
    --eval-data corpora/squad/squad_eval.json \
    --out results/titans.json \
    --n-examples 500

uv run python -m scripts.plot_qa_recall \
    --input results/vanilla.json results/titans.json \
    --label "Vanilla GPT-2" "TITANS" \
    --out docs/figures/qa_recall.png
```

Writeup goes in `docs/EXPERIMENTS.md` or a new `RESULTS.md`.

## Pinned configuration

| Knob | Value | Rationale |
|---|---|---|
| Model size | gpt2_small | Fits on consumer GPU |
| Steps | 5000 optimizer | Past NMM warm-up (~1k steps) |
| Effective batch | 16 (1 × 16 grad-accum) | Standard fine-tune scale |
| Chunk/block size | 1024 | Full GPT-2 context |
| NMM block size | 64 | Documented recipe |
| LR | base_lr_gpt2=3e-5, base_lr_nmm=9e-5 | EXPERIMENTS.md §2.1 |
| Eval distances | [0, 256, 512, 1024, 2048, 3072] | Spans both regimes |
| Eval examples per bucket | 500 | ±2% CI per bucket |
| Scoring | First-token argmax | Consistent across distance regimes |

## Why first-token scoring (not substring match)

At distances `> block_size`, `prepare_decode_chunked` chunks the prompt
through forward() and can only sample ONE token before `forward_step`
would push wpe out of bounds. To keep the metric consistent across all
distance buckets (so the chart is interpretable), every bucket uses
first-token argmax. This loses some sensitivity in the easy regime —
some answers' first tokens are ambiguous — but the cross-condition
comparison stays valid because both models are scored identically.

## Risks

| Risk | Mitigation |
|---|---|
| NMM doesn't turn on (`out_scale` stuck near 0) | Sanity check at step 1000 |
| Vanilla scores too high at long distances | Tighten distance buckets / check for prompt-construction leak |
| Both score ~50% at all distances | Broken scoring; run step-0 checkpoint as sanity (should be near 0%) |
| NaN loss | Existing `train.py` recovery resets `nmm_states`; should self-heal per RUNBOOK §NaN loss |

## Not committed

- Checkpoints (already in `.gitignore` via `ckpts/`)
- Prepared corpora — regenerable; added to `.gitignore` under `corpora/`
- Eval result JSONs — regenerable; added to `.gitignore` under `results/`

## Time budget

| Phase | Engineering | GPU | Wall-clock |
|---|---|---|---|
| 1. Data prep | 4 h | — | done |
| 2. Vanilla control | 1 h | — | done |
| 3. Eval infrastructure | 3 h | — | done |
| 4. Vanilla training | — | 1–2 h | 2 h |
| 4. TITANS training | — | 21 h | overnight |
| 5. Eval + plot + writeup | 3 h | <1 h | 3 h |
