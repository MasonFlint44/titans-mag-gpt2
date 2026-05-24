# Experiments

What experiments to run to validate the implementation, and what success looks like for each. Use this as a checklist: until every experiment in §1 passes, the implementation is not "done" regardless of code review.

> "If something seems to work but you haven't run the experiments below, you don't know if it works."

---

## 1. Correctness gates (must pass before any real training)

These are not ablations — they are pass/fail gates. If any fail, stop and fix before running anything in §2.

### 1.1 HF GPT-2 logit parity at init
**Goal:** With NMM zeroed (`out_scale=0` via `finetune_mode=True`) and `N_p=0`, a freshly loaded model produces identical logits to vanilla HF GPT-2.

**Setup:**
```python
cfg = TitansConfig.gpt2_small(nmm_n_persistent=0)  # finetune_mode=True default
ours = build_and_load_pretrained(cfg).eval()
hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
x = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
diff = (hf(x).logits - ours(x)[0]).abs().max()
```
**Pass:** `diff < 1e-4`

**Why this matters:** If this fails, weight loading (Conv1D transpose, residual scale, wte/lm_head tie) is broken. Every later experiment is meaningless until this passes.

### 1.2 HF perplexity parity on held-out data
**Goal:** Perplexity on a held-out corpus matches HF GPT-2 within 5% when NMM is zeroed.

**Setup:** WikiText-103 validation, 1024-token windows, no NMM contribution.

**Pass:** `|ppl_ours - ppl_hf| / ppl_hf < 0.05`

### 1.3 Overfit a single batch
**Goal:** Train on a single fixed batch for 200 steps. Loss must reach <0.1.

**Pass:** Final loss < 0.1; loss curve monotonically decreasing (no spikes after step 20).

**Why this matters:** If the model can't memorize one batch, the inner loop is wrong. Run this before any other training experiment.

### 1.4 KV memorization
**Goal:** Inject a unique key→value pair early in a synthetic sequence; the model recalls the value when prompted with the key 500 tokens later.

**Setup:** `[k1, v1, ...random_tokens..., k1, ?]` — measure top-1 accuracy on the masked position. Repeat 100 times with different (k, v) pairs.

**Pass:** Top-1 accuracy > 80% (vanilla GPT-2 baseline: ~5%).

**Why this matters:** This is the smallest test that proves the NMM is actually storing and retrieving across distance. If it's at baseline, the memory branch is disconnected.

### 1.5 Needle-in-haystack
**Goal:** Inject a single fact early in a long context (2048 tokens), ask about it at the end.

**Setup:** "The secret code is 47291. ...2000 tokens of distractor text... What was the secret code?"

**Pass:** Correct answer in greedy generation for ≥ 70% of trials at 2048-token context. Drops gracefully (not catastrophically) at longer contexts.

### 1.6 Long-context loss curve
**Goal:** Per-position next-token loss does NOT spike at chunk boundaries (every 1024 tokens).

**Setup:** Take a 4096-token continuous text, plot loss[t] for t in 0..4095. Look for the chunk boundaries at 1024, 2048, 3072.

**Pass:** Loss at chunk boundary is within ±10% of loss just before. If there's a visible spike, NMM state continuity is broken.

### 1.7 Scan vs sequential parity (if Phase 6 implemented)
**Goal:** `_forward_chunk_scan` produces NMM output within 5% relative error of `_forward_chunk_sequential`.

**Pass:** `(out_scan - out_seq).abs() / out_seq.abs().mean() < 0.05` on a 512-token chunk.

---

## 2. Headline experiments (proves the implementation works)

The experiments that justify writing this code.

### 2.1 Fine-tuning GPT-2 small on WikiText-103
**Setup:**
- `TitansConfig.gpt2_small()`, `finetune_mode=True`
- Load `openai-community/gpt2` pretrained weights
- Train for 10K steps, `chunk_size=512`, `batch_size=8`, `grad_accum=4` (effective batch 32)
- LR: `BASE_LR_GPT2=3e-5`, `BASE_LR_NMM=9e-5` (10× lower than from-scratch since pretrained)
- Hold out 5% for validation

**Measure:**
- Validation perplexity at steps {0, 1K, 5K, 10K}
- Compare to vanilla GPT-2 fine-tuned on the same data (no NMM)

**Expected:**
- Step 0: identical to vanilla (parity test, §1.1)
- Step 10K: ≥3% lower perplexity than vanilla fine-tune at the same step
- Step 10K NMM weights have moved meaningfully from init (norm ratio > 1.5)

**Failure modes:**
- No improvement over vanilla → NMM not contributing (check `out_scale` gradient, see RUNBOOK §NMM not learning)
- Worse than vanilla → NMM is degrading the residual stream (check gate formula, finetune_mode)

### 2.2 Training from scratch on a small corpus
**Setup:**
- `TitansConfig.gpt2_small(finetune_mode=False, chunk_size=1024)`
- Random init (no pretrained load)
- Train for 50K steps on OpenWebText (subset) or similar
- 4× GPU DDP, `batch_size=8`/GPU, `grad_accum=2`

**Measure:**
- Validation perplexity vs. a GPT-2 small trained on identical data without the NMM (same total compute)

**Expected:**
- TITANS-MAG achieves equal or lower perplexity at the same step count
- Loss curve is stable (no NaN spikes, no plateaus past warmup)

### 2.3 Long-context generation quality
**Setup:** Generate 2048 tokens with `chunk_size=1024`, comparing:
- Vanilla GPT-2 (truncates context to last 1024)
- TITANS-MAG (slides attention, NMM carries state)

**Measure:**
- Human / LLM-judge eval for coherence beyond position 1024
- Perplexity on positions 1024-2048 of held-out text

**Expected:** TITANS-MAG shows visible coherence improvement; perplexity gap widens past 1024.

---

## 3. Ablations

Compare against a baseline run of §2.1 with each modification.

| Ablation | Hypothesis | Expected outcome |
|---|---|---|
| `nmm_spectral_norm=False` (no NS5) | Training will diverge or plateau | Loss spikes or NaN within 1K steps |
| `nmm_n_persistent=0` (no persistent tokens) | Negligible effect alone | <1% perplexity change |
| `nmm_depth=1` (linear memory) | Significant degradation | Paper ablation: L_M=2 ≫ 1 |
| `nmm_conv_kernel=1` (no temporal mixing) | Modest degradation | +1-2 perplexity per paper §4.4 |
| `nmm_expansion=1` (small hidden) | Modest degradation, 4× less state memory | Trade-off; should still beat vanilla |
| `use_swa=True, swa_window=256` | Acceptable for from-scratch | Verify training is stable; not for fine-tune |
| `finetune_mode=False` with pretrained init | Breaks at init | Logits do NOT match HF; loss spikes at step 0-100 |
| NMM frozen, GPT-2 trained | Should still work but no memory benefit | Matches vanilla fine-tune |
| GPT-2 frozen, NMM trained (DON'T) | Fails per Titans Revisited | Loss plateaus, NMM doesn't learn |

The last row is informative: it's the "frozen backbone" experiment that fails per Di Nepi et al. 2025. Run it once to confirm; never as a default.

---

## 4. Scaling experiments (optional, expensive)

### 4.1 Backbone size sweep
- gpt2_small, gpt2_medium, gpt2_large, (gpt2_xl if budget allows)
- Same training data and total tokens
- Measure: does the NMM benefit scale with backbone size?

### 4.2 NMM capacity sweep
- Fix backbone at gpt2_small, vary `nmm_expansion ∈ {1, 2, 4, 8}`
- Measure: how does perplexity vs. NMM state memory trade off?

### 4.3 Chunk size sweep
- `chunk_size ∈ {256, 512, 1024}`
- Larger chunk → more in-chunk attention, smaller chunk → more reliance on NMM
- Measure: long-context perplexity (positions ≥ chunk_size)

---

## 5. Performance benchmarks

### 5.1 Throughput
- Tokens/sec at `gpt2_small`, `batch_size=8`, `chunk_size=1024`, single A100
- Baseline: vanilla GPT-2 with same config (no NMM)
- Expected: TITANS-MAG runs at 0.4-0.7× vanilla throughput (the inner loop is the bottleneck)
- If <0.3×, the inner loop has a bug — check that the chunk-forward runs the conv on the full chunk, not per-token (G154)

### 5.2 Memory footprint
- Peak GPU memory during a 1024-token chunk forward+backward
- Baseline: vanilla GPT-2 same config
- Expected: +650 MB for NMM state at gpt2_small (~3× vanilla activations); +grad checkpointing reduces backward peak by ~50%

### 5.3 Generation latency
- Time per token at T=1 step, with KV cache
- Compare to vanilla GPT-2 generation
- Expected: TITANS-MAG ~2-3× slower per token due to NMM update; Phase 6 scan brings this down for batched generation but doesn't help T=1

---

## 6. What to track per run

Every training run should log:
- Step, train loss, val loss
- Per-group grad norms (4 groups)
- Per-layer NMM state norms (`compute_nmm_norm(state)`)
- LR per group
- Tokens/sec, GPU memory
- NaN-skip count (should be 0 in a healthy run)
- Doc-boundary reset count (should match input data)

W&B / TensorBoard recommended. Without per-group grad norms you can't diagnose §RUNBOOK "NMM not learning".

---

## 7. Success criteria (the actual bar)

The implementation is **validated** when all of these hold:

- [ ] All §1 gates pass (1.1–1.6 minimum; 1.7 if Phase 6 implemented)
- [ ] §2.1 shows ≥3% perplexity improvement over vanilla fine-tune at matched compute
- [ ] §2.2 trains stably for 50K steps from scratch without NaN
- [ ] §3 ablations show the expected qualitative behaviors (no_spectral_norm diverges, frozen_backbone fails, etc.)
- [ ] §5.1 throughput is within 0.4-0.7× vanilla (not slower than that)

Anything below this bar means there's an unidentified bug or the implementation is not exercising the NMM properly. Don't ship until all are green.

---

## 8. What we are NOT testing here

Out of scope for this implementation's validation:
- TITANS MAC or MAL variants — only MAG
- Multi-head NMM — not in paper, optional
- Token-level meta-learning beyond per-token surprise — paper's setup only
- Tasks beyond next-token prediction (classification, RL) — language modeling only

For these, fork the repo and extend.
