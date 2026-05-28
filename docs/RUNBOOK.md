# Runbook — When Training Breaks

Operational guide for diagnosing and recovering from common failures. Each section names the symptom, the most likely root cause, the verification step, and the fix.

When in doubt: read **§ First moves** first.

---

## First moves (always do these)

Before diving into a specific failure mode:

1. **Check the git diff vs. last known-good.** Did anything change in `model/`, `train.py`, or `config.py`?
2. **Capture the config**: log `TitansConfig` at the top of every run. If you don't know what was running, you can't reproduce.
3. **Save the last NMM state norms.** `compute_nmm_norm(state)` returns one float per layer. A sudden jump from O(1) → O(100) is the canary for what's about to crash.
4. **Rule out the loader.** Print a few tokens of `idx[0, :10]` and confirm `doc_boundaries` is plausible (mostly False, occasionally True at document starts).

---

## NaN loss

**Symptom.** Loss prints `nan` or `inf`. Gradients then propagate the NaN to every parameter and the model is dead.

### Diagnose
```python
# in train_step, after forward:
if not torch.isfinite(loss):
    print(f"NaN at step {step}")
    print(f"  logits min/max: {logits.min().item()}, {logits.max().item()}")
    print(f"  nmm_state norms: {compute_nmm_norm(nmm_states)}")
    print(f"  last grad norm: {grad_norm_history[-3:]}")
```

### Most likely causes (in order)

1. **Newton-Schulz overflow under bf16 autocast.** The single most common silent killer. `.float()` inside the NS5 iteration is *not enough* — ambient `autocast` re-casts matmul inputs to bf16. Wrap the iteration in `torch.amp.autocast(device_type=device.type, enabled=False)`.

   **Verify:** put a `print(G.dtype)` inside the iteration. If it says `bfloat16`, that's it.

2. **`θ_t` applied pre-NS.** NS divides by the Frobenius norm — pre-scaling by θ cancels exactly. Then S has no LR control and blows up.

   **Verify:** Read the NMM step code. The order must be `g̃ = NS(g); S = η·S - θ·g̃` — NOT `g̃ = NS(θ·g)` or `g = θ·∇ℓ; g̃ = NS(g)`.

3. **Inner-loss reduction mismatched with `nmm_spectral_norm`.** If `nmm_spectral_norm=False` and reduction is still `'sum'`, gradients are `d_model×` too large. See `CONFIG_REFERENCE.md` — flip both together.

4. **Backward in bf16.** Backward + clip + step must run in fp32 even under bf16 autocast. The forward exits the autocast context before `loss.backward()`.

### Recovery (in-loop)

`train_step` should already have the NaN-skip path:

```python
if not torch.isfinite(loss):
    optimizer.zero_grad(set_to_none=True)
    return loss, None, None        # caller re-initializes nmm_states
```

The caller MUST reset `nmm_states` to `None` after a NaN — otherwise the corrupted state persists and the next step NaNs again immediately. Also reset in the DDP accumulation block.

### Recovery (post-mortem)

If params themselves went NaN, the run is dead. Resume from the last clean checkpoint with `nmm_states=None`.

---

## Logit parity broken at init

**Symptom.** Right after `load_pretrained`, the model's logits do not match HF GPT-2 to <1e-4 max diff (with NMM zeroed and `N_p=0`).

### Diagnose
```python
import torch
from transformers import GPT2LMHeadModel
hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
ours = build_and_load_pretrained(cfg).eval()
ours.set_nmm_zero()             # out_scale=0, N_p=0
x = torch.tensor([[1,2,3,4,5]])
with torch.no_grad():
    diff = (hf(x).logits - ours(x)[0]).abs().max()
print(diff)                      # must be < 1e-4
```

### Most likely causes

1. **Conv1D transpose wrong.** HF stores attention weights as `[in, out]`; we store `[out, in]`. `load_pretrained` must transpose `attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight`, `mlp.c_proj.weight`.

2. **`out_scale ≠ 0` for the parity test.** Default is zero only when `finetune_mode=True`. If you built with `finetune_mode=False`, `out_scale` is ones and the NMM contributes from step 0.

3. **`N_p ≠ 0` for the parity test.** Persistent tokens change the attention mask shape. Set `nmm_n_persistent=0` for parity.

4. **HF model name mismatch.** `n_embd=768` must pull `openai-community/gpt2` (small), not `gpt2-medium`. Derive from `n_embd`, not a hardcoded string.

5. **wte/lm_head tie missing.** `lm_head = wte.weight.T`. Without the tie, the head is random init and logits diverge instantly.

6. **Residual scale init wrong.** `attn.proj.weight.std() ≈ 0.02/√(2·n_layer)`. If you copied without applying the scale, residuals saturate.

### Fix
Patch the specific transpose / tie / scale, re-run the parity test. Add it to CI (TEST_PLAN.md §9).

---

## DDP hang

**Symptom.** Multi-rank training stops printing. `nvidia-smi` shows GPUs at 0% util. After a while NCCL times out.

### Most likely causes

1. **Missing `model.no_sync()` during gradient accumulation.** Without it, all-reduce fires on every micro-batch and ranks desync.

2. **Partial-cycle off-by-one.** The check must be:
   ```python
   is_partial_cycle = (batch is None) and (accum_i > 0)
   # NOT: accum_i < ACCUM_STEPS - 1
   ```
   When `StopIteration` fires at iteration `K-1`, the off-by-one version routes one rank into the no_sync branch while the other does not → all-reduce mismatch → hang.

3. **Per-rank seed identical** (inverse). If ranks have the same seed, dropout masks are identical and there's nothing to all-reduce — looks like a hang but is actually working. **Verify by checking gradient diversity across ranks.**

4. **NCCL communicator already destroyed** by an earlier exception path that didn't go through `try/finally`. Subsequent ranks block waiting on a dead communicator.

### Diagnose
```bash
# attach to the hung process and dump py-spy
py-spy dump --pid <hung_rank_pid>
```
Look for `dist.all_reduce` in the stack — that confirms a collective wait.

### Fix
- Verify `model.no_sync()` wraps non-final accumulation steps.
- Verify `is_partial_cycle` uses the `batch is None` form.
- Verify `try/finally: dist.destroy_process_group()` wraps the entire train loop.
- Add the indentation static check to CI — inconsistent indent in the try body can silently skip cleanup.

---

## Resuming a training run

Use `--resume-from PATH` on `cli/finetune.py` (or `train.py`). The flag accepts a step-N checkpoint (`step_NNNNNNN.pt`) or `latest.pt` and restores:

- **Model weights** via `load_state_dict` (with `_unwrap` to strip `_orig_mod.` / `module.` prefixes from compile/DDP wrapping).
- **Optimizer state** — m, v moments and step counter survive across resume. Including for 8-bit AdamW (bitsandbytes).
- **Step counter** — set to `saved_step + 1` so the resume picks up at the next cycle, not the saved one. (The saved value is off-by-one with respect to "completed cycles" — `save_checkpoint` records the counter before the post-cycle `step += 1` increment.)
- **Config** — reconstructed from `dataclasses.asdict(config)` via `TitansConfig.from_dict`, which silently drops keys listed in `_REMOVED_CONFIG_KEYS` so checkpoints from older schemas still load.

Architecture-affecting CLI flags (`--size`, `--chunk-size`, `--nmm-*`) are IGNORED in resume mode (warning printed to stderr). Changing them would invalidate the loaded optimizer state's param-group shape. Scaffolding flags (`--max-steps`, `--save-dir`, `--save-every`, `--warmup-steps`, `--grad-accum`) stay user-controlled — you can extend a run, redirect saves, or change effective batch size.

Loader state is NOT restored — the data iterator restarts at the corpus head on each `run_training` call. For multi-epoch training where the loader gets recycled anyway, this is a wash; for partial-epoch resume you re-read the same head-of-corpus chunks once. Document caveat, not a bug.

### Examples

Extend a finished 5000-step run by 5000 more:
```bash
uv run python cli/finetune.py \
    --resume-from ckpts/titans/latest.pt \
    --data corpus.txt \
    --max-steps 10000 --save-dir ckpts/titans
```

Recover from a crash mid-training:
```bash
uv run python cli/finetune.py \
    --resume-from ckpts/titans/latest.pt \
    --data corpus.txt \
    --max-steps 5000 --save-dir ckpts/titans \
    --compile-model --optim8bit  # same flags as the original
```

The `[finetune] resumed from ... (saved at step N, continuing from cycle N+1)` line on stderr confirms the resume worked.

---

## LR deflation across resumes

**Symptom.** Each time you resume from a checkpoint, the effective LR is lower than the previous run. After 3-4 resumes the model barely trains.

### Cause
`base_lrs` was read from `optimizer.param_groups[i]['lr']` *after* `optimizer.load_state_dict(...)`. The loaded state contains the current (scheduled-down) LR, not the original peak. Each resume re-anchors the schedule on the deflated value.

### Verify
```python
print(f"base_lrs after load: {base_lrs}")
print(f"expected (constants): {[BASE_LR_GPT2, BASE_LR_GPT2, BASE_LR_NMM, BASE_LR_NMM]}")
```

### Fix
Source `base_lrs` from the constants module:
```python
from constants import BASE_LR_GPT2, BASE_LR_NMM
base_lrs = [BASE_LR_GPT2, BASE_LR_GPT2, BASE_LR_NMM, BASE_LR_NMM]
```
NEVER:
```python
base_lrs = [g['lr'] for g in optimizer.param_groups]  # BUG
```

---

## OOM during forward

**Symptom.** `CUDA out of memory` at chunk forward, despite working batches earlier.

### Most likely causes (in order)

1. **Sequential per-token NMM at long T.** The sequential path
   (`nmm_block_size=1`) retains the full per-token autograd graph for
   the chunk. On a 16 GiB consumer card it tops out around
   `B=1, T≈64-128`. **Fix:** switch to the blockwise path
   (`nmm_block_size >= 16`) — faster (TC engagement) and the per-block
   transient is `T / block_size` smaller than the per-token one.

2. **fp32 NMM state.** Per-step `(M, S)` buffers in fp32 dominate at long
   T. **Fix:** set `nmm_state_dtype="bf16"` to halve them, or
   `"int8"` (blockwise-only) to quarter them. NS5 still runs in fp32
   internally (invariant preserved).

3. **NMM state base size too big.** State per layer scales with `B·d²`.
   **Fix:** `nmm_low_rank=64` factors `memory_mlp` weights and shrinks
   per-step state ~10×. Use `nmm_expansion=1` for an additional ~4×
   reduction (paper ablation — minor capacity loss).

4. **`chunk_size` too large.** Even with all NMM knobs on, very long
   chunks blow up activations. Halve `chunk_size`. T=1024 + B>1 on a
   16 GiB consumer card needs `nmm_low_rank=64` + blockwise + bf16.

5. **Stuck reference to old `nmm_states`.** Detaching between chunks
   releases the graph; not detaching means the graph keeps every
   chunk's intermediates pinned. Verify `detach_states` is actually
   called between TBPTT chunks.

### Diagnose
```python
print(torch.cuda.memory_summary(device=0, abbreviated=True))
```

Look for "Active memory" — if it's much larger than "Allocated memory" expectation, you have an autograd graph leak (probably detach missing).

### Recovery checklist (in fix-cost order)

```python
# 1. Switch to the blockwise NMM path — biggest single-knob win at long T.
#    Approximate (paper's per-token M_{t-1} becomes per-block M_{block-1})
#    but trains stably and engages tensor cores via batched matmul.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_block_size=64,
    nmm_state_dtype="bf16",
)

# 2. The unlock for T=1024 + B>1 on a 16 GiB card: low-rank NMM.
#    Factors memory_mlp weights so per-step state is ~10x smaller.
#    Loses some NMM capacity vs paper full-rank; measure loss vs
#    baseline before committing.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_block_size=64,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
)

# 3. If you need MAX speed at T=1024 and accept lower NMM capacity
#    (paper applies NMM at every block — reducing to 4-of-12 cuts
#    NMM-recompute cost ~3x).
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_block_size=64,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
    nmm_layer_indices=[0, 3, 6, 9],
)

# 4. Inner-loop compile: 1.7-1.9x step-time speedup with
#    paper-faithful sequential semantics. Stacks on top of all of the
#    above. First step pays a one-time torch.compile cost (~30-60s).
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_block_size=64,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
)
# Blockwise throughput at gpt2_small, RTX 5070 Ti, T=1024:
#   block_size=64  -> 1.88 s/step ( 45x over sequential), 50k steps in ~31 h
#   block_size=128 -> 0.99 s/step ( 86x), 50k-steps in 14h
#   block_size=256 -> 0.53 s/step (160x), 50k-steps in 7.5h
#   block_size=512 -> 0.31 s/step (274x), 50k-steps in 4.3h
```

The same knobs are exposed as `--nmm-*` flags on `train.py` and
`cli/finetune.py` — no need to edit the script. The recommended
**consumer-GPU default** (full-rank, 16 GiB VRAM, T=1024) is:

```bash
python -m cli.finetune --data corpus.txt \
    --chunk-size 1024 --batch-size 1 --grad-accum 16 \
    --nmm-block-size 64 \
    --nmm-state-dtype bf16 \
    --nmm-detach-state-between-blocks \
    --nmm-use-gram-ns5 \
    --freeze-embeddings \
    --nmm-gate-ramp-steps 100 \
    --nmm-gate-ramp-target 0.1 \
    --compile-model \
    --optim8bit
```

The two training-regime flags address things found in needle-in-
haystack diagnostics:

- `--freeze-embeddings`: TPTT-style freeze of `wte`, `wpe`, `ln_f` so
  the limited fine-tune data doesn't distort the input/output
  representations. Transformer blocks (attention + MLP) stay trainable
  so they can adapt to attend to NMM-modulated tokens. (There's also
  `--freeze-backbone` for a more aggressive freeze, but empirically it
  collapses short-distance accuracy because attention can't compensate
  for the injected NMM signal.)
- `--nmm-gate-ramp-*`: forces the memory gate open on a schedule
  instead of waiting for LM loss alone to slowly open it.

The `--nmm-aux-loss-weight α` flag (direct supervision on `y_mem`)
exists but is **NOT** in the default recipe. It was tested at α=0.5 on
the needle corpus; the result was that short-distance accuracy
saturated but long-distance recall stayed at chance, and the
diff-under-swap magnitude at long distance actually _decreased_ — the
calibrated optimizer correctly suppressed noisy `y_mem` rather than
manufacturing signal that wasn't there. Preserved as a diagnostic
tool. See `CLAUDE.md` for the detailed analysis.

Drop the flags entirely for the original full-fine-tune behavior.

Measured on RTX 5070 Ti without `--nmm-use-gram-ns5`: ~1.11 s/step,
8.5 GiB peak. The `--nmm-use-gram-ns5` flag swaps NS5 for the
Gram-iteration variant (Tri Dao et al., POLAR_EXPRESS coefficients).
Implemented locally in `model/nmm.py` — no external dep.
`--nmm-use-cans` is the alternative for a quality-favouring tradeoff;
drop both for paper-faithful NS5.

The `--nmm-detach-state-between-blocks`
flag is what makes full-rank fit at T=1024 — it bounds the backward
graph to a single block. Tradeoff: outer NMM-related params learn from
64-token windows instead of full-chunk BPTT. For standard LM training
this is fine; truncated BPTT is long-established practice.

If you find detach materially hurts your task, the alternatives are:
- `--nmm-low-rank 64` (factor MemoryMLP weights, lose some capacity)
- `--nmm-layer-indices 3,8` (NMM on a subset of blocks)
- a 24 GiB+ GPU (eliminates the constraint entirely)

---

## Long-context generation drift

**Symptom.** Generation quality degrades sharply past `block_size` (1024 tokens for GPT-2 small). Sometimes degrades earlier.

### Causes

1. **Conv window not maintained at T=1 step.** The depthwise conv is stateless (intentionally — not in `(M, S)`). At training time the conv sees a `kernel_size`-token window; at T=1 step it sees a 1-token window. Mitigation: keep a rolling conv buffer externally, see `diagrams/inference_sequence.mmd`.

2. **Position embedding past `block_size`.** GPT-2's wpe table only covers 0..1023. Chunks reset positions, but within a single forward call you cannot exceed `block_size`. For long generation, chunk the prompt + carry NMM state.

3. **From-scratch model with `chunk_size < block_size`.** Untrained wpe rows beyond `chunk_size` silently degrade long-context generation. The config emits a `warnings.warn` for this case — check the run logs.

4. **(FIXED in v2 via Option B — KV cache + `step_with_conv` for NMM.)** Previously `generate.py` re-fed the entire `block_size` window through the NMM at every decoded token, compounding state updates ~`block_size`× per generated token. The current `generate.py` uses `model.prepare_decode` + `model.forward_step`: each decoded token gets exactly one NMM update via `step_with_conv` (full k-token conv context via a conv buffer) and one attention pass via KV cache. Per-step decode cost is now O(1) for NMM and O(T) for attention, matching standard transformer decoding. `evaluation.needle_in_haystack` was on the same broken path in v1 and was ported to the cached pipeline in the same fix; a call-counting regression test (`tests/integration/test_needle_smoke.py::test_needle_in_haystack_uses_cached_decode_path`) defends against accidental reintroduction. If you're debugging older checkpoints/scripts that still re-feed the window, port to the new entry points or accept the drift.

### Fix
- Chunk long prompts through the model — never feed >`block_size` tokens in one forward (the new `generate.py` handles this automatically via the chunked warm-up path).
- For from-scratch runs intended for long context, set `chunk_size = block_size`.
- For #1's conv-window limitation, the new `NMM.step_with_conv` + per-step conv buffer mitigates it at decode time — the conv now sees a full k-token window via the buffer.

---

## Rank divergence (silent)

**Symptom.** Training appears to work but eval loss is much worse than expected, or different runs at the same step give wildly different results.

### Causes

1. **Same seed on all ranks (inverse of).** Dropout masks must diverge across ranks; if they don't, you're effectively training with a much smaller effective batch.

2. **Different config across ranks.** Verify `dist.barrier()` after config load, and that all ranks read the same config file.

3. **DDP `find_unused_parameters=True`** (when not strictly needed). Some params getting different gradients across ranks → all-reduce inconsistency.

### Diagnose
```python
# after a forward, before backward:
for n, p in model.named_parameters():
    if p.grad is not None:
        max_diff = collect_max_across_ranks(p.grad)
        if max_diff > 1e-3:
            print(f"divergence in {n}: {max_diff}")
```

### Fix
- Per-rank seed: `torch.manual_seed(base_seed + rank)` AFTER model construction.
- Set `find_unused_parameters=False` unless you genuinely need it (it's the default we recommend).

---

## NMM not learning (loss plateaus immediately)

**Symptom.** Training loss is identical to GPT-2 baseline loss for hundreds of steps with no improvement.

### Causes

1. **`out_scale` stuck at zero.** In `finetune_mode=True`, init is zero. It must be in the `nmm_no_decay` group AND receiving gradients. Verify:
   ```python
   for n, p in model.named_parameters():
       if 'out_scale' in n:
           print(n, p.requires_grad, p.grad.abs().max() if p.grad is not None else None)
   ```

2. **NMM is in the wrong optimizer group.** Routing rule: `'nmm' in name` → NMM groups. If your block exposes the NMM under a different attribute name, the router misses it and the NMM trains at GPT-2 LR (3× too low).

3. **NMM gradients zero.** Inner-loop updates produce per-token weight changes, but those propagate to `memory_mlp.W*.weight` ONLY through the outer-loop `loss.backward()`. If the autograd graph through the chunk is broken (e.g., `detach()` called inside the chunk by accident, not between chunks), the outer optimizer never sees NMM gradients.

### Diagnose
```python
# after backward:
nmm_grad_norm = sum(p.grad.norm()**2 for n, p in model.named_parameters() if 'nmm' in n)**0.5
gpt2_grad_norm = sum(p.grad.norm()**2 for n, p in model.named_parameters() if 'nmm' not in n)**0.5
print(f"NMM/GPT-2 grad norm ratio: {nmm_grad_norm/gpt2_grad_norm}")
```

If the ratio is ~0 the NMM is disconnected from the graph.

---

## Checkpoint won't load

**Symptom.** `torch.load` raises, or `load_state_dict` raises `Unexpected key(s)` / `Missing key(s)`.

### Causes

1. **`weights_only=True`.** Our checkpoint contains nested dicts (optimizer state, scheduler state, step). `torch.load(..., weights_only=False)` is required.

2. **`_orig_mod.` prefix from `torch.compile`.** If the checkpoint was saved on a compiled model, all keys are prefixed. Save via `_unwrap(model).state_dict()` to strip.

3. **Missing `'optimizer'` key.** HF-initialized checkpoints have no optimizer state. Resume must tolerate this — only load optimizer if the key exists.

4. **NMM state in checkpoint.** It shouldn't be there. If a forked checkpoint format includes `nmm_states`, ignore them on load and reset via `init_state`.

### Fix
Per case above. The canonical resume order:
1. Build model from config
2. `model.load_state_dict(ckpt['model'])`
3. Wrap with DDP
4. Build optimizer
5. `optimizer.load_state_dict(ckpt['optimizer'])` if key exists
6. `model.train()`

---

## When all else fails

- Re-run with `nmm_spectral_norm=False`, `finetune_mode=True`, `N_p=0`, `chunk_size=block_size`. This isolates the NMM down to its minimal contribution. If training is still broken, the bug is in the GPT-2 path.
- Run the parity test (TEST_PLAN.md §9). If it fails, weight loading is wrong, not training.
- Run `pytest tests/unit/ -x` to localize to a single failing component.
