# Runbook — When Training Breaks

Operational guide for diagnosing and recovering from common failures. Each section names the symptom, the most likely root cause, the verification step, and the fix. Gap IDs (Gnnn) link to `GAP_HISTORY.md` for the original incident.

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

1. **Newton-Schulz overflow under bf16 autocast (G226).** The single most common silent killer. `.float()` inside the NS5 iteration is *not enough* — ambient `autocast` re-casts matmul inputs to bf16. Wrap the iteration in `torch.amp.autocast(device_type=device.type, enabled=False)`.

   **Verify:** put a `print(G.dtype)` inside the iteration. If it says `bfloat16`, that's it.

2. **`θ_t` applied pre-NS (G140).** NS divides by the Frobenius norm — pre-scaling by θ cancels exactly. Then S has no LR control and blows up.

   **Verify:** Read the NMM step code. The order must be `g̃ = NS(g); S = η·S - θ·g̃` — NOT `g̃ = NS(θ·g)` or `g = θ·∇ℓ; g̃ = NS(g)`.

3. **Inner-loss reduction mismatched with `nmm_spectral_norm` (G160).** If `nmm_spectral_norm=False` and reduction is still `'sum'`, gradients are `d_model×` too large. See `CONFIG_REFERENCE.md` — flip both together.

4. **Backward in bf16 (G159).** Backward + clip + step must run in fp32 even under bf16 autocast. The forward exits the autocast context before `loss.backward()`.

### Recovery (in-loop)

`train_step` should already have the NaN-skip path (G213, G217):

```python
if not torch.isfinite(loss):
    optimizer.zero_grad(set_to_none=True)
    return loss, None, None        # caller re-initializes nmm_states
```

The caller MUST reset `nmm_states` to `None` after a NaN — otherwise the corrupted state persists and the next step NaNs again immediately. Also reset in the DDP accumulation block (G217).

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

4. **HF model name mismatch.** `n_embd=768` must pull `openai-community/gpt2` (small), not `gpt2-medium`. Derive from `n_embd`, not a hardcoded string (G216).

5. **wte/lm_head tie missing.** `lm_head = wte.weight.T`. Without the tie, the head is random init and logits diverge instantly.

6. **Residual scale init wrong.** `attn.proj.weight.std() ≈ 0.02/√(2·n_layer)`. If you copied without applying the scale, residuals saturate.

### Fix
Patch the specific transpose / tie / scale, re-run the parity test. Add it to CI (TEST_PLAN.md §9).

---

## DDP hang

**Symptom.** Multi-rank training stops printing. `nvidia-smi` shows GPUs at 0% util. After a while NCCL times out.

### Most likely causes

1. **Missing `model.no_sync()` during gradient accumulation (G200).** Without it, all-reduce fires on every micro-batch and ranks desync.

2. **Partial-cycle off-by-one (G222).** The check must be:
   ```python
   is_partial_cycle = (batch is None) and (accum_i > 0)
   # NOT: accum_i < ACCUM_STEPS - 1
   ```
   When `StopIteration` fires at iteration `K-1`, the off-by-one version routes one rank into the no_sync branch while the other does not → all-reduce mismatch → hang.

3. **Per-rank seed identical** (G204 inverse). If ranks have the same seed, dropout masks are identical and there's nothing to all-reduce — looks like a hang but is actually working. **Verify by checking gradient diversity across ranks.**

4. **NCCL communicator already destroyed** by an earlier exception path that didn't go through `try/finally` (G225). Subsequent ranks block waiting on a dead communicator.

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
- Add the indentation static check (G227) to CI — inconsistent indent in the try body can silently skip cleanup.

---

## LR deflation across resumes

**Symptom.** Each time you resume from a checkpoint, the effective LR is lower than the previous run. After 3-4 resumes the model barely trains.

### Cause (G162)
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

1. **`_forward_chunk_sequential`'s per-token autograd graph.** This is the dominant OOM mode at `chunk_size >= 64` on consumer GPUs. The per-token NMM loop retains `(M_t, S_t, g_t)` snapshots for every t, growing linearly with `chunk_size`. **Fix:** set `nmm_grad_checkpoint=True` (and tune `nmm_grad_checkpoint_segment_len` — start at 64, drop to 32 or 16 if it still OOMs). Backward recomputes intermediates instead of storing them. See `docs/CONFIG_REFERENCE.md` "Memory-saving knobs".

2. **fp32 NMM state.** With checkpoint on, the fp32 `(M, S)` snapshots at segment boundaries plus inner-loop intermediates dominate. **Fix:** set `nmm_state_dtype="bf16"` to halve them. Combines with the checkpoint flag multiplicatively. NS5 still runs in fp32 internally (G226 invariant preserved).

3. **NMM state base size too big.** State per layer is `~24·B·d²·dtype_bytes`. For `gpt2_medium` (`d_model=1024`) at `B=8` fp32: ~6 GB across 24 layers on top of model weights, AdamW moments, and activations. Try `nmm_expansion=1` (halves hidden dim) or smaller `B`.

4. **`chunk_size` too large.** Even with both knobs above on, very long chunks blow up activations. Halve `chunk_size`. On a 16 GiB consumer card with `gpt2_small`, expect to fit `T` up to ~256 with bf16 + ckpt seg=16; `T=1024` realistically needs ≥ 24 GiB.

5. **Stuck reference to old `nmm_states`.** Detaching between chunks releases the graph; not detaching means the graph keeps every chunk's intermediates pinned. Verify `detach_states` is actually called between TBPTT chunks.

### Diagnose
```python
print(torch.cuda.memory_summary(device=0, abbreviated=True))
```

Look for "Active memory" — if it's much larger than "Allocated memory" expectation, you have an autograd graph leak (probably detach missing).

### Recovery checklist (in fix-cost order)

```python
# 1. Enable checkpointing first — biggest win, no accuracy risk.
cfg = TitansConfig.gpt2_small(
    chunk_size=T,
    block_size=T,
    nmm_grad_checkpoint=True,
    nmm_grad_checkpoint_segment_len=32,
)

# 2. Add bf16 state if still OOM. Accuracy risk is minor but measure
#    loss curves vs fp32 baseline before committing to it.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=16,
    nmm_state_dtype="bf16",
)

# 3. Reduce capacity if still OOM.
cfg = TitansConfig.gpt2_small(
    chunk_size=T // 2, block_size=T // 2,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=16,
    nmm_state_dtype="bf16",
    nmm_expansion=1,
)

# 4. Last resort if you NEED long T and don't care about step time:
#    enable CPU-offload of the checkpoint boundaries. Adds 5-10x to
#    step time at long T due to PCIe transfers + extra recompute.
#    Suitable for correctness work, not production training.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_cpu_offload_segments=True,
)

# 5. Alternative to #4: block-level checkpointing. Wraps each
#    TitansMAGBlock.forward in a recompute boundary so only block I/O
#    lives across the stack. Comparable VRAM win to cpu_offload at the
#    same step-time scale (~5-10x slower than no checkpointing).
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_block_grad_checkpoint=True,
)

# 6. The unlock for T=1024 + B>1 on a 16 GiB card: low-rank NMM.
#    Factors memory_mlp weights so per-step state is ~10x smaller.
#    Loses some NMM capacity vs paper full-rank; measure loss vs
#    baseline before committing.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
)

# 7. If you need MAX speed at T=1024 and accept lower NMM capacity
#    (paper applies NMM at every block — reducing to 4-of-12 cuts
#    NMM-recompute cost ~3x).
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
    nmm_layer_indices=[0, 3, 6, 9],
)

# 8. Inner-loop compile (G264a): 1.7-1.9x step-time speedup with
#    paper-faithful sequential semantics. The single most impactful
#    speed knob — recommended for any T >= 256 training run.
#    First step pays a one-time torch.compile cost (~30-60s);
#    subsequent steps are fast.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
    nmm_compile_inner_loop=True,   # the speedup
    nmm_fused_kernel=True,         # +5% on top (analytical inner gradient)
)
# T=1024 measured: 158s/step (ref) -> 83s/step (combined). 1.90x.
# T=256 measured:   40s/step (ref) -> 21s/step (combined). 1.90x.

# 9. Blockwise NMM (G266): chunk-as-update aggregation — ONE memory
#    update per `nmm_block_size` tokens, not per token. Per-block
#    forward becomes a batched matmul (TC engages). Approximate
#    (paper's per-token M_{t-1} replaced by per-block M_{block-1})
#    but trains stably and unlocks REAL training throughput on
#    consumer hardware. This is the recommended path for any
#    serious training run at T >= 256.
cfg = TitansConfig.gpt2_small(
    chunk_size=T, block_size=T,
    nmm_grad_checkpoint=True, nmm_grad_checkpoint_segment_len=32,
    nmm_state_dtype="bf16",
    nmm_low_rank=64,
    nmm_block_size=64,             # 45x faster, 16 blocks per T=1024 chunk
)
# T=1024 measured: 83s/step (sequential best) -> 1.88s/step. 45x.
# Loss trajectory @ blk=64 over 20 steps on fixed batch: 10.94 -> 6.94.
# 50k-step run: ~31 hours @ blk=64 (vs 7 weeks sequential).
# Larger block_size = faster but coarser approximation:
#   block_size=128 -> 0.99s/step ( 86x), 50k-steps in 14h
#   block_size=256 -> 0.53s/step (160x), 50k-steps in 7.5h
#   block_size=512 -> 0.31s/step (274x), 50k-steps in 4.3h
```

---

## Long-context generation drift

**Symptom.** Generation quality degrades sharply past `block_size` (1024 tokens for GPT-2 small). Sometimes degrades earlier.

### Causes

1. **Conv window not maintained at T=1 step.** The depthwise conv is stateless (intentionally — not in `(M, S)`). At training time the conv sees a `kernel_size`-token window; at T=1 step it sees a 1-token window. Mitigation: keep a rolling conv buffer externally, see `diagrams/inference_sequence.mmd`.

2. **Position embedding past `block_size`.** GPT-2's wpe table only covers 0..1023. Chunks reset positions, but within a single forward call you cannot exceed `block_size`. For long generation, chunk the prompt + carry NMM state (G176).

3. **From-scratch model with `chunk_size < block_size`.** Untrained wpe rows beyond `chunk_size` silently degrade long-context generation. The config emits a `warnings.warn` for this case — check the run logs.

4. **(FIXED in v2 via Option B — KV cache + `step_with_conv` for NMM.)** Previously `generate.py` re-fed the entire `block_size` window through the NMM at every decoded token, compounding state updates ~`block_size`× per generated token. The current `generate.py` uses `model.prepare_decode` + `model.forward_step`: each decoded token gets exactly one NMM update via `step_with_conv` (full k-token conv context via a conv buffer) and one attention pass via KV cache. Per-step decode cost is now O(1) for NMM and O(T) for attention, matching standard transformer decoding. `eval.needle_in_haystack` was on the same broken path in v1 and was ported to the cached pipeline in the same fix; a call-counting regression test (`tests/integration/test_needle_smoke.py::test_needle_in_haystack_uses_cached_decode_path`) defends against accidental reintroduction. If you're debugging older checkpoints/scripts that still re-feed the window, port to the new entry points or accept the drift.

### Fix
- Chunk long prompts through the model — never feed >`block_size` tokens in one forward (the new `generate.py` handles this automatically via the chunked warm-up path).
- For from-scratch runs intended for long context, set `chunk_size = block_size`.
- For #1's conv-window limitation, the new `NMM.step_with_conv` + per-step conv buffer mitigates it at decode time — the conv now sees a full k-token window via the buffer.

---

## Rank divergence (silent)

**Symptom.** Training appears to work but eval loss is much worse than expected, or different runs at the same step give wildly different results.

### Causes

1. **Same seed on all ranks (inverse of G204).** Dropout masks must diverge across ranks; if they don't, you're effectively training with a much smaller effective batch.

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
- Per-rank seed: `torch.manual_seed(base_seed + rank)` AFTER model construction (G204).
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

1. **`weights_only=True`.** Our checkpoint contains nested dicts (optimizer state, scheduler state, step). `torch.load(..., weights_only=False)` is required (G168).

2. **`_orig_mod.` prefix from `torch.compile`.** If the checkpoint was saved on a compiled model, all keys are prefixed. Save via `_unwrap(model).state_dict()` to strip (G184).

3. **Missing `'optimizer'` key.** HF-initialized checkpoints have no optimizer state. Resume must tolerate this (G219) — only load optimizer if the key exists.

4. **NMM state in checkpoint.** It shouldn't be there. If a forked checkpoint format includes `nmm_states`, ignore them on load and reset via `init_state` (G198).

### Fix
Per case above. The canonical resume order (G209):
1. Build model from config
2. `model.load_state_dict(ckpt['model'])`
3. Wrap with DDP
4. Build optimizer
5. `optimizer.load_state_dict(ckpt['optimizer'])` if key exists
6. `model.train()` (G221)

---

## When all else fails

- Re-run with `nmm_spectral_norm=False`, `finetune_mode=True`, `N_p=0`, `chunk_size=block_size`. This isolates the NMM down to its minimal contribution. If training is still broken, the bug is in the GPT-2 path.
- Run the parity test (TEST_PLAN.md §9). If it fails, weight loading is wrong, not training.
- Run `pytest tests/unit/ -x` to localize to a single failing component.
- Read `GAP_HISTORY.md` for the symptom — 227 entries, search for keywords.
