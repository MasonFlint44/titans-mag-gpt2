# Glossary

Terms used across this codebase, in roughly conceptual order (foundations → variants → implementation details). For equations and architectural decisions see `ARCHITECTURE.md`; for the paper itself see Sun et al. 2025 (arXiv:2501.00663).

---

## Core TITANS terms

**TITANS.** Family of architectures introduced by Sun et al. 2025 that add a **Neural Memory Module** with online weight updates to a transformer backbone. Three variants — **MAG**, **MAC**, **MAL** — differ in how the memory is wired in.

**NMM (Neural Memory Module).** A small MLP whose weights act as recurrent state. The weights update at every token via gradient descent on a **surprise** signal (associative-memory loss). Distinct from a standard recurrent state vector: the *parameters* of the MLP are the state, not the activations.

**Surprise.** The associative-memory loss `ℓ_t = ‖f_M(k̂_t) - v_t‖²` (paper Eq. 12). When the current memory can't reconstruct `v_t` from `k̂_t`, the loss is high → large gradient → big update to `M`. The model learns more from things it didn't expect.

**Test-time learning.** The NMM continues updating during inference. This is the whole TITANS premise: memory adapts to the live context, not just the training corpus. Implementation: `torch.func.grad` operates independently of `torch.no_grad()`, so wrapping a `generate()` call in `eval()` does NOT freeze the NMM.

---

## TITANS variants (we implement MAG)

**MAG — Memory as a Gate.** Attention and memory branches combine via an element-wise learned gate: `o = silu(γ_a · y_attn) ⊗ silu(γ_m · y_mem)` (paper §4.2). This is what we implement.

**MAC — Memory as a Context.** The NMM output is prepended to the sequence as additional context tokens; attention then operates over a concatenated [memory, real] sequence. Not implemented here.

**MAL — Memory as a Layer.** The NMM is its own layer, applied sequentially before/after attention rather than in parallel. Not implemented here.

**Fine-tune-compatible MAG gate.** Our modification to the MAG formula: `o = y_attn + silu(γ_m · y_mem) * y_attn` (equivalent to `y_attn · (1 + silu(γ_m · y_mem))`). At init with `out_scale=0` → `y_mem≈0` → `o = y_attn` exactly. Preserves pretrained GPT-2 residual stream so fine-tuning starts from vanilla GPT-2 logits.

---

## NMM internals

**MemoryMLP.** The actual MLP whose weights are the recurrent state. Two-layer (`L_M=2`) SiLU-GLU: `h = silu(W1·x) ⊙ sigmoid(W_gate·x); out = norm(W2·h) + x`. The `{W1, W_gate, W2}` weights ARE the state; LayerNorm params are fixed (not recurrent).

**SiLU-GLU.** Gated MLP using SiLU activation and a sigmoid gate: `silu(W1·x) * sigmoid(W_gate·x)`. Distinct from **SwiGLU** which uses a linear (no-sigmoid) gate. Don't confuse them.

**ResidualNorm.** `norm(W2·h) + x` — applies LayerNorm to the W2 output then adds the residual. Stabilizes memory output scale (from lucidrains; not in paper).

**`out_scale`.** Learnable `[d_model]` parameter multiplied element-wise onto the NMM output. Init = zeros when `finetune_mode=True` (so memory contribution starts at zero exactly), ones when `False`. The only reliable way to silence the NMM at fine-tune init given that ResidualNorm passes `x` through even when `W2≈0` (G123).

**`θ_t, η_t, α_t`.** Per-token data-dependent scalars produced by three `Linear(d_model, 1)` projections of the input token:
- `θ_t = sigmoid(W_θ · x_t)` — inner-loop learning rate
- `η_t = sigmoid(W_η · x_t)` — momentum decay
- `α_t = sigmoid(W_α · x_t)` — forgetting rate

Per paper §3.2 ("functions of tokens"). All three are scalars `[B, T]` (not per-element).

**Momentum buffer `S`.** Per-layer recurrent state alongside `M`. Equation 14: `S_t = η_t · S_{t-1} - θ_t · g̃_t`. Same shapes as `M`.

**Persistent memory tokens (`N_p`).** Learned, input-independent tokens prepended to each block's input before attention. Paper Eq. 19. Distinct from the NMM — they're static parameters trained by the outer optimizer only. Default `N_p=4`. Per-block in our implementation (paper doesn't specify granularity).

---

## Update rule details

**Newton-Schulz spectral normalization (NS5).** Five-step iteration that drives a matrix's singular values toward 1 (spectral norm ≈ 1). Applied to the per-token gradient `g_t` to prevent inner-loop blow-up. Must run in fp32 with autocast disabled (G226). Transpose tall matrices before iteration, back after (G198) — NS converges on wide (cols ≥ rows) matrices.

**Write-then-read.** Retrieval ordering: update `M_{t-1} → M_t` first, then read `y_t = MemoryMLP(M_t; q̂_t)`. The current token's query sees the freshly-updated memory. Contrast with read-then-write (`y_t = MemoryMLP(M_{t-1}; q̂_t)`, paper Eq. 15) — both are valid; we follow lucidrains for marginally better behavior on associative recall tasks.

**`torch.func.grad` + `vmap`.** PyTorch functional autograd. Computes the per-sample gradient of the inner loss w.r.t. the memory weights without hand-coding the backward. Required because we want per-token gradients through a stateful inner loop, which standard autograd can't easily express.

**Functional call.** `torch.func.functional_call(module, params, x)` runs `module.forward` using `params` (a dict, not the module's own parameters). Lets us treat the MemoryMLP weights as a state we pass through, not parameters we mutate.

---

## Training mechanics

**TBPTT (Truncated BackPropagation Through Time).** Standard technique for training recurrent models: process a long sequence in chunks, backprop within each chunk, **detach** the state between chunks to bound the graph. Without detach, every chunk extends the autograd graph and memory blows up linearly.

**`detach_states`.** Method that calls `.detach()` on every leaf of the nested `(M, S)` state. Called between TBPTT chunks. Must handle `state=None` (G149).

**`chunk_size`.** Length of one TBPTT chunk. Bounded by `block_size` (GPT-2 position embedding table covers 0..1023). Positions reset to 0 at each chunk boundary; cross-chunk context lives in the NMM state, not in the attention.

**Doc boundary.** Boolean `[B, T]` tensor marking document starts within the chunk. The NMM resets `(M, S)` to init values at boundaries via `torch.where` (NOT in-place — autograd tensors). Prevents cross-document memory leakage.

**`ParallelStreamLoader`.** Our TBPTT-aware data loader. Maintains B independent sub-streams; position `i` of every batch in a row continues the same document stream across calls. A naive `DataLoader(shuffle=False)` does NOT give this property — see `diagrams/data_pipeline.mmd`.

**Gradient accumulation.** Compute K micro-batches of gradients, all-reduce once, optimizer step once. Under DDP, the first K-1 micro-batches run inside `model.no_sync()` to suppress all-reduce; the final one triggers the all-reduce (G200).

**`no_sync()`.** PyTorch DDP context manager that disables gradient all-reduce on backward. Used during gradient accumulation to defer sync to the last micro-batch.

**Partial cycle.** When `next(loader)` raises `StopIteration` mid-accumulation. The skip condition must be `(batch is None) and (accum_i > 0)`, NOT `accum_i < ACCUM_STEPS - 1` (G222) — the latter silently desyncs ranks.

---

## Precision and numerical

**bf16 autocast.** PyTorch mixed-precision context: forward ops run in bfloat16, backward and optimizer in fp32. Wraps the *forward* pass only; backward + clip + step must run in fp32 (G159).

**Autocast leakage.** When `.float()` inside an `autocast(enabled=True)` region is silently undone — matmul inputs get re-cast to bf16. Fix: wrap the explicitly-fp32 block in `autocast(enabled=False)` (G226). This is the single most common silent killer in the inner loop.

**`fp32 master weights`.** Optimizer keeps fp32 copies of the parameters; the bf16 cast is only for the forward computation. Standard mixed-precision training pattern.

---

## Inference

**Conv buffer.** Stateless — NOT part of `(M, S)`. At training time the chunk forward sees a full `kernel_size`-token window; at T=1 step it sees only 1 token. Mitigated at inference by maintaining a rolling conv window externally (`diagrams/inference_sequence.mmd`).

**KV cache.** Standard attention key/value cache for autoregressive generation. Separate from NMM state.

**Sliding window context strategy.** For generation past `block_size`, slide the attention window but keep the NMM state continuous. The NMM provides the long-range memory; attention covers local context.

---

## Optimizer

**4-group optimizer.** Params split into four groups: `gpt2_decay`, `gpt2_no_decay`, `nmm_decay`, `nmm_no_decay`. NMM groups use 3× the GPT-2 LR (paper ratio). `out_scale`, `gamma_*`, `persistent_mem` are no-decay (decay would shrink them toward zero).

**`base_lrs`.** The peak learning rates per group, fed to `apply_lr` to compute the scheduled LR. **Must come from code-level constants, not from `optimizer.param_groups[i]['lr']`** — the latter compounds LR deflation across resumes (G162).

**Cosine schedule with warmup.** Linear warmup for `warmup_steps`, then cosine decay to `min_ratio × peak` over `max_steps - warmup_steps`. Standard for transformer training.

---

## DDP

**Rank.** Process index in the distributed group. Rank 0 is conventionally the "main" rank that owns checkpoint saving and logging.

**`world_size`.** Total number of ranks (typically = number of GPUs).

**All-reduce.** Collective operation that sums gradients across ranks and broadcasts the result. Triggered automatically by DDP on `loss.backward()` unless inside `no_sync()`.

**Per-rank seed.** Each rank gets a different RNG seed (typically `base_seed + rank`) **set after model construction** so the model is identical across ranks but dropout masks diverge (G204).

---

## Implementation references

**G-number (Gnnn).** Audit gap identifier — e.g., G226 is the bf16-autocast leakage gap. See `GAP_HISTORY.md` for the full incident description, root cause, and fix.

**lucidrains/titans-pytorch.** Reference implementation by Phil Wang. Has many enhancements beyond the paper (spectral norm, `torch.func.grad`, ResidualNorm). **We do NOT depend on it as a runtime dependency** — implement from scratch, reference only.

**Titans Revisited.** Follow-up paper (Di Nepi et al. 2025, arXiv:2510.09551). Notable finding: frozen-backbone + NMM-only training fails. Hence our policy: do NOT freeze GPT-2 weights when fine-tuning.

**TPTT.** Different architecture (Zohar et al. 2025, arXiv:2506.17671) that uses DeltaProduct (Householder linear attention), NOT original NMM. Not a cross-reference for this work.
