# TITANS MAG GPT-2 — Implementation Roadmap

A condensed guide to implementing TITANS MAG on top of GPT-2. Read this end-to-end
before writing code; consult `PLAN.md` for full code sketches and the rationale
behind every non-obvious choice.

## Document map

| File | Role | When to read |
|---|---|---|
| `README.md` | Front door: install, train, generate, common-symptom pointers | Before anything else if new to the repo |
| `ARCHITECTURE.md` | Design decisions, equations, block diagram, file layout | First — understand what you're building |
| `ROADMAP.md` (this file) | Phase-by-phase task list with key gotchas | Second — plan your work |
| `PLAN.md` | Full implementation details, code sketches, every gap-driven safeguard | Reference per task as you write it |
| `TEST_PLAN.md` | Comprehensive test plan: unit/integration/parity/behavior/DDP/performance + regression matrix | When writing tests for each component |
| `CONFIG_REFERENCE.md` | Every `TitansConfig` knob with type, range, default, boundary behavior | When tuning hyperparameters or choosing factories |
| `RUNBOOK.md` | Failure-mode recovery: NaN loss, DDP hang, OOM, logit drift, LR deflation | When training breaks |
| `GLOSSARY.md` | TITANS terms (NMM, MAG, NS5, surprise, write-then-read, ...) | When a term in another doc is unfamiliar |
| `EXPERIMENTS.md` | Validation gates, headline experiments, ablations, success criteria | After implementation — proves the thing works |
| `GAP_HISTORY.md` | Audit log: 227 gaps over 53 review passes. Each entry traces a near-miss with its fix | Background — only consult when a `PLAN.md` snippet references a specific G-number you don't understand |
| `diagrams/architecture.mmd` | Component hierarchy: model → block → NMM → update loop | Visualizing the static structure |
| `diagrams/training_sequence.mmd` | Training loop: setup → accumulation → TBPTT → step → teardown | Understanding the dynamic training flow |
| `diagrams/inference_sequence.mmd` | Generation: prompt warm-up → autoregressive loop → test-time NMM learning | Understanding inference |
| `diagrams/nmm_state_lifecycle.mmd` | (M, S) state machine: init → update → detach → reset → NaN-recover | When reasoning about state correctness |
| `diagrams/ddp_no_sync.mmd` | 2-rank × K-accumulation with all-reduce suppression and G222 pitfall | When debugging DDP behavior |
| `diagrams/data_pipeline.mmd` | ParallelStreamLoader: shards → sub-streams → batches → doc_boundaries | When tracing data flow into training |
| `diagrams/newton_schulz.mmd` | NS5 with fp32 autocast-disable region and transpose-tall guard | When the inner loop is misbehaving |
| `diagrams/phase_dag.mmd` | Cross-phase task dependencies; what can be parallelized | When planning implementation order |

Throughout this file:
- **→ PLAN.md §X.Y** points to the detailed section.
- **⚠** marks silent-failure modes — read the linked detail before you write the code.
- **Gnnn** are gap identifiers — full context in `GAP_HISTORY.md`.

## How to use this document

1. Skim all phases once to understand scope. The project is ~7 components: an NMM, a block that combines NMM+attention via a learned gate, the full model with GPT-2 weight loading, a TBPTT training loop, a data pipeline that preserves stream continuity across batches, a generator that handles cross-chunk state, and an optional fast-inference scan path.
2. Implement in phase order. Within a phase, tasks with no listed dependency can be parallelized.
3. After each task, run the matching row(s) in `PLAN.md`'s **Testing Checkpoints** table. These regress every silent failure caught during audit.
4. **Do NOT use `titans-pytorch` as a dependency.** Implement from scratch; reference only.

---

## Phase 0 — Scaffolding

Establish the package layout and the canonical config that every later phase reads from.

### 0.1 Repo skeleton
Create the empty file tree from `ARCHITECTURE.md` (model/, data/, train.py, etc.) plus a `requirements.txt` with **lower-bound-pinned** deps:

```
torch>=2.3,<3      # >=2.8 also needed for Phase 6 scan
tiktoken>=0.5
transformers>=4.30
datasets>=2.14
numpy>=1.24
```

**⚠** Unpinned deps are a silent-failure source — a `transformers` minor bump can change the Conv1D weight layout that Phase 2.6 unpacks.

→ PLAN.md §0.1

### 0.2 `config.py` — `TitansConfig`
Standard `@dataclass` with: GPT-2 dims (n_layer/n_head/n_embd/block_size/vocab/dropout), NMM hyperparams (nmm_depth, nmm_expansion, nmm_conv_kernel, nmm_spectral_norm, nmm_n_persistent, chunk_size), attention flags (use_swa, swa_window), and `finetune_mode`. Factory classmethods `gpt2_small/medium/large/xl()` set the four pretrained backbone sizes.

**⚠ Use `raise ValueError`, NOT `assert`, in `__post_init__`** — `python -O` strips asserts. Validate at config time:
- `chunk_size > block_size` → reject (would OOB wpe)
- `use_swa=True and swa_window < 1` → reject (softmax NaN at step 0)
- `nmm_n_persistent < 0` → reject
- `nmm_expansion < 1` → reject
- `n_embd % n_head != 0` → reject (G223 — fail at config build, not model build)
- From-scratch + `chunk_size < block_size` → `warnings.warn` (untrained wpe rows beyond chunk_size silently degrade long-context generation)

**⚠ Factory methods must merge via `**{**defaults, **overrides}`** — passing dims as fixed kwargs alongside `**overrides` raises `TypeError: multiple values for keyword argument` whenever a caller overrides one.

→ PLAN.md §0.2

---

## Phase 1 — Neural Memory Module

The core novel component. Everything else is GPT-2 with this module spliced in.

### 1.1 Depthwise 1D conv
`CausalDepthwiseConv1d(dim, kernel_size=4)`: pad left-only by `k-1`; no padding inside `nn.Conv1d`; transpose in/out so the public API is `[B, T, dim]`.

**⚠** Strict causality requires *left-only* padding. Symmetric padding leaks future tokens.

→ PLAN.md §1.1

### 1.2 Q/K/V projection modules
`NMMProjection`: `Linear(d, d, bias=False) → CausalDepthwiseConv1d`. Three identical instances (`k_proj`, `q_proj`, `v_proj`). **No activation inside the module** — SiLU + L2 are applied at the call site:

```python
k_hat = F.normalize(F.silu(self.k_proj(x)), dim=-1)
q_hat = F.normalize(F.silu(self.q_proj(x)), dim=-1)
v     = F.silu(self.v_proj(x))   # NO L2 on v
```

**⚠** Submodule names matter for optimizer grouping in §4.1 — don't rename `linear`/`conv` to anything containing `norm`, `bias`, or `gamma`.

→ PLAN.md §1.2

### 1.3 Data-dependent update params
`W_θ`, `W_η`, `W_α` = three `Linear(d, 1, bias=False)`. Apply `sigmoid(...).squeeze(-1)` so outputs are `[B, T]` (not `[B, T, 1]`). Shape matters for broadcasting in §1.7 and for `grad()` to receive a scalar loss in §1.5.

→ PLAN.md §1.3

### 1.4 `MemoryMLP` (gated, L_M = 2)
SiLU-GLU two-layer MLP: `silu(W1·x) * sigmoid(W_gate·x)` → hidden → `W2 · h`. Forward returns `self.norm(W2(h)) + x` (ResidualNorm). Init `W1/W_gate/W2` Xavier-uniform.

**Critical:** `norm.weight/bias` are NOT recurrent state. Only the 2D weights `{W1, W_gate, W2}` are. (Newton-Schulz only works on 2D matrices; the norm is a fixed stabilizer.)

Add to `NeuralMemoryModule.__init__`:
- `out_scale` = `Parameter([d_model])`, init zeros when `finetune_mode=True`, ones when False. This is the only reliable way to get `y_mem=0` at finetune init given ResidualNorm passes `x` through.
- `_build_init_M(device)` helper returns the per-sample-batched initial M dict. Use `.to(device).clone()` order to avoid a wasted source-device copy (G207).

→ PLAN.md §1.4

### 1.5 Gradient via `torch.func`
Build once in `__init__`:
```python
def inner_loss(params, k_hat, v):
    pred = functional_call(self.memory_mlp, params, k_hat)
    # ⚠ reduction='sum' for spectral_norm=True; 'mean' for False (G160)
    return F.mse_loss(pred, v, reduction=reduction)
self.per_sample_grad_fn = vmap(grad(inner_loss), in_dims=(0, 0, 0))
```

**⚠** Construct `vmap(grad(...))` once in `__init__`, NOT per forward call.

**⚠** The reduction choice is config-driven. A user toggling `nmm_spectral_norm=False` while the reduction stays `'sum'` gets gradients d_model× too large → silent divergence.

→ PLAN.md §1.5

### 1.6 Newton-Schulz spectral normalization
5-step `NS5(G, steps=5, eps=1e-7)`: cast to fp32, transpose tall matrices (rows>cols) so they're wide, normalize by Frobenius, iterate `G = a·G + (b·A + c·A²)·G` where `A = G·Gᵀ`, transpose back, cast to orig dtype.

**⚠ Wrap the entire iteration in `torch.amp.autocast(device_type=..., enabled=False)`** (G226). A bare `.float()` is silently undone by ambient bf16 autocast — matmul inputs get cast back to bf16 even when explicitly fp32, and the iteration drifts.

**⚠ The transpose guard matters** — NS converges on wide matrices. W1/W_gate are `[4d, d]` (tall), W2 is `[d, 4d]` (wide). Without the transpose guard, NS on the tall gradients is suboptimal.

→ PLAN.md §1.6

### 1.7 Sequential memory step (single-token inference)
Per-token recurrence:
```
g̃_t = NS(∇_M ℓ(M_{t-1}, k̂_t, v_t))
S_t = η_t · S_{t-1} − θ_t · g̃_t       # momentum (θ applied POST-NS)
M_t = (1 − α_t) · M_{t-1} + S_t        # forget + integrate
y_t = out_scale ⊙ MemoryMLP(M_t, q̂_t) # write-then-read (retrieve from M_t)
```

**⚠ θ_t must scale POST-NS, not pre-NS.** NS divides by Frobenius norm — pre-scaling cancels exactly.

**⚠ Use `torch.where` for doc-boundary state resets, not in-place assignment.** In-place on autograd tensors raises `RuntimeError`.

→ PLAN.md §1.7

### 1.8 Chunked forward (training)
The training-time hot path. Two implementations (selected by `_HAS_ASSOC_SCAN` and grad mode):
- `_forward_chunk_sequential`: per-token Python loop, exact, autograd-friendly.
- `_forward_chunk_scan` (Phase 6): pre-computed gradients + `associative_scan`, ~10× faster but approximate.

For `_forward_chunk_sequential`:
- Run the conv on the full chunk in one shot (not per-token) — that's the whole point vs. T-many `step()` calls (G154).
- Precompute the per-t boundary mask on CPU; do NOT slice into a CUDA tensor inside the per-token loop (G202).
- Lazy-build `init_M` only when a doc boundary actually fires (G211).
- Optional `nmm_grad_checkpoint` flag: rematerialize each per-token update on backward (the GRADIENT+NS+MOMENTUM+RETRIEVAL block — not `step()`, which would recompute the conv on a 1-token slice).

→ PLAN.md §1.8

### 1.9 State management
Three methods on `NeuralMemoryModule`:
- `init_state(batch_size, device)` — returns `(M, S)` from `_build_init_M`; `S` zeros-like.
- `reset_state(state, mask)` — `torch.where(mask, init, state)` per tensor; handles `state=None`.
- `detach_states(states)` — detach every leaf in the nested structure; handle `None` (G149).

→ PLAN.md §1.9

---

## Phase 2 — Block and Full Model

Compose the NMM with attention into a TITANS MAG block, then stack into a full model that can load pretrained GPT-2 weights.

### 2.0 `CausalSelfAttention` + `GPT2MLP`
Standard GPT-2 attention/MLP. CausalSelfAttention takes an explicit `mask` arg (so the block can pass an augmented persistent+causal+SWA mask). `GPT2MLP` is `Linear → gelu(approximate='tanh') → Linear` matching HF parity.

**⚠** CausalSelfAttention must reject `n_head` not dividing `n_embd` via `raise ValueError`, NOT `assert` (G220 — same `-O` strip issue as §0.2).

→ PLAN.md §2.0

### 2.1 Persistent memory tokens + augmented mask
Each block owns `persistent_mem = Parameter([N_p, d_model])`. Concatenate to `x` along the time axis. Build a block-structured mask:
- persistent↔persistent: full
- persistent→real: -inf
- real→persistent: full
- real→real: causal

→ PLAN.md §2.1

### 2.2 Separate `ln_nmm`
Add `ln_nmm = LayerNorm(d_model)` to the block — distinct from `ln_1` (which feeds attention). NMM receives `ln_nmm(x)` of **real tokens only** (not the persistent-augmented `x̃`).

→ PLAN.md §2.2

### 2.3 MAG combination — fine-tuning compatible gate
Two formulas selected by `finetune_mode`:

```python
# finetune_mode=True (preserve pretrained residual at init):
o = y_attn + silu(gamma_mem * y_mem) * y_attn   # additive; out_scale=0 → o=y_attn

# finetune_mode=False (paper formula, from-scratch):
o = silu(gamma_attn * y_attn) * silu(gamma_mem * y_mem)
```

`gamma_attn` exists only when `finetune_mode=False`; `gamma_mem` is always created. Both init to ones.

→ PLAN.md §2.3

### 2.4 `TitansMAGBlock.forward`
Prepend persistent tokens, run attention with the augmented mask, drop the persistent prefix from the output, run NMM on `ln_nmm(x)`, combine via the gate from §2.3, MLP residual.

The NMM call is `nmm.forward_chunk(x_norm, state, doc_boundaries)` — 3 args. **⚠** Calling with only 2 silently drops boundary handling.

SWA implementation: if `use_swa=True`, add a banded-far-past mask. Persistent tokens remain fully visible regardless (paper Fig. 3b).

→ PLAN.md §2.4

### 2.5 `TitansMAGGPT2`
Wrap `wte + wpe → drop → N×TitansMAGBlock → ln_f`. Tied `lm_head = wte.weight.T`. Forward signature:
```python
def forward(idx, nmm_states=None, doc_boundaries=None) -> (logits, new_nmm_states)
```

`nmm_states=None` → initialize all blocks via `nmm.init_state(...)`.

**⚠** `_apply_gpt2_init` must skip NMM-internal modules by **id-set**, not by name (G203 — renaming `self.nmm` would silently break a name-based skip).

**⚠** `_apply_gpt2_init` uses relative import (`from .nmm import NeuralMemoryModule`), NOT `from model.nmm` (G224 — breaks under any top-level package name).

**⚠** Post-init: `wte.weight.std() ≈ 0.02`, `attn.proj.weight.std() ≈ 0.02/√(2·n_layer)` (residual scaling).

→ PLAN.md §2.5

### 2.6 GPT-2 weight loading
`load_pretrained(model, config)`: read HF `openai-community/gpt2` (or medium/large/xl, **derived from `config.n_embd`** — G216), transpose Conv1D weights (HF stores `[in, out]`, we store `[out, in]`), copy wte/wpe/ln/attn/mlp.

Leave NMM weights at their init. After loading, the model must produce logits identical to HF's GPT-2 (with NMM zeroed via `out_scale=0` and `N_p=0`) up to <1e-4 max diff.

→ PLAN.md §2.6

---

## Phase 3 — Data Pipeline

The non-obvious challenge: TBPTT requires that **position `i` of every batch in a row continues the same document stream** across calls. A naive `DataLoader(shuffle=False)` does NOT give you this.

### 3.1 Tokenizer
`tiktoken` GPT-2 encoding. Wrap with a small adapter that exposes `encode/decode/eot_token` and treats literal `<|endoftext|>` in the source text as BPE tokens (NOT as the EOT id — G152). Default `encode_corpus(file_or_iterable)` is whole-file by default (line-per-doc only on opt-in, with a warning if a raw file handle is passed — G210).

→ PLAN.md §3.1

### 3.2 Chunked document dataset (optional)
Only needed for from-scratch experiments with per-document boundary signals. For most fine-tuning runs, skip — feed continuous concatenated streams to §3.3.

→ PLAN.md §3.2

### 3.3 `ParallelStreamLoader` — TBPTT-aware batching
Maintains B independent sub-streams. Each `__iter__` yields `(idx_BT, doc_boundaries_BT)`. Sub-stream `b` continues from where it left off the previous call — i.e., `batch_k+1[b, 0]` follows `batch_k[b, -1]` in the same document.

**⚠ Shard streams across DDP ranks correctly** — each rank constructs the loader with its own `rank` and `world_size`. Don't have all ranks see the same data (G151).

→ PLAN.md §3.3

---

## Phase 4 — Training

### 4.1 Optimizer — four parameter groups
`gpt2_decay`, `gpt2_no_decay`, `nmm_decay`, `nmm_no_decay`. Routing:
- `'nmm' in name` → nmm groups; else gpt2 groups.
- `'bias'`, `'ln'`, `'norm'`, `'out_scale'`, `'gamma'`, `'persistent'` in name → no_decay.

Betas `(0.9, 0.95)`. NMM groups get 3× the LR (paper ratio).

**⚠** No param may appear in two groups. Add a runtime check.

**⚠** `out_scale`, `gamma_*`, and `persistent_mem` are intentionally no-decay — decay shrinks them toward zero, suppressing the memory branch and the prefix capacity.

→ PLAN.md §4.1

### 4.2 TBPTT `train_step`
Each call processes ONE chunk:
1. `.to(device)` on the batch (do NOT pre-move in the loader — G167).
2. Forward through `model(idx, nmm_states, doc_boundaries)`.
3. Loss = cross-entropy.
4. NaN guard: if loss is NaN/inf, zero grads and return `(loss, None, None)` — caller re-initializes `nmm_states` to `None` (G213). Verify a NaN injection does NOT corrupt params (G158).
5. Backward, clip, step. Run **backward + clip + step in fp32** even under bf16 autocast (G159).
6. Return `(loss, nmm_states_detached, grad_norm)`.

`detach_states` between chunks. The forward updates the NMM weights via differentiable ops; detach prevents the autograd graph from growing across chunks.

→ PLAN.md §4.2

### 4.3 LR schedule, logging, checkpointing
- `apply_lr(opt, step, base_lrs, max_steps, warmup_steps)` returns `lr_mul`. Warmup → cosine to `min_ratio` of peak.
- **⚠ Derive `base_lrs` from CODE-LEVEL CONSTANTS, NOT from `optimizer.param_groups[i]['lr']`.** Each save/resume cycle silently compounds LR deflation if you capture from `param_groups` after `load_state_dict` (G162).
- **⚠ Thread `max_steps` and `warmup_steps` explicitly into `apply_lr`.** Don't rely on defaults (G175).
- Checkpoint save fires on rank 0 only, followed by `dist.barrier()` (G199).
- `torch.load(..., weights_only=False)` — `weights_only=True` would reject our nested checkpoint (G168).
- `compute_nmm_norm(state)` returns `None` when state is `None`, else one float per layer (G172).
- Resume: build model → load state_dict → wrap DDP → build optimizer → load optimizer (in this order — G209). End with `model.train()` (G221). Tolerate missing `'optimizer'` key for HF-init checkpoints (G219).

→ PLAN.md §4.3

### 4.4 Fine-tuning entry point
Thin wrapper that constructs the model from `gpt2_small()`, calls `load_pretrained`, then runs the train loop. Sets `finetune_mode=True`.

→ PLAN.md §4.4

### 4.5 Training-from-scratch entry point
The consolidated training driver. Key structural rules:
- Build `config` BEFORE the loader references `config.chunk_size` (G205).
- Move model to device, then wrap with DDP, then build optimizer (G201).
- `init_process_group` called once before DDP wrap; `destroy_process_group` at the end (G201).
- Per-rank seed differs **after model construction** so dropout masks diverge (G204).
- Gradient accumulation under DDP uses `model.no_sync()` for all but the last micro-batch (G200).
- **⚠ Partial-cycle skip check:** `is_partial_cycle = (batch is None) and (accum_i > 0)` — NOT `accum_i < ACCUM_STEPS - 1` (G222 — silent off-by-one when StopIteration fires at iter K-1, causes DDP rank divergence).
- NaN-skip in the accumulation block must ALSO reset `nmm_states` to None (G217).
- **⚠ Wrap the entire training loop in `try: ... finally: dist.destroy_process_group()`** (G225) so cleanup fires on exceptions. Use consistent 4-space indentation throughout the try body (G227).

→ PLAN.md §4.5

---

## Phase 5 — Generation and Evaluation

### 5.1 Autoregressive generation
`generate(model, prompt, max_new_tokens, temperature, top_k, tokenizer=None, ...)`:
- If `len(prompt) > block_size`, chunk the prompt through the model so the NMM sees the full prefix (G176). Carry `nmm_states` across chunks.
- Order: temperature → top_k → softmax → multinomial (G173).
- Reuse caller-supplied tokenizer instance (don't construct a new one — G208).
- `try/finally` to restore `model.training` (G161).
- For contexts past `block_size`, slide the attention window but keep the NMM state continuous.

→ PLAN.md §5.1

### 5.2 Perplexity evaluation
Standard CE over a held-out corpus. NMM zeroed, eval mode + no_grad. Baseline must match HF GPT-2 within 5% (G156). `try/finally` for `model.training` restoration.

→ PLAN.md §5.2

### 5.3 Needle-in-haystack
Synthetic long-context recall test. Inject a unique key-value pair early in a long random context, ask the model to recall it at the end. Tests whether the NMM is actually storing across chunks.

→ PLAN.md §5.3

---

## Phase 6 — Associative Scan (optional inference speedup)

Pre-computes gradients at chunk-start `M_0` (an approximation), then uses `torch.associative_scan` for the per-token integration. ~10× faster inference; <5% relative error vs. sequential.

### 6.1 Pre-compute gradients + scan
Inside NMM:
- Compute all per-t gradients in one vmap call against `M_0` (not `M_{t-1}`).
- Apply NS per gradient.
- Define an associative op `(a₁, b₁) ⊕ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)` to integrate `M_t = (1-α_t)·M_{t-1} + S_t` as a prefix scan.

**⚠** The dispatcher gates on `torch.is_grad_enabled()`, NOT `self.training` (G164). Scan + autograd requires `torch.compile`; without compile, scan is inference-only.

→ PLAN.md §6.1

### 6.2 Integrate with `torch.associative_scan`
- Resolve `torch.associative_scan` through documented path first, fall back to private `torch._higher_order_ops.associative_scan` (PyTorch 2.6/2.7) — G215. Bind to module-level `_associative_scan`.
- `allow_scan_training(model, True)` sets the flag on every `block.nmm` (G180).
- For `torch.compile` checkpoints, save `_unwrap(model).state_dict()` to strip `_orig_mod.` prefixes (G184).

→ PLAN.md §6.2

---

## Critical Invariants (top gap-driven gotchas)

Not exhaustive — every section above flags its own. These are the highest-severity ones, all silent-failure modes that production runs hit even with passing unit tests.

| # | Invariant | Why it matters |
|---|---|---|
| 1 | `__post_init__` and `__init__` validation use `raise ValueError`, never `assert` | `python -O` strips asserts; invalid configs ship silently (G190, G220, G223) |
| 2 | Newton-Schulz wrapped in `autocast(enabled=False)` | Ambient bf16 autocast silently undoes `.float()` (G226) |
| 3 | NS applied per-gradient, BEFORE momentum; θ scales POST-NS | Pre-NS θ cancels in Frobenius division (paper §3.2) |
| 4 | `out_scale` init = zeros when finetune_mode=True | Only reliable way to get y_mem=0 at init (G123) |
| 5 | `base_lrs` from constants, not from `optimizer.param_groups` | Each resume compounds LR deflation otherwise (G162) |
| 6 | Backward + clip + step in fp32 under bf16 autocast | Mixed precision must not extend past the forward (G159) |
| 7 | Partial DDP accumulation: `(batch is None) and (accum_i > 0)` | Off-by-one causes silent rank divergence (G222) |
| 8 | DDP gradient accumulation uses `model.no_sync()` for non-final micro-batches | Otherwise allreduce fires per micro-batch (slow) or diverges (G200) |
| 9 | Training loop wrapped in `try/finally: destroy_process_group()` | NCCL communicator leak on exception path (G225) |
| 10 | `_apply_gpt2_init` skips NMM modules by id, with relative import | Robust to renames and package nesting (G203, G224) |
| 11 | `torch.where` for doc-boundary state resets | In-place assignment on autograd tensors raises RuntimeError |
| 12 | `_forward_chunk_sequential` runs the conv on the full chunk, not per-token | Per-token conv loses the lookback that makes the conv useful (G154) |
| 13 | NaN-skip in train_step resets `nmm_states` to None | Otherwise a corrupt state persists across the next call (G213, G217) |
| 14 | Per-rank seed differs after model construction | Dropout masks must diverge across ranks (G204) |
| 15 | Build config BEFORE loader references it | The naive ordering uses an undefined name (G205) |

For the full audit (all 227 gaps across 53 passes) see `GAP_HISTORY.md`.

---

## Recommended implementation order

1. **Phase 0** end-to-end. Run `python -c "import config; TitansConfig.gpt2_small()"`. Verify the rejection tests for invalid configs (Testing Checkpoints in PLAN.md).
2. **Phase 1.1 – 1.4** (conv, projections, update params, MemoryMLP). Test each module's shape and basic forward in isolation.
3. **Phase 1.5 – 1.6** (gradient via torch.func, NS). Test that the NS spectral norm bound holds for random tall/wide matrices, including under bf16 autocast (the G198/G226 tests).
4. **Phase 1.7 – 1.9** (sequential step, chunked forward, state mgmt). Overfit a single key→value pair. If loss doesn't go to ~0, the inner loop is wrong before you go further.
5. **Phase 2.0 – 2.2** (attention, MLP, persistent tokens, ln_nmm). Test the augmented mask block structure.
6. **Phase 2.3 – 2.5** (MAG gate, block forward, full model). Confirm `out_scale=0 → o = y_attn` exactly.
7. **Phase 2.6** (GPT-2 weight loading). The big checkpoint: with NMM zeroed and N_p=0, max logit diff vs. HF GPT-2 must be < 1e-4. If it isn't, the Conv1D transpose or wpe/wte tie is wrong.
8. **Phase 3** (data pipeline). Verify position-i continuity by printing sub-stream contents across two consecutive batches.
9. **Phase 4.1 – 4.3** (optimizer, train_step, schedule). Run a 100-step overfit on one batch. Loss must decrease monotonically.
10. **Phase 4.4** (fine-tune entry). One short run on real data. Watch for the NaN-skip path firing (it shouldn't, frequently).
11. **Phase 5** (generation + eval). Perplexity on held-out must match HF within 5% when NMM is zeroed.
12. **Phase 4.5** (from-scratch entry, DDP). Stand up multi-GPU with K=2 grad accum. Confirm the partial-cycle skip never fires unexpectedly.
13. **Phase 6** (optional scan). Compare scan vs. sequential output to <5% relative error before using for production inference.

---

## File structure

```
titans-mag-gpt2/
├── ARCHITECTURE.md
├── ROADMAP.md           (this file)
├── PLAN.md              (detailed plan; 4100+ lines)
├── GAP_HISTORY.md       (audit log)
├── config.py            (TitansConfig)
├── model/
│   ├── __init__.py
│   ├── nmm.py           (NeuralMemoryModule)
│   ├── block.py         (TitansMAGBlock)
│   └── titans_gpt2.py   (TitansMAGGPT2)
├── data/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── dataset.py
│   └── dataloader.py    (ParallelStreamLoader)
├── train.py             (consolidated training driver)
├── generate.py
├── eval.py
├── scripts/
│   ├── load_pretrained.py
│   └── finetune.py
└── requirements.txt
```

## References

- Sun et al. (2025). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663 — primary
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
- Di Nepi et al. (2025). *Titans Revisited.* arXiv:2510.09551
- lucidrains/titans-pytorch — reference only; do NOT depend on
