# TITANS MAG GPT-2 — Test Plan

A comprehensive test suite specification covering correctness, integration, parity,
behavior, performance, and regression of every silent-failure mode caught during the
53-pass / 227-gap audit.

## Document map
- `ARCHITECTURE.md` — what each test is verifying
- `ROADMAP.md` — order of implementation; this test plan mirrors that order
- `PLAN.md` — full code context; § references throughout this file
- `GAP_HISTORY.md` — G-numbers referenced under **Defends** point here

---

## Goals

1. **Correctness** — every module's contract holds in isolation.
2. **Integration** — components compose into a working block, model, training loop.
3. **Parity** — with NMM disabled, the model matches HF GPT-2 numerically.
4. **Behavior** — emergent properties (memorization, convergence) are present.
5. **Regression** — every audited gap has at least one defending test.
6. **Robustness** — known failure modes (NaN, OOM, rank divergence) are caught early.
7. **Performance** — silent throughput regressions are flagged at PR time.

The 227-gap regression suite is the highest priority — these are bugs that *passed
unit tests in earlier drafts*. Each test in §11 (regression matrix) defends at least
one logged G-number; reverting the corresponding fix must make the test fail loudly.

---

## 1. Test infrastructure

### 1.1 Repository layout
```
tests/
├── conftest.py                 # shared fixtures, markers, RNG seeding
├── unit/
│   ├── test_config.py
│   ├── test_conv.py
│   ├── test_projections.py
│   ├── test_update_params.py
│   ├── test_memory_mlp.py
│   ├── test_grad_fn.py
│   ├── test_newton_schulz.py
│   ├── test_step.py
│   ├── test_forward_chunk.py
│   ├── test_state_mgmt.py
│   ├── test_attention.py
│   ├── test_gpt2_mlp.py
│   ├── test_persistent_mask.py
│   ├── test_mag_gate.py
│   ├── test_block.py
│   ├── test_full_model.py
│   ├── test_weight_loading.py
│   ├── test_tokenizer.py
│   ├── test_dataloader.py
│   ├── test_optimizer.py
│   ├── test_train_step.py
│   ├── test_lr_schedule.py
│   ├── test_checkpoint.py
│   └── test_generate.py
├── integration/
│   ├── test_block_forward.py
│   ├── test_model_forward.py
│   ├── test_train_loop.py
│   ├── test_resume.py
│   └── test_scan_dispatcher.py
├── parity/
│   ├── test_hf_logit_parity.py
│   ├── test_hf_perplexity_parity.py
│   └── test_scan_vs_sequential.py
├── behavior/
│   ├── test_overfit_batch.py
│   ├── test_kv_memorization.py
│   ├── test_needle_in_haystack.py
│   └── test_long_context_loss.py
├── ddp/
│   ├── test_ddp_setup.py
│   ├── test_ddp_gradient_accumulation.py
│   ├── test_ddp_convergence.py
│   └── test_ddp_cleanup.py
├── performance/
│   ├── test_throughput.py
│   ├── test_memory_footprint.py
│   └── test_generation_latency.py
└── failure_modes/
    ├── test_nan_injection.py
    ├── test_invalid_configs.py
    └── test_resource_leak.py
```

### 1.2 Pytest markers

| Marker | Meaning | CI tier |
|---|---|---|
| (none) | CPU, single-process, <1s | every commit |
| `slow` | >5s but <60s, CPU-only | every commit (parallel) |
| `gpu` | requires CUDA | every commit (single GPU runner) |
| `slow_gpu` | >60s on GPU | nightly |
| `ddp` | requires ≥2 GPUs, `torchrun` | nightly |
| `compile` | requires `torch.compile` | nightly |
| `perf` | performance gate (deterministic threshold) | nightly |

Run filters:
- Pre-merge: `pytest -m "not slow_gpu and not ddp and not compile and not perf"` (~10 min)
- Nightly: `pytest` (everything; ~2 hours)
- Performance: `pytest -m perf` (with `--benchmark-compare` against baseline)

### 1.3 Shared fixtures (`conftest.py`)

```python
@pytest.fixture
def tiny_config():
    """Small config for fast unit tests."""
    return TitansConfig(
        n_layer=2, n_head=2, n_embd=32, vocab_size=256,
        block_size=64, chunk_size=32, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
    )

@pytest.fixture
def small_config():
    """gpt2-small dimensions for parity tests."""
    return TitansConfig.gpt2_small()

@pytest.fixture(autouse=True)
def deterministic_rng():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    yield

@pytest.fixture
def cpu_batch():
    """[B=2, T=16] integer batch on CPU — exercises G167 transfer path."""
    ...

@pytest.fixture
def fake_doc_boundaries():
    """[B=2, T=16] bool tensor with boundary at t=8 on row 0."""
    ...

@pytest.fixture(scope="session")
def hf_gpt2_model():
    """Cached HF GPT-2 small for parity comparisons."""
    return AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
```

Markers configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["slow", "gpu", "slow_gpu", "ddp", "compile", "perf"]
```

### 1.4 Determinism

Every test sets `torch.manual_seed(0)` via `autouse` fixture. Tests that involve sampling
(top-k, multinomial) further seed `torch.Generator` explicitly. Numerical-stability tests
use a fixed `eps = 1e-5` for comparisons except where higher precision is documented.

---

## 2. Unit tests — Phase 0 (Config)

### `test_config.py` — TitansConfig

- **Factory dims** — `TitansConfig.gpt2_small()` returns n_embd=768, n_head=12, n_layer=12. Same for medium/large/xl. **Defends:** G143, G150.
- **Factory accepts overrides** — `TitansConfig.gpt2_small(dropout=0.1).dropout == 0.1`. **Defends:** G150 (kwargs merge bug).
- **chunk_size > block_size rejected** — `pytest.raises(ValueError)` with `chunk_size=2048, block_size=1024`. Must NOT be `AssertionError`. **Defends:** G190, G206.
- **n_embd not divisible by n_head rejected at config time** — `TitansConfig.gpt2_small(n_head=10)` raises `ValueError` immediately (not on model construction). **Defends:** G223.
- **use_swa=True + swa_window=0 rejected** — raises `ValueError` mentioning the softmax-NaN failure mode. **Defends:** G166.
- **nmm_n_persistent < 0 rejected.**
- **nmm_expansion < 1 rejected.**
- **From-scratch + chunk_size < block_size warns** — `TitansConfig(finetune_mode=False, chunk_size=512, block_size=1024)` triggers `UserWarning` mentioning G163; fine-tune mode does NOT warn. **Defends:** G163.
- **Validation survives `python -O`** — `subprocess.run([sys.executable, '-O', '-c', 'from config import TitansConfig; TitansConfig(chunk_size=2048, block_size=1024)'])` exits non-zero (assert→ValueError migration). **Defends:** G190.
- **Repr is informative** — error messages quote the offending values, not just generic "invalid".

---

## 3. Unit tests — Phase 1 (NMM)

### `test_conv.py` — CausalDepthwiseConv1d

- **Shape preservation** — input `[B=2, T=16, d=32]` → output same shape, any T.
- **Strict causality** — for two inputs identical at positions ≤ t, outputs at positions ≤ t are bit-identical; perturbing position t+k does not change output at t.
- **Depthwise** — gradient of output channel c w.r.t. input channel c′≠c is zero.
- **Left-padding** — output at t=0 sees only x[0] (not future tokens).
- **Kernel size variation** — works at k=1, 4, 8.

### `test_projections.py` — NMMProjection

- **q̂, k̂ are L2-normalized** at the *call site*: `‖q_hat‖_2 ≈ 1` and `‖k_hat‖_2 ≈ 1` along the last dim, `‖v‖_2` unconstrained (varies by input).
- **No double-SiLU** — verify by running with a known input that the call-site `silu(self.k_proj(x))` produces a `silu`-once activation, not `silu(silu(x))`. **Defends:** documented in PLAN.md §1.2.
- **Submodule names** — `proj.linear.weight` and `proj.conv.conv.weight` exist in `named_parameters()`. Renaming would break optimizer routing. **Defends:** §4.1 routing.

### `test_update_params.py` — W_θ, W_η, W_α

- **Output range** — `sigmoid(W_θ(x)) ∈ (0, 1)` for arbitrary `x`.
- **Output shape `[B, T]`, NOT `[B, T, 1]`** — required for §1.5 grad scalar-loss and §1.7 broadcasting. **Defends:** broadcasting bug noted in §1.3.
- **All three independent** — perturbing W_α weights doesn't affect θ output (independent linears).

### `test_memory_mlp.py` — MemoryMLP

- **Output shape `[B, T, d]`** matches input.
- **ResidualNorm passes x through** — at `W2.weight = 0`, output ≈ `norm(0) + x` ≈ `x` (within LN affine init).
- **Xavier-uniform init** — `W1.weight.std()` matches `sqrt(2 / (fan_in + fan_out))` to 5%.
- **`norm` NOT in recurrent state** — `_build_init_M(...)` returns dict with keys `{W1.weight, W_gate.weight, W2.weight}` only; `norm.weight` and `norm.bias` are not present. **Defends:** architecture invariant — NS5 only operates on 2D tensors.

### `test_grad_fn.py` — torch.func gradient

- **Functional grad matches autograd** — compute `∇_M ℓ` via `vmap(grad(inner_loss))` and via `torch.autograd.functional.jacobian` on a tiny MemoryMLP; max abs diff < 1e-4.
- **Vmap correctness** — for batch_size=4, the b-th gradient equals the gradient of the b-th sample run in isolation.
- **Reduction switch** — with `spectral_norm=True` (reduction='sum') the gradient is `d_model` times larger than with `spectral_norm=False` (reduction='mean'). Verify the ratio for a fixed input. **Defends:** G160.
- **Grad fn constructed once** — `id(self.per_sample_grad_fn)` after first forward equals id after the tenth forward (no re-construction in hot path). **Defends:** §1.5 construction-cost rule.

### `test_newton_schulz.py` — NS5

- **Spectral norm bound** — for random `[m, n]` matrices (m=n=64, m=4n, n=4m), post-NS5 `‖G‖_2 ≈ 1` to within 1%.
- **Transpose guard** — tall (m>n) and wide (m<n) both produce post-NS5 spectral norm ≈1. Without the guard, the tall path is noticeably worse. **Defends:** convergence note in §1.6.
- **fp32 internal under bf16 autocast** — with ambient `torch.autocast(dtype=torch.bfloat16)`, NS5 still returns a result whose post-NS spectral norm is ≈1 (would drift toward ~0.7 under bf16 matmul). **Defends:** G198, G226.
- **Explicit autocast disable** — patch `torch.matmul` to record dtype, run NS5 under bf16 autocast, verify recorded dtype is `float32` (not `bfloat16`). **Defends:** G226 — `.float()` alone is silently undone by autocast policy.
- **Iteration count** — output at steps=5 vs steps=10 differs by <1e-3 (5 steps is converged).
- **Eps non-zero output** — passing all-zero G doesn't NaN; returns zero matrix.

### `test_step.py` — sequential single-token

- **Reference parity** — implement a hand-coded loop of the equations in `ARCHITECTURE.md` Equations 12-14 (`g̃_t = NS(∇ℓ)`, `S_t = η·S − θ·g̃`, `M_t = (1−α)·M + S`, `y_t = MLP(M_t, q̂_t)`); for a T=8 sequence, `step()` matches the reference to 1e-5.
- **θ applied POST-NS** — perturb θ by 10× and verify the change in `y_t` matches the post-NS scaling math, not pre-NS (which NS would cancel). **Defends:** ordering specified in `ARCHITECTURE.md` "Memory update rule".
- **Doc boundary reset (non-mutating)** — calling step with `boundary=True` returns state equal to `init_state(...)`; the input state tensor remains unchanged (no in-place write). **Defends:** torch.where rationale in §1.7.
- **out_scale=0 → y_t=0 exactly** — at finetune init. **Defends:** G123.

### `test_forward_chunk.py` — chunked training-mode forward

- **Output shape `[B, T, d]` + new state shape matches `_build_init_M`.**
- **Conv sees full chunk** — `_forward_chunk_sequential(x)` ≠ `[step(x[:,t:t+1])` stacked]: the per-token conv lookback differs. **Defends:** G154.
- **Boundary mask precomputed on CPU** — patch `torch.Tensor.__getitem__` to log accesses; verify no per-token GPU-resident indexing inside the for-loop. **Defends:** G202.
- **Lazy init_M build** — patch `_build_init_M` to count calls; with `doc_boundaries=None`, called 0 times; with one true boundary mid-chunk, called once. **Defends:** G211.
- **Gradient checkpointing parity** — with `nmm_grad_checkpoint=True`, output and gradients match the non-checkpointed path to 1e-5 (rematerialization is value-preserving).
- **Multi-step BPTT** — calling `_forward_chunk_sequential` for two consecutive chunks with `detach_states` between, then `.backward()` on the second chunk's loss, produces non-NaN gradients for `memory_mlp.W*.weight`.

### `test_state_mgmt.py` — init/reset/detach

- **`init_state(B, device)` shape** — returns `(M, S)` tuple; M has 3 entries (W1, W_gate, W2 weights), all with leading batch dim `B`; S is zeros-like.
- **`reset_state(state, mask)`** — entries where `mask=True` equal `init_state`; entries where `mask=False` are byte-identical to input. **Defends:** G149.
- **`detach_states(None) is None`** — handles first-step case without crashing. **Defends:** G149.
- **`detach_states` is recursive** — for nested dicts/tuples, every leaf tensor's `.requires_grad` is `False` post-detach.
- **`_build_init_M` device ordering** — patch `Tensor.to` to log; verify `.to(device).clone()` order (not `.clone().to(device)`). **Defends:** G207.

---

## 4. Unit tests — Phase 2 (Block & Full Model)

### `test_attention.py` — CausalSelfAttention

- **Shape `[B, T, d]` matches input.**
- **Mask honored** — supplying an `_aug_mask`-style mask zeroes attention weights at the masked positions.
- **n_head not dividing n_embd → ValueError** at construction, NOT `AssertionError`. Survives `python -O`. **Defends:** G220.
- **Output deterministic at dropout=0.**

### `test_gpt2_mlp.py` — GPT2MLP

- **HF parity** — output matches `F.gelu(c_fc(x), approximate='tanh')` elementwise to 1e-6 (matches HF's exact MLP).

### `test_persistent_mask.py` — augmented mask

- **Block structure** — given N_p=2, T=4, mask shape `[6, 6]` with:
  - persistent×persistent (top-left 2×2): all 0
  - persistent×real (top-right 2×4): all `-inf`
  - real×persistent (bottom-left 4×2): all 0
  - real×real (bottom-right 4×4): standard causal
- **Persistent row sum is normalizable** — no row is fully `-inf` (would cause softmax NaN).

### `test_mag_gate.py` — MAG combination

- **Additive form at out_scale=0** — `o = y_attn + silu(γ_m * y_mem) * y_attn = y_attn` when y_mem=0. Exact equality (not approximate). **Defends:** G123.
- **Pure paper form (from-scratch)** — `o = silu(γ_a·y_attn) ⊗ silu(γ_m·y_mem)` shape matches `y_attn`.
- **gamma_attn exists only when finetune_mode=False** — `'gamma_attn' in dict(block.named_parameters())` ⇔ `finetune_mode=False`.
- **gamma_mem always exists.**
- **Both γ init to ones.**

### `test_block.py` — TitansMAGBlock

- **Forward shape `[B, T, d]` matches input** (persistent prefix is dropped before residual).
- **NMM receives real tokens only** — patch `nmm.forward_chunk` to record `x.shape[1]`; verify T (real tokens), not T + N_p.
- **NMM called with 3 args** — `forward_chunk(x_norm, state, doc_boundaries)`, not 2. **Defends:** §2.4 doc-boundary handling.
- **SWA banded mask** — with `use_swa=True, swa_window=2`, attention at position t attends to t-2, t-1, t (not t-3). Persistent tokens remain fully visible. **Defends:** G136.
- **`ln_nmm` in state_dict** — distinct from `ln_1`. **Defends:** §2.2 separate norms.

### `test_full_model.py` — TitansMAGGPT2

- **Forward signature** — `forward(idx, nmm_states=None, doc_boundaries=None)` returns `(logits [B, T, V], new_nmm_states)`.
- **nmm_states=None initializes** — all blocks' returned states match `_build_init_M`.
- **Tied lm_head** — `model.lm_head.weight is model.wte.weight` (parameter-sharing tied).
- **`_apply_gpt2_init` post-init stds** — `wte.weight.std() ≈ 0.02` (NOT ≈1 — would mean uninitialized N(0,1)). **Defends:** G155.
- **Residual scaling** — `attn.proj.weight.std() ≈ 0.02 / sqrt(2 * n_layer)`. **Defends:** G155 residual scaling.
- **Init skips NMM by id, not name** — rename `self.nmm` → `self.foo` (via monkeypatch + reinit), confirm NMM params still skipped. **Defends:** G203.
- **Init uses relative import** — `from .nmm import NeuralMemoryModule` works when the package is installed under a non-`model` top-level name (e.g., `mypkg.nmm`). **Defends:** G224.

### `test_weight_loading.py` — load_pretrained

- **HF model name derived from n_embd** — `gpt2_small() → "openai-community/gpt2"`; `gpt2_medium() → "openai-community/gpt2-medium"`; etc. **Defends:** G216.
- **Conv1D transpose correct** — HF stores `c_attn.weight` as `[d, 3d]`; our `attn.c_attn.weight` is `[3d, d]`; verify transpose applied.
- **All gpt2 params loaded** — count of loaded params equals count of pretrained params (no silently-skipped tensors).

---

## 5. Unit tests — Phase 3 (Data)

### `test_tokenizer.py` — Tokenizer

- **Round-trip ASCII** — `decode(encode("hello world")) == "hello world"`.
- **EOT id == 50256** — matches HF GPT-2 vocab.
- **Literal `<|endoftext|>` in text is BPE-encoded** — `tok.encode("a <|endoftext|> b")` does NOT contain `tok.eot_token` as a single id; it splits into BPE tokens. **Defends:** G152.
- **`encode_corpus(open(path))` warns** — line-per-doc behavior on raw file handles emits `UserWarning`. **Defends:** G210.
- **Whole-file pattern works** — `encode_corpus(open(path).read())` treats as one document.

### `test_dataloader.py` — ParallelStreamLoader

- **Stream continuity** — for sub-stream b, the last token of `batch_k[b]` matches the position just before the first token of `batch_{k+1}[b]` in the source corpus. **Defends:** G151.
- **doc_boundaries shape** — yields `[B, T]` bool tensor.
- **Boundary firing** — when a sub-stream crosses a document boundary mid-batch, the corresponding position in `doc_boundaries` is `True`.
- **Rank sharding** — with `rank=0, world_size=2`, sub-stream 0 sees disjoint data from `rank=1, world_size=2`. No overlap.
- **Empty corpus handling** — raises a clear error, not a silent infinite loop.

---

## 6. Unit tests — Phase 4 (Training)

### `test_optimizer.py` — 4-group construction

- **Exactly 4 groups** — `len(optimizer.param_groups) == 4`. **Defends:** G117.
- **No param in two groups** — `sum(len(g['params']) for g in groups) == len(list(model.parameters()))`. **Defends:** G117.
- **Routing correctness** — `'nmm' in name` → nmm group; `'bias'/'ln'/'norm'/'out_scale'/'gamma'/'persistent' in name` → no_decay group. **Defends:** G153.
- **Betas (0.9, 0.95)** for all groups.
- **NMM LR is 3× GPT-2 LR.**
- **Decay groups have weight_decay > 0; no_decay groups have weight_decay == 0.**

### `test_train_step.py` — TBPTT step

- **CPU batch → GPU works** — pass CPU `input_ids`, model on `cuda`; step succeeds. **Defends:** G167.
- **Returns 3-tuple** — `(loss, nmm_states, grad_norm)`. **Defends:** G158.
- **Loss decreases on overfit** — 100 steps on a fixed batch, loss monotonically non-increasing (allow ±5% noise).
- **NaN-loss injection does NOT corrupt params** — patch loss to `loss + nan_tensor`; verify param hashes equal pre-step hashes; verify `optimizer.zero_grad` was called. **Defends:** G158.
- **NaN-loss returns nmm_states=None** — caller's next forward must re-init. **Defends:** G213.
- **bf16 autocast: backward + clip + step are fp32** — patch `torch.matmul` in clip/step to record dtype; verify fp32. **Defends:** G159.
- **`detach_states` actually breaks autograd** — call train_step twice, backward on second loss; verify first chunk's parameter gradients do NOT receive contributions from the second backward.

### `test_lr_schedule.py` — apply_lr

- **Warmup linear** — at step=0, lr_mul=0; at step=warmup_steps, lr_mul=1.
- **Cosine to min_ratio** — at step=max_steps, lr_mul=min_ratio.
- **base_lrs from constants survive deflation** — capture base_lrs from code constants, save+load optimizer state (which mutates `param_groups[i]['lr']` to a deflated value), call `apply_lr` again; verify post-call LRs match `code_constant * lr_mul`, not `deflated * lr_mul * lr_mul`. **Defends:** G162.
- **Scales all 4 groups** — `apply_lr` multiplies every group's LR. **Defends:** G157.
- **Preserves 1:1:3:3 ratio** post-call.
- **Respects user-supplied max_steps/warmup_steps** — call `apply_lr(opt, step=500, max_steps=1000, warmup_steps=100)` produces a different result than default `max_steps=100000`. **Defends:** G175.

### `test_checkpoint.py` — save/load

- **Round-trip** — save model + optimizer, construct fresh model + optimizer, load, parameters and optimizer state are byte-identical.
- **`torch.load(weights_only=False)` succeeds.** **Defends:** G168.
- **Resume from HF-init checkpoint (no 'optimizer' key)** — does NOT raise KeyError; optimizer state is left at its fresh init. **Defends:** G219.
- **Resume sequence ends with `model.train()`** — patch `model.train` to set a flag; verify flag after resume. **Defends:** G221.
- **Resume order: load state_dict → wrap DDP → build optimizer → load optimizer.** **Defends:** G209.
- **DDP-aware save fires on rank 0 only** — patch `torch.save` to count calls; with world_size=2, only rank 0 writes. **Defends:** G199.
- **`dist.barrier()` follows save.**
- **`compute_nmm_norm(None) is None`; non-None returns one float per layer.** **Defends:** G172.

### `test_generate.py` — autoregressive

- **Order: temperature → top_k → softmax → multinomial** — patch `F.softmax` to record `input.std()` before/after temp scaling; verify temp is applied first. **Defends:** G173.
- **Long prompt chunked, not truncated** — pass prompt of length 2048 to a model with block_size=1024; verify NMM state shows TWO chunk-forward updates (not one truncated to last 1024). **Defends:** G176.
- **Caller-supplied tokenizer reused** — pass a tokenizer with a custom attribute; verify `generate()` uses that instance (attribute survives). **Defends:** G208.
- **`model.training` restored on exit** — call `generate()` from a `model.train()` context; verify `model.training is True` after return. **Defends:** G161.
- **`model.training` restored on exception** — patch model.forward to raise; verify training mode still restored. **Defends:** G161 (try/finally).
- **Empty prompt handled** — `generate(model, "")` produces output starting from EOT.
- **EOT terminates generation early.**

---

## 7. Unit tests — Phase 6 (Scan)

### `test_scan_dispatcher.py` — dispatcher gating

- **Gates on `torch.is_grad_enabled()`, NOT `self.training`** — set model to train mode but wrap in `torch.no_grad()`: dispatcher selects scan path. **Defends:** G164.
- **Sequential under grad** — `with torch.enable_grad()`: dispatcher selects sequential. **Defends:** G164.
- **`allow_scan_training` flips every block** — `allow_scan_training(model, True)` sets `_allow_scan_training=True` on every `block.nmm`. **Defends:** G180.
- **`_HAS_ASSOC_SCAN` resolution** — patch `torch.associative_scan` and `torch._higher_order_ops.associative_scan` independently; verify the module finds the function in either location. **Defends:** G215.

---

## 8. Integration tests

Multi-component, single-process correctness.

### `test_block_forward.py`
- Full block forward pass on `[B=2, T=16, d=32]`: output shape correct, gradient flows to every parameter (`grad is not None` for each named parameter after `loss.backward()`).
- Multi-block stack: 3 blocks in a row, output shape and gradient flow.
- Persistent token gradient — `block.persistent_mem.grad` is non-zero after backward.

### `test_model_forward.py`
- `TitansMAGGPT2.forward(idx, None, None)` end-to-end on `[B=2, T=16]`: logits shape `[B, T, V]`, returned `nmm_states` is a list of length `n_layer`.
- `nmm_states` carries: pass `(M_1, S_1) = forward(idx)`; pass again with `forward(idx, nmm_states)`; second call's NMM internal state differs from first call's (memory is accumulating).
- doc_boundaries=all_true → state resets every position; equivalent to passing `nmm_states=None` each step.

### `test_train_loop.py`
- Run 10 steps with `train_step`, `apply_lr`, gradient clipping. No NaN. Loss decreases.
- `nmm_states` detached between steps; no autograd graph growth (measure `torch.autograd.graph.get_gradient_edge_count`-style via leaf tracking or by depth).
- Build config BEFORE loader references it — try constructing the loader before config and verify a clear `NameError`/`AttributeError` (not a silent default). **Defends:** G205.

### `test_resume.py`
- Train 5 steps → save → load in fresh process → train 5 more steps. Final params match an uninterrupted 10-step run to 1e-5.
- Resume from HF-init checkpoint (no optimizer key) does not crash; training proceeds. **Defends:** G219.
- Resume after NaN-skip step: ensure no corrupted state persists.

---

## 9. Parity tests

The strongest single signal of correctness.

### `test_hf_logit_parity.py`
- Load HF `openai-community/gpt2`. Build `TitansMAGGPT2.gpt2_small()`, call `load_pretrained`, set `out_scale=0` and `N_p=0` (disable NMM contribution; remove persistent prefix).
- Feed identical `input_ids = [50256, 1, 2, ..., 1023]` (full block).
- **Max logit diff < 1e-4** across all positions and vocab entries. Larger means weight loading is wrong (Conv1D transpose, wpe/wte tie, or LN epsilon).

### `test_hf_perplexity_parity.py`
- Compute perplexity on 1000 tokens of wikitext-2 with HF GPT-2 directly.
- Compute on TitansMAGGPT2 with NMM zeroed, `model.eval() + torch.no_grad()`.
- **Within 5%** of HF perplexity. **Defends:** G156.
- Without `model.eval()`: perplexity is silently higher (dropout). Test that `assert not model.training` is enforced in `perplexity()` setup.

### `test_scan_vs_sequential.py` (requires `slow_gpu` or `compile`)
- Build a small NMM, generate input `[B=2, T=64, d=32]` with random doc boundaries.
- Run `_forward_chunk_sequential` and `_forward_chunk_scan`.
- **Relative L2 error < 5%** (scan is an approximation: gradients pre-computed at `M_0`).
- Scan + autograd requires `torch.compile`; verify error message when calling scan without compile and grad enabled.

---

## 10. End-to-end behavior tests

Verify that the architecture's *intended* behavior emerges. Slow but essential.

### `test_overfit_batch.py` (`slow`)
- Tiny config, batch_size=2, T=64. Train 200 steps on a single fixed batch.
- **Final loss < 0.1** (overfitting works → the gradient flow is complete and correct).
- If this fails, the model isn't actually learning — investigate before further tests.

### `test_kv_memorization.py` (`slow_gpu`)
- Construct a synthetic sequence with key `[K1, K2, K3]` at position 100 and value `[V1, V2, V3]` at position 200.
- Train 500 steps; verify the model's NMM, when queried with the K-pattern later in the sequence, retrieves the V-pattern.
- This is the *associative memory* property the paper claims; without it the NMM is decorative.

### `test_needle_in_haystack.py` (`slow_gpu`)
- Inject `"The magic number is 42."` at random position in 2048 tokens of random Wikipedia.
- Generate completion of `"Question: what is the magic number? Answer:"`.
- **Top-1 token includes "42"** for ≥80% of positions. Tests cross-chunk memory.

### `test_long_context_loss.py` (`slow_gpu`)
- Compare per-position loss at positions [0, 256, 512, 768] of a 1024-length sequence.
- **Loss should be roughly stable or decrease across positions** (NMM provides more context further in). A sharp increase at boundary positions indicates state-reset bugs.

---

## 11. DDP / multi-process tests

Require `torchrun --nproc-per-node=2`. Mark `ddp`.

### `test_ddp_setup.py`
- `init_process_group` called once before DDP wrap, `destroy_process_group` at end. **Defends:** G201.
- Model `.to(device)` happens BEFORE `DDP(model)`. **Defends:** G201.
- Per-rank seed differs after model construction; dropout masks diverge across ranks. **Defends:** G204.

### `test_ddp_gradient_accumulation.py`
- With `K=4` accumulation steps:
  - Iterations 0–2 use `model.no_sync()` (verify via patched `dist.all_reduce` call count). **Defends:** G200.
  - Iteration 3 (last) does NOT use `no_sync`; allreduce fires.
  - **Result equivalence:** K-step accumulation produces gradient equal (to ±1%) to single-step with K× batch on each rank.
- **Partial-cycle check:** `is_partial_cycle = (batch is None) and (accum_i > 0)`:
  - Cycle of 4 batches: no partial — optimizer.step() fires after iter 3.
  - StopIteration at iter 0 (mid-cycle K=0): partial detected, optimizer.step() skipped, no rank divergence.
  - StopIteration at iter K-1 (the off-by-one boundary): partial detected. **Defends:** G222.

### `test_ddp_convergence.py` (`slow_gpu`, `ddp`)
- Train 100 steps on the same data with world_size=1 vs world_size=2 (gradient-equivalent batch). Final loss matches to 5%.

### `test_ddp_cleanup.py`
- Inject an exception inside the training loop; verify `dist.destroy_process_group()` was still called (try/finally fires). **Defends:** G225.
- After exception path, calling `init_process_group` again succeeds (no stale communicator). Tests under both clean exit and KeyboardInterrupt simulation.
- Indentation of the try block — static AST inspection: every statement in the try body has indent column 4, 8, or 12 (multiples of 4); no mixed step. **Defends:** G227.

---

## 12. Performance smoke tests (`perf`)

Catch silent throughput / memory regressions. Each test stores a baseline; PRs that regress more than the threshold trigger a CI signal (not block).

### `test_throughput.py`
- gpt2-small, batch=4, chunk=512, A100: tokens/sec ≥ baseline × 0.9.
- Sequential vs scan path: scan ≥ 5× sequential for inference.

### `test_memory_footprint.py`
- gpt2-small, B=4, chunk=512: peak GPU memory ≤ baseline + 10%.
- NMM state alone: ~54 MB/layer × 12 layers ≈ 650 MB (verify within 10%).
- Gradient checkpointing reduces activation memory ≥30%.

### `test_generation_latency.py`
- Generate 100 tokens from a 512-token prompt: latency ≤ baseline + 10%.
- Sliding-window generation does NOT grow unboundedly with output length.

---

## 13. Failure-mode tests

### `test_nan_injection.py`
- Inject NaN into loss → params unchanged (already in §6 — repeated here for completeness).
- Inject NaN into NMM state → train_step returns `nmm_states=None`; subsequent step rebuilds. **Defends:** G213.
- Inject NaN into accumulation block partial sum → reset path fires; no corruption propagates to next macro-batch. **Defends:** G217.

### `test_invalid_configs.py`
- All rejection cases from §2 — fuzzed with random invalid combinations to confirm error messages are informative.

### `test_resource_leak.py`
- Run 100 training-step start/destroy cycles in a single process. Active CUDA contexts and NCCL groups don't grow.
- Long-running test under `pytest-leak-detector`.

---

## 14. Regression coverage matrix

Every gap in `GAP_HISTORY.md` that introduced a silent-failure mode (or near-miss) has a defending test. Gaps marked **structural** are code-organization improvements that are not testable — they're listed for completeness.

| Gap | Defending test(s) | Section |
|---|---|---|
| G117 | `test_optimizer::test_exactly_4_groups`, `::test_no_param_in_two_groups` | §6 |
| G123 | `test_memory_mlp::test_out_scale_zero_init_finetune`, `test_mag_gate::test_additive_at_out_scale_zero`, `test_step::test_out_scale_zero` | §3, §4 |
| G134 | `test_checkpoint::test_round_trip` | §6 |
| G136 | `test_block::test_swa_banded_mask` | §4 |
| G143 | `test_config::test_factory_dims` | §2 |
| G147 | `test_state_mgmt::test_build_init_m_helper` | §3 |
| G149 | `test_state_mgmt::test_reset_state_byte_identical`, `::test_detach_states_none_safe` | §3 |
| G150 | `test_config::test_factory_accepts_overrides` | §2 |
| G151 | `test_dataloader::test_rank_sharding` | §5 |
| G152 | `test_tokenizer::test_literal_endoftext_bpe_encoded` | §5 |
| G153 | `test_optimizer::test_routing_correctness`, `test_checkpoint::test_round_trip` | §6 |
| G154 | `test_forward_chunk::test_conv_sees_full_chunk` | §3 |
| G155 | `test_full_model::test_apply_gpt2_init_stds` | §4 |
| G156 | `test_hf_perplexity_parity::test_within_5_percent`, `test_generate::test_eval_mode` | §9, §6 |
| G157 | `test_lr_schedule::test_scales_all_four_groups` | §6 |
| G158 | `test_train_step::test_nan_does_not_corrupt`, `::test_returns_3_tuple` | §6 |
| G159 | `test_train_step::test_bf16_backward_clip_step_fp32` | §6 |
| G160 | `test_grad_fn::test_reduction_switch` | §3 |
| G161 | `test_generate::test_model_training_restored`, `::test_restored_on_exception` | §6 |
| G162 | `test_lr_schedule::test_base_lrs_survive_deflation` | §6 |
| G163 | `test_config::test_from_scratch_warning` | §2 |
| G164 | `test_scan_dispatcher::test_gates_on_grad_enabled` | §7 |
| G166 | `test_config::test_swa_window_zero_rejected` | §2 |
| G167 | `test_train_step::test_cpu_batch_to_gpu` | §6 |
| G168 | `test_checkpoint::test_weights_only_false_succeeds` | §6 |
| G172 | `test_checkpoint::test_compute_nmm_norm_none` | §6 |
| G173 | `test_generate::test_temp_topk_softmax_order` | §6 |
| G175 | `test_lr_schedule::test_respects_user_max_steps` | §6 |
| G176 | `test_generate::test_long_prompt_chunked` | §6 |
| G180 | `test_scan_dispatcher::test_allow_scan_training_all_blocks` | §7 |
| G181 | (structural — this whole document) | — |
| G184 | `test_checkpoint::test_unwrap_compile_state_dict` | §6 |
| G189 | `test_train_loop::test_documents_resource_managed` | §8 |
| G190 | `test_config::test_validation_survives_O_flag` | §2 |
| G198 | `test_newton_schulz::test_fp32_under_bf16_autocast` | §3 |
| G199 | `test_checkpoint::test_ddp_save_rank_zero_only`, `test_ddp_setup::test_barrier_after_save` | §6, §11 |
| G200 | `test_ddp_gradient_accumulation::test_no_sync_for_non_final` | §11 |
| G201 | `test_ddp_setup::test_wrap_order`, `::test_init_destroy_pairing` | §11 |
| G202 | `test_forward_chunk::test_boundary_mask_cpu_precomputed` | §3 |
| G203 | `test_full_model::test_apply_gpt2_init_skips_nmm_by_id` | §4 |
| G204 | `test_ddp_setup::test_per_rank_seed_diverges` | §11 |
| G205 | `test_train_loop::test_config_before_loader` | §8 |
| G206 | `test_config::test_chunk_size_gt_block_size_rejected` | §2 |
| G207 | `test_state_mgmt::test_build_init_m_device_ordering` | §3 |
| G208 | `test_generate::test_caller_supplied_tokenizer_reused` | §6 |
| G209 | `test_checkpoint::test_resume_order` | §6 |
| G210 | `test_tokenizer::test_encode_corpus_file_handle_warns` | §5 |
| G211 | `test_forward_chunk::test_lazy_init_m` | §3 |
| G213 | `test_train_step::test_nan_returns_none_states`, `test_nan_injection::test_nmm_state_nan` | §6, §13 |
| G214 | `test_ddp_gradient_accumulation::test_partial_cycle_skip_step` | §11 |
| G215 | `test_scan_dispatcher::test_has_assoc_scan_path_resolution` | §7 |
| G216 | `test_weight_loading::test_hf_model_name_from_n_embd` | §4 |
| G217 | `test_nan_injection::test_accumulation_block_reset_nmm` | §13 |
| G219 | `test_checkpoint::test_resume_without_optimizer_key` | §6 |
| G220 | `test_attention::test_n_head_value_error_under_O` | §4 |
| G221 | `test_checkpoint::test_resume_ends_with_train_mode` | §6 |
| G222 | `test_ddp_gradient_accumulation::test_partial_cycle_off_by_one` | §11 |
| G223 | `test_config::test_n_embd_n_head_divisibility_at_config_time` | §2 |
| G224 | `test_full_model::test_apply_gpt2_init_relative_import` | §4 |
| G225 | `test_ddp_cleanup::test_destroy_on_exception` | §11 |
| G226 | `test_newton_schulz::test_explicit_autocast_disable` | §3 |
| G227 | `test_ddp_cleanup::test_try_block_indentation` | §11 |

Gaps in `GAP_HISTORY.md` not appearing here are structural (documentation reorganization, comment additions, file renames) and are not separately testable. See `GAP_HISTORY.md` for full per-gap context.

---

## 15. CI tier strategy

### Tier 1 — every commit (~10 min)
Run filter: `pytest -m "not slow_gpu and not ddp and not compile and not perf"`

Includes all of §2–§7 (unit) and the fast portions of §8 (integration). Excludes:
- `slow_gpu` end-to-end behavior tests
- DDP tests (require ≥2 GPUs)
- compile-dependent scan tests
- perf gates

Failure here BLOCKS the PR.

### Tier 2 — pre-merge gate (~30 min)
Adds the `slow` (CPU) and single-GPU `gpu` tests:
- `test_overfit_batch` (verifies the model actually trains)
- `test_hf_logit_parity` (catches weight loading bugs)

Failure here BLOCKS merge to main.

### Tier 3 — nightly (~2 hours)
Everything:
- `slow_gpu`, `ddp`, `compile`, `perf`
- `test_kv_memorization`, `test_needle_in_haystack`, `test_long_context_loss`
- `test_ddp_convergence`
- `test_scan_vs_sequential`
- All performance gates

Failure here notifies; does not block, but must be resolved before the next release.

### Tier 4 — release gate
Full Tier 3 plus:
- Run for ≥1000 training steps on real data; verify no NaN, loss trajectory matches a reference run within ±5%.
- Run `eval.py` on a held-out set; perplexity matches baseline within 1%.
- Generate 10 samples at temperature=0.8; manual review for degeneracy.

---

## 16. Known testing gaps

Items the audit has noted but for which no test exists yet — these are open work, not blockers:

- **Memory analysis under autograd** — quantify per-token NMM state retention; the audit notes ~58–230 GB at GPT-2-small chunk_size=512 may require gradient checkpointing as default rather than optional. A test that asserts a memory ceiling is straightforward but requires a real GPU and stable measurement.
- **`_unwrap` attribute-based detection** — could falsely unwrap user modules with a `.module` attribute. No isinstance-based variant test exists.
- **`model.no_sync()` AttributeError** — if `is_distributed=True` but model not actually DDP-wrapped, the call fails. No regression test covers this misconfig edge case.
- **First-chunk `boundary[:, 0] = True` redundancy** — wasted `_build_init_M` allocation; performance impact, not correctness.
- **`compute_nmm_norm` log-parser compatibility** — `None` return during NaN-reset cycle could break downstream log parsers. No test verifies the contract with a representative parser.
- **PyTorch 2.6/2.7 vs 2.8+ `_HAS_ASSOC_SCAN` signature compatibility** — partly covered by §7, but only at the import level. Function-signature drift across versions is untested.

These are tracked in PLAN.md context and should become tests if/when the corresponding component is touched.

---

## 17. References

- `PLAN.md` Testing Checkpoints table (subsumed and expanded by this document)
- `ROADMAP.md` Critical Invariants (top-15 distillation)
- `GAP_HISTORY.md` per-gap rationale for every entry in §14
