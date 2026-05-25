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
- **Harness API:** `eval.needle_in_haystack_sweep(model, tokenizer, device, haystack, ...)` runs the (position × secret) grid and returns `{"recall": float, "per_position": dict, "per_secret": dict, ...}` — the headline `recall ≥ 0.8` invariant maps to `result["recall"]`. Six structural / aggregation tests in `tests/integration/test_needle_smoke.py` lock in the harness contract; the recall behavior test waits on a trained checkpoint (G251).

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
| G117 | `test_optimizer::test_exactly_four_param_groups`, `::test_no_parameter_appears_in_two_groups`, `::test_every_trainable_param_is_in_some_group` | §6 |
| G123 | `test_memory_mlp::test_out_scale_init_zeros_in_finetune_mode`, `test_mag_gate::test_finetune_additive_gate_at_init_equals_y_attn_exactly`, `test_step::test_step_at_finetune_init_returns_zero_y` | §3, §4 |
| G134 | `test_checkpoint::test_save_load_roundtrip_preserves_state_dict`, `::test_save_load_optimizer_state_populated_after_step` | §6 |
| G136 | `test_persistent_mask::test_swa_banded_mask_attends_only_to_window` | §4 |
| G143 | `test_config::test_gpt2_small_factory_dims`, `::test_gpt2_medium_factory_dims`, `::test_gpt2_large_factory_dims`, `::test_gpt2_xl_factory_dims` | §2 |
| G147 | `test_memory_mlp::test_init_state_returns_M_and_zero_S`, `::test_init_state_shapes_match_build_init_M` | §3 |
| G149 | `test_state_mgmt::test_reset_state_unmasked_entries_byte_identical`, `::test_detach_states_passes_None_through` | §3 |
| G150 | `test_config::test_factory_accepts_dim_override`, `::test_factory_accepts_chunk_size_override`, `::test_factory_accepts_dropout_override` | §2 |
| G151 | `test_dataloader::test_position_i_streams_are_contiguous_across_batches`, `::test_ddp_rank_partition_gives_disjoint_segments` | §5 |
| G152 | `test_tokenizer::test_encode_corpus_literal_endoftext_is_BPE_not_special_id` | §5 |
| G153 | `test_optimizer::test_layernorm_params_routed_to_no_decay`, `::test_out_scale_routed_to_nmm_no_decay`, `::test_gamma_mem_routed_to_nmm_no_decay`, `::test_persistent_mem_routed_to_nmm_no_decay`, `test_checkpoint::test_save_load_roundtrip_preserves_state_dict` | §6 |
| G154 | `test_forward_chunk::test_forward_chunk_NOT_equal_to_T_many_step_calls` | §3 |
| G155 | `test_full_model::test_wte_init_std_is_002_not_default_1`, `::test_wpe_init_std_is_002`, `::test_attn_output_projection_has_residual_scaling`, `::test_mlp_c_proj_has_residual_scaling`, `::test_attn_qkv_projections_use_unscaled_std` | §4 |
| G156 | `test_hf_perplexity_parity` (`@pytest.mark.slow`), `test_generate::test_generate_keeps_eval_mode_if_caller_was_in_eval`, `test_perplexity_restores_training_mode` | §9, §6 |
| G157 | `test_lr_schedule::test_apply_lr_scales_all_four_groups_proportionally`, `::test_apply_lr_preserves_3x_nmm_ratio` | §6 |
| G158 | `test_train_step::test_nan_gradient_does_not_corrupt_parameters`, `::test_train_step_returns_three_values` | §6 |
| G159 | `test_bf16_autocast_gpu` (gpu-tier) | §6 |
| G160 | `test_grad_fn::test_reduction_switch_scales_gradient_by_inverse_d` | §3 |
| G161 | `test_generate::test_generate_restores_training_mode_when_called_in_train_mode`, `::test_generate_keeps_eval_mode_if_caller_was_in_eval`, `test_perplexity_restores_training_mode` | §6 |
| G162 | `test_lr_schedule::test_base_lrs_from_constants_does_not_read_optimizer_state` | §6 |
| G163 | `test_config::test_from_scratch_short_chunk_warns`, `::test_finetune_short_chunk_does_not_warn`, `::test_from_scratch_equal_chunk_does_not_warn` | §2 |
| G164 | `test_scan_dispatcher::test_dispatcher_uses_sequential_when_grad_enabled`, `::test_dispatcher_uses_scan_under_no_grad_when_no_boundaries` | §7 |
| G166 | `test_config::test_swa_zero_window_rejected`, `::test_swa_negative_window_rejected` | §2 |
| G167 | `test_train_step::test_train_step_handles_cpu_batch_via_to_device_transfer` | §6 |
| G168 | `test_checkpoint::test_load_works_under_weights_only_false` | §6 |
| G172 | `test_checkpoint::test_compute_nmm_norm_returns_None_when_states_is_None`, `::test_compute_nmm_norm_returns_one_float_per_layer` | §6 |
| G173 | `test_generate::test_temperature_zero_is_deterministic_argmax`, `::test_top_k_actually_filters_to_top_k_tokens` | §6 |
| G175 | `test_lr_schedule::test_apply_lr_respects_user_max_and_warmup_steps` | §6 |
| G176 | `test_generate::test_generate_chunks_long_prompts_through_NMM` | §6 |
| G180 | `test_scan_dispatcher::test_allow_scan_training_propagates_to_every_nmm`, `::test_top_level_model_attr_set_does_NOT_enable_scan` | §7 |
| G181 | (structural — this whole document) | — |
| G184 | `test_compile_save_load_gpu` (gpu-tier; covers `_unwrap` round-trip) | §6 |
| G189 | `test_resource_leak::test_finetune_opens_corpus_in_with_block`, `::test_train_main_opens_corpus_in_with_block` | §13 |
| G190 | `test_config::test_validation_fires_under_python_O` | §2 |
| G198 | `test_newton_schulz::test_internal_matmul_runs_fp32_under_bf16_autocast`, `::test_spectral_norm_bound_holds_under_bf16_autocast` | §3 |
| G199 | `test_ddp_save_barrier` (covers rank-0 save + dist.barrier afterwards) | §6, §11 |
| G200 | `test_ddp_gradient_accumulation` (ddp-tier, requires torchrun) | §11 |
| G201 | `test_train_main_structure::test_init_process_group_called_before_ddp_wrap`, `::test_model_to_device_called_before_ddp_wrap`, `::test_ddp_wrap_before_optimizer_construction`, `::test_init_destroy_process_group_pair_present` (AST-level structural) | §11 |
| G202 | `test_forward_chunk::test_boundary_mask_cpu_precomputed_not_per_token_indexed`, `::test_boundary_precomputation_does_not_fire_for_none_boundaries` | §3 |
| G203 | `test_full_model::test_nmm_internal_inits_preserved_after_apply_gpt2_init`, `::test_renaming_self_nmm_does_not_break_id_skip_pattern` | §4 |
| G204 | `test_train_main_structure::test_per_rank_seed_set_after_model_construction` (AST-level) | §11 |
| G205 | `test_train_loop::test_run_training_executes_full_loop` (verifies config-before-loader ordering implicitly via run completion) | §8 |
| G206 | `test_config::test_chunk_size_exceeds_block_size_rejected` | §2 |
| G207 | `test_memory_mlp::test_build_init_M_is_cloned_not_a_view` (covers the clone part; the `.to().clone()` vs `.clone().to()` ordering is structural — no per-call test) | §3 |
| G208 | `test_generate::test_generate_accepts_caller_supplied_tokenizer` | §6 |
| G209 | `test_checkpoint::test_resume_with_no_optimizer_key_does_not_raise`, `test_train_loop::test_resume_advances_params_and_loads_optimizer_state` | §6 |
| G210 | `test_tokenizer::test_encode_corpus_warns_on_file_handle`, `::test_encode_corpus_does_not_warn_on_list_of_documents`, `::test_encode_corpus_does_not_warn_on_generator_of_strings` | §5 |
| G211 | `test_forward_chunk::test_forward_chunk_no_boundaries_does_not_build_init_M`, `::test_forward_chunk_builds_init_M_only_once_per_chunk_with_boundaries` | §3 |
| G213 | `test_train_step::test_nan_skip_returns_None_nmm_states` | §6, §13 |
| G214 | `test_ddp_gradient_accumulation` (ddp-tier) | §11 |
| G215 | `test_scan_dispatcher::test_associative_scan_resolution_is_consistent` | §7 |
| G216 | `test_weight_loading::test_hf_model_name_from_n_embd_table`, `::test_load_pretrained_rejects_unsupported_n_embd`, `::test_load_pretrained_factory_overrides_route_to_correct_hf_name` | §4 |
| G217 | `test_train_step::test_run_training_accumulation_cycle_resets_nmm_states_on_nan`, `::test_run_training_continues_after_nan_skip_recovers` | §13 |
| G219 | `test_checkpoint::test_resume_with_no_optimizer_key_does_not_raise` | §6 |
| G220 | `test_attention::test_rejects_n_head_not_dividing_n_embd_via_ValueError`, `::test_attention_validation_survives_python_O` | §4 |
| G221 | `test_checkpoint::test_resume_ends_with_model_train_mode` | §6 |
| G222 | `test_train_loop::test_is_partial_cycle_at_K_minus_one`, `::test_is_partial_cycle_mid_cycle`, `::test_is_partial_cycle_at_cycle_start` (pure-function tests; ddp runtime in `test_ddp_gradient_accumulation`) | §11 |
| G223 | `test_config::test_n_embd_not_divisible_by_n_head_rejected`, `::test_n_embd_divisibility_error_quotes_values` | §2 |
| G224 | `test_full_model::test_apply_gpt2_init_uses_relative_import` | §4 |
| G225 | `test_train_main_structure::test_main_wraps_training_in_try_finally`, `::test_destroy_process_group_called_in_finally` | §11 |
| G226 | `test_newton_schulz::test_internal_matmul_runs_fp32_under_bf16_autocast` | §3 |
| G227 | `test_train_main_structure::test_try_body_uses_consistent_4_space_indentation` | §11 |
| G228 | (documentation drift — no defending test; see `tests/unit/test_config.py::test_factory_accepts_chunk_size_override` for override behavior) | §2 |
| G229 | (documentation drift — field intentionally absent from config per YAGNI; no test) | — |
| G230 | `test_newton_schulz::test_spectral_norm_bound_for_any_shape` (loosened bound `(0.80, 1.25)`) | §3 |
| G231 | `test_hf_logit_parity` shape sweep `[(1, 2), (2, 4)]` (CPU-affordable) | §9 |
| G232 | `test_scan_dispatcher::test_scan_implementation_matches_M0_approx_sequential_exactly`, `test_scan_vs_sequential_trained::test_scan_sequential_gap_increases_with_chunk_size` | §7, §10 |
| G233 | `test_train_loop::test_run_training_restarts_loader_on_exhaustion_to_reach_max_steps` | §8 |
| G234 | `test_hf_logit_parity_gpu::test_logit_parity_at_realistic_size` (scaled tolerance) | §9 |
| G235 | `test_needle_smoke::test_needle_in_haystack_uses_cached_decode_path` | §10 |
| G236 | `test_generate::test_generate_at_prompt_len_equals_block_size_returns_one_token`, `::test_generate_short_prompt_respects_new_cap` | §6 |
| G237 | `test_cached_generate_parity::test_cached_decode_long_prompt_uses_full_context` (rewritten to logit-level invariant) | §10 |
| G238 | `test_attention_kv_cache::test_cached_forward_swa_masks_far_past_real_positions`, `::test_cached_forward_swa_with_window_geq_real_positions_is_noop`, `::test_cached_forward_swa_persistent_prefix_always_visible`, `test_decode_parity::test_cached_decode_matches_full_forward_with_swa` | §4, §8 |
| G239 | (dead code removed — no defending test; existing parity tests use a local `_zero_conv_buffer` helper) | — |
| G240 | `test_decode_parity::test_prepare_decode_rejects_wrong_length_initial_nmm_states` | §8 |
| G241 | `test_decode_parity::test_cached_decode_matches_full_forward_multi_step_batched` | §8 |
| G242 | `test_needle_smoke::test_needle_in_haystack_long_prompt_uses_cached_decode_path` | §10 |
| G243 | `test_decode_parity::test_prepare_decode_rejects_train_mode`, `::test_forward_step_rejects_train_mode` | §8 |
| G244 | `test_decode_parity::test_prepare_decode_rejects_wrong_batch_dim_initial_nmm_states` | §8 |
| G245 / G246 / G247 / G248 | (documentation patches — no defending tests) | — |
| G249 | `test_decode_parity::test_prepare_decode_chunked_short_prompt_matches_prepare_decode`, `::test_prepare_decode_chunked_long_prompt_threads_nmm_state_across_prefix`, `::test_prepare_decode_chunked_rejects_train_mode` | §8 |
| G250 | (dead branch removed — existing `test_full_model::test_apply_gpt2_init_*` defend the init invariants) | §4 |
| G251 | `test_needle_smoke::test_sweep_default_grid_returns_expected_structure`, `::test_sweep_per_position_per_secret_aggregation_correct`, `::test_sweep_explicit_secrets_and_positions_override_defaults`, `::test_sweep_seed_determinism`, `::test_sweep_different_seed_produces_different_secrets`, `::test_sweep_one_secret_one_position_runs_one_pair` | §10 |
| G252 (T1) | `test_weight_loading::test_load_pretrained_transposes_c_attn_into_q_k_v_correctly`, `::test_load_pretrained_transposes_c_proj_correctly`, `::test_load_pretrained_copies_layernorms_without_transpose`, `::test_load_pretrained_copies_embeddings`, `::test_load_pretrained_does_not_touch_nmm_params`, `::test_load_pretrained_n_layer_mismatch_raises`, `::test_load_pretrained_n_head_mismatch_raises` | §4 |
| G252 (T2) | `test_train_loop::test_resume_matches_uninterrupted_training_in_param_space`, `::test_resume_advances_params_and_loads_optimizer_state`, `::test_resume_after_nan_skip_does_not_crash` | §8 |
| G252 (T3) | `test_full_model::test_doc_boundaries_all_true_isolates_positions_from_earlier_input_changes`, `::test_doc_boundaries_no_reset_path_DOES_propagate_position_zero_changes`, `::test_doc_boundaries_all_true_state_differs_from_no_reset_path` | §4 |
| G252 (T4) | `test_train_step::test_overfit_batch_drives_loss_near_zero` (slow) | §6 |
| G252 (T5) | `test_train_step::test_run_training_accumulation_cycle_resets_nmm_states_on_nan`, `::test_run_training_continues_after_nan_skip_recovers` | §6 |
| G252 (T6) | `test_full_model::test_multi_block_stack_forward_and_gradient_flow_to_every_param` | §4 |
| G252 (T7) | `test_train_main_structure::test_init_process_group_called_before_ddp_wrap`, `::test_model_to_device_called_before_ddp_wrap`, `::test_ddp_wrap_before_optimizer_construction`, `::test_per_rank_seed_set_after_model_construction`, `::test_init_destroy_process_group_pair_present` | §6 |
| G252 (T8) | `test_perf_smoke::test_train_step_throughput_is_non_degenerate`, `::test_train_step_does_not_leak_memory_across_steps`, `::test_generate_per_token_latency_does_not_grow_with_output_length`, `::test_generate_completes_within_reasonable_wall_time` | §12 |
| G252 (T9) | `test_config::test_fuzzed_invalid_config_raises_value_error_with_informative_message` (40 parametrized) | §13 |
| G252 (T10) | `test_tokenizer::test_encode_corpus_whole_file_pattern_produces_one_document` | §5 |
| G253 (T11) | `test_forward_chunk::test_boundary_mask_cpu_precomputed_not_per_token_indexed`, `::test_boundary_precomputation_does_not_fire_for_none_boundaries` | §3 |
| G253 (T12) | `test_block::test_nmm_receives_only_real_tokens_not_persistent_augmented` | §4 |
| G253 (T13) | `test_block::test_nmm_forward_chunk_called_with_doc_boundaries_arg`, `::test_nmm_forward_chunk_called_with_none_when_doc_boundaries_none` | §4 |
| G253 (T14) | `test_persistent_mask::test_aug_mask_no_row_is_fully_inf_standard_causal`, `::test_aug_mask_no_row_is_fully_inf_with_swa_at_edge_window`, `::test_aug_mask_softmax_produces_no_nan_in_attention_forward` | §4 |
| G253 (T15) | `test_attention::test_attention_output_deterministic_at_dropout_zero`, `::test_attention_output_deterministic_at_dropout_zero_with_causal_mask`, `::test_attention_output_deterministic_in_train_mode_at_dropout_zero` | §4 |
| G254 (retrieval_from_M_prev) | `test_paper_strict_flags::test_step_retrieval_from_M_prev_differs_from_default`, `::test_step_retrieval_from_M_prev_state_update_identical`, `::test_forward_chunk_retrieval_from_M_prev_first_token_uses_init_M`, `::test_forward_chunk_default_first_token_uses_M_1_not_init` | §3 |
| G254 (feed_persistent_to_nmm) | `test_paper_strict_flags::test_block_feed_persistent_flag_propagates_to_attribute`, `::test_block_feed_persistent_changes_output_shape_invariant`, `::test_block_feed_persistent_changes_nmm_output`, `::test_block_feed_persistent_handles_doc_boundaries` | §4 |
| G254 (nmm_n_heads) | `test_paper_strict_flags::test_block_n_heads_1_uses_single_head_NeuralMemoryModule`, `::test_block_n_heads_gt_1_uses_MultiHeadNMM`, `::test_multi_head_nmm_init_state_returns_list_of_per_head_states`, `::test_multi_head_block_forward_shape_invariant`, `::test_multi_head_block_is_differentiable`, `::test_multi_head_state_threads_across_calls`, `::test_multi_head_full_model_forward_shape_and_grad` | §4 |
| G254 (nested state plumbing) | `test_paper_strict_flags::test_detach_states_handles_multi_head_nested_structure`, `::test_detach_states_single_head_unchanged`, `::test_compute_nmm_norm_handles_multi_head_nested_structure`, `::test_compute_nmm_norm_returns_none_on_none_multi_head_safe` | §3, §6 |
| G254 (defaults + combined) | `test_paper_strict_flags::test_retrieval_from_M_prev_default_is_False`, `::test_feed_persistent_to_nmm_default_is_False`, `::test_nmm_n_heads_default_is_1`, `::test_nmm_n_heads_negative_rejected`, `::test_nmm_n_heads_not_dividing_n_embd_rejected`, `::test_all_three_flags_together_does_not_crash` | §2 |

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
