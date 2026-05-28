# Quack-kernels NS5 investigation (2026-05)

We evaluated whether [Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)'s
custom kernels (the `quack-kernels` package) could speed up the inner-loop NS5
that dominates training time (~78% of CUDA time per the README profile).
**Conclusion: no win at our shapes / hardware. Threw the branch away.**

## What we tried

1. **Per-op autograd wrappers.** Quack's `gemm`, `gemm_symmetric`, and
   `gemm_add` are forward-only — the upstream use case is Muon, which
   orthogonalises gradients in the optimizer step (no backward through
   NS5). Our TITANS NMM differentiates *through* NS5 (the inner-loop
   gradient flows back to the outer loss), so we wrapped each quack op
   in a `torch.autograd.Function` with hand-derived matmul backward
   (gradient of `A @ B` is just two more matmuls).

2. **Backend-agnostic `gram_newton_schulz`.** Added a `backend` argument
   to the existing function so the same iteration could run on either
   `torch.baddbmm` or the wrapped quack ops via a four-op interface
   (`sym_mm`, `sym_baddbmm`, `mm`, `mm_add`).

3. **Param-key batching at the call site.** The NMM dispatches NS5 once
   per parameter key per step (3 for full-rank: `W1`, `W_gate`, `W2`;
   6 for low-rank). After the transpose-to-wide step, full-rank keys
   all reduce to `[B, 768, 3072]` — same shape, batchable along dim 0.
   `batched_ns5_over_dict` groups same-wide-shape keys and stacks them
   into a single call.

## Results

Hardware: RTX 5070 Ti (consumer Blackwell, SM 12.0), CUDA 13.0.
Benchmark: median ms per NS5 dispatch (3 keys), 100 reps after 10
warmups. Each cell is `ms (speedup-vs-torch-per-key)`. Configurations:

- `torch + per-key` — current default (`torch.baddbmm` × 3 keys).
- `quack + per-key` — quack ops × 3 keys.
- `torch + batched` — `torch.baddbmm` × 1 batched call.
- `quack + batched` — quack ops × 1 batched call (the proposed default).

### Forward-only

```
config                  B=1            B=2            B=4            B=8           B=16
torch + per-key         3.870 (1.00x)  8.502 (1.00x)  14.232 (1.00x) 26.996 (1.00x) 49.811 (1.00x)
quack + per-key         7.992 (0.48x)  8.419 (1.01x)  18.080 (0.79x) 28.642 (0.94x) 52.407 (0.95x)
torch + batched         3.916 (0.99x)  7.924 (1.07x)  14.044 (1.01x) 31.542 (0.86x) 61.169 (0.81x)
quack + batched         3.853 (1.00x)  8.041 (1.06x)  17.183 (0.83x) 32.174 (0.84x) 61.661 (0.81x)
```

### Forward + Backward (matches training-time cost)

```
config                  B=1            B=2            B=4            B=8           B=16
torch + per-key        12.389 (1.00x) 31.358 (1.00x) 52.170 (1.00x) 95.318 (1.00x) 186.650 (1.00x)
quack + per-key        23.070 (0.54x) 32.374 (0.97x) 62.829 (0.83x) 118.801 (0.80x) 233.321 (0.80x)
torch + batched        12.263 (1.01x) 27.194 (1.15x) 51.148 (1.02x) 98.737 (0.97x) 204.262 (0.91x)
quack + batched        16.896 (0.73x) 31.961 (0.98x) 61.840 (0.84x) 128.526 (0.74x) 252.115 (0.74x)
```

### Per-op head-to-head (the answer in one table)

At our exact iteration shapes, `quack` and `torch.baddbmm` are within
5% of each other — sometimes quack is slightly slower:

```
B=1:   X @ X.mT    torch 2.28ms   quack_sym 2.40ms   0.95x
       baddbmm     torch 2.25ms   quack_sym 2.27ms   0.99x
       Q @ X       torch 2.27ms   quack_gemm 2.28ms  1.00x
B=16:  X @ X.mT    torch 2.88ms   quack_sym 3.13ms   0.92x
       baddbmm     torch 2.45ms   quack_sym 2.45ms   1.00x
       Q @ X       torch 2.90ms   quack_gemm 2.84ms  1.02x
```

## Why it didn't work

1. **cuBLAS is already excellent at our shapes.** `768 × 3072` matmuls
   on consumer Blackwell are not where quack's hardware-aware kernels
   show their advantage. Their "up to 2× faster" claim targets H100 /
   B200 at larger problem sizes.

2. **The symmetric-kernel FLOP halving doesn't materialise as
   wall-time.** Theoretical 2× savings on the symmetric matmul aren't
   visible at our matrix dimensions on this GPU — we're not
   FLOP-bound, the kernel is memory-bandwidth-bound at these sizes.

3. **Batching doesn't help when the baseline isn't launch-bound.**
   Each `torch.baddbmm` at our shapes is ~2.3 ms of *actual compute*,
   not launch overhead. Stacking three calls into one doesn't recover
   meaningful time. At higher batches the batched path is actually
   slower (likely cache / memory-pressure effects on the 3× larger
   stacked tensor).

4. **Autograd.Function overhead is non-trivial.** The wrappers we needed
   to make quack's forward-only ops differentiable add Python dispatch
   cost per call. With ~9 matmul ops per NS5 iteration, that's ~100–
   450µs of dispatch on top of an already-fast 3.9ms baseline at B=1.

## What might change the picture (not pursued)

- **Larger backbones.** At gpt2_medium (d=1024) or gpt2_large
  (d=1280), the Gram matrix grows from 768² to 1024² / 1280², the
  FLOP advantage of the symmetric kernel is more significant, and
  cuBLAS's tuning may not cover the regime as tightly. We didn't
  test these because we train at gpt2_small.
- **H100 / B200 hardware.** Quack's documented gains are on server
  Hopper / Blackwell. Consumer Blackwell (SM 12.0) is supported but
  not the primary target.
- **Single-Function whole-iteration wrapper + CUDA graph.** Would
  eliminate per-op autograd dispatch and might let CUDA-graph capture
  amortise launches. Would only matter if launch overhead were the
  bottleneck — at 2.3 ms/op it isn't.

## Decision

Throw the branch away. Keep the current pure-PyTorch
`gram_newton_schulz` (out-of-place F-norm, no optional dep, no
autograd wrappers). The comment block at the top of that function in
`model/nmm.py` references this doc.

## Reproducing

The branch was deleted. The benchmark output for the four-config sweep
above lives only in this doc. To redo the experiment from scratch:

1. `uv pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130`
   (or `[cu12]` on CUDA 12.x).
2. Add an `autograd.Function` wrapper around each of `gemm`,
   `gemm_symmetric`, `gemm_add` with matmul-gradient backward.
3. Thread a `backend` argument through `gram_newton_schulz` that
   selects between `torch.baddbmm` and the wrapped quack ops.
4. Write a benchmark that sweeps batch size and times forward-only
   plus forward+backward, both per-key and batched.

The per-op head-to-head table above is the load-bearing finding —
if quack ever beats torch.baddbmm on your hardware at your shapes,
revisit; otherwise the integration work won't net out.
