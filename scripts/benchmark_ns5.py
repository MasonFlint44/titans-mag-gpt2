"""Benchmark Newton-Schulz orthogonalization variants.

Compares three approaches to orthogonalizing a matrix gradient (the TITANS
NMM uses NS5 per token-block per layer, so NS5 is ~78% of CUDA time):

  1. Stock NS5         — fixed (a,b,c)=(3.4445,-4.7750,2.0315), n steps
  2. gram-NS5          — Tri Dao CUDA kernel: POLAR_EXPRESS per-step coefficients
                         + Gram-matrix iteration with reset at step 2.
                         NOTE: this is a DIFFERENT algorithm from NS5, not just
                         better coefficients.  The reset re-orthogonalises the
                         intermediate result, which is why it converges faster.
  3. CANS-stationary   — same standard NS5 structure but minimax-optimal (a,b,c)
                         derived via differential_evolution on the actual sv range
  4. CANS-nonstationary — per-step (aₖ,bₖ,cₖ) via LP on the propagated sv range

CANS (2506.10935) claims fewer iterations than NS5 by using Chebyshev-optimal
coefficients.  We derive coefficients numerically.  Important caveat: the Gram
iteration in gram-NS5 adds an orthogonality-reset benefit that pure coefficient
optimisation on the standard NS5 structure cannot replicate.

Singular-value range is estimated from the test matrices (10th-percentile floor
to exclude near-zero SVs that no polynomial can orthogonalise in finite steps).

Usage:
    uv run python -m scripts.benchmark_ns5 [--out profiles/ns5_benchmark.txt]
    uv run python -m scripts.benchmark_ns5 --cpu   # no GPU required
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Singular-value range estimation
# ─────────────────────────────────────────────────────────────────────────────

def _sv_range(
    shapes: list[tuple], device: torch.device, n_samples: int = 8, seed: int = 42,
    pct_lo: float = 10.0,
) -> tuple[float, float]:
    """[p10, max] singular values after F-norm normalisation, across shapes.

    The p10 floor excludes near-zero SVs that can never converge in finite steps
    (f(0)=0 for any polynomial, so optimising over σ≈0 is meaningless).
    """
    torch.manual_seed(seed)
    all_svs: list[float] = []
    for rows, cols, _ in shapes:
        for _ in range(n_samples):
            G = torch.randn(rows, cols, device=device)
            X = G / (G.norm() + 1e-7)
            sv = torch.linalg.svdvals(X.float()).cpu().numpy()
            all_svs.extend(sv.tolist())
    arr = np.array(all_svs)
    return float(np.percentile(arr, pct_lo)), float(np.max(arr))


# ─────────────────────────────────────────────────────────────────────────────
# CANS stationary: DE on [sv_min, sv_max]
# ─────────────────────────────────────────────────────────────────────────────

_cans_stat_cache: dict[tuple, tuple[float, float, float]] = {}


def cans_stationary(
    sv_min: float, sv_max: float, n_steps: int = 5,
    n_grid: int = 400, seed: int = 42,
) -> tuple[float, float, float]:
    """Global optimisation of (a,b,c) minimising max|f^n(σ)-1| over [sv_min,sv_max]."""
    key = (round(sv_min, 4), round(sv_max, 4), n_steps)
    if key in _cans_stat_cache:
        return _cans_stat_cache[key]

    from scipy.optimize import differential_evolution

    sigmas = np.linspace(sv_min, sv_max, n_grid)

    def objective(params):
        a, b, c = params
        x = sigmas.copy()
        for _ in range(n_steps):
            x = a * x + b * x ** 3 + c * x ** 5
            x = np.clip(x, 0.0, 5.0)
        return float(np.max(np.abs(x - 1.0)))

    result = differential_evolution(
        objective,
        bounds=[(1.0, 20.0), (-80.0, 0.0), (0.0, 70.0)],
        seed=seed, maxiter=3000, tol=1e-12,
        polish=True, init="latinhypercube",
    )
    coeffs = tuple(float(v) for v in result.x)
    _cans_stat_cache[key] = coeffs
    print(
        f"  CANS-stationary(steps={n_steps}): a={coeffs[0]:.4f}, "
        f"b={coeffs[1]:.4f}, c={coeffs[2]:.4f}  (obj={result.fun:.4e})",
        file=sys.stderr,
    )
    return coeffs


# ─────────────────────────────────────────────────────────────────────────────
# CANS non-stationary: per-step LP
# ─────────────────────────────────────────────────────────────────────────────

_cans_nstat_cache: dict[tuple, list[tuple[float, float, float]]] = {}
_NS5_FALLBACK = (3.4445, -4.7750, 2.0315)
# Coefficient bounds for the LP: large enough to include known good solutions
# (POLAR_EXPRESS step-1 uses c≈13.6; NS5 uses c≈2.03) but capped to prevent
# absurdly large values that cause bf16 precision issues.
_LP_BOUNDS_A = (0.0, 100.0)
_LP_BOUNDS_B = (-5000.0, 0.0)
_LP_BOUNDS_C = (0.0, 5000.0)


def _lp_step(lo: float, hi: float, max_out: float = 1.5, n_grid: int = 400,
) -> tuple[float, float, float]:
    from scipy.optimize import linprog
    # If the range is already very narrow the LP has many near-equivalent
    # solutions; the unbounded solver picks degenerate corners (e.g. c=1,
    # a=b=0 → x^5) that diverge outside [lo,hi].  Use bounded variables to
    # steer toward the stable branch.
    x = np.linspace(lo, hi, n_grid)
    phi = np.column_stack([x, x ** 3, x ** 5])
    n = len(x)
    ones = np.ones(n)
    A = np.vstack([
        np.column_stack([ phi, -ones]),           # f - E ≤ 1
        np.column_stack([-phi, -ones]),           # -f - E ≤ -1
        np.column_stack([ phi, np.zeros(n)]),     # f ≤ max_out (stability)
        np.column_stack([-phi, np.zeros(n)]),     # f ≥ 0
    ])
    b_vec = np.concatenate([ones, -ones, max_out * ones, np.zeros(n)])
    bounds = [_LP_BOUNDS_A, _LP_BOUNDS_B, _LP_BOUNDS_C, (None, None)]
    res = linprog([0, 0, 0, 1], A_ub=A, b_ub=b_vec,
                  bounds=bounds, method="highs")
    if not res.success:
        return _NS5_FALLBACK
    return (float(res.x[0]), float(res.x[1]), float(res.x[2]))


def cans_nonstationary(
    sv_min: float, sv_max: float, n_steps: int = 7,
) -> list[tuple[float, float, float]]:
    key = (round(sv_min, 4), round(sv_max, 4), n_steps)
    if key in _cans_nstat_cache:
        return _cans_nstat_cache[key]
    coefs: list[tuple[float, float, float]] = []
    lo, hi = sv_min, sv_max
    for _ in range(n_steps):
        a, b, c = _lp_step(lo, hi)
        coefs.append((a, b, c))
        x = np.linspace(lo, hi, 400)
        out = a * x + b * x ** 3 + c * x ** 5
        lo = max(0.001, float(np.min(out)))
        hi = min(2.5, float(np.max(out)))
    _cans_nstat_cache[key] = coefs
    return coefs


# ─────────────────────────────────────────────────────────────────────────────
# NS variants
# ─────────────────────────────────────────────────────────────────────────────

_NS5_COEFS = (3.4445, -4.7750, 2.0315)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = _NS5_COEFS
    X = G.bfloat16()
    t = X.size(-2) > X.size(-1)
    if t: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
    return X.mT if t else X


def _ns_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    A = X @ X.mT; B = b * A + c * (A @ A); return a * X + B @ X


def cans_stationary_fn(G: torch.Tensor, a: float, b: float, c: float,
                        steps: int = 5, eps: float = 1e-7,
                        dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    X = G.to(dtype)
    t = X.size(-2) > X.size(-1)
    if t: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps): X = _ns_step(X, a, b, c)
    return X.mT if t else X


def cans_nonstationary_fn(G: torch.Tensor,
                           coefs: list[tuple[float, float, float]],
                           eps: float = 1e-7,
                           dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    X = G.to(dtype)
    t = X.size(-2) > X.size(-1)
    if t: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in coefs: X = _ns_step(X, a, b, c)
    return X.mT if t else X


def gram_ns5_fn(G: torch.Tensor) -> torch.Tensor:
    from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
    return GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
    )(G)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def orth_error(X: torch.Tensor) -> float:
    """||X^T X - I||_F (uses smaller gram matrix for tall/wide inputs)."""
    Xf = X.float()
    gram = Xf.mT @ Xf if Xf.size(-2) >= Xf.size(-1) else Xf @ Xf.mT
    I = torch.eye(gram.size(-1), device=X.device, dtype=torch.float32)
    return (gram - I).norm().item()


def time_ms(fn, warmup: int = 10, repeats: int = 200, cuda: bool = True) -> float:
    for _ in range(warmup): fn()
    if cuda: torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

SHAPES = [
    (768, 3072, "768×3072 (NMM W1)"),
    (3072, 768, "3072×768 (NMM W2)"),
]
BENCH_STEPS = [3, 4, 5]
MAX_STEPS = 7


def _h(t: str) -> str:
    return f"\n{'─'*80}\n{t}\n{'─'*80}"


def run_benchmark(device: torch.device, out: Path) -> None:
    cuda = device.type == "cuda"
    lines: list[str] = []

    # ── 0. SV range ──────────────────────────────────────────────────────────
    print("[benchmark] sampling singular value range (p10 floor)…", file=sys.stderr)
    sv_min, sv_max = _sv_range(SHAPES, device)
    msg = (f"Singular value range (p10→max) after F-norm normalisation: "
           f"[{sv_min:.5f}, {sv_max:.5f}]")
    print(f"  {msg}", file=sys.stderr)
    lines.append(msg)

    # ── 1. Coefficients ───────────────────────────────────────────────────────
    # Derive CANS-stationary coefficients *separately* for each target step
    # count.  CANS-k means: (a,b,c) that minimise max|f^k(σ)-1| over the sv
    # range, applied for exactly k steps.  Using 5-step coefficients for a
    # 3-step run is NOT CANS-3 — it's just stopping NS5 early.
    lines.append(_h("Coefficients"))
    lines.append(f"  NS5 (standard):  a={_NS5_COEFS[0]:.4f}, b={_NS5_COEFS[1]:.4f}, c={_NS5_COEFS[2]:.4f}")

    cans_coefs: dict[int, tuple[float, float, float]] = {}
    for k in BENCH_STEPS:
        print(f"\n[benchmark] deriving CANS-stationary coefficients (DE, steps={k})…",
              file=sys.stderr)
        a, b, c = cans_stationary(sv_min, sv_max, n_steps=k)
        cans_coefs[k] = (a, b, c)
        lines.append(
            f"\n  CANS-stationary (DE, {k}-step, sv=[{sv_min:.4f},{sv_max:.4f}]):\n"
            f"    a={a:.4f}, b={b:.4f}, c={c:.4f}"
        )

    print(f"\n[benchmark] deriving CANS-nonstationary coefficients (LP per step)…",
          file=sys.stderr)
    nstat_coefs = cans_nonstationary(sv_min, sv_max, n_steps=MAX_STEPS)
    lines.append(f"\n  CANS-nonstationary (LP per step):")
    for i, (a, b, c) in enumerate(nstat_coefs):
        lines.append(f"    step {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}")
        print(f"  step {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}", file=sys.stderr)

    try:
        from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS
        gram_available = True
        lines.append("\n  gram-NS5 POLAR_EXPRESS (per-step) + Gram iteration + reset@step2:")
        for i, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
            lines.append(f"    step {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    except ImportError:
        gram_available = False
        lines.append("  gram-NS5: not installed")

    # ── 2. Convergence ────────────────────────────────────────────────────────
    # Each CANS-k row shows the error when the *correct* k-step coefficients
    # are used for exactly k steps.  This is the fair comparison: CANS-k at k
    # steps vs NS5-5 at 5 steps.
    # For CANS-nonstationary we show both bf16 and fp32 to diagnose numerical
    # precision effects from the large step-1 LP coefficients.
    lines.append(_h("Convergence: ||X^T X - I||_F  — CANS-k uses coefficients derived for k steps"))
    lines.append("(nstat bf16/fp32 differ only in compute dtype; coefficients are identical)")

    for rows, cols, label in SHAPES:
        lines.append(f"\n  Shape: {label}")
        hdr = (f"  {'k':>4}  {'NS5':>10}  {'CANS-stat':>10}"
               f"  {'nstat bf16':>10}  {'nstat fp32':>10}")
        lines.append(hdr)
        lines.append("  " + "─" * (len(hdr) - 2))
        torch.manual_seed(42)
        G = torch.randn(rows, cols, device=device)
        for k in range(1, MAX_STEPS + 1):
            e_ns  = orth_error(newton_schulz5(G, steps=k))
            ak, bk, ck = cans_coefs.get(k, cans_stationary(sv_min, sv_max, n_steps=k))
            e_cs  = orth_error(cans_stationary_fn(G, ak, bk, ck, steps=k))
            e_cn16 = orth_error(cans_nonstationary_fn(G, nstat_coefs[:k], dtype=torch.bfloat16))
            e_cn32 = orth_error(cans_nonstationary_fn(G, nstat_coefs[:k], dtype=torch.float32))
            marker = " ← NS5-5 target" if k == 5 else ""
            lines.append(
                f"  {k:>4}  {e_ns:>10.4f}  {e_cs:>10.4f}"
                f"  {e_cn16:>10.4f}  {e_cn32:>10.4f}{marker}"
            )

    # ── 3. Speed vs quality (primary comparison) ──────────────────────────────
    lines.append(_h("Speed vs quality: CANS-k (k-step coefficients, k steps) vs NS5-5"))
    lines.append("Primary question: can CANS-k match NS5-5 quality at fewer steps?\n")
    lines.append(f"{'Method':<32} {'Steps':>5} {'Shape':<24} {'ms':>8} {'error':>10} {'vs NS5-5 err':>14} {'speedup vs NS5-5':>17}")
    lines.append("─" * 102)

    for rows, cols, label in SHAPES:
        torch.manual_seed(42)
        G = torch.randn(rows, cols, device=device)
        ns5_ref_err = orth_error(newton_schulz5(G, steps=5))
        ns5_ref_ms  = time_ms(lambda: newton_schulz5(G, steps=5), cuda=cuda)

        # NS5 baseline rows
        for k in BENCH_STEPS:
            fn = lambda s=k: newton_schulz5(G, steps=s)
            ms = time_ms(fn, cuda=cuda)
            err = orth_error(fn())
            marker = " ← baseline" if k == 5 else ""
            lines.append(
                f"{'NS5':<32} {k:>5} {label:<24} {ms:>8.3f} {err:>10.4f}"
                f" {'—':>14} {ns5_ref_ms/ms:>16.2f}×{marker}"
            )

        # CANS-stationary: each k uses coefficients derived for exactly k steps
        for k in BENCH_STEPS:
            a, b, c = cans_coefs[k]
            fn = lambda s=k, _a=a, _b=b, _c=c: cans_stationary_fn(G, _a, _b, _c, steps=s)
            ms = time_ms(fn, cuda=cuda)
            err = orth_error(fn())
            ratio = err / ns5_ref_err
            speedup = ns5_ref_ms / ms
            tag = " ✓ better" if err < ns5_ref_err else f" {ratio:.1f}× worse"
            lines.append(
                f"{'CANS-stat-' + str(k):<32} {k:>5} {label:<24} {ms:>8.3f} {err:>10.4f}"
                f" {tag:>14} {speedup:>16.2f}×"
            )

        # CANS-nonstationary bf16 and fp32
        for tag_label, dt in [("CANS-nstat bf16", torch.bfloat16),
                               ("CANS-nstat fp32", torch.float32)]:
            for k in BENCH_STEPS:
                fn = lambda s=k, _dt=dt: cans_nonstationary_fn(G, nstat_coefs[:s], dtype=_dt)
                ms = time_ms(fn, cuda=cuda)
                err = orth_error(fn())
                ratio = err / ns5_ref_err
                speedup = ns5_ref_ms / ms
                tag = " ✓ better" if err < ns5_ref_err else f" {ratio:.1f}× worse"
                lines.append(
                    f"{tag_label + '-' + str(k):<32} {k:>5} {label:<24} {ms:>8.3f} {err:>10.4f}"
                    f" {tag:>14} {speedup:>16.2f}×"
                )

        if gram_available:
            fn = lambda: gram_ns5_fn(G)
            ms = time_ms(fn, cuda=cuda)
            err = orth_error(fn())
            ratio = err / ns5_ref_err
            speedup = ns5_ref_ms / ms
            tag = " ✓ better" if err < ns5_ref_err else f" {ratio:.1f}× worse"
            lines.append(
                f"{'gram-NS5 (POLAR+Gram+reset)':<32} {'5*':>5} {label:<24} {ms:>8.3f} {err:>10.4f}"
                f" {tag:>14} {speedup:>16.2f}×"
            )
        lines.append("")

    lines.append("* gram-NS5 is a different algorithm (Gram iteration + reset at step 2).")

    # ── Output ───────────────────────────────────────────────────────────────
    text = "\n".join(lines)
    print("\n" + text, file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[benchmark] wrote {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("profiles/ns5_benchmark.txt"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Pass --cpu to run on CPU.")
    device = torch.device("cpu" if args.cpu else "cuda")
    print(f"[benchmark] device={device}", file=sys.stderr)
    run_benchmark(device, args.out)


if __name__ == "__main__":
    main()
