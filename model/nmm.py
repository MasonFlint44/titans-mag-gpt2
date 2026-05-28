"""Neural Memory Module: SiLU-GLU gated MLP with online surprise-driven weight updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from . import nmm_fused as _fused


# Map config strings -> torch dtypes for the recurrent (M, S) state and
# per-step update buffers. fp16 is intentionally excluded — it would need
# GradScaler wiring and the NMM's surprise gradient can overshoot fp16 range
# in early training. bf16 has fp32-equivalent range and "Just Works" with
# our bf16-autocast forward pipeline.
_STATE_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

# G275 — int8 state quantization.
#
# Halves the bytes-per-element of the recurrent (M, S) state vs bf16
# (and quarters vs fp32). Within the blockwise path the math runs in
# fp32 — the int8 form is only the storage representation between
# block boundaries.
#
# Storage representation: each state-key tensor is split into TWO
# tensors in the same dict:
#   key:           int8 tensor with the quantized values, same shape as
#                   the original fp32/bf16 tensor.
#   key + "_qs":   fp16 scalar per sample. The dequantized value is
#                   `int8_tensor.float() * scale.view(B, 1, 1)`.
#
# Per-sample-per-tensor scaling: each (sample b, state-key) gets its
# own scalar scale, sized symmetrically as `max_abs / 127`. This gives
# ~127 levels of resolution per row, which matches typical KV-cache
# int8 quantization in transformer literature.
#
# Constraints:
#   - Blockwise path only. Sequential, scan v2, step(), step_with_conv()
#     raise NotImplementedError when fed int8 state — those paths
#     update state per-token and would dequantize/requantize on every
#     step, which is both slow and amplifies noise. Config validation
#     enforces `nmm_block_size > 1` when state_dtype is int8.
#   - Quantization noise enters once per block. Over many blocks the
#     noise accumulates; loss-curve drift vs bf16 grows with T/block_size.
#     For best behavior, pair int8 with the standard production
#     blockwise config (`block_size=64`) and validate loss curves on
#     your data before committing to a long run.


def _quantize_int8(t: torch.Tensor):
    """Per-sample per-tensor int8 quantization.

    Args:
        t: [B, ...] fp32/bf16 tensor.

    Returns:
        (int8_tensor [B, ...], scale [B] fp32) where the dequantized
        value is `int8_tensor.float() * scale.view(B, 1, ..., 1)`.

    Uses symmetric quantization centered at zero: scale = max_abs / 127.
    Tiny-eps clamp on the divisor so all-zero tensors don't NaN out.
    """
    B = t.shape[0]
    t_f = t.float().detach()
    max_abs = t_f.view(B, -1).abs().amax(dim=-1)        # [B]
    scale = (max_abs / 127.0).clamp(min=1e-8)            # [B]
    s_view = scale.view(B, *([1] * (t.ndim - 1)))
    q = (t_f / s_view).round().clamp(min=-128, max=127).to(torch.int8)
    return q, scale


def _dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of `_quantize_int8`. Always returns fp32."""
    B = q.shape[0]
    s_view = scale.view(B, *([1] * (q.ndim - 1)))
    return q.float() * s_view


def _is_int8_dict(d: dict) -> bool:
    """Detect whether a state dict uses the int8 representation.

    True iff any key has a corresponding `key + '_qs'` companion (a fp32
    scale tensor). False for normal fp32/bf16 dicts (no `_qs` companions).
    """
    return any(k.endswith("_qs") for k in d.keys())


def _dequant_dict(d: dict, state_keys: tuple) -> dict:
    """Convert an int8-representation dict (with `_qs` scale entries) to
    a plain fp32 dict over `state_keys`. No-op if `d` is already plain.
    """
    if not _is_int8_dict(d):
        return d
    return {k: _dequantize_int8(d[k], d[k + "_qs"]) for k in state_keys}


def _quant_dict(d: dict, state_keys: tuple) -> dict:
    """Convert a plain fp32/bf16 dict to int8 representation with `_qs`
    scale companions. No-op if `d` already has the int8 representation.
    """
    if _is_int8_dict(d):
        return d
    out = {}
    for k in state_keys:
        q, s = _quantize_int8(d[k])
        out[k] = q
        out[k + "_qs"] = s
    return out


def _where_dict(mask: torch.Tensor, new_dict: dict, old_dict: dict) -> dict:
    """Per-batch torch.where over matching dict tensors.

    `mask` [B] bool: True → take from new_dict; False → keep old_dict.
    Returns a fresh dict; tensors are autograd-friendly (no in-place writes).
    """
    out = {}
    for k, new_v in new_dict.items():
        old_v = old_dict[k]
        m = mask.view(mask.shape[0], *([1] * (old_v.ndim - 1)))
        out[k] = torch.where(m, new_v, old_v)
    return out


def _reset_M_S(M: dict, S, mask: torch.Tensor, init_M: dict):
    """Reset masked batch entries of (M, S). Does NOT touch conv_buf —
    used by the inner per-token loop where conv_buf is irrelevant (the
    conv ran once at chunk start; the inner loop iterates over already-
    convolved k_hat / q_hat / v values).

    Returns (M_new, S_new) — note 2-tuple, not state."""
    M_new = _where_dict(mask, init_M, M)
    if isinstance(S, (list, tuple)):
        S_new = tuple(
            _where_dict(mask, {k: torch.zeros_like(v) for k, v in S_lvl.items()}, S_lvl)
            for S_lvl in S
        )
    else:
        S_new = _where_dict(mask, {k: torch.zeros_like(v) for k, v in S.items()}, S)
    return M_new, S_new


def reset_state(state: tuple, mask: torch.Tensor, init_M: dict) -> tuple:
    """Reset masked batch entries of a full per-layer state.

    Handles both shapes:
      - 2-tuple `(M, S)` — returns `(M_new, S_new)`. Used by callers that
        only track M/S (e.g. the per-token inner loop).
      - 3-tuple `(M, S, conv_buf)` — returns `(M_new, S_new, conv_buf_new)`.
        Used by the chunk-level reset.

    `mask`: [B] bool. True → init values for that batch row; False → keep.

    Inner-loop callers that operate on (M, S) directly should call
    `_reset_M_S` to avoid an unused conv_buf round-trip.
    """
    if len(state) == 2:
        M, S = state
        return _reset_M_S(M, S, mask, init_M)
    M, S, conv_buf = state
    M_new, S_new = _reset_M_S(M, S, mask, init_M)
    cb_new = _where_dict(
        mask, {k: torch.zeros_like(v) for k, v in conv_buf.items()}, conv_buf,
    )
    return (M_new, S_new, cb_new)


def _detach_per_layer(layer_state):
    """Detach a single per-layer NMM state. Handles all shapes:
      - None: pass through (plain non-NMM block, G261).
      - single-head, momentum_order=1: `(M, S_dict, conv_buf)` triple.
      - single-head, momentum_order>1: `(M, S_tuple, conv_buf)` where S_tuple
        is a tuple/list of N dicts (G272).
      - multi-head: `[(M_h, S_h, conv_buf_h), ...]` (G254).
      - int8 state (G275): dicts may carry `_qs` scale companion entries
        alongside value entries. `dict.items()` returns both;
        `.detach()` is dtype-agnostic so detaching both is safe.

    `conv_buf` is the last-k-1 Q/K/V linear projections carried across
    chunk boundaries (item 6). Always detached at chunk boundary so the
    backward graph does not extend across chunks.
    """
    if layer_state is None:
        return None
    if isinstance(layer_state, list):
        return [_detach_per_layer(s) for s in layer_state]
    M, S, conv_buf = layer_state
    M_det = {k: v.detach() for k, v in M.items()}
    if isinstance(S, (list, tuple)):
        S_det = tuple({k: v.detach() for k, v in S_lvl.items()} for S_lvl in S)
    else:
        S_det = {k: v.detach() for k, v in S.items()}
    cb_det = {k: v.detach() for k, v in conv_buf.items()}
    return (M_det, S_det, cb_det)


def detach_states(states):
    """Detach every leaf tensor in a per-layer list of NMM states.

    Pass-through on None — at the very first training step nmm_states is
    None and the model's forward initializes it; this helper must not
    explode on that case (G149).

    Now recursive (G254): per-layer state can be either a `(M, S)` tuple
    (single-head NMM) or a list-of-tuples (multi-head NMM via `MultiHeadNMM`).
    The recursion in `_detach_per_layer` handles both transparently.
    """
    if states is None:
        return None
    return [_detach_per_layer(s) for s in states]



def _apply_lookahead_v(v_chunk: torch.Tensor, theta_chunk: torch.Tensor) -> tuple:
    """G269 lookahead-value transform: shift v left by 1, zero θ at last pos.

    Inner loss becomes `||M(k_t) - v_{t+1}||²` for t < T-1; for t = T-1
    there's no v_T+1 within this chunk, so its surprise gradient is zeroed
    (the M update at that step is just (1-α)·M_{t-1} decay). This matches
    lucidrains' `store_with_lookahead_value` flag.

    Both `v_chunk` and `theta_chunk` are passed in and out so the caller
    can use the shifted values directly. Autograd-safe (no in-place writes).

        v_chunk:     [B, T, D]
        theta_chunk: [B, T]      (default)  or  [B, T, K]  (per_param_lr)
    """
    v_shift = torch.cat(
        [v_chunk[:, 1:], torch.zeros_like(v_chunk[:, :1])], dim=1
    )
    zero_tail = torch.zeros_like(theta_chunk[:, -1:])
    theta_shift = torch.cat([theta_chunk[:, :-1], zero_tail], dim=1)
    return v_shift, theta_shift


def _scale(scalar_B: torch.Tensor, tensor_dict: dict) -> dict:
    """Broadcast a per-sample scalar [B] across each [B, ...] tensor in the dict."""
    out = {}
    for k, g in tensor_dict.items():
        s = scalar_B.view(scalar_B.shape[0], *([1] * (g.ndim - 1)))
        out[k] = s * g
    return out


def _scale_per_key(scalars, tensor_dict: dict) -> dict:
    """Scale a tensor dict by either:
      - a shared scalar tensor (same shape as `_scale`'s arg) — applied
        uniformly to every key, OR
      - a per-key dict of scalars `{key: [B, ...]}` — applied independently
        per key.

    G270: enables per-parameter LR modulation where each state key (W1,
    W_gate, W2 for full-rank; six factors for low-rank) gets its own
    data-dependent learning rate θ[key].
    """
    if isinstance(scalars, dict):
        out = {}
        for k, g in tensor_dict.items():
            s = scalars[k]
            s_view = s.view(s.shape[0], *([1] * (g.ndim - 1)))
            out[k] = s_view * g
        return out
    return _scale(scalars, tensor_dict)


def _step_momentum(S_prev, g_tilde: dict, theta_t, eta_t, order: int):
    """One step of the (possibly higher-order) momentum recurrence.

    Order 1 (paper default):
        S_t = η_1 · S_{t-1} - θ · g_t                          (single dict)

    Order N > 1 (G272):
        S_1_t = η_1 · S_1_{t-1} - θ · g_t
        S_k_t = η_k · S_k_{t-1} + S_{k-1}_t        for k=2..N
        S_t = tuple(S_1, S_2, ..., S_N)                         (tuple-of-dicts)

    The M update reads the topmost level S_N (the smoothest), so the
    caller does `M += S_top` where S_top = `S_t` (order 1) or
    `S_t[-1]` (order > 1).

    Args:
        S_prev: dict (N=1) or tuple/list of N dicts (N>1).
        g_tilde: dict {state_key: per-sample gradient tensor}.
        theta_t: scalar tensor [B] or per-key dict (per_param_lr).
        eta_t:   scalar tensor [B] (N=1) or [B, N] (N>1).
        order:   N.

    Returns: same structure as `S_prev`.
    """
    surprise = _scale_per_key(theta_t, g_tilde)
    if order == 1:
        return _dict_sub(_scale(eta_t, S_prev), surprise)

    # eta_t is [B, N] (chunk path will pre-slice when needed). Per-level
    # scalars: eta_t[..., k] is [B].
    eta_levels = [eta_t[..., k] for k in range(order)]
    S_new = []
    S_1_new = _dict_sub(_scale(eta_levels[0], S_prev[0]), surprise)
    S_new.append(S_1_new)
    for k in range(1, order):
        S_k_new = _dict_add(_scale(eta_levels[k], S_prev[k]), S_new[k - 1])
        S_new.append(S_k_new)
    return tuple(S_new)


def _S_top(S, order: int):
    """Return the topmost momentum level, matching state structure."""
    return S if order == 1 else S[-1]


def _project_theta(
    W_theta: nn.Module,
    x: torch.Tensor,
    state_keys: tuple,
    per_param_lr: bool,
):
    """Project x through W_theta -> per-token θ_t.

    Returns one of:
      - shape [B] or [B, T] tensor when per_param_lr=False (default), OR
      - dict {key: [B] or [B, T]} when per_param_lr=True (one independent
        θ per state-key).

    W_theta's output_dim is 1 in the default case and len(state_keys) in
    the per_param case. The split is along the last dim.
    """
    z = torch.sigmoid(W_theta(x))  # [..., n_theta]
    if per_param_lr:
        return {k: z[..., i] for i, k in enumerate(state_keys)}
    return z.squeeze(-1)


def _dict_add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _dict_sub(a: dict, b: dict) -> dict:
    return {k: a[k] - b[k] for k in a}


def softclamp_grad_norm(
    t: torch.Tensor, max_value: float, eps: float = 1e-6
) -> torch.Tensor:
    """Tanh-based soft norm clamping (G265 — lucidrains/titans-pytorch).

    Caps the Frobenius norm of `t` (over the last two dims) at roughly
    `max_value`, but smoothly via tanh — unlike hard clipping
    (`torch.nn.utils.clip_grad_norm_`), which has a discontinuity at the
    threshold and a zero-gradient region beyond it. The soft variant has
    nonzero gradient everywhere; backprop never "dead-ends" on
    over-threshold inputs.

    Per-matrix behavior:
        norm = ||t||_F                                  computed over (-2, -1)
        target = max_value * tanh(norm / max_value)
        t_clamped = t * (target / (norm + eps))

    When `norm << max_value`: tanh(x) ≈ x, target ≈ norm, scale ≈ 1, no-op.
    When `norm >> max_value`: tanh saturates to 1, target ≈ max_value,
        scale ≈ max_value / norm, t_clamped has norm ≈ max_value.
    Transition is smooth and differentiable; suitable for use inside the
    NMM inner loop without disrupting outer autograd.

    Use this as a safety net BEFORE Newton-Schulz when the per-token
    gradient magnitude can spike (e.g., early training, scan path with
    M_0-fixed inner-loss). NS5 already Frobenius-normalises but its
    iteration is only well-conditioned when the input's spectral norm
    is bounded; tail spikes can push it into bad numerics.
    """
    norm = t.norm(dim=(-2, -1), keepdim=True)
    target = max_value * torch.tanh(norm / max_value)
    scale = target / (norm + eps)
    return t * scale


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """5-step Newton-Schulz iteration: drives the spectral norm of G toward 1.

    Two non-obvious correctness guards:

    1. The entire iteration runs under `autocast(enabled=False)`. A bare
       `G.float()` under an ambient bf16 autocast is silently undone — matmul
       inputs get re-cast to bf16, the iteration accumulates in bf16, and the
       spectral-norm fixed point ends up in [0.7, 1.4] instead of ~1. The
       disabled-autocast wrapper makes the fp32 cast persist across matmuls.

    2. NS converges on wide matrices (cols >= rows). Tall gradients such as
       W1/W_gate (shape [4d, d]) must be transposed before the iteration and
       transposed back after; W2 ([d, 4d]) is already wide.

    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are from Jordan et al.
    (Muon / nanogpt), tuned to the fp32 fixed point.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        G = G.float()
        should_transpose = G.shape[-2] > G.shape[-1]
        if should_transpose:
            G = G.mT
        G = G / (G.norm(dim=(-2, -1), keepdim=True) + eps)
        for _ in range(steps):
            A = G @ G.mT
            G = a * G + (b * A + c * (A @ A)) @ G
        if should_transpose:
            G = G.mT
    return G.to(orig_dtype)


# Chebyshev-optimised Newton-Schulz (CANS), stationary 3-step variant.
# arxiv 2506.10935 — same polynomial form as NS5 but with coefficients
# minimax-optimised for the *actual* post-F-norm singular value range of the
# inputs, instead of NS5's coefficients which are tuned for an idealised
# fixed-point. Cheaper (3 iterations vs 5) and ~4.6× better orthogonalisation
# error than NS5-5 at the NMM shapes the coefficients were tuned for.
#
# Per-shape derivation: the singular value range of a F-norm-normalised
# random matrix depends on the matrix shape (Marchenko-Pastur). Baking a
# single (a, b, c) tuned for one shape gives an OK answer at other shapes
# but is sub-optimal — and at very different aspect ratios (e.g. low-rank
# factors) the polynomial may not even converge. We derive coefficients
# per shape at NMM construction and cache them in a module-level dict
# keyed by `(max(rows, cols), min(rows, cols))`.
_CANS_STATIONARY_3STEP_STEPS = 3

# Fallback coefficients — gpt2_small full-rank NMM MemoryMLP (W1 shape
# 3072×768, sv range [0.0228, 0.0542]). Used if `_derive_cans_coefficients`
# is called with a shape that hasn't been pre-cached AND the live derivation
# fails. Kept as a guard rather than as a primary path.
_CANS_STATIONARY_3STEP_FALLBACK = (3.8641, -9.7196, 9.7101)

# Module-level cache of per-shape CANS coefficients. Key: normalised shape
# tuple `(max_dim, min_dim)`. Value: (a, b, c). Shared across all
# NeuralMemoryModule instances in the process — every layer at the same
# model size derives the same coefficients, so caching by shape avoids
# redundant scipy.DE runs.
_CANS_COEFFICIENT_CACHE: dict[tuple[int, int], tuple[float, float, float]] = {}


def _derive_cans_coefficients(
    rows: int,
    cols: int,
    n_steps: int = _CANS_STATIONARY_3STEP_STEPS,
    n_samples: int = 8,
    pct_lo: float = 10.0,
    n_grid: int = 400,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Derive CANS-stationary (a, b, c) for a given matrix shape.

    Algorithm:
      1. Sample `n_samples` random Gaussian matrices of shape (rows, cols),
         F-norm normalise, take singular values.
      2. SV range = [p10, max] across all samples (the p10 floor excludes
         near-zero SVs that no polynomial can orthogonalise in finite steps).
      3. Differential-evolution minimise max|f^n_steps(σ) - 1| over the SV
         range, where f(σ) = aσ + bσ³ + cσ⁵.

    Cached in `_CANS_COEFFICIENT_CACHE` keyed by (max(rows, cols),
    min(rows, cols)). The SV distribution depends only on the aspect ratio
    after F-norm normalisation, so swapping rows/cols hits the same cache
    entry (legitimate — NS5/CANS transpose tall matrices anyway).

    Pure-CPU numpy + scipy; ~1-5s per unique shape. Pre-warm at NMM
    construction so the first training step doesn't pay this latency.
    """
    import numpy as np
    from scipy.optimize import differential_evolution

    key = (max(rows, cols), min(rows, cols))
    if key in _CANS_COEFFICIENT_CACHE:
        return _CANS_COEFFICIENT_CACHE[key]

    rng = np.random.default_rng(seed)
    all_svs: list[float] = []
    for _ in range(n_samples):
        G = rng.standard_normal((rows, cols))
        # F-norm normalise to match the NS5/CANS prologue.
        G = G / (np.linalg.norm(G) + 1e-7)
        sv = np.linalg.svd(G, compute_uv=False)
        all_svs.extend(sv.tolist())
    sv_arr = np.asarray(all_svs)
    sv_min = float(np.percentile(sv_arr, pct_lo))
    sv_max = float(np.max(sv_arr))

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
    _CANS_COEFFICIENT_CACHE[key] = coeffs
    return coeffs


def _cans_dispatch(G: torch.Tensor) -> torch.Tensor:
    """Per-call dispatcher used by NMM's `_ns5_fn` when `use_cans=True`.

    Reads the cached coefficients for the input's transposed-shape key and
    forwards to `cans_stationary`. The cache is pre-warmed at NMM
    construction in `_prewarm_cans_coefficients`, so the cache miss path
    (live derivation in `cans_stationary`) is a safety net only.
    """
    rows = G.shape[-2]
    cols = G.shape[-1]
    if rows > cols:
        # NS5/CANS transpose tall to wide; the cache is keyed by the
        # post-transpose shape so we mirror that here.
        rows, cols = cols, rows
    coefs = _CANS_COEFFICIENT_CACHE.get(
        (cols, rows), _CANS_STATIONARY_3STEP_FALLBACK,
    )
    return cans_stationary(G, coefs=coefs)


def _prewarm_cans_coefficients(n_embd: int, expansion: int, low_rank=None) -> None:
    """Pre-derive `(a, b, c)` for every unique recurrent-weight shape an
    NMM with these hyperparameters will produce, populating the module-
    level `_CANS_COEFFICIENT_CACHE`. Idempotent (cached entries are skipped
    inside `_derive_cans_coefficients`).

    Full-rank produces ONE unique gradient shape: `(expansion * n_embd, n_embd)`.
    Low-rank produces TWO: `(n_embd, r)` and `(expansion * n_embd, r)`.
    """
    h = n_embd * expansion
    if low_rank is None:
        shapes = [(h, n_embd)]
    else:
        r = int(low_rank)
        shapes = [(n_embd, r), (h, r)]
    for rows, cols in shapes:
        _derive_cans_coefficients(rows, cols)


def cans_stationary(
    G: torch.Tensor,
    coefs: tuple[float, float, float] | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """3-step CANS-stationary iteration.

    `coefs`: explicit `(a, b, c)` tuple. When provided, used directly —
    this is the production path (NMM passes its pre-derived per-shape
    coefficients via the `_ns5_fn` closure). When `None`, looks up the
    cache by the input's transposed-shape key; falls back to the
    gpt2_small full-rank coefficients if the cache misses.

    Step count (3) is baked into the coefficients — they're DE-derived for
    *exactly* 3 iterations over the input's SV range. Running for more or
    fewer steps degrades convergence sharply.

    Same fp32 autocast guard and transpose guard as `newton_schulz5`.
    """
    orig_dtype = G.dtype
    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        G = G.float()
        should_transpose = G.shape[-2] > G.shape[-1]
        if should_transpose:
            G = G.mT
        if coefs is None:
            rows, cols = G.shape[-2], G.shape[-1]
            key = (max(rows, cols), min(rows, cols))
            coefs = _CANS_COEFFICIENT_CACHE.get(
                key, _CANS_STATIONARY_3STEP_FALLBACK,
            )
        a, b, c = coefs
        G = G / (G.norm(dim=(-2, -1), keepdim=True) + eps)
        for _ in range(_CANS_STATIONARY_3STEP_STEPS):
            A = G @ G.mT
            G = a * G + (b * A + c * (A @ A)) @ G
        if should_transpose:
            G = G.mT
    return G.to(orig_dtype)


# Gram-iteration Newton-Schulz orthogonalization (Tri Dao et al., POLAR_EXPRESS
# coefficients with restart at iteration 2). Reimplemented locally in pure
# PyTorch — replaces the external `gram_newton_schulz` library.
#
# Why we have our own copy:
#   1. The upstream library does an in-place divide on the F-norm step
#      (`X /= X.norm(...) + eps`), which mutates a tensor saved for the
#      norm's backward. Under torch.compile + AOT autograd this trips the
#      version-check and raises `BackendCompilerFailed`. We do an
#      out-of-place divide here, so the compiled graph is well-formed.
#   2. The library wraps `__call__` with `torch.compile(mode='reduce-
#      overhead')` by default. With our outer `--compile-model`, that
#      double-compile and its CUDA-graph memory pools produced ~10 GiB
#      of extra per-layer activation memory at full recipe scale,
#      blowing past 16 GiB VRAM.
#   3. The kernel-backed path requires Hopper/Blackwell GPU + CUDA 12.9+
#      + quack-kernels + nvidia-cutlass-dsl. At our batch=1 NMM shapes
#      the quack kernels are slower than cuBLAS anyway (per the benchmark
#      in scripts/benchmark_ns5.py), so the dependency wasn't paying its
#      way. Dropping it removes the optional dep entirely.
#
# Algorithm: maintain a small n×n Gram matrix R = X X^T and accumulate
# a polynomial Q over five iterations of POLAR_EXPRESS coefficients,
# with a reset at iteration 2 that re-orthogonalizes the intermediate
# (X ← Q @ X, R ← X X^T, Q ← 0). Result: X ← Q_final @ X.
#
# At gpt2_small NMM weight shapes (α = m/n = 4, m=3072, n=768), the
# rectangular matmuls in the outer X-update are replaced by n×n
# operations in the inner loop — 42% FLOP reduction vs stock NS5's 2T
# rectangular matmuls (Tri Dao paper claim).

# POLAR_EXPRESS per-iteration coefficients, scaled by a 1.05x safety
# factor (arxiv 2505.16932 §5). Copied verbatim from the upstream
# `gram_newton_schulz.coefficients` module so we don't depend on it.
_POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563 / 1.05, -23.595886519098837 / 1.05 ** 3,
     17.300387312530933 / 1.05 ** 5),
    (4.107059111542203 / 1.05, -2.9478499167379106 / 1.05 ** 3,
     0.5448431082926601 / 1.05 ** 5),
    (3.9486908534822946 / 1.05, -2.908902115962949 / 1.05 ** 3,
     0.5518191394370137 / 1.05 ** 5),
    (3.3184196573706015 / 1.05, -2.488488024314874 / 1.05 ** 3,
     0.51004894012372 / 1.05 ** 5),
    (2.300652019954817 / 1.05, -1.6689039845747493 / 1.05 ** 3,
     0.4188073119525673 / 1.05 ** 5),
)
# Iteration indices (0-indexed) at which to re-orthogonalize the intermediate.
_GRAM_RESET_ITERATIONS: frozenset = frozenset({2})


def gram_newton_schulz(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Gram-iteration Newton-Schulz orthogonalization (pure PyTorch).

    Two dtype phases inside an `autocast(enabled=False)` scope:

      1. **fp32 for the F-norm step.** The Frobenius norm of a small-valued
         gradient can underflow in fp16 — keep it in fp32 for stability.
      2. **fp16 for the Gram iteration.** Unlike stock NS5 (Muon coefficients,
         G226 fp32 invariant), POLAR_EXPRESS coefficients with reset at
         iter 2 tolerate fp16 just fine — measured |orth error| difference
         vs fp32 is < 1e-2 at NMM shapes. fp16 buys ~2.3× speed on consumer
         Blackwell tensor cores (matches what the upstream library does).

    Same shape contract as `newton_schulz5`: input and output have the same
    shape; spectral norm of the output is driven toward 1.
    """
    orig_dtype = G.dtype
    orig_shape = G.shape

    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        X = G.float()
        # Normalize to (B, m, n) so the iteration's matmuls are batched.
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])

        # Iterate on the smaller-of-(m, n) Gram matrix. The transpose is a
        # view (no copy); we transpose back at the end so the caller's
        # output shape matches.
        should_transpose = X.size(-2) > X.size(-1)
        if should_transpose:
            X = X.mT

        # F-norm normalize, OUT-OF-PLACE. This is the load-bearing fix vs
        # the upstream library's `X /= ...`: the in-place mutation breaks
        # AOT autograd's tensor-version check, because `X.norm(...)`
        # saves X at version 0 and the in-place divide bumps it to 1.
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

        # Drop to fp16 for the iteration — half-precision tensor cores are
        # ~2× faster than fp32 on consumer Blackwell, and POLAR_EXPRESS +
        # reset at iter 2 is robust to fp16 precision (measured: |sv-1|
        # within 1e-2 of fp32 at NMM shapes).
        X = X.to(torch.float16)

        # R = X X^T is the small n×n Gram matrix.
        R = X @ X.mT
        I = torch.eye(R.size(-1), device=R.device, dtype=R.dtype)
        Q = None

        for i, (a, b, c) in enumerate(_POLAR_EXPRESS_COEFFICIENTS):
            if i in _GRAM_RESET_ITERATIONS and i != 0:
                # Re-orthogonalize: apply the accumulated polynomial to X,
                # then recompute R from the partial result. This is the
                # mechanism that lets POLAR_EXPRESS converge in 5 iters
                # what plain NS5 needs Muon-tuned coefficients to match.
                X = Q @ X
                R = X @ X.mT
                Q = None

            # Z = b·R + c·R²  (one fused baddbmm: beta=b · R + alpha=c · R@R)
            Z = torch.baddbmm(R, R, R, alpha=c, beta=b)

            if i == 0 or i in _GRAM_RESET_ITERATIONS:
                Q = Z + a * I
            else:
                # Q ← a·Q + Q·Z = Q · (a·I + Z) — right-mult; equivalent to
                # composing polynomial steps from the right.
                Q = torch.baddbmm(Q, Q, Z, beta=a)

            # Propagate R for the next iter, unless next iter resets it.
            is_last = i == len(_POLAR_EXPRESS_COEFFICIENTS) - 1
            next_is_reset = (i + 1) in _GRAM_RESET_ITERATIONS
            if not is_last and not next_is_reset:
                # R_next = (a·I + Z)·R·(a·I + Z), two fused baddbmms.
                RZ = torch.baddbmm(R, R, Z, beta=a)
                R = torch.baddbmm(RZ, Z, RZ, beta=a)

        # Final apply: X ← Q · X gives the orthogonalized rectangular
        # output. The polynomial Q is the product Π(a_i·I + Z_i) for the
        # iterations after the reset, applied to X via this one rectangular
        # matmul. This is where the FLOP win lives: only two rectangular
        # matmuls overall (initial R and this final Q·X), vs 2T for stock NS.
        X = Q @ X

        if should_transpose:
            X = X.mT
        # `.reshape` (not `.view`) because the final `.mT` above leaves X as a
        # non-contiguous view of a contiguous matmul output. `.view` would
        # error on the 4-D path (`per_token_ns5`: [B, T, h, d]) where we need
        # to unflatten the leading dim back to (B, T). `.reshape` returns the
        # existing tensor when strides happen to be compatible (the 2-D / 3-D
        # cases hit this fast path) and falls back to a contiguous copy only
        # on the path that genuinely needs one.
        X = X.reshape(orig_shape)

    return X.to(orig_dtype)


def _make_grad_fn(memory_mlp: nn.Module, spectral_norm: bool):
    """Build the cached vmap(grad(inner_loss)) per-sample gradient function.

    The reduction is derived from `spectral_norm`, NOT hardcoded:
      - True: 'sum' — Newton-Schulz normalises the gradient's spectral norm
        to 1, so the d_model factor cancels and sum vs mean is invisible
        downstream. Matches paper Eq. 12 (squared L2 norm).
      - False: 'mean' — without NS, 'sum' would let the gradient grow with
        d_model, scaling W_theta's effective LR by ~d. 'mean' keeps the
        gradient magnitude independent of d.

    Mismatch between this flag and the downstream NS branch is silent and
    catastrophic: a user toggling nmm_spectral_norm=False after construction
    keeps the cached 'sum' reduction (no NS to cancel the d factor),
    inflating per-token LR ~768x at gpt2_small dims. Treat
    nmm_spectral_norm as construction-time-only; rebuild the NMM to change it.
    """
    reduction = "sum" if spectral_norm else "mean"

    def inner_loss(params, k_hat, v):
        # params: dict of per-sample weights ([h,d] / [d,h]); k_hat, v: [d]
        pred = functional_call(memory_mlp, params, k_hat)
        return F.mse_loss(pred, v, reduction=reduction)

    # argnums=0 (default) -> grad w.r.t. params (must stay first arg).
    return vmap(grad(inner_loss), in_dims=(0, 0, 0))


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D conv with strict causality (left-only padding).

    Public API operates on `[B, T, dim]` (transformer convention); the
    underlying nn.Conv1d expects `[B, dim, T]`. Padding inside Conv1d would
    be symmetric, which leaks future tokens; we pad left-only by `k-1`.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1 (got {kernel_size})")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=0, groups=dim, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        x = x.transpose(1, 2)                              # [B, dim, T]
        x = F.pad(x, (self.kernel_size - 1, 0))            # left-pad only
        x = self.conv(x)                                    # [B, dim, T]
        return x.transpose(1, 2)                           # [B, T, dim]


class NMMProjection(nn.Module):
    """Q/K/V projection: Linear -> CausalDepthwiseConv1d -> Pointwise (1x1).

    Implements depthwise-separable convolution per paper §4.4 in the
    stricter reading where the Q/K/V `linear` is logically distinct from
    the (depthwise + pointwise) separable conv. Note: the leading
    `linear` is also a per-token channel mix and could in principle play
    the pointwise role on its own (looser reading); we keep both so the
    paper-text "depthwise-separable" is satisfied verbatim. Adds ~d²
    params per Q/K/V per layer; in our consumer-GPU recipe (d=768,
    n_layer=12) that's ~21M extra params total, modest vs the model's
    ~150M backbone.

    SiLU + L2-norm are applied at the call site, not inside the module —
    putting SiLU inside would silently produce silu(silu(x)) at the call site.

    Submodule names `linear`, `conv`, `pointwise` are load-bearing for §4.1
    optimizer routing: param paths like `blocks.X.nmm.k_proj.linear.weight`
    get into the NMM group via the `'nmm'` substring; renaming to anything
    containing `'norm'`, `'bias'`, or `'gamma'` would misroute to no_decay.
    """

    def __init__(self, n_embd: int, kernel_size: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.conv = CausalDepthwiseConv1d(n_embd, kernel_size)
        self.pointwise = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv(self.linear(x)))


class MemoryMLP(nn.Module):
    """SiLU-GLU gated two-layer MLP (L_M = 2) with ResidualNorm.

    Full-rank (default):
        silu(W1 x) * sigmoid(W_gate x)  ->  W2  ->  norm(.) + x
        State: 3 weight matrices {W1, W_gate, W2}.

    Low-rank (G262, `low_rank=r`): each of the three weight matrices is
    factored into two `nn.Linear` modules with intermediate dim `r`:
        W1_b(W1_a(x))   instead of W1(x)
    Per-step recurrent state grows from 3 keys to 6 keys but each key is
    much smaller; at gpt2_small d=768, expansion=4, r=64 the state
    footprint drops ~10x. Newton-Schulz still operates on each 2D matrix
    independently and converges on the factored rectangles.

    `norm` is a fixed stabilizer trained by the outer optimizer only — it
    is NOT recurrent state regardless of rank choice. Recurrent state is
    discovered at runtime by `_collect_state_keys` (everything except
    `norm.*`) so the architecture stays parametric.

    Dtype handling: when `functional_call` overrides recurrent weights
    with bf16 state, `x` flows through this MLP in bf16, but
    `self.norm.weight` / `self.norm.bias` remain fp32 (they are
    outer-trained params that AdamW expects in fp32). We explicitly cast
    through fp32 around the norm to unblock `state_dtype="bf16"` (G256).
    """

    def __init__(self, d: int, expansion: int = 4, low_rank=None):
        super().__init__()
        h = d * expansion
        self.d = d
        self.h = h
        self.low_rank = low_rank
        if low_rank is None:
            # Full-rank path (paper-faithful).
            self.W1 = nn.Linear(d, h, bias=False)
            self.W_gate = nn.Linear(d, h, bias=False)
            self.W2 = nn.Linear(h, d, bias=False)
        else:
            # Low-rank factorization: each Wx becomes Wx_b @ Wx_a (G262).
            # We use the suffix '_a' for the d->r (or h->r) projection and
            # '_b' for the r->h (or r->d) projection.
            r = int(low_rank)
            self.W1_a = nn.Linear(d, r, bias=False)
            self.W1_b = nn.Linear(r, h, bias=False)
            self.W_gate_a = nn.Linear(d, r, bias=False)
            self.W_gate_b = nn.Linear(r, h, bias=False)
            self.W2_a = nn.Linear(h, r, bias=False)
            self.W2_b = nn.Linear(r, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank is None:
            h = F.silu(self.W1(x)) * torch.sigmoid(self.W_gate(x))
            y = self.W2(h)
        else:
            h = F.silu(self.W1_b(self.W1_a(x))) * torch.sigmoid(
                self.W_gate_b(self.W_gate_a(x))
            )
            y = self.W2_b(self.W2_a(h))
        # Run LayerNorm in fp32 regardless of (x, W*) dtype; cast back to
        # match the input. Identity-cast is free in fp32, ~negligible in
        # bf16, and unblocks `state_dtype="bf16"` mode (G256).
        orig_dtype = y.dtype
        y = self.norm(y.float()).to(orig_dtype)
        return y + x.to(orig_dtype)


class NeuralMemoryModule(nn.Module):
    """Online memory module with surprise-driven weight updates.

    The constructor wires together the components from docs/PLAN.md §1.2–§1.4:
    Q/K/V projections, the three data-dependent update params, the
    MemoryMLP (whose W*.weight ARE the meta-learned initial state), and
    a learnable per-channel output scale.

    Phase 1.4 surface: __init__, _build_init_M, init_state. The cached
    per-sample gradient function (§1.5), Newton-Schulz (§1.6), step
    (§1.7), and forward_chunk (§1.8) are added by subsequent tasks.
    """

    def __init__(
        self,
        n_embd: int,
        expansion: int = 4,
        kernel_size: int = 4,
        spectral_norm: bool = True,
        finetune_mode: bool = True,
        retrieval_from_M_prev: bool = True,
        state_dtype: str = "fp32",
        low_rank=None,
        softclamp_max=None,
        block_size: int = 1,
        per_token_ns5: bool = False,
        detach_state_between_blocks: bool = False,
        lookahead_value: bool = False,
        per_param_lr_modulation: bool = False,
        momentum_order: int = 1,
        ns5_steps: int = 5,
        use_gram_ns5: bool = False,
        use_cans: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.nmm_spectral_norm = spectral_norm
        self.finetune_mode = finetune_mode
        self.low_rank = low_rank
        # The analytical inner-gradient kernel (model/nmm_fused.py) is the
        # only path on the sequential `block_size=1` recurrence — the
        # earlier `fused_kernel` toggle is gone (always-on now). Decode
        # paths (`step`, `step_with_conv`) still use the vmap-based
        # `per_sample_grad_fn` reference because they fire once per token
        # at inference time and aren't on any training hot path.
        # Soft norm-clamp threshold applied to surprise gradients BEFORE NS5.
        # None = disabled (paper-strict). See `softclamp_grad_norm` docstring.
        self.softclamp_max = softclamp_max
        # Block size for chunk-as-update aggregation. 1 = paper-strict
        # per-token recurrence (existing sequential path). >1 = blockwise:
        # one update per `block_size` tokens, with TC-engaged batched matmul
        # forward. See `_forward_chunk_blockwise` and SPEC §5.8.
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1 (got {block_size})")
        self.block_size = int(block_size)
        # G267: per-token NS5 + per-token θ weighting in the blockwise path.
        # When True, paper Eq 16's `Σ_t θ_t · NS5(∇_t)` is implemented
        # exactly (instead of v1's `θ_mean · NS5(Σ_t ∇_t)`). Costs more
        # memory (per-token gradient tensors of shape [B, block, H, D]
        # per state key) — reach for `nmm_low_rank` if memory is tight.
        # No-op when block_size=1 (single-token block has theta_mean = θ_t
        # and per-token NS5 = single NS5).
        self.per_token_ns5 = bool(per_token_ns5)
        # G268: truncated BPTT — detach (M, S) at each block boundary in the
        # blockwise path so the backward graph spans one block instead of the
        # full chunk. No-op when block_size=1 (only one block per chunk).
        self.detach_state_between_blocks = bool(detach_state_between_blocks)
        # G269: predictive inner loss — use v_{t+1} as the target for token t
        # (next-token reconstruction) instead of v_t (same-token
        # reconstruction). Last token in chunk has no v_{t+1}; its inner-loss
        # contribution is dropped (the chunk-level helpers below produce
        # zero gradient for that position).
        self.lookahead_value = bool(lookahead_value)
        # G270: per-state-key θ — W_theta projects to K independent scalars
        # (one per recurrent weight key) instead of one shared scalar.
        self.per_param_lr_modulation = bool(per_param_lr_modulation)
        # G272: order of the momentum recurrence (>=1). At N>1, S becomes a
        # list of N momenta with their own η projections (W_eta gains N
        # output dims). M's update reads the deepest level (S_N).
        if momentum_order < 1:
            raise ValueError(f"momentum_order must be >= 1 (got {momentum_order})")
        self.momentum_order = int(momentum_order)
        self.use_gram_ns5 = bool(use_gram_ns5)
        self.use_cans = bool(use_cans)
        if self.use_gram_ns5 and self.use_cans:
            raise ValueError(
                "use_gram_ns5 and use_cans are mutually exclusive — both "
                "replace the stock NS5 path. Pick one."
            )
        # Number of Newton-Schulz iterations. 5 = paper-faithful (Muon
        # coefficients tuned for this fixed point); lower drifts the
        # spectral norm away from 1 (see config.nmm_ns5_steps docstring).
        if not isinstance(ns5_steps, int) or ns5_steps < 1:
            raise ValueError(f"ns5_steps must be a positive int (got {ns5_steps!r})")
        self.ns5_steps = int(ns5_steps)
        # Resolved NS5 callable — used by every path that applies NS5.
        # Bind `steps` here so call sites stay `self._ns5_fn(g)` with no
        # extra argument threading. Three options (mutually exclusive):
        #   1. use_cans=True → 3-step CANS-stationary with coefficients tuned
        #      for the gpt2_small NMM sv range (see scripts/benchmark_ns5.py).
        #      Ignores ns5_steps.
        #   2. use_gram_ns5=True → Gram-iteration Newton-Schulz (POLAR_EXPRESS
        #      coefficients + reset at iter 2). Different algorithm; the
        #      coefficient table is fixed, so ns5_steps is ignored.
        #   3. default → plain stock NS5 with `ns5_steps` iterations.
        # All three paths follow the same lambda shape:
        #   self._ns5_fn = lambda g, _f=<callable>, _s=<int>: _f(g)
        # The `_s` slot lets introspection tests inspect the bound step
        # count even when the callable itself ignores it.
        if self.use_cans:
            # Pre-derive per-shape coefficients for every recurrent-weight
            # gradient this NMM will see. We resolve `state_keys` to their
            # `memory_mlp` params (built immediately below — but ordering is
            # safe because the params already exist via __init__'s call
            # chain). Caching is module-level and shape-keyed, so multiple
            # NMM instances at the same model size share the cache.
            #
            # Deferred to a post-memory_mlp-build pre-warm helper so the
            # order-of-init in this __init__ stays linear and readable. See
            # `_prewarm_cans_coefficients` below; it's invoked after the
            # memory_mlp construction further down.
            _steps = _CANS_STATIONARY_3STEP_STEPS
            self._ns5_fn = lambda g, _s=_steps: _cans_dispatch(g)
        elif self.use_gram_ns5:
            _ns5_base = gram_newton_schulz
            _steps = len(_POLAR_EXPRESS_COEFFICIENTS)
            self._ns5_fn = lambda g, _f=_ns5_base, _s=_steps: _f(g)
        else:
            _ns5_base = newton_schulz5
            _steps = self.ns5_steps
            self._ns5_fn = lambda g, _f=_ns5_base, _s=_steps: _f(g, steps=_s)
        # Paper Eq. 15: y_t = M(q_t) where M is M_{t-1} (read-then-write).
        # Default False = lucidrains "write-then-read" (retrieve from M_t).
        self.retrieval_from_M_prev = retrieval_from_M_prev

        allowed_state_dtypes = set(_STATE_DTYPE_MAP) | {"int8"}
        if state_dtype not in allowed_state_dtypes:
            raise ValueError(
                f"state_dtype must be one of {sorted(allowed_state_dtypes)} "
                f"(got {state_dtype!r})."
            )
        self.state_dtype_name = state_dtype
        # For int8 state, the "fp32 inner" arithmetic dtype is fp32 — the
        # actual stored state is int8 + a fp32 scale, but every operation
        # inside forward dequantizes to fp32 first.
        self.state_dtype = _STATE_DTYPE_MAP.get(state_dtype, torch.float32)
        self.int8_state = (state_dtype == "int8")

        # Q/K/V projections — SiLU/L2 applied at call site, not inside.
        self.k_proj = NMMProjection(n_embd, kernel_size)
        self.q_proj = NMMProjection(n_embd, kernel_size)
        self.v_proj = NMMProjection(n_embd, kernel_size)

        # Per-token data-dependent update params (sigmoid+squeeze at call site).
        # W_theta output dim: 1 (default) or K = len(state_keys) when
        # per_param_lr_modulation is True (G270). Resolve state_keys here
        # via memory_mlp's expected layout — we need this BEFORE building
        # memory_mlp itself, so derive it from `low_rank`.
        if self.per_param_lr_modulation:
            n_theta = 6 if low_rank is not None else 3
        else:
            n_theta = 1
        self.W_theta = nn.Linear(n_embd, n_theta, bias=False)
        # W_eta output dim: 1 (default) or momentum_order N (G272). Each
        # level of the momentum stack uses an independently-learned η.
        self.W_eta = nn.Linear(n_embd, self.momentum_order, bias=False)
        self.W_alpha = nn.Linear(n_embd, 1, bias=False)
        # Cache the per-θ output dim so call sites don't rebranch on the flag.
        self._n_theta = n_theta

        # MemoryMLP. Its weight tensors ARE the meta-learned initial values
        # of M; _build_init_M reads them at sequence/document start to seed
        # the recurrent state. Architecture is full-rank or low-rank
        # depending on the `low_rank` flag.
        self.memory_mlp = MemoryMLP(n_embd, expansion, low_rank=low_rank)

        # CANS pre-warm: now that we know the recurrent-weight shapes, derive
        # per-shape coefficients before the first forward. Without this,
        # `_cans_dispatch` would call `cans_stationary` with the fallback
        # coefficients (gpt2_small tuning) on the first call until the cache
        # is lazily populated. Pre-warming is ~1-5s per unique shape, paid
        # once per model size per process.
        if self.use_cans:
            _prewarm_cans_coefficients(n_embd, expansion, low_rank=low_rank)
        # Xavier-uniform every recurrent weight, both full-rank (W1, W_gate,
        # W2) and low-rank (W*_a, W*_b).
        for name, p in self.memory_mlp.named_parameters():
            if name.startswith("norm."):
                continue  # LayerNorm is not recurrent.
            nn.init.xavier_uniform_(p)

        # Discover the recurrent state's key set ONCE, sorted for
        # determinism. Used by _build_init_M, the flat-tensor checkpoint
        # plumbing, and the block-level state flatten/unflatten. The set
        # depends on whether MemoryMLP is full-rank (3 keys) or low-rank
        # (6 keys).
        self.state_keys = tuple(sorted(
            name for name, _ in self.memory_mlp.named_parameters()
            if not name.startswith("norm.")
        ))

        # finetune_mode=True: zero-init silences y_mem at step 0, preserving
        # the pretrained GPT-2 residual exactly (ResidualNorm makes a bare
        # W2=0 not enough; out_scale is the only reliable mute).
        # finetune_mode=False: NMM contributes from step 1, no residual to preserve.
        if finetune_mode:
            self.out_scale = nn.Parameter(torch.zeros(n_embd))
        else:
            self.out_scale = nn.Parameter(torch.ones(n_embd))

        # Per-sample inner gradient — built ONCE; recreating vmap(grad(...)) in
        # forward/step is measurably slower at T=512. Reduction is baked in via
        # spectral_norm at construction time; mutating self.nmm_spectral_norm
        # later does NOT change the cached reduction. Lock the construction
        # value so downstream sanity checks can detect drift.
        self.per_sample_grad_fn = _make_grad_fn(
            self.memory_mlp, spectral_norm=self.nmm_spectral_norm
        )
        self._spectral_norm_at_init = self.nmm_spectral_norm

        # Per-sample retrieval — same caching rationale. functional_call reads
        # self.memory_mlp at call time, so device moves of NMM after __init__
        # are still respected.
        def _retrieve_one_sample(m_dict, q):
            return functional_call(
                self.memory_mlp, m_dict, q.unsqueeze(0)
            ).squeeze(0)

        self._batched_retrieve = vmap(_retrieve_one_sample, in_dims=(0, 0))

    def _build_init_M(self, B: int, device) -> dict:
        """Per-sample-batched initial M dict from `memory_mlp` parameters,
        cast to `self.state_dtype` (fp32 by default, bf16 when configured).

        Iterates over `self.state_keys` so the dict shape adapts to the
        MemoryMLP architecture (3 keys for full-rank, 6 for low-rank).

        `.expand(B, -1, -1)` is a stride-0 view; vmap with in_dims=0 over
        such views is undefined in the batched autograd interpreter, so
        we `.clone()` to materialize normal strides. Order is
        `.to(device, dtype).clone()`: on the same device the dtype cast
        happens before the materialization, avoiding a fp32 staging copy
        when state_dtype is bf16.
        """
        dt = self.state_dtype
        result = {}
        for key in self.state_keys:
            # The key is like "W1.weight" or "W1_a.weight"; resolve via
            # named_parameters lookup to handle nested names robustly.
            p = dict(self.memory_mlp.named_parameters())[key]
            result[key] = (
                p.unsqueeze(0).expand(B, -1, -1).to(device, dtype=dt).clone()
            )
        return result

    def init_state(self, B: int, device) -> tuple:
        M = self._build_init_M(B, device)
        if self.momentum_order == 1:
            S = {k: torch.zeros_like(v) for k, v in M.items()}
        else:
            # G272: N-th order momentum -> N independent zero-initialized
            # S levels.
            S = tuple(
                {k: torch.zeros_like(v) for k, v in M.items()}
                for _ in range(self.momentum_order)
            )
        # G275 int8 state: quantize both M and S immediately on construction
        # so the returned state is in int8 form (with _qs scale companions
        # alongside the value entries in the dict). For a zero S the scale
        # clamp gives a tiny non-zero scale and all int8 values are 0 —
        # dequant produces zero correctly.
        if self.int8_state:
            M = _quant_dict(M, self.state_keys)
            if self.momentum_order == 1:
                S = _quant_dict(S, self.state_keys)
            else:
                S = tuple(_quant_dict(s, self.state_keys) for s in S)
        # Conv buffer: last (k-1) linear projections of Q/K/V from the
        # previous chunk, used to seed the depthwise conv at the current
        # chunk's start so it sees a full k-token window across the chunk
        # boundary (paper §4.4 conv loses context at chunk boundaries
        # otherwise). At fresh-init the buffer is zeros — equivalent to
        # the legacy left-pad-with-zeros behavior. Dtype matches
        # `state_dtype`; the forward paths cast to the current chunk's
        # dtype if it differs.
        conv_buf = self._init_conv_buf(B, device)
        return (M, S, conv_buf)

    def _project_qkv_with_buf(self, x_chunk: torch.Tensor, conv_buf: dict):
        """Apply Q/K/V projections (linear -> buffer-aware conv -> pointwise).

        Item 6: the chunk-path conv used to left-pad each chunk with k-1
        zeros, losing context at chunk boundaries. Now we maintain a
        rolling buffer of the last k-1 linear projections and prepend
        them to the chunk's linear projections before the conv runs.

        `conv_buf`: dict {"q","k","v"} of [B, k-1, d] linear projections
        from the previous chunk's tail. At fresh init the entries are
        zeros (matches legacy left-pad).

        Returns `(q_pw, k_pw, v_pw, new_conv_buf)` where:
          - q_pw/k_pw/v_pw are post-pointwise outputs [B, T, d] (caller
            still applies silu + L2 as needed).
          - new_conv_buf is the rolled buffer for the NEXT chunk: the last
            (k-1) linear projections produced HERE, detached for TBPTT.
        """
        k_sz = self.k_proj.conv.kernel_size

        def _one(projection: NMMProjection, buf_x: torch.Tensor):
            lin = projection.linear(x_chunk)  # [B, T, d]
            # Buffer is always stored in `state_dtype` (see new_buf cast
            # below); the current chunk's Linear output may be bf16 under
            # autocast even when state_dtype is fp32. Promote the buffer
            # to the compute dtype so `torch.cat` doesn't error.
            if buf_x.dtype != lin.dtype:
                buf_x = buf_x.to(lin.dtype)
            if k_sz <= 1:
                # No conv context to maintain — k=1 has no kernel reach.
                conv_out = lin
                new_buf = lin.new_zeros(lin.shape[0], 0, lin.shape[2])
            else:
                combined = torch.cat([buf_x, lin], dim=1)  # [B, T+k-1, d]
                # Apply the underlying nn.Conv1d directly (no left-pad)
                # since the buffer already provides the k-1 context. The
                # public conv.forward would left-pad with zeros, defeating
                # the whole point.
                combined_t = combined.transpose(1, 2)  # [B, d, T+k-1]
                conv_out = projection.conv.conv(combined_t).transpose(1, 2)
                # New buf: last (k-1) of THIS chunk's linear projections.
                # Detached so the autograd graph doesn't span chunks, and
                # cast back to `state_dtype` so the buffer's storage dtype
                # is invariant across chunks (matches M and S semantics —
                # without this cast, an autocast-bf16 chunk's tail would
                # silently downgrade an otherwise-fp32 buffer for all
                # subsequent chunks).
                if lin.shape[1] >= k_sz - 1:
                    new_buf = lin[:, -(k_sz - 1):, :].detach()
                else:
                    # Edge case: chunk shorter than k-1 (rare). Take from
                    # the combined buffer's tail.
                    new_buf = combined[:, -(k_sz - 1):, :].detach()
                new_buf = new_buf.to(self.state_dtype)
            pw_out = projection.pointwise(conv_out)
            return pw_out, new_buf

        q_pw, new_q = _one(self.q_proj, conv_buf["q"])
        k_pw, new_k = _one(self.k_proj, conv_buf["k"])
        v_pw, new_v = _one(self.v_proj, conv_buf["v"])
        return q_pw, k_pw, v_pw, {"q": new_q, "k": new_k, "v": new_v}

    @staticmethod
    def _reset_conv_buf_at_chunk_start(
        conv_buf: dict, doc_boundaries
    ) -> dict:
        """Zero per-batch conv_buf entries whose chunk starts at a doc
        boundary. The buffer's only consumption point is chunk start (sub-
        chunk 0); mid-chunk boundaries don't affect it. So we only check
        `doc_boundaries[:, 0]`.

        Returns the conv_buf dict, possibly with entries replaced via
        `torch.where` (autograd-safe; doesn't break the rolling
        cross-chunk path)."""
        if doc_boundaries is None:
            return conv_buf
        bdry_at_0 = doc_boundaries[:, 0]
        if not bool(bdry_at_0.any()):
            return conv_buf
        mask = bdry_at_0.view(bdry_at_0.shape[0], 1, 1)
        return {
            k: torch.where(mask, torch.zeros_like(v), v)
            for k, v in conv_buf.items()
        }

    def _init_conv_buf(self, B: int, device) -> dict:
        """Zero-init conv buffer, shape `{q, k, v: [B, k-1, n_embd]}`.

        The buffer's dtype is `state_dtype` (fp32 default, bf16 when set).
        Both `_project_qkv_with_buf` and `step_with_conv` cast the rolled
        buffer back to `state_dtype` before storing, so this invariant
        holds across chunks even when the per-step compute runs in
        autocast bf16. `int8` state mode uses fp32 for the buffer
        (quantizing per-step would amplify rolling-context noise — same
        rationale as the int8-blockwise-only rule for M and S)."""
        k_sz = self.k_proj.conv.kernel_size
        dt = self.state_dtype if not self.int8_state else torch.float32
        if k_sz <= 1:
            shape = (B, 0, self.n_embd)
        else:
            shape = (B, k_sz - 1, self.n_embd)
        return {
            "q": torch.zeros(shape, device=device, dtype=dt),
            "k": torch.zeros(shape, device=device, dtype=dt),
            "v": torch.zeros(shape, device=device, dtype=dt),
        }


    def step_with_conv(
        self,
        x_t: torch.Tensor,
        state: tuple,
    ) -> tuple:
        """Decode-path single-token step.

        Item 6: the conv buffer now lives INSIDE the state tuple as the
        third element. The previous separate `conv_buffer` parameter is
        gone; callers pass `state = (M, S, conv_buf)` and receive the
        same shape back.

        x_t: [B, d]; state: (M, S, conv_buf) where `conv_buf` is a dict
        `{"q","k","v"}` each `[B, k-1, d]` of prior Linear projections.

        Returns `(y_t [B, d], new_state)` where new_state has the rolled
        conv_buf (new entry: this token's linear projection appended,
        oldest dropped).
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm})."
            )
        if self.int8_state:
            raise NotImplementedError(
                "step_with_conv() does not support nmm_state_dtype='int8'."
            )

        M_prev, S_prev, conv_buffer = state
        x_unsq = x_t.unsqueeze(1)  # [B, 1, d]

        # Linear projection only (no conv yet, no activation).
        q_lin = self.q_proj.linear(x_unsq)  # [B, 1, d]
        k_lin = self.k_proj.linear(x_unsq)
        v_lin = self.v_proj.linear(x_unsq)

        # Cast buffer to lin dtype if needed (e.g. fp32 state buf vs bf16
        # current token under autocast).
        cb_q = conv_buffer["q"]
        cb_k = conv_buffer["k"]
        cb_v = conv_buffer["v"]
        if cb_q.dtype != q_lin.dtype:
            cb_q = cb_q.to(q_lin.dtype)
            cb_k = cb_k.to(k_lin.dtype)
            cb_v = cb_v.to(v_lin.dtype)

        # Concat with buffer (last k-1 prior linear projections) -> length-k
        # input. Run conv; take the LAST position (conv at that position uses
        # the full [buffer | new] context).
        q_input = torch.cat([cb_q, q_lin], dim=1)  # [B, k, d]
        k_input = torch.cat([cb_k, k_lin], dim=1)
        v_input = torch.cat([cb_v, v_lin], dim=1)
        q_conv = self.q_proj.conv(q_input)[:, -1, :]  # [B, d]
        k_conv = self.k_proj.conv(k_input)[:, -1, :]
        v_conv = self.v_proj.conv(v_input)[:, -1, :]
        # Apply the pointwise component of the depthwise-separable conv,
        # matching the full NMMProjection.forward chain (linear -> conv ->
        # pointwise). Without this, decode would diverge from training.
        q_pw = self.q_proj.pointwise(q_conv)
        k_pw = self.k_proj.pointwise(k_conv)
        v_pw = self.v_proj.pointwise(v_conv)

        # Call-site SiLU + L2 (same pattern as the chunked forward path).
        k_hat = F.normalize(F.silu(k_pw), dim=-1)
        q_hat = F.normalize(F.silu(q_pw), dim=-1)
        v = F.silu(v_pw)

        theta_t = _project_theta(
            self.W_theta, x_t, self.state_keys, self.per_param_lr_modulation,
        )
        eta_t = torch.sigmoid(self.W_eta(x_t))
        if self.momentum_order == 1:
            eta_t = eta_t.squeeze(-1)
        alpha_t = torch.sigmoid(self.W_alpha(x_t)).squeeze(-1)

        # Cast to state_dtype — see forward_chunk note. Decode path runs
        # without autocast (the `prepare_decode` eval-mode contract), so
        # without this cast inputs would be fp32 vs. bf16 M weights.
        if self.state_dtype != k_hat.dtype:
            k_hat = k_hat.to(self.state_dtype)
            q_hat = q_hat.to(self.state_dtype)
            v = v.to(self.state_dtype)
            if isinstance(theta_t, dict):
                theta_t = {k: vv.to(self.state_dtype) for k, vv in theta_t.items()}
            else:
                theta_t = theta_t.to(self.state_dtype)
            eta_t = eta_t.to(self.state_dtype)
            alpha_t = alpha_t.to(self.state_dtype)

        g_t = self.per_sample_grad_fn(M_prev, k_hat, v)
        if self.softclamp_max is not None:
            g_t = {key: softclamp_grad_norm(g, self.softclamp_max) for key, g in g_t.items()}
        if self.nmm_spectral_norm:
            g_tilde = {key: self._ns5_fn(g) for key, g in g_t.items()}
        else:
            g_tilde = g_t

        S_t = _step_momentum(S_prev, g_tilde, theta_t, eta_t, self.momentum_order)
        M_t = _dict_add(_scale(1.0 - alpha_t, M_prev), _S_top(S_t, self.momentum_order))
        M_for_retrieval = M_prev if self.retrieval_from_M_prev else M_t
        y_t = self.out_scale * self._batched_retrieve(M_for_retrieval, q_hat)

        # Roll conv buffer: drop oldest, append the new linear projection.
        # Cast back to `state_dtype` so the buffer's storage dtype stays
        # invariant across steps (matches M/S semantics; without this an
        # autocast-bf16 step would downgrade an otherwise-fp32 buffer).
        new_conv_buf = {
            "q": torch.cat([cb_q[:, 1:, :], q_lin], dim=1).to(self.state_dtype),
            "k": torch.cat([cb_k[:, 1:, :], k_lin], dim=1).to(self.state_dtype),
            "v": torch.cat([cb_v[:, 1:, :], v_lin], dim=1).to(self.state_dtype),
        }
        return y_t, (M_t, S_t, new_conv_buf)

    def _run_inner_loop(
        self,
        k_hat_seg: torch.Tensor,
        q_hat_seg: torch.Tensor,
        v_seg: torch.Tensor,
        theta_seg: torch.Tensor,
        eta_seg: torch.Tensor,
        alpha_seg: torch.Tensor,
        *flat_state_db_init,
    ) -> tuple:
        """Run the per-token NMM update loop over a (sub)chunk and return:
          `(y_seg, *new_M_flat, *new_S_flat)`.

        All state inputs/outputs are flat tensors (not dicts) so this is
        directly wrappable in `torch.utils.checkpoint.checkpoint`, which
        requires tensor-only signatures with `use_reentrant=True`.

        The variadic tail (`*flat_state_db_init`) is unpacked as:
            M_tensors[K], S_tensors[K], db_seg, init_M_tensors[K]
        where K = `len(self.state_keys)` (3 for full-rank MemoryMLP, 6 for
        low-rank). Variable-arity is required because `torch.utils.checkpoint`
        doesn't pass kwargs to the wrapped callable; the alternative is
        K hard-coded argument names, which would tie this method to
        full-rank shape only (G262).

        `db_seg` is the [B, T_seg] bool mask of document boundaries for
        this segment (or None). When a boundary fires, the corresponding
        rows of M/S get reset from `init_M_*` (zeros for S). `init_M_*`
        is precomputed once at the chunk level (so it doesn't have to be
        re-built per segment) and passed in as flat tensors too — None
        sentinels are passed through the variadic tail when no boundary
        fires in the chunk.
        """
        K = len(self.state_keys)
        N = self.momentum_order
        M_flat = flat_state_db_init[:K]
        # S layout: K tensors per momentum level, N levels total.
        S_total = K * N
        S_flat = flat_state_db_init[K : K + S_total]
        db_seg = flat_state_db_init[K + S_total]
        init_M_flat = flat_state_db_init[K + S_total + 1 :]
        T_seg = k_hat_seg.shape[1]
        M = dict(zip(self.state_keys, M_flat))
        if N == 1:
            S = dict(zip(self.state_keys, S_flat))
        else:
            S = tuple(
                dict(zip(self.state_keys, S_flat[lvl * K : (lvl + 1) * K]))
                for lvl in range(N)
            )
        init_M = (
            dict(zip(self.state_keys, init_M_flat))
            if (init_M_flat and init_M_flat[0] is not None) else None
        )
        # Cache the analytical-path constants up front so the per-token
        # branch stays branch-free. Norm params are outer-trained (fp32)
        # and shared across batch + time.
        _norm_w = self.memory_mlp.norm.weight
        _norm_b = self.memory_mlp.norm.bias
        _norm_eps = self.memory_mlp.norm.eps
        # Match the reduction contract that the analytical kernel expects.
        _reduction = "sum" if self.nmm_spectral_norm else "mean"

        y_list = []
        for t in range(T_seg):
            if db_seg is not None and bool(db_seg[:, t].any()):
                # `init_M` is guaranteed non-None at this branch by the caller —
                # if any boundary in the WHOLE chunk fires, the caller builds it.
                M, S = reset_state((M, S), db_seg[:, t], init_M)

            k_hat_t = k_hat_seg[:, t, :]
            q_hat_t = q_hat_seg[:, t, :]
            v_t = v_seg[:, t, :]
            # theta_seg layout: [B, T_seg, n_theta]. For default
            # (per_param_lr=False) we squeeze back to [B]; for per_param_lr=True
            # we split into a per-key dict.
            theta_slice = theta_seg[:, t, :]                # [B, n_theta]
            if self.per_param_lr_modulation:
                theta_t = {
                    k: theta_slice[:, i] for i, k in enumerate(self.state_keys)
                }
            else:
                theta_t = theta_slice.squeeze(-1)           # [B]
            # eta_seg layout: [B, T_seg, N]. For N=1 squeeze to [B]; for N>1
            # keep [B, N] (consumed by _step_momentum's per-level slice).
            eta_slice = eta_seg[:, t, :]                    # [B, N]
            eta_t = eta_slice.squeeze(-1) if N == 1 else eta_slice
            alpha_t = alpha_seg[:, t]

            M_prev = M

            g_t = _fused.analytical_inner_grad(
                M, k_hat_t, v_t, _norm_w, _norm_b, _norm_eps, _reduction,
            )
            if self.softclamp_max is not None:
                g_t = {key: softclamp_grad_norm(g, self.softclamp_max) for key, g in g_t.items()}
            if self.nmm_spectral_norm:
                g_tilde = {key: self._ns5_fn(g) for key, g in g_t.items()}
            else:
                g_tilde = g_t

            S = _step_momentum(S, g_tilde, theta_t, eta_t, N)
            M = _dict_add(_scale(1.0 - alpha_t, M), _S_top(S, N))

            M_for_retrieval = M_prev if self.retrieval_from_M_prev else M
            y_t = self.out_scale * _fused.batched_retrieve(
                M_for_retrieval, q_hat_t, _norm_w, _norm_b, _norm_eps,
            )
            y_list.append(y_t)

        y_seg = torch.stack(y_list, dim=1)
        # Return tensors in self.state_keys order to match the input layout.
        out_M = tuple(M[k] for k in self.state_keys)
        if N == 1:
            out_S = tuple(S[k] for k in self.state_keys)
        else:
            out_S = tuple(
                S[lvl][k]
                for lvl in range(N)
                for k in self.state_keys
            )
        return (y_seg,) + out_M + out_S

    def _forward_chunk_sequential(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries,
    ) -> tuple:
        """Training-path chunked forward: pre-project the full chunk, then run
        the per-token recurrence over the chunk.

        Per-token step() in a training loop would feed the conv a 1-token
        window (3 of 4 kernel weights dead). The pre-projection here lets
        the conv see up-to-k tokens of causal context for every output.

        Memory note: this path retains the full per-token autograd graph for
        the chunk. At long T on a memory-constrained card, prefer the
        blockwise path (`nmm_block_size >= 16`) — it engages TC via batched
        matmul and its peak transient scales with `block_size` rather than
        full T.
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm}). The cached "
                "per_sample_grad_fn's reduction is locked at __init__; "
                "rebuild the module to change spectral_norm."
            )
        # G275: int8 state is supported ONLY by the blockwise path.
        if self.int8_state:
            raise NotImplementedError(
                "nmm_state_dtype='int8' is supported only by the blockwise "
                "forward path (nmm_block_size > 1). The sequential path "
                "iterates per-token and would dequantize/requantize at every "
                "step, which is both slow and accumulates quantization "
                "noise. Set nmm_block_size >= 16 to use int8 state."
            )

        B, T, _ = x_chunk.shape
        # Unpack state including the rolling conv buffer (item 6). Reset
        # the buffer per-batch where chunk position 0 hits a doc boundary —
        # otherwise the previous document's tail would leak across the
        # boundary via the conv context.
        M_dict, S_state, conv_buf = state_in
        conv_buf = self._reset_conv_buf_at_chunk_start(conv_buf, doc_boundaries)

        # Full-chunk projection — conv sees up-to-k tokens per output via
        # the buffer of the previous chunk's tail (or zeros at fresh init).
        q_pw, k_pw, v_pw, new_conv_buf = self._project_qkv_with_buf(
            x_chunk, conv_buf,
        )
        k_hat_chunk = F.normalize(F.silu(k_pw), dim=-1)
        q_hat_chunk = F.normalize(F.silu(q_pw), dim=-1)
        v_chunk = F.silu(v_pw)
        # Keep the LAST dim of theta/eta (no squeeze): shape is
        # [B, T, n_theta] / [B, T, momentum_order]. _run_inner_loop reads
        # the right slice per token based on `per_param_lr_modulation`
        # / `momentum_order`. At their defaults (n_theta=1, order=1) this
        # is [B, T, 1] which is one more dim than the old behavior, but
        # the inner loop normalizes back to [B] before scaling — no
        # numerical drift, only an extra unsqueeze/squeeze pair.
        theta_chunk = torch.sigmoid(self.W_theta(x_chunk))   # [B, T, n_theta]
        eta_chunk = torch.sigmoid(self.W_eta(x_chunk))       # [B, T, N]
        alpha_chunk = torch.sigmoid(self.W_alpha(x_chunk)).squeeze(-1)

        # When state_dtype is bf16 the per-step buffers (M, S, g_t, etc.)
        # are bf16, but the Q/K/V projections still produce whatever
        # dtype x_chunk is (fp32 if no autocast, bf16 under autocast).
        # Inside `per_sample_grad_fn` the params dict overrides
        # memory_mlp's bf16 weights, so the inner forward expects bf16
        # inputs — without these casts, F.linear errors with "expected
        # Float but found BFloat16". Casting once at the chunk boundary
        # is cheaper than per-step casts.
        if self.state_dtype != x_chunk.dtype:
            k_hat_chunk = k_hat_chunk.to(self.state_dtype)
            q_hat_chunk = q_hat_chunk.to(self.state_dtype)
            v_chunk = v_chunk.to(self.state_dtype)
            theta_chunk = theta_chunk.to(self.state_dtype)
            eta_chunk = eta_chunk.to(self.state_dtype)
            alpha_chunk = alpha_chunk.to(self.state_dtype)

        # G269 lookahead-value: target for token t becomes v_{t+1}. Last
        # token's θ is zeroed (no v_T+1 within chunk) — the M update at
        # that step is just (1-α)·M decay.
        if self.lookahead_value:
            v_chunk, theta_chunk = _apply_lookahead_v(v_chunk, theta_chunk)

        # M_dict, S_state were unpacked above (along with conv_buf).

        # init_M is built only when a doc boundary fires anywhere in the
        # chunk (zero-cost on the common no-boundary chunk).
        any_boundary = (
            doc_boundaries is not None and bool(doc_boundaries.any())
        )
        K = len(self.state_keys)
        N = self.momentum_order
        if any_boundary:
            init_M = self._build_init_M(B, x_chunk.device)
            init_M_flat = tuple(init_M[k] for k in self.state_keys)
        else:
            init_M_flat = (None,) * K

        # Pull state tensors out in canonical (self.state_keys) order so
        # the layout is consistent for low-rank (6 keys) and full-rank (3).
        # S layout: K tensors for level-0, then K for level-1, ... K for
        # level-(N-1). At N=1 this is just K S tensors (current default).
        M_flat = tuple(M_dict[k] for k in self.state_keys)
        if N == 1:
            S_flat = tuple(S_state[k] for k in self.state_keys)
        else:
            # S_state is a tuple of N dicts; flatten level-major.
            S_flat = tuple(
                S_state[lvl][k]
                for lvl in range(N)
                for k in self.state_keys
            )

        outs = self._run_inner_loop(
            k_hat_chunk, q_hat_chunk, v_chunk,
            theta_chunk, eta_chunk, alpha_chunk,
            *M_flat, *S_flat,
            doc_boundaries,
            *init_M_flat,
        )

        # outs layout: (y_chunk, *M_flat_new[K], *S_flat_new[K*N]).
        y_chunk = outs[0]
        M_flat = tuple(outs[1 : 1 + K])
        S_flat = tuple(outs[1 + K : 1 + K + K * N])

        if N == 1:
            S_out = dict(zip(self.state_keys, S_flat))
        else:
            S_out = tuple(
                dict(zip(self.state_keys, S_flat[lvl * K : (lvl + 1) * K]))
                for lvl in range(N)
            )
        new_state = (
            dict(zip(self.state_keys, M_flat)),
            S_out,
            new_conv_buf,  # item 6: thread the rolling conv buffer forward
        )
        return y_chunk, new_state

    def _forward_chunk_blockwise(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries,
    ) -> tuple:
        """Chunk-as-update path (G266 — lucidrains-style aggregation).

        Within each block of `self.block_size` tokens, ONE memory update is
        produced from the aggregate per-block gradient (sum of per-token
        gradients). All tokens in a block share the block-start M for both
        the surprise loss and retrieval. Between blocks, the state threads
        sequentially.

        At `block_size = 1` this reduces to the paper-strict per-token
        recurrence (equivalent to `_forward_chunk_sequential` modulo
        operation-order round-off; tests lock the numerical match).

        At `block_size > 1` the per-block forward through MemoryMLP is a
        single batched matmul over the block's tokens (`einsum("bhd,btd->bth", ...)`
        and friends) — TC engages at `block_size ≥ 16`.

        The current implementation:
        - Computes per-block gradient via `analytical_chunk_grad` (TC-engaged).
        - Averages theta/eta/alpha across the block (one scalar per block
          instead of per-token).
        - Sequential threading between blocks via a Python loop (16 blocks
          per T=1024 chunk at block_size=64 is negligible loop overhead vs
          the per-block matmul time).

        Doc-boundary handling: the chunk is pre-split at boundary positions
        (union across batch) into sub-chunks with no internal boundary.
        `reset_state` fires once at each sub-chunk start, masked by the
        per-batch boundary indicator at that exact position. Tokens before
        a boundary keep their pre-boundary M; tokens at-or-after the
        boundary run against init_M for batches that hit the boundary at
        that position. This restores per-token-correct reset semantics
        without per-token loop overhead. The trade-off: boundary-containing
        chunks produce smaller block matmuls at the sub-chunk seams, which
        gives up some TC engagement near the seam.
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm})."
            )

        B, T, _ = x_chunk.shape
        # T may not be a multiple of block_size — the trailing block will
        # have whatever leftover tokens remain. The persistent-mem prefix
        # in TitansMAGBlock prepends `nmm_n_persistent` tokens to each
        # chunk, so the NMM-effective T = chunk_size + n_persistent, which
        # rarely divides evenly. Handle the trailing block specially below.

        # Unpack state including the rolling conv buffer (item 6); reset
        # per-batch at chunk pos 0 if a doc boundary fires there.
        M, S, conv_buf = state_in
        conv_buf = self._reset_conv_buf_at_chunk_start(conv_buf, doc_boundaries)

        # Project k, q, v + adaptive params once across the full chunk.
        # Keep the trailing dim of theta/eta (no squeeze) so per_param_lr
        # (n_theta=K) and momentum_order>1 (N>1) paths can slice in.
        q_pw, k_pw, v_pw, new_conv_buf = self._project_qkv_with_buf(
            x_chunk, conv_buf,
        )
        k_hat_chunk = F.normalize(F.silu(k_pw), dim=-1)
        q_hat_chunk = F.normalize(F.silu(q_pw), dim=-1)
        v_chunk = F.silu(v_pw)
        theta_chunk = torch.sigmoid(self.W_theta(x_chunk))   # [B, T, n_theta]
        eta_chunk = torch.sigmoid(self.W_eta(x_chunk))       # [B, T, N]
        alpha_chunk = torch.sigmoid(self.W_alpha(x_chunk)).squeeze(-1)

        if self.state_dtype != x_chunk.dtype:
            k_hat_chunk = k_hat_chunk.to(self.state_dtype)
            q_hat_chunk = q_hat_chunk.to(self.state_dtype)
            v_chunk = v_chunk.to(self.state_dtype)
            theta_chunk = theta_chunk.to(self.state_dtype)
            eta_chunk = eta_chunk.to(self.state_dtype)
            alpha_chunk = alpha_chunk.to(self.state_dtype)

        # G269: shift v left by 1, zero θ at the last position. With theta
        # carrying the last n_theta dim, the helper still works — it
        # slices on the T (dim 1) only.
        if self.lookahead_value:
            v_chunk, theta_chunk = _apply_lookahead_v(v_chunk, theta_chunk)

        norm_w = self.memory_mlp.norm.weight
        norm_b = self.memory_mlp.norm.bias
        norm_eps = self.memory_mlp.norm.eps
        reduction = "sum" if self.nmm_spectral_norm else "mean"

        # M, S, conv_buf were unpacked at the top of this function.
        N = self.momentum_order
        # G275 int8 state: dequantize on entry. Within the block the math
        # runs in the standard state_dtype (fp32 here); at chunk end we
        # requantize before returning.
        was_int8 = self.int8_state and (
            _is_int8_dict(M)
            or (isinstance(S, dict) and _is_int8_dict(S))
            or (isinstance(S, (list, tuple)) and any(_is_int8_dict(s) for s in S))
        )
        if was_int8:
            M = _dequant_dict(M, self.state_keys)
            if isinstance(S, (list, tuple)):
                S = tuple(_dequant_dict(s, self.state_keys) for s in S)
            else:
                S = _dequant_dict(S, self.state_keys)
        # Sub-chunk pre-split at doc boundaries. Each sub-chunk has no
        # internal boundary by construction; reset_state fires once at
        # sub-chunk start (per-batch via the boundary mask at that exact
        # position). This restores per-token-correct semantics — tokens
        # BEFORE a boundary keep their pre-boundary M, tokens AT-OR-AFTER
        # the boundary run against init_M for batches that hit the
        # boundary.
        #
        # Common case (no boundary in chunk): exactly one sub-chunk, zero
        # overhead. Boundary-containing chunks pay smaller block matmuls
        # at the sub-chunk seams, which costs some TC engagement; the
        # alternative was the retroactive-reset bug.
        any_boundary = (
            doc_boundaries is not None and bool(doc_boundaries.any())
        )
        init_M = self._build_init_M(B, x_chunk.device) if any_boundary else None

        if any_boundary:
            bdry_any = doc_boundaries.any(dim=0)             # [T] bool
            bdry_positions = bdry_any.nonzero(as_tuple=True)[0].tolist()
        else:
            bdry_positions = []
        # sub_starts[0] is always 0; subsequent entries are boundary
        # positions in order. A boundary at position 0 produces a (0, 0)
        # range that's skipped, with the reset firing at sub_idx == 1.
        sub_starts = [0] + bdry_positions
        sub_ends = sub_starts[1:] + [T]

        y_blocks = []
        for sub_idx, (sub_s, sub_e) in enumerate(zip(sub_starts, sub_ends)):
            if sub_e <= sub_s:
                continue  # boundary at position 0 (or duplicate); skip empty range.

            # Reset at sub-chunk start. `sub_idx == 0` is the chunk's
            # leading sub-chunk (starts at position 0); no prior state to
            # reset against, so we never reset here. For sub_idx > 0 the
            # sub-chunk starts at a boundary position; reset_state masks
            # by `doc_boundaries[:, sub_s]` so only the batches that have
            # the boundary at exactly this position get reset.
            if sub_idx > 0:
                M, S = reset_state((M, S), doc_boundaries[:, sub_s], init_M)

            # Inner loop: block_size-aligned partition of [sub_s, sub_e).
            # No boundary checks inside — the sub-chunk has none by
            # construction.
            for s in range(sub_s, sub_e, self.block_size):
                e = min(s + self.block_size, sub_e)

                # G268 truncated BPTT — detach at every block boundary
                # across the whole chunk. Redundant right after a reset
                # (init_M is already detached) but harmless.
                if self.detach_state_between_blocks and s > 0:
                    M = {k: v.detach() for k, v in M.items()}
                    if isinstance(S, (list, tuple)):
                        S = tuple(
                            {k: v.detach() for k, v in S_lvl.items()} for S_lvl in S
                        )
                    else:
                        S = {k: v.detach() for k, v in S.items()}

                k_blk = k_hat_chunk[:, s:e]                          # [B, block, D]
                q_blk = q_hat_chunk[:, s:e]
                v_blk = v_chunk[:, s:e]
                # Block-aggregate scalars: mean over the block's T dim.
                # theta_chunk: [B, T, n_theta] -> [B, n_theta] after T-mean.
                # eta_chunk:   [B, T, N]       -> [B, N]      after T-mean.
                # Squeeze when the trailing dim is 1 (default scalar case).
                theta_blk_full = theta_chunk[:, s:e].mean(dim=1)     # [B, n_theta]
                eta_blk_full   = eta_chunk[:, s:e].mean(dim=1)       # [B, N]
                if self.per_param_lr_modulation:
                    # Per-key dict; consumed by `_scale_per_key` downstream.
                    theta_blk = {
                        k: theta_blk_full[:, i]
                        for i, k in enumerate(self.state_keys)
                    }
                else:
                    theta_blk = theta_blk_full.squeeze(-1)           # [B]
                eta_blk = eta_blk_full if N > 1 else eta_blk_full.squeeze(-1)
                alpha_blk = alpha_chunk[:, s:e].mean(dim=-1)

                M_block_start = M
                # Per-token θ (block-local). Shape [B, block, n_theta]; for
                # default n_theta=1 the unsqueezed form is fine when fed
                # into `_scale_per_key` and per-token NS5 path's einsum
                # (which operates on a 2D scalar tensor — squeeze when
                # n_theta=1).
                theta_chunk_slice = theta_chunk[:, s:e]              # [B, block, n_theta]
                if self.per_param_lr_modulation:
                    theta_per_token = {
                        k: theta_chunk_slice[..., i]
                        for i, k in enumerate(self.state_keys)
                    }
                else:
                    theta_per_token = theta_chunk_slice.squeeze(-1)   # [B, block]

                if self.per_token_ns5:
                    # G267 paper-faithful: per-token NS5 then per-token θ
                    # weighting then sum. Matches paper Eq 16's
                    # `Σ_t θ_t · NS5(∇_t)`.
                    per_token_grad = _fused.analytical_per_token_grad(
                        M, k_blk, v_blk, norm_w, norm_b, norm_eps, reduction,
                    )
                    if self.softclamp_max is not None:
                        per_token_grad = {
                            key: softclamp_grad_norm(g, self.softclamp_max)
                            for key, g in per_token_grad.items()
                        }
                    if self.nmm_spectral_norm:
                        per_token_tilde = {
                            key: self._ns5_fn(g)
                            for key, g in per_token_grad.items()
                        }
                    else:
                        per_token_tilde = per_token_grad
                    # Σ_t θ_t · NS5(∇_t) — per-token θ weighting OUTSIDE NS5.
                    # When per_param_lr is on, theta_per_token is a per-key
                    # dict; otherwise a shared [B, block] tensor.
                    if isinstance(theta_per_token, dict):
                        chunk_theta_grad = {
                            key: torch.einsum("bt,bthd->bhd", theta_per_token[key], g)
                            for key, g in per_token_tilde.items()
                        }
                    else:
                        chunk_theta_grad = {
                            key: torch.einsum("bt,bthd->bhd", theta_per_token, g)
                            for key, g in per_token_tilde.items()
                        }
                    # S/M update via _step_momentum (handles N>=1).
                    # Per-token NS5 path treats theta as "already-applied"
                    # in chunk_theta_grad; pass a unit-θ to _step_momentum
                    # so the surprise term is chunk_theta_grad as-is.
                    # Concretely: re-do the recurrence by hand here since
                    # we already have θ-weighted grads. For N=1:
                    #   S = η·S - chunk_theta_grad
                    # For N>1: feed chunk_theta_grad as the level-0 surprise.
                    if N == 1:
                        S = _dict_sub(_scale(eta_blk, S), chunk_theta_grad)
                    else:
                        eta_levels = [eta_blk[..., k] for k in range(N)]
                        S_new = []
                        S_new.append(_dict_sub(_scale(eta_levels[0], S[0]), chunk_theta_grad))
                        for k in range(1, N):
                            S_new.append(_dict_add(_scale(eta_levels[k], S[k]), S_new[k-1]))
                        S = tuple(S_new)
                    M = _dict_add(_scale(1.0 - alpha_blk, M_block_start), _S_top(S, N))
                else:
                    # Cheap blockwise (no per-token θ refinement): NS5 on
                    # the plain aggregate, mean θ scaling. At block_size=1
                    # this is bit-equivalent to the sequential recurrence.
                    grad_blk = _fused.analytical_chunk_grad(
                        M, k_blk, v_blk, norm_w, norm_b, norm_eps, reduction,
                    )
                    if self.softclamp_max is not None:
                        grad_blk = {
                            key: softclamp_grad_norm(g, self.softclamp_max)
                            for key, g in grad_blk.items()
                        }
                    if self.nmm_spectral_norm:
                        grad_tilde = {
                            key: self._ns5_fn(g) for key, g in grad_blk.items()
                        }
                    else:
                        grad_tilde = grad_blk
                    S = _step_momentum(S, grad_tilde, theta_blk, eta_blk, N)
                    M = _dict_add(_scale(1.0 - alpha_blk, M_block_start), _S_top(S, N))

                # Retrieval per-token within the block (all tokens see the same M).
                M_for_retrieval = M_block_start if self.retrieval_from_M_prev else M
                y_blk = _fused.batched_retrieve_chunk(
                    M_for_retrieval, q_blk, norm_w, norm_b, norm_eps,
                )                                                    # [B, block, D]
                y_blk = self.out_scale * y_blk
                y_blocks.append(y_blk)

        y_chunk = torch.cat(y_blocks, dim=1)
        # G275: requantize before returning so the caller-visible state
        # is in int8 form again.
        if was_int8:
            M = _quant_dict(M, self.state_keys)
            if isinstance(S, (list, tuple)):
                S = tuple(_quant_dict(s, self.state_keys) for s in S)
            else:
                S = _quant_dict(S, self.state_keys)
        return y_chunk, (M, S, new_conv_buf)

    def forward_chunk(self, x_chunk, state_in, doc_boundaries):
        """Dispatch between blockwise and sequential paths.

        Priority:
          1. **Blockwise** when `block_size > 1` — chunk-as-update path
             (G266). TC-engaged, sequential between blocks. Approximate at
             block_size > 1; bit-equivalent to sequential at block_size=1
             (but using analytical-grad ops instead of `torch.func.grad`).
          2. **Sequential** otherwise — paper-strict per-token recurrence.
        """
        if self.block_size > 1:
            return self._forward_chunk_blockwise(x_chunk, state_in, doc_boundaries)
        return self._forward_chunk_sequential(x_chunk, state_in, doc_boundaries)


class MultiHeadNMM(nn.Module):
    """N parallel NeuralMemoryModules, each on `head_dim = n_embd // n_heads`.

    Lucidrains enhancement (NOT in the paper proper) — gated by config field
    `nmm_n_heads`. Default `nmm_n_heads=1` uses the single-head
    `NeuralMemoryModule` directly (no wrapper); `nmm_n_heads > 1` instantiates
    this wrapper. Exposes the same API as `NeuralMemoryModule` so the block
    treats it as a drop-in replacement.

    State structure: a list of per-head `(M, S)` tuples. `detach_states` and
    `compute_nmm_norm` are now recursive to handle the nested structure
    (the per-layer entry in `nmm_states` becomes a list-of-states instead
    of a single `(M, S)` tuple).
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        expansion: int = 4,
        kernel_size: int = 4,
        spectral_norm: bool = True,
        finetune_mode: bool = True,
        retrieval_from_M_prev: bool = True,
        state_dtype: str = "fp32",
        low_rank=None,
        softclamp_max=None,
        block_size: int = 1,
        per_token_ns5: bool = False,
        detach_state_between_blocks: bool = False,
        lookahead_value: bool = False,
        per_param_lr_modulation: bool = False,
        momentum_order: int = 1,
        ns5_steps: int = 5,
        use_gram_ns5: bool = False,
        use_cans: bool = False,
        per_head_learned_params: bool = True,
    ):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1 (got {n_heads})")
        if n_embd % n_heads != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads}); "
                f"head_dim would be {n_embd // n_heads}, which gives "
                f"{n_heads * (n_embd // n_heads)}, not {n_embd}."
            )
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.nmm_spectral_norm = spectral_norm
        self.finetune_mode = finetune_mode
        self.retrieval_from_M_prev = retrieval_from_M_prev

        # The conv-kernel `step_with_conv` semantic carries through — each
        # head's NMM has its own conv buffer of last (k-1) head-dim tokens.
        # state_dtype propagates per-head.
        self.per_head_learned_params = bool(per_head_learned_params)
        self.heads = nn.ModuleList([
            NeuralMemoryModule(
                n_embd=self.head_dim,
                expansion=expansion,
                kernel_size=kernel_size,
                spectral_norm=spectral_norm,
                finetune_mode=finetune_mode,
                retrieval_from_M_prev=retrieval_from_M_prev,
                state_dtype=state_dtype,
                low_rank=low_rank,
                softclamp_max=softclamp_max,
                block_size=block_size,
                per_token_ns5=per_token_ns5,
                detach_state_between_blocks=detach_state_between_blocks,
                lookahead_value=lookahead_value,
                per_param_lr_modulation=per_param_lr_modulation,
                momentum_order=momentum_order,
                ns5_steps=ns5_steps,
                use_gram_ns5=use_gram_ns5,
                use_cans=use_cans,
            )
            for _ in range(n_heads)
        ])
        # G271: per_head_learned_params=False — point every head's MemoryMLP
        # at the SAME nn.Module instance so the recurrent weight inits are
        # shared. Per-head LayerNorm / out_scale / Q/K/V projections /
        # update-param Linears all remain head-private. State (M, S) per-head
        # is independent at runtime (each head threads its own state), but
        # both heads' `init_state()` calls now resolve to the same shared
        # weight tensors.
        #
        # Implementation detail: we swap `head.memory_mlp = head_0.memory_mlp`
        # in place AFTER construction so each head's per-sample-grad cache
        # (`per_sample_grad_fn`) was built against ITS OWN MemoryMLP — but
        # `_make_grad_fn` closes over `memory_mlp` via `functional_call`,
        # which reads the module at call time (params override the module's
        # buffers, not its identity). So sharing the module post-hoc is safe.
        # ALSO update `state_keys` resolution to follow the shared module.
        if (not self.per_head_learned_params) and n_heads > 1:
            shared_mlp = self.heads[0].memory_mlp
            for h in self.heads[1:]:
                h.memory_mlp = shared_mlp
                # Rebuild the per-sample grad fn / retrieve fn now that
                # memory_mlp is the shared instance. `_make_grad_fn` reads
                # `memory_mlp` from the closure; rebuilding ensures all
                # heads point at the same callable target.
                h.per_sample_grad_fn = _make_grad_fn(
                    shared_mlp, spectral_norm=spectral_norm
                )
                # _batched_retrieve also closes over memory_mlp via
                # `functional_call(self.memory_mlp, ...)`. Since we're
                # mutating `self.memory_mlp` to point at the shared instance
                # and `_retrieve_one_sample` reads `self.memory_mlp` at call
                # time (not at vmap-binding time), no rebuild is strictly
                # required — but rebuilding here keeps the per-head call
                # caches symmetrical, simpler to reason about.
                def _retrieve_one_sample(m_dict, q, _mlp=shared_mlp):
                    return functional_call(
                        _mlp, m_dict, q.unsqueeze(0)
                    ).squeeze(0)
                h._batched_retrieve = vmap(_retrieve_one_sample, in_dims=(0, 0))

    # --- Properties / helpers ----------------------------------------------

    @property
    def memory_mlp(self):
        """For backward-compat with code paths that read `nmm.memory_mlp.W*`
        for shape / dtype probes (e.g., `_apply_gpt2_init`'s
        NMM-skip-by-id). Returns the FIRST head's MemoryMLP — sufficient
        for shape/dtype-only consumers."""
        return self.heads[0].memory_mlp

    # --- Same-API methods as NeuralMemoryModule ----------------------------

    def init_state(self, B: int, device) -> list:
        """Per-head init states. Returns a list[n_heads] of `(M, S)` tuples."""
        return [h.init_state(B, device) for h in self.heads]

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape last dim into (n_heads, head_dim). Works for [B, T, d] or
        [B, d] inputs (the step path)."""
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    def _merge_heads(self, head_outputs: list) -> torch.Tensor:
        """Concatenate per-head outputs back into a d_model tensor."""
        return torch.cat(head_outputs, dim=-1)

    def forward_chunk(self, x_chunk, state_in, doc_boundaries):
        """Per-head dispatch of forward_chunk. doc_boundaries is shared
        across heads (a token-level event is the same for every head)."""
        x_split = self._split_heads(x_chunk)  # [B, T, n_heads, head_dim]
        outputs = []
        new_states = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()  # [B, T, head_dim]
            y_h, state_h = head.forward_chunk(x_h, state_in[i], doc_boundaries)
            outputs.append(y_h)
            new_states.append(state_h)
        return self._merge_heads(outputs), new_states

    def step_with_conv(self, x_t, state):
        """Per-head step. `state` is a `list[n_heads]` of per-head
        `(M, S, conv_buf)` tuples (item 6 — conv_buf is in state now).
        x_t is [B, d_model]; split into per-head [B, head_dim] slices,
        run each head's step_with_conv, concatenate outputs.

        Returns (y [B, d_model], new_state) where new_state has the same
        nested shape as state with per-head rolled conv buffers."""
        x_split = self._split_heads(x_t)  # [B, n_heads, head_dim]
        outputs = []
        new_states = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()
            y_h, s_h = head.step_with_conv(x_h, state[i])
            outputs.append(y_h)
            new_states.append(s_h)
        return self._merge_heads(outputs), new_states

