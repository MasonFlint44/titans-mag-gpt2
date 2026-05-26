"""Analytical-gradient fused path for the NMM inner loop.

The reference path in `nmm.py` uses `vmap(grad(inner_loss))` which pays
~10 ms of Python / autograd-graph-construction overhead per token. At
T=1024 over 12 NMM layers that's ~12 000 calls per chunk — the
dominant cost in TITANS-MAG training (G264 — see SPEC §5.7).

This module computes the per-token gradient of the inner MSE loss
w.r.t. each `MemoryMLP` recurrent weight in closed form. Every
operation is plain batched PyTorch, so outer autograd handles the
backward to `k_hat`, `v`, `M_init`, etc. automatically. The numerical
result is identical to the reference within fp32 round-off; this is
locked by tests in `tests/unit/test_nmm_fused.py`.

Shapes (full-rank, per-sample batched [B, ...]):

    Inputs (one sample):
        k_hat, v      : [d]
        W1, W_gate    : [h, d]                       (rows = output)
        W2            : [d, h]
        norm.weight,
        norm.bias     : [d]                          (LayerNorm gamma/beta)

    Forward intermediates (saved for backward):
        pre1   = W1 @ k_hat                          # [h]
        preg   = W_gate @ k_hat                      # [h]
        silu1  = silu(pre1)                          # [h]
        sigg   = sigmoid(preg)                       # [h]
        a      = silu1 * sigg                        # [h]
        y      = W2 @ a                              # [d]
        y_hat  = (y - mean(y)) * inv_std             # [d]   normalized
        y_norm = norm.weight * y_hat + norm.bias     # [d]
        out    = y_norm + k_hat                      # [d]   residual
        L      = ||out - v||^2  (sum or mean)        # scalar

    Backward (returns d_W1, d_W_gate, d_W2):
        d_out      = 2 * (out - v)               (sum reduction)
                   = 2 * (out - v) / d           (mean reduction)
        d_y_norm   = d_out                       (residual = identity grad)
        # LayerNorm backward (gamma/beta NOT trained inside inner loss):
        d_y_hat    = d_y_norm * norm.weight
        d_y        = (1/d) * inv_std *
                     (d * d_y_hat - sum(d_y_hat) - y_hat * sum(d_y_hat * y_hat))
        # Linear back through W2:
        d_W2       = outer(d_y, a)                            # [d, h]
        d_a        = W2.T @ d_y                               # [h]
        # SwiGLU split:
        d_silu1    = d_a * sigg
        d_sigg     = d_a * silu1
        d_pre1     = d_silu1 * silu_grad(pre1)
        d_preg     = d_sigg  * sigmoid_grad(preg)
        # Linear back through W1, W_gate:
        d_W1       = outer(d_pre1, k_hat)                     # [h, d]
        d_W_gate   = outer(d_preg, k_hat)                     # [h, d]

Low-rank (`low_rank=r`) chains an extra (r,d)-(h,r) factor through each
of W1, W_gate, W2; the analytical chain rule is the obvious extension
documented inline below.

Reduction: matches the reference. `spectral_norm=True` (default) uses
'sum'; `False` uses 'mean'. NS5 normalises the spectral norm
post-grad, so for the spectral_norm=True path the `d` factor cancels
downstream — the choice is invisible. But for the no-NS5 path it
matters that we use 'mean' to keep gradient magnitude independent of
`d`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation derivatives.
# Pure functions — no module state. silu / sigmoid grads are tiny ops and
# composing through them analytically is much cheaper than letting autograd
# trace the per-token versions.
# ---------------------------------------------------------------------------

def _silu_grad(x: torch.Tensor, sig: torch.Tensor | None = None) -> torch.Tensor:
    """d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Accepts a precomputed `sig = sigmoid(x)` if the caller already has it.
    """
    if sig is None:
        sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def _sigmoid_grad(sig: torch.Tensor) -> torch.Tensor:
    """d/dx sigmoid(x) given sig = sigmoid(x) (already computed)."""
    return sig * (1.0 - sig)


# ---------------------------------------------------------------------------
# Forward + analytical-gradient kernel for one token.
# `analytical_inner_grad` mirrors `vmap(grad(inner_loss))` in nmm.py for a
# SINGLE timestep: given current M dict, k_hat [B,d], v [B,d], returns a
# dict of per-sample gradients with the same keys & shapes as M.
# ---------------------------------------------------------------------------


def _layer_norm_forward(y: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """LayerNorm with `normalized_shape = (d,)`. Operates on last dim of y.

    Returns (out, y_hat, inv_std):
        out:     [B, d]   normalized + affine output (same dtype as y)
        y_hat:   [B, d]   pre-affine normalized output
        inv_std: [B, 1]   1/sqrt(var + eps), needed by backward

    Runs the normalisation in fp32 even when y is bf16 — matches
    `MemoryMLP.forward`, where `self.norm(y.float())` is used to keep the
    LayerNorm path numerically clean under bf16 state.
    """
    orig_dtype = y.dtype
    y32 = y.float()
    mu = y32.mean(dim=-1, keepdim=True)
    var = y32.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = (var + eps).rsqrt()
    y_hat = (y32 - mu) * inv_std
    out = y_hat * gamma.float() + beta.float()
    return out.to(orig_dtype), y_hat, inv_std


def _layer_norm_backward_x(
    d_out: torch.Tensor,
    y_hat: torch.Tensor,
    inv_std: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Backward of LayerNorm w.r.t. input only (gamma/beta are outer-trained
    and not part of the per-token inner gradient).

    d_x_i = (1/d) * inv_std * (d * d_y_hat_i - sum(d_y_hat) - y_hat_i * sum(d_y_hat * y_hat))
    where d_y_hat = d_out * gamma.

    All computation in fp32, output cast back to d_out's original dtype.
    """
    orig_dtype = d_out.dtype
    d_out32 = d_out.float()
    gamma32 = gamma.float()
    d_y_hat = d_out32 * gamma32
    d_local = d_y_hat.shape[-1]
    sum_dy = d_y_hat.sum(dim=-1, keepdim=True)
    sum_dy_yhat = (d_y_hat * y_hat).sum(dim=-1, keepdim=True)
    d_x = (1.0 / d_local) * inv_std * (
        d_local * d_y_hat - sum_dy - y_hat * sum_dy_yhat
    )
    return d_x.to(orig_dtype)


def _memory_mlp_forward_fullrank(
    k_hat: torch.Tensor,           # [B, d]
    W1: torch.Tensor,              # [B, h, d]
    W_gate: torch.Tensor,          # [B, h, d]
    W2: torch.Tensor,              # [B, d, h]
    norm_weight: torch.Tensor,     # [d]   (shared, outer-trained)
    norm_bias: torch.Tensor,       # [d]
    norm_eps: float,
):
    """Returns (out, intermediates_dict) where `out` matches `MemoryMLP(k_hat)`
    bit-for-bit (within rounding) and `intermediates_dict` carries everything
    the backward needs."""
    # pre1[b, i] = sum_j W1[b, i, j] * k_hat[b, j]
    pre1 = torch.einsum("bij,bj->bi", W1, k_hat)
    preg = torch.einsum("bij,bj->bi", W_gate, k_hat)
    silu1 = F.silu(pre1)
    sigg = torch.sigmoid(preg)
    a = silu1 * sigg
    y = torch.einsum("bij,bj->bi", W2, a)
    y_norm, y_hat, inv_std = _layer_norm_forward(y, norm_weight, norm_bias, norm_eps)
    out = y_norm + k_hat
    return out, {
        "pre1": pre1, "preg": preg,
        "silu1": silu1, "sigg": sigg, "a": a,
        "y": y, "y_hat": y_hat, "inv_std": inv_std,
    }


def _memory_mlp_backward_fullrank(
    d_out: torch.Tensor,       # [B, d]
    k_hat: torch.Tensor,       # [B, d]
    W2: torch.Tensor,          # [B, d, h]
    norm_weight: torch.Tensor, # [d]
    interm: dict,
):
    """Returns d_W1, d_W_gate, d_W2 — each [B, h_or_d, h_or_d] matching
    forward-time per-sample weight shapes.

    Derivative chain follows the docstring at module top.
    """
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out                                           # residual passes through
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    # d_W2 = outer(d_y, a)
    d_W2 = d_y.unsqueeze(-1) * a.unsqueeze(-2)                 # [B, d, h]
    # d_a = W2.T @ d_y
    d_a = torch.einsum("bij,bi->bj", W2, d_y)                  # [B, h]
    d_silu1 = d_a * sigg
    d_sigg = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)
    d_preg = d_sigg * _sigmoid_grad(sigg)
    d_W1 = d_pre1.unsqueeze(-1) * k_hat.unsqueeze(-2)          # [B, h, d]
    d_W_gate = d_preg.unsqueeze(-1) * k_hat.unsqueeze(-2)
    return d_W1, d_W_gate, d_W2


def _memory_mlp_forward_lowrank(
    k_hat: torch.Tensor,
    W1_a: torch.Tensor, W1_b: torch.Tensor,
    Wg_a: torch.Tensor, Wg_b: torch.Tensor,
    W2_a: torch.Tensor, W2_b: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
):
    """Low-rank factored forward. Shapes (per sample):
        W1_a: [r, d]   W1_b: [h, r]   (so W1 = W1_b @ W1_a is [h, d])
        Wg_a: [r, d]   Wg_b: [h, r]
        W2_a: [r, h]   W2_b: [d, r]
    """
    t1 = torch.einsum("brd,bd->br", W1_a, k_hat)
    pre1 = torch.einsum("bhr,br->bh", W1_b, t1)
    tg = torch.einsum("brd,bd->br", Wg_a, k_hat)
    preg = torch.einsum("bhr,br->bh", Wg_b, tg)
    silu1 = F.silu(pre1); sigg = torch.sigmoid(preg)
    a = silu1 * sigg
    t2 = torch.einsum("brh,bh->br", W2_a, a)
    y = torch.einsum("bdr,br->bd", W2_b, t2)
    y_norm, y_hat, inv_std = _layer_norm_forward(y, norm_weight, norm_bias, norm_eps)
    out = y_norm + k_hat
    return out, {
        "pre1": pre1, "preg": preg, "silu1": silu1, "sigg": sigg, "a": a,
        "t1": t1, "tg": tg, "t2": t2,
        "y_hat": y_hat, "inv_std": inv_std,
    }


def _memory_mlp_backward_lowrank(
    d_out: torch.Tensor,
    k_hat: torch.Tensor,
    W1_b: torch.Tensor, Wg_b: torch.Tensor,
    W2_a: torch.Tensor, W2_b: torch.Tensor,
    norm_weight: torch.Tensor,
    interm: dict,
):
    """Returns gradient dict in canonical sorted key order:
        W1_a.weight, W1_b.weight, W2_a.weight, W2_b.weight,
        W_gate_a.weight, W_gate_b.weight
    (alphabetical — matches `tuple(sorted(state_keys))`.)
    """
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    t1 = interm["t1"]; tg = interm["tg"]; t2 = interm["t2"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    # y = W2_b @ t2, t2 = W2_a @ a
    d_W2_b = d_y.unsqueeze(-1) * t2.unsqueeze(-2)               # [B, d, r]
    d_t2 = torch.einsum("bdr,bd->br", W2_b, d_y)                 # [B, r]
    d_W2_a = d_t2.unsqueeze(-1) * a.unsqueeze(-2)                # [B, r, h]
    d_a = torch.einsum("brh,br->bh", W2_a, d_t2)                 # [B, h]
    d_silu1 = d_a * sigg
    d_sigg = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)
    d_preg = d_sigg * _sigmoid_grad(sigg)
    # pre1 = W1_b @ t1, t1 = W1_a @ k_hat
    d_W1_b = d_pre1.unsqueeze(-1) * t1.unsqueeze(-2)             # [B, h, r]
    d_t1 = torch.einsum("bhr,bh->br", W1_b, d_pre1)              # [B, r]
    d_W1_a = d_t1.unsqueeze(-1) * k_hat.unsqueeze(-2)            # [B, r, d]
    d_Wg_b = d_preg.unsqueeze(-1) * tg.unsqueeze(-2)
    d_tg = torch.einsum("bhr,bh->br", Wg_b, d_preg)
    d_Wg_a = d_tg.unsqueeze(-1) * k_hat.unsqueeze(-2)
    return {
        "W1_a.weight": d_W1_a,
        "W1_b.weight": d_W1_b,
        "W2_a.weight": d_W2_a,
        "W2_b.weight": d_W2_b,
        "W_gate_a.weight": d_Wg_a,
        "W_gate_b.weight": d_Wg_b,
    }


def analytical_inner_grad(
    M: dict,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    reduction: str,
) -> dict:
    """Analytical per-sample gradient of the inner MSE loss w.r.t. each
    weight in `M`. Matches `vmap(grad(inner_loss))` numerically.

    M: dict — same keys as MemoryMLP's recurrent state. Full-rank has
        {'W1.weight', 'W_gate.weight', 'W2.weight'};  low-rank has
        {'W1_a.weight', 'W1_b.weight', 'W_gate_a.weight', 'W_gate_b.weight',
         'W2_a.weight', 'W2_b.weight'}.
    reduction: 'sum' (when spectral_norm=True) or 'mean' (when False).
        Sum/mean only differ by the 1/d factor on d_out — NS5 cancels the
        scale downstream so the choice is silent for the default path but
        deliberate for ablations.
    """
    # Detect full-rank vs low-rank by checking the canonical key set.
    keys = set(M.keys())
    full_rank_keys = {"W1.weight", "W_gate.weight", "W2.weight"}
    low_rank_keys = {
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    }
    is_full_rank = keys == full_rank_keys
    is_low_rank = keys == low_rank_keys
    if not (is_full_rank or is_low_rank):
        raise ValueError(
            f"analytical_inner_grad: unrecognised M key set {sorted(keys)}. "
            f"Expected full-rank {sorted(full_rank_keys)} or low-rank "
            f"{sorted(low_rank_keys)}."
        )

    if is_full_rank:
        out, interm = _memory_mlp_forward_fullrank(
            k_hat,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            norm_weight, norm_bias, norm_eps,
        )
    else:
        out, interm = _memory_mlp_forward_lowrank(
            k_hat,
            M["W1_a.weight"], M["W1_b.weight"],
            M["W_gate_a.weight"], M["W_gate_b.weight"],
            M["W2_a.weight"], M["W2_b.weight"],
            norm_weight, norm_bias, norm_eps,
        )

    # Loss = sum (or mean) of (out - v)^2. d_loss/d_out = 2*(out - v) for sum,
    # or 2*(out - v) / d for mean.
    diff = out - v
    if reduction == "sum":
        d_out = 2.0 * diff
    elif reduction == "mean":
        d_out = (2.0 / diff.shape[-1]) * diff
    else:
        raise ValueError(f"reduction must be 'sum' or 'mean' (got {reduction!r}).")

    if is_full_rank:
        d_W1, d_W_gate, d_W2 = _memory_mlp_backward_fullrank(
            d_out, k_hat, M["W2.weight"], norm_weight, interm,
        )
        return {
            "W1.weight": d_W1,
            "W_gate.weight": d_W_gate,
            "W2.weight": d_W2,
        }
    return _memory_mlp_backward_lowrank(
        d_out, k_hat,
        M["W1_b.weight"], M["W_gate_b.weight"],
        M["W2_a.weight"], M["W2_b.weight"],
        norm_weight, interm,
    )


# ---------------------------------------------------------------------------
# Chunk-aggregate analytical gradient (G266) — sums per-token gradients into
# one weight gradient per chunk via batched matmuls. The per-token math is
# unchanged from `analytical_inner_grad`; the difference is that all T tokens
# share M_block_start and contribute additively to one update gradient.
#
# Used by the blockwise NMM path (`_forward_chunk_blockwise`) when
# `nmm_block_size > 1`. At `nmm_block_size = 1` this reduces to
# `analytical_inner_grad` (T=1 trivially sums to one term).
#
# Tensor-core engagement: every matmul here has shape  M=H or D, N=H or D,
# K=T (the token batch). With T ≥ 16, cuBLAS GEMM routes to TC.
# ---------------------------------------------------------------------------


def _memory_mlp_forward_fullrank_chunk(
    k_chunk: torch.Tensor,        # [B, T, D]
    W1: torch.Tensor,             # [B, H, D]
    W_gate: torch.Tensor,         # [B, H, D]
    W2: torch.Tensor,             # [B, D, H]
    norm_weight: torch.Tensor,    # [D]
    norm_bias: torch.Tensor,      # [D]
    norm_eps: float,
):
    """Chunk-batched forward. All T tokens share the same per-sample weights;
    the [B, T, ...] layout puts T into the matmul N dim, enabling TC.

    Returns (out_chunk [B, T, D], intermediates with [B, T, ...] shapes).
    """
    # pre1[b, t, h] = sum_d W1[b, h, d] * k[b, t, d]
    # einsum routes this as bmm of [B, T, D] @ [B, D, H] = [B, T, H]
    pre1 = torch.einsum("bhd,btd->bth", W1, k_chunk)        # [B, T, H]
    preg = torch.einsum("bhd,btd->bth", W_gate, k_chunk)
    silu1 = F.silu(pre1)
    sigg = torch.sigmoid(preg)
    a = silu1 * sigg                                          # [B, T, H]
    # y[b, t, d] = sum_h W2[b, d, h] * a[b, t, h]
    y = torch.einsum("bdh,bth->btd", W2, a)                  # [B, T, D]
    # LayerNorm runs over the LAST dim (D) per token — same as single-token.
    # _layer_norm_forward handles arbitrary leading dims.
    y_norm, y_hat, inv_std = _layer_norm_forward(y, norm_weight, norm_bias, norm_eps)
    out = y_norm + k_chunk
    return out, {
        "pre1": pre1, "preg": preg,
        "silu1": silu1, "sigg": sigg, "a": a,
        "y_hat": y_hat, "inv_std": inv_std,
    }


def _memory_mlp_backward_fullrank_chunk(
    d_out: torch.Tensor,          # [B, T, D]
    k_chunk: torch.Tensor,        # [B, T, D]
    W2: torch.Tensor,             # [B, D, H]
    norm_weight: torch.Tensor,    # [D]
    interm: dict,
    theta_per_token=None,         # [B, T] or None
):
    """Chunk-aggregate analytical backward. Returns d_W1, d_W_gate, d_W2.

    When `theta_per_token` is None: returns the plain sum-over-T gradient
    (matches `grad(sum_t loss_t)`). This is the v1 chunk-aggregate behavior.

    When `theta_per_token` is provided: returns `Σ_t θ_t · ∇_t` — each
    per-token gradient gets weighted by its own θ_t before being summed
    into the chunk aggregate. This is the G267 paper-faithful refinement
    (matches paper Eq 16's `Σ_i θ_i ∇ℓ(M_0; x_i)` term without needing to
    materialise per-token gradient tensors). NO extra memory cost vs the
    plain aggregate path — θ is folded into the existing einsum.

    Critical TC-friendly ops (all use the T dim as the K of a GEMM):
        d_W2[b, d, h] = sum_t θ_t · d_y[b, t, d] · a[b, t, h]
        d_W1[b, h, d] = sum_t θ_t · d_pre1[b, t, h] · k[b, t, d]
    """
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out                                          # [B, T, D]
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    # d_a per token = W2.T @ d_y per token; vectorized over T.
    d_a = torch.einsum("bdh,btd->bth", W2, d_y)               # [B, T, H]
    d_silu1 = d_a * sigg
    d_sigg  = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)                       # [B, T, H]
    d_preg = d_sigg * _sigmoid_grad(sigg)
    if theta_per_token is None:
        d_W2     = torch.einsum("btd,bth->bdh", d_y, a)
        d_W1     = torch.einsum("bth,btd->bhd", d_pre1, k_chunk)
        d_W_gate = torch.einsum("bth,btd->bhd", d_preg, k_chunk)
    else:
        # θ-weighted aggregation — TC still engages with the (T) inner sum.
        d_W2     = torch.einsum("bt,btd,bth->bdh", theta_per_token, d_y, a)
        d_W1     = torch.einsum("bt,bth,btd->bhd", theta_per_token, d_pre1, k_chunk)
        d_W_gate = torch.einsum("bt,bth,btd->bhd", theta_per_token, d_preg, k_chunk)
    return d_W1, d_W_gate, d_W2


def _memory_mlp_forward_lowrank_chunk(
    k_chunk: torch.Tensor,
    W1_a: torch.Tensor, W1_b: torch.Tensor,
    Wg_a: torch.Tensor, Wg_b: torch.Tensor,
    W2_a: torch.Tensor, W2_b: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
):
    """Low-rank chunk-batched forward. Same factor structure as the
    per-token version with an added T dim in the activations."""
    # k_chunk: [B, T, D]; W1_a: [B, r, D]; W1_b: [B, H, r]
    t1   = torch.einsum("brd,btd->btr", W1_a, k_chunk)           # [B, T, r]
    pre1 = torch.einsum("bhr,btr->bth", W1_b, t1)                 # [B, T, H]
    tg   = torch.einsum("brd,btd->btr", Wg_a, k_chunk)
    preg = torch.einsum("bhr,btr->bth", Wg_b, tg)
    silu1 = F.silu(pre1); sigg = torch.sigmoid(preg)
    a    = silu1 * sigg                                            # [B, T, H]
    t2   = torch.einsum("brh,bth->btr", W2_a, a)                  # [B, T, r]
    y    = torch.einsum("bdr,btr->btd", W2_b, t2)                 # [B, T, D]
    y_norm, y_hat, inv_std = _layer_norm_forward(y, norm_weight, norm_bias, norm_eps)
    out = y_norm + k_chunk
    return out, {
        "pre1": pre1, "preg": preg, "silu1": silu1, "sigg": sigg, "a": a,
        "t1": t1, "tg": tg, "t2": t2,
        "y_hat": y_hat, "inv_std": inv_std,
    }


def _memory_mlp_backward_lowrank_chunk(
    d_out: torch.Tensor,
    k_chunk: torch.Tensor,
    W1_b: torch.Tensor, Wg_b: torch.Tensor,
    W2_a: torch.Tensor, W2_b: torch.Tensor,
    norm_weight: torch.Tensor,
    interm: dict,
    theta_per_token=None,         # [B, T] or None
):
    """Low-rank chunk-aggregate backward — returns dict of gradients in
    canonical sorted-low-rank order. When `theta_per_token` is provided,
    each per-token gradient gets weighted by θ_t before being summed
    (paper Eq 16, G267). No extra memory vs the plain aggregate path."""
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    t1 = interm["t1"]; tg = interm["tg"]; t2 = interm["t2"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    d_t2 = torch.einsum("bdr,btd->btr", W2_b, d_y)               # [B, T, r]
    d_a = torch.einsum("brh,btr->bth", W2_a, d_t2)               # [B, T, H]
    d_silu1 = d_a * sigg
    d_sigg  = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)
    d_preg = d_sigg * _sigmoid_grad(sigg)
    d_t1   = torch.einsum("bhr,bth->btr", W1_b, d_pre1)          # [B, T, r]
    d_tg   = torch.einsum("bhr,bth->btr", Wg_b, d_preg)
    if theta_per_token is None:
        d_W2_b = torch.einsum("btd,btr->bdr", d_y, t2)
        d_W2_a = torch.einsum("btr,bth->brh", d_t2, a)
        d_W1_b = torch.einsum("bth,btr->bhr", d_pre1, t1)
        d_W1_a = torch.einsum("btr,btd->brd", d_t1, k_chunk)
        d_Wg_b = torch.einsum("bth,btr->bhr", d_preg, tg)
        d_Wg_a = torch.einsum("btr,btd->brd", d_tg, k_chunk)
    else:
        # θ-weighted aggregation, no extra memory vs the plain version.
        d_W2_b = torch.einsum("bt,btd,btr->bdr", theta_per_token, d_y, t2)
        d_W2_a = torch.einsum("bt,btr,bth->brh", theta_per_token, d_t2, a)
        d_W1_b = torch.einsum("bt,bth,btr->bhr", theta_per_token, d_pre1, t1)
        d_W1_a = torch.einsum("bt,btr,btd->brd", theta_per_token, d_t1, k_chunk)
        d_Wg_b = torch.einsum("bt,bth,btr->bhr", theta_per_token, d_preg, tg)
        d_Wg_a = torch.einsum("bt,btr,btd->brd", theta_per_token, d_tg, k_chunk)
    return {
        "W1_a.weight": d_W1_a,
        "W1_b.weight": d_W1_b,
        "W2_a.weight": d_W2_a,
        "W2_b.weight": d_W2_b,
        "W_gate_a.weight": d_Wg_a,
        "W_gate_b.weight": d_Wg_b,
    }


def analytical_chunk_grad(
    M: dict,
    k_chunk: torch.Tensor,            # [B, T, D]
    v_chunk: torch.Tensor,            # [B, T, D]
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    reduction: str,
    theta_per_token=None,             # [B, T] or None
) -> dict:
    """Chunk-aggregate analytical gradient.

    When `theta_per_token` is None (default): returns
    `Σ_t grad(loss_t)` over the chunk's T tokens — semantically equivalent
    to summing `analytical_inner_grad` per-token.

    When `theta_per_token` is provided: returns `Σ_t θ_t · grad(loss_t)` —
    the paper-faithful (G267) per-token weighted aggregation. Used by the
    blockwise path to match paper Eq 16's `Σ_i θ_i ∇ℓ(M_0; x_i)` structure
    while keeping memory cost identical to the plain aggregate (θ is folded
    into the existing einsum, no per-token gradient tensors materialised).

    Both modes use batched matmuls with the T dim as the K of a GEMM —
    TC engages at T ≥ 16.
    """
    keys = set(M.keys())
    full_rank_keys = {"W1.weight", "W_gate.weight", "W2.weight"}
    low_rank_keys = {
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    }
    is_full_rank = keys == full_rank_keys
    is_low_rank = keys == low_rank_keys
    if not (is_full_rank or is_low_rank):
        raise ValueError(
            f"analytical_chunk_grad: unrecognised M key set {sorted(keys)}."
        )

    if is_full_rank:
        out, interm = _memory_mlp_forward_fullrank_chunk(
            k_chunk,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            norm_weight, norm_bias, norm_eps,
        )
    else:
        out, interm = _memory_mlp_forward_lowrank_chunk(
            k_chunk,
            M["W1_a.weight"], M["W1_b.weight"],
            M["W_gate_a.weight"], M["W_gate_b.weight"],
            M["W2_a.weight"], M["W2_b.weight"],
            norm_weight, norm_bias, norm_eps,
        )

    # Loss per token: same reduction as the single-token path.
    # Total loss = sum over tokens; gradient is naturally the sum of per-token
    # gradients by linearity of differentiation.
    diff = out - v_chunk                                           # [B, T, D]
    if reduction == "sum":
        d_out = 2.0 * diff
    elif reduction == "mean":
        d_out = (2.0 / diff.shape[-1]) * diff
    else:
        raise ValueError(f"reduction must be 'sum' or 'mean' (got {reduction!r}).")

    if is_full_rank:
        d_W1, d_W_gate, d_W2 = _memory_mlp_backward_fullrank_chunk(
            d_out, k_chunk, M["W2.weight"], norm_weight, interm,
            theta_per_token=theta_per_token,
        )
        return {
            "W1.weight": d_W1,
            "W_gate.weight": d_W_gate,
            "W2.weight": d_W2,
        }
    return _memory_mlp_backward_lowrank_chunk(
        d_out, k_chunk,
        M["W1_b.weight"], M["W_gate_b.weight"],
        M["W2_a.weight"], M["W2_b.weight"],
        norm_weight, interm,
        theta_per_token=theta_per_token,
    )


# ---------------------------------------------------------------------------
# Per-token analytical gradient over a chunk (G267, paper Eq 16 refinement)
# Returns gradients with the per-TOKEN dim retained — caller can apply NS5
# and per-token theta WEIGHTING before summing, matching the paper's chunked
# formulation `Σ_i θ_i NS5(∇ℓ(M_0; x_i))` rather than our v1 blockwise
# simplification `θ_mean * NS5(Σ_i ∇ℓ(M_0; x_i))`.
#
# Storage cost: gradient tensors have an extra T (block) leading dim. At
# gpt2_small full-rank, [B=1, T=64, H=3072, D=768] = 600 MB per state-key.
# At low_rank=64: ~30 MB per key. Manageable inside one block; allocated
# transiently and freed at block end.
# ---------------------------------------------------------------------------


def _memory_mlp_backward_fullrank_per_token(
    d_out: torch.Tensor,          # [B, T, D]
    k_chunk: torch.Tensor,        # [B, T, D]
    W2: torch.Tensor,             # [B, D, H]
    norm_weight: torch.Tensor,
    interm: dict,
):
    """Per-token analytical backward — returns per-token gradients with
    shape [B, T, ...] for each weight. Unlike `_memory_mlp_backward_fullrank_chunk`
    which SUMS over T, this keeps the T dim intact.

    The forward intermediates already have the T dim (from
    `_memory_mlp_forward_fullrank_chunk`), so the only difference from the
    aggregate backward is in the W-gradient computation: outer product
    per (b, t) instead of summed-over-t einsum."""
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out                                          # [B, T, D]
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    # d_W2 per-token: outer(d_y_t, a_t) for each (b, t).
    # einsum("btd,bth->btdh", d_y, a) -> [B, T, D, H]
    d_W2 = torch.einsum("btd,bth->btdh", d_y, a)              # [B, T, D, H]
    # d_a per token = W2.T @ d_y per-token; same as the aggregate path.
    d_a = torch.einsum("bdh,btd->bth", W2, d_y)               # [B, T, H]
    d_silu1 = d_a * sigg
    d_sigg  = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)
    d_preg = d_sigg * _sigmoid_grad(sigg)
    # d_W1, d_W_gate per-token: outer(d_pre1_t, k_t) for each (b, t).
    d_W1 = torch.einsum("bth,btd->bthd", d_pre1, k_chunk)     # [B, T, H, D]
    d_W_gate = torch.einsum("bth,btd->bthd", d_preg, k_chunk)
    return d_W1, d_W_gate, d_W2


def _memory_mlp_backward_lowrank_per_token(
    d_out: torch.Tensor,
    k_chunk: torch.Tensor,
    W1_b: torch.Tensor, Wg_b: torch.Tensor,
    W2_a: torch.Tensor, W2_b: torch.Tensor,
    norm_weight: torch.Tensor,
    interm: dict,
):
    """Low-rank per-token backward. Returns dict in canonical sorted order
    with [B, T, ...] shapes per key."""
    pre1 = interm["pre1"]; preg = interm["preg"]
    silu1 = interm["silu1"]; sigg = interm["sigg"]; a = interm["a"]
    t1 = interm["t1"]; tg = interm["tg"]; t2 = interm["t2"]
    y_hat = interm["y_hat"]; inv_std = interm["inv_std"]

    d_y_norm = d_out
    d_y = _layer_norm_backward_x(d_y_norm, y_hat, inv_std, norm_weight)
    # Per-token: same as aggregate but keep T dim instead of summing.
    d_W2_b = torch.einsum("btd,btr->btdr", d_y, t2)           # [B, T, D, r]
    d_t2 = torch.einsum("bdr,btd->btr", W2_b, d_y)
    d_W2_a = torch.einsum("btr,bth->btrh", d_t2, a)           # [B, T, r, H]
    d_a = torch.einsum("brh,btr->bth", W2_a, d_t2)
    d_silu1 = d_a * sigg
    d_sigg  = d_a * silu1
    d_pre1 = d_silu1 * _silu_grad(pre1)
    d_preg = d_sigg * _sigmoid_grad(sigg)
    d_W1_b = torch.einsum("bth,btr->bthr", d_pre1, t1)        # [B, T, H, r]
    d_t1   = torch.einsum("bhr,bth->btr", W1_b, d_pre1)
    d_W1_a = torch.einsum("btr,btd->btrd", d_t1, k_chunk)     # [B, T, r, D]
    d_Wg_b = torch.einsum("bth,btr->bthr", d_preg, tg)
    d_tg   = torch.einsum("bhr,bth->btr", Wg_b, d_preg)
    d_Wg_a = torch.einsum("btr,btd->btrd", d_tg, k_chunk)
    return {
        "W1_a.weight": d_W1_a,
        "W1_b.weight": d_W1_b,
        "W2_a.weight": d_W2_a,
        "W2_b.weight": d_W2_b,
        "W_gate_a.weight": d_Wg_a,
        "W_gate_b.weight": d_Wg_b,
    }


def analytical_per_token_grad(
    M: dict,
    k_chunk: torch.Tensor,            # [B, T, D]
    v_chunk: torch.Tensor,            # [B, T, D]
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    reduction: str,
) -> dict:
    """Per-token analytical gradient. Returns dict with [B, T, ...] tensors
    per state key — one gradient per (sample, token) instead of one
    aggregate per sample.

    Semantically: ∇ℓ(M; x_i) for each i in 0..T-1, all computed against
    the SHARED M (= chunk-start). Equivalent to vmapping
    `analytical_inner_grad` over the T dim.

    Per-token gradients are what the paper's Eq 16 needs: NS5 is applied
    PER token, then the per-token results are weighted by θ_i and summed.
    This differs from `analytical_chunk_grad`'s aggregate-then-NS5
    approach.
    """
    keys = set(M.keys())
    full_rank_keys = {"W1.weight", "W_gate.weight", "W2.weight"}
    low_rank_keys = {
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    }
    is_full_rank = keys == full_rank_keys
    is_low_rank = keys == low_rank_keys
    if not (is_full_rank or is_low_rank):
        raise ValueError(
            f"analytical_per_token_grad: unrecognised M key set {sorted(keys)}."
        )

    if is_full_rank:
        out, interm = _memory_mlp_forward_fullrank_chunk(
            k_chunk,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            norm_weight, norm_bias, norm_eps,
        )
    else:
        out, interm = _memory_mlp_forward_lowrank_chunk(
            k_chunk,
            M["W1_a.weight"], M["W1_b.weight"],
            M["W_gate_a.weight"], M["W_gate_b.weight"],
            M["W2_a.weight"], M["W2_b.weight"],
            norm_weight, norm_bias, norm_eps,
        )

    diff = out - v_chunk                                       # [B, T, D]
    if reduction == "sum":
        d_out = 2.0 * diff
    elif reduction == "mean":
        d_out = (2.0 / diff.shape[-1]) * diff
    else:
        raise ValueError(f"reduction must be 'sum' or 'mean' (got {reduction!r}).")

    if is_full_rank:
        d_W1, d_W_gate, d_W2 = _memory_mlp_backward_fullrank_per_token(
            d_out, k_chunk, M["W2.weight"], norm_weight, interm,
        )
        return {
            "W1.weight": d_W1,
            "W_gate.weight": d_W_gate,
            "W2.weight": d_W2,
        }
    return _memory_mlp_backward_lowrank_per_token(
        d_out, k_chunk,
        M["W1_b.weight"], M["W_gate_b.weight"],
        M["W2_a.weight"], M["W2_b.weight"],
        norm_weight, interm,
    )


def batched_retrieve_chunk(
    M: dict,
    q_chunk: torch.Tensor,           # [B, T, D]
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """MemoryMLP forward over per-sample M and a [B, T, D] query batch.
    Returns [B, T, D]. TC engages on the matmuls with N=T."""
    keys = set(M.keys())
    if keys == {"W1.weight", "W_gate.weight", "W2.weight"}:
        out, _ = _memory_mlp_forward_fullrank_chunk(
            q_chunk,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            norm_weight, norm_bias, norm_eps,
        )
        return out
    if keys == {
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    }:
        out, _ = _memory_mlp_forward_lowrank_chunk(
            q_chunk,
            M["W1_a.weight"], M["W1_b.weight"],
            M["W_gate_a.weight"], M["W_gate_b.weight"],
            M["W2_a.weight"], M["W2_b.weight"],
            norm_weight, norm_bias, norm_eps,
        )
        return out
    raise ValueError(
        f"batched_retrieve_chunk: unrecognised M key set {sorted(keys)}."
    )


def batched_retrieve(
    M: dict,
    q_hat: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """MemoryMLP forward over a per-sample-batched M and per-sample q_hat
    [B, d]. Returns [B, d]. Same numerical contract as the reference's
    `vmap(_retrieve_one_sample, in_dims=(0, 0))`.
    """
    keys = set(M.keys())
    if keys == {"W1.weight", "W_gate.weight", "W2.weight"}:
        out, _ = _memory_mlp_forward_fullrank(
            q_hat,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            norm_weight, norm_bias, norm_eps,
        )
        return out
    if keys == {
        "W1_a.weight", "W1_b.weight",
        "W_gate_a.weight", "W_gate_b.weight",
        "W2_a.weight", "W2_b.weight",
    }:
        out, _ = _memory_mlp_forward_lowrank(
            q_hat,
            M["W1_a.weight"], M["W1_b.weight"],
            M["W_gate_a.weight"], M["W_gate_b.weight"],
            M["W2_a.weight"], M["W2_b.weight"],
            norm_weight, norm_bias, norm_eps,
        )
        return out
    raise ValueError(
        f"batched_retrieve: unrecognised M key set {sorted(keys)}. "
        f"Expected full-rank or low-rank canonical keys."
    )
