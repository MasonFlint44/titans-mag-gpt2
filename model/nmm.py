"""Neural Memory Module: SiLU-GLU gated MLP with online surprise-driven weight updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _checkpoint
from torch.func import functional_call, grad, vmap


# Map config strings -> torch dtypes for the recurrent (M, S) state and
# per-step update buffers. fp16 is intentionally excluded — it would need
# GradScaler wiring and the NMM's surprise gradient can overshoot fp16 range
# in early training. bf16 has fp32-equivalent range and "Just Works" with
# our bf16-autocast forward pipeline.
_STATE_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class _CPUOffloadCheckpoint(torch.autograd.Function):
    """Gradient-checkpoint variant that stashes saved tensors on CPU.

    Standard `torch.utils.checkpoint.checkpoint(use_reentrant=True)` saves
    its inputs on the SAME device (GPU) for later recompute. With many
    blocks × many segments, those boundary `(M, S)` tensors dominate VRAM
    at long T (G257 — 12 blocks × 64 segments × ~28 MiB per layer at
    gpt2_small, ~20 GiB just for boundaries at T=1024).

    This Function moves the saved input tensors to CPU after forward and
    moves them back to GPU during backward, recomputes forward there with
    autograd enabled, and runs backward through it. Net effect: identical
    gradients to GPU checkpoint, ~20× less GPU memory used by saved
    boundaries, at the cost of CPU↔GPU transfers per segment per backward
    (~21 GB of transfers per training step at T=1024 — PCIe 4.0 x16 caps
    at ~16 GB/s, so add ~1-2 s/step).

    Why a custom Function and not `torch.autograd.graph.save_on_cpu`:
    the latter uses `saved_tensors_hooks`, which `torch.func.grad`
    (per_sample_grad_fn) rejects at runtime. The reentrant custom
    Function path is hook-free, just like `use_reentrant=True` in
    `torch.utils.checkpoint`.

    Caller contract: pass a callable `fn` and ALL its tensor inputs as
    positional args. Non-tensor inputs (None, bool masks that we slice
    from larger tensors, etc.) get threaded through `non_tensor_args`
    keyword. The callable must be deterministic given its inputs (no
    hidden state changes between forward and recompute).
    """

    @staticmethod
    def forward(ctx, fn, n_tensors, *args):
        # Separate tensor inputs (must be backed up to CPU) from non-tensor
        # constants (doc-boundary slices, None init_M_* sentinels).
        tensor_args = args[:n_tensors]
        other_args = args[n_tensors:]

        ctx.fn = fn
        ctx.other_args = other_args
        ctx.n_tensors = n_tensors
        # Save on CPU as DETACHED clones — backward will re-attach with
        # requires_grad_ to rebuild the local autograd graph for recompute.
        # non_blocking=True is silently ignored without pinned memory; we
        # accept synchronous transfers here since the alternative
        # (pin_memory allocation per call) is its own bottleneck.
        ctx.cpu_tensors = tuple(
            t.detach().to("cpu") for t in tensor_args
        )
        ctx.tensor_devices = tuple(t.device for t in tensor_args)
        ctx.tensor_dtypes = tuple(t.dtype for t in tensor_args)
        ctx.tensor_requires_grad = tuple(t.requires_grad for t in tensor_args)

        # Run the actual forward in no_grad — we'll redo it WITH grad
        # during backward.
        with torch.no_grad():
            outputs = fn(*tensor_args, *other_args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Pull saved tensors back to GPU and rebuild a local autograd graph
        # by setting requires_grad on the ones that originally required it.
        gpu_inputs = []
        for cpu_t, dev, dtype, rg in zip(
            ctx.cpu_tensors,
            ctx.tensor_devices,
            ctx.tensor_dtypes,
            ctx.tensor_requires_grad,
        ):
            gt = cpu_t.to(dev, dtype=dtype)
            if rg:
                gt = gt.detach().requires_grad_(True)
            gpu_inputs.append(gt)

        with torch.enable_grad():
            outputs = ctx.fn(*gpu_inputs, *ctx.other_args)

        # outputs can be a tuple. Pair with incoming gradients; non-tensor
        # outputs (if any) are dropped on backward.
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        # Only backprop through outputs that have a grad_fn (some scalar
        # constants returned from fn would otherwise crash autograd.backward).
        outs_with_grad = []
        grads_with_grad = []
        for o, g in zip(outputs, grad_outputs):
            if isinstance(o, torch.Tensor) and o.requires_grad and g is not None:
                outs_with_grad.append(o)
                grads_with_grad.append(g)
        if outs_with_grad:
            torch.autograd.backward(outs_with_grad, grads_with_grad)

        # Collect input grads in the original arg order.
        input_grads = tuple(
            (gt.grad if isinstance(gt, torch.Tensor) and rg else None)
            for gt, rg in zip(gpu_inputs, ctx.tensor_requires_grad)
        )
        # Return tuple matching forward()'s signature: (fn, n_tensors, *args).
        # First two are None (non-tensor); rest are grads for tensor_args
        # then None for other_args.
        return (None, None) + input_grads + (None,) * len(ctx.other_args)


def cpu_offload_checkpoint(fn, *args):
    """Wrapper around `_CPUOffloadCheckpoint` that auto-detects which
    args are tensors. Tensor args (in order) come first; non-tensor args
    must come AFTER all tensors. This is matched by the call site in
    `_forward_chunk_sequential` which passes the segment-slice tensors
    first and the bool / None constants last.
    """
    n_tensors = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            n_tensors += 1
        else:
            break
    # Sanity: no tensor allowed after the first non-tensor.
    for a in args[n_tensors:]:
        if isinstance(a, torch.Tensor):
            raise RuntimeError(
                "cpu_offload_checkpoint requires all tensor args to come "
                "before any non-tensor arg; got a tensor after a "
                "non-tensor in the arg list."
            )
    return _CPUOffloadCheckpoint.apply(fn, n_tensors, *args)


# Resolve torch.associative_scan across PyTorch versions (G215). 2.8+ exposes
# it at `torch.associative_scan`; 2.6/2.7 only at the private path. Bind to
# a module-level name so callers don't redo the lookup.
try:
    from torch import associative_scan as _associative_scan
    _HAS_ASSOC_SCAN = True
except ImportError:
    try:
        from torch._higher_order_ops import associative_scan as _associative_scan
        _HAS_ASSOC_SCAN = True
    except ImportError:
        _associative_scan = None
        _HAS_ASSOC_SCAN = False


def allow_scan_training(model, enabled: bool = True) -> None:
    """Opt every block's NMM into the scan path during training.

    Only meaningful when the model is wrapped in torch.compile — without it,
    associative_scan lacks autograd and the scan path silently zeros NMM
    gradients (G180). The setattr-on-top-level-model pattern is a silent
    no-op since the dispatcher reads `self._allow_scan_training` on the NMM,
    not on the model.
    """
    for block in model.blocks:
        block.nmm._allow_scan_training = bool(enabled)


def reset_state(state: tuple, mask: torch.Tensor, init_M: dict) -> tuple:
    """Reset masked batch entries to init values (autograd-safe via torch.where).

    In-place index assignment on tensors in the autograd graph raises
    RuntimeError. torch.where is non-mutating and differentiable.

    mask: [B] bool. Where True, the entry gets init_M / zeros_S; where False,
    it keeps its current value bit-identically.
    """
    M, S = state
    zeros_S = {k: torch.zeros_like(v) for k, v in S.items()}

    def _where_dict(new_dict, old_dict):
        out = {}
        for k, new_v in new_dict.items():
            old_v = old_dict[k]
            m = mask.view(mask.shape[0], *([1] * (old_v.ndim - 1)))
            out[k] = torch.where(m, new_v, old_v)
        return out

    return (_where_dict(init_M, M), _where_dict(zeros_S, S))


def _detach_per_layer(layer_state):
    """Detach a single per-layer NMM state. Handles both shapes:
      - single-head: `(M, S)` tuple of dicts
      - multi-head: `[(M_h, S_h), ...]` list of per-head tuples (G254)
    """
    if isinstance(layer_state, list):
        return [_detach_per_layer(s) for s in layer_state]
    M, S = layer_state
    return (
        {k: v.detach() for k, v in M.items()},
        {k: v.detach() for k, v in S.items()},
    )


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


def _scale(scalar_B: torch.Tensor, tensor_dict: dict) -> dict:
    """Broadcast a per-sample scalar [B] across each [B, ...] tensor in the dict."""
    out = {}
    for k, g in tensor_dict.items():
        s = scalar_B.view(scalar_B.shape[0], *([1] * (g.ndim - 1)))
        out[k] = s * g
    return out


def _dict_add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _dict_sub(a: dict, b: dict) -> dict:
    return {k: a[k] - b[k] for k in a}


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
    """Q/K/V projection: Linear -> CausalDepthwiseConv1d, no activation.

    SiLU + L2-norm are applied at the call site, not inside the module —
    putting SiLU inside would silently produce silu(silu(x)) at the call site.

    Submodule names `linear` and `conv` are load-bearing for §4.1 optimizer
    routing: param paths like `blocks.X.nmm.k_proj.linear.weight` get into
    the NMM decay group via the `'nmm'` substring; renaming to anything
    containing `'norm'`, `'bias'`, or `'gamma'` would misroute to no_decay.
    """

    def __init__(self, n_embd: int, kernel_size: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.conv = CausalDepthwiseConv1d(n_embd, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.linear(x))


class MemoryMLP(nn.Module):
    """SiLU-GLU gated two-layer MLP (L_M = 2) with ResidualNorm.

    silu(W1 x) * sigmoid(W_gate x)  ->  W2  ->  norm(.) + x

    norm is a fixed stabilizer trained by the outer optimizer only — it is
    NOT recurrent state. Only the three 2D weights {W1, W_gate, W2} live in
    the recurrent (M, S); Newton-Schulz operates on 2D matrices.

    Dtype handling: when `functional_call` overrides W1/W_gate/W2 with bf16
    state, `x` flows through this MLP in bf16, but `self.norm.weight` /
    `self.norm.bias` remain fp32 (they are outer-trained params that
    AdamW expects in fp32). LayerNorm under autocast.bf16 already runs in
    fp32 internally, but `torch.func.grad` (used in the NMM inner loop)
    disables autocast. So we explicitly cast through fp32 around the norm
    — matches the autocast policy and lets bf16-state NMM forward without
    a dtype mismatch (G256).
    """

    def __init__(self, d: int, expansion: int = 4):
        super().__init__()
        h = d * expansion
        self.W1 = nn.Linear(d, h, bias=False)
        self.W_gate = nn.Linear(d, h, bias=False)
        self.W2 = nn.Linear(h, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.W1(x)) * torch.sigmoid(self.W_gate(x))
        y = self.W2(h)
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
        retrieval_from_M_prev: bool = False,
        state_dtype: str = "fp32",
        grad_checkpoint: bool = False,
        grad_checkpoint_segment_len: int = 64,
        cpu_offload_segments: bool = False,
        allow_scan_training: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.nmm_spectral_norm = spectral_norm
        self.finetune_mode = finetune_mode
        # Paper Eq. 15: y_t = M(q_t) where M is M_{t-1} (read-then-write).
        # Default False = lucidrains "write-then-read" (retrieve from M_t).
        self.retrieval_from_M_prev = retrieval_from_M_prev

        if state_dtype not in _STATE_DTYPE_MAP:
            raise ValueError(
                f"state_dtype must be one of {sorted(_STATE_DTYPE_MAP)} "
                f"(got {state_dtype!r})."
            )
        self.state_dtype_name = state_dtype
        self.state_dtype = _STATE_DTYPE_MAP[state_dtype]
        self.grad_checkpoint = grad_checkpoint
        if grad_checkpoint_segment_len < 1:
            raise ValueError(
                f"grad_checkpoint_segment_len must be >= 1 "
                f"(got {grad_checkpoint_segment_len})."
            )
        self.grad_checkpoint_segment_len = grad_checkpoint_segment_len
        if cpu_offload_segments and not grad_checkpoint:
            raise ValueError(
                "cpu_offload_segments=True requires grad_checkpoint=True; "
                "see TitansConfig validation note."
            )
        self.cpu_offload_segments = cpu_offload_segments
        # `_allow_scan_training` was previously a runtime-only attribute set
        # by the `allow_scan_training(model, True)` helper. Now also
        # settable at construction via the corresponding config flag so the
        # user doesn't need a separate post-construction step.
        self._allow_scan_training = bool(allow_scan_training)

        # Q/K/V projections — SiLU/L2 applied at call site, not inside.
        self.k_proj = NMMProjection(n_embd, kernel_size)
        self.q_proj = NMMProjection(n_embd, kernel_size)
        self.v_proj = NMMProjection(n_embd, kernel_size)

        # Per-token data-dependent update params (sigmoid+squeeze at call site).
        self.W_theta = nn.Linear(n_embd, 1, bias=False)
        self.W_eta = nn.Linear(n_embd, 1, bias=False)
        self.W_alpha = nn.Linear(n_embd, 1, bias=False)

        # MemoryMLP. Its W*.weight ARE the meta-learned initial values of M;
        # _build_init_M reads them at sequence/document start to seed state.
        self.memory_mlp = MemoryMLP(n_embd, expansion)
        nn.init.xavier_uniform_(self.memory_mlp.W1.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W_gate.weight)
        nn.init.xavier_uniform_(self.memory_mlp.W2.weight)

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
        """Per-sample-batched initial M dict from memory_mlp.W*.weight,
        cast to `self.state_dtype` (fp32 by default, bf16 when configured).

        `.expand(B, -1, -1)` is a stride-0 view; vmap with in_dims=0 over
        such views is undefined in the batched autograd interpreter, so
        we `.clone()` to materialize normal strides. Order is
        `.to(device, dtype).clone()`: on the same device the dtype cast
        happens before the materialization, avoiding a fp32 staging copy
        when state_dtype is bf16.
        """
        W1 = self.memory_mlp.W1.weight
        W_gate = self.memory_mlp.W_gate.weight
        W2 = self.memory_mlp.W2.weight
        dt = self.state_dtype
        return {
            "W1.weight":     W1.unsqueeze(0).expand(B, -1, -1).to(device, dtype=dt).clone(),
            "W_gate.weight": W_gate.unsqueeze(0).expand(B, -1, -1).to(device, dtype=dt).clone(),
            "W2.weight":     W2.unsqueeze(0).expand(B, -1, -1).to(device, dtype=dt).clone(),
        }

    def init_state(self, B: int, device) -> tuple:
        M = self._build_init_M(B, device)
        S = {k: torch.zeros_like(v) for k, v in M.items()}
        return (M, S)

    def step(self, x_t: torch.Tensor, state: tuple) -> tuple:
        """Process a single token. Inference-time path.

        Do NOT call in a training loop — the conv (kernel_size=4) only sees a
        1-token window per call (3 of 4 weights are masked by left-pad). The
        training path is `_forward_chunk_sequential` which pre-projects the
        full chunk so the conv sees up-to-k context.

        x_t: [B, d]; state: (M_prev, S_prev), each dict of [B, h, d] / [B, d, h].
        Returns: (y_t, new_state) where y_t: [B, d].
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm}). The cached "
                "per_sample_grad_fn's reduction is locked at __init__; "
                "rebuild the module to change spectral_norm."
            )
        M_prev, S_prev = state

        x_seq = x_t.unsqueeze(1)
        k_raw = self.k_proj(x_seq).squeeze(1)
        q_raw = self.q_proj(x_seq).squeeze(1)
        v_raw = self.v_proj(x_seq).squeeze(1)
        k_hat = F.normalize(F.silu(k_raw), dim=-1)
        q_hat = F.normalize(F.silu(q_raw), dim=-1)
        v = F.silu(v_raw)

        theta_t = torch.sigmoid(self.W_theta(x_t)).squeeze(-1)
        eta_t = torch.sigmoid(self.W_eta(x_t)).squeeze(-1)
        alpha_t = torch.sigmoid(self.W_alpha(x_t)).squeeze(-1)

        # Cast inputs to state_dtype so per_sample_grad_fn's overridden
        # weights (state_dtype) match input dtype — see forward_chunk note.
        if self.state_dtype != k_hat.dtype:
            k_hat = k_hat.to(self.state_dtype)
            q_hat = q_hat.to(self.state_dtype)
            v = v.to(self.state_dtype)
            theta_t = theta_t.to(self.state_dtype)
            eta_t = eta_t.to(self.state_dtype)
            alpha_t = alpha_t.to(self.state_dtype)

        g_t = self.per_sample_grad_fn(M_prev, k_hat, v)
        if self.nmm_spectral_norm:
            g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
        else:
            g_tilde = g_t

        # Momentum + memory update; theta POST-NS so the scale survives Frobenius division.
        S_t = _dict_sub(_scale(eta_t, S_prev), _scale(theta_t, g_tilde))
        M_t = _dict_add(_scale(1.0 - alpha_t, M_prev), S_t)

        # Retrieval source per config: M_prev (paper Eq. 15, read-then-write)
        # or M_t (lucidrains default, write-then-read).
        M_for_retrieval = M_prev if self.retrieval_from_M_prev else M_t
        y_t = self.out_scale * self._batched_retrieve(M_for_retrieval, q_hat)
        return y_t, (M_t, S_t)

    def init_conv_buffer_from_prompt(self, x_chunk: torch.Tensor) -> dict:
        """Seed conv buffers from the last (k-1) tokens of a warm-up prompt.

        At decode time, step_with_conv needs the last (k-1) Linear-projected
        values for each of q/k/v so the conv can see a full k-token window
        (instead of T=1 step()'s zero-padded 1-token window). This helper
        re-projects the prompt's tail; cheaper than threading buffer capture
        through forward_chunk's batched projection path.

        x_chunk: [B, T, d] — post-ln_nmm prompt input.
        Returns dict {'q', 'k', 'v'} each [B, k-1, d]. Left-padded with
        zeros if T < k-1.
        """
        k = self.k_proj.conv.kernel_size
        pad_size = k - 1
        B, T, d = x_chunk.shape
        if T >= pad_size:
            last = x_chunk[:, -pad_size:, :]
        else:
            zero_pad = torch.zeros(
                B, pad_size - T, d,
                device=x_chunk.device, dtype=x_chunk.dtype,
            )
            last = torch.cat([zero_pad, x_chunk], dim=1)
        return {
            "q": self.q_proj.linear(last),
            "k": self.k_proj.linear(last),
            "v": self.v_proj.linear(last),
        }

    def step_with_conv(
        self,
        x_t: torch.Tensor,
        state: tuple,
        conv_buffer: dict,
    ) -> tuple:
        """Decode-path single-token step that uses a conv buffer so the
        depthwise conv sees a full k-token window (vs. step()'s T=1 with
        zero-padding which silently disables 3 of 4 kernel weights).

        x_t: [B, d]; state: (M_prev, S_prev); conv_buffer: dict {q, k, v}
        each [B, k-1, d] of prior Linear projections.

        Returns (y_t [B, d], new_state, new_conv_buffer).
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm})."
            )

        M_prev, S_prev = state
        x_unsq = x_t.unsqueeze(1)  # [B, 1, d]

        # Linear projection only (no conv yet, no activation).
        q_lin = self.q_proj.linear(x_unsq)  # [B, 1, d]
        k_lin = self.k_proj.linear(x_unsq)
        v_lin = self.v_proj.linear(x_unsq)

        # Concat with buffer (last k-1 prior linear projections) -> length-k
        # input. Run conv; take the LAST position (conv at that position uses
        # the full [buffer | new] context).
        q_input = torch.cat([conv_buffer["q"], q_lin], dim=1)  # [B, k, d]
        k_input = torch.cat([conv_buffer["k"], k_lin], dim=1)
        v_input = torch.cat([conv_buffer["v"], v_lin], dim=1)
        q_conv = self.q_proj.conv(q_input)[:, -1, :]  # [B, d]
        k_conv = self.k_proj.conv(k_input)[:, -1, :]
        v_conv = self.v_proj.conv(v_input)[:, -1, :]

        # Call-site SiLU + L2 (same as step()).
        k_hat = F.normalize(F.silu(k_conv), dim=-1)
        q_hat = F.normalize(F.silu(q_conv), dim=-1)
        v = F.silu(v_conv)

        theta_t = torch.sigmoid(self.W_theta(x_t)).squeeze(-1)
        eta_t = torch.sigmoid(self.W_eta(x_t)).squeeze(-1)
        alpha_t = torch.sigmoid(self.W_alpha(x_t)).squeeze(-1)

        # Cast to state_dtype — see forward_chunk note. Decode path runs
        # without autocast (the `prepare_decode` eval-mode contract), so
        # without this cast inputs would be fp32 vs. bf16 M weights.
        if self.state_dtype != k_hat.dtype:
            k_hat = k_hat.to(self.state_dtype)
            q_hat = q_hat.to(self.state_dtype)
            v = v.to(self.state_dtype)
            theta_t = theta_t.to(self.state_dtype)
            eta_t = eta_t.to(self.state_dtype)
            alpha_t = alpha_t.to(self.state_dtype)

        g_t = self.per_sample_grad_fn(M_prev, k_hat, v)
        if self.nmm_spectral_norm:
            g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
        else:
            g_tilde = g_t

        S_t = _dict_sub(_scale(eta_t, S_prev), _scale(theta_t, g_tilde))
        M_t = _dict_add(_scale(1.0 - alpha_t, M_prev), S_t)
        M_for_retrieval = M_prev if self.retrieval_from_M_prev else M_t
        y_t = self.out_scale * self._batched_retrieve(M_for_retrieval, q_hat)

        # Update conv buffer: drop oldest, append the new linear projection.
        new_buffer = {
            "q": torch.cat([conv_buffer["q"][:, 1:, :], q_lin], dim=1),
            "k": torch.cat([conv_buffer["k"][:, 1:, :], k_lin], dim=1),
            "v": torch.cat([conv_buffer["v"][:, 1:, :], v_lin], dim=1),
        }
        return y_t, (M_t, S_t), new_buffer

    def _run_inner_loop(
        self,
        k_hat_seg: torch.Tensor,
        q_hat_seg: torch.Tensor,
        v_seg: torch.Tensor,
        theta_seg: torch.Tensor,
        eta_seg: torch.Tensor,
        alpha_seg: torch.Tensor,
        M_W1, M_Wg, M_W2,
        S_W1, S_Wg, S_W2,
        db_seg,
        init_M_W1, init_M_Wg, init_M_W2,
    ) -> tuple:
        """Run the per-token NMM update loop over a (sub)chunk and return:
          (y_seg [B, T_seg, d], M_W1, M_Wg, M_W2, S_W1, S_Wg, S_W2).

        All state inputs/outputs are flat tensors (not dicts) so this is
        directly wrappable in `torch.utils.checkpoint.checkpoint`, which
        requires tensor-only signatures with `use_reentrant=False`.

        `db_seg` is the [B, T_seg] bool mask of document boundaries for
        this segment (or None). When a boundary fires, the corresponding
        rows of M/S get reset from `init_M_*` (zeros for S). `init_M_*`
        is precomputed once at the chunk level (so it doesn't have to be
        re-built per segment) and passed in as flat tensors too — None
        sentinels are not allowed through checkpoint, but the boundary-
        free common case still avoids the reset altogether via the mask
        being all-False.
        """
        T_seg = k_hat_seg.shape[1]
        M = {"W1.weight": M_W1, "W_gate.weight": M_Wg, "W2.weight": M_W2}
        S = {"W1.weight": S_W1, "W_gate.weight": S_Wg, "W2.weight": S_W2}
        init_M = (
            {"W1.weight": init_M_W1, "W_gate.weight": init_M_Wg,
             "W2.weight": init_M_W2}
            if init_M_W1 is not None else None
        )
        y_list = []
        for t in range(T_seg):
            if db_seg is not None and bool(db_seg[:, t].any()):
                # `init_M` is guaranteed non-None at this branch by the caller —
                # if any boundary in the WHOLE chunk fires, the caller builds it.
                M, S = reset_state((M, S), db_seg[:, t], init_M)

            k_hat_t = k_hat_seg[:, t, :]
            q_hat_t = q_hat_seg[:, t, :]
            v_t = v_seg[:, t, :]
            theta_t = theta_seg[:, t]
            eta_t = eta_seg[:, t]
            alpha_t = alpha_seg[:, t]

            M_prev = M

            g_t = self.per_sample_grad_fn(M, k_hat_t, v_t)
            if self.nmm_spectral_norm:
                g_tilde = {key: newton_schulz5(g) for key, g in g_t.items()}
            else:
                g_tilde = g_t

            S = _dict_sub(_scale(eta_t, S), _scale(theta_t, g_tilde))
            M = _dict_add(_scale(1.0 - alpha_t, M), S)

            M_for_retrieval = M_prev if self.retrieval_from_M_prev else M
            y_t = self.out_scale * self._batched_retrieve(M_for_retrieval, q_hat_t)
            y_list.append(y_t)

        y_seg = torch.stack(y_list, dim=1)
        return (
            y_seg,
            M["W1.weight"], M["W_gate.weight"], M["W2.weight"],
            S["W1.weight"], S["W_gate.weight"], S["W2.weight"],
        )

    def _forward_chunk_sequential(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries,
    ) -> tuple:
        """Training-path chunked forward: pre-project the full chunk, then loop
        only over the recurrent state update.

        Per-token step() in a training loop would feed the conv a 1-token
        window (3 of 4 kernel weights dead). The pre-projection here lets
        the conv see up-to-k tokens of causal context for every output.

        Two performance/memory guards:
        - init_M is lazy-built only when a boundary actually fires in the
          whole chunk (zero-cost on the common no-boundary chunk).
        - The per-position "any boundary?" mask is computed once on CPU
          to avoid T implicit GPU->CPU syncs from `tensor.any()` inside
          a Python `if`.

        When `self.grad_checkpoint` is True, the per-token inner loop is
        broken into `grad_checkpoint_segment_len`-token segments, each
        wrapped in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
        so backward recomputes the segment's intermediates instead of
        storing them. This trades one extra forward per segment during
        backward for ~5-10x peak-memory headroom — the difference between
        T=32 and T=1024 fitting on a 16 GiB consumer card (G257).
        """
        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction "
                f"(init={self._spectral_norm_at_init}, "
                f"now={self.nmm_spectral_norm}). The cached "
                "per_sample_grad_fn's reduction is locked at __init__; "
                "rebuild the module to change spectral_norm."
            )

        B, T, _ = x_chunk.shape
        # Full-chunk projection — conv sees up-to-k tokens per output.
        k_hat_chunk = F.normalize(F.silu(self.k_proj(x_chunk)), dim=-1)
        q_hat_chunk = F.normalize(F.silu(self.q_proj(x_chunk)), dim=-1)
        v_chunk = F.silu(self.v_proj(x_chunk))
        theta_chunk = torch.sigmoid(self.W_theta(x_chunk)).squeeze(-1)
        eta_chunk = torch.sigmoid(self.W_eta(x_chunk)).squeeze(-1)
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

        M_dict, S_dict = state_in

        # Eagerly build init_M iff any boundary fires anywhere in the chunk.
        # The grad-checkpointed segment loop can't lazily build init_M from
        # inside the checkpointed callable (calling _build_init_M during
        # backward-recompute would silently re-read potentially-grad-tracked
        # MemoryMLP weights and reshape the autograd graph), so we hoist the
        # decision here and pass init_M_* down as tensors (or Nones).
        any_boundary = (
            doc_boundaries is not None and bool(doc_boundaries.any())
        )
        if any_boundary:
            init_M = self._build_init_M(B, x_chunk.device)
            init_M_W1 = init_M["W1.weight"]
            init_M_Wg = init_M["W_gate.weight"]
            init_M_W2 = init_M["W2.weight"]
        else:
            init_M_W1 = init_M_Wg = init_M_W2 = None

        seg_len = (
            self.grad_checkpoint_segment_len
            if (self.grad_checkpoint and torch.is_grad_enabled())
            else T
        )

        M_W1 = M_dict["W1.weight"]
        M_Wg = M_dict["W_gate.weight"]
        M_W2 = M_dict["W2.weight"]
        S_W1 = S_dict["W1.weight"]
        S_Wg = S_dict["W_gate.weight"]
        S_W2 = S_dict["W2.weight"]

        y_segments = []
        for start in range(0, T, seg_len):
            end = min(start + seg_len, T)
            k_seg = k_hat_chunk[:, start:end]
            q_seg = q_hat_chunk[:, start:end]
            v_seg = v_chunk[:, start:end]
            theta_seg = theta_chunk[:, start:end]
            eta_seg = eta_chunk[:, start:end]
            alpha_seg = alpha_chunk[:, start:end]
            db_seg = (
                doc_boundaries[:, start:end] if doc_boundaries is not None else None
            )

            args = (
                k_seg, q_seg, v_seg, theta_seg, eta_seg, alpha_seg,
                M_W1, M_Wg, M_W2, S_W1, S_Wg, S_W2,
                db_seg,
                init_M_W1, init_M_Wg, init_M_W2,
            )
            if self.grad_checkpoint and torch.is_grad_enabled() and seg_len < T:
                if self.cpu_offload_segments:
                    # Same recompute semantics as use_reentrant=True
                    # checkpoint, but with all tensor inputs stashed on
                    # CPU between forward and backward (G258 — VRAM win
                    # at the cost of CPU↔GPU transfers). Argument order
                    # MUST place all tensors before any non-tensor
                    # constant (the `db_seg` bool slice / `init_M_*`
                    # potentially-None sentinels) — see
                    # `cpu_offload_checkpoint`.
                    y_seg, M_W1, M_Wg, M_W2, S_W1, S_Wg, S_W2 = cpu_offload_checkpoint(
                        self._run_inner_loop, *args,
                    )
                else:
                    # use_reentrant=True is REQUIRED here even though the
                    # modern default is False: `use_reentrant=False`
                    # enables `disable_saved_tensors_hooks`, which
                    # `torch.func.grad` (used inside `per_sample_grad_fn`)
                    # rejects at runtime. The legacy reentrant path goes
                    # through `torch.autograd.function.Function` and does
                    # NOT touch saved-tensor hooks, so it composes with
                    # torch.func.
                    #
                    # Reentrant-path quirks we already handle:
                    # - All tensor inputs (state, init_M_*, segment slices)
                    #   must be properly tracked: the chunk-level state
                    #   tensors flow through `memory_mlp.W*.weight` which
                    #   has requires_grad=True, so autograd is connected.
                    # - boundary masks are bool / non-grad — fine,
                    #   checkpoint silently passes non-floats through.
                    # - autocast: reentrant captures + restores the
                    #   autocast state on recompute by default in modern
                    #   torch.
                    y_seg, M_W1, M_Wg, M_W2, S_W1, S_Wg, S_W2 = _checkpoint.checkpoint(
                        self._run_inner_loop, *args,
                        use_reentrant=True,
                    )
            else:
                y_seg, M_W1, M_Wg, M_W2, S_W1, S_Wg, S_W2 = self._run_inner_loop(*args)

            y_segments.append(y_seg)

        y_chunk = torch.cat(y_segments, dim=1)
        new_state = (
            {"W1.weight": M_W1, "W_gate.weight": M_Wg, "W2.weight": M_W2},
            {"W1.weight": S_W1, "W_gate.weight": S_Wg, "W2.weight": S_W2},
        )
        return y_chunk, new_state

    def _forward_chunk_scan(
        self,
        x_chunk: torch.Tensor,
        state_in: tuple,
        doc_boundaries,
    ) -> tuple:
        """Associative-scan path. Approximation: all per-token gradients are
        computed against the chunk-start M_0 (not against M_{t-1}). This breaks
        the recurrence's true sequential dependency in exchange for parallelism.

        Caller (forward_chunk dispatcher) MUST ensure:
          - doc_boundaries is None or has no True entries (the scan can't do
            mid-chunk state resets).
          - Either grad is disabled OR the model is wrapped in torch.compile
            (associative_scan lacks autograd otherwise).
        """
        if not _HAS_ASSOC_SCAN:
            raise RuntimeError("_forward_chunk_scan called but associative_scan unavailable")

        if self.nmm_spectral_norm != self._spectral_norm_at_init:
            raise RuntimeError(
                "nmm_spectral_norm was mutated after construction — see step()."
            )

        M_state, S_state = state_in

        k_hat_chunk = F.normalize(F.silu(self.k_proj(x_chunk)), dim=-1)
        q_hat_chunk = F.normalize(F.silu(self.q_proj(x_chunk)), dim=-1)
        v_chunk = F.silu(self.v_proj(x_chunk))
        theta_chunk = torch.sigmoid(self.W_theta(x_chunk)).squeeze(-1)
        eta_chunk = torch.sigmoid(self.W_eta(x_chunk)).squeeze(-1)
        alpha_chunk = torch.sigmoid(self.W_alpha(x_chunk)).squeeze(-1)

        # Mirror the chunk-level cast from `_forward_chunk_sequential` —
        # state_dtype=bf16 requires inputs to per_sample_grad_fn match the
        # bf16 (M, S) tensors or F.linear fails inside the nested
        # vmap(grad(...)) (G256).
        if self.state_dtype != x_chunk.dtype:
            k_hat_chunk = k_hat_chunk.to(self.state_dtype)
            q_hat_chunk = q_hat_chunk.to(self.state_dtype)
            v_chunk = v_chunk.to(self.state_dtype)
            theta_chunk = theta_chunk.to(self.state_dtype)
            eta_chunk = eta_chunk.to(self.state_dtype)
            alpha_chunk = alpha_chunk.to(self.state_dtype)

        # All gradients in parallel: outer vmap over T (dim 1 of chunks), inner
        # vmap over B (already inside per_sample_grad_fn). M_state shared (None).
        all_grads = vmap(self.per_sample_grad_fn, in_dims=(None, 1, 1))(
            M_state, k_hat_chunk, v_chunk
        )  # dict of [T, B, h, d]

        th = theta_chunk.T.unsqueeze(-1).unsqueeze(-1)         # [T, B, 1, 1]
        eta = eta_chunk.T.unsqueeze(-1).unsqueeze(-1)
        one_minus_alpha = (1.0 - alpha_chunk.T).unsqueeze(-1).unsqueeze(-1)

        # NS5 batched over the T,B leading dims; theta POST-NS as elsewhere.
        scaled_grads = {}
        for key, g in all_grads.items():
            if self.nmm_spectral_norm:
                g = newton_schulz5(g)
            scaled_grads[key] = th * g

        # Associative op for S_t = decay * S_{t-1} + delta_t:
        #   (decay_a, delta_a) ⊗ (decay_b, delta_b) =
        #     (decay_b * decay_a, decay_b * delta_a + delta_b)
        def assoc_op(carry_a, carry_b):
            decay_a, delta_a = carry_a
            decay_b, delta_b = carry_b
            return (decay_b * decay_a, decay_b * delta_a + delta_b)

        one_B11 = eta.new_ones(1, *eta.shape[1:])

        S_chunk = {}
        M_chunk = {}
        for key, dg in scaled_grads.items():
            S0 = S_state[key].unsqueeze(0)              # [1, B, h, d]
            M0 = M_state[key].unsqueeze(0)

            # S scan: prepend (1, S_0) so the scan output at index t > 0
            # is the correct S_t with S_0 incorporated. Slice [1:] to drop
            # the synthetic prepended element.
            eta_aug = torch.cat([one_B11, eta], dim=0)
            neg_dg_aug = torch.cat([S0, -dg], dim=0)
            _, S_aug = _associative_scan(
                assoc_op, (eta_aug, neg_dg_aug), dim=0, combine_mode="generic"
            )
            S_chunk[key] = S_aug[1:]

            # M scan: same augmented trick with M_0 + S_chunk as deltas.
            alpha_aug = torch.cat([one_B11, one_minus_alpha], dim=0)
            delta_aug = torch.cat([M0, S_chunk[key]], dim=0)
            _, M_aug = _associative_scan(
                assoc_op, (alpha_aug, delta_aug), dim=0, combine_mode="generic"
            )
            M_chunk[key] = M_aug[1:]

        # Retrieval source: M_chunk[t] is M_t (post-update). For paper-Eq.-15
        # ordering we instead want M_{t-1} at position t — prepend M_state
        # (= M_0) and drop the last entry so position t reads from M_{t-1}.
        if self.retrieval_from_M_prev:
            M_for_retrieval = {}
            for k, v in M_chunk.items():
                # [T, B, h, d]; prepend M_state[k] shape [B, h, d] -> [1, B, h, d]
                M_for_retrieval[k] = torch.cat(
                    [M_state[k].unsqueeze(0), v[:-1]], dim=0,
                )
        else:
            M_for_retrieval = M_chunk

        # Retrieval: outer vmap over T, inner is the cached _batched_retrieve.
        y_raw = vmap(self._batched_retrieve, in_dims=(0, 1))(M_for_retrieval, q_hat_chunk)
        y_chunk = self.out_scale * y_raw.transpose(0, 1)  # [B, T, d]

        state_out = (
            {k: v[-1] for k, v in M_chunk.items()},
            {k: v[-1] for k, v in S_chunk.items()},
        )
        return y_chunk, state_out

    def forward_chunk(self, x_chunk, state_in, doc_boundaries):
        """Dispatch between scan and sequential paths.

        Scan is taken only when:
          - associative_scan is available, AND
          - no doc boundaries in this chunk (scan can't reset mid-chunk), AND
          - autograd is disabled OR _allow_scan_training is set (the latter
            opted in only when the model is wrapped in torch.compile —
            otherwise scan zeroes NMM gradients silently).

        Gate uses torch.is_grad_enabled (G164), not self.training. The two
        are independent: a `model.eval()`-then-forgot-to-`train()` pattern
        leaves self.training=False during a training loop with autograd on,
        which the old self.training gate misread as "safe to scan" — silent
        NMM-gradient-freeze. Probing autograd directly is robust to that.
        """
        can_scan = _HAS_ASSOC_SCAN and (
            doc_boundaries is None or not bool(doc_boundaries.any())
        )
        if torch.is_grad_enabled() and not getattr(
            self, "_allow_scan_training", False
        ):
            can_scan = False
        if can_scan:
            return self._forward_chunk_scan(x_chunk, state_in, doc_boundaries)
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
        retrieval_from_M_prev: bool = False,
        state_dtype: str = "fp32",
        grad_checkpoint: bool = False,
        grad_checkpoint_segment_len: int = 64,
        cpu_offload_segments: bool = False,
        allow_scan_training: bool = False,
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
        # state_dtype / grad_checkpoint / cpu_offload / allow_scan_training
        # propagate per-head. Per-head segment_len is the same since heads
        # share the chunk-time dimension.
        self.heads = nn.ModuleList([
            NeuralMemoryModule(
                n_embd=self.head_dim,
                expansion=expansion,
                kernel_size=kernel_size,
                spectral_norm=spectral_norm,
                finetune_mode=finetune_mode,
                retrieval_from_M_prev=retrieval_from_M_prev,
                state_dtype=state_dtype,
                grad_checkpoint=grad_checkpoint,
                grad_checkpoint_segment_len=grad_checkpoint_segment_len,
                cpu_offload_segments=cpu_offload_segments,
                allow_scan_training=allow_scan_training,
            )
            for _ in range(n_heads)
        ])

    # --- Properties / helpers ----------------------------------------------

    @property
    def memory_mlp(self):
        """For backward-compat with code paths that read `nmm.memory_mlp.W*`
        for shape / dtype probes (e.g., `init_conv_buffer_from_prompt` dtype
        inference, `_apply_gpt2_init`'s NMM-skip-by-id). Returns the FIRST
        head's MemoryMLP — sufficient for shape/dtype-only consumers."""
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

    def init_conv_buffer_from_prompt(self, x_chunk: torch.Tensor) -> list:
        """Per-head conv buffer; the block stores a list[n_heads] of buffer
        dicts in place of the single-head dict."""
        x_split = self._split_heads(x_chunk)
        return [
            head.init_conv_buffer_from_prompt(x_split[..., i, :].contiguous())
            for i, head in enumerate(self.heads)
        ]

    def step_with_conv(self, x_t, state, conv_buffer):
        """Per-head step. x_t is [B, d_model]; split into per-head [B, head_dim]
        slices, run each head's step_with_conv, concatenate outputs.
        `conv_buffer` is a list[n_heads] of per-head buffer dicts."""
        x_split = self._split_heads(x_t)  # [B, n_heads, head_dim]
        outputs = []
        new_states = []
        new_buffers = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()
            y_h, s_h, b_h = head.step_with_conv(x_h, state[i], conv_buffer[i])
            outputs.append(y_h)
            new_states.append(s_h)
            new_buffers.append(b_h)
        return self._merge_heads(outputs), new_states, new_buffers

    def step(self, x_t, state):
        """Legacy single-token step (no conv buffer). Each head's step has
        its own zero-padded conv window; per-head dispatch."""
        x_split = self._split_heads(x_t)
        outputs = []
        new_states = []
        for i, head in enumerate(self.heads):
            x_h = x_split[..., i, :].contiguous()
            y_h, s_h = head.step(x_h, state[i])
            outputs.append(y_h)
            new_states.append(s_h)
        return self._merge_heads(outputs), new_states
