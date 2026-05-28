"""GPU-tier: train_step with bf16 autocast actually works numerically.

On CPU the autocast(bf16) path through torch.func.grad fails at backward
with 'expected BFloat16 but found Float' due to limited CPU op coverage —
train_step defaults autocast_dtype=None there. On CUDA the path works;
this test verifies (a) forward+backward+step doesn't crash, (b) loss
trends down across a few steps, (c) param tensors remain fp32 (autocast
forward only invariant).
"""

import pytest
import torch

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import build_optimizer, run_training, train_step


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def _tiny_setup():
    _gpu_or_skip()
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=32, vocab_size=64,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg).cuda()
    optimizer = build_optimizer(model)
    return cfg, model, optimizer


def test_bf16_train_step_does_not_crash():
    """The CPU-disabled autocast path runs on CUDA — verify a single
    bf16-autocast step completes (no 'expected BFloat16 but found Float')."""
    cfg, model, opt = _tiny_setup()
    idx = torch.randint(0, cfg.vocab_size, (2, 4), device="cuda")
    db = torch.zeros(2, 4, dtype=torch.bool, device="cuda")
    db[:, 0] = True
    loss, _, gn = train_step(
        model, (idx, db), None, opt, torch.device("cuda"),
        autocast_dtype=torch.bfloat16,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(gn))


def test_bf16_params_stay_fp32_after_step():
    """autocast wraps the forward only — model parameters must stay
    fp32 throughout. Otherwise AdamW updates round to zero on fp16-like dtypes."""
    cfg, model, opt = _tiny_setup()
    idx = torch.randint(0, cfg.vocab_size, (2, 4), device="cuda")
    db = torch.zeros(2, 4, dtype=torch.bool, device="cuda")
    db[:, 0] = True
    train_step(
        model, (idx, db), None, opt, torch.device("cuda"),
        autocast_dtype=torch.bfloat16,
    )
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32, (
            f"param {name} became {p.dtype} after bf16-autocast step — "
            f"autocast leaked beyond forward (violation)"
        )


def test_bf16_loss_decreases_on_short_overfit():
    """Smoke training: 30 steps on a tiny synthetic stream, loss must trend
    down. Validates the bf16 forward + fp32 backward graph round-trips
    cleanly with the inner torch.func.grad."""
    cfg, model, opt = _tiny_setup()
    torch.manual_seed(0)
    stream = torch.randint(1, cfg.vocab_size, (2 * 4 * 8,), device="cuda")
    loader = ParallelStreamLoader(stream, 2, 4, eot_id=99999)
    # Use train_step in a loop (run_training already covered separately).
    losses = []
    states = None
    iter_batches = iter(loader)
    device = torch.device("cuda")
    for _ in range(30):
        try:
            batch = next(iter_batches)
        except StopIteration:
            iter_batches = iter(loader)
            batch = next(iter_batches)
            states = None
        loss, states, _ = train_step(
            model, batch, states, opt, device, autocast_dtype=torch.bfloat16,
        )
        losses.append(loss)
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, (
        f"bf16 training did not reduce loss: initial avg={initial:.3f}, "
        f"final avg={final:.3f}"
    )
