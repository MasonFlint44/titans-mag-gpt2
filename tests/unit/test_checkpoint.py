"""Phase 4.3 — checkpoint save/load round-trip + compute_nmm_norm helper."""

import dataclasses
import os
import tempfile
from pathlib import Path

import pytest
import torch

from config import TitansConfig
from model.titans_gpt2 import TitansMAGGPT2
from train import (
    BASE_LR_GPT2,
    BASE_LR_NMM,
    LATEST_CKPT_NAME,
    base_lrs_from_constants,
    build_optimizer,
    compute_nmm_norm,
    list_step_checkpoints,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
    save_checkpoint_rotating,
    train_step,
)


def _tiny_setup(finetune_mode=False):
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=finetune_mode,
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)
    return cfg, model, optimizer


def _fake_batch(cfg, B=2, T=4):
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    db = torch.zeros(B, T, dtype=torch.bool)
    db[:, 0] = True
    return (idx, db)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_state_dict():
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=42, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert ckpt["step"] == 42
        assert set(ckpt["state_dict"].keys()) == set(model.state_dict().keys())
        for k in ckpt["state_dict"]:
            assert torch.equal(ckpt["state_dict"][k], model.state_dict()[k])


def test_save_load_round_trip_preserves_config():
    cfg, model, opt = _tiny_setup(finetune_mode=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        # Round-trip via dict -> TitansConfig must reconstruct equivalent config.
        cfg2 = TitansConfig(**ckpt["config"])
        assert dataclasses.asdict(cfg) == dataclasses.asdict(cfg2)


def test_save_load_optimizer_state_populated_after_step():
    """G153: after a real .step(), Adam moments exp_avg / exp_avg_sq exist."""
    cfg, model, opt = _tiny_setup()
    batch = _fake_batch(cfg)
    train_step(model, batch, None, opt, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=1, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert "optimizer" in ckpt
        # The state dict's 'state' must be non-empty (at least one param has Adam moments).
        opt_state = ckpt["optimizer"]["state"]
        assert len(opt_state) > 0, "Adam moments not saved"


def test_load_works_under_weights_only_false():
    """G168: torch.load(weights_only=True) (the PyTorch 2.6+ default) would
    reject our nested optimizer state on some version combos. load_checkpoint
    passes weights_only=False explicitly — verify it returns the dict."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert isinstance(ckpt, dict)


# ---------------------------------------------------------------------------
# G219 — HF-init checkpoint with no 'optimizer' key
# ---------------------------------------------------------------------------

def test_resume_with_no_optimizer_key_does_not_raise():
    """G219: scripts/load_pretrained.py emits {state_dict, config, step}
    without 'optimizer'. The resume path must tolerate the missing key.
    Simulate the path: load checkpoint, build fresh optimizer, only call
    load_state_dict if the key is present."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hf_init.pt")
        # Save like load_pretrained does — no optimizer key.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": dataclasses.asdict(cfg),
                "step": 0,
            },
            path,
        )
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        # Caller-side pattern from G219:
        cfg2 = TitansConfig(**ckpt["config"])
        model2 = TitansMAGGPT2(cfg2)
        model2.load_state_dict(ckpt["state_dict"])
        opt2 = build_optimizer(model2)
        if "optimizer" in ckpt:
            opt2.load_state_dict(ckpt["optimizer"])
        # No exception — the missing-optimizer branch is allowed.
        assert opt2 is not None


# ---------------------------------------------------------------------------
# G221 — resume sequence ends in train()
# ---------------------------------------------------------------------------

def test_resume_ends_with_model_train_mode():
    """G221: explicit model.train() at end of resume protects against any
    intervening eval-mode code (smoke perplexity, sample generation).
    The resume sequence is caller-owned, but verify the pattern works."""
    cfg, model, opt = _tiny_setup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, opt, step=0, config=cfg)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        cfg2 = TitansConfig(**ckpt["config"])
        model2 = TitansMAGGPT2(cfg2)
        model2.load_state_dict(ckpt["state_dict"])
        model2.eval()  # simulate a smoke-test eval that came before final train()
        model2.train()  # the G221 defensive call
        assert model2.training


# ---------------------------------------------------------------------------
# G172 — compute_nmm_norm
# ---------------------------------------------------------------------------

def test_compute_nmm_norm_returns_None_when_states_is_None():
    assert compute_nmm_norm(None) is None


def test_compute_nmm_norm_returns_one_float_per_layer():
    cfg, model, opt = _tiny_setup()
    states = [block.nmm.init_state(2, torch.device("cpu")) for block in model.blocks]
    norms = compute_nmm_norm(states)
    assert isinstance(norms, list)
    assert len(norms) == cfg.n_layer
    for n in norms:
        assert isinstance(n, float)
        assert n > 0  # at init the Xavier-uniform weights have non-zero Frobenius


def test_compute_nmm_norm_increases_when_M_is_larger():
    """Sanity: doubling all entries of M should ~double the reported norm."""
    cfg, model, opt = _tiny_setup()
    states = [block.nmm.init_state(2, torch.device("cpu")) for block in model.blocks]
    norms1 = compute_nmm_norm(states)
    # Scale all M entries by 2.0.
    for M, S, _ in states:
        for k in M:
            M[k] = M[k] * 2.0
    norms2 = compute_nmm_norm(states)
    for n1, n2 in zip(norms1, norms2):
        assert abs(n2 / n1 - 2.0) < 1e-5


def test_run_training_continues_across_epochs_on_single_gpu():
    """Regression: when `max_steps > batches_per_epoch`, run_training MUST
    rebuild the iterator and continue into the next epoch — not silently
    early-stop after one pass. The docstring promises this; the single-GPU
    partial-cycle branch previously broke out of the outer loop, truncating
    a 5000-step run to ~one epoch with no checkpoint (the bug observed on
    vanilla GPT-2 SQuAD training).

    We exercise the path by giving the loader fewer batches than
    `max_steps * accum_steps` and asserting that the training loop sees
    every optimizer step.
    """
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        nmm_layer_indices=[],  # vanilla — exercises the same code path
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)

    # Loader yields exactly 3 micro-batches per pass. With accum_steps=2,
    # that's 1 full cycle + 1 partial cycle per epoch (the single-GPU
    # partial-cycle branch hits the break we just fixed). max_steps=4
    # therefore requires AT LEAST 3 epoch restarts to complete.
    class _SmallLoader:
        def __iter__(self):
            for _ in range(3):
                yield _fake_batch(cfg, B=1, T=4)

    loader = _SmallLoader()

    # Snapshot optimizer step count to verify we actually advanced.
    from train import run_training
    run_training(
        model=model, optimizer=optimizer, loader=loader,
        device=torch.device("cpu"),
        max_steps=4, warmup_steps=0, accum_steps=2,
        log_every=1, save_every=None, save_dir=None,
        config=cfg, autocast_dtype=None,
        show_progress=False,
    )

    # The first param's step counter is the most reliable signal: the
    # optimizer only steps on completed cycles. We REQUIRE max_steps
    # optimizer steps — anything less means epoch-restart didn't kick in.
    p = next(iter(model.parameters()))
    state = optimizer.state.get(p, {})
    n_steps = int(state.get("step", torch.tensor(0)).item()) if state else 0
    assert n_steps == 4, (
        f"Expected 4 optimizer steps (max_steps), got {n_steps}. "
        f"run_training likely truncated to one epoch — the partial-cycle "
        f"break has regressed."
    )


# ---------------------------------------------------------------------------
# run_training start_step / resume
# ---------------------------------------------------------------------------


def test_run_training_respects_start_step():
    """`start_step=N` makes the training loop begin counting from N instead
    of 0. With max_steps=N+K we should see exactly K optimizer steps,
    leaving the final step counter at N+K.

    This is the load-bearing piece of resume: without it, --resume-from
    would re-do every step from 0, undoing the previous run."""
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        nmm_layer_indices=[],
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)

    class _Loader:
        def __iter__(self):
            for _ in range(20):
                yield _fake_batch(cfg, B=1, T=4)
    loader = _Loader()

    from train import run_training
    run_training(
        model=model, optimizer=optimizer, loader=loader,
        device=torch.device("cpu"),
        max_steps=5, warmup_steps=0, accum_steps=1,
        log_every=1, save_every=None, save_dir=None,
        config=cfg, autocast_dtype=None,
        show_progress=False,
        start_step=3,  # resume from step 3
    )

    # Two optimizer steps should have happened (3 -> 4 -> 5).
    p = next(iter(model.parameters()))
    state = optimizer.state.get(p, {})
    n_steps = int(state.get("step", torch.tensor(0)).item()) if state else 0
    assert n_steps == 2, (
        f"Expected 2 optimizer steps (max_steps=5, start_step=3), got "
        f"{n_steps}. run_training is ignoring start_step."
    )


def test_run_training_rejects_negative_start_step():
    """start_step < 0 is nonsense (no prior step counter would be negative).
    Reject loudly instead of silently coercing — catches the typo
    --resume-from=path → start_step=-1 from a bug."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False, nmm_layer_indices=[],
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)
    from train import run_training
    with pytest.raises(ValueError, match="start_step must be >= 0"):
        run_training(
            model=model, optimizer=optimizer, loader=iter([]),
            device=torch.device("cpu"),
            max_steps=5, warmup_steps=0, accum_steps=1,
            log_every=1, save_every=None, save_dir=None,
            config=cfg, autocast_dtype=None,
            show_progress=False,
            start_step=-1,
        )


def test_run_training_rejects_start_step_at_or_past_max_steps():
    """If `start_step >= max_steps` there's nothing to train — the loop
    would no-op and the user would silently get a "done" with zero work.
    Common cause: forgot to bump --max-steps when resuming. Fail loud."""
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False, nmm_layer_indices=[],
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)
    from train import run_training
    with pytest.raises(ValueError, match=r"start_step.*>=.*max_steps"):
        run_training(
            model=model, optimizer=optimizer, loader=iter([]),
            device=torch.device("cpu"),
            max_steps=5, warmup_steps=0, accum_steps=1,
            log_every=1, save_every=None, save_dir=None,
            config=cfg, autocast_dtype=None,
            show_progress=False,
            start_step=5,  # equal to max_steps
        )


def test_resume_round_trip_via_save_and_load(tmp_path):
    """End-to-end resume integration: a continuous N-step run produces the
    same final params as a (save at N//2) + (resume to N) run.

    Both paths use the SAME max_steps (so the cosine LR schedule is
    identical across cycles) and the SAME data trajectory (a constant-
    batch loader removes loader-state-on-resume as a confounder). With
    those held fixed, resume MUST be bit-equivalent — anything else
    surfaces a real bug in start_step plumbing, state-dict / optimizer
    loading, or from_dict's reconstruction.
    """
    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=8, vocab_size=16,
        block_size=16, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False, nmm_layer_indices=[],
    )

    # Constant-batch loader: every iteration sees the same input. Removes
    # the "loader restarts at batch 0 on resume" confounder so we can
    # directly test the resume-state mechanism.
    torch.manual_seed(7)
    const_batch = _fake_batch(cfg, B=1, T=4)
    class _ConstLoader:
        def __iter__(self):
            while True:
                yield const_batch

    from train import run_training, load_checkpoint

    # === Continuous run: max_steps=4, save_every=2 so we capture step 2 ===
    torch.manual_seed(0)
    model_cont = TitansMAGGPT2(cfg)
    opt_cont = build_optimizer(model_cont)
    save_dir = tmp_path / "cont_ckpts"
    run_training(
        model=model_cont, optimizer=opt_cont, loader=_ConstLoader(),
        device=torch.device("cpu"),
        max_steps=4, warmup_steps=0, accum_steps=1,
        log_every=1, save_every=2, save_dir=str(save_dir), keep_last_n=10,
        config=cfg, autocast_dtype=None,
        show_progress=False,
    )
    cont_final_params = {
        n: p.detach().clone() for n, p in model_cont.named_parameters()
    }

    # === Resume from the step=2 checkpoint, run through step 4 ===
    # The saved `step=2` is the counter value at the save call site,
    # which fires BEFORE the post-cycle `step += 1` — so the model has
    # actually completed 3 cycles. To resume without redoing cycle 3,
    # start_step = saved_step + 1 = 3. This matches the off-by-one fix
    # in finetune.py / train.py's resume path.
    ckpt = load_checkpoint(save_dir / "step_0000002.pt",
                           device=torch.device("cpu"))
    cfg_loaded = TitansConfig.from_dict(ckpt["config"])
    # Different init seed → resume must override via load_state_dict.
    torch.manual_seed(99)
    model_b = TitansMAGGPT2(cfg_loaded)
    model_b.load_state_dict(ckpt["state_dict"])
    opt_b = build_optimizer(model_b)
    opt_b.load_state_dict(ckpt["optimizer"])
    run_training(
        model=model_b, optimizer=opt_b, loader=_ConstLoader(),
        device=torch.device("cpu"),
        max_steps=4, warmup_steps=0, accum_steps=1,
        log_every=1, save_every=None, save_dir=None,
        config=cfg_loaded, autocast_dtype=None,
        show_progress=False,
        start_step=int(ckpt["step"]) + 1,
    )
    resume_params = {
        n: p.detach().clone() for n, p in model_b.named_parameters()
    }

    # Bit-equivalent (within fp32 round-off). If this drifts, the resume
    # state-restoration is incomplete.
    for name in cont_final_params:
        assert torch.allclose(
            cont_final_params[name], resume_params[name],
            atol=1e-6, rtol=1e-6,
        ), (
            f"Resume diverged on param {name}: max_abs="
            f"{(cont_final_params[name] - resume_params[name]).abs().max().item():.2e}"
        )


def test_run_training_handles_all_none_nmm_norms():
    """Regression: under --vanilla-gpt2 every block is a PlainGPT2Block, so
    `compute_nmm_norm(states)` returns `[None, None, ...]` — a list that's
    truthy as a Python object but whose elements can't be summed. The
    per-step aggregator in `run_training` must filter Nones BEFORE summing,
    otherwise step 0 of vanilla training raises
    `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.

    This test exercises the smallest possible vanilla config end-to-end —
    1 step, batch=1, T=4 — so a CI run catches the regression in seconds.
    """
    cfg = TitansConfig(
        n_layer=2, n_head=2, n_embd=8, vocab_size=32,
        block_size=64, chunk_size=4, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
        nmm_layer_indices=[],  # vanilla mode — every block is PlainGPT2Block
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)

    # Fake loader yielding one (idx, doc_boundaries) pair.
    class _OneShotLoader:
        def __iter__(self):
            yield _fake_batch(cfg, B=1, T=4)
    loader = _OneShotLoader()

    # Import locally to avoid contaminating test_checkpoint's top-level imports.
    from train import run_training
    # Single step must complete without raising. `show_progress=False` keeps
    # tqdm quiet under pytest capture.
    run_training(
        model=model, optimizer=optimizer, loader=loader,
        device=torch.device("cpu"),
        max_steps=1, warmup_steps=0, accum_steps=1,
        log_every=1, save_every=None, save_dir=None,
        config=cfg, autocast_dtype=None,
        show_progress=False,
    )


# ---------------------------------------------------------------------------
# save_checkpoint_rotating + prune_old_checkpoints
# ---------------------------------------------------------------------------


def _save(tmp_path, step, cfg=None, model=None, opt=None, keep_last_n=3):
    """Save one rotated checkpoint with the tiny setup. Returns the
    `(saved_path, save_dir)` pair so tests don't have to recompute paths."""
    if cfg is None:
        cfg, model, opt = _tiny_setup()
    saved = save_checkpoint_rotating(
        tmp_path, model, opt, step=step, config=cfg, keep_last_n=keep_last_n,
    )
    return saved, Path(tmp_path)


def test_save_rotating_creates_step_file_with_padded_name(tmp_path):
    """Step filename is zero-padded to 7 digits so lex order == numeric
    order even when steps span 6 -> 7 digit widths (cf. 9 vs 10 sorting
    as '10' < '9' in lex)."""
    saved, _ = _save(tmp_path, step=42)
    assert saved.name == "step_0000042.pt"
    assert saved.exists()


def test_save_rotating_writes_latest_pt_as_copy_of_step(tmp_path):
    """latest.pt must be a real file (not a symlink — Windows / cross-FS
    safety) and identical content to the just-saved step file."""
    saved, save_dir = _save(tmp_path, step=5)
    latest = save_dir / LATEST_CKPT_NAME
    assert latest.exists() and latest.is_file()
    assert not latest.is_symlink()
    assert latest.read_bytes() == saved.read_bytes()


def test_save_rotating_latest_pt_tracks_most_recent_step(tmp_path):
    """After saves at steps {1, 2, 3}, latest.pt's `step` field must be 3."""
    cfg, model, opt = _tiny_setup()
    for s in (1, 2, 3):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=10,
        )
    ckpt = load_checkpoint(tmp_path / LATEST_CKPT_NAME, torch.device("cpu"))
    assert ckpt["step"] == 3


def test_save_rotating_prunes_oldest_beyond_keep_last_n(tmp_path):
    """keep_last_n=2 + 5 saves -> only the last 2 step files remain."""
    cfg, model, opt = _tiny_setup()
    for s in range(5):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=2,
        )
    step_files = list_step_checkpoints(tmp_path)
    steps = [s for s, _p in step_files]
    assert steps == [3, 4], (
        f"expected step_0000003.pt + step_0000004.pt only; got {steps}"
    )
    # latest.pt still present and unaffected by pruning.
    assert (tmp_path / LATEST_CKPT_NAME).exists()


def test_save_rotating_keep_last_n_none_disables_pruning(tmp_path):
    """keep_last_n=None: every save sticks around. This is the explicit
    opt-out for users who want full step history (e.g., to plot loss
    curves from intermediate ckpts)."""
    cfg, model, opt = _tiny_setup()
    for s in range(4):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=None,
        )
    steps = [s for s, _p in list_step_checkpoints(tmp_path)]
    assert steps == [0, 1, 2, 3]


def test_save_rotating_keep_last_n_zero_also_disables_pruning(tmp_path):
    """`<= 0` is the same as None — convenient for users passing
    `--keep-last-n 0` on the CLI."""
    cfg, model, opt = _tiny_setup()
    for s in range(4):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=0,
        )
    assert len(list_step_checkpoints(tmp_path)) == 4


def test_save_rotating_does_not_touch_unrelated_files(tmp_path):
    """Pruning only deletes files matching the `step_NNNNNNN.pt` pattern.
    User-placed files (notes, configs, logs) and `latest.pt` are left
    alone even when keep_last_n=1 triggers aggressive pruning."""
    cfg, model, opt = _tiny_setup()
    # Drop a user file BEFORE any saves.
    user_note = tmp_path / "README.txt"
    user_note.write_text("don't touch me")
    # A masquerading file that LOOKS like a step ckpt but isn't (wrong digit
    # width) — must not be touched either (regex anchors are strict).
    masquerade = tmp_path / "step_42.pt"  # 2 digits, not 7
    masquerade.write_text("not really a step ckpt")

    for s in range(3):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=1,
        )
    # Pruning to 1 keeps step_0000002.pt only.
    steps = [s for s, _p in list_step_checkpoints(tmp_path)]
    assert steps == [2]
    assert user_note.exists() and user_note.read_text() == "don't touch me"
    assert masquerade.exists()
    assert (tmp_path / LATEST_CKPT_NAME).exists()


def test_save_rotating_creates_save_dir_if_missing(tmp_path):
    """Auto-mkdir so the CLI default `ckpts/finetune` works in fresh repos
    without the user having to `mkdir -p ckpts/finetune` first."""
    target = tmp_path / "fresh" / "deep" / "ckpts"
    assert not target.exists()
    cfg, model, opt = _tiny_setup()
    save_checkpoint_rotating(target, model, opt, step=0, config=cfg)
    assert target.is_dir()
    assert (target / "step_0000000.pt").exists()


def test_save_rotating_step_files_round_trip_with_load_checkpoint(tmp_path):
    """A rotated step file must load through `load_checkpoint` just like
    a regular `save_checkpoint` file would — same dict layout."""
    cfg, model, opt = _tiny_setup()
    save_checkpoint_rotating(tmp_path, model, opt, step=7, config=cfg)
    ckpt = load_checkpoint(tmp_path / "step_0000007.pt", torch.device("cpu"))
    assert ckpt["step"] == 7
    assert set(ckpt["state_dict"].keys()) == set(model.state_dict().keys())


def test_list_step_checkpoints_returns_sorted_ascending(tmp_path):
    """Out-of-order saves still come back sorted by numeric step.
    Important for `prune_old_checkpoints` to identify the truly oldest."""
    cfg, model, opt = _tiny_setup()
    for s in (5, 1, 10, 3):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=None,
        )
    steps = [s for s, _p in list_step_checkpoints(tmp_path)]
    assert steps == [1, 3, 5, 10]


def test_list_step_checkpoints_empty_dir_returns_empty_list(tmp_path):
    assert list_step_checkpoints(tmp_path) == []


def test_list_step_checkpoints_missing_dir_returns_empty_list(tmp_path):
    """Defensive: passing a path that doesn't exist (e.g., during a
    pre-flight check before the first save) returns [], not raises."""
    assert list_step_checkpoints(tmp_path / "does-not-exist") == []


def test_prune_old_checkpoints_returns_paths_it_deleted(tmp_path):
    """The function's return value is contractual — tests + the CLI rely
    on it to log what was removed."""
    cfg, model, opt = _tiny_setup()
    for s in range(4):
        save_checkpoint_rotating(
            tmp_path, model, opt, step=s, config=cfg, keep_last_n=None,
        )
    deleted = prune_old_checkpoints(tmp_path, keep_last_n=2)
    deleted_names = sorted(p.name for p in deleted)
    assert deleted_names == ["step_0000000.pt", "step_0000001.pt"]
    # The kept files are still present.
    remaining = [s for s, _p in list_step_checkpoints(tmp_path)]
    assert remaining == [2, 3]


def test_save_rotating_overwrite_same_step_does_not_duplicate(tmp_path):
    """Re-saving step N (e.g., a retry after a NaN-skip cycle in
    run_training) overwrites the existing step_N file rather than
    creating a duplicate. The file count stays the same; latest.pt
    refreshes."""
    cfg, model, opt = _tiny_setup()
    save_checkpoint_rotating(tmp_path, model, opt, step=1, config=cfg)
    save_checkpoint_rotating(tmp_path, model, opt, step=1, config=cfg)
    assert len(list_step_checkpoints(tmp_path)) == 1
