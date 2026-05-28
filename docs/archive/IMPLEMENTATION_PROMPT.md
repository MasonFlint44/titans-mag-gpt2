# Implementation Prompt — TITANS MAG + GPT-2

> Paste this entire file as your opening message to Claude in a fresh session
> opened in `/home/mason/git/titans-mag-gpt2/`. The prompt assumes Claude has
> access to the working directory and Bash/Read/Edit/Write tools.

---

You are implementing **TITANS MAG (Memory as a Gate)** on top of **GPT-2** in
PyTorch, from scratch. The full design, plan, tests, and audit history already
exist in this directory — your job is to translate them into working code.
Treat this as a serious, production-grade implementation. The bar is "totally
correct and complete," not "passes a smoke test."

## 0. Scope

**In scope:** implement the code, write tests, run the test suite, fix bugs.

**Out of scope:** running the headline experiments (`EXPERIMENTS.md` §2–4),
full-scale training runs, hyperparameter sweeps, ablation campaigns,
performance benchmarking on real workloads. Those happen in a separate phase
once the code is ready. You may run the *tests* in `TEST_PLAN.md` (which
include short behavior tests like 200-step single-batch overfit and synthetic
KV memorization) because those are minutes-scale regression coverage, not
experiments. Do not start a real training run.

If a task seems to require running an experiment to "verify" something, that
verification probably belongs in a test in `TEST_PLAN.md` — write it there.

## 1. Hard constraints (read these first; do not violate)

1. **Do NOT use `titans-pytorch` (lucidrains) as a dependency.** Reference its
   source for clarification if you want, but every line of code in this repo
   must be implemented from scratch.
2. **Do not skip or weaken any guard documented in PLAN.md or GAP_HISTORY.md.**
   Every one of those guards was added because something silently broke
   without it. If a guard seems redundant, it isn't — read the linked G-number
   in `GAP_HISTORY.md` before considering whether to remove it.
3. **Validation uses `raise ValueError`, never `assert`.** `python -O` strips
   asserts. (G190, G220, G223.)
4. **Newton-Schulz 5-step runs in fp32 with `autocast(enabled=False)`.** Bare
   `.float()` inside an ambient bf16 autocast is silently undone (G226). This
   is the single most failure-prone subroutine; treat it as load-bearing.
5. **Backward + clip + step run in fp32 under bf16 autocast** (G159). Exit the
   autocast region before `loss.backward()`.
6. **`base_lrs` come from code-level constants, never from
   `optimizer.param_groups[i]['lr']`** (G162). Otherwise LR deflates every
   resume.
7. **DDP gradient accumulation uses `model.no_sync()` for non-final
   micro-batches** (G200). The partial-cycle skip condition is
   `(batch is None) and (accum_i > 0)`, **not** `accum_i < ACCUM_STEPS - 1`
   (G222).
8. **NMM state is not in the checkpoint.** It's per-sequence recurrent state;
   re-init from `memory_mlp.W*.weight` on resume.
9. **The entire training loop is wrapped in `try: ... finally:
   destroy_process_group()`** (G225) with consistent 4-space indentation
   throughout the try body (G227).

If you find yourself wanting to violate one of these, stop and ask.

## 2. The documents you have

You are working with a complete pre-implementation design corpus. Read in this
order before writing any code:

| Order | File | Purpose |
|---|---|---|
| 1 | `README.md` | Front door — what this is |
| 2 | `ARCHITECTURE.md` | The model, equations, design decisions |
| 3 | `diagrams/architecture.mmd` | Visual companion to ARCHITECTURE.md |
| 4 | `ROADMAP.md` | Phase-by-phase task list + critical invariants |
| 5 | `GLOSSARY.md` | Skim once so terminology lands |
| 6 | `CONFIG_REFERENCE.md` | Every config knob and its boundary behavior |
| 7 | `PLAN.md` | Detailed code sketches per task. **The authoritative reference.** Read the relevant §N.M before implementing task N.M — **do not read cover-to-cover; it is 4000+ lines and reading it whole will waste context for no benefit** |
| 8 | `TEST_PLAN.md` | What tests to write per component |
| 9 | `RUNBOOK.md` | Failure-mode recovery — skim, but consult when things break |
| 10 | `EXPERIMENTS.md` | The validation gates that define "done" |

Diagrams to consult as you build the relevant component:

- `diagrams/nmm_state_lifecycle.mmd` — when implementing Phase 1.7–1.9 and 4.2
- `diagrams/newton_schulz.mmd` — when implementing Phase 1.6
- `diagrams/training_sequence.mmd` — when implementing Phase 4
- `diagrams/data_pipeline.mmd` — when implementing Phase 3
- `diagrams/ddp_no_sync.mmd` — when implementing Phase 4.5
- `diagrams/inference_sequence.mmd` — when implementing Phase 5
- `diagrams/phase_dag.mmd` — when deciding what to work on next

`GAP_HISTORY.md` is reference-only. Do not read it cover-to-cover; consult by
G-number when a PLAN.md snippet cites one you don't understand.

### When the docs disagree

These documents were written over many sessions and some drift is expected.
If you find a contradiction between them, do not silently pick one — flag it
in your status update and use this tiebreaker order:

1. **`PLAN.md`** — authoritative for code-level details (function signatures,
   shapes, control flow, specific guards).
2. **`ARCHITECTURE.md`** — authoritative for design decisions (equations,
   variant choices, why the gate has its specific form).
3. **`TEST_PLAN.md`** — authoritative for test specifications (what to assert,
   what fixture to use, what marker to apply).
4. **The diagrams** — visual companions, occasionally simplified. If a diagram
   contradicts a doc, trust the doc.
5. **`ROADMAP.md` / `README.md` / `CONFIG_REFERENCE.md` / `GLOSSARY.md` /
   `RUNBOOK.md`** — derived summaries; trust the authoritative sources above.
6. **The paper** — only when even the above don't resolve the question.

When you find a discrepancy, log it as a gap (new G-number) noting which doc
was wrong and which you followed. Do not edit the docs to "fix" the
discrepancy on your own — flag and ask.

## 3. Implementation order

Follow `ROADMAP.md` §"Recommended implementation order" (the 13 numbered
checkpoints). Summary:

1. Phase 0 end-to-end (config + skeleton)
2. Phase 1.1–1.4 (NMM building blocks)
3. Phase 1.5–1.6 (gradient + Newton-Schulz)
4. Phase 1.7–1.9 (step + chunked forward + state mgmt) — **overfit a single k→v before moving on**
5. Phase 2.0–2.2 (attention + persistent tokens + ln_nmm)
6. Phase 2.3–2.5 (MAG gate + block + full model) — **confirm `out_scale=0 → o = y_attn` exactly**
7. Phase 2.6 (GPT-2 weight loading) — **HARD GATE: HF logit parity < 1e-4**
8. Phase 3 (data pipeline) — **verify sub-stream continuity by printing**
9. Phase 4.1–4.3 (optimizer + train_step + schedule) — **100-step single-batch overfit**
10. Phase 4.4 (fine-tune entry)
11. Phase 5 (generate + perplexity within 5% of HF when NMM zeroed)
12. Phase 4.5 (multi-GPU DDP with K=2 grad accum) — **partial-cycle skip never fires unexpectedly**
13. Phase 6 (optional scan) — **scan vs sequential < 5% relative error**

Some tasks within a phase can be parallelized (see `diagrams/phase_dag.mmd`),
but the recommended sequential order above is calibrated to surface bugs at
the right time. Do not jump ahead to make progress feel faster.

## 4. Per-task workflow

For each task N.M:

1. **Read `PLAN.md` §N.M end-to-end.** If it cites a G-number, look up that
   entry in `GAP_HISTORY.md`. Do not start coding until you understand every
   safeguard listed.
2. **Read the relevant ROADMAP section** for the ⚠ gotchas summary.
3. **Sketch the code in your head** — what files change, what interfaces, what
   shapes pass through.
4. **Implement** in the file structure documented in
   `ARCHITECTURE.md` §"File Structure". Match the structure exactly.
5. **Write tests as you go.** Find the matching row(s) in `TEST_PLAN.md` for
   this task and implement those tests. Do not move to the next task with
   failing or absent tests for the current one. Use the pytest markers and
   fixtures defined in TEST_PLAN.md §3.
6. **Run the tests.** All unit tests for the current task must pass. Run
   `pytest tests/unit/ -x` after every task to confirm nothing earlier
   regressed.
7. **If anything surprises you, log it.** See §6 below on gap logging.
8. **Update PLAN.md's "Testing Checkpoints" table** to mark the task complete
   (a `✓` next to the row).

## 5. Testing standard

- **Every component gets unit tests.** No exceptions. See TEST_PLAN.md §4–9
  for the per-component test list.
- **Integration tests** (TEST_PLAN.md §8) run after each phase wraps.
- **Parity tests** (TEST_PLAN.md §9) run after Phase 2.6 and are the hardest
  correctness signal. If the HF parity test fails, do NOT proceed to Phase 3;
  fix the weight loading.
- **Behavior tests** (TEST_PLAN.md §10) — overfit, KV memorization,
  needle-in-haystack, long-context loss curve. Implement these after Phase 4.
- **DDP tests** (TEST_PLAN.md §11) after Phase 4.5. Use the `ddp` pytest
  marker; spawn 2-rank subprocesses.
- **Performance tests** (TEST_PLAN.md §12) are guardrails, not gates. Run
  before declaring done.
- **Failure-mode tests** (TEST_PLAN.md §13) — NaN injection, invalid configs,
  resource leak detection. These confirm the guards actually fire.
- **The regression matrix in TEST_PLAN.md §14 is the inverse index**: every
  G-number has a defending test. If you remove or weaken a guard, the
  corresponding test must fail — that is the contract.

Run modes:
- Local dev loop: `pytest -m "not slow and not gpu"` should complete in
  ~10 minutes.
- Pre-merge: `pytest -m "not gpu"` adds slow CPU tests.
- Nightly: everything including GPU and DDP tests.
- Release gate: nightly + behavior + perf benchmarks.

If you don't have a GPU available, skip the `gpu`-marked tests and note that
they're untested. Do NOT delete them.

## 6. Gap logging protocol

If during implementation you encounter:
- A subtle issue not covered in PLAN.md
- A failure mode the existing guards don't catch
- A spec ambiguity that required you to make a judgment call
- A discrepancy between the paper, lucidrains, and what the code naturally does

Log it. The format follows `GAP_HISTORY.md`:

```markdown
### Gnnn — <short title>

**Found:** Pass 54 / implementation phase X.Y
**Symptom:** <what fails or risks failing>
**Root cause:** <why>
**Fix:** <what you did>
**Test:** <which test in TEST_PLAN.md defends this>
**Affects:** <files / functions touched>
```

Append new entries at the end of GAP_HISTORY.md. Increment the gap counter
(current highest is G227 — yours start at G228). If the gap defends an
invariant that should be in ROADMAP.md's Critical Invariants table, also
update the table.

If you're unsure whether something rises to "gap" level, err toward logging.
The cost of an unneeded entry is one paragraph of text; the cost of a missed
gap is a silent production failure.

## 7. What "done" means

The implementation is complete when:

- [ ] Every task in ROADMAP.md Phases 0–5 is implemented (Phase 6 optional).
- [ ] Every test in TEST_PLAN.md §§4–13 passes on the appropriate hardware
      tier. `gpu`/`ddp`/`slow` tests pass where hardware is available;
      explicitly note any that were skipped and why.
- [ ] The regression matrix in TEST_PLAN.md §14 has full coverage — every
      G-number has at least one defending test that's actually green (or
      explicitly listed as not-testable like G181).
- [ ] Test infrastructure (markers, fixtures, conftest) matches
      TEST_PLAN.md §3.
- [ ] PLAN.md's "Testing Checkpoints" table is fully checked off.
- [ ] No `TODO`, `FIXME`, or `XXX` comments in committed code.
- [ ] Any new gaps you found are logged in GAP_HISTORY.md.
- [ ] `pytest` runs clean on the appropriate hardware tier.

The §1 correctness gates listed in `EXPERIMENTS.md` are *tests*, not
experiments — they correspond to specific entries in TEST_PLAN.md (§9 parity
tests, §10 behavior tests). They are covered by "every test in TEST_PLAN.md
passes" above. The §2–4 headline experiments and ablations in
`EXPERIMENTS.md` are explicitly out of scope for this implementation phase
(see §0).

Performance benchmarks (EXPERIMENTS.md §5) are also out of scope here.

## 8. When to ask vs decide

**Ask** when:
- A docstring/spec genuinely contradicts itself or the paper
- A hard constraint (§1 above) appears to require violation to proceed
- You'd need to delete an existing test to make a new one pass
- You'd need to remove a guard cited in a G-number
- A dependency version bump would be needed (lockfile changes)
- Hardware you don't have access to is required to validate (state it
  explicitly, don't claim untested code as done)

**Decide** when:
- Naming a private helper
- Choosing variable names within a function
- Code style within the existing conventions
- Test parametrization values within reasonable ranges
- Which fixture to use among several plausible ones

Default to deciding. Ask sparingly and only for genuine ambiguities.

### When stuck

If a test fails or an implementation step doesn't work, follow this protocol:

1. **Try a focused fix.** Read the failure carefully, form a specific
   hypothesis, change one thing, re-run.
2. **Try a second fix.** If the first hypothesis was wrong, form a new one.
   Do not change multiple things at once; you want to learn what fixed it.
3. **Try a third fix.** Same rules.
4. **Stop and report.** After three failed attempts, do not continue
   thrashing. Write a status update describing:
   - What you were trying to make work
   - The exact failure (error message, stack trace, unexpected output)
   - The three hypotheses you tried and what each produced
   - What you'd try next if you had more attempts, and why you're uncertain

   Then ask. I would much rather get a clean "I'm stuck on X, here's what I
   know" than a chain of speculative commits.

**Never** weaken a test, lower a tolerance, remove an assertion, delete a
guard, or wrap a failing call in `try/except` to make a failure go away. If
a test seems wrong, that is exactly the kind of thing to ask about — do not
edit it on your own initiative. The 227-entry audit history exists because
several of those guards looked redundant until they weren't; the same will
be true of any test you're tempted to weaken.

Exception: if a test fails because *you wrote the test incorrectly* during
this session (typo, wrong import, off-by-one in the test setup), fixing the
test is fine. The rule is about tests that came from `TEST_PLAN.md` or that
defend a logged gap.

## 9. Common anti-patterns to avoid

These come straight from RUNBOOK.md and GAP_HISTORY.md. Do not commit any of:

- `assert` instead of `raise ValueError` in `__post_init__` or `__init__`
- `.float()` inside an autocast region without `autocast(enabled=False)`
- `base_lrs = [g['lr'] for g in optimizer.param_groups]`
- `in-place` assignment for doc-boundary state resets (use `torch.where`)
- Running the conv per-token inside `_forward_chunk_sequential` instead of on
  the full chunk
- `nmm.forward_chunk(x_norm, state)` (2 args) — must be 3 args including
  `doc_boundaries`
- `accum_i < ACCUM_STEPS - 1` for the partial-cycle skip
- Identity per-rank seeds
- `torch.load(..., weights_only=True)` for our checkpoint format
- Constructing `vmap(grad(...))` inside the forward (must be in `__init__`)
- Name-based skip in `_apply_gpt2_init` (must be id-set based)
- Absolute import `from model.nmm import ...` in `_apply_gpt2_init` (must be
  relative)
- Pre-scaling gradient by `θ` before Newton-Schulz
- Saving `nmm_states` to checkpoint
- Forgetting to set `model.train()` after `load_state_dict`
- Treating literal `<|endoftext|>` in source text as the EOT token id
- Pre-moving data to CUDA inside the loader's worker process
- Citing G-numbers in source code comments — they belong in commit messages,
  `GAP_HISTORY.md` entries, the `ROADMAP.md` Critical Invariants table, and
  test names where TEST_PLAN.md already establishes the pattern. A bare
  `# G226` comment tells the reader where to look without telling them what
  to know

If you catch yourself about to do one of these, that's a sign you're tired —
take a break, re-read the PLAN.md section, and resume.

## 10. Output expectations per session

Each working session should produce:

1. **Working code** for one or more tasks in ROADMAP.md, in the file structure
   from ARCHITECTURE.md.
2. **Passing tests** for everything that was just implemented.
3. **A short status update** at the end of the session noting:
   - Which tasks are now complete (with a `✓` in PLAN.md's table)
   - Which tests pass / are skipped (and why skipped)
   - Any new gaps logged in GAP_HISTORY.md (with G-numbers)
   - Any open questions for me

Do not produce documentation churn. PLAN.md, ROADMAP.md, ARCHITECTURE.md, and
TEST_PLAN.md are stable artifacts; touch them only to mark tasks complete or
update the Critical Invariants table when a new G-number requires it. Do not
create new top-level docs unless asked.

### Git commit cadence

Commit per ROADMAP task once its tests pass — not per session, not per phase,
not per file. Each commit should leave the repo in a working state with the
relevant tests green.

Commit message format:

```
phase X.Y: <task name from ROADMAP>

<one or two lines on what was implemented>

Tests: <list of new test files or functions>
Gaps defended: G123, G124   (omit line if none)
New gaps: G228              (omit line if none)
```

Example:

```
phase 1.6: Newton-Schulz 5-step spectral normalization

NS5 in fp32 with autocast(enabled=False), transpose-tall guard
for [4d, d] gradients, 5-step polynomial iteration.

Tests: tests/unit/test_ns5.py (spectral bound, transpose, autocast)
Gaps defended: G198, G226
```

Rules:
- One ROADMAP task per commit. Do not bundle phases.
- Tests for the task must be in the same commit as the implementation.
- Do not commit failing tests. If a test fails, the commit waits until it
  passes (or until you've followed the "When stuck" protocol and reported).
- Do not push, force-push, rebase, amend published commits, or delete
  branches without asking.
- Do not skip pre-commit hooks (`--no-verify`). If a hook fails, fix the
  underlying issue.

## 11. Tone and pacing

- Work in small, verifiable units. After each task, run the relevant tests.
- It is far better to deliver 3 correct tasks with passing tests than 10
  half-tasks with TODOs.
- If you find yourself writing speculative code "in case we need it later,"
  delete it. YAGNI.
- The 227 audited gaps mean this design has been stress-tested by reading;
  trust it. Your job is to translate, not to redesign. If you think the
  design is wrong, ask before deviating.

## 12. Start here

Begin with:

1. Run `ls -la` to confirm you're in `/home/mason/git/titans-mag-gpt2/`.
2. Read `README.md`, `ARCHITECTURE.md`, `ROADMAP.md` in full.
3. Read `diagrams/architecture.mmd` and `diagrams/phase_dag.mmd`.
4. Skim `GLOSSARY.md` and `CONFIG_REFERENCE.md`.
5. Then begin Phase 0.1: create the repo skeleton (empty files per the file
   structure in ARCHITECTURE.md) and `pyproject.toml` with the pinned deps
   under `[project].dependencies`; run `uv lock` to generate `uv.lock`.
6. Move to Phase 0.2: implement `TitansConfig`. Run the matching unit tests in
   TEST_PLAN.md §4 to confirm validation fires correctly (including under
   `python -O`).
7. Continue through the 13-step recommended order in ROADMAP.md, stopping at
   each numbered checkpoint to verify the gate criterion.

Confirm you've read this prompt and the documents listed in §2, then proceed
with Phase 0.1. Do not produce a recap of the design — just start.

---

> **Reminder:** The papers are at arXiv:2501.00663 (Sun et al. 2025, TITANS),
> arXiv:2510.09551 (Di Nepi et al. 2025, Titans Revisited), and the GPT-2 tech
> report. Consult them only when a specific equation or claim in PLAN.md needs
> verification — they are not a substitute for the docs in this directory.
