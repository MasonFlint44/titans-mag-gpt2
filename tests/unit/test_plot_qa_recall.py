"""Tests for scripts.plot_qa_recall pure helpers.

Matplotlib-rendering is intentionally NOT tested here — that's an
environment-coupled visual concern, and the layout is best validated by
eye after a real run. We DO test the numeric helpers (bootstrap CI,
per-distance aggregation) so the chart's numbers are trustworthy.
"""

from __future__ import annotations

from scripts.plot_qa_recall import aggregate_per_distance, bootstrap_ci


def test_bootstrap_ci_empty_trials_returns_zero():
    """Empty input must not crash — return a degenerate CI so downstream
    plotting handles "no data at this bucket" without special-casing."""
    lo, hi = bootstrap_ci([])
    assert lo == 0.0 and hi == 0.0


def test_bootstrap_ci_all_correct_gives_high_ci_near_one():
    """A perfect-score bucket should have a CI bracketed near 1.0. The
    lower bound can dip below 1 due to bootstrap-resample variance, but
    must be reasonably tight (>=0.95 at n=200 trials)."""
    trials = [True] * 200
    lo, hi = bootstrap_ci(trials, n_boot=500, seed=0)
    assert hi == 1.0
    assert lo >= 0.95


def test_bootstrap_ci_brackets_the_mean():
    """Sanity: the empirical mean must fall inside the bootstrap CI. If
    it doesn't, the resampling logic is broken."""
    trials = [True, False] * 100  # mean = 0.5
    lo, hi = bootstrap_ci(trials, n_boot=500, seed=0)
    assert lo <= 0.5 <= hi


def test_bootstrap_ci_is_deterministic_with_seed():
    """Same seed → same CI across runs. Required for reproducible plots
    (the chart shouldn't shift on re-run)."""
    trials = [i % 3 == 0 for i in range(150)]
    a = bootstrap_ci(trials, n_boot=300, seed=42)
    b = bootstrap_ci(trials, n_boot=300, seed=42)
    assert a == b


def test_aggregate_per_distance_groups_by_distance():
    """The aggregator must group trials by their `distance` field. Cross-
    contamination would smear bucket accuracies and the chart would
    misrepresent the headline regime split."""
    run = {
        "results": [
            {"distance": 0, "correct": True},
            {"distance": 0, "correct": True},
            {"distance": 0, "correct": False},
            {"distance": 1024, "correct": False},
            {"distance": 1024, "correct": False},
        ],
    }
    out = aggregate_per_distance(run)
    assert set(out.keys()) == {0, 1024}
    assert out[0]["n"] == 3
    assert out[0]["accuracy"] == 2 / 3
    assert out[1024]["n"] == 2
    assert out[1024]["accuracy"] == 0.0


def test_aggregate_per_distance_handles_singleton_bucket():
    """A bucket with one trial still produces a CI (degenerate but
    non-crashing)."""
    run = {"results": [{"distance": 500, "correct": True}]}
    out = aggregate_per_distance(run)
    assert out[500]["n"] == 1
    assert out[500]["accuracy"] == 1.0
    # Bootstrap from one sample always gives [1.0, 1.0].
    assert out[500]["ci_lo"] == 1.0
    assert out[500]["ci_hi"] == 1.0
