"""Feature drift: real shifts detected, nothing fabricated below the floor.

The instrument's failure modes are epistemic: a fabricated "stable" reading
on a degenerate feature, a symbol-mix change masquerading as drift, a
reference that quietly absorbs the drifted past, or a number served from
insufficient history would all be confident nonsense. Each is pinned here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from services.continuous_learning.feature_drift import (
    MIN_VECTORS_PER_SIDE,
    PSI_SIGNIFICANT,
    build_report,
    psi,
)

_NOW = datetime(2026, 8, 20, tzinfo=UTC)


# ---------------------------------------------------------------------------
# The PSI core
# ---------------------------------------------------------------------------


def test_identical_distributions_score_near_zero() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 4000)
    rec = rng.normal(0, 1, 4000)
    assert abs(psi(ref, rec)) < 0.05


def test_a_planted_mean_shift_is_significant() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 4000)
    rec = rng.normal(2.0, 1, 4000)  # two sigmas of drift
    assert psi(ref, rec) > PSI_SIGNIFICANT


def test_recent_values_beyond_the_reference_range_still_count() -> None:
    """Outer bins are opened to +/-inf: a regime jump outside the reference
    range must register as drift, not silently vanish from the histogram."""
    rng = np.random.default_rng(2)
    ref = rng.uniform(0, 1, 2000)
    rec = rng.uniform(5, 6, 2000)  # entirely outside the reference support
    assert psi(ref, rec) > PSI_SIGNIFICANT


def test_a_degenerate_feature_is_unmeasurable_not_stable() -> None:
    ref = np.full(2000, 3.14)
    rec = np.full(2000, 3.14)
    assert psi(ref, rec) is None


# ---------------------------------------------------------------------------
# The report: per-symbol stratification, fixed reference, floors
# ---------------------------------------------------------------------------


def _rows(
    n: int,
    start: datetime,
    symbol: str = "BTC/USDT",
    digest: str = "aaaa",
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int = 7,
    dims: int = 3,
    step_minutes: int = 5,
) -> list[tuple[datetime, str, str, list[float]]]:
    rng = np.random.default_rng(seed)
    return [
        (
            start + timedelta(minutes=step_minutes * i),
            symbol,
            digest,
            list(rng.normal(loc, scale, dims)),
        )
        for i in range(n)
    ]


def test_planted_drift_in_one_feature_is_localised() -> None:
    reference = _rows(1500, _NOW - timedelta(days=10))
    recent = _rows(700, _NOW - timedelta(days=1), seed=8, step_minutes=2)
    drifted = [
        (t, s, h, [v[0], v[1] + 3.0, v[2]]) for t, s, h, v in recent
    ]  # only index 1 moves
    report = build_report(reference + drifted, _NOW)
    assert report.computable
    by_index = {f.index: f.psi for f in report.features}
    assert by_index[1] > PSI_SIGNIFICANT
    assert by_index[0] < 0.1
    assert by_index[2] < 0.1
    assert report.worst is not None and report.worst[1] == 1


def test_symbol_mix_shift_is_not_drift() -> None:
    """THE pooling regression (review 2026-08-09): two symbols whose own
    distributions never move, five orders of magnitude apart, with the row
    share flipping 70/30 -> 30/70 between windows. Pooled PSI would scream;
    per-symbol PSI must stay quiet."""
    ref_start, rec_start = _NOW - timedelta(days=10), _NOW - timedelta(days=1)
    rows = (
        _rows(1050, ref_start, "BTC/USDT", loc=60_000.0, seed=3)
        + _rows(450, ref_start, "ADA/USDT", loc=0.5, seed=4)
        + _rows(310, rec_start, "BTC/USDT", loc=60_000.0, seed=5, step_minutes=2)
        + _rows(700, rec_start, "ADA/USDT", loc=0.5, seed=6, step_minutes=2)
    )
    report = build_report(rows, _NOW)
    assert report.computable
    assert report.n_symbols_measured == 2
    assert all(f.psi is not None and f.psi < 0.1 for f in report.features)


def test_one_symbols_drift_survives_the_other_symbols_calm() -> None:
    ref_start, rec_start = _NOW - timedelta(days=10), _NOW - timedelta(days=1)
    rows = (
        _rows(800, ref_start, "BTC/USDT", loc=60_000.0, scale=100.0, seed=11)
        + _rows(800, ref_start, "ADA/USDT", loc=0.5, scale=0.01, seed=12)
        + _rows(500, rec_start, "BTC/USDT", loc=60_000.0, scale=100.0, seed=13,
                step_minutes=2)
        + _rows(500, rec_start, "ADA/USDT", loc=0.9, scale=0.01, seed=14,
                step_minutes=2)  # ADA moved 40 of its sigmas
    )
    report = build_report(rows, _NOW)
    assert report.computable
    assert report.worst is not None and report.worst[0] == "ADA/USDT"
    # Mean across 2 symbols halves the signal but must still be visible.
    assert all(f.psi is not None and f.psi > 0.5 for f in report.features)


def test_reference_is_the_early_window_not_everything_before_recent() -> None:
    """A slow walk must be judged against where the generation STARTED. Rows
    between the reference and recent windows (already drifted) must not
    dilute the baseline."""
    gen_start = _NOW - timedelta(days=20)
    early = _rows(1500, gen_start, seed=20)  # first ~5 days: loc 0
    middle = _rows(1500, gen_start + timedelta(days=9), loc=3.0, seed=21)
    recent = _rows(700, _NOW - timedelta(days=1), loc=3.0, seed=22, step_minutes=2)
    report = build_report(early + middle + recent, _NOW)
    assert report.computable
    # vs the EARLY window this is 3 sigmas of drift. An expanding reference
    # (early+middle pooled) would have scored it far lower.
    assert all(f.psi is not None and f.psi > PSI_SIGNIFICANT for f in report.features)
    assert report.reference_end == gen_start + timedelta(days=7)


def test_older_generation_rows_are_excluded_and_reference_reanchors() -> None:
    """After a pipeline change, column i means something else: rows with the
    previous hash must not enter either side, and the reference must be the
    NEW generation's own first days."""
    old = _rows(3000, _NOW - timedelta(days=30), digest="oldhash", loc=50.0)
    new = _rows(1500, _NOW - timedelta(days=10), digest="newhash", seed=9) + _rows(
        700, _NOW - timedelta(days=1), digest="newhash", seed=10, step_minutes=2
    )
    report = build_report(old + new, _NOW)
    assert report.computable
    assert report.generation_hash == "newhash"
    assert report.reference_start == _NOW - timedelta(days=10)
    # The old generation's wild shift must not have manufactured drift.
    assert all(f.psi is None or f.psi < 0.1 for f in report.features)


def test_thin_history_refuses_to_compute() -> None:
    report = build_report(_rows(300, _NOW - timedelta(days=1)), _NOW)
    assert not report.computable
    assert report.features == []
    assert "insufficient" in (report.reason or "")


def test_a_symbol_below_the_floor_is_excluded_not_padded() -> None:
    ref_start, rec_start = _NOW - timedelta(days=10), _NOW - timedelta(days=1)
    rows = (
        _rows(1000, ref_start, "BTC/USDT", seed=30)
        + _rows(500, rec_start, "BTC/USDT", seed=31, step_minutes=2)
        # DOT has plenty of reference but a starved recent side.
        + _rows(1000, ref_start, "DOT/USDT", seed=32)
        + _rows(MIN_VECTORS_PER_SIDE - 50, rec_start, "DOT/USDT", seed=33,
                step_minutes=2)
    )
    report = build_report(rows, _NOW)
    assert report.computable
    assert report.n_symbols_measured == 1
    assert report.worst is not None and report.worst[0] == "BTC/USDT"


def test_no_symbol_above_the_floor_is_a_named_refusal() -> None:
    rows = _rows(2000, _NOW - timedelta(days=10)) + _rows(
        100, _NOW - timedelta(days=1), seed=34, step_minutes=2
    )
    report = build_report(rows, _NOW)
    assert not report.computable
    assert "no symbol" in (report.reason or "")


def test_no_rows_at_all_is_a_named_absence() -> None:
    report = build_report([], _NOW)
    assert not report.computable
    assert report.generation_hash is None
    assert report.n_reference == 0 and report.n_recent == 0


def test_ragged_vectors_within_a_generation_refuse_loudly() -> None:
    rows = _rows(1500, _NOW - timedelta(days=7)) + [
        (_NOW - timedelta(days=1), "BTC/USDT", "aaaa", [1.0, 2.0])  # wrong length
    ]
    report = build_report(rows, _NOW)
    assert not report.computable
    assert "inconsistent" in (report.reason or "")
