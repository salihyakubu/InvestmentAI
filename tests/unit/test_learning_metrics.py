"""Learning metrics: honest by construction, and able to detect learning.

Both directions, as always: planted improvement across eras must show up,
flat noise must not; the zero-return guard (the PR #59 lesson) and the
gated-only calibration restriction are pinned structurally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.continuous_learning.learning_metrics import (
    MIN_DAILY_OBSERVATIONS,
    ResolvedPrediction,
    build_report,
    calibration_bins,
    daily_quality,
    era_boundaries,
    era_summaries,
)

_BASE = datetime(2026, 7, 20, tzinfo=UTC)


def _rows(
    n_days: int,
    per_day: int = 120,
    ic_strength: float = 0.0,
    seed: int = 0,
    start: datetime = _BASE,
    p_flat: float = 0.5,
) -> list[ResolvedPrediction]:
    rng = np.random.default_rng(seed)
    rows: list[ResolvedPrediction] = []
    for d in range(n_days):
        for i in range(per_day):
            expected = rng.normal(0.0014, 0.0011)
            noise = rng.normal(0, 0.0018)
            actual = ic_strength * (expected - 0.0014) + noise
            actually_flat = rng.random() < p_flat
            rows.append(
                ResolvedPrediction(
                    predicted_at=start + timedelta(days=d, minutes=5 * i),
                    symbol="BTC/USDT",
                    expected_return=float(expected),
                    confidence=float(np.clip(p_flat + rng.normal(0, 0.05), 0, 1)),
                    direction="flat",
                    actual_return=0.0 if actually_flat and rng.random() < 0.1
                    else float(actual),
                    actual_direction="flat" if actually_flat else "long",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Daily quality
# ---------------------------------------------------------------------------


def test_thin_days_report_none_not_zero() -> None:
    rows = _rows(1, per_day=MIN_DAILY_OBSERVATIONS - 1)
    day = daily_quality(rows)[0]
    assert day.ic is None
    assert day.brier_flat is None
    assert day.n == MIN_DAILY_OBSERVATIONS - 1


def test_a_real_signal_produces_positive_daily_ic() -> None:
    rows = _rows(3, ic_strength=2.0, seed=1)
    days = daily_quality(rows)
    ics = [d.ic for d in days if d.ic is not None]
    assert len(ics) == 3
    assert min(ics) > 0.3


def test_noise_produces_ic_near_zero() -> None:
    rows = _rows(3, ic_strength=0.0, seed=2)
    ics = [d.ic for d in daily_quality(rows) if d.ic is not None]
    assert all(abs(ic) < 0.2 for ic in ics)


def test_zero_return_outcomes_are_excluded_from_agreement() -> None:
    """The PR #59 guard, structural: a day whose non-flat outcomes are
    perfectly predicted scores ~1.0 agreement even when a third of the bars
    did not move at all."""
    rows: list[ResolvedPrediction] = []
    rng = np.random.default_rng(3)
    for i in range(150):
        expected = float(rng.normal(0, 0.001))
        if i % 3 == 0:
            actual, direction = 0.0, "flat"  # unchanged bar
        else:
            actual, direction = (0.002 if expected > 0 else -0.002), "long"
        rows.append(
            ResolvedPrediction(
                predicted_at=_BASE + timedelta(minutes=5 * i), symbol="X",
                expected_return=expected, confidence=0.5, direction="flat",
                actual_return=actual, actual_direction=direction,
            )
        )
    day = daily_quality(rows)[0]
    assert day.sign_agreement is not None
    assert day.sign_agreement > 0.9  # zeros did not dilute it toward 0.66


def test_abstention_rate_is_reported_even_on_thin_days() -> None:
    rows = _rows(1, per_day=10)
    assert daily_quality(rows)[0].abstention_rate == 1.0


# ---------------------------------------------------------------------------
# Eras
# ---------------------------------------------------------------------------


def test_same_day_promotions_form_one_era_boundary() -> None:
    stamps = [
        datetime(2026, 7, 28, 9, 0, tzinfo=UTC),
        datetime(2026, 7, 28, 9, 5, tzinfo=UTC),  # ensemble refresh, same day
        datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    ]
    bounds = era_boundaries(stamps)
    assert [b[1].isoformat() for b in bounds] == ["2026-07-24", "2026-07-28"]


def test_eras_detect_planted_learning() -> None:
    """Era 2's live IC must beat era 1's when the signal genuinely improved
    after the promotion -- the metric the whole module exists for."""
    before = _rows(4, ic_strength=0.0, seed=4, start=_BASE)
    after = _rows(4, ic_strength=2.0, seed=5, start=_BASE + timedelta(days=4))
    promos = [_BASE, _BASE + timedelta(days=4)]
    eras = era_summaries(daily_quality(before + after), era_boundaries(promos))
    assert len(eras) == 2
    assert eras[0].mean_ic is not None and abs(eras[0].mean_ic) < 0.2
    assert eras[1].mean_ic is not None and eras[1].mean_ic > 0.3
    assert eras[1].ic_t_stat is not None and eras[1].ic_t_stat > 2


def test_flat_noise_across_eras_shows_no_improvement() -> None:
    """Churning, not learning, must be visible as such."""
    a = _rows(4, ic_strength=0.0, seed=6, start=_BASE)
    b = _rows(4, ic_strength=0.0, seed=7, start=_BASE + timedelta(days=4))
    eras = era_summaries(
        daily_quality(a + b), era_boundaries([_BASE, _BASE + timedelta(days=4)])
    )
    for era in eras:
        assert era.mean_ic is not None
        assert abs(era.mean_ic) < 0.2


def test_open_final_era_has_no_end() -> None:
    eras = era_summaries(daily_quality(_rows(2)), era_boundaries([_BASE]))
    assert eras[-1].end is None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_uses_only_gated_predictions() -> None:
    """Suppressed directional views carry a confidence that is NOT p(flat);
    they must not pollute the reliability table."""
    rows = _rows(2, seed=8)
    # Add directional (non-gated) rows with absurd confidence; bins must not move.
    polluted = rows + [
        ResolvedPrediction(
            predicted_at=_BASE + timedelta(days=1, minutes=i), symbol="X",
            expected_return=0.001, confidence=0.99, direction="long",
            actual_return=0.001, actual_direction="long",
        )
        for i in range(200)
    ]
    assert calibration_bins(rows) == calibration_bins(polluted)


def test_calibrated_forecasts_show_small_gaps() -> None:
    """When p(flat) is drawn honestly from the true flat rate, every bin's
    realized frequency must sit near its forecast."""
    rng = np.random.default_rng(9)
    rows: list[ResolvedPrediction] = []
    for i in range(4000):
        p = float(rng.uniform(0.35, 0.75))
        rows.append(
            ResolvedPrediction(
                predicted_at=_BASE + timedelta(minutes=5 * i), symbol="X",
                expected_return=0.001, confidence=p, direction="flat",
                actual_return=0.0,
                actual_direction="flat" if rng.random() < p else "long",
            )
        )
    for bin_row in calibration_bins(rows):
        assert abs(bin_row["gap"]) < 0.06


def test_report_carries_the_method_notes() -> None:
    report = build_report(_rows(2), [_BASE])
    assert any("PR #59" in note for note in report.notes)
    assert any("p(flat)" in note for note in report.notes)
    assert report.eras and report.daily


@pytest.mark.parametrize("per_day", [MIN_DAILY_OBSERVATIONS, 200])
def test_daily_counts_are_exact(per_day: int) -> None:
    days = daily_quality(_rows(2, per_day=per_day))
    assert [d.n for d in days] == [per_day, per_day]
