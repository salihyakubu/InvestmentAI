"""Horizon ladder: annualisation must be right, and blending must be honest.

Two failure modes could each manufacture a strategy out of arithmetic:

* Annualising with the wrong bar size. Using the hourly constant on daily
  bars multiplies every Sharpe by sqrt(24).
* Declaring a blend successful when its components are the same bet. If they
  are perfectly correlated the blend cannot beat them, and any harness that
  says otherwise is wrong.

Both are tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.backtesting.cross_section import (
    DAYS_PER_YEAR,
    HOURS_PER_YEAR,
    cross_sectional_zscore,
    evaluate_factor,
    forward_returns,
)
from services.backtesting.horizon_ladder import (
    MIN_REBALANCES,
    blend_returns,
    evaluate_blend,
    evaluate_ladder,
    factor_correlations,
    format_ladder,
)


def _prices(n_periods: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(0, 0.01, (n_periods, n_symbols)), axis=0)


def _signal_with_edge(n_periods: int, n_symbols: int, seed: int, strength: float):
    """A cross-sectional signal that genuinely leads next-period returns."""
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.004, (n_periods, n_symbols))
    rel[1:] += score[:-1] * strength
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    return cross_sectional_zscore(score), close


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


def test_bar_size_changes_annualised_figures() -> None:
    """The same data annualised as hourly vs daily must differ by sqrt(24) in
    Sharpe -- proof the constant is actually being used, not ignored."""
    signal, close = _signal_with_edge(2_000, 50, 1, 0.01)
    forward = forward_returns(close, 1)
    hourly = evaluate_factor("f", signal, forward, 1, 0.0, 1, HOURS_PER_YEAR)
    daily = evaluate_factor("f", signal, forward, 1, 0.0, 1, DAYS_PER_YEAR)
    assert hourly.sharpe == pytest.approx(
        daily.sharpe * np.sqrt(HOURS_PER_YEAR / DAYS_PER_YEAR), rel=1e-9
    )
    assert hourly.rebalances_per_year == pytest.approx(HOURS_PER_YEAR)
    assert daily.rebalances_per_year == pytest.approx(DAYS_PER_YEAR)


def test_annual_net_scales_with_rebalance_frequency() -> None:
    """The whole point of annualising: the same per-rebalance edge is worth
    far more when collected hourly than weekly."""
    signal, close = _signal_with_edge(3_000, 50, 2, 0.01)
    fast = evaluate_factor(
        "f", signal, forward_returns(close, 1), 1, 0.0, 1, HOURS_PER_YEAR
    )
    slow = evaluate_factor(
        "f", signal, forward_returns(close, 24), 24, 0.0, 1, HOURS_PER_YEAR
    )
    assert fast.rebalances_per_year == pytest.approx(24 * slow.rebalances_per_year)
    # Per-rebalance numbers are not comparable; annualised ones are.
    assert fast.annual_net_pct != pytest.approx(fast.net_bps[0.0])


def test_annual_net_is_consistent_with_its_parts() -> None:
    signal, close = _signal_with_edge(2_000, 50, 3, 0.01)
    r = evaluate_factor(
        "f", signal, forward_returns(close, 4), 4, 2.0, 1, HOURS_PER_YEAR
    )
    assert r.annual_net_pct == pytest.approx(
        r.net_bps[2.0] * r.rebalances_per_year / 100.0, rel=1e-9
    )


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_rungs_with_too_few_rebalances_are_not_judged() -> None:
    """A 30-day hold over two years is ~24 observations. No statistic computed
    on that may reach a capital decision."""
    close = _prices(600, 40, 4)
    factors = {"f": cross_sectional_zscore(np.random.default_rng(5).normal(0, 1, (600, 40)))}
    rungs = evaluate_ladder(close, factors, (1, 200), 2.0, HOURS_PER_YEAR)
    by_horizon = {r.horizon: r for r in rungs}
    assert by_horizon[1].judged is True
    assert by_horizon[200].judged is False
    assert by_horizon[200].holdout.periods < MIN_REBALANCES


def test_ladder_scores_both_halves_for_every_cell() -> None:
    close = _prices(1_500, 40, 6)
    rng = np.random.default_rng(7)
    factors = {
        "a": cross_sectional_zscore(rng.normal(0, 1, (1_500, 40))),
        "b": cross_sectional_zscore(rng.normal(0, 1, (1_500, 40))),
    }
    rungs = evaluate_ladder(close, factors, (1, 4), 2.0, HOURS_PER_YEAR)
    assert len(rungs) == 4
    for r in rungs:
        assert r.in_sample.periods > 0
        assert r.holdout.periods > 0


def test_ladder_output_reports_how_many_rungs_survived_both_halves() -> None:
    close = _prices(1_200, 40, 8)
    factors = {"noise": cross_sectional_zscore(np.random.default_rng(9).normal(0, 1, (1_200, 40)))}
    text = format_ladder(evaluate_ladder(close, factors, (1, 6), 2.0, HOURS_PER_YEAR))
    assert "positive in BOTH" in text
    assert "enough non-overlapping periods" in text


# ---------------------------------------------------------------------------
# The blend
# ---------------------------------------------------------------------------


def test_identical_components_cannot_beat_themselves() -> None:
    """Perfectly correlated components have nothing to diversify. A harness
    that reports a win here is broken."""
    signal, close = _signal_with_edge(2_000, 50, 10, 0.01)
    factors = {"a": signal, "b": signal.copy()}
    result = evaluate_blend(close, factors, 1, 0.0, HOURS_PER_YEAR)
    assert result.mean_correlation > 0.95
    assert result.beats_best_component is False


def test_uncorrelated_components_diversify() -> None:
    """Two independent real signals must blend to a better Sharpe than either,
    which is the entire premise of a multi-horizon book."""
    rng = np.random.default_rng(11)
    n_periods, n_symbols = 4_000, 50
    a = rng.normal(0, 1, (n_periods, n_symbols))
    b = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.006, (n_periods, n_symbols))
    rel[1:] += (a[:-1] + b[:-1]) * 0.004
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    factors = {"a": cross_sectional_zscore(a), "b": cross_sectional_zscore(b)}
    result = evaluate_blend(close, factors, 1, 0.0, HOURS_PER_YEAR)
    assert abs(result.mean_correlation) < 0.5
    assert result.beats_best_component is True
    assert result.sharpe > result.best_component_sharpe


def test_blend_leaves_absent_cells_absent() -> None:
    """A symbol with no component score must not be blended to zero -- zero is
    a tradeable rank, absent is not."""
    close = _prices(400, 40, 12)
    a = np.full((400, 40), np.nan)
    b = np.full((400, 40), np.nan)
    a[:, :15] = 1.0
    b[:, :15] = -1.0
    returns, _ = blend_returns(close, {"a": a, "b": b}, 1)
    # Only 15 symbols ever have a score -- strictly below the 20-symbol
    # cross-section minimum. If NaN cells were blended to 0 instead, all 40
    # would count as scored and a book would be built from invented ranks.
    assert np.all(np.isnan(returns))


def test_correlation_of_a_stream_with_itself_is_one() -> None:
    signal, close = _signal_with_edge(1_500, 50, 13, 0.01)
    corr = factor_correlations(close, {"a": signal, "b": signal.copy()}, 1)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_blend_on_pure_noise_reports_no_advantage() -> None:
    rng = np.random.default_rng(14)
    close = _prices(3_000, 50, 15)
    factors = {
        "a": cross_sectional_zscore(rng.normal(0, 1, (3_000, 50))),
        "b": cross_sectional_zscore(rng.normal(0, 1, (3_000, 50))),
    }
    result = evaluate_blend(close, factors, 1, 2.0, HOURS_PER_YEAR, n_trials=8)
    assert result.deflated_sharpe < 0.95


def test_blend_handles_a_window_too_short_to_score() -> None:
    close = _prices(30, 40, 16)
    factors = {"a": cross_sectional_zscore(np.random.default_rng(17).normal(0, 1, (30, 40)))}
    result = evaluate_blend(close, factors, 24, 2.0, HOURS_PER_YEAR)
    assert result.sharpe == 0.0
    assert result.beats_best_component is False
