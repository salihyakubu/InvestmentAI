"""Cross-sectional research harness: must find real structure and refuse noise.

Same discipline as the live-signal evaluator. A tool that only ever says "no
edge" is worthless as evidence for a null, so every guard here is paired with
a positive control proving the tool would have caught a real effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.backtesting.cross_section import (
    MIN_SYMBOLS_PER_PERIOD,
    build_factors,
    cross_sectional_zscore,
    evaluate,
    evaluate_factor,
    forward_returns,
    long_short_returns,
    momentum,
    period_ic,
    profile_monotonicity,
    quantile_profile,
    tail_dominance,
)


def _random_prices(n_periods: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, (n_periods, n_symbols))
    return 100.0 * np.cumprod(1 + steps, axis=0)


def _volume(n_periods: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(10.0, 1.0, (n_periods, n_symbols))


# ---------------------------------------------------------------------------
# The null
# ---------------------------------------------------------------------------


def test_random_walks_show_no_cross_sectional_edge() -> None:
    """Independent random walks have no relative structure to find."""
    close = _random_prices(3_000, 60, 1)
    volume = _volume(3_000, 60, 2)
    report = evaluate(close, volume, horizon=1, cost_bps=10.0)
    assert report.has_edge is False
    assert "NO CROSS-SECTIONAL EDGE" in report.verdict or "not profitable" in report.verdict.lower()


def test_no_factor_passes_on_noise_despite_many_being_tested() -> None:
    close = _random_prices(3_000, 60, 3)
    volume = _volume(3_000, 60, 4)
    report = evaluate(close, volume, horizon=1, cost_bps=10.0)
    assert all(not f.has_edge for f in report.factors)


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------


def test_a_planted_cross_sectional_signal_is_found() -> None:
    """A factor that genuinely ranks next-period relative returns must pass,
    or a null result from this harness would mean nothing."""
    rng = np.random.default_rng(5)
    n_periods, n_symbols = 3_000, 60
    # Each symbol gets a persistent relative-strength score; realised
    # relative returns follow it with heavy noise.
    score = rng.normal(0, 1, (n_periods, n_symbols))
    # The score must LEAD the return it predicts: forward_returns(close)[t]
    # is close[t+1]/close[t]-1, i.e. the return realised in bar t+1.
    rel = rng.normal(0, 0.004, (n_periods, n_symbols))
    rel[1:] += score[:-1] * 0.02
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    forward = forward_returns(close, 1)
    planted = cross_sectional_zscore(score)
    result = evaluate_factor("planted", planted, forward, 1, 10.0, n_trials=6)
    assert result.mean_ic > 0.2
    assert result.ic_t_stat > 5
    assert result.has_edge is True


def test_a_real_but_tiny_signal_is_reported_unprofitable() -> None:
    """Gross IC real, net of 20bps turnover cost negative -> not an edge."""
    rng = np.random.default_rng(6)
    n_periods, n_symbols = 3_000, 60
    score = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.01, (n_periods, n_symbols))
    rel[1:] += score[:-1] * 0.0004  # real, but far too small to pay 20bps
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    forward = forward_returns(close, 1)
    planted = cross_sectional_zscore(score)
    result = evaluate_factor("tiny", planted, forward, 1, 20.0, n_trials=6)
    assert result.gross_bps > 0
    assert result.net_bps[20.0] < 0
    assert result.has_edge is False


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_zscore_is_computed_within_each_timestamp() -> None:
    values = np.array([[1.0, 2.0, 3.0] * 10, [10.0, 20.0, 30.0] * 10])
    z = cross_sectional_zscore(values)
    # Both rows standardise to the same shape despite different levels: the
    # comparison is peer-relative, never across time.
    assert np.allclose(z[0], z[1])
    assert np.isclose(np.nanmean(z[0]), 0.0, atol=1e-9)


def test_zscore_skips_a_cross_section_that_is_too_thin() -> None:
    values = np.full((3, MIN_SYMBOLS_PER_PERIOD - 1), 1.0)
    values += np.random.default_rng(7).normal(0, 1, values.shape)
    assert np.all(np.isnan(cross_sectional_zscore(values)))


def test_unlisted_symbols_stay_absent_rather_than_filled() -> None:
    """A symbol that had not listed yet must not appear in the cross-section
    at a stale price -- that would invent tradeable history."""
    close = _random_prices(500, 40, 8)
    close[:200, 0] = np.nan  # symbol 0 lists late
    z = cross_sectional_zscore(momentum(close, 24))
    assert np.all(np.isnan(z[:200, 0]))
    assert np.isfinite(z[300:, 0]).any()


def test_forward_returns_are_market_neutral() -> None:
    """Each row must sum to ~0, so a signal cannot score by being long a
    rising market."""
    close = _random_prices(500, 40, 9)
    fwd = forward_returns(close, 1)
    rows = fwd[np.isfinite(fwd).sum(axis=1) >= MIN_SYMBOLS_PER_PERIOD]
    assert rows.size > 0
    assert np.allclose(np.nanmean(rows, axis=1), 0.0, atol=1e-12)


def test_long_short_weights_are_dollar_neutral() -> None:
    rng = np.random.default_rng(10)
    signal = rng.normal(0, 1, (50, 40))
    forward = rng.normal(0, 0.01, (50, 40))
    returns, turnover = long_short_returns(signal, forward)
    assert np.isfinite(returns).all()
    # Turnover of a fully re-drawn book is bounded by the two legs.
    assert np.nanmax(turnover) <= 4.0 + 1e-9


def test_turnover_is_zero_when_the_book_does_not_change() -> None:
    signal = np.tile(np.arange(40, dtype=float), (30, 1))
    forward = np.zeros((30, 40))
    _, turnover = long_short_returns(signal, forward)
    assert turnover[0] > 0  # first period builds the book
    assert np.allclose(turnover[1:], 0.0)  # identical ranking -> no trading


def test_period_ic_ignores_thin_cross_sections() -> None:
    signal = np.random.default_rng(11).normal(0, 1, (10, MIN_SYMBOLS_PER_PERIOD - 2))
    forward = np.random.default_rng(12).normal(0, 1, signal.shape)
    assert np.all(np.isnan(period_ic(signal, forward)))


def test_momentum_uses_only_past_bars() -> None:
    close = np.arange(1, 21, dtype=float).reshape(20, 1)
    mom = momentum(close, 5)
    assert np.all(np.isnan(mom[:5]))
    # At t=5, return is close[5]/close[0]-1 = 6/1-1 = 5.
    assert mom[5, 0] == pytest.approx(5.0)


def test_factor_battery_is_a_declared_small_set() -> None:
    """Every extra factor inflates the multiple-testing penalty, so the
    battery is pre-declared rather than searched."""
    close = _random_prices(400, 40, 13)
    volume = _volume(400, 40, 14)
    factors = build_factors(close, volume)
    assert set(factors) == {
        "reversal_1h",
        "reversal_6h",
        "momentum_24h",
        "momentum_168h",
        "low_volatility_24h",
        "volume_surge_24h",
    }


def test_short_history_reports_insufficient_not_no_edge() -> None:
    close = _random_prices(60, 40, 15)
    volume = _volume(60, 40, 16)
    report = evaluate(close, volume, horizon=1)
    assert report.has_edge is False
    assert "insufficient" in report.verdict


def test_survivorship_caveat_is_always_reported() -> None:
    close = _random_prices(1_000, 40, 17)
    volume = _volume(1_000, 40, 18)
    report = evaluate(close, volume, horizon=1)
    assert any("SURVIVORSHIP" in note for note in report.notes)


def test_breakeven_cost_is_gross_alpha_per_unit_turnover() -> None:
    """The number that decides whether a real signal is harvestable at all."""
    rng = np.random.default_rng(30)
    n_periods, n_symbols = 2_000, 60
    score = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.006, (n_periods, n_symbols))
    rel[1:] += score[:-1] * 0.01
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    forward = forward_returns(close, 1)
    result = evaluate_factor(
        "planted", cross_sectional_zscore(score), forward, 1, 10.0, n_trials=6
    )
    assert result.turnover > 0
    assert result.breakeven_bps == pytest.approx(
        result.gross_bps / result.turnover, rel=1e-6
    )
    # Net at exactly the breakeven cost must be ~zero.
    net_at_breakeven = result.gross_bps - result.turnover * result.breakeven_bps
    assert net_at_breakeven == pytest.approx(0.0, abs=1e-6)


def test_a_tail_driven_spread_is_rejected_as_a_factor() -> None:
    """The guard that caught momentum_168h.

    A signal whose top bucket alone carries the whole spread, with a flat
    middle, is a tail artifact. Its headline top-minus-bottom number looks
    excellent and it must still be refused.
    """
    rng = np.random.default_rng(40)
    n_periods, n_symbols = 3_000, 60
    score = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.004, (n_periods, n_symbols))
    # Only the most extreme scores get any predictive relationship at all.
    extreme = score[:-1] > 2.0
    rel[1:] += np.where(extreme, 0.05, 0.0)
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    forward = forward_returns(close, 1)
    result = evaluate_factor(
        "tail_only", cross_sectional_zscore(score), forward, 1, 0.0, n_trials=6
    )
    assert result.gross_bps > 0  # the headline spread looks good...
    assert result.tail_dominance > 0.5  # ...but one jump carries it
    assert result.has_edge is False


def test_a_monotone_factor_reports_high_monotonicity() -> None:
    rng = np.random.default_rng(41)
    n_periods, n_symbols = 2_000, 60
    score = rng.normal(0, 1, (n_periods, n_symbols))
    rel = rng.normal(0, 0.004, (n_periods, n_symbols))
    rel[1:] += score[:-1] * 0.01  # relationship holds across the whole range
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    result = evaluate_factor(
        "monotone", cross_sectional_zscore(score), forward_returns(close, 1),
        1, 0.0, n_trials=6,
    )
    assert result.monotonicity > 0.9


def test_quantile_profile_orders_low_to_high_signal() -> None:
    signal = np.tile(np.arange(50, dtype=float), (200, 1))
    forward = np.tile(np.arange(50, dtype=float) * 1e-4, (200, 1))
    profile = quantile_profile(signal, forward, n_buckets=5)
    assert profile == sorted(profile)
    assert profile_monotonicity(profile) == pytest.approx(1.0)


def test_tail_dominance_separates_an_even_ladder_from_one_big_jump() -> None:
    """The exact profiles observed on real data."""
    # reversal_1h: rises evenly.
    even = [-1.378, -0.689, -0.069, 0.528, 1.664]
    # momentum_168h: flat middle, then one jump into the top bucket.
    jumpy = [-2.235, -1.226, -0.210, -0.417, 6.694]
    assert tail_dominance(even) < 0.5
    assert tail_dominance(jumpy) > 0.5
    # Rank correlation cannot tell them apart -- both score 0.9+.
    assert profile_monotonicity(jumpy) >= 0.8
