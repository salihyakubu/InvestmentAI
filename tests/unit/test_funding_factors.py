"""Funding factors: causality first, then the usual both-ways discipline.

The single most dangerous bug in this module would be a one-stamp lookahead.
Funding settles every 8 hours; using the rate published AFTER the moment of
decision would invent a spectacular strategy out of nothing, and it would
look entirely plausible. Causality is therefore tested directly and first.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.backtesting.cross_section import (
    cross_sectional_zscore,
    evaluate_factor,
    forward_returns,
)
from services.backtesting.funding_factors import (
    build_funding_factors,
    combined_factors,
    funding_zscore,
    trailing_mean,
    trailing_std,
)

sys_rng = np.random.default_rng


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_trailing_mean_never_sees_the_future() -> None:
    """A spike at t must not appear in any row before t."""
    values = np.zeros((10, 2))
    values[7] = 100.0
    mean = trailing_mean(values, 4)
    assert np.allclose(mean[:7], 0.0)
    assert mean[7, 0] > 0


def test_trailing_std_never_sees_the_future() -> None:
    values = np.zeros((10, 2))
    values[6] = 50.0
    std = trailing_std(values, 3)
    assert np.allclose(std[:6], 0.0)
    assert std[6, 0] > 0


def test_a_factor_built_from_future_funding_would_be_caught() -> None:
    """Positive control for the causality tests themselves.

    If the guards above were vacuous, a deliberately shifted (cheating)
    factor would score the same as the honest one. It must not.
    """
    rng = sys_rng(1)
    n_periods, n_symbols = 800, 40
    funding = rng.normal(0, 0.0005, (n_periods, n_symbols))
    # The bar-t return is driven by the funding stamped at t, so a factor
    # peeking one stamp ahead (funding[t+1]) predicts forward[t] perfectly
    # while the honest factor (funding[t]) predicts nothing.
    rel = -funding * 5.0 + rng.normal(0, 0.001, (n_periods, n_symbols))
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    fwd = forward_returns(close, 1)

    honest = build_funding_factors(funding)["funding_level"]
    cheating = cross_sectional_zscore(-np.roll(funding, -1, axis=0))

    honest_ic = evaluate_factor("honest", honest, fwd, 1, 0.0, 4).mean_ic
    cheating_ic = evaluate_factor("cheat", cheating, fwd, 1, 0.0, 4).mean_ic
    # The cheat sees the future and must score far higher. If these were
    # close, the honest factor would be leaking.
    assert cheating_ic > 0.3
    assert abs(honest_ic) < cheating_ic / 2


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_battery_is_a_small_declared_set_with_a_declared_sign() -> None:
    funding = sys_rng(2).normal(0, 0.0005, (500, 40))
    factors = build_funding_factors(funding)
    assert set(factors) == {
        "funding_level",
        "funding_carry_24h",
        "funding_carry_72h",
        "funding_zscore_7d",
    }


def test_high_funding_maps_to_a_negative_score() -> None:
    """Sign convention: crowded longs are expected to underperform, so the
    most expensive contract must rank LAST, not first."""
    funding = np.tile(np.linspace(-0.001, 0.001, 40), (300, 1))
    level = build_funding_factors(funding)["funding_level"]
    row = level[-1]
    assert row[0] > row[-1]  # cheapest funding scores above most expensive
    assert np.argmax(row) == 0
    assert np.argmin(row) == len(row) - 1


def test_funding_zscore_is_relative_to_each_contracts_own_history() -> None:
    """A structurally expensive contract sitting at its own normal level must
    not look crowded."""
    n = 400
    funding = np.zeros((n, 2))
    funding[:, 0] = 0.001  # always expensive, never unusual
    funding[:, 1] = 0.0
    funding[-1, 1] = 0.001  # normally cheap, suddenly expensive
    z = funding_zscore(funding, 100)
    # A constant series has no meaningful z-score; it must be NaN, not the
    # order-1 garbage that float residue in nanstd would otherwise produce.
    assert np.isnan(z[-1, 0])
    assert z[-1, 1] > 1.0  # a genuine departure


def test_zscore_is_undefined_before_any_history_exists() -> None:
    funding = np.full((5, 3), np.nan)
    funding[3:] = 0.0005
    z = funding_zscore(funding, 24)
    assert np.all(np.isnan(z[:3]))


def test_combined_battery_keeps_the_price_factors_as_a_control() -> None:
    """Funding must be judged against the price factors on the same data, or
    'it works' has no baseline."""
    rng = sys_rng(3)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, (600, 40)), axis=0)
    volume = rng.lognormal(10, 1, (600, 40))
    funding = rng.normal(0, 0.0005, (600, 40))
    factors = combined_factors(close, volume, funding)
    assert "reversal_1h" in factors  # price control retained
    assert "funding_level" in factors
    assert len(factors) == 10


# ---------------------------------------------------------------------------
# Both ways
# ---------------------------------------------------------------------------


def test_random_funding_yields_no_edge() -> None:
    rng = sys_rng(4)
    n_periods, n_symbols = 2_000, 50
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, (n_periods, n_symbols)), axis=0)
    funding = rng.normal(0, 0.0005, (n_periods, n_symbols))
    fwd = forward_returns(close, 8)
    for name, sig in build_funding_factors(funding).items():
        result = evaluate_factor(name, sig, fwd, 8, 2.0, n_trials=4)
        assert result.has_edge is False, f"{name} passed on random funding"


def test_a_genuine_funding_effect_is_detected() -> None:
    """If crowded longs really did underperform, the battery must find it."""
    rng = sys_rng(5)
    n_periods, n_symbols = 2_000, 50
    funding = rng.normal(0, 0.0005, (n_periods, n_symbols))
    rel = rng.normal(0, 0.002, (n_periods, n_symbols))
    # Next period's relative return is driven by THIS period's funding,
    # negatively: crowded longs underperform.
    rel[1:] += -funding[:-1] * 8.0
    close = 100.0 * np.cumprod(1 + rel, axis=0)
    fwd = forward_returns(close, 1)
    level = build_funding_factors(funding)["funding_level"]
    result = evaluate_factor("funding_level", level, fwd, 1, 0.0, n_trials=4)
    assert result.mean_ic > 0.1
    assert result.ic_t_stat > 5


def test_funding_factors_turn_over_less_than_price_reversal() -> None:
    """The structural claim: an 8-hourly input churns less than an hourly one.
    If this failed, funding would inherit the problem it exists to avoid."""
    rng = sys_rng(6)
    n_periods, n_symbols = 1_500, 40
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, (n_periods, n_symbols)), axis=0)
    volume = rng.lognormal(10, 1, (n_periods, n_symbols))
    # Funding moves on an 8-hour grid, as it does in reality.
    stamps = rng.normal(0, 0.0005, (n_periods // 8 + 1, n_symbols))
    funding = np.repeat(stamps, 8, axis=0)[:n_periods]
    factors = combined_factors(close, volume, funding)
    fwd = forward_returns(close, 1)
    reversal = evaluate_factor("reversal_1h", factors["reversal_1h"], fwd, 1, 2.0, 10)
    level = evaluate_factor("funding_level", factors["funding_level"], fwd, 1, 2.0, 10)
    assert level.turnover < reversal.turnover


def test_constant_funding_carries_no_cross_sectional_information() -> None:
    funding = np.full((500, 40), 0.0001)
    level = build_funding_factors(funding)["funding_level"]
    assert np.all(np.isnan(level)) or np.allclose(np.nan_to_num(level), 0.0)


@pytest.mark.parametrize("lookback", [24, 72, 168])
def test_trailing_windows_shrink_gracefully_at_the_start(lookback: int) -> None:
    values = np.ones((10, 3))
    mean = trailing_mean(values, lookback)
    assert np.allclose(mean, 1.0)  # partial windows still produce a value
