"""Unit tests for the Stage-2 edge-gate logic (scripts/validate_model.py).

These cover the pure, deterministic pieces -- net-of-cost returns, annualized
Sharpe, cross-period stability, the per-symbol gate, and the cross-symbol
aggregate verdict -- without training any model, so a broken gate is caught in CI.
"""

from __future__ import annotations

import numpy as np

from scripts.validate_model import (
    aggregate_verdict,
    annualized_sharpe,
    net_period_returns,
    stability_fraction,
    symbol_has_edge,
)


def _result(edge: bool, insufficient: bool = False, name: str = "SYM") -> dict:
    return {"name": name, "edge": edge, "insufficient": insufficient}


# --- net_period_returns ---------------------------------------------------

def test_net_period_returns_non_overlapping_and_charges_cost() -> None:
    dirs = np.array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0])
    fwd = np.full(10, 0.02)
    net = net_period_returns(dirs, fwd, horizon=5, cost_bps=10.0)  # 10 bps = 0.001
    # Sampled every 5th bar -> positions [1, -1]; gross [+0.02, -0.02] minus 0.001 cost each.
    assert net.size == 2
    assert np.allclose(net, [0.02 - 0.001, -0.02 - 0.001])


def test_net_period_returns_no_cost_when_flat() -> None:
    dirs = np.zeros(10, dtype=int)
    net = net_period_returns(dirs, np.full(10, 0.02), horizon=5, cost_bps=25.0)
    assert np.allclose(net, 0.0)  # no position -> no gross, no cost


# --- annualized_sharpe ----------------------------------------------------

def test_sharpe_zero_when_mean_zero() -> None:
    assert annualized_sharpe(np.array([0.01, -0.01, 0.01, -0.01]), 50.4) == 0.0


def test_sharpe_zero_for_degenerate_inputs() -> None:
    assert annualized_sharpe(np.array([0.01]), 50.4) == 0.0        # < 2 points
    assert annualized_sharpe(np.array([0.01, 0.01, 0.01]), 50.4) == 0.0  # zero variance


def test_sharpe_matches_manual_formula() -> None:
    r = np.array([0.02, 0.01, 0.03, 0.00])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(50.4)
    assert annualized_sharpe(r, 50.4) == float(expected)
    assert annualized_sharpe(r, 50.4) > 0


# --- stability_fraction ---------------------------------------------------

def test_stability_all_positive_periods() -> None:
    assert stability_fraction(np.ones(8), n_folds=4) == 1.0


def test_stability_half_positive() -> None:
    r = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    assert stability_fraction(r, n_folds=4) == 0.5


def test_stability_too_little_data_is_zero() -> None:
    assert stability_fraction(np.array([1.0, 1.0, 1.0]), n_folds=4) == 0.0


# --- symbol_has_edge ------------------------------------------------------

def test_symbol_edge_requires_all_three_thresholds() -> None:
    assert symbol_has_edge(hit_rate=0.55, sharpe=0.8, stability=0.75) is True
    assert symbol_has_edge(hit_rate=0.51, sharpe=0.8, stability=0.75) is False   # hit
    assert symbol_has_edge(hit_rate=0.55, sharpe=0.40, stability=0.75) is False  # sharpe
    assert symbol_has_edge(hit_rate=0.55, sharpe=0.8, stability=0.50) is False   # stability


# --- aggregate_verdict ----------------------------------------------------

def test_single_symbol_pass_is_not_a_green_light() -> None:
    text, green = aggregate_verdict([_result(edge=True)])
    assert green is False
    assert "not sufficient" in text.lower()


def test_single_symbol_fail() -> None:
    text, green = aggregate_verdict([_result(edge=False)])
    assert green is False
    assert "NO demonstrable edge" in text


def test_three_symbols_all_pass_is_confirmed() -> None:
    text, green = aggregate_verdict([_result(True), _result(True), _result(True)])
    assert green is True
    assert "3/3" in text


def test_majority_of_three_passes() -> None:
    text, green = aggregate_verdict([_result(True), _result(True), _result(False)])
    assert green is True
    assert "2/3" in text


def test_minority_pass_is_no_edge() -> None:
    _text, green = aggregate_verdict([_result(True), _result(False), _result(False)])
    assert green is False


def test_two_symbols_below_confirm_minimum() -> None:
    text, green = aggregate_verdict([_result(True), _result(True)])
    assert green is False
    assert "fewer than 3" in text


def test_all_insufficient_cannot_assess() -> None:
    text, green = aggregate_verdict([_result(False, insufficient=True)])
    assert green is False
    assert "enough data" in text
