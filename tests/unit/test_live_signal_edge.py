"""The live-signal evaluator must be able to return NO EDGE.

A measurement tool that cannot report a null is not a measurement tool. These
pin both directions: pure noise must come back NO EDGE, a planted signal must
be found, and a signal too small to pay its transaction costs must be
reported as unprofitable even though it is real.

This mirrors the invariant already enforced on the daily harness (its
random-walk self-test must report no edge) -- the honesty property is a
tested property here, not a claim.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.backtesting.live_signal import (
    Prediction,
    confidence_is_inverted,
    confidence_strata,
    decile_table,
    deflated_sharpe,
    deoverlap,
    evaluate,
    information_coefficient,
    monotonicity,
    scan_horizons,
    strategy_returns,
)

_START = datetime(2026, 4, 1, tzinfo=UTC)


def _predictions(
    signal: np.ndarray, outcome: np.ndarray, symbols: list[str] | None = None
) -> list[Prediction]:
    symbols = symbols or ["BTC/USDT"]
    per = len(signal) // len(symbols)
    out: list[Prediction] = []
    for s_i, sym in enumerate(symbols):
        lo, hi = s_i * per, (s_i + 1) * per
        for i, (sig, act) in enumerate(zip(signal[lo:hi], outcome[lo:hi], strict=False)):
            out.append(
                Prediction(
                    symbol=sym,
                    predicted_at=_START + timedelta(minutes=i),
                    expected_return=float(sig),
                    confidence=0.5,
                    actual_return=float(act),
                )
            )
    return out


# ---------------------------------------------------------------------------
# The null must hold
# ---------------------------------------------------------------------------


def test_pure_noise_reports_no_edge() -> None:
    """The invariant. If this ever passes, the tool is broken, not the models."""
    rng = np.random.default_rng(0)
    n = 30_000
    signal = rng.normal(0.0014, 0.0011, n)  # same moments as the live signal
    outcome = rng.normal(0.0, 0.0018, n)  # independent of it
    report = evaluate(_predictions(signal, outcome, ["A", "B", "C"]))
    assert report.has_edge is False
    assert "NO EDGE" in report.verdict
    assert abs(report.ic) < 0.05


def test_noise_across_many_symbols_still_reports_no_edge() -> None:
    """Enough symbols that one will pass by chance; the aggregate must not."""
    rng = np.random.default_rng(1)
    n = 60_000
    signal = rng.normal(0.0014, 0.0011, n)
    outcome = rng.normal(0.0, 0.0018, n)
    symbols = [f"S{i}" for i in range(10)]
    report = evaluate(_predictions(signal, outcome, symbols))
    assert report.has_edge is False


# ---------------------------------------------------------------------------
# ...and the tool must still find a real signal
# ---------------------------------------------------------------------------


def test_planted_edge_is_detected() -> None:
    """A signal large enough to clear costs must be found, or the null result
    on real data would be meaningless."""
    rng = np.random.default_rng(2)
    n = 30_000
    signal = rng.normal(0.0, 0.0011, n)
    # Realised return follows the signal, plus heavy noise.
    outcome = signal * 3.0 + rng.normal(0.0, 0.0009, n)
    report = evaluate(_predictions(signal, outcome, ["A", "B", "C"]), cost_bps=5.0)
    assert report.has_edge is True
    assert report.ic > 0.3
    assert report.cost_ladder[5.0] > 0


def test_real_but_uneconomic_signal_is_reported_unprofitable() -> None:
    """A genuine signal that cannot pay 10bps of slippage is still a losing
    strategy, and must be reported as one."""
    rng = np.random.default_rng(3)
    n = 30_000
    signal = rng.normal(0.0, 0.0011, n)
    outcome = signal * 0.05 + rng.normal(0.0, 0.0018, n)  # tiny true relationship
    report = evaluate(_predictions(signal, outcome, ["A", "B", "C"]), cost_bps=10.0)
    assert report.has_edge is False
    assert report.cost_ladder[10.0] < 0


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_deoverlap_reduces_sample_by_the_overlap_factor() -> None:
    assert deoverlap(np.arange(100), 5).size == 20
    assert deoverlap(np.arange(100), 1).size == 100


def test_deflated_sharpe_penalises_a_wider_search() -> None:
    """The same Sharpe is worth less when more strategies were tried."""
    few = deflated_sharpe(0.05, 5_000, n_trials=1)
    many = deflated_sharpe(0.05, 5_000, n_trials=500)
    assert few > many


def test_strategy_returns_centre_the_signal() -> None:
    """expected_return is almost always positive; using its raw sign would be
    a constant long bet measuring drift, not skill."""
    signal = np.full(100, 0.0014)  # constant, therefore no information
    outcome = np.full(100, 0.002)  # market drifted up
    net = strategy_returns(signal, outcome, cost_bps=0.0)
    assert np.allclose(net, 0.0)  # centred -> no position -> no fake profit


def test_strategy_returns_charge_cost_only_on_open_positions() -> None:
    """A flat period pays nothing; a position pays the full round trip."""
    # Median is 0.0, so the middle observation takes no position.
    signal = np.array([1.0, 0.0, -1.0])
    outcome = np.zeros(3)
    net = strategy_returns(signal, outcome, cost_bps=10.0)
    assert net[1] == 0.0
    assert net[0] == pytest.approx(-10.0 / 1e4)
    assert net[2] == pytest.approx(-10.0 / 1e4)


def test_deflated_sharpe_is_high_for_a_strong_signal_and_low_for_noise() -> None:
    """The gate must be passable, or every verdict is NO EDGE by construction."""
    # A per-observation Sharpe of 0.05 over 6000 periods is a huge edge.
    assert deflated_sharpe(0.05, 6_000, n_trials=24) > 0.95
    # A per-observation Sharpe of essentially zero is not.
    assert deflated_sharpe(0.001, 6_000, n_trials=24) < 0.95


def test_information_coefficient_on_a_constant_signal_is_zero() -> None:
    ic, p = information_coefficient(np.ones(500), np.random.default_rng(4).normal(size=500))
    assert ic == 0.0
    assert p == 1.0


def test_decile_ladder_is_monotonic_for_a_real_signal() -> None:
    rng = np.random.default_rng(5)
    signal = rng.normal(0, 0.001, 20_000)
    outcome = signal * 2 + rng.normal(0, 0.0005, 20_000)
    rows = decile_table(signal, outcome)
    assert len(rows) == 10
    assert monotonicity(rows) > 0.9


def test_decile_ladder_is_not_monotonic_for_noise() -> None:
    rng = np.random.default_rng(6)
    signal = rng.normal(0, 0.001, 20_000)
    outcome = rng.normal(0, 0.0018, 20_000)
    assert abs(monotonicity(decile_table(signal, outcome))) < 0.9


def test_empty_input_reports_cannot_assess_not_no_edge() -> None:
    """Absence of data is a different answer from absence of edge."""
    report = evaluate([])
    assert report.has_edge is False
    assert "cannot assess" in report.verdict


def test_symbols_below_the_minimum_are_skipped_not_judged(  ) -> None:
    rng = np.random.default_rng(8)
    n = 6_000
    signal = rng.normal(0.0014, 0.0011, n)
    outcome = rng.normal(0.0, 0.0018, n)
    preds = _predictions(signal, outcome, ["BIG"])
    preds += [
        Prediction("TINY", _START + timedelta(minutes=i), 0.001, 0.5, 0.0)
        for i in range(10)
    ]
    report = evaluate(preds, min_per_symbol=500)
    tiny = next(s for s in report.per_symbol if s.symbol == "TINY")
    assert tiny.insufficient is True
    assert tiny.has_edge is False


@pytest.mark.parametrize("cost", [0.0, 2.0, 5.0, 10.0])
def test_cost_ladder_is_monotonically_worse(cost: float) -> None:
    rng = np.random.default_rng(9)
    n = 20_000
    signal = rng.normal(0.0, 0.0011, n)
    outcome = signal * 2 + rng.normal(0.0, 0.0009, n)
    report = evaluate(_predictions(signal, outcome, ["A", "B", "C"]))
    ladder = report.cost_ladder
    assert ladder[0.0] >= ladder[cost]


# ---------------------------------------------------------------------------
# Horizon scan + confidence calibration
# ---------------------------------------------------------------------------


def _walk(n: int, seed: int, drift_from: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0008, n)
    if drift_from is not None:
        steps = steps + drift_from
    return 100.0 * np.cumprod(1 + steps)


def test_horizon_scan_finds_the_horizon_a_signal_actually_works_at() -> None:
    """A signal that predicts the next 60 bars, not the next 5, must show up
    at 60 and not at 5 -- that distinction is the whole point of the scan."""
    rng = np.random.default_rng(11)
    n = 20_000
    # Slow component that persists for ~60 bars.
    slow = np.repeat(rng.normal(0, 0.0002, n // 60 + 1), 60)[:n]
    close = 100.0 * np.cumprod(1 + slow + rng.normal(0, 0.0006, n))
    bar_index = np.arange(0, n - 300)
    signal = slow[bar_index] + rng.normal(0, 0.00002, bar_index.size)

    results = {r.horizon: r for r in scan_horizons(signal, close, bar_index)}
    assert results[60].ic > results[5].ic
    assert results[60].p_value < 0.05


def test_horizon_scan_reports_nothing_for_a_noise_signal() -> None:
    rng = np.random.default_rng(12)
    n = 20_000
    close = _walk(n, 13)
    bar_index = np.arange(0, n - 300)
    signal = rng.normal(0, 0.001, bar_index.size)
    for r in scan_horizons(signal, close, bar_index):
        assert abs(r.ic) < 0.25
        assert r.deflated_sharpe < 0.95


def test_horizon_scan_shows_cost_eating_a_small_signal() -> None:
    """Gross positive, net negative: the signal exists but cannot pay its way."""
    rng = np.random.default_rng(14)
    n = 20_000
    slow = np.repeat(rng.normal(0, 0.00003, n // 60 + 1), 60)[:n]
    close = 100.0 * np.cumprod(1 + slow + rng.normal(0, 0.0009, n))
    bar_index = np.arange(0, n - 300)
    signal = slow[bar_index]
    results = {r.horizon: r for r in scan_horizons(signal, close, bar_index)}
    r60 = results[60]
    assert r60.net_bps[0.0] > r60.net_bps[10.0]


def test_confidence_strata_detects_correct_calibration() -> None:
    rng = np.random.default_rng(15)
    n = 30_000
    confidence = rng.uniform(0.3, 0.8, n)
    signal = rng.normal(0, 0.001, n)
    # Noise shrinks as confidence rises -> agreement should rise.
    outcome = signal + rng.normal(0, 0.002, n) * (0.9 - confidence)
    strata = confidence_strata(confidence, signal, outcome)
    assert len(strata) == 5
    assert strata[-1]["sign_agreement"] > strata[0]["sign_agreement"]
    assert confidence_is_inverted(strata) is False


def test_confidence_strata_detects_inverted_calibration() -> None:
    """The failure mode that matters: most confident, most wrong. The
    abstention gate would then discard the good predictions."""
    rng = np.random.default_rng(16)
    n = 30_000
    confidence = rng.uniform(0.3, 0.8, n)
    signal = rng.normal(0, 0.001, n)
    # Sign flips as confidence rises.
    flip = np.where(confidence > 0.55, -1.0, 1.0)
    outcome = signal * flip + rng.normal(0, 0.0005, n)
    strata = confidence_strata(confidence, signal, outcome)
    assert strata[-1]["sign_agreement"] < strata[0]["sign_agreement"]
    assert confidence_is_inverted(strata) is True


def test_confidence_strata_needs_enough_data() -> None:
    assert confidence_strata(np.zeros(50), np.zeros(50), np.zeros(50)) == []
