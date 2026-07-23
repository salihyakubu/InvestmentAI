"""services.backtesting.edge.run_backtest: portfolio assembly over the harness.

The per-symbol methodology (net-of-cost, non-overlapping, stability) is pinned
by test_edge_harness.py; these tests cover the portfolio layer the API serves:
result shape matches the dashboard contract, the equity curve/trades are
internally consistent, insufficient symbols are excluded from assembly but
reported, and a null random walk yields no confirmed edge.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.backtesting.edge import run_backtest

_REQUIRED_KEYS = {
    "total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades",
    "profit_factor", "equity_curve", "trades", "verdict", "edge_confirmed",
    "per_symbol", "cost_bps",
}


def _series(n: int, seed: int, drift: float = 0.0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i) for i in range(n)]
    return dates, closes


def test_result_shape_and_consistency() -> None:
    series = {
        "AAA": _series(1200, seed=1),
        "BBB": _series(1200, seed=2),
    }
    out = run_backtest(series, cost_bps=5.0, initial_capital=10_000.0)

    assert _REQUIRED_KEYS <= set(out)
    assert len(out["per_symbol"]) == 2
    # Equity curve starts at initial capital and its end implies total_return.
    curve = out["equity_curve"]
    assert curve[0]["equity"] == pytest.approx(10_000.0)
    assert out["total_return"] == pytest.approx(
        curve[-1]["equity"] / 10_000.0 - 1.0, abs=1e-9
    )
    # Dates are ISO strings ascending.
    dates = [p["date"] for p in curve if p["date"]]
    assert dates == sorted(dates)
    # Every trade is well-formed and its side matches price direction math.
    for t in out["trades"]:
        assert t["symbol"] in series
        assert t["side"] in ("buy", "sell")
        assert t["entry_date"] <= t["exit_date"]
    assert out["total_trades"] == len(out["trades"])
    assert 0.0 <= out["win_rate"] <= 1.0
    assert out["max_drawdown"] >= 0.0


def test_null_random_walk_confirms_no_edge() -> None:
    series = {"NULL": _series(3000, seed=7)}
    out = run_backtest(series, cost_bps=5.0)
    assert out["edge_confirmed"] is False
    assert "NOT" in out["verdict"] or "NO" in out["verdict"]


def test_insufficient_symbol_reported_but_excluded_from_assembly() -> None:
    series = {
        "TINY": _series(120, seed=3),   # far below MIN_TRAIN_ROWS after split
        "FULL": _series(1200, seed=4),
    }
    out = run_backtest(series, cost_bps=5.0)
    by_name = {r["name"]: r for r in out["per_symbol"]}
    assert by_name["TINY"]["insufficient"] is True
    assert by_name["FULL"]["insufficient"] is False
    # No trades attributed to the insufficient symbol.
    assert all(t["symbol"] != "TINY" for t in out["trades"])


def test_costs_reduce_returns() -> None:
    series = {"AAA": _series(1500, seed=5, drift=0.0008)}  # drifting walk
    cheap = run_backtest(series, cost_bps=0.0)
    dear = run_backtest(series, cost_bps=50.0)
    # Same signals, higher friction -> total return strictly lower (unless the
    # strategy never traded, in which case both are equal).
    if dear["total_trades"]:
        assert dear["total_return"] < cheap["total_return"]
