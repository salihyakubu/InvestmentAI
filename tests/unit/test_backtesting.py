"""Backtest correctness: no look-ahead, correct short accounting, buying-power,
timestamp alignment, and timeframe-aware metrics."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np

from backtesting.engine import BacktestEngine, Signal
from backtesting.performance import PerformanceAnalyzer
from backtesting.simulator import Bar
from config.settings import Settings
from core.enums import OrderSide, OrderType, TimeFrame


def _bar(sym: str, o: float, h: float, lo: float, c: float, ts: datetime | None = None) -> Bar:
    return Bar(
        symbol=sym,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(lo)),
        close=Decimal(str(c)),
        volume=Decimal("1000"),
        timestamp=ts,
    )


def _engine(mock_settings: Settings, capital: str = "10000", symbols=("AAPL",)) -> BacktestEngine:
    return BacktestEngine(
        settings=mock_settings,
        initial_capital=Decimal(capital),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        symbols=list(symbols),
        timeframe=TimeFrame.D1,
        slippage_bps=Decimal("0"),
        commission_per_share=Decimal("0"),
    )


def _counter_strategy(signals_by_call: dict[int, list[Signal]]):
    state = {"n": 0}

    def strategy(bars, portfolio):  # noqa: ANN001
        i = state["n"]
        state["n"] += 1
        return signals_by_call.get(i, [])

    return strategy


def test_no_lookahead_fills_at_next_bar_open(mock_settings: Settings) -> None:
    """A signal decided on bar i must fill at bar i+1's OPEN, not bar i's close."""
    eng = _engine(mock_settings)
    eng.load_bars(
        "AAPL",
        [
            _bar("AAPL", 100, 110, 99, 110),   # bar0: decide BUY here
            _bar("AAPL", 120, 125, 119, 121),  # bar1: BUY fills at open=120
            _bar("AAPL", 130, 131, 129, 130),  # bar2: SELL fills at open=130
        ],
    )
    strat = _counter_strategy(
        {
            0: [Signal("AAPL", OrderSide.BUY, OrderType.MARKET, Decimal("1"))],
            1: [Signal("AAPL", OrderSide.SELL, OrderType.MARKET, Decimal("1"))],
        }
    )
    result = eng.run(strat)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Filled at next-bar opens (120, 130) -- NOT the decision bars' closes (110, 121).
    assert trade.entry_price == 120.0
    assert trade.exit_price == 130.0
    assert trade.pnl == 10.0


def test_short_round_trip_cash_signs(mock_settings: Settings) -> None:
    """Short then cover: opening credits proceeds, covering debits; PnL correct."""
    eng = _engine(mock_settings, capital="10000")
    eng.load_bars(
        "AAPL",
        [
            _bar("AAPL", 100, 101, 99, 100),  # decide SHORT
            _bar("AAPL", 90, 91, 89, 90),     # short opens at open=90; decide COVER
            _bar("AAPL", 80, 81, 79, 80),     # cover at open=80
        ],
    )
    strat = _counter_strategy(
        {
            0: [Signal("AAPL", OrderSide.SELL, OrderType.MARKET, Decimal("1"))],
            1: [Signal("AAPL", OrderSide.BUY, OrderType.MARKET, Decimal("1"))],
        }
    )
    result = eng.run(strat)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "sell"  # the position was short
    assert trade.entry_price == 90.0
    assert trade.exit_price == 80.0
    assert trade.pnl == 10.0  # shorted at 90, covered at 80 -> +10
    # Profit of 10 on a 10k account.
    assert result.equity_curve[-1] == 10010.0


def test_buying_power_blocks_oversized_buy(mock_settings: Settings) -> None:
    """A buy whose cost exceeds cash is rejected; equity never goes negative."""
    eng = _engine(mock_settings, capital="100")
    eng.load_bars(
        "AAPL",
        [
            _bar("AAPL", 50, 51, 49, 50),       # decide BUY 10
            _bar("AAPL", 100, 101, 99, 100),    # would cost 1000 >> 100 cash
        ],
    )
    strat = _counter_strategy(
        {0: [Signal("AAPL", OrderSide.BUY, OrderType.MARKET, Decimal("10"))]}
    )
    result = eng.run(strat)

    assert result.trades == []
    assert float(np.min(result.equity_curve)) >= 0.0
    assert result.equity_curve[-1] == 100.0


def test_timestamp_alignment_forward_fills_missing_symbol(mock_settings: Settings) -> None:
    """Symbols are aligned on the union of timestamps; a gap is forward-filled."""
    eng = _engine(mock_settings, symbols=("A", "B"))
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 1, 2)
    t2 = datetime(2024, 1, 3)
    eng.load_bars(
        "A",
        [_bar("A", 1, 1, 1, 10, t0), _bar("A", 1, 1, 1, 11, t1), _bar("A", 1, 1, 1, 12, t2)],
    )
    eng.load_bars(
        "B",
        [_bar("B", 1, 1, 1, 20, t0), _bar("B", 1, 1, 1, 22, t2)],  # B missing t1
    )

    steps = eng._build_steps()

    assert len(steps) == 3  # union of {t0, t1, t2}
    idx1, ts1, bars1 = steps[1]
    assert ts1 == t1
    # B has no t1 bar -> forward-filled with its t0 bar (close 20).
    assert bars1["B"].close == Decimal("20")
    assert bars1["A"].close == Decimal("11")


def test_sharpe_is_timeframe_aware() -> None:
    """Annualisation scales with periods_per_year (hourly vs daily)."""
    returns = np.array([0.01, -0.005, 0.02, 0.0, 0.01])
    daily = PerformanceAnalyzer.sharpe_ratio(returns, periods_per_year=252)
    hourly = PerformanceAnalyzer.sharpe_ratio(returns, periods_per_year=252 * 6.5)
    assert hourly > daily
    assert abs(hourly / daily - np.sqrt(6.5)) < 1e-9


def test_sortino_uses_full_period_downside_deviation() -> None:
    """Downside deviation divides by ALL periods, not just losing ones."""
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    expected_dd = np.sqrt((np.minimum(returns, 0.0) ** 2).mean())
    expected = returns.mean() / expected_dd * np.sqrt(252)
    got = PerformanceAnalyzer.sortino_ratio(returns, periods_per_year=252)
    assert abs(got - expected) < 1e-9
