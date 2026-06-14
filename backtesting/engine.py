"""Core backtesting engine for strategy evaluation."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

import numpy as np

from config.settings import Settings
from core.enums import OrderSide, OrderType, TimeFrame

from backtesting.performance import PerformanceAnalyzer, TradeRecord
from backtesting.simulator import Bar, MarketSimulator, SimulatedFill, SimulatedOrder

logger = logging.getLogger(__name__)

# Bars per year by timeframe (equity convention: 252 trading days, 6.5h
# sessions) -- used to annualise risk metrics correctly for the timeframe.
_PERIODS_PER_YEAR: dict[TimeFrame, float] = {
    TimeFrame.M1: 252 * 6.5 * 60,
    TimeFrame.M5: 252 * 6.5 * 12,
    TimeFrame.M15: 252 * 6.5 * 4,
    TimeFrame.H1: 252 * 6.5,
    TimeFrame.H4: 252 * 6.5 / 4,
    TimeFrame.D1: 252,
    TimeFrame.W1: 52,
}


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class Signal:
    """A trading signal emitted by a strategy function."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None


@dataclass
class PortfolioState:
    """Snapshot of portfolio state passed to the strategy function."""

    cash: Decimal
    equity: Decimal
    positions: dict[str, "OpenPosition"]
    timestamp: datetime | None = None


@dataclass
class OpenPosition:
    """A currently open position tracked by the engine."""

    position_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    entry_bar_idx: int


@dataclass
class ClosedTrade:
    """A completed round-trip trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    entry_bar_idx: int
    exit_bar_idx: int


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    equity_curve: np.ndarray
    trades: list[ClosedTrade]
    metrics: dict[str, Any]
    drawdown_curve: np.ndarray
    bars_processed: int
    symbols: list[str]


# Type alias for the strategy callable.
# strategy_fn(bars_dict, portfolio_state) -> list[Signal]
StrategyFn = Callable[
    [dict[str, Bar], PortfolioState],
    list[Signal],
]


class BacktestEngine:
    """Event-driven backtesting engine supporting multiple symbols.

    Parameters
    ----------
    settings:
        Application settings (used for risk thresholds).
    initial_capital:
        Starting cash balance.
    start_date:
        Inclusive start of the backtest window.
    end_date:
        Inclusive end of the backtest window.
    symbols:
        List of symbols to backtest.
    timeframe:
        Bar timeframe for the backtest.
    slippage_bps:
        Slippage in basis points for market / stop fills.
    commission_per_share:
        Commission charged per share on each fill.
    """

    def __init__(
        self,
        settings: Settings,
        initial_capital: Decimal,
        start_date: date,
        end_date: date,
        symbols: list[str],
        timeframe: TimeFrame = TimeFrame.D1,
        slippage_bps: Decimal = Decimal("5"),
        commission_per_share: Decimal = Decimal("0.005"),
    ) -> None:
        self._settings = settings
        self._initial_capital = initial_capital
        self._start_date = start_date
        self._end_date = end_date
        self._symbols = symbols
        self._timeframe = timeframe

        self._simulator = MarketSimulator(
            commission_per_share=commission_per_share,
            default_slippage_bps=slippage_bps,
        )

        # Internal state -- reset each run.
        self._cash = Decimal("0")
        self._positions: dict[str, OpenPosition] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._equity_history: list[float] = []

        # Bar data: symbol -> list of Bar (chronologically ordered).
        self._bar_data: dict[str, list[Bar]] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_bars(self, symbol: str, bars: list[Bar]) -> None:
        """Load historical bar data for *symbol*.

        Bars must be in chronological order.
        """
        self._bar_data[symbol] = bars
        logger.info("Loaded %d bars for %s", len(bars), symbol)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, strategy_fn: StrategyFn) -> BacktestResult:
        """Execute the backtest.

        Parameters
        ----------
        strategy_fn:
            ``strategy_fn(bars_dict, portfolio_state) -> list[Signal]``
            Called once per bar with the current bar for each symbol and
            the portfolio state.

        Returns
        -------
        BacktestResult
        """
        self._cash = self._initial_capital
        self._positions.clear()
        self._closed_trades.clear()
        self._equity_history.clear()

        # Determine the number of bars to iterate (use shortest series).
        if not self._bar_data:
            logger.warning("No bar data loaded -- nothing to backtest")
            equity = np.array([float(self._initial_capital)])
            return BacktestResult(
                equity_curve=equity,
                trades=[],
                metrics={},
                drawdown_curve=np.zeros(1),
                bars_processed=0,
                symbols=self._symbols,
            )

        steps = self._build_steps()
        n_bars = len(steps)
        logger.info(
            "Starting backtest: %d steps, %d symbols, capital=%s",
            n_bars,
            len(self._bar_data),
            self._initial_capital,
        )

        # Signals decided on one bar execute on the NEXT bar (at its open), so a
        # strategy cannot trade on information from the bar it is deciding on.
        # This is the fix for look-ahead bias.
        pending_signals: list[Signal] = []
        for idx, ts, current_bars in steps:
            # 1. Execute signals queued on the previous step against this bar.
            for signal in pending_signals:
                self._process_signal(signal, current_bars, bar_idx=idx)
            pending_signals = []

            # 2. Mark-to-market AFTER fills, at this bar's close.
            equity = self._compute_equity(current_bars)
            self._equity_history.append(float(equity))

            # 3. Ask the strategy for signals; they execute on the next step.
            state = PortfolioState(
                cash=self._cash,
                equity=equity,
                positions=dict(self._positions),
                timestamp=ts,
            )
            pending_signals = strategy_fn(current_bars, state) or []

        equity_curve = np.array(self._equity_history, dtype=np.float64)

        # Build trade records for performance analysis.
        trade_records = [
            TradeRecord(
                symbol=t.symbol,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                quantity=t.quantity,
                pnl=t.pnl,
                commission=t.commission,
            )
            for t in self._closed_trades
        ]

        metrics = PerformanceAnalyzer.compute_metrics(
            equity_curve,
            trade_records,
            periods_per_year=_PERIODS_PER_YEAR.get(self._timeframe, 252),
        )

        # Drawdown curve.
        running_max = np.maximum.accumulate(equity_curve)
        drawdown_curve = np.where(
            running_max == 0, 0, (equity_curve - running_max) / running_max
        )

        result = BacktestResult(
            equity_curve=equity_curve,
            trades=self._closed_trades,
            metrics=metrics,
            drawdown_curve=drawdown_curve,
            bars_processed=n_bars,
            symbols=self._symbols,
        )

        report = PerformanceAnalyzer.generate_report(metrics, equity_curve)
        logger.info("\n%s", report)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_steps(self) -> list[tuple[int, datetime | None, dict[str, Bar]]]:
        """Ordered ``(idx, timestamp, {symbol: Bar})`` steps for the run loop.

        If every loaded bar carries a timestamp, align symbols on the union of
        timestamps (forward-filling each symbol's most recent bar) so a
        multi-symbol backtest is date-aligned rather than index-aligned.
        Otherwise fall back to positional alignment, warning if series lengths
        differ (silent truncation would otherwise hide missing data).
        """
        if not self._bar_data:
            return []

        has_ts = all(
            all(b.timestamp is not None for b in bars)
            for bars in self._bar_data.values()
        )

        if has_ts:
            all_ts = sorted(
                {b.timestamp for bars in self._bar_data.values() for b in bars}
            )
            by_ts = {
                sym: {b.timestamp: b for b in bars}
                for sym, bars in self._bar_data.items()
            }
            steps: list[tuple[int, datetime | None, dict[str, Bar]]] = []
            last_bar: dict[str, Bar] = {}
            for idx, ts in enumerate(all_ts):
                for sym in self._bar_data:
                    bar = by_ts[sym].get(ts)
                    if bar is not None:
                        last_bar[sym] = bar
                steps.append((idx, ts, dict(last_bar)))
            return steps

        lengths = {sym: len(bars) for sym, bars in self._bar_data.items()}
        n = min(lengths.values())
        if len(set(lengths.values())) > 1:
            logger.warning(
                "Symbol bar counts differ (%s); truncating to %d. Positional "
                "alignment assumes bar i is the same date for every symbol -- "
                "attach timestamps to bars for correct alignment.",
                lengths,
                n,
            )
        steps = []
        for i in range(n):
            snapshot = {sym: self._bar_data[sym][i] for sym in self._bar_data}
            steps.append((i, None, snapshot))
        return steps

    def _compute_equity(self, current_bars: dict[str, Bar]) -> Decimal:
        """Cash plus the signed mark-to-market value of open positions.

        Cash already reflects every trade cash flow (including short-sale
        proceeds and buy-to-cover payments), so a long contributes +mark*qty and
        a short contributes -mark*qty (the liability to repurchase the shares).
        """
        equity = self._cash
        for pos in self._positions.values():
            bar = current_bars.get(pos.symbol)
            if bar is None:
                continue
            if pos.side == OrderSide.BUY:
                equity += bar.close * pos.quantity
            else:
                equity -= bar.close * pos.quantity
        return equity

    def _process_signal(
        self,
        signal: Signal,
        current_bars: dict[str, Bar],
        bar_idx: int,
    ) -> None:
        """Convert a signal to a simulated order and process the fill."""
        bar = current_bars.get(signal.symbol)
        if bar is None:
            return

        order = SimulatedOrder(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=signal.side,
            order_type=signal.order_type,
            quantity=signal.quantity,
            limit_price=signal.limit_price,
            stop_price=signal.stop_price,
        )

        fill = self._simulator.simulate_fill(order, bar)
        if fill is None:
            return

        self._apply_fill(fill, bar_idx)

    def _apply_fill(self, fill: SimulatedFill, bar_idx: int) -> None:
        """Update cash and positions based on a fill."""
        # Check if this fill closes an existing position.
        existing = self._find_opposing_position(fill.symbol, fill.side)

        if existing is not None:
            # Close (part of) the opposing position.
            close_qty = min(fill.quantity, existing.quantity)
            if existing.side == OrderSide.BUY:
                # Closing a long: sell -> receive proceeds.
                pnl = float(
                    (fill.price - existing.entry_price) * close_qty
                    - fill.commission
                )
                self._cash += fill.price * close_qty - fill.commission
            else:
                # Closing a short: buy to cover -> pay.
                pnl = float(
                    (existing.entry_price - fill.price) * close_qty
                    - fill.commission
                )
                self._cash -= fill.price * close_qty + fill.commission

            remaining = existing.quantity - close_qty

            self._closed_trades.append(
                ClosedTrade(
                    symbol=fill.symbol,
                    side=existing.side.value,
                    entry_price=float(existing.entry_price),
                    exit_price=float(fill.price),
                    quantity=float(close_qty),
                    pnl=pnl,
                    commission=float(fill.commission),
                    entry_bar_idx=existing.entry_bar_idx,
                    exit_bar_idx=bar_idx,
                )
            )

            if remaining <= Decimal("0"):
                del self._positions[existing.position_id]
            else:
                existing.quantity = remaining
        else:
            # Open a new position.
            if fill.side == OrderSide.BUY:
                cost = fill.price * fill.quantity + fill.commission
                if cost > self._cash:
                    logger.warning(
                        "Insufficient cash to buy %s: need %.2f, have %.2f -- rejected",
                        fill.symbol,
                        float(cost),
                        float(self._cash),
                    )
                    return
                self._cash -= cost
            else:
                # Short sale: receive proceeds net of commission.
                self._cash += fill.price * fill.quantity - fill.commission

            pos_id = str(uuid.uuid4())
            self._positions[pos_id] = OpenPosition(
                position_id=pos_id,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_bar_idx=bar_idx,
            )

    def _find_opposing_position(
        self, symbol: str, fill_side: OrderSide
    ) -> OpenPosition | None:
        """Find an existing position in the opposite direction."""
        for pos in self._positions.values():
            if pos.symbol != symbol:
                continue
            if fill_side == OrderSide.SELL and pos.side == OrderSide.BUY:
                return pos
            if fill_side == OrderSide.BUY and pos.side == OrderSide.SELL:
                return pos
        return None
