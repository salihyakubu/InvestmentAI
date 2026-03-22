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

        n_bars = min(len(bars) for bars in self._bar_data.values())
        logger.info(
            "Starting backtest: %d bars, %d symbols, capital=%s",
            n_bars,
            len(self._bar_data),
            self._initial_capital,
        )

        for i in range(n_bars):
            # Build current bar dict.
            current_bars: dict[str, Bar] = {}
            for symbol, bars_list in self._bar_data.items():
                current_bars[symbol] = bars_list[i]

            # Compute equity before strategy.
            equity = self._compute_equity(current_bars)
            self._equity_history.append(float(equity))

            # Build portfolio state.
            state = PortfolioState(
                cash=self._cash,
                equity=equity,
                positions=dict(self._positions),
            )

            # Ask strategy for signals.
            signals = strategy_fn(current_bars, state)

            # Process signals.
            for signal in signals:
                self._process_signal(signal, current_bars, bar_idx=i)

        # Final equity snapshot using last bars.
        if n_bars > 0:
            last_bars = {
                sym: bars_list[n_bars - 1]
                for sym, bars_list in self._bar_data.items()
            }
            final_eq = self._compute_equity(last_bars)
            if len(self._equity_history) > 0:
                self._equity_history[-1] = float(final_eq)

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

        metrics = PerformanceAnalyzer.compute_metrics(equity_curve, trade_records)

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

    def _compute_equity(self, current_bars: dict[str, Bar]) -> Decimal:
        """Cash + mark-to-market value of all open positions."""
        equity = self._cash
        for pos in self._positions.values():
            bar = current_bars.get(pos.symbol)
            if bar is None:
                continue
            if pos.side == OrderSide.BUY:
                equity += (bar.close - pos.entry_price) * pos.quantity
            else:
                equity += (pos.entry_price - bar.close) * pos.quantity
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
            # Close the position.
            close_qty = min(fill.quantity, existing.quantity)
            if existing.side == OrderSide.BUY:
                pnl = float(
                    (fill.price - existing.entry_price) * close_qty
                    - fill.commission
                )
            else:
                pnl = float(
                    (existing.entry_price - fill.price) * close_qty
                    - fill.commission
                )

            self._cash += fill.price * close_qty - fill.commission
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
            cost = fill.price * fill.quantity + fill.commission
            self._cash -= cost

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
