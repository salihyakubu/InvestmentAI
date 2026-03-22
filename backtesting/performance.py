"""Performance analytics for backtest results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Annualisation factor assuming 252 trading days.
_TRADING_DAYS = 252


@dataclass
class TradeRecord:
    """A completed round-trip trade for performance analysis."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    entry_time: str | None = None
    exit_time: str | None = None


class PerformanceAnalyzer:
    """Computes standard backtest performance metrics.

    All metric methods are static / class-level so they can be used
    independently or via the convenience :meth:`compute_metrics` method.
    """

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @classmethod
    def compute_metrics(
        cls,
        equity_curve: np.ndarray,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        """Compute a full suite of performance metrics.

        Parameters
        ----------
        equity_curve:
            1-D array of portfolio equity values over time.
        trades:
            List of completed :class:`TradeRecord` objects.

        Returns
        -------
        dict
            Metric name -> value.
        """
        if len(equity_curve) < 2:
            return {"error": "Equity curve too short for analysis"}

        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[np.isfinite(returns)]

        total_ret = cls.total_return(equity_curve)
        annual_ret = cls.annualized_return(equity_curve)
        dd, peak_idx, trough_idx = cls.max_drawdown(equity_curve)

        return {
            "total_return": total_ret,
            "annualized_return": annual_ret,
            "sharpe_ratio": cls.sharpe_ratio(returns),
            "sortino_ratio": cls.sortino_ratio(returns),
            "calmar_ratio": cls.calmar_ratio(annual_ret, dd),
            "max_drawdown": dd,
            "max_drawdown_peak_idx": int(peak_idx),
            "max_drawdown_trough_idx": int(trough_idx),
            "win_rate": cls.win_rate(trades),
            "profit_factor": cls.profit_factor(trades),
            "avg_trade_return": cls.avg_trade_return(trades),
            "max_consecutive_losses": cls.max_consecutive_losses(trades),
            "recovery_factor": cls.recovery_factor(total_ret, dd),
            "total_trades": len(trades),
        }

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        """Total return as a fraction (e.g. 0.25 = 25%)."""
        if equity_curve[0] == 0:
            return 0.0
        return float((equity_curve[-1] / equity_curve[0]) - 1.0)

    @staticmethod
    def annualized_return(equity_curve: np.ndarray) -> float:
        """Annualised return assuming 252 trading days per bar."""
        if equity_curve[0] == 0 or len(equity_curve) < 2:
            return 0.0
        total = equity_curve[-1] / equity_curve[0]
        n_years = len(equity_curve) / _TRADING_DAYS
        if n_years <= 0:
            return 0.0
        if total <= 0:
            return -1.0
        return float(total ** (1.0 / n_years) - 1.0)

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
        """Annualised Sharpe ratio.

        Parameters
        ----------
        returns:
            Array of periodic (daily) returns.
        rf:
            Risk-free rate per period (default 0).
        """
        if len(returns) == 0:
            return 0.0
        excess = returns - rf
        std = float(np.std(excess, ddof=1))
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(_TRADING_DAYS))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
        """Annualised Sortino ratio (uses downside deviation only)."""
        if len(returns) == 0:
            return 0.0
        excess = returns - rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf") if np.mean(excess) > 0 else 0.0
        downside_std = float(np.sqrt(np.mean(downside**2)))
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(_TRADING_DAYS))

    @staticmethod
    def calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """Calmar ratio = annualised return / max drawdown."""
        if max_drawdown == 0:
            return float("inf") if annual_return > 0 else 0.0
        return annual_return / abs(max_drawdown)

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def max_drawdown(
        equity_curve: np.ndarray,
    ) -> tuple[float, int, int]:
        """Maximum drawdown as a negative fraction.

        Returns
        -------
        (max_dd, peak_idx, trough_idx)
            max_dd is negative (e.g. -0.15 = -15%).
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0

        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / np.where(
            running_max == 0, 1, running_max
        )

        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(equity_curve[:trough_idx + 1]))
        max_dd = float(drawdowns[trough_idx])

        return max_dd, peak_idx, trough_idx

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def win_rate(trades: list[TradeRecord]) -> float:
        """Fraction of trades that were profitable."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def profit_factor(trades: list[TradeRecord]) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def avg_trade_return(trades: list[TradeRecord]) -> float:
        """Average PnL per trade."""
        if not trades:
            return 0.0
        return sum(t.pnl for t in trades) / len(trades)

    @staticmethod
    def max_consecutive_losses(trades: list[TradeRecord]) -> int:
        """Longest streak of consecutive losing trades."""
        if not trades:
            return 0
        max_streak = 0
        current_streak = 0
        for t in trades:
            if t.pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @staticmethod
    def recovery_factor(total_return: float, max_drawdown: float) -> float:
        """Recovery factor = total return / abs(max drawdown)."""
        if max_drawdown == 0:
            return float("inf") if total_return > 0 else 0.0
        return total_return / abs(max_drawdown)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @classmethod
    def generate_report(
        cls,
        metrics: dict[str, Any],
        equity_curve: np.ndarray,
    ) -> str:
        """Produce a human-readable text performance report."""
        lines = [
            "=" * 60,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            "--- Returns ---",
            f"  Total Return:        {metrics.get('total_return', 0):.2%}",
            f"  Annualised Return:   {metrics.get('annualized_return', 0):.2%}",
            "",
            "--- Risk-Adjusted ---",
            f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.4f}",
            f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.4f}",
            f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.4f}",
            "",
            "--- Drawdown ---",
            f"  Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}",
            f"  Peak Index:          {metrics.get('max_drawdown_peak_idx', 0)}",
            f"  Trough Index:        {metrics.get('max_drawdown_trough_idx', 0)}",
            f"  Recovery Factor:     {metrics.get('recovery_factor', 0):.4f}",
            "",
            "--- Trade Stats ---",
            f"  Total Trades:        {metrics.get('total_trades', 0)}",
            f"  Win Rate:            {metrics.get('win_rate', 0):.2%}",
            f"  Profit Factor:       {metrics.get('profit_factor', 0):.4f}",
            f"  Avg Trade Return:    ${metrics.get('avg_trade_return', 0):.2f}",
            f"  Max Consec. Losses:  {metrics.get('max_consecutive_losses', 0)}",
            "",
            "--- Equity Curve ---",
            f"  Start:               ${equity_curve[0]:,.2f}",
            f"  End:                 ${equity_curve[-1]:,.2f}",
            f"  Bars:                {len(equity_curve)}",
            "=" * 60,
        ]
        return "\n".join(lines)
