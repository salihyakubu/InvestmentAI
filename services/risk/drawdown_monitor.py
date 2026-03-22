"""Drawdown monitoring and daily/total loss limit tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Snapshot of current drawdown metrics."""

    current_drawdown_pct: float
    max_drawdown_pct: float
    peak_equity: float
    is_daily_limit_hit: bool
    is_total_limit_hit: bool


class DrawdownMonitor:
    """Track intra-day and total drawdown from peak equity.

    Parameters
    ----------
    max_daily_drawdown:
        Maximum allowed intra-day drawdown as a fraction (e.g. 0.05 = 5 %).
    max_total_drawdown:
        Maximum allowed drawdown from all-time peak as a fraction.
    """

    def __init__(
        self,
        max_daily_drawdown: float,
        max_total_drawdown: float,
    ) -> None:
        if max_daily_drawdown <= 0.0 or max_daily_drawdown > 1.0:
            raise ValueError(f"max_daily_drawdown must be in (0, 1], got {max_daily_drawdown}")
        if max_total_drawdown <= 0.0 or max_total_drawdown > 1.0:
            raise ValueError(f"max_total_drawdown must be in (0, 1], got {max_total_drawdown}")

        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown

        # All-time peak equity (updated on every call to `update`)
        self._peak_equity: float = 0.0
        # Max drawdown from all-time peak observed so far
        self._max_drawdown_pct: float = 0.0

        # Daily tracking
        self._daily_start_equity: float = 0.0
        self._daily_peak_equity: float = 0.0
        self._daily_max_drawdown_pct: float = 0.0

        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, current_equity: float) -> DrawdownState:
        """Update drawdown tracking with the latest equity value.

        Parameters
        ----------
        current_equity:
            The current total account equity.

        Returns
        -------
        DrawdownState
            Current drawdown metrics and whether limits are breached.
        """
        if current_equity < 0.0:
            current_equity = 0.0

        if not self._initialised:
            self._peak_equity = current_equity
            self._daily_start_equity = current_equity
            self._daily_peak_equity = current_equity
            self._initialised = True

        # Update all-time peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Update daily peak
        if current_equity > self._daily_peak_equity:
            self._daily_peak_equity = current_equity

        # Current drawdown from all-time peak
        if self._peak_equity > 0.0:
            current_dd_pct = (self._peak_equity - current_equity) / self._peak_equity
        else:
            current_dd_pct = 0.0

        # Track maximum drawdown
        if current_dd_pct > self._max_drawdown_pct:
            self._max_drawdown_pct = current_dd_pct

        # Daily drawdown from daily start equity
        if self._daily_start_equity > 0.0:
            daily_dd_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity
        else:
            daily_dd_pct = 0.0

        if daily_dd_pct > self._daily_max_drawdown_pct:
            self._daily_max_drawdown_pct = daily_dd_pct

        is_daily_hit = daily_dd_pct >= self.max_daily_drawdown
        is_total_hit = current_dd_pct >= self.max_total_drawdown

        if is_daily_hit:
            logger.warning(
                "Daily drawdown limit hit: %.2f%% >= %.2f%%",
                daily_dd_pct * 100,
                self.max_daily_drawdown * 100,
            )
        if is_total_hit:
            logger.warning(
                "Total drawdown limit hit: %.2f%% >= %.2f%%",
                current_dd_pct * 100,
                self.max_total_drawdown * 100,
            )

        return DrawdownState(
            current_drawdown_pct=current_dd_pct,
            max_drawdown_pct=self._max_drawdown_pct,
            peak_equity=self._peak_equity,
            is_daily_limit_hit=is_daily_hit,
            is_total_limit_hit=is_total_hit,
        )

    def reset_daily(self, current_equity: float | None = None) -> None:
        """Reset daily drawdown tracking at the start of a new trading day.

        Parameters
        ----------
        current_equity:
            If provided, sets the daily start equity explicitly. Otherwise
            the daily start equity is set to the last known peak equity.
        """
        if current_equity is not None:
            self._daily_start_equity = current_equity
            self._daily_peak_equity = current_equity
        else:
            self._daily_start_equity = self._peak_equity
            self._daily_peak_equity = self._peak_equity

        self._daily_max_drawdown_pct = 0.0
        logger.info(
            "Daily drawdown reset. Start equity=%.2f",
            self._daily_start_equity,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_days_to_recovery(
        current: float,
        peak: float,
        avg_daily_return: float,
    ) -> float | None:
        """Estimate number of trading days to recover from drawdown.

        Parameters
        ----------
        current:
            Current equity.
        peak:
            Peak equity to recover to.
        avg_daily_return:
            Average daily return (as a fraction, e.g. 0.001 for 0.1 %).

        Returns
        -------
        float or None
            Estimated trading days to recovery, or ``None`` if recovery
            is not possible (zero/negative return or already at peak).
        """
        if current <= 0.0:
            return None

        if current >= peak:
            return 0.0

        if avg_daily_return <= 0.0:
            return None

        # Days = ln(peak / current) / ln(1 + avg_daily_return)
        import math

        ratio = peak / current
        daily_growth = math.log1p(avg_daily_return)

        if daily_growth <= 0.0:
            return None

        return math.log(ratio) / daily_growth

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    @property
    def daily_drawdown_pct(self) -> float:
        return self._daily_max_drawdown_pct
