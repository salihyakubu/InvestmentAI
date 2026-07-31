"""Persist live risk state so the dashboard stops flying blind.

The circuit breaker and every portfolio risk metric live inside the worker's
``RiskManagerService``; the API is a separate process that reads the
``risk_metrics`` table -- which, until this writer existed, had readers and
NO writer. Zero rows, ever. The dashboard first rendered that absence as a
green "all systems operating normally" (the one failure mode a safety
indicator must never have), then -- after PR #56 -- as an honest grey
UNKNOWN. This writer supplies the truth the card was waiting for.

It also fixes a quieter, worse defect: ``update_returns()`` and
``update_position()`` had no callers anywhere, so the risk engine's return
history was permanently empty. VaR and CVaR computed to 0.0 structurally,
which meant ``MaxVaRRule`` and ``MaxCorrelationRule`` APPROVED EVERY ORDER
without checking anything. Each write cycle now feeds the engine its inputs
first -- positions from the broker's book, per-symbol returns from stored
1-minute bars -- so the pre-trade rules evaluate real numbers again. The
write is the visible half; the re-armed rules are the half that matters.

Reporting only: this service never blocks an order itself, and a failed
iteration is logged and skipped. The risk engine keeps enforcing in-process
regardless of whether its state gets persisted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.market_data import OHLCVRecord
from core.models.risk import RiskMetric
from services.risk.service import RiskManagerService

logger = structlog.get_logger(__name__)

# How many 1m closes feed each symbol's return series. ~200 returns is the
# same order the audit prescribed: enough for a stable historical VaR without
# reaching back into a different regime.
_RETURN_BARS = 201
# Risk state moves with prices; two minutes keeps the dashboard honest
# without hammering the DB.
_INTERVAL_SECONDS = 120.0


class RiskMetricsWriter:
    """Feed the risk engine its inputs, then persist its verdict."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        risk_service: RiskManagerService,
        positions_provider: Callable[[], Awaitable[list[dict[str, Any]]]],
    ) -> None:
        self._session_factory = session_factory
        self._risk = risk_service
        self._positions_provider = positions_provider

    async def run(self, interval_seconds: float = _INTERVAL_SECONDS) -> None:
        """Write immediately, then on the interval, until cancelled."""
        first = True
        while True:
            try:
                if not first:
                    await asyncio.sleep(interval_seconds)
                first = False
                await self.write_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Risk metrics iteration failed")

    async def write_once(self) -> None:
        """One full cycle: feed inputs, compute, persist."""
        held = await self._feed_positions()
        await self._feed_returns(held)

        # check_portfolio_risk is synchronous, cheap numpy -- called directly.
        report = self._risk.check_portfolio_risk()
        extended = self._risk.extended_metrics()
        drawdown = report.drawdown_state

        now = datetime.now(UTC)
        row = RiskMetric(
            time=now,
            var_95=report.portfolio_var_95 or None,
            var_99=extended["var_99"],
            cvar_95=report.portfolio_cvar_95 or None,
            cvar_99=extended["cvar_99"],
            max_drawdown=(drawdown.max_drawdown_pct if drawdown else None),
            current_drawdown=(drawdown.current_drawdown_pct if drawdown else None),
            # beta deliberately stays NULL: nothing computes it, and a
            # fabricated 0.00 would render as "no market risk".
            beta=None,
            correlation_max=extended["correlation_max"],
            concentration_max=report.max_weight or None,
            circuit_breaker_active=report.circuit_breaker_state != "closed",
            details={
                "circuit_breaker_state": report.circuit_breaker_state,
                "volatility": extended["volatility"],
                "daily_pnl_pct": report.daily_pnl_pct,
                "hhi": report.hhi,
                "effective_positions": report.effective_positions,
                "correlation_violations": report.correlation_violations,
                "positions_fed": len(held),
                "rules_failed": [
                    r.rule_name for r in report.rule_results if not r.passed
                ],
            },
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        logger.debug(
            "risk_metrics_written",
            breaker=report.circuit_breaker_state,
            var_95=report.portfolio_var_95,
            positions=len(held),
        )

    async def _feed_positions(self) -> list[str]:
        """Sync the engine's position map to the broker's book. Returns held symbols."""
        positions = await self._positions_provider()
        seen: set[str] = set()
        for p in positions:
            symbol = str(p["symbol"])
            value = abs(float(p.get("market_value", 0) or 0))
            if value > 0:
                self._risk.update_position(symbol, value)
                seen.add(symbol)
        # Positions closed since the last cycle must leave the map, or stale
        # exposure keeps inflating concentration and VaR forever.
        for symbol in list(self._risk.positions):
            if symbol not in seen:
                self._risk.update_position(symbol, 0.0)
        return sorted(seen)

    async def _feed_returns(self, symbols: list[str]) -> None:
        """Load recent 1m closes per held symbol into the return history."""
        if not symbols:
            return
        async with self._session_factory() as session:
            for symbol in symbols:
                closes = await self._recent_closes(session, symbol)
                if len(closes) < 3:
                    continue
                returns = [
                    closes[i] / closes[i - 1] - 1.0
                    for i in range(1, len(closes))
                    if closes[i - 1] > 0
                ]
                if returns:
                    self._risk.update_returns(symbol, returns)

    @staticmethod
    async def _recent_closes(session: AsyncSession, symbol: str) -> list[float]:
        result = await session.execute(
            select(OHLCVRecord.close)
            .where(OHLCVRecord.symbol == symbol, OHLCVRecord.timeframe == "1m")
            .order_by(OHLCVRecord.time.desc())
            .limit(_RETURN_BARS)
        )
        closes = [float(row[0]) for row in result.all()]
        closes.reverse()  # chronological
        return closes
