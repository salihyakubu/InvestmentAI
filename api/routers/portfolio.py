"""Portfolio endpoints."""

from __future__ import annotations

import time as _time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse
from api.schemas.portfolio import AllocationSchema, PortfolioSummary
from config.settings import Settings, get_settings
from core.models.orders import Fill, Order
from core.models.portfolio import PortfolioSnapshot
from services.portfolio.metrics import EquityPoint, LedgerFill, PerformanceMetrics
from services.portfolio.metrics import compute as compute_metrics

router = APIRouter(prefix="/portfolio")

# Performance is derived from the whole stored history, which the summary
# endpoint is polled against every few seconds. The inputs move on the
# snapshot interval (minutes), so a short cache keeps the scan off the hot
# path without the numbers ever being visibly stale.
_METRICS_TTL_SECONDS = 60.0
_METRICS_WINDOW_DAYS = 365
_METRICS_MAX_ROWS = 200_000
_metrics_cache: dict[str, tuple[float, PerformanceMetrics]] = {}


async def _performance(db: AsyncSession, trading_mode: str) -> PerformanceMetrics:
    """Compute (and briefly cache) performance over the stored history."""
    cached = _metrics_cache.get(trading_mode)
    now = _time.monotonic()
    if cached is not None and now - cached[0] < _METRICS_TTL_SECONDS:
        return cached[1]

    since = datetime.now(UTC) - timedelta(days=_METRICS_WINDOW_DAYS)
    rows = (
        await db.execute(
            select(PortfolioSnapshot.time, PortfolioSnapshot.total_equity)
            .where(
                PortfolioSnapshot.time >= since,
                PortfolioSnapshot.trading_mode == trading_mode,
            )
            .order_by(PortfolioSnapshot.time)
            .limit(_METRICS_MAX_ROWS)
        )
    ).all()
    points = [EquityPoint(time=t, equity=Decimal(str(e))) for t, e in rows]

    fill_rows = (
        await db.execute(
            select(Order.symbol, Order.side, Fill.price, Fill.quantity, Fill.commission)
            .join(Fill, Fill.order_id == Order.id)
            .where(Fill.filled_at >= since, Order.trading_mode == trading_mode)
            .order_by(Fill.filled_at)
        )
    ).all()
    fills = [
        LedgerFill(
            symbol=symbol,
            side=side,
            price=Decimal(str(price)),
            quantity=Decimal(str(quantity)),
            commission=Decimal(str(commission or 0)),
        )
        for symbol, side, price, quantity, commission in fill_rows
    ]

    metrics = compute_metrics(points, fills)
    _metrics_cache[trading_mode] = (now, metrics)
    return metrics


@router.get(
    "/summary",
    response_model=PortfolioSummary,
    summary="Current portfolio state",
)
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PortfolioSummary:
    """Return the latest snapshot plus performance over stored history."""
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.trading_mode == settings.trading_mode)
        .order_by(PortfolioSnapshot.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    snapshot = result.scalar_one_or_none()

    if snapshot is None:
        return PortfolioSummary(
            total_equity=Decimal("0"),
            cash=Decimal("0"),
            positions_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            daily_return_pct=None,
            total_return_pct=None,
            daily_pnl_pct=None,
            sharpe_ratio=None,
            max_drawdown=None,
            win_rate=None,
        )

    metrics = await _performance(db, settings.trading_mode)
    return PortfolioSummary(
        total_equity=snapshot.total_equity,
        cash=snapshot.cash,
        positions_value=snapshot.positions_value,
        unrealized_pnl=snapshot.unrealized_pnl,
        realized_pnl=snapshot.realized_pnl,
        daily_return_pct=metrics.daily_pnl_pct,
        daily_pnl=metrics.daily_pnl,
        daily_pnl_pct=metrics.daily_pnl_pct,
        total_return=metrics.total_return,
        total_return_pct=metrics.total_return_pct,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        closed_trades=metrics.closed_trades,
        position_count=snapshot.position_count,
    )


@router.get(
    "/allocations",
    response_model=list[AllocationSchema],
    summary="Current portfolio allocations",
)
async def get_allocations(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> list[AllocationSchema]:
    """Return the current asset allocations from the latest snapshot."""
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.trading_mode == settings.trading_mode)
        .order_by(PortfolioSnapshot.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    snapshot = result.scalar_one_or_none()

    if snapshot is None or not snapshot.allocations:
        return []

    allocations: list[AllocationSchema] = []
    for symbol, data in snapshot.allocations.items():
        allocations.append(
            AllocationSchema(
                symbol=symbol,
                weight=data.get("weight", 0.0),
                value=Decimal(str(data.get("value", 0))),
            )
        )
    return allocations


@router.get(
    "/snapshots",
    summary="Historical portfolio snapshots",
)
async def get_snapshots(
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    days: int = Query(default=30, ge=1, le=3650),
    max_points: int = Query(default=750, ge=10, le=5000),
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> list[dict[str, Any]]:
    """Return historical snapshots in CHRONOLOGICAL order.

    Rows are stored on a 5-minute cadence, so a naive row limit covers only
    hours and every chart label collapses to a single date. The window is
    therefore expressed in *days* and evenly downsampled to *max_points*,
    keeping the newest point so the curve always ends at the present.
    """
    window_start = start or (datetime.now(UTC) - timedelta(days=days))
    stmt = (
        select(PortfolioSnapshot)
        .where(
            PortfolioSnapshot.time >= window_start,
            PortfolioSnapshot.trading_mode == settings.trading_mode,
        )
        .order_by(PortfolioSnapshot.time)
    )
    if end is not None:
        stmt = stmt.where(PortfolioSnapshot.time <= end)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    if len(rows) > max_points:
        # Stride from the end so the most recent observation always survives.
        step = len(rows) // max_points + 1
        rows = list(reversed(rows[::-1][::step]))

    return [
        {
            "time": row.time.isoformat(),
            "total_equity": float(row.total_equity),
            "cash": float(row.cash),
            "positions_value": float(row.positions_value),
            "unrealized_pnl": float(row.unrealized_pnl),
            "realized_pnl": float(row.realized_pnl),
            "daily_return_pct": row.daily_return_pct,
            "position_count": row.position_count,
        }
        for row in rows
    ]


# The do-nothing benchmark (registered GO_LIVE.md 2026-08-02): equal weight
# across these, bought once at inception and never touched. The platform's
# own gauntlet showed buy-and-hold beating its models on 2/3 of symbols;
# this puts that bar on the dashboard, where it can indict or absolve every
# strategy live.
BENCHMARK_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SPY")
BENCHMARK_INCEPTION = datetime(2026, 8, 2, tzinfo=UTC)


@router.get(
    "/benchmark",
    summary="Do-nothing benchmark equity (equal-weight buy-and-hold)",
)
async def get_benchmark(
    days: int = Query(default=30, ge=1, le=3650),
    max_points: int = Query(default=750, ge=10, le=5000),
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> list[dict[str, Any]]:
    """Benchmark equity sampled at the portfolio's own snapshot times.

    Normalised to the account's equity AT INCEPTION, so the two curves start
    from the same dollar and diverge only by decisions. Each symbol
    contributes its last known 1m close at or before each snapshot time
    (strictly causal); before inception the series is empty rather than
    backfilled -- a benchmark that predates its own creation would be
    hindsight.
    """
    from sqlalchemy import and_

    window_start = max(
        BENCHMARK_INCEPTION, datetime.now(UTC) - timedelta(days=days)
    )
    snap_rows = (
        await db.execute(
            select(PortfolioSnapshot.time, PortfolioSnapshot.total_equity)
            .where(
                PortfolioSnapshot.time >= window_start,
                PortfolioSnapshot.trading_mode == settings.trading_mode,
            )
            .order_by(PortfolioSnapshot.time)
        )
    ).all()
    if not snap_rows:
        return []

    from core.models.market_data import OHLCVRecord

    def _aware(dt: datetime) -> datetime:
        # sqlite hands back naive datetimes, postgres aware ones; comparisons
        # against the aware inception constant must work on both.
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt

    series: dict[str, tuple[list[datetime], list[float]]] = {}
    for symbol in BENCHMARK_SYMBOLS:
        rows = (
            await db.execute(
                select(OHLCVRecord.time, OHLCVRecord.close)
                .where(
                    and_(
                        OHLCVRecord.symbol == symbol,
                        OHLCVRecord.timeframe == "1m",
                        OHLCVRecord.time >= BENCHMARK_INCEPTION - timedelta(days=3),
                    )
                )
                .order_by(OHLCVRecord.time)
            )
        ).all()
        if rows:
            series[symbol] = (
                [_aware(r[0]) for r in rows],
                [float(r[1]) for r in rows],
            )
    if len(series) < 2:  # a one-symbol "benchmark" is not the registered one
        return []

    import bisect

    def last_close(symbol: str, when: datetime) -> float | None:
        times, closes = series[symbol]
        i = bisect.bisect_right(times, when) - 1
        return closes[i] if i >= 0 else None

    # Base price per symbol: last close at or before inception (or first
    # available after, for symbols that begin trading post-inception).
    base: dict[str, float] = {}
    for symbol in series:
        price = last_close(symbol, BENCHMARK_INCEPTION)
        if price is None:
            price = series[symbol][1][0]
        base[symbol] = price

    inception_equity: float | None = None
    out: list[dict[str, Any]] = []
    for raw_when, equity in snap_rows:
        when = _aware(raw_when)
        if inception_equity is None:
            inception_equity = float(equity)
        ratios = []
        for symbol in series:
            price = last_close(symbol, when)
            if price is not None:
                ratios.append(price / base[symbol])
        if not ratios:
            continue
        out.append(
            {
                "time": when.isoformat(),
                "benchmark_equity": inception_equity * float(np.mean(ratios)),
            }
        )

    if len(out) > max_points:
        step = len(out) // max_points + 1
        out = list(reversed(out[::-1][::step]))
    return out


@router.post(
    "/rebalance",
    response_model=SuccessResponse,
    summary="Trigger manual rebalance",
)
async def trigger_rebalance(
    _user: dict[str, Any] = Depends(get_current_user),
) -> SuccessResponse:
    """Trigger a manual portfolio rebalance.

    This enqueues a rebalance task; the actual rebalancing is handled
    asynchronously by the portfolio service.
    """
    # TODO: Publish rebalance event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Rebalance service not yet implemented",
    )
