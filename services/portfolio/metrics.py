"""Portfolio performance metrics computed from persisted history.

The dashboard's summary tiles used to render fields the API never returned,
which the frontend normaliser coerced to ``0`` -- so a losing account showed a
0.00% return, a 0.00 Sharpe and a 0.0% win rate, all indistinguishable from
"no data". These functions compute the metrics that the stored history can
support, and return ``None`` for the ones it cannot, so the UI can say
"insufficient history" instead of inventing a number.

That distinction is the point: this platform never displays a performance
figure the data does not support.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from math import sqrt

# A Sharpe ratio over a handful of observations is noise. Twenty daily returns
# is already generous for a point estimate; below it we report nothing.
MIN_DAILY_OBSERVATIONS = 20

# Trading days per year, for annualising a daily-return Sharpe.
_TRADING_DAYS = 252


@dataclass(frozen=True)
class EquityPoint:
    """One equity observation."""

    time: datetime
    equity: Decimal


@dataclass(frozen=True)
class LedgerFill:
    """One persisted fill, for round-trip matching."""

    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    commission: Decimal = Decimal("0")


def max_drawdown(equities: list[Decimal]) -> float | None:
    """Largest peak-to-trough decline as a positive fraction (0.05 = -5%)."""
    if len(equities) < 2:
        return None
    peak = equities[0]
    worst = 0.0
    for value in equities:
        if value > peak:
            peak = value
        if peak > 0:
            drop = float((peak - value) / peak)
            worst = max(worst, drop)
    return worst


def daily_closes(points: list[EquityPoint]) -> list[tuple[date, Decimal]]:
    """Collapse an intraday series to one closing equity per UTC day."""
    closes: dict[date, tuple[datetime, Decimal]] = {}
    for point in points:
        day = point.time.date()
        current = closes.get(day)
        if current is None or point.time >= current[0]:
            closes[day] = (point.time, point.equity)
    return [(day, value) for day, (_, value) in sorted(closes.items())]


def sharpe_ratio(points: list[EquityPoint]) -> float | None:
    """Annualised Sharpe of the daily return series (zero risk-free rate).

    ``None`` when there are fewer than ``MIN_DAILY_OBSERVATIONS`` daily
    returns, or when the series has no variance to speak of.
    """
    closes = daily_closes(points)
    returns: list[float] = []
    for (_, prev), (_, curr) in zip(closes, closes[1:], strict=False):
        if prev > 0:
            returns.append(float((curr - prev) / prev))
    if len(returns) < MIN_DAILY_OBSERVATIONS:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    return (mean / sqrt(variance)) * sqrt(_TRADING_DAYS)


def round_trip_stats(fills: list[LedgerFill]) -> tuple[float | None, int, Decimal]:
    """FIFO-match the fill ledger into closed round trips.

    Returns ``(win_rate, closed_trades, realised_pnl)``; the win rate is
    ``None`` until at least one round trip has closed. Matching mirrors the
    learning loop's attribution so the dashboard and the feedback loop agree
    on what a closed trade is.
    """
    open_lots: dict[str, deque[list[Decimal]]] = defaultdict(deque)
    wins = 0
    closed = 0
    realised = Decimal("0")

    for fill in fills:
        lots = open_lots[fill.symbol]
        remaining = fill.quantity
        opening = Decimal("1") if fill.side == "buy" else Decimal("-1")

        while remaining > 0 and lots and lots[0][0] * opening < 0:
            lot_qty, lot_price = lots[0]
            matched = min(remaining, abs(lot_qty))
            direction = Decimal("1") if lot_qty > 0 else Decimal("-1")
            pnl = direction * matched * (fill.price - lot_price)
            realised += pnl
            closed += 1
            if pnl > 0:
                wins += 1
            remaining -= matched
            if abs(lot_qty) <= matched:
                lots.popleft()
            else:
                lots[0][0] = lot_qty - direction * matched

        if remaining > 0:
            lots.append([opening * remaining, fill.price])

        realised -= fill.commission

    win_rate = (wins / closed) if closed else None
    return win_rate, closed, realised


@dataclass(frozen=True)
class PerformanceMetrics:
    """Everything the summary tiles need, with honest gaps."""

    total_return: Decimal
    total_return_pct: float | None
    daily_pnl: Decimal
    daily_pnl_pct: float | None
    max_drawdown: float | None
    sharpe_ratio: float | None
    win_rate: float | None
    closed_trades: int


def compute(
    points: list[EquityPoint],
    fills: list[LedgerFill],
    baseline: Decimal | None = None,
) -> PerformanceMetrics:
    """Derive the summary metrics from an equity series and a fill ledger.

    *points* must be chronological. *baseline* defaults to the first observed
    equity, which is the account's own starting point and therefore survives a
    change to the configured initial capital.
    """
    win_rate, closed_trades, _ = round_trip_stats(fills)
    if not points:
        return PerformanceMetrics(
            total_return=Decimal("0"),
            total_return_pct=None,
            daily_pnl=Decimal("0"),
            daily_pnl_pct=None,
            max_drawdown=None,
            sharpe_ratio=None,
            win_rate=win_rate,
            closed_trades=closed_trades,
        )

    equities = [p.equity for p in points]
    current = equities[-1]
    start = baseline if baseline is not None else equities[0]

    total_return = current - start
    total_return_pct = float(total_return / start * 100) if start else None

    # Day boundary in UTC: the equity at the last close before today.
    today = points[-1].time.date()
    closes = daily_closes(points)
    prior = [value for day, value in closes if day < today]
    day_open = prior[-1] if prior else equities[0]
    daily_pnl = current - day_open
    daily_pnl_pct = float(daily_pnl / day_open * 100) if day_open else None

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        max_drawdown=max_drawdown(equities),
        sharpe_ratio=sharpe_ratio(points),
        win_rate=win_rate,
        closed_trades=closed_trades,
    )
