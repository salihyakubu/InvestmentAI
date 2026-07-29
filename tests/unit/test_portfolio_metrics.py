"""Dashboard performance metrics: real numbers, or an honest gap.

The summary tiles used to render zeros for metrics the API never sent, which
is indistinguishable from a measured 0.00. These pin the two halves of the
contract: the arithmetic is right when the data supports it, and the result
is ``None`` -- never 0 -- when it does not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from services.portfolio.metrics import (
    MIN_DAILY_OBSERVATIONS,
    EquityPoint,
    LedgerFill,
    compute,
    daily_closes,
    max_drawdown,
    round_trip_stats,
    sharpe_ratio,
)

_BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _series(values: list[float], step_minutes: int = 5) -> list[EquityPoint]:
    return [
        EquityPoint(
            time=_BASE + timedelta(minutes=step_minutes * i), equity=Decimal(str(v))
        )
        for i, v in enumerate(values)
    ]


def _daily(values: list[float]) -> list[EquityPoint]:
    return [
        EquityPoint(time=_BASE + timedelta(days=i), equity=Decimal(str(v)))
        for i, v in enumerate(values)
    ]


def _fill(symbol: str, side: str, price: float, qty: float, commission: float = 0.0):
    return LedgerFill(
        symbol=symbol,
        side=side,
        price=Decimal(str(price)),
        quantity=Decimal(str(qty)),
        commission=Decimal(str(commission)),
    )


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_is_peak_to_trough_not_first_to_last() -> None:
    # Rises to 120, falls to 90 (-25% from peak), recovers to 110.
    assert max_drawdown([Decimal(x) for x in ("100", "120", "90", "110")]) == pytest.approx(0.25)


def test_max_drawdown_needs_two_points() -> None:
    assert max_drawdown([]) is None
    assert max_drawdown([Decimal("100")]) is None


def test_drawdown_measures_against_prior_peak_only() -> None:
    """A later high must not create a drawdown for an earlier dip."""
    assert max_drawdown([Decimal(x) for x in ("100", "99", "200")]) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Daily resampling + Sharpe
# ---------------------------------------------------------------------------


def test_daily_closes_takes_the_last_observation_of_each_day() -> None:
    points = [
        EquityPoint(time=datetime(2026, 7, 1, 9, tzinfo=UTC), equity=Decimal("100")),
        EquityPoint(time=datetime(2026, 7, 1, 21, tzinfo=UTC), equity=Decimal("103")),
        EquityPoint(time=datetime(2026, 7, 2, 10, tzinfo=UTC), equity=Decimal("101")),
    ]
    assert daily_closes(points) == [
        (datetime(2026, 7, 1).date(), Decimal("103")),
        (datetime(2026, 7, 2).date(), Decimal("101")),
    ]


def test_sharpe_is_none_below_the_observation_floor() -> None:
    """Ten days of a paper soak cannot produce a Sharpe ratio, and the tile
    must say so rather than render 0.00."""
    assert sharpe_ratio(_daily([100 + i * 0.1 for i in range(10)])) is None


def test_sharpe_computed_once_history_is_long_enough() -> None:
    values, equity = [], 100.0
    # Alternating drift so the series has genuine variance.
    for i in range(MIN_DAILY_OBSERVATIONS + 5):
        equity *= 1.004 if i % 3 else 0.997
        values.append(equity)
    result = sharpe_ratio(_daily(values))
    assert result is not None
    assert result > 0  # net-positive drift


def test_sharpe_none_on_a_perfectly_flat_series() -> None:
    assert sharpe_ratio(_daily([100.0] * (MIN_DAILY_OBSERVATIONS + 5))) is None


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_round_trip_win_rate_and_realised_pnl() -> None:
    fills = [
        _fill("BTC/USDT", "buy", 100.0, 1.0),
        _fill("BTC/USDT", "sell", 110.0, 1.0),  # +10 win
        _fill("ETH/USDT", "buy", 50.0, 2.0),
        _fill("ETH/USDT", "sell", 45.0, 2.0),  # -10 loss
    ]
    win_rate, closed, realised = round_trip_stats(fills)
    assert closed == 2
    assert win_rate == pytest.approx(0.5)
    assert realised == pytest.approx(Decimal("0"))


def test_round_trips_match_fifo_across_partial_closes() -> None:
    fills = [
        _fill("SOL/USDT", "buy", 100.0, 1.0),
        _fill("SOL/USDT", "buy", 120.0, 1.0),
        _fill("SOL/USDT", "sell", 110.0, 1.5),  # closes lot 1 (+10), half lot 2 (-5)
    ]
    win_rate, closed, realised = round_trip_stats(fills)
    assert closed == 2
    assert win_rate == pytest.approx(0.5)
    assert realised == pytest.approx(Decimal("5"))  # +10 and -5


def test_commission_reduces_realised_pnl() -> None:
    fills = [
        _fill("BTC/USDT", "buy", 100.0, 1.0, commission=0.5),
        _fill("BTC/USDT", "sell", 110.0, 1.0, commission=0.5),
    ]
    _, _, realised = round_trip_stats(fills)
    assert realised == pytest.approx(Decimal("9"))


def test_win_rate_is_none_with_no_closed_round_trips() -> None:
    """An open position is not a result. Reporting 0.0% would read as
    'every trade lost'."""
    win_rate, closed, _ = round_trip_stats([_fill("BTC/USDT", "buy", 100.0, 1.0)])
    assert win_rate is None
    assert closed == 0


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_compute_returns_mark_to_market_total_return() -> None:
    metrics = compute(_series([100.0, 99.5, 98.85]), [])
    assert metrics.total_return == pytest.approx(Decimal("-1.15"))
    assert metrics.total_return_pct == pytest.approx(-1.15)
    assert metrics.max_drawdown == pytest.approx(0.0115)


def test_compute_daily_pnl_anchors_to_the_prior_close() -> None:
    points = [
        EquityPoint(time=datetime(2026, 7, 1, 23, tzinfo=UTC), equity=Decimal("100")),
        EquityPoint(time=datetime(2026, 7, 2, 9, tzinfo=UTC), equity=Decimal("101")),
        EquityPoint(time=datetime(2026, 7, 2, 15, tzinfo=UTC), equity=Decimal("102")),
    ]
    metrics = compute(points, [])
    assert metrics.daily_pnl == pytest.approx(Decimal("2"))  # vs the Jul-1 close
    assert metrics.daily_pnl_pct == pytest.approx(2.0)


def test_compute_on_empty_history_reports_gaps_not_zeros() -> None:
    metrics = compute([], [])
    assert metrics.total_return_pct is None
    assert metrics.daily_pnl_pct is None
    assert metrics.sharpe_ratio is None
    assert metrics.max_drawdown is None
    assert metrics.win_rate is None


def test_short_history_reports_returns_but_not_sharpe() -> None:
    """The exact shape of the live soak: real return and drawdown, honest
    silence on the statistic that needs a longer record."""
    metrics = compute(_series([100.0, 99.9, 99.94]), [_fill("BTC/USDT", "buy", 1.0, 1.0)])
    assert metrics.total_return_pct is not None
    assert metrics.max_drawdown is not None
    assert metrics.sharpe_ratio is None
    assert metrics.win_rate is None
