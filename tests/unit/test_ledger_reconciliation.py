"""Trade statistics are withheld when the fill ledger cannot support them.

The live defect this pins (measured 2026-09-12): a handful of closes never
persisted, leaving unmatched lots that FIFO paired against months-old
prices. The dashboard reported +$8.40 realised and a 34.1% win rate on an
account whose broker state said -$1.53. A wrong win rate is worse than an
absent one -- it flatters exactly the account that needs scrutiny.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from services.portfolio.metrics import (
    EquityPoint,
    LedgerFill,
    compute,
    ledger_net_positions,
    ledger_reconciles,
)


def _fill(symbol: str, side: str, price: str, qty: str) -> LedgerFill:
    return LedgerFill(
        symbol=symbol, side=side, price=Decimal(price), quantity=Decimal(qty)
    )


def _points(n: int = 5) -> list[EquityPoint]:
    base = datetime(2026, 9, 1, tzinfo=UTC)
    return [
        EquityPoint(time=base + timedelta(days=i), equity=Decimal("100") - i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_a_balanced_ledger_reconciles_with_a_flat_book() -> None:
    fills = [
        _fill("ADA/USDT", "buy", "0.21", "14"),
        _fill("ADA/USDT", "sell", "0.22", "14"),
    ]
    assert ledger_net_positions(fills)["ADA/USDT"] == Decimal("0")
    assert ledger_reconciles(fills, book={})


def test_an_open_position_reconciles_when_the_book_agrees() -> None:
    fills = [_fill("BTC/USDT", "buy", "70000", "0.00003")]
    assert ledger_reconciles(fills, book={"BTC/USDT": Decimal("0.00003")})


def test_a_missing_close_is_caught() -> None:
    """The live shape: buys outnumber sells, the broker is flat, and the
    ledger silently claims a position that does not exist."""
    fills = [
        _fill("ADA/USDT", "buy", "0.21", "14"),
        _fill("ADA/USDT", "sell", "0.21", "14"),
        _fill("ADA/USDT", "buy", "0.21", "14"),  # its close never persisted
    ]
    assert not ledger_reconciles(fills, book={})


def test_reconciliation_is_measured_in_notional_not_units() -> None:
    """A 0.5-unit drift is $0.10 on a token and $35,000 on BTC; only the
    second is a broken ledger."""
    token = [_fill("ADA/USDT", "buy", "0.20", "0.5")]
    assert ledger_reconciles(token, book={})  # $0.10 drift: dust
    btc = [_fill("BTC/USDT", "buy", "70000", "0.5")]
    assert not ledger_reconciles(btc, book={})  # $35k drift: broken


def test_marks_override_stale_last_fill_prices() -> None:
    fills = [_fill("BTC/USDT", "buy", "0.001", "3")]  # absurd stale price
    assert not ledger_reconciles(
        fills, book={}, marks={"BTC/USDT": Decimal("70000")}
    )


# ---------------------------------------------------------------------------
# The served metrics
# ---------------------------------------------------------------------------


def test_unreconciled_ledger_withholds_trade_stats_but_keeps_equity_metrics() -> None:
    fills = [
        _fill("ADA/USDT", "buy", "0.21", "14"),
        _fill("ADA/USDT", "sell", "0.25", "14"),
        _fill("BTC/USDT", "buy", "70000", "0.01"),  # phantom: book is flat
    ]
    metrics = compute(_points(), fills, book={})
    assert metrics.win_rate is None
    assert metrics.closed_trades == 0
    assert "does not reconcile" in (metrics.trade_stats_unavailable or "")
    # Equity-derived metrics come from the snapshot series and must survive.
    assert metrics.total_return == Decimal("-4")


def test_reconciled_ledger_reports_trade_stats_normally() -> None:
    fills = [
        _fill("ADA/USDT", "buy", "0.21", "14"),
        _fill("ADA/USDT", "sell", "0.25", "14"),  # a winner
        _fill("ADA/USDT", "buy", "0.30", "14"),
        _fill("ADA/USDT", "sell", "0.28", "14"),  # a loser
    ]
    metrics = compute(_points(), fills, book={})
    assert metrics.trade_stats_unavailable is None
    assert metrics.closed_trades == 2
    assert metrics.win_rate == 0.5


def test_omitting_the_book_preserves_the_previous_contract() -> None:
    """Callers that cannot supply a book (tests, tools) still get stats."""
    fills = [
        _fill("ADA/USDT", "buy", "0.21", "14"),
        _fill("ADA/USDT", "sell", "0.25", "14"),
    ]
    metrics = compute(_points(), fills)
    assert metrics.win_rate == 1.0
    assert metrics.trade_stats_unavailable is None
