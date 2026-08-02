"""Gauntlet glue: date-aligned pooling, beta control, era slicing.

The heavy machinery (evaluate_symbol, deflated_sharpe) is tested elsewhere;
these pin the three pieces of glue where a silent bug would corrupt the
audit itself.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gauntlet_daily_harness",
    Path(__file__).resolve().parents[2] / "scripts" / "gauntlet_daily_harness.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _dates(n: int, offset_days: int = 0) -> list[datetime]:
    return [_BASE + timedelta(days=offset_days + 5 * i) for i in range(n)]


def test_pooling_aligns_on_calendar_dates_not_array_position() -> None:
    """Two symbols whose OOS windows start on different dates must only be
    averaged where their periods actually coincide."""
    a_dates, b_dates = _dates(4), _dates(4, offset_days=10)  # overlap on 2
    a = np.array([0.01, 0.02, 0.03, 0.04])
    b = np.array([0.10, 0.20, 0.30, 0.40])
    dates, pooled = _MOD.pool_by_date({"A": (a_dates, a), "B": (b_dates, b)})
    assert len(dates) == 6  # 4 + 4 - 2 overlapping
    by_date = dict(zip(dates, pooled, strict=True))
    # Overlapping dates average the two symbols; lone dates pass through.
    assert by_date[a_dates[2]] == pytest.approx((0.03 + 0.10) / 2)
    assert by_date[a_dates[0]] == pytest.approx(0.01)
    assert by_date[b_dates[-1]] == pytest.approx(0.40)


def test_era_slice_is_strictly_after_the_boundary() -> None:
    dates = _dates(5)
    returns = np.arange(5, dtype=float)
    cut = dates[2]
    sliced = _MOD.era_slice(dates, returns, cut)
    # Strictly after: the boundary period itself belongs to the seen era.
    assert list(sliced) == [3.0, 4.0]


def test_buy_and_hold_control_is_costless_and_positive_in_a_bull_run() -> None:
    """The control the strategy must beat: monotonically rising prices give
    buy-and-hold a large positive Sharpe."""
    prices = 100.0 * np.cumprod(np.full(50, 1.01))
    assert _MOD.buy_and_hold_sharpe(prices) > 5.0


def test_buy_and_hold_needs_enough_periods() -> None:
    assert _MOD.buy_and_hold_sharpe(np.array([100.0, 101.0])) == 0.0


def test_registered_universe_is_unchanged() -> None:
    """The universe is fixed in the pre-registration (GO_LIVE.md 2026-08-02).
    If this test fails, someone edited the universe after registration --
    which is exactly the move the registration exists to prevent."""
    # 65 names: the registration PROSE said "three plus 61" (an off-by-one
    # in prose); the registered LIST -- which is authoritative -- has 65 and
    # matches GO_LIVE.md verbatim. Recorded in the results section.
    assert len(_MOD.UNIVERSE) == 65
    assert _MOD.UNIVERSE[:3] == ["AAPL", "MSFT", "SPY"]
    assert _MOD.N_TRIALS == 24
    assert _MOD.ORIGINAL_END == datetime(2025, 12, 31, tzinfo=UTC)
