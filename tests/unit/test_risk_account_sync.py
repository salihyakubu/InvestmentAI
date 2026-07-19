"""The risk engine must consume live account equity so the drawdown monitor and
circuit breaker act on real capital. Previously ``update_equity``/daily-PnL were
never fed from the broker account, leaving those controls blind.
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from config.settings import Settings
from core.events.base import InProcessEventBus
from services.risk.service import RiskManagerService


def _svc(mock_settings: Settings) -> RiskManagerService:
    svc = RiskManagerService(event_bus=InProcessEventBus(), settings=mock_settings)
    svc.update_equity(Decimal("10000"))
    svc.reset_daily()  # day-start equity = 10000, drawdown peak = 10000
    return svc


def test_sync_account_computes_daily_pnl(mock_settings: Settings) -> None:
    svc = _svc(mock_settings)
    svc.sync_account(Decimal("9500"))  # -5% on the day
    assert svc._daily_pnl_pct == pytest.approx(-0.05)


def test_sync_account_trips_circuit_breaker_on_large_loss(mock_settings: Settings) -> None:
    svc = _svc(mock_settings)
    svc.sync_account(Decimal("8000"))  # -20% intraday -> well past the breaker
    decision = svc.pre_trade_check({"symbol": "AAPL", "target_weight": 0.05})
    assert not decision.approved
    assert any("CircuitBreaker" in r for r in decision.rejections)


def test_sync_account_healthy_allows_trading(mock_settings: Settings) -> None:
    svc = _svc(mock_settings)
    svc.sync_account(Decimal("10050"))  # +0.5%, no breach
    decision = svc.pre_trade_check({"symbol": "AAPL", "target_weight": 0.05})
    assert decision.approved


@pytest.mark.asyncio
async def test_account_sync_loop_feeds_risk(mock_settings: Settings) -> None:
    svc = _svc(mock_settings)
    calls = {"n": 0}

    async def provider() -> Decimal:
        calls["n"] += 1
        return Decimal("8000")  # -20% intraday

    task = asyncio.create_task(svc.run_account_sync(provider, interval_seconds=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert calls["n"] >= 1
    assert svc._daily_pnl_pct == pytest.approx(-0.20)
    # The synced loss must have tripped the circuit breaker.
    decision = svc.pre_trade_check({"symbol": "AAPL", "target_weight": 0.05})
    assert not decision.approved


async def _wait_for_calls(calls: dict[str, int], at_least: int) -> None:
    """Poll until the provider has been invoked *at_least* times (max ~2s)."""
    for _ in range(200):
        if calls["n"] >= at_least:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"provider reached only {calls['n']} calls, wanted {at_least}")


@pytest.mark.asyncio
async def test_account_sync_no_daily_reset_while_date_unchanged(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _svc(mock_settings)
    monkeypatch.setattr(
        "services.risk.service._utc_today", lambda: date(2026, 7, 19)
    )

    resets = {"n": 0}
    original_reset = svc.reset_daily

    def spy_reset() -> None:
        resets["n"] += 1
        original_reset()

    monkeypatch.setattr(svc, "reset_daily", spy_reset)

    calls = {"n": 0}

    async def provider() -> Decimal:
        calls["n"] += 1
        return Decimal("10000")

    task = asyncio.create_task(svc.run_account_sync(provider, interval_seconds=0.01))
    await _wait_for_calls(calls, 3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert resets["n"] == 0


@pytest.mark.asyncio
async def test_account_sync_resets_daily_once_when_utc_date_advances(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _svc(mock_settings)
    fake_today = {"value": date(2026, 7, 19)}
    monkeypatch.setattr(
        "services.risk.service._utc_today", lambda: fake_today["value"]
    )

    resets = {"n": 0}
    original_reset = svc.reset_daily

    def spy_reset() -> None:
        resets["n"] += 1
        original_reset()

    monkeypatch.setattr(svc, "reset_daily", spy_reset)

    calls = {"n": 0}

    async def provider() -> Decimal:
        calls["n"] += 1
        return Decimal("9000")  # -10% on the day until the rollover

    task = asyncio.create_task(svc.run_account_sync(provider, interval_seconds=0.01))
    await _wait_for_calls(calls, 2)
    assert resets["n"] == 0
    assert svc._daily_pnl_pct == pytest.approx(-0.10)

    # Advance the UTC calendar date: the very next iteration must reset once.
    fake_today["value"] = date(2026, 7, 20)
    seen = calls["n"]
    await _wait_for_calls(calls, seen + 3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert resets["n"] == 1
    # The reset re-baselined day-start equity to current equity (9000), so the
    # subsequent syncs at 9000 report a flat day, not a -10% one.
    assert svc._day_start_equity == pytest.approx(9000.0)
    assert svc._daily_pnl_pct == pytest.approx(0.0)
