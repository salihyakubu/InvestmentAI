"""The risk engine must consume live account equity so the drawdown monitor and
circuit breaker act on real capital. Previously ``update_equity``/daily-PnL were
never fed from the broker account, leaving those controls blind.
"""

from __future__ import annotations

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
