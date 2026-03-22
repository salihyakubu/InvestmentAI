"""Unit tests for the risk management service."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from config.settings import Settings
from core.events.base import InProcessEventBus
from services.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from services.risk.position_sizer import KellyCriterionSizer
from services.risk.service import RiskManagerService
from services.risk.var_calculator import VaRCalculator


class TestPositionSizeLimit:
    """Verify that the risk manager rejects orders that exceed position limits."""

    def test_position_size_limit(
        self, event_bus: InProcessEventBus, mock_settings: Settings
    ) -> None:
        """An order requesting more than max_position_pct of equity
        should be rejected."""
        svc = RiskManagerService(event_bus=event_bus, settings=mock_settings)
        svc.update_equity(Decimal("10000"))
        svc.drawdown_monitor.update(10000.0)

        # Request 20% allocation -- exceeds 10% limit.
        decision = svc.pre_trade_check(
            {"symbol": "AAPL", "target_weight": 0.20}
        )
        assert not decision.approved
        assert any("MaxPositionSizeRule" in r for r in decision.rejections)


class TestDrawdownHalt:
    """Verify trading halts when drawdown limits are hit."""

    def test_drawdown_halt(
        self, event_bus: InProcessEventBus, mock_settings: Settings
    ) -> None:
        """Trading should be blocked once total drawdown exceeds the
        configured limit."""
        svc = RiskManagerService(event_bus=event_bus, settings=mock_settings)

        # Start at 10000, then drop to 8000 => 20% drawdown > 15% limit.
        svc.update_equity(Decimal("10000"))
        svc.drawdown_monitor.update(10000.0)
        svc.drawdown_monitor.update(8000.0)

        decision = svc.pre_trade_check(
            {"symbol": "AAPL", "target_weight": 0.05}
        )
        assert not decision.approved
        assert any("MaxDrawdownRule" in r for r in decision.rejections)


class TestCircuitBreakerTrip:
    """Verify that the circuit breaker trips on large daily losses."""

    def test_circuit_breaker_trip(
        self, event_bus: InProcessEventBus, mock_settings: Settings
    ) -> None:
        """When daily PnL loss exceeds threshold, circuit breaker
        should trip and block trading."""
        svc = RiskManagerService(event_bus=event_bus, settings=mock_settings)
        svc.update_equity(Decimal("10000"))
        svc.drawdown_monitor.update(10000.0)

        # Set daily PnL to -8% (threshold is 7%).
        svc.update_daily_pnl(-0.08)

        decision = svc.pre_trade_check(
            {"symbol": "AAPL", "target_weight": 0.05}
        )
        assert not decision.approved
        assert svc.circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_cooldown(self) -> None:
        """After cooldown, breaker should transition to HALF_OPEN."""
        cb = CircuitBreaker(loss_threshold=0.07, cooldown_minutes=0)
        cb.trip("test loss")

        assert cb.state == CircuitBreakerState.OPEN

        # With zero-minute cooldown, attempt_reset should succeed
        # immediately.
        reset = cb.attempt_reset()
        assert reset is True
        assert cb.state == CircuitBreakerState.HALF_OPEN


class TestKellyCriterionSizing:
    """Verify the Kelly criterion position sizer."""

    def test_kelly_criterion_sizing(self) -> None:
        """With a 60% win rate and 1.5:1 win/loss ratio, Kelly
        should produce a positive allocation."""
        sizer = KellyCriterionSizer(kelly_fraction=0.25)

        size = sizer.size(
            win_rate=0.60,
            win_loss_ratio=1.5,
            equity=Decimal("10000"),
            max_pct=0.10,
        )

        # Kelly pct = 0.60 - (0.40 / 1.5) = 0.3333
        # Quarter Kelly = 0.0833
        # Capped at 0.10 -> stays at 0.0833
        assert size > Decimal("0")
        assert size <= Decimal("10000") * Decimal("0.10")


class TestVaRHistorical:
    """Verify historical VaR calculation."""

    def test_var_historical(self) -> None:
        """Historical VaR at 95% confidence should be within a
        reasonable range for normal-like returns."""
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(loc=0.0005, scale=0.02, size=252)

        calc = VaRCalculator()
        var = calc.historical_var(returns, confidence=0.95, portfolio_value=10000.0)

        # VaR should be positive (it represents a loss amount).
        assert var > 0.0
        # For 2% daily vol, 95% VaR should roughly be around 2-4% of
        # portfolio value.
        assert var < 10000.0 * 0.10  # Sanity: less than 10% of portfolio
