"""Unit tests for the circuit breaker state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
)


@pytest.fixture
def breaker() -> CircuitBreaker:
    """Return a circuit breaker with 7% threshold and 30-minute cooldown."""
    return CircuitBreaker(loss_threshold=0.07, cooldown_minutes=30)


class TestInitialState:
    """Verify the circuit breaker starts in the correct state."""

    def test_initial_state_closed(self, breaker: CircuitBreaker) -> None:
        """A freshly created circuit breaker should be in CLOSED state."""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.tripped_at is None


class TestTripping:
    """Verify the circuit breaker trips correctly."""

    def test_trip_on_threshold(self, breaker: CircuitBreaker) -> None:
        """A daily loss exceeding the threshold should trip the breaker."""
        decision = breaker.check(current_daily_pnl_pct=-0.08)

        assert breaker.state == CircuitBreakerState.OPEN
        assert not decision.allow_trading
        assert breaker.tripped_at is not None


class TestCooldown:
    """Verify cooldown behaviour."""

    def test_cooldown_period(self, breaker: CircuitBreaker) -> None:
        """The breaker should remain OPEN during the cooldown window."""
        breaker.trip("test loss")

        # Immediately after tripping, attempt_reset should fail.
        reset = breaker.attempt_reset()
        assert reset is False
        assert breaker.state == CircuitBreakerState.OPEN


class TestHalfOpen:
    """Verify the HALF_OPEN state."""

    def test_half_open_state(self, breaker: CircuitBreaker) -> None:
        """After cooldown elapses, the breaker should transition to
        HALF_OPEN and allow trading with caution."""
        breaker.trip("test loss")

        # Simulate time passage beyond cooldown.
        past = datetime.now(timezone.utc) - timedelta(minutes=31)
        breaker._tripped_at = past

        reset = breaker.attempt_reset()
        assert reset is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Trading should be allowed in HALF_OPEN.
        decision = breaker.check(current_daily_pnl_pct=-0.02)
        assert decision.allow_trading is True


class TestReset:
    """Verify manual reset to CLOSED state."""

    def test_reset_to_closed(self, breaker: CircuitBreaker) -> None:
        """Manual reset should return the breaker to CLOSED state."""
        breaker.trip("test loss")
        assert breaker.state == CircuitBreakerState.OPEN

        breaker.reset()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.tripped_at is None

        # Trading should be allowed again.
        decision = breaker.check(current_daily_pnl_pct=-0.01)
        assert decision.allow_trading is True
