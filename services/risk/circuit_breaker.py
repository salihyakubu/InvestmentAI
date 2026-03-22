"""Circuit breaker state machine for halting trading during adverse conditions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitBreakerState(StrEnum):
    """Possible circuit breaker states."""

    CLOSED = "closed"       # Normal operation -- trading allowed
    OPEN = "open"           # Halted -- trading blocked
    HALF_OPEN = "half_open" # Cooldown expired -- testing with caution


@dataclass
class CircuitBreakerDecision:
    """Result of a circuit breaker check."""

    allow_trading: bool
    state: CircuitBreakerState
    reason: str


class CircuitBreaker:
    """Circuit breaker that trips when daily losses exceed a threshold.

    State machine:

        CLOSED  --(loss >= threshold)--> OPEN
        OPEN    --(cooldown expires)---> HALF_OPEN
        HALF_OPEN --(manual reset)-----> CLOSED
        HALF_OPEN --(loss again)-------> OPEN

    Parameters
    ----------
    loss_threshold:
        Daily PnL percentage loss that trips the breaker (e.g. 0.07 = 7 %).
    cooldown_minutes:
        Minutes to wait before transitioning from OPEN to HALF_OPEN.
    """

    def __init__(
        self,
        loss_threshold: float,
        cooldown_minutes: int = 30,
    ) -> None:
        if loss_threshold <= 0.0 or loss_threshold > 1.0:
            raise ValueError(f"loss_threshold must be in (0, 1], got {loss_threshold}")
        if cooldown_minutes < 0:
            raise ValueError(f"cooldown_minutes must be >= 0, got {cooldown_minutes}")

        self.loss_threshold = loss_threshold
        self.cooldown_minutes = cooldown_minutes

        self._state = CircuitBreakerState.CLOSED
        self._tripped_at: datetime | None = None
        self._trip_reason: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def tripped_at(self) -> datetime | None:
        return self._tripped_at

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self, current_daily_pnl_pct: float) -> CircuitBreakerDecision:
        """Evaluate whether trading should be allowed.

        Parameters
        ----------
        current_daily_pnl_pct:
            Current daily PnL as a fraction (negative means loss).

        Returns
        -------
        CircuitBreakerDecision
        """
        # Auto-trip if loss exceeds threshold (loss is negative PnL)
        if current_daily_pnl_pct <= -self.loss_threshold:
            if self._state == CircuitBreakerState.CLOSED:
                self.trip(
                    f"Daily loss {current_daily_pnl_pct:.2%} exceeded "
                    f"threshold {-self.loss_threshold:.2%}"
                )
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self.trip(
                    f"Loss continued in HALF_OPEN state: {current_daily_pnl_pct:.2%}"
                )

        # Attempt auto-reset from OPEN to HALF_OPEN if cooldown has elapsed
        if self._state == CircuitBreakerState.OPEN:
            self.attempt_reset()

        # Build the decision
        if self._state == CircuitBreakerState.CLOSED:
            return CircuitBreakerDecision(
                allow_trading=True,
                state=self._state,
                reason="Circuit breaker closed; trading allowed.",
            )
        elif self._state == CircuitBreakerState.HALF_OPEN:
            return CircuitBreakerDecision(
                allow_trading=True,
                state=self._state,
                reason=(
                    "Circuit breaker half-open after cooldown; "
                    "trading allowed with caution."
                ),
            )
        else:
            remaining = self._remaining_cooldown_seconds()
            return CircuitBreakerDecision(
                allow_trading=False,
                state=self._state,
                reason=(
                    f"Circuit breaker OPEN: {self._trip_reason}. "
                    f"Cooldown remaining: {remaining:.0f}s."
                ),
            )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def trip(self, reason: str) -> None:
        """Transition to OPEN state and record the timestamp."""
        previous = self._state
        self._state = CircuitBreakerState.OPEN
        self._tripped_at = datetime.now(timezone.utc)
        self._trip_reason = reason

        logger.warning(
            "Circuit breaker tripped: %s -> OPEN. Reason: %s",
            previous,
            reason,
        )

    def attempt_reset(self) -> bool:
        """Transition from OPEN to HALF_OPEN if the cooldown has elapsed.

        Returns
        -------
        bool
            True if the transition occurred.
        """
        if self._state != CircuitBreakerState.OPEN:
            return False

        if self._tripped_at is None:
            return False

        elapsed = (datetime.now(timezone.utc) - self._tripped_at).total_seconds()
        cooldown_seconds = self.cooldown_minutes * 60

        if elapsed >= cooldown_seconds:
            self._state = CircuitBreakerState.HALF_OPEN
            logger.info(
                "Circuit breaker cooldown elapsed (%.0fs). "
                "Transitioning OPEN -> HALF_OPEN.",
                elapsed,
            )
            return True

        return False

    def reset(self) -> None:
        """Manually transition to CLOSED state."""
        previous = self._state
        self._state = CircuitBreakerState.CLOSED
        self._tripped_at = None
        self._trip_reason = ""

        logger.info("Circuit breaker reset: %s -> CLOSED.", previous)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remaining_cooldown_seconds(self) -> float:
        """Return seconds remaining in the cooldown period."""
        if self._tripped_at is None:
            return 0.0

        elapsed = (datetime.now(timezone.utc) - self._tripped_at).total_seconds()
        remaining = max(self.cooldown_minutes * 60 - elapsed, 0.0)
        return remaining
