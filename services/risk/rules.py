"""Composable risk rules evaluated during pre-trade checks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from services.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Outcome of evaluating a single risk rule."""

    passed: bool
    rule_name: str
    message: str
    current_value: float
    limit_value: float


class RiskRule(ABC):
    """Abstract base class for a single risk check."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""

    @abstractmethod
    def check(self, context: dict[str, Any]) -> RuleResult:
        """Evaluate the rule against the provided context.

        Parameters
        ----------
        context:
            Dictionary containing all data required by risk rules.
            Expected keys vary per rule; see each rule's docstring.

        Returns
        -------
        RuleResult
        """


class MaxPositionSizeRule(RiskRule):
    """Reject orders that would make a single position exceed max weight.

    Context keys:
        - ``proposed_position_pct``: float -- weight of the position after
          the proposed order.
        - ``max_position_pct``: float -- hard limit.
    """

    @property
    def name(self) -> str:
        return "MaxPositionSize"

    def check(self, context: dict[str, Any]) -> RuleResult:
        proposed = float(context.get("proposed_position_pct", 0.0))
        limit = float(context.get("max_position_pct", 0.10))

        passed = proposed <= limit
        return RuleResult(
            passed=passed,
            rule_name=self.name,
            message=(
                f"Position size {proposed:.2%} within limit {limit:.2%}."
                if passed
                else f"Position size {proposed:.2%} exceeds limit {limit:.2%}."
            ),
            current_value=proposed,
            limit_value=limit,
        )


class MaxDrawdownRule(RiskRule):
    """Block trading when drawdown exceeds the configured maximum.

    Context keys:
        - ``current_drawdown_pct``: float
        - ``max_total_drawdown_pct``: float
    """

    @property
    def name(self) -> str:
        return "MaxDrawdown"

    def check(self, context: dict[str, Any]) -> RuleResult:
        current = float(context.get("current_drawdown_pct", 0.0))
        limit = float(context.get("max_total_drawdown_pct", 0.15))

        passed = current < limit
        return RuleResult(
            passed=passed,
            rule_name=self.name,
            message=(
                f"Drawdown {current:.2%} within limit {limit:.2%}."
                if passed
                else f"Drawdown {current:.2%} exceeds limit {limit:.2%}."
            ),
            current_value=current,
            limit_value=limit,
        )


class MaxCorrelationRule(RiskRule):
    """Flag when proposed position is too correlated with existing holdings.

    Context keys:
        - ``max_pairwise_correlation``: float
        - ``correlation_violations``: int -- number of pairs exceeding the limit.
    """

    @property
    def name(self) -> str:
        return "MaxCorrelation"

    def check(self, context: dict[str, Any]) -> RuleResult:
        limit = float(context.get("max_pairwise_correlation", 0.85))
        violations = int(context.get("correlation_violations", 0))

        passed = violations == 0
        return RuleResult(
            passed=passed,
            rule_name=self.name,
            message=(
                "No pairwise correlation violations."
                if passed
                else f"{violations} pair(s) exceed correlation limit {limit:.2f}."
            ),
            current_value=float(violations),
            limit_value=0.0,
        )


class MaxConcentrationRule(RiskRule):
    """Reject when the portfolio is too concentrated (HHI or max weight).

    Context keys:
        - ``max_weight``: float -- largest single position weight.
        - ``max_position_pct``: float -- per-position cap.
        - ``effective_positions``: float
    """

    @property
    def name(self) -> str:
        return "MaxConcentration"

    def check(self, context: dict[str, Any]) -> RuleResult:
        max_weight = float(context.get("max_weight", 0.0))
        limit = float(context.get("max_position_pct", 0.10))

        passed = max_weight <= limit
        return RuleResult(
            passed=passed,
            rule_name=self.name,
            message=(
                f"Max position weight {max_weight:.2%} within limit {limit:.2%}."
                if passed
                else f"Max position weight {max_weight:.2%} exceeds limit {limit:.2%}."
            ),
            current_value=max_weight,
            limit_value=limit,
        )


class MaxVaRRule(RiskRule):
    """Block when portfolio VaR exceeds the configured ceiling.

    Context keys:
        - ``portfolio_var_pct``: float -- portfolio VaR as fraction of equity.
        - ``max_portfolio_var_95``: float -- limit.
    """

    @property
    def name(self) -> str:
        return "MaxVaR"

    def check(self, context: dict[str, Any]) -> RuleResult:
        current = float(context.get("portfolio_var_pct", 0.0))
        limit = float(context.get("max_portfolio_var_95", 0.03))

        passed = current <= limit
        return RuleResult(
            passed=passed,
            rule_name=self.name,
            message=(
                f"Portfolio VaR {current:.2%} within limit {limit:.2%}."
                if passed
                else f"Portfolio VaR {current:.2%} exceeds limit {limit:.2%}."
            ),
            current_value=current,
            limit_value=limit,
        )


class CircuitBreakerRule(RiskRule):
    """Block all trading when the circuit breaker is OPEN.

    Context keys:
        - ``circuit_breaker``: CircuitBreaker instance.
        - ``current_daily_pnl_pct``: float -- today's PnL as fraction.
    """

    @property
    def name(self) -> str:
        return "CircuitBreaker"

    def check(self, context: dict[str, Any]) -> RuleResult:
        cb: CircuitBreaker | None = context.get("circuit_breaker")
        daily_pnl = float(context.get("current_daily_pnl_pct", 0.0))

        if cb is None:
            return RuleResult(
                passed=True,
                rule_name=self.name,
                message="No circuit breaker configured.",
                current_value=0.0,
                limit_value=0.0,
            )

        decision = cb.check(daily_pnl)

        return RuleResult(
            passed=decision.allow_trading,
            rule_name=self.name,
            message=decision.reason,
            current_value=daily_pnl,
            limit_value=-cb.loss_threshold,
        )
