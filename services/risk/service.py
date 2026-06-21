"""Main risk management service -- orchestrates all risk checks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from core.enums import OrderSide
from core.events.base import Event, EventBus
from core.events.order_events import OrderFilledEvent
from core.events.risk_events import (
    RebalanceRequestEvent,
    RiskApprovedEvent,
    RiskBreachedEvent,
)
from core.events.streams import ORDERS, REBALANCE, RISK_APPROVED, RISK_BREACHED
from services.risk.circuit_breaker import CircuitBreaker
from services.risk.correlation_monitor import CorrelationMonitor
from services.risk.drawdown_monitor import DrawdownMonitor, DrawdownState
from services.risk.position_sizer import (
    FixedFractionalSizer,
    PositionSizer,
)
from services.risk.rules import (
    CircuitBreakerRule,
    MaxConcentrationRule,
    MaxCorrelationRule,
    MaxDrawdownRule,
    MaxPositionSizeRule,
    MaxVaRRule,
    RiskRule,
    RuleResult,
)
from services.risk.var_calculator import VaRCalculator

logger = logging.getLogger(__name__)

# Minimum trade size (USD) below which a rebalance weight delta is ignored.
_MIN_TRADE_NOTIONAL = 1.0


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class RiskDecision:
    """Result of a pre-trade risk evaluation."""

    approved: bool
    rejections: list[str] = field(default_factory=list)
    sized_quantity: Decimal = Decimal("0")


@dataclass
class RiskReport:
    """Comprehensive snapshot of all portfolio risk metrics."""

    drawdown_state: DrawdownState | None = None
    portfolio_var_95: float = 0.0
    portfolio_cvar_95: float = 0.0
    correlation_violations: int = 0
    hhi: float = 0.0
    effective_positions: float = 0.0
    max_weight: float = 0.0
    circuit_breaker_state: str = "closed"
    daily_pnl_pct: float = 0.0
    rule_results: list[RuleResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class RiskManagerService:
    """Central risk management service.

    Subscribes to ``RebalanceRequestEvent`` and ``OrderFilledEvent`` on
    the event bus and performs pre-trade risk checks and post-trade
    portfolio updates.

    Parameters
    ----------
    event_bus:
        Shared event bus for publishing/subscribing.
    settings:
        Application settings containing risk thresholds.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.settings = settings or get_settings()

        # Components
        self.circuit_breaker = CircuitBreaker(
            loss_threshold=self.settings.circuit_breaker_loss_pct,
            cooldown_minutes=self.settings.circuit_breaker_cooldown_minutes,
        )
        self.drawdown_monitor = DrawdownMonitor(
            max_daily_drawdown=self.settings.max_daily_drawdown_pct,
            max_total_drawdown=self.settings.max_total_drawdown_pct,
        )
        self.var_calculator = VaRCalculator()
        self.correlation_monitor = CorrelationMonitor()
        self.position_sizer: PositionSizer = FixedFractionalSizer(
            fraction=self.settings.max_single_order_pct,
        )

        # Rules
        self._rules: list[RiskRule] = [
            MaxPositionSizeRule(),
            MaxDrawdownRule(),
            MaxCorrelationRule(),
            MaxConcentrationRule(),
            MaxVaRRule(),
            CircuitBreakerRule(),
        ]

        # Portfolio state (updated via post_trade_update)
        self._positions: dict[str, float] = {}  # symbol -> dollar value
        self._equity: Decimal = self.settings.initial_capital
        self._daily_pnl_pct: float = 0.0
        self._returns_history: dict[str, list[float]] = {}  # symbol -> returns
        # Equity at the start of the trading day, for daily-PnL computation.
        self._day_start_equity: float = float(self.settings.initial_capital)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant event streams."""
        await self.event_bus.subscribe(
            stream=REBALANCE,
            group="risk_manager",
            consumer="risk-1",
            handler=self._on_rebalance_request,
        )
        await self.event_bus.subscribe(
            stream=ORDERS,
            group="risk_manager",
            consumer="risk-1",
            handler=self._on_order_filled,
        )
        logger.info("RiskManagerService started.")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_rebalance_request(self, event: Event) -> None:
        """Risk-check each requested symbol, size approved weight deltas into
        share quantities, and publish a RiskApprovedEvent per approved order."""
        rebalance = RebalanceRequestEvent.model_validate(event.model_dump())
        logger.info(
            "Received RebalanceRequestEvent for %d symbols.",
            len(rebalance.target_allocations),
        )

        total_equity = float(self._equity)
        if total_equity <= 0.0:
            logger.warning(
                "Rebalance skipped: non-positive equity (%.2f).", total_equity
            )
            return

        prices = rebalance.reference_prices or {}

        for symbol, target_weight in rebalance.target_allocations.items():
            decision = self.pre_trade_check(
                {"symbol": symbol, "target_weight": target_weight}
            )

            if not decision.approved:
                for rejection in decision.rejections:
                    await self.event_bus.publish(
                        RISK_BREACHED,
                        RiskBreachedEvent(
                            rule_name=rejection,
                            current_value=0.0,
                            limit_value=0.0,
                            action="block",
                            source_service="risk_manager",
                        ),
                    )
                continue

            # Size the weight delta into a share quantity. The risk manager has
            # no price feed, so it relies on the reference price supplied by the
            # producer (which has market data).
            price = float(prices.get(symbol, 0.0))
            if price <= 0.0:
                logger.warning("No reference price for %s; skipping sizing.", symbol)
                continue

            target_dollar = target_weight * total_equity
            current_dollar = self._positions.get(symbol, 0.0)
            delta_dollar = target_dollar - current_dollar
            notional = abs(delta_dollar)
            if notional < _MIN_TRADE_NOTIONAL:
                continue

            side = OrderSide.BUY if delta_dollar > 0 else OrderSide.SELL
            quantity = notional / price
            if quantity <= 0.0:
                continue

            await self.event_bus.publish(
                RISK_APPROVED,
                RiskApprovedEvent(
                    order_id=f"rebal-{event.event_id}-{symbol}",
                    symbol=symbol,
                    side=str(side),
                    order_type="market",
                    quantity=quantity,
                    reference_price=price,
                    source_service="risk_manager",
                ),
            )

    async def _on_order_filled(self, event: Event) -> None:
        """Handle an order fill by updating portfolio risk state.

        The execution engine publishes several event types to the shared
        ``orders`` stream; only fills update risk state.
        """
        if event.event_type != "OrderFilledEvent":
            return
        fill = OrderFilledEvent.model_validate(event.model_dump())
        self.post_trade_update(fill)

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def pre_trade_check(self, order_request: dict[str, Any]) -> RiskDecision:
        """Run all risk rules against a proposed order.

        Parameters
        ----------
        order_request:
            Dictionary with at least ``symbol`` and ``target_weight``.

        Returns
        -------
        RiskDecision
        """
        symbol = order_request.get("symbol", "")
        target_weight = float(order_request.get("target_weight", 0.0))

        # Build rule evaluation context
        total_equity = float(self._equity) if self._equity > 0 else 1.0
        proposed_position_pct = target_weight

        # Drawdown state
        dd_state = self.drawdown_monitor.update(total_equity)

        # Correlation
        returns_dict = {
            s: np.array(r)
            for s, r in self._returns_history.items()
            if len(r) >= 2
        }
        if returns_dict:
            symbols_sorted = sorted(returns_dict.keys())
            corr_matrix = self.correlation_monitor.compute_correlation_matrix(returns_dict)
            violations = self.correlation_monitor.check_pairwise_limits(
                corr_matrix,
                symbols_sorted,
                max_correlation=self.settings.max_pairwise_correlation,
            )
            n_violations = len(violations)
        else:
            n_violations = 0

        # Concentration
        weights = {}
        for s, v in self._positions.items():
            if total_equity > 0:
                weights[s] = v / total_equity
        # Include proposed position
        weights[symbol] = target_weight
        concentration = self.correlation_monitor.portfolio_concentration(weights)
        # Largest position as a fraction of *equity* (cash is the remainder, so
        # the weights need not sum to 1). Do not use portfolio_concentration's
        # renormalised max_weight here, or a single position always reads 100%.
        max_weight_frac = max(weights.values()) if weights else 0.0

        # VaR
        portfolio_var_pct = self._compute_portfolio_var_pct()

        context: dict[str, Any] = {
            # Position size
            "proposed_position_pct": proposed_position_pct,
            "max_position_pct": self.settings.max_position_pct,
            # Drawdown
            "current_drawdown_pct": dd_state.current_drawdown_pct,
            "max_total_drawdown_pct": self.settings.max_total_drawdown_pct,
            # Correlation
            "max_pairwise_correlation": self.settings.max_pairwise_correlation,
            "correlation_violations": n_violations,
            # Concentration
            "max_weight": max_weight_frac,
            "effective_positions": concentration["effective_positions"],
            # VaR
            "portfolio_var_pct": portfolio_var_pct,
            "max_portfolio_var_95": self.settings.max_portfolio_var_95,
            # Circuit breaker
            "circuit_breaker": self.circuit_breaker,
            "current_daily_pnl_pct": self._daily_pnl_pct,
        }

        rejections: list[str] = []
        for rule in self._rules:
            result = rule.check(context)
            if not result.passed:
                rejections.append(f"{result.rule_name}: {result.message}")
                logger.warning("Risk rule failed: %s", result.message)

        approved = len(rejections) == 0

        # Size the position if approved
        sized_quantity = Decimal("0")
        if approved:
            sized_quantity = self.position_sizer.size(equity=self._equity)

        decision = RiskDecision(
            approved=approved,
            rejections=rejections,
            sized_quantity=sized_quantity,
        )

        logger.info(
            "Pre-trade check for %s: approved=%s, rejections=%d",
            symbol,
            approved,
            len(rejections),
        )

        return decision

    # ------------------------------------------------------------------
    # Post-trade updates
    # ------------------------------------------------------------------

    def post_trade_update(self, fill: OrderFilledEvent) -> None:
        """Update internal risk state after an order fill.

        Parameters
        ----------
        fill:
            The fill event from the execution layer.
        """
        # Prefer the explicit symbol/side fields; fall back to payload/order_id.
        symbol = fill.symbol or fill.payload.get("symbol", fill.order_id)
        side = (fill.side or fill.payload.get("side", "")).lower()

        fill_value = fill.fill_price * fill.fill_quantity
        current = self._positions.get(symbol, 0.0)
        # Buys increase dollar exposure; sells reduce it.
        if side == "sell":
            self._positions[symbol] = current - fill_value
        else:
            self._positions[symbol] = current + fill_value

        # Deduct commission from equity
        self._equity -= Decimal(str(fill.commission))

        # Update drawdown tracking with the latest equity.
        self.drawdown_monitor.update(float(self._equity))

        logger.debug(
            "Post-trade update: %s filled %.4f @ %.4f (commission=%.4f)",
            symbol,
            fill.fill_quantity,
            fill.fill_price,
            fill.commission,
        )

    # ------------------------------------------------------------------
    # Portfolio risk report
    # ------------------------------------------------------------------

    def check_portfolio_risk(self) -> RiskReport:
        """Generate a comprehensive risk report for the current portfolio.

        Returns
        -------
        RiskReport
        """
        total_equity = float(self._equity) if self._equity > 0 else 1.0

        # Drawdown
        dd_state = self.drawdown_monitor.update(total_equity)

        # Correlation
        returns_dict = {
            s: np.array(r)
            for s, r in self._returns_history.items()
            if len(r) >= 2
        }
        n_violations = 0
        if returns_dict:
            symbols_sorted = sorted(returns_dict.keys())
            corr_matrix = self.correlation_monitor.compute_correlation_matrix(returns_dict)
            violations = self.correlation_monitor.check_pairwise_limits(
                corr_matrix,
                symbols_sorted,
                max_correlation=self.settings.max_pairwise_correlation,
            )
            n_violations = len(violations)

        # Concentration
        weights: dict[str, float] = {}
        for s, v in self._positions.items():
            if total_equity > 0:
                weights[s] = v / total_equity
        concentration = self.correlation_monitor.portfolio_concentration(weights)
        max_weight_frac = max(weights.values()) if weights else 0.0

        # VaR
        portfolio_var_pct = self._compute_portfolio_var_pct()
        portfolio_cvar_pct = self._compute_portfolio_cvar_pct()

        # Circuit breaker
        cb_decision = self.circuit_breaker.check(self._daily_pnl_pct)

        # Run all rules for the report
        context: dict[str, Any] = {
            "proposed_position_pct": max_weight_frac,
            "max_position_pct": self.settings.max_position_pct,
            "current_drawdown_pct": dd_state.current_drawdown_pct,
            "max_total_drawdown_pct": self.settings.max_total_drawdown_pct,
            "max_pairwise_correlation": self.settings.max_pairwise_correlation,
            "correlation_violations": n_violations,
            "max_weight": max_weight_frac,
            "effective_positions": concentration["effective_positions"],
            "portfolio_var_pct": portfolio_var_pct,
            "max_portfolio_var_95": self.settings.max_portfolio_var_95,
            "circuit_breaker": self.circuit_breaker,
            "current_daily_pnl_pct": self._daily_pnl_pct,
        }

        rule_results = [rule.check(context) for rule in self._rules]

        return RiskReport(
            drawdown_state=dd_state,
            portfolio_var_95=portfolio_var_pct * total_equity,
            portfolio_cvar_95=portfolio_cvar_pct * total_equity,
            correlation_violations=n_violations,
            hhi=concentration["hhi"],
            effective_positions=concentration["effective_positions"],
            max_weight=max_weight_frac,
            circuit_breaker_state=cb_decision.state.value,
            daily_pnl_pct=self._daily_pnl_pct,
            rule_results=rule_results,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def update_equity(self, equity: Decimal) -> None:
        """Externally set the current equity (e.g. from account sync)."""
        self._equity = equity

    def update_daily_pnl(self, pnl_pct: float) -> None:
        """Set the current daily PnL percentage."""
        self._daily_pnl_pct = pnl_pct

    def sync_account(self, equity: Decimal | float) -> None:
        """Feed the live broker account into the risk engine.

        Sets current equity, recomputes daily PnL versus the day's starting
        equity, and advances the drawdown monitor -- so the drawdown rule and
        circuit breaker act on real account state, not stale/empty values. A
        worker should call this periodically with ``broker.get_account()``.
        """
        self._equity = Decimal(str(equity))
        eq = float(self._equity)
        if self._day_start_equity > 0:
            self._daily_pnl_pct = (eq - self._day_start_equity) / self._day_start_equity
        self.drawdown_monitor.update(eq)

    async def run_account_sync(
        self,
        account_provider: Callable[[], Awaitable[Decimal | float | None]],
        interval_seconds: float = 30.0,
    ) -> None:
        """Background loop: periodically pull live equity via *account_provider*
        and feed it into the risk engine, so the drawdown monitor and circuit
        breaker track real capital. Runs until cancelled."""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                equity = await account_provider()
                if equity is not None:
                    self.sync_account(equity)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Account sync iteration failed")

    def update_returns(self, symbol: str, returns: list[float]) -> None:
        """Set the historical return series for a symbol."""
        self._returns_history[symbol] = returns

    def update_position(self, symbol: str, value: float) -> None:
        """Set the current dollar value for a position."""
        if value <= 0.0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = value

    def reset_daily(self) -> None:
        """Reset daily counters at the start of a new trading day."""
        self._daily_pnl_pct = 0.0
        self._day_start_equity = float(self._equity)
        self.drawdown_monitor.reset_daily(float(self._equity))
        self.circuit_breaker.reset()
        logger.info("Risk manager daily reset complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_portfolio_var_pct(self) -> float:
        """Compute portfolio VaR as a percentage of equity."""
        all_returns = self._aggregate_portfolio_returns()
        if len(all_returns) < 2:
            return 0.0

        var_dollar = self.var_calculator.historical_var(
            all_returns,
            confidence=0.95,
            portfolio_value=1.0,  # returns are already pct
        )
        return var_dollar

    def _compute_portfolio_cvar_pct(self) -> float:
        """Compute portfolio CVaR as a percentage of equity."""
        all_returns = self._aggregate_portfolio_returns()
        if len(all_returns) < 2:
            return 0.0

        cvar_dollar = self.var_calculator.cvar(
            all_returns,
            confidence=0.95,
            portfolio_value=1.0,
        )
        return cvar_dollar

    def _aggregate_portfolio_returns(self) -> np.ndarray:
        """Compute weighted portfolio returns from per-asset returns."""
        total_equity = float(self._equity) if float(self._equity) > 0 else 1.0

        if not self._returns_history or not self._positions:
            return np.array([])

        # Find common length
        lengths = [
            len(r)
            for s, r in self._returns_history.items()
            if s in self._positions and len(r) > 0
        ]
        if not lengths:
            return np.array([])

        min_len = min(lengths)
        if min_len < 2:
            return np.array([])

        portfolio_returns = np.zeros(min_len)
        for symbol, returns in self._returns_history.items():
            if symbol not in self._positions:
                continue
            weight = self._positions[symbol] / total_equity
            asset_returns = np.array(returns[-min_len:])
            portfolio_returns += weight * asset_returns

        return portfolio_returns
