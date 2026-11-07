"""Main risk management service -- orchestrates all risk checks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import numpy as np

from config.settings import Settings, get_settings
from core.enums import OrderSide
from core.events.base import Event, EventBus
from core.events.order_events import OrderFilledEvent
from core.events.risk_events import (
    CircuitBreakerEvent,
    RebalanceRequestEvent,
    RiskApprovedEvent,
    RiskBreachedEvent,
)
from core.events.streams import ORDERS, REBALANCE, RISK_APPROVED, RISK_BREACHED
from services.risk.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from services.persistence.safety_state import SafetyStateStore
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


def _utc_today() -> date:
    """Current UTC calendar date (module-level so tests can monkeypatch it)."""
    return datetime.now(UTC).date()


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
    position_provider:
        Async callable returning the broker's open positions (the shape of
        ``BaseBroker.get_positions``). Exits are sized from its signed
        quantities; without it the engine falls back to the dollar book
        divided by the reference price, which leaves dust when the price has
        moved (GO_LIVE.md 2026-11-07).
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
        state_store: SafetyStateStore | None = None,
        position_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.settings = settings or get_settings()
        # Durable daily baseline / peak / breaker state (None = memory only).
        self._state_store = state_store
        self._position_provider = position_provider

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
        # Seeded from the FIRST account sync (or restored), not from
        # initial_capital: a $92 account booting against a $100 baseline read
        # -8% on the day and tripped the breaker on every restart.
        self._day_start_equity: float = float(self.settings.initial_capital)
        self._day_start_pending: bool = True
        self._session_date: date = _utc_today()
        # Last breaker state published; a transition publishes exactly once.
        self._published_breaker_state: str = self.circuit_breaker.state.value

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
        # The venue's signed book, when a provider is wired: exits are sized
        # from it. None means "no provider" (or a failed call, logged): sizing
        # falls back to the dollar book, which is how dust used to arise.
        held = await self._held_quantities()

        for symbol, target_weight in rebalance.target_allocations.items():
            price = float(prices.get(symbol, 0.0))
            target_dollar = target_weight * total_equity
            current_dollar = self._positions.get(symbol, 0.0)
            held_qty = held.get(symbol, Decimal("0")) if held is not None else None

            # A risk-REDUCING order: a full exit, a reduction of a long, or the
            # buy that closes a short. It is exempt from the entry rules (not
            # from the execution halt) and sized from what is actually held.
            if held_qty is not None:
                reducing = held_qty != 0 and (
                    target_weight <= 0.0
                    or held_qty < 0
                    or (price > 0.0 and target_dollar / price < float(held_qty))
                )
            else:
                reducing = current_dollar > 0.0 and target_dollar < current_dollar

            decision = self.pre_trade_check(
                {"symbol": symbol, "target_weight": target_weight},
                risk_reducing=reducing,
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

            if reducing and held_qty is not None:
                # Exact held quantity for a full exit or a short close; the
                # surplus above the target for a reduction. Never the dollar
                # book divided by a moved price, and never more than is held.
                if held_qty < 0 or target_weight <= 0.0 or price <= 0.0:
                    quantity = float(abs(held_qty))
                else:
                    quantity = float(held_qty) - target_dollar / price
                side = OrderSide.SELL if held_qty > 0 else OrderSide.BUY
                if quantity <= 0.0:
                    continue
            else:
                # Size the weight delta into a share quantity. The risk manager
                # has no price feed, so it relies on the reference price
                # supplied by the producer (which has market data).
                if price <= 0.0:
                    logger.warning("No reference price for %s; skipping sizing.", symbol)
                    continue
                delta_dollar = target_dollar - current_dollar
                notional = abs(delta_dollar)
                if notional < _MIN_TRADE_NOTIONAL:
                    continue
                side = OrderSide.BUY if delta_dollar > 0 else OrderSide.SELL
                quantity = notional / price
                if quantity <= 0.0:
                    continue

            # Exploration trades are tagged in the correlation id so fills,
            # audit rows, and P&L reporting can separate learning trades from
            # conviction trades.
            prefix = (
                "explore" if symbol in rebalance.exploration_symbols else "rebal"
            )
            await self.event_bus.publish(
                RISK_APPROVED,
                RiskApprovedEvent(
                    order_id=f"{prefix}-{event.event_id}-{symbol}",
                    symbol=symbol,
                    side=str(side),
                    order_type="market",
                    quantity=quantity,
                    reference_price=price,
                    source_service="risk_manager",
                ),
            )
        await self.publish_breaker_transition()

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

    def pre_trade_check(
        self, order_request: dict[str, Any], *, risk_reducing: bool = False
    ) -> RiskDecision:
        """Run all risk rules against a proposed order.

        Parameters
        ----------
        order_request:
            Dictionary with at least ``symbol`` and ``target_weight``.
        risk_reducing:
            The order lowers exposure (an exit, a reduction, a short close).
            Every pre-trade rule exists to limit exposure, so none may be
            the reason exposure cannot fall: such an order is approved
            without evaluation. The execution halt still applies to it
            downstream (GO_LIVE.md 2026-11-07).

        Returns
        -------
        RiskDecision
        """
        symbol = order_request.get("symbol", "")
        target_weight = float(order_request.get("target_weight", 0.0))

        if risk_reducing:
            logger.info(
                "Pre-trade check for %s: risk-reducing order, entry rules not applied.",
                symbol,
            )
            return RiskDecision(approved=True, rejections=[], sized_quantity=Decimal("0"))

        # Build rule evaluation context
        total_equity = float(self._equity) if self._equity > 0 else 1.0
        proposed_position_pct = target_weight

        # Drawdown state
        dd_state = self.drawdown_monitor.update(total_equity)

        # Correlation: the proposed symbol against real holdings only.
        n_violations = self._entry_correlation_violations(symbol)

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
        if self._day_start_pending:
            self._day_start_equity = eq
            self._day_start_pending = False
            self.drawdown_monitor.reset_daily(eq)
            logger.info("risk.day_start_seeded_from_account equity=%.2f", eq)
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
        breaker track real capital. Also rolls the daily risk session over when
        the UTC calendar date advances, so multi-day runs reset the daily
        drawdown baseline and circuit breaker. Runs until cancelled."""
        last_seen_date = _utc_today()
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                today = _utc_today()
                if today > last_seen_date:
                    self.reset_daily()
                    await self.persist_safety_state()
                    logger.info(
                        "risk.daily_reset: UTC session rolled over from %s to %s",
                        last_seen_date,
                        today,
                    )
                    last_seen_date = today
                equity = await account_provider()
                if equity is not None:
                    self.sync_account(equity)
                    # Evaluate the breaker on the live feed, not only when a
                    # rebalance happens to run, and publish any transition.
                    self.circuit_breaker.check(self._daily_pnl_pct)
                await self.publish_breaker_transition()
                await self.persist_safety_state()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Account sync iteration failed")

    async def publish_breaker_transition(self) -> bool:
        """Publish a CircuitBreakerEvent if the breaker changed state.

        Until 2026-11-02 the breaker only logged; the event type existed, the
        monitoring handler for it existed, and nothing ever published it, so a
        trip reached neither Slack nor audit_logs. Returns True when an event
        was published.
        """
        state = self.circuit_breaker.state.value
        if state == self._published_breaker_state:
            return False
        previous = self._published_breaker_state
        self._published_breaker_state = state
        reason = self.circuit_breaker.trip_reason or f"{previous} -> {state}"
        await self.event_bus.publish(
            RISK_BREACHED,
            CircuitBreakerEvent(state=state, reason=reason, source_service="risk_manager"),
        )
        logger.warning("Circuit breaker %s -> %s published (%s)", previous, state, reason)
        return True

    @property
    def positions(self) -> dict[str, float]:
        """Current position dollar values (symbol -> value). Copy, not live."""
        return dict(self._positions)

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
        self._day_start_pending = False
        self._session_date = _utc_today()
        self.drawdown_monitor.reset_daily(float(self._equity))
        self.circuit_breaker.reset()
        logger.info("Risk manager daily reset complete.")

    # ------------------------------------------------------------------
    # Durable safety state
    # ------------------------------------------------------------------

    async def restore_safety_state(self) -> None:
        """Re-apply the persisted daily baseline, peak and breaker state.

        Same UTC day: baseline, peak and breaker come back exactly (an OPEN
        breaker stays OPEN). An older row: the all-time peak is kept, the
        daily baseline is re-seeded from the first account sync, and the
        breaker starts the new day CLOSED as reset_daily would have left it.
        """
        if self._state_store is None:
            return
        try:
            state = await self._state_store.load()
        except Exception:
            logger.exception("risk.safety_state_load_failed")
            return
        if state is None:
            return
        today = _utc_today()
        peak = float(state.peak_equity) if state.peak_equity is not None else None
        if state.session_date == today and state.day_start_equity is not None:
            day_start = float(state.day_start_equity)
            self._day_start_equity = day_start
            self._day_start_pending = False
            self.drawdown_monitor.restore(
                peak_equity=peak if peak is not None else day_start,
                daily_start_equity=day_start,
            )
            if state.breaker_state != "closed":
                tripped_at = state.breaker_tripped_at
                if tripped_at is not None and tripped_at.tzinfo is None:
                    tripped_at = tripped_at.replace(tzinfo=UTC)
                self.circuit_breaker.restore(
                    state.breaker_state,
                    tripped_at,
                    state.breaker_reason or "restored from safety_state",
                )
                self._published_breaker_state = state.breaker_state
            logger.warning(
                "risk.safety_state_restored",
                extra={
                    "day_start_equity": day_start,
                    "peak_equity": peak,
                    "breaker_state": state.breaker_state,
                },
            )
        else:
            if peak is not None:
                self.drawdown_monitor.restore(peak_equity=peak)
            self._day_start_pending = True
            logger.info("risk.safety_state_stale_new_session", extra={"peak_equity": peak})
        self._session_date = today

    async def persist_safety_state(self) -> None:
        """Write the daily baseline, peak and breaker state (best effort)."""
        if self._state_store is None or self._day_start_pending:
            return
        try:
            await self._state_store.save_risk(
                breaker_state=self.circuit_breaker.state.value,
                breaker_tripped_at=self.circuit_breaker.tripped_at,
                breaker_reason=self.circuit_breaker.trip_reason or None,
                day_start_equity=self._day_start_equity,
                peak_equity=self.drawdown_monitor.peak_equity,
                session_date=self._session_date,
            )
        except Exception:
            logger.exception("risk.safety_state_save_failed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _held_quantities(self) -> dict[str, Decimal] | None:
        """Signed quantity per symbol from the position provider, or None."""
        if self._position_provider is None:
            return None
        try:
            positions = await self._position_provider()
        except Exception:
            logger.exception("risk.position_provider_failed; sizing from the dollar book")
            return None
        held: dict[str, Decimal] = {}
        for position in positions:
            symbol = str(position.get("symbol", ""))
            try:
                quantity = Decimal(str(position.get("quantity", "0")))
            except InvalidOperation:
                continue
            if symbol and quantity != 0:
                held[symbol] = held.get(symbol, Decimal("0")) + quantity
        return held

    def _entry_correlation_violations(self, symbol: str) -> int:
        """Pairs (proposed symbol, held symbol) above the correlation limit.

        Only holdings at or above the minimum trade notional count: dust
        cannot block an entry, and the book's own internal correlations are
        the risk report's business, not this order's. Until 2026-11-07 the
        rule counted every pair in the book and ignored the symbol proposed,
        so four dust residues blocked every entry and every exit.
        """
        proposed = self._returns_history.get(symbol)
        if proposed is None or len(proposed) < 2:
            return 0
        candidates = {
            s: r
            for s, r in self._returns_history.items()
            if s != symbol
            and len(r) >= 2
            and self._positions.get(s, 0.0) >= _MIN_TRADE_NOTIONAL
        }
        if not candidates:
            return 0
        length = min(len(proposed), *(len(r) for r in candidates.values()))
        returns_dict = {symbol: np.array(proposed[-length:])}
        for s, r in candidates.items():
            returns_dict[s] = np.array(r[-length:])
        symbols_sorted = sorted(returns_dict)
        corr_matrix = self.correlation_monitor.compute_correlation_matrix(returns_dict)
        violations = self.correlation_monitor.check_pairwise_limits(
            corr_matrix,
            symbols_sorted,
            max_correlation=self.settings.max_pairwise_correlation,
        )
        return sum(1 for v in violations if symbol in (v.symbol_a, v.symbol_b))

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

    def extended_metrics(self) -> dict[str, float | None]:
        """Tail and dispersion metrics beyond the 95% figures in RiskReport.

        Used by the risk-metrics writer to persist what the dashboard renders.
        Every value is None -- never zero -- when the return history cannot
        support it: a fabricated 0.00 VaR reads as "no risk", which is the
        exact lie the dashboard used to tell.
        """
        rets = self._aggregate_portfolio_returns()
        if rets.size < 2:
            return {
                "var_99": None,
                "cvar_99": None,
                "volatility": None,
                "correlation_max": None,
            }
        equity = float(self._equity) if float(self._equity) > 0 else 1.0
        var_99 = self.var_calculator.historical_var(
            rets, confidence=0.99, portfolio_value=1.0
        )
        cvar_99 = self.var_calculator.cvar(rets, confidence=0.99, portfolio_value=1.0)
        volatility = float(np.std(rets, ddof=1))

        correlation_max: float | None = None
        returns_dict = {
            s: np.array(r) for s, r in self._returns_history.items() if len(r) >= 2
        }
        if len(returns_dict) >= 2:
            matrix = self.correlation_monitor.compute_correlation_matrix(returns_dict)
            arr = np.asarray(matrix, dtype=float)
            off_diagonal = arr[~np.eye(arr.shape[0], dtype=bool)]
            if off_diagonal.size:
                correlation_max = float(np.nanmax(np.abs(off_diagonal)))

        return {
            "var_99": float(var_99 * equity),
            "cvar_99": float(cvar_99 * equity),
            "volatility": volatility,
            "correlation_max": correlation_max,
        }

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
