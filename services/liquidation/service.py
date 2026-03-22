"""Main liquidation manager service coordinating all stop/liquidation logic."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from config.settings import Settings
from core.events.base import Event, EventBus
from core.events.market_events import PriceUpdateEvent
from core.events.risk_events import RiskBreachedEvent
from core.events.system_events import LiquidationTriggeredEvent

from services.liquidation.emergency import EmergencyLiquidator
from services.liquidation.stop_loss import StopLossManager
from services.liquidation.trailing_stop import TrailingStopManager

logger = logging.getLogger(__name__)

# Stream names used by the event bus.
STREAM_MARKET = "market.prices"
STREAM_RISK = "risk.breaches"
STREAM_LIQUIDATION = "liquidation.triggered"


class LiquidationManagerService:
    """Orchestrates stop-loss, trailing-stop, and emergency liquidation.

    Subscribes to :class:`PriceUpdateEvent` and :class:`RiskBreachedEvent`
    from the event bus, evaluates all registered stops on each tick, and
    publishes :class:`LiquidationTriggeredEvent` when a position must be
    closed.

    Parameters
    ----------
    event_bus:
        The platform event bus (Redis-backed or in-process).
    settings:
        Application settings (used for risk thresholds).
    execution_service_ref:
        An optional reference to the execution service for forwarding
        liquidation orders.  Kept as ``Any`` to avoid circular imports.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        execution_service_ref: Any = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._execution_service = execution_service_ref

        self.stop_loss_manager = StopLossManager()
        self.trailing_stop_manager = TrailingStopManager()
        self.emergency_liquidator = EmergencyLiquidator()

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant event streams and begin processing."""
        if self._running:
            return

        self._running = True
        logger.info("LiquidationManagerService starting")

        # Register handlers -- for InProcessEventBus these return
        # immediately; for Redis EventBus they would be run as tasks.
        await self._event_bus.subscribe(
            stream=STREAM_MARKET,
            group="liquidation-svc",
            consumer="liquidation-1",
            handler=self.handle_price_update,
        )
        await self._event_bus.subscribe(
            stream=STREAM_RISK,
            group="liquidation-svc",
            consumer="liquidation-1",
            handler=self.handle_risk_breach,
        )
        logger.info("LiquidationManagerService subscribed to events")

    async def stop(self) -> None:
        """Mark the service as stopped."""
        self._running = False
        logger.info("LiquidationManagerService stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_price_update(self, event: Event) -> None:
        """Handle a :class:`PriceUpdateEvent` -- check all stops."""
        if not isinstance(event, PriceUpdateEvent):
            # Deserialise from generic Event if needed.
            symbol = event.payload.get("symbol", "")
            price = Decimal(str(event.payload.get("price", 0)))
        else:
            symbol = event.symbol
            price = Decimal(str(event.price))

        if not symbol:
            return

        current_prices: dict[str, Decimal] = {symbol: price}

        # 1. Fixed stop losses
        for stop in self.stop_loss_manager.check_stops(current_prices):
            await self._trigger_liquidation(
                position_id=stop.position_id,
                symbol=stop.symbol,
                reason=stop.reason,
                quantity=0.0,  # full close implied
            )

        # 2. Trailing stops
        for tstop in self.trailing_stop_manager.update_prices(current_prices):
            await self._trigger_liquidation(
                position_id=tstop.position_id,
                symbol=tstop.symbol,
                reason=tstop.reason,
                quantity=0.0,
            )

    async def handle_risk_breach(self, event: Event) -> None:
        """Handle a :class:`RiskBreachedEvent` -- emergency liquidation if needed."""
        if isinstance(event, RiskBreachedEvent):
            rule_name = event.rule_name
            action = event.action
            current_value = event.current_value
            limit_value = event.limit_value
        else:
            rule_name = event.payload.get("rule_name", "unknown")
            action = event.payload.get("action", "warn")
            current_value = event.payload.get("current_value", 0)
            limit_value = event.payload.get("limit_value", 0)

        logger.warning(
            "Risk breach received: rule=%s action=%s value=%s limit=%s",
            rule_name,
            action,
            current_value,
            limit_value,
        )

        if action in ("reduce", "liquidate"):
            reason = (
                f"Risk breach [{rule_name}]: "
                f"value {current_value} > limit {limit_value}"
            )
            orders = self.emergency_liquidator.liquidate_all(reason)
            for order in orders:
                await self._trigger_liquidation(
                    position_id=order.position_id,
                    symbol=order.symbol,
                    reason=order.reason,
                    quantity=float(order.quantity),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _trigger_liquidation(
        self,
        position_id: str,
        symbol: str,
        reason: str,
        quantity: float,
    ) -> None:
        """Publish a :class:`LiquidationTriggeredEvent` on the event bus."""
        event = LiquidationTriggeredEvent(
            position_id=position_id,
            symbol=symbol,
            reason=reason,
            quantity=quantity,
            source_service="LiquidationManagerService",
        )
        await self._event_bus.publish(STREAM_LIQUIDATION, event)
        logger.warning(
            "Liquidation triggered: position=%s symbol=%s reason=%s",
            position_id,
            symbol,
            reason,
        )
