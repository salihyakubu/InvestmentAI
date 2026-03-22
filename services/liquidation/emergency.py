"""Emergency liquidation for rapid portfolio de-risking."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from core.enums import AssetClass, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class LiquidationOrder:
    """Represents a market order created for emergency liquidation."""

    order_id: str
    position_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    reason: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PositionSnapshot:
    """Minimal position info needed for liquidation decisions."""

    position_id: str
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    quantity: Decimal


class EmergencyLiquidator:
    """Creates market orders to immediately exit positions.

    The liquidator does not execute orders itself -- it produces
    :class:`LiquidationOrder` objects that the execution layer
    can process.

    It maintains an in-memory log of all emergency actions for
    audit purposes.
    """

    def __init__(self) -> None:
        self._positions: dict[str, PositionSnapshot] = {}
        self._action_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Position registry (kept in sync by the service layer)
    # ------------------------------------------------------------------

    def register_position(self, snapshot: PositionSnapshot) -> None:
        """Register or update a position snapshot."""
        self._positions[snapshot.position_id] = snapshot

    def unregister_position(self, position_id: str) -> None:
        """Remove a closed position from the registry."""
        self._positions.pop(position_id, None)

    # ------------------------------------------------------------------
    # Liquidation actions
    # ------------------------------------------------------------------

    def liquidate_all(self, reason: str) -> list[LiquidationOrder]:
        """Create market orders to liquidate **every** open position.

        Returns a list of :class:`LiquidationOrder` objects.
        """
        orders: list[LiquidationOrder] = []
        for snapshot in list(self._positions.values()):
            order = self._build_liquidation_order(snapshot, reason)
            orders.append(order)

        self._log_action("liquidate_all", reason, orders)
        logger.critical(
            "EMERGENCY LIQUIDATE ALL: %d orders created -- %s",
            len(orders),
            reason,
        )
        return orders

    def liquidate_asset_class(
        self, asset_class: AssetClass, reason: str
    ) -> list[LiquidationOrder]:
        """Liquidate all positions belonging to *asset_class*."""
        orders: list[LiquidationOrder] = []
        for snapshot in list(self._positions.values()):
            if snapshot.asset_class == asset_class:
                order = self._build_liquidation_order(snapshot, reason)
                orders.append(order)

        self._log_action(
            f"liquidate_asset_class:{asset_class}", reason, orders
        )
        logger.critical(
            "EMERGENCY LIQUIDATE %s: %d orders -- %s",
            asset_class,
            len(orders),
            reason,
        )
        return orders

    def liquidate_position(
        self, position_id: str, reason: str
    ) -> LiquidationOrder | None:
        """Liquidate a single position by *position_id*.

        Returns ``None`` if the position is not found.
        """
        snapshot = self._positions.get(position_id)
        if snapshot is None:
            logger.warning(
                "Position %s not found for emergency liquidation", position_id
            )
            return None

        order = self._build_liquidation_order(snapshot, reason)
        self._log_action(
            f"liquidate_position:{position_id}", reason, [order]
        )
        logger.critical(
            "EMERGENCY LIQUIDATE %s (%s): %s",
            position_id,
            snapshot.symbol,
            reason,
        )
        return order

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_liquidation_order(
        self, snapshot: PositionSnapshot, reason: str
    ) -> LiquidationOrder:
        """Build a market order that closes *snapshot*."""
        # To close: sell if long, buy if short.
        close_side = (
            OrderSide.SELL if snapshot.side == OrderSide.BUY else OrderSide.BUY
        )
        return LiquidationOrder(
            order_id=str(uuid.uuid4()),
            position_id=snapshot.position_id,
            symbol=snapshot.symbol,
            side=close_side,
            quantity=snapshot.quantity,
            order_type=OrderType.MARKET,
            reason=reason,
        )

    def _log_action(
        self,
        action: str,
        reason: str,
        orders: list[LiquidationOrder],
    ) -> None:
        self._action_log.append(
            {
                "action": action,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_ids": [o.order_id for o in orders],
                "symbols": [o.symbol for o in orders],
            }
        )

    @property
    def action_log(self) -> list[dict[str, Any]]:
        """Return the full emergency action audit log."""
        return list(self._action_log)
