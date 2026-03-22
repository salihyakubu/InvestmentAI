"""Alpaca broker adapter for US equities execution."""

from __future__ import annotations

from decimal import Decimal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.enums import OrderStatus
from config.settings import Settings
from services.execution.brokers.base import BaseBroker, BrokerOrder

logger = structlog.get_logger(__name__)

# Map Alpaca order statuses to our internal OrderStatus.
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL_FILL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "pending_new": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
}


class AlpacaBroker(BaseBroker):
    """Broker adapter for Alpaca Markets (US equities)."""

    name = "alpaca"
    asset_class = "stock"
    supports_live = True

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def _get_client(self):
        """Lazy-initialise the Alpaca TradingClient."""
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._settings.alpaca_api_key.get_secret_value(),
                secret_key=self._settings.alpaca_secret_key.get_secret_value(),
                paper=("paper" in self._settings.alpaca_base_url),
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to Alpaca and return the Alpaca order ID."""
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        client = self._get_client()

        side = AlpacaSide.BUY if order.side == "buy" else AlpacaSide.SELL

        if order.order_type == "market":
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type == "limit":
            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(order.limit_price) if order.limit_price else None,
            )
        elif order.order_type == "stop":
            request = StopOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                stop_price=float(order.stop_price) if order.stop_price else None,
            )
        elif order.order_type == "stop_limit":
            request = StopLimitOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
            )
        else:
            raise ValueError(f"Unsupported order type for Alpaca: {order.order_type}")

        alpaca_order = client.submit_order(request)
        external_id = str(alpaca_order.id)

        logger.info(
            "alpaca_order_submitted",
            external_id=external_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=str(order.quantity),
        )
        return external_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    async def cancel_order(self, external_id: str) -> bool:
        """Cancel an order on Alpaca."""
        client = self._get_client()
        try:
            client.cancel_order_by_id(external_id)
            logger.info("alpaca_order_cancelled", external_id=external_id)
            return True
        except Exception as exc:
            logger.warning(
                "alpaca_cancel_failed",
                external_id=external_id,
                error=str(exc),
            )
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    async def get_order_status(self, external_id: str) -> dict:
        """Fetch order status from Alpaca and map to internal status."""
        client = self._get_client()
        alpaca_order = client.get_order_by_id(external_id)
        raw_status = str(alpaca_order.status).lower()
        mapped_status = _STATUS_MAP.get(raw_status, OrderStatus.PENDING)

        return {
            "external_id": external_id,
            "status": mapped_status,
            "filled_qty": str(alpaca_order.filled_qty or 0),
            "filled_avg_price": str(alpaca_order.filled_avg_price or 0),
            "symbol": alpaca_order.symbol,
            "side": str(alpaca_order.side).lower(),
            "order_type": str(alpaca_order.order_type).lower(),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    async def get_positions(self) -> list[dict]:
        """Return all current Alpaca positions."""
        client = self._get_client()
        positions = client.get_all_positions()
        return [
            {
                "symbol": pos.symbol,
                "quantity": str(pos.qty),
                "market_value": str(pos.market_value),
                "avg_entry_price": str(pos.avg_entry_price),
                "unrealized_pnl": str(pos.unrealized_pl),
            }
            for pos in positions
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    async def get_account(self) -> dict:
        """Return Alpaca account summary."""
        client = self._get_client()
        account = client.get_account()
        return {
            "equity": str(account.equity),
            "cash": str(account.cash),
            "buying_power": str(account.buying_power),
            "positions_value": str(
                Decimal(str(account.equity)) - Decimal(str(account.cash))
            ),
        }

    async def health_check(self) -> bool:
        """Check if Alpaca API is reachable."""
        try:
            client = self._get_client()
            client.get_account()
            return True
        except Exception as exc:
            logger.warning("alpaca_health_check_failed", error=str(exc))
            return False
