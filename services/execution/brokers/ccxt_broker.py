"""CCXT broker adapter for Binance cryptocurrency execution."""

from __future__ import annotations

from decimal import Decimal

import structlog

from config.settings import Settings
from core.enums import OrderStatus
from services.execution.brokers.base import BaseBroker, BrokerOrder

logger = structlog.get_logger(__name__)

# Map CCXT/Binance order statuses to our internal OrderStatus.
_STATUS_MAP = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


class CCXTBroker(BaseBroker):
    """Broker adapter for Binance via the CCXT library."""

    name = "ccxt_binance"
    asset_class = "crypto"
    supports_live = True

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exchange = None

    async def _get_exchange(self):
        """Lazy-initialise the async Binance exchange client."""
        if self._exchange is None:
            import ccxt.async_support as ccxt_async

            self._exchange = ccxt_async.binance({
                "apiKey": self._settings.binance_api_key.get_secret_value(),
                "secret": self._settings.binance_secret_key.get_secret_value(),
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
            })
            await self._exchange.load_markets()
        return self._exchange

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to Binance via CCXT."""
        exchange = await self._get_exchange()

        # Adjust quantity to match Binance lot size precision.
        symbol = order.symbol
        quantity = float(order.quantity)
        quantity = self._adjust_quantity(symbol, quantity)

        params: dict = {}

        if order.order_type == "market":
            ccxt_type = "market"
        elif order.order_type == "limit":
            ccxt_type = "limit"
        elif order.order_type == "stop":
            ccxt_type = "market"
            if order.stop_price is not None:
                params["stopPrice"] = float(order.stop_price)
        elif order.order_type == "stop_limit":
            ccxt_type = "limit"
            if order.stop_price is not None:
                params["stopPrice"] = float(order.stop_price)
        else:
            raise ValueError(f"Unsupported order type for Binance: {order.order_type}")

        price = float(order.limit_price) if order.limit_price else None

        try:
            result = await exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=order.side,
                amount=quantity,
                price=price,
                params=params,
            )
            external_id = str(result["id"])
            logger.info(
                "ccxt_order_submitted",
                external_id=external_id,
                symbol=symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=str(quantity),
            )
            return external_id
        except Exception as exc:
            logger.error(
                "ccxt_order_submit_failed",
                symbol=symbol,
                error=str(exc),
            )
            raise

    async def cancel_order(self, external_id: str, symbol: str = "") -> bool:
        """Cancel an open order on Binance."""
        exchange = await self._get_exchange()
        try:
            await exchange.cancel_order(external_id, symbol=symbol or None)
            logger.info("ccxt_order_cancelled", external_id=external_id)
            return True
        except Exception as exc:
            logger.warning(
                "ccxt_cancel_failed",
                external_id=external_id,
                error=str(exc),
            )
            return False

    async def get_order_status(self, external_id: str, symbol: str = "") -> dict:
        """Fetch order status from Binance."""
        exchange = await self._get_exchange()
        try:
            order = await exchange.fetch_order(external_id, symbol=symbol or None)
            raw_status = order.get("status", "").lower()
            mapped_status = _STATUS_MAP.get(raw_status, OrderStatus.PENDING)

            return {
                "external_id": external_id,
                "status": mapped_status,
                "filled_qty": str(order.get("filled", 0)),
                "filled_avg_price": str(order.get("average", 0) or 0),
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "order_type": order.get("type", ""),
                "remaining": str(order.get("remaining", 0)),
            }
        except Exception as exc:
            logger.error(
                "ccxt_get_status_failed",
                external_id=external_id,
                error=str(exc),
            )
            raise

    async def get_positions(self) -> list[dict]:
        """Return current positions from Binance balances."""
        exchange = await self._get_exchange()
        try:
            balance = await exchange.fetch_balance()
            positions = []
            for currency, info in balance.get("total", {}).items():
                amount = float(info) if info else 0.0
                if amount > 0 and currency != "USDT":
                    positions.append({
                        "symbol": f"{currency}/USDT",
                        "quantity": str(amount),
                        "market_value": "0",  # would need ticker to compute
                        "avg_entry_price": "0",
                        "unrealized_pnl": "0",
                    })
            return positions
        except Exception as exc:
            logger.error("ccxt_get_positions_failed", error=str(exc))
            raise

    async def get_account(self) -> dict:
        """Return account summary from Binance."""
        exchange = await self._get_exchange()
        try:
            balance = await exchange.fetch_balance()
            usdt_free = float(balance.get("free", {}).get("USDT", 0))
            usdt_total = float(balance.get("total", {}).get("USDT", 0))
            return {
                "equity": str(usdt_total),
                "cash": str(usdt_free),
                "buying_power": str(usdt_free),
                "positions_value": str(usdt_total - usdt_free),
            }
        except Exception as exc:
            logger.error("ccxt_get_account_failed", error=str(exc))
            raise

    async def health_check(self) -> bool:
        """Check if Binance is reachable."""
        try:
            exchange = await self._get_exchange()
            await exchange.fetch_time()
            return True
        except Exception as exc:
            logger.warning("ccxt_health_check_failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjust_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to Binance lot-size precision."""
        if self._exchange is None or symbol not in self._exchange.markets:
            return quantity

        market = self._exchange.markets[symbol]
        precision = market.get("precision", {}).get("amount")
        if precision is not None:
            quantity = float(
                Decimal(str(quantity)).quantize(Decimal(10) ** -int(precision))
            )

        # Enforce minimum order size.
        limits = market.get("limits", {}).get("amount", {})
        min_amount = limits.get("min")
        if min_amount and quantity < float(min_amount):
            logger.warning(
                "ccxt_quantity_below_minimum",
                symbol=symbol,
                quantity=quantity,
                min_amount=min_amount,
            )
        return quantity
