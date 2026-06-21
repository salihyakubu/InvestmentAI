"""Smart order router that selects brokers and execution algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from config.settings import Settings
from core.enums import AssetClass, OrderType
from services.execution.brokers.base import BaseBroker

logger = structlog.get_logger(__name__)

# Thresholds as fraction of average daily volume.
_TWAP_THRESHOLD = Decimal("0.10")
_VWAP_THRESHOLD = Decimal("0.05")


@dataclass
class ChildOrder:
    """A sub-order produced by algorithmic splitting."""

    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    scheduled_time: float | None = None  # seconds offset from now
    limit_price: Decimal | None = None


@dataclass
class RoutingDecision:
    """The result of the smart router's analysis."""

    broker_name: str | None  # None when no broker is safe to route to
    algo_type: str  # "direct", "twap", or "vwap"
    child_orders: list[ChildOrder] = field(default_factory=list)


class SmartOrderRouter:
    """Routes orders to the appropriate broker and execution algorithm.

    Selection logic:
    1. Choose broker based on asset class.
    2. If order value > 10% of daily volume -> TWAP.
    3. If order value > 5% of daily volume -> VWAP.
    4. Otherwise -> direct market/limit submission.
    """

    def __init__(
        self,
        brokers: dict[str, BaseBroker],
        settings: Settings,
    ) -> None:
        self._brokers = brokers
        self._settings = settings

    def route(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        avg_daily_volume: Decimal | None = None,
        limit_price: Decimal | None = None,
    ) -> RoutingDecision:
        """Determine broker and algorithm for the given order parameters."""

        broker_name = self._select_broker(symbol)

        # Determine participation rate if volume data is available.
        if avg_daily_volume and avg_daily_volume > 0:
            participation = quantity / avg_daily_volume
        else:
            participation = Decimal("0")

        algo_type = "direct"
        child_orders: list[ChildOrder] = []

        if participation >= _TWAP_THRESHOLD:
            algo_type = "twap"
            logger.info(
                "smart_router_twap",
                symbol=symbol,
                participation=str(participation),
            )
            child_orders = [
                ChildOrder(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.TWAP,
                    quantity=quantity,
                    limit_price=limit_price,
                )
            ]
        elif participation >= _VWAP_THRESHOLD:
            algo_type = "vwap"
            logger.info(
                "smart_router_vwap",
                symbol=symbol,
                participation=str(participation),
            )
            child_orders = [
                ChildOrder(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.VWAP,
                    quantity=quantity,
                    limit_price=limit_price,
                )
            ]
        else:
            child_orders = [
                ChildOrder(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    limit_price=limit_price,
                )
            ]

        decision = RoutingDecision(
            broker_name=broker_name,
            algo_type=algo_type,
            child_orders=child_orders,
        )

        logger.info(
            "smart_router_decision",
            symbol=symbol,
            broker=broker_name,
            algo_type=algo_type,
            child_count=len(child_orders),
        )
        return decision

    def _select_broker(self, symbol: str) -> str | None:
        """Select a broker for *symbol* by asset class, honouring trading mode.

        Safety: outside live mode we only ever consider brokers that cannot
        place real-money orders (``supports_live is False``). This filter --
        not dict insertion order -- is what protects paper/backtest traffic
        from reaching a live venue. Reordering, removing, or rebuilding the
        brokers dict therefore cannot accidentally route a "paper" crypto
        order to the real Binance venue.

        Returns the chosen broker name, or ``None`` when no broker is eligible
        (e.g. only live brokers are wired in a non-live mode); the caller must
        refuse to dispatch in that case rather than fall back to a live venue.
        """
        # Crypto symbols typically contain a slash (e.g. BTC/USDT).
        if "/" in symbol:
            target_class = AssetClass.CRYPTO
        else:
            target_class = AssetClass.STOCK

        live_mode = self._settings.trading_mode == "live"
        eligible = [
            (name, broker)
            for name, broker in self._brokers.items()
            if live_mode or not broker.supports_live
        ]

        for name, broker in eligible:
            if broker.asset_class == target_class or broker.asset_class == "all":
                return name

        # No asset-class match among eligible brokers; fall back to the first
        # eligible broker (guaranteed non-live outside live mode).
        if eligible:
            first = eligible[0][0]
            logger.warning(
                "smart_router_no_class_match",
                symbol=symbol,
                target_class=target_class,
                fallback=first,
            )
            return first

        # Nothing safe to route to. Refuse rather than touch a live venue.
        logger.error(
            "smart_router_no_safe_broker",
            symbol=symbol,
            target_class=target_class,
            trading_mode=self._settings.trading_mode,
        )
        return None
