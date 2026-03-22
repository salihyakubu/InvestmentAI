"""Market simulator for backtesting order fills."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from core.enums import OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar used by the simulator."""

    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class SimulatedOrder:
    """Order submitted to the simulator."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None


@dataclass
class SimulatedFill:
    """Result of a simulated order execution."""

    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    commission: Decimal
    slippage: Decimal


class MarketSimulator:
    """Simulates order fills against historical OHLCV bars.

    Parameters
    ----------
    commission_per_share:
        Fixed commission charged per share filled.
    default_slippage_bps:
        Default slippage in basis points applied to market orders
        if not overridden per call.
    """

    def __init__(
        self,
        commission_per_share: Decimal = Decimal("0.005"),
        default_slippage_bps: Decimal = Decimal("5"),
    ) -> None:
        self._commission_per_share = commission_per_share
        self._default_slippage_bps = default_slippage_bps

    def simulate_fill(
        self,
        order: SimulatedOrder,
        bar: Bar,
        slippage_bps: Decimal | None = None,
    ) -> SimulatedFill | None:
        """Attempt to fill *order* against *bar*.

        Returns a :class:`SimulatedFill` if the order can be filled on
        this bar, or ``None`` if the fill conditions are not met.

        Fill logic by order type:

        * **MARKET**: filled at bar close with slippage.
        * **LIMIT**: filled if the bar's price range crosses the limit
          price.  Fill price is the limit price (best case).
        * **STOP**: filled if the bar's price range crosses the stop
          price.  Simulates gap risk by filling at the stop price with
          slippage applied.
        """
        if slippage_bps is None:
            slippage_bps = self._default_slippage_bps

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar, slippage_bps)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar, slippage_bps)
        else:
            logger.warning(
                "Unsupported order type %s for simulation", order.order_type
            )
            return None

    # ------------------------------------------------------------------
    # Private fill methods
    # ------------------------------------------------------------------

    def _fill_market(
        self,
        order: SimulatedOrder,
        bar: Bar,
        slippage_bps: Decimal,
    ) -> SimulatedFill:
        """Market orders fill at bar close +/- slippage."""
        slip_factor = slippage_bps / Decimal("10000")

        if order.side == OrderSide.BUY:
            fill_price = bar.close * (Decimal("1") + slip_factor)
        else:
            fill_price = bar.close * (Decimal("1") - slip_factor)

        slippage_amount = abs(fill_price - bar.close)
        commission = self._commission_per_share * order.quantity

        return SimulatedFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price.quantize(Decimal("0.0001")),
            quantity=order.quantity,
            commission=commission.quantize(Decimal("0.01")),
            slippage=slippage_amount.quantize(Decimal("0.0001")),
        )

    def _fill_limit(
        self,
        order: SimulatedOrder,
        bar: Bar,
    ) -> SimulatedFill | None:
        """Limit orders fill if the bar crosses the limit price."""
        if order.limit_price is None:
            return None

        if order.side == OrderSide.BUY:
            # Buy limit: fill if bar low <= limit price
            if bar.low <= order.limit_price:
                fill_price = order.limit_price
            else:
                return None
        else:
            # Sell limit: fill if bar high >= limit price
            if bar.high >= order.limit_price:
                fill_price = order.limit_price
            else:
                return None

        commission = self._commission_per_share * order.quantity

        return SimulatedFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission.quantize(Decimal("0.01")),
            slippage=Decimal("0"),
        )

    def _fill_stop(
        self,
        order: SimulatedOrder,
        bar: Bar,
        slippage_bps: Decimal,
    ) -> SimulatedFill | None:
        """Stop orders fill if the bar crosses the stop price (with gap risk)."""
        if order.stop_price is None:
            return None

        slip_factor = slippage_bps / Decimal("10000")

        if order.side == OrderSide.BUY:
            # Buy stop: triggered when price rises above stop
            if bar.high >= order.stop_price:
                fill_price = order.stop_price * (Decimal("1") + slip_factor)
            else:
                return None
        else:
            # Sell stop: triggered when price drops below stop
            if bar.low <= order.stop_price:
                fill_price = order.stop_price * (Decimal("1") - slip_factor)
            else:
                return None

        slippage_amount = abs(fill_price - order.stop_price)
        commission = self._commission_per_share * order.quantity

        return SimulatedFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price.quantize(Decimal("0.0001")),
            quantity=order.quantity,
            commission=commission.quantize(Decimal("0.01")),
            slippage=slippage_amount.quantize(Decimal("0.0001")),
        )
