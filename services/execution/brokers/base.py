"""Abstract base broker interface for all execution brokers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass
class BrokerOrder:
    """Broker-level order representation sent to the execution venue."""

    external_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None


@dataclass
class BrokerFill:
    """A single fill (partial or complete) returned by the broker."""

    external_id: str
    order_external_id: str
    price: Decimal
    quantity: Decimal
    commission: Decimal
    filled_at: datetime


class BaseBroker(ABC):
    """Abstract broker that all venue adapters must implement."""

    name: str
    asset_class: str
    supports_live: bool

    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order and return the broker-assigned external ID."""
        ...

    @abstractmethod
    async def cancel_order(self, external_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""
        ...

    @abstractmethod
    async def get_order_status(self, external_id: str) -> dict:
        """Return current order status from the broker."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""
        ...

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account summary (equity, cash, buying power, etc.)."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the broker connection is healthy."""
        ...
