"""Trading-mode safety for broker routing: a non-live mode must never reach a
live venue, and ordering of the brokers dict must NOT be what protects us.

These tests pin the invariant directly at the routing layer (the previous worker
wiring only kept us safe because the PaperBroker happened to be inserted first).
Here we deliberately insert a live crypto broker *before* the PaperBroker and
prove that a "paper"-mode BTC/USDT order still never dispatches to the live venue
-- both through the smart router's selection and the execution engine's hard
guard. See also tests/unit/test_worker_broker_wiring.py for the wiring side.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.settings import Settings
from core.enums import OrderStatus
from core.events.base import InProcessEventBus
from core.events.order_events import OrderRejectedEvent
from services.execution.brokers.base import BaseBroker, BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.execution.smart_router import RoutingDecision, SmartOrderRouter


class _TripwireLiveBroker(BaseBroker):
    """A live crypto broker that screams if anything tries to dispatch to it.

    Mirrors CCXTBroker's class attributes (``asset_class="crypto"``,
    ``supports_live=True``) so a naive asset-class-only router would pick it for
    a BTC/USDT order. Every dispatch/IO method raises so a routing regression
    surfaces loudly in tests instead of silently hitting a real venue.
    """

    name = "ccxt"
    asset_class = "crypto"
    supports_live = True

    def __init__(self) -> None:
        self.submit_called = False

    async def submit_order(self, order: BrokerOrder) -> str:
        self.submit_called = True
        raise AssertionError("live broker reached in non-live mode")

    async def cancel_order(self, external_id: str) -> bool:
        raise AssertionError("live broker reached in non-live mode")

    async def get_order_status(self, external_id: str) -> dict:
        raise AssertionError("live broker reached in non-live mode")

    async def get_positions(self) -> list[dict]:
        raise AssertionError("live broker reached in non-live mode")

    async def get_account(self) -> dict:
        raise AssertionError("live broker reached in non-live mode")

    async def health_check(self) -> bool:
        return True


def _settings(trading_mode: str) -> Settings:
    # Pass an explicit strong JWT secret so live-mode construction satisfies the
    # settings guard regardless of the ambient environment. Constructor kwargs
    # take precedence over any .env, so the test is hermetic (CI has no .env;
    # locally we must not depend on the developer's real secret either).
    return Settings(
        trading_mode=trading_mode,
        initial_capital=Decimal("100000"),
        jwt_secret="test-secret-not-the-published-default-value",
    )


def _brokers_ccxt_first() -> tuple[_TripwireLiveBroker, PaperBroker, dict]:
    """Build a brokers dict with the live ccxt broker inserted BEFORE paper.

    If insertion order were what protected us, this ordering would route crypto
    to the live venue -- which is exactly what these tests prove cannot happen.
    """
    live = _TripwireLiveBroker()
    paper = PaperBroker(initial_cash=Decimal("100000"))
    paper.update_price("BTC/USDT", Decimal("50000"))
    # dict literal preserves insertion order: ccxt first, paper second.
    return live, paper, {"ccxt": live, "paper": paper}


# ---------------------------------------------------------------------------
# Router-level selection
# ---------------------------------------------------------------------------


def test_select_broker_skips_live_crypto_in_paper_mode() -> None:
    """Paper mode: crypto routes to the (non-live) paper broker, not ccxt,
    even though ccxt is first in the dict and matches the asset class."""
    _live, _paper, brokers = _brokers_ccxt_first()
    router = SmartOrderRouter(brokers, _settings("paper"))

    assert router._select_broker("BTC/USDT") == "paper"

    decision = router.route(
        symbol="BTC/USDT", side="buy", order_type="market", quantity=Decimal("0.001")
    )
    assert decision.broker_name == "paper"


def test_select_broker_uses_live_crypto_in_live_mode() -> None:
    """Positive control: in live mode the live crypto broker IS eligible and is
    selected for a crypto symbol -- proving the filter only bites outside live."""
    _live, _paper, brokers = _brokers_ccxt_first()
    router = SmartOrderRouter(brokers, _settings("live"))

    assert router._select_broker("BTC/USDT") == "ccxt"


def test_select_broker_returns_none_when_only_live_brokers_in_paper_mode() -> None:
    """No safe broker available -> refuse (None) rather than fall back to live."""
    live = _TripwireLiveBroker()
    router = SmartOrderRouter({"ccxt": live}, _settings("paper"))

    assert router._select_broker("BTC/USDT") is None


# ---------------------------------------------------------------------------
# Engine-level dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_order_never_dispatches_to_live_broker_in_paper_mode() -> None:
    """End-to-end: a paper-mode BTC/USDT order fills on the paper broker and the
    live broker is never touched, regardless of dict ordering."""
    bus = InProcessEventBus()
    live, paper, brokers = _brokers_ccxt_first()
    engine = ExecutionEngineService(event_bus=bus, settings=_settings("paper"), brokers=brokers)

    order = await engine.submit_order(
        symbol="BTC/USDT", side="buy", order_type="market", quantity=Decimal("0.001")
    )

    assert live.submit_called is False
    assert order is not None
    assert order.broker_name == "paper"

    positions = await paper.get_positions()
    btc = next((p for p in positions if p["symbol"] == "BTC/USDT"), None)
    assert btc is not None
    assert Decimal(btc["quantity"]) == Decimal("0.001")


@pytest.mark.asyncio
async def test_submit_order_hard_guard_rejects_forced_live_broker() -> None:
    """Defence in depth: even if routing is subverted to pick a live broker, the
    engine's hard guard rejects the order outside live mode."""
    bus = InProcessEventBus()
    live, _paper, brokers = _brokers_ccxt_first()
    engine = ExecutionEngineService(event_bus=bus, settings=_settings("paper"), brokers=brokers)

    # Force the router to choose the live broker, bypassing its safety filter.
    def _force_live(*_args: object, **_kwargs: object) -> RoutingDecision:
        return RoutingDecision(broker_name="ccxt", algo_type="direct", child_orders=[])

    engine._router.route = _force_live  # type: ignore[method-assign]

    order = await engine.submit_order(
        symbol="BTC/USDT", side="buy", order_type="market", quantity=Decimal("0.001")
    )

    assert live.submit_called is False
    rejected = [e for _s, e in bus.history if isinstance(e, OrderRejectedEvent)]
    assert rejected, "expected an OrderRejectedEvent for the blocked live dispatch"
    assert "live_broker_blocked" in rejected[-1].reason

    status = engine._order_manager.get_order(order.order_id).status
    assert status == OrderStatus.REJECTED
