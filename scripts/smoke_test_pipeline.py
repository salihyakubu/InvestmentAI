"""End-to-end smoke test against REAL infrastructure (Redis + the paper broker).

Run after `alembic upgrade head`, with Redis reachable at settings.redis_url:

    PYTHONPATH=. python scripts/smoke_test_pipeline.py

Verifies the things unit tests can't: that the real Redis-backed EventBus starts
its consumer loops as background tasks, and that an order flows
rebalance -> risk -> execution -> broker fill -> risk feedback over the wire.
Exits non-zero on failure so it can gate a deploy.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

from config.settings import get_settings
from core.events.base import EventBus
from core.events.risk_events import RebalanceRequestEvent
from core.events.streams import REBALANCE
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.risk.service import RiskManagerService


async def main() -> int:
    settings = get_settings()
    print(f"[1/5] Connecting to Redis at {settings.redis_url} ...")
    bus = EventBus(redis_url=settings.redis_url)
    redis = await bus._get_redis()
    await redis.ping()
    print("      Redis reachable.")

    broker = PaperBroker(initial_cash=Decimal("10000"))
    broker.update_price("AAPL", Decimal("100"))

    risk = RiskManagerService(event_bus=bus, settings=settings)
    risk.update_equity(Decimal("10000"))
    risk.drawdown_monitor.update(10000.0)
    execution = ExecutionEngineService(
        event_bus=bus, settings=settings, brokers={"paper": broker}
    )

    print("[2/5] Starting risk + execution services (real Redis consumer loops) ...")
    await risk.start()
    await execution.start()
    await asyncio.sleep(1.5)  # let consumer groups attach
    print("      Services started.")

    print("[3/5] Publishing RebalanceRequest (AAPL 5% of $10k @ $100 = 5 shares) ...")
    await bus.publish(
        REBALANCE,
        RebalanceRequestEvent(
            target_allocations={"AAPL": 0.05},
            reference_prices={"AAPL": 100.0},
            source_service="smoke",
        ),
    )

    print("[4/5] Waiting for the order to flow over Redis and fill ...")
    broker_ok = risk_ok = False
    for _ in range(30):
        await asyncio.sleep(0.5)
        positions = await broker.get_positions()
        broker_ok = any(p["symbol"] == "AAPL" for p in positions)
        risk_ok = risk._positions.get("AAPL", 0.0) > 0
        if broker_ok and risk_ok:
            break

    print(f"      broker positions: {await broker.get_positions()}")
    print(f"      risk position state: {risk._positions}")

    print("[5/5] Tearing down ...")
    await execution.stop()
    await bus.close()

    if broker_ok and risk_ok:
        print("\nSMOKE TEST PASSED: rebalance -> risk -> execution -> fill -> risk feedback (over Redis).")
        return 0
    print("\nSMOKE TEST FAILED: order did not complete the round trip.")
    print(f"  broker filled: {broker_ok} | risk saw fill: {risk_ok}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
