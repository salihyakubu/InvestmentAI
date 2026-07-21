"""Redis-backed EventBus must redeliver crashed-handler messages.

The consume loop reads only '>' (never-delivered), so a message whose handler
crashed sits in the pending-entries list forever without XAUTOCLAIM. These
tests pin the reclaim path: claimed entries re-run through handler + ack, and
poison messages (delivered more than MAX_DELIVERIES times) are dead-lettered
(acked + logged) instead of retrying forever.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import core.events.base as base_mod
from core.events.base import (
    MAX_DELIVERIES,
    PENDING_MIN_IDLE_MS,
    Event,
    EventBus,
)


def _event_json(event_type: str = "TestEvent") -> str:
    return Event(event_type=event_type, source_service="test").model_dump_json()


class _FakeRedis:
    """Fake redis-py asyncio client covering the reclaim + ack surface.

    Responses mirror redis-py's parsed shapes: xautoclaim returns
    [next_start_id, [(id, fields), ...], [deleted_ids]] and xpending_range
    returns dicts with 'message_id' / 'times_delivered' keys.
    """

    def __init__(
        self,
        claimed: list[tuple[str | None, dict[str, Any] | None]] | None = None,
        pending: list[dict[str, Any]] | None = None,
    ) -> None:
        self.claimed = claimed or []
        self.pending = pending or []
        self.acked: list[str] = []
        self.xautoclaim_calls: list[dict[str, Any]] = []
        self.xpending_calls: list[dict[str, Any]] = []
        self.xreadgroup_calls = 0

    async def xautoclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        start_id: str = "0-0",
        count: int | None = None,
    ) -> list[Any]:
        self.xautoclaim_calls.append(
            {
                "stream": stream,
                "group": group,
                "consumer": consumer,
                "min_idle_time": min_idle_time,
                "start_id": start_id,
                "count": count,
            }
        )
        return ["0-0", self.claimed, []]

    async def xpending_range(self, stream: str, group: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.xpending_calls.append({"stream": stream, "group": group, **kwargs})
        return self.pending

    async def xack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append(message_id)

    async def xgroup_create(self, stream: str, group: str, id: str = "0", mkstream: bool = False) -> None:
        return None

    async def xreadgroup(self, *args: Any, **kwargs: Any) -> list[Any]:
        # First call: idle stream; second call: shut the consume loop down.
        self.xreadgroup_calls += 1
        if self.xreadgroup_calls >= 2:
            raise asyncio.CancelledError
        return []


class _RecordingHandler:
    def __init__(self, fail_types: set[str] | None = None) -> None:
        self.events: list[Event] = []
        self._fail_types = fail_types or set()

    async def __call__(self, event: Event) -> None:
        if event.event_type in self._fail_types:
            raise RuntimeError(f"handler crash for {event.event_type}")
        self.events.append(event)


def _bus_with(fake: _FakeRedis) -> EventBus:
    bus = EventBus(redis_url="redis://unused")
    bus._redis = fake  # pre-seed the lazy connection with the fake
    return bus


@pytest.mark.asyncio
async def test_claimed_message_runs_through_handler_and_acks() -> None:
    fake = _FakeRedis(
        claimed=[("1-0", {"event": _event_json()})],
        pending=[{"message_id": "1-0", "consumer": "c", "time_since_delivered": 400_000, "times_delivered": 2}],
    )
    bus = _bus_with(fake)
    handler = _RecordingHandler()

    await bus._claim_stale_pending(fake, "orders", "grp", "c", handler, 10)

    assert [e.event_type for e in handler.events] == ["TestEvent"]
    assert fake.acked == ["1-0"]
    # The claim must target entries idle past the 5-minute threshold.
    call = fake.xautoclaim_calls[0]
    assert call["min_idle_time"] == PENDING_MIN_IDLE_MS
    assert call["start_id"] == "0-0"


@pytest.mark.asyncio
async def test_poison_message_is_dead_lettered_without_handler_call() -> None:
    fake = _FakeRedis(
        claimed=[("1-0", {"event": _event_json()})],
        pending=[
            {
                "message_id": "1-0",
                "consumer": "c",
                "time_since_delivered": 400_000,
                "times_delivered": MAX_DELIVERIES + 1,
            }
        ],
    )
    bus = _bus_with(fake)
    handler = _RecordingHandler()

    await bus._claim_stale_pending(fake, "orders", "grp", "c", handler, 10)

    assert handler.events == []  # poison payloads never reach the handler
    assert fake.acked == ["1-0"]  # but they leave the PEL for good


@pytest.mark.asyncio
async def test_mixed_batch_dead_letters_poison_and_replays_the_rest() -> None:
    fake = _FakeRedis(
        claimed=[
            ("1-0", {"event": _event_json("PoisonEvent")}),
            (None, None),  # entry trimmed from the stream but still in the PEL
            ("3-0", {"event": _event_json("HealthyEvent")}),
        ],
        pending=[
            {"message_id": "1-0", "consumer": "c", "time_since_delivered": 400_000, "times_delivered": 9},
            {"message_id": "3-0", "consumer": "c", "time_since_delivered": 400_000, "times_delivered": 1},
        ],
    )
    bus = _bus_with(fake)
    handler = _RecordingHandler()

    await bus._claim_stale_pending(fake, "orders", "grp", "c", handler, 10)

    assert [e.event_type for e in handler.events] == ["HealthyEvent"]
    assert sorted(fake.acked) == ["1-0", "3-0"]


@pytest.mark.asyncio
async def test_crashing_handler_leaves_message_pending() -> None:
    """A still-failing reclaimed message must stay in the PEL (no ack)."""
    fake = _FakeRedis(
        claimed=[("1-0", {"event": _event_json("CrashEvent")})],
        pending=[{"message_id": "1-0", "consumer": "c", "time_since_delivered": 400_000, "times_delivered": 3}],
    )
    bus = _bus_with(fake)
    handler = _RecordingHandler(fail_types={"CrashEvent"})

    await bus._claim_stale_pending(fake, "orders", "grp", "c", handler, 10)

    assert fake.acked == []


@pytest.mark.asyncio
async def test_consume_loop_runs_claim_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consume loop itself must fire XAUTOCLAIM once the interval elapses."""
    monkeypatch.setattr(base_mod, "_jittered_claim_interval", lambda: 0.0)
    fake = _FakeRedis()
    bus = _bus_with(fake)
    handler = _RecordingHandler()

    # xreadgroup cancels on its second call, ending the loop like close() does.
    await bus._consume_loop("orders", "grp", "c", handler, 10, 10)

    assert fake.xautoclaim_calls, "claim cycle never ran inside the consume loop"


def test_jittered_claim_interval_bounds() -> None:
    """Jitter spreads consumers across 0.5x-1.5x of the base interval."""
    for _ in range(200):
        interval = base_mod._jittered_claim_interval()
        assert 0.5 * base_mod.CLAIM_INTERVAL_S <= interval <= 1.5 * base_mod.CLAIM_INTERVAL_S
