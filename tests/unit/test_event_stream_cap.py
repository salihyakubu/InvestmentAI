"""Redis-backed EventBus.publish must cap streams via XADD MAXLEN ~.

Consumer-group acks never delete stream entries, so an uncapped stream grows
until Redis OOMs and every publish fails. The bus therefore publishes with an
approximate maxlen; these tests pin the exact kwargs sent to redis-py.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.events.base import STREAM_MAXLEN, Event, EventBus


class _FakeRedis:
    """Records xadd calls so tests can assert the capping kwargs."""

    def __init__(self) -> None:
        self.xadd_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def xadd(
        self, stream: str, fields: dict[str, Any], **kwargs: Any
    ) -> str:
        self.xadd_calls.append((stream, fields, kwargs))
        return f"{len(self.xadd_calls)}-0"


@pytest.mark.asyncio
async def test_publish_caps_stream_with_approximate_maxlen() -> None:
    bus = EventBus(redis_url="redis://unused")
    fake = _FakeRedis()
    bus._redis = fake  # pre-seed the lazy connection with the fake

    event = Event(event_type="TestEvent", source_service="test")
    message_id = await bus.publish("market.bars", event)

    assert message_id == "1-0"
    assert len(fake.xadd_calls) == 1
    stream, fields, kwargs = fake.xadd_calls[0]
    assert stream == "market.bars"
    assert "event" in fields  # serialized event payload still intact
    assert kwargs["maxlen"] == STREAM_MAXLEN
    assert kwargs["approximate"] is True


@pytest.mark.asyncio
async def test_every_publish_carries_the_cap() -> None:
    """The cap must apply per-call (no first-call-only trimming)."""
    bus = EventBus(redis_url="redis://unused")
    fake = _FakeRedis()
    bus._redis = fake

    for _ in range(3):
        await bus.publish("orders", Event(event_type="E", source_service="test"))

    assert len(fake.xadd_calls) == 3
    for _stream, _fields, kwargs in fake.xadd_calls:
        assert kwargs["maxlen"] == STREAM_MAXLEN
        assert kwargs["approximate"] is True


def test_stream_maxlen_value() -> None:
    """100k covers ~5-8 days of the busiest stream, far above consumer lag."""
    assert STREAM_MAXLEN == 100_000
