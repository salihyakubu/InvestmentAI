"""Integration tests for the in-process event bus."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.events.base import Event, InProcessEventBus


@pytest.mark.asyncio
async def test_publish_and_subscribe(event_bus: InProcessEventBus) -> None:
    """Publishing an event should invoke the registered handler."""
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await event_bus.subscribe(
        stream="test.stream",
        group="test-group",
        consumer="consumer-1",
        handler=handler,
    )

    test_event = Event(
        event_type="TestEvent",
        source_service="test",
        payload={"key": "value"},
    )
    message_id = await event_bus.publish("test.stream", test_event)

    assert len(received) == 1
    assert received[0].event_type == "TestEvent"
    assert received[0].payload == {"key": "value"}
    assert message_id  # Non-empty message ID


@pytest.mark.asyncio
async def test_multiple_consumers(event_bus: InProcessEventBus) -> None:
    """Multiple subscribers on the same stream should all receive
    the event."""
    results_a: list[str] = []
    results_b: list[str] = []

    async def handler_a(event: Event) -> None:
        results_a.append(event.event_type)

    async def handler_b(event: Event) -> None:
        results_b.append(event.event_type)

    await event_bus.subscribe(
        stream="shared.stream",
        group="group-a",
        consumer="consumer-a",
        handler=handler_a,
    )
    await event_bus.subscribe(
        stream="shared.stream",
        group="group-b",
        consumer="consumer-b",
        handler=handler_b,
    )

    event = Event(event_type="SharedEvent", source_service="test")
    await event_bus.publish("shared.stream", event)

    assert results_a == ["SharedEvent"]
    assert results_b == ["SharedEvent"]


@pytest.mark.asyncio
async def test_event_acknowledgment(event_bus: InProcessEventBus) -> None:
    """Acknowledgement should be a no-op on InProcessEventBus but
    should not raise. Published events should appear in history."""
    async def noop_handler(event: Event) -> None:
        pass

    await event_bus.subscribe(
        stream="ack.stream",
        group="ack-group",
        consumer="ack-consumer",
        handler=noop_handler,
    )

    event = Event(event_type="AckEvent", source_service="test")
    msg_id = await event_bus.publish("ack.stream", event)

    # ack should not raise.
    await event_bus.ack("ack.stream", "ack-group", msg_id)

    # History should contain the published event.
    assert len(event_bus.history) == 1
    stream_name, recorded_event = event_bus.history[0]
    assert stream_name == "ack.stream"
    assert recorded_event.event_type == "AckEvent"


@pytest.mark.asyncio
async def test_no_cross_stream_delivery(event_bus: InProcessEventBus) -> None:
    """Events published to one stream should not be delivered to
    handlers registered on a different stream."""
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    await event_bus.subscribe(
        stream="stream.a",
        group="group",
        consumer="c1",
        handler=handler,
    )

    event = Event(event_type="WrongStream", source_service="test")
    await event_bus.publish("stream.b", event)

    assert len(received) == 0
