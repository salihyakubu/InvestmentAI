"""Core event infrastructure with Redis Streams transport."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type alias for async event handler callbacks.
EventHandler = Callable[["Event"], Awaitable[None]]


class Event(BaseModel):
    """Base event that flows through the platform event bus."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_service: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    # extra="allow" is essential: the bus deserialises every message as the base
    # Event (`Event.model_validate_json`), so without this, subclass fields
    # (symbol, side, quantity, ...) would be silently dropped on the Redis path
    # and only survive on the in-process bus. Allowing extras preserves them and
    # keeps them accessible via attribute access and model_dump().
    model_config = {"frozen": False, "extra": "allow"}

    def model_post_init(self, _context: Any) -> None:  # noqa: ANN401
        if not self.event_type:
            self.event_type = self.__class__.__name__


class EventBus:
    """Publish / subscribe event bus backed by Redis Streams.

    Each logical stream corresponds to a topic (e.g. ``market.bars``).
    Consumer groups allow competing consumers for horizontal scaling.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._redis: Any | None = None  # lazy import / connect
        self._consumer_tasks: list[asyncio.Task[Any]] = []

    async def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._redis

    async def close(self) -> None:
        """Cancel consumer loops and close the underlying Redis connection."""
        for task in self._consumer_tasks:
            task.cancel()
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        self._consumer_tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, stream: str, event: Event) -> str:
        """Publish an event to a Redis Stream.

        Returns the message ID assigned by Redis.
        """
        r = await self._get_redis()
        data = {"event": event.model_dump_json()}
        message_id: str = await r.xadd(stream, data)
        logger.debug("Published %s to %s (msg=%s)", event.event_type, stream, message_id)
        return message_id

    # ------------------------------------------------------------------
    # Consumer group management
    # ------------------------------------------------------------------

    async def create_consumer_group(
        self, stream: str, group: str, start_id: str = "0"
    ) -> None:
        """Create a consumer group on *stream*, ignoring if it already exists."""
        r = await self._get_redis()
        try:
            await r.xgroup_create(stream, group, id=start_id, mkstream=True)
            logger.info("Created consumer group %s on stream %s", group, stream)
        except Exception as exc:
            # BUSYGROUP means the group already exists – that is fine.
            if "BUSYGROUP" in str(exc):
                logger.debug("Consumer group %s already exists on %s", group, stream)
            else:
                raise

    # ------------------------------------------------------------------
    # Subscribing / consuming
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: EventHandler,
        *,
        batch_size: int = 10,
        block_ms: int = 2000,
    ) -> None:
        """Start a background consumer loop for *stream* and return.

        The loop runs as a tracked asyncio task (cancelled on ``close()``) so a
        single ``start()`` can subscribe to several streams without blocking --
        mirroring :class:`InProcessEventBus` semantics.
        """
        task = asyncio.create_task(
            self._consume_loop(stream, group, consumer, handler, batch_size, block_ms),
            name=f"consume-{group}-{stream}",
        )
        self._consumer_tasks.append(task)

    async def _consume_loop(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: EventHandler,
        batch_size: int,
        block_ms: int,
    ) -> None:
        """Infinite consumer loop: read, dispatch one-at-a-time, then ack."""
        r = await self._get_redis()
        await self.create_consumer_group(stream, group)

        logger.info(
            "Consumer %s/%s subscribed to %s", group, consumer, stream
        )

        while True:
            try:
                entries = await r.xreadgroup(
                    group, consumer, {stream: ">"}, count=batch_size, block=block_ms
                )
                if not entries:
                    continue

                for _stream_name, messages in entries:
                    for message_id, fields in messages:
                        try:
                            event = Event.model_validate_json(fields["event"])
                            await handler(event)
                            await self.ack(stream, group, message_id)
                        except Exception:
                            logger.exception(
                                "Error processing message %s from %s",
                                message_id,
                                stream,
                            )
            except asyncio.CancelledError:
                logger.info("Consumer %s/%s shutting down", group, consumer)
                break
            except Exception:
                logger.exception("Consumer loop error on %s", stream)
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge a message so it is not redelivered."""
        r = await self._get_redis()
        await r.xack(stream, group, message_id)


class InProcessEventBus:
    """In-memory event bus for unit / integration tests (no Redis required).

    Handlers are stored per-stream and invoked sequentially on publish.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[tuple[str, Event]] = []

    async def publish(self, stream: str, event: Event) -> str:
        """Publish an event and invoke all registered handlers."""
        message_id = str(uuid.uuid4())
        self._history.append((stream, event))

        for handler in self._handlers.get(stream, []):
            await handler(event)

        return message_id

    async def subscribe(
        self,
        stream: str,
        group: str,          # noqa: ARG002
        consumer: str,       # noqa: ARG002
        handler: EventHandler,
        **_kwargs: Any,
    ) -> None:
        """Register a handler for *stream*. Does not block."""
        self._handlers[stream].append(handler)

    async def create_consumer_group(
        self, stream: str, group: str, start_id: str = "0"  # noqa: ARG002
    ) -> None:
        """No-op for in-process bus."""

    async def ack(
        self, stream: str, group: str, message_id: str  # noqa: ARG002
    ) -> None:
        """No-op for in-process bus."""

    async def close(self) -> None:
        """No-op for in-process bus."""

    @property
    def history(self) -> list[tuple[str, Event]]:
        """Return all published (stream, event) pairs for test assertions."""
        return list(self._history)
