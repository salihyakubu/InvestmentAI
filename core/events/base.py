"""Core event infrastructure with Redis Streams transport."""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type alias for async event handler callbacks.
EventHandler = Callable[["Event"], Awaitable[None]]

# Approximate per-stream cap for XADD MAXLEN. Consumer-group acks never delete
# entries, so without a cap streams grow until Redis OOMs. 100k covers ~5-8
# days of the busiest stream — far above worst-case consumer lag.
STREAM_MAXLEN = 100_000

# The consume loop reads only '>' (never-delivered messages), so a message
# whose handler crashed mid-run sits in the pending-entries list forever — a
# restart never re-reads it. Each consumer therefore runs XAUTOCLAIM roughly
# every CLAIM_INTERVAL_S (jittered per consumer so a fleet restart doesn't
# stampede Redis) to reclaim entries pending longer than PENDING_MIN_IDLE_MS.
CLAIM_INTERVAL_S = 60.0
PENDING_MIN_IDLE_MS = 5 * 60 * 1000

# A message delivered more than this many times keeps crashing its handler:
# treat it as poison and dead-letter (ack + log) instead of retrying forever.
MAX_DELIVERIES = 5


def _jittered_claim_interval() -> float:
    """Next XAUTOCLAIM tick: CLAIM_INTERVAL_S scaled by a per-tick 0.5-1.5x."""
    return CLAIM_INTERVAL_S * random.uniform(0.5, 1.5)


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
        message_id: str = await r.xadd(
            stream, data, maxlen=STREAM_MAXLEN, approximate=True
        )
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

        loop = asyncio.get_running_loop()
        next_claim_at = loop.time() + _jittered_claim_interval()

        while True:
            try:
                # Claim before the blocking read so a claim-cycle error cannot
                # discard a freshly read batch. xreadgroup blocks at most
                # block_ms, so the tick fires within ~block_ms of schedule
                # even on idle streams.
                if loop.time() >= next_claim_at:
                    next_claim_at = loop.time() + _jittered_claim_interval()
                    await self._claim_stale_pending(
                        r, stream, group, consumer, handler, batch_size
                    )

                entries = await r.xreadgroup(
                    group, consumer, {stream: ">"}, count=batch_size, block=block_ms
                )
                if not entries:
                    continue

                for _stream_name, messages in entries:
                    for message_id, fields in messages:
                        await self._dispatch(stream, group, message_id, fields, handler)
            except asyncio.CancelledError:
                logger.info("Consumer %s/%s shutting down", group, consumer)
                break
            except Exception:
                logger.exception("Consumer loop error on %s", stream)
                await asyncio.sleep(1)

    async def _dispatch(
        self,
        stream: str,
        group: str,
        message_id: str,
        fields: dict[str, Any],
        handler: EventHandler,
    ) -> None:
        """Run one message through handler + ack; a crash leaves it pending."""
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

    async def _claim_stale_pending(
        self,
        r: Any,
        stream: str,
        group: str,
        consumer: str,
        handler: EventHandler,
        batch_size: int,
    ) -> None:
        """Reclaim PEL entries idle > PENDING_MIN_IDLE_MS and re-dispatch them.

        Poison messages (delivered more than MAX_DELIVERIES times, per the
        XPENDING delivery counter — XAUTOCLAIM does not return it) are acked
        and logged as dead-lettered so one bad payload cannot loop forever.
        """
        result = await r.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=PENDING_MIN_IDLE_MS,
            start_id="0-0",
            count=batch_size,
        )
        # redis-py parses XAUTOCLAIM to [next_start_id, messages, deleted_ids]
        # (deleted_ids only on Redis >= 7.0). Entries trimmed from the stream
        # but still in the PEL come back as (None, None) pairs — skip them.
        raw_messages = result[1] if len(result) > 1 else []
        messages = [
            (message_id, fields)
            for message_id, fields in raw_messages
            if message_id is not None and fields is not None
        ]
        if not messages:
            return

        delivery_counts: dict[str, int] = {}
        pending = await r.xpending_range(
            stream,
            group,
            min=messages[0][0],
            max=messages[-1][0],
            count=len(messages),
            consumername=consumer,
        )
        for entry in pending:
            delivery_counts[entry["message_id"]] = int(entry["times_delivered"])

        for message_id, fields in messages:
            if delivery_counts.get(message_id, 1) > MAX_DELIVERIES:
                await self.ack(stream, group, message_id)
                logger.error(
                    "event.dead_lettered stream=%s id=%s deliveries=%s",
                    stream,
                    message_id,
                    delivery_counts.get(message_id),
                )
                continue
            logger.warning(
                "Reclaimed stale pending message %s from %s (deliveries=%s)",
                message_id,
                stream,
                delivery_counts.get(message_id),
            )
            await self._dispatch(stream, group, message_id, fields, handler)

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
