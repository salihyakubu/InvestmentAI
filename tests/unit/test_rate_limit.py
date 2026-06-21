"""Unit tests for the Redis sliding-window rate limiter.

The limiter was implemented but wired to no routes (review H1); it is now applied
to login and order submission. Route-level 429s need a live Redis, so the limiter
logic is tested here against a fake Redis pipeline.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.middleware.rate_limit import RateLimiter


class _FakePipe:
    def __init__(self, count: int) -> None:
        self._count = count

    def zremrangebyscore(self, *args, **kwargs):
        return self

    def zadd(self, *args, **kwargs):
        return self

    def zcard(self, *args, **kwargs):
        return self

    def expire(self, *args, **kwargs):
        return self

    async def execute(self):
        # Results for: zremrangebyscore, zadd, zcard, expire -> count is index 2.
        return [0, 1, self._count, True]


class _FakeRedis:
    def __init__(self, count: int) -> None:
        self._count = count

    def pipeline(self):
        return _FakePipe(self._count)


class _Request:
    """Minimal stand-in for a Starlette Request the limiter inspects."""

    def __init__(self, redis) -> None:
        self.app = type("_App", (), {"state": type("_S", (), {"redis": redis})()})()
        self.state = type("_St", (), {"user": None})()
        self.client = type("_C", (), {"host": "203.0.113.7"})()


@pytest.mark.asyncio
async def test_under_limit_allowed() -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    await limiter(_Request(_FakeRedis(count=3)))  # no exception


@pytest.mark.asyncio
async def test_over_limit_raises_429() -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    with pytest.raises(HTTPException) as exc:
        await limiter(_Request(_FakeRedis(count=6)))
    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "60"


@pytest.mark.asyncio
async def test_no_redis_fails_open() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    # Redis unavailable -> limiting disabled, no exception even past the limit.
    await limiter(_Request(redis=None))
