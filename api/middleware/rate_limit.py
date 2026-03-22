"""Sliding-window rate limiter backed by Redis."""

from __future__ import annotations

import logging
import time

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter using Redis sorted sets.

    Each request is scored by its timestamp.  Expired entries are pruned
    on every check so the window always reflects the most recent period.

    Usage as a FastAPI dependency::

        limiter = RateLimiter(max_requests=100, window_seconds=60)

        @router.get("/resource")
        async def get_resource(
            request: Request,
            _: None = Depends(limiter),
        ):
            ...
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "rl",
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    def _client_key(self, request: Request) -> str:
        """Derive a rate-limit key from the request.

        Prefers the authenticated user id if available, otherwise falls back
        to the client IP address.
        """
        user = getattr(request.state, "user", None)
        if user and isinstance(user, dict) and "user_id" in user:
            identifier = str(user["user_id"])
        else:
            identifier = request.client.host if request.client else "unknown"
        return f"{self.key_prefix}:{identifier}"

    async def __call__(self, request: Request) -> None:
        """Check the rate limit for the current request.

        Raises ``HTTPException(429)`` when the limit is exceeded.
        """
        redis_client = getattr(request.app.state, "redis", None)
        if redis_client is None:
            # If Redis is not available, skip rate limiting
            logger.warning("Redis not available; rate limiting disabled")
            return

        key = self._client_key(request)
        now = time.time()
        window_start = now - self.window_seconds

        pipe = redis_client.pipeline()
        # Remove entries outside the current window
        pipe.zremrangebyscore(key, 0, window_start)
        # Add the current request
        pipe.zadd(key, {f"{now}": now})
        # Count requests in the window
        pipe.zcard(key)
        # Set TTL so keys don't linger forever
        pipe.expire(key, self.window_seconds + 10)
        results = await pipe.execute()

        request_count: int = results[2]

        if request_count > self.max_requests:
            retry_after = int(self.window_seconds)
            logger.warning(
                "Rate limit exceeded for %s (%d/%d)",
                key,
                request_count,
                self.max_requests,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
