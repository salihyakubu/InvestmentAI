"""Health checking for platform services and dependencies."""

from __future__ import annotations

import enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, enum.Enum):
    """Possible health states for a checked component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthChecker:
    """Checks connectivity and health of external dependencies.

    Parameters:
        database_url: PostgreSQL connection string.
        redis_url: Redis connection string.
    """

    def __init__(
        self,
        database_url: str = "postgresql+asyncpg://investai:investai@localhost:5432/investai",
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self._database_url = database_url
        self._redis_url = redis_url

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def check_database(self) -> bool:
        """Return ``True`` if the database responds to a simple query."""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(self._database_url, echo=False)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return True
        except Exception:
            logger.warning("health.database.unreachable", exc_info=True)
            return False

    async def check_redis(self) -> bool:
        """Return ``True`` if Redis responds to PING."""
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(self._redis_url, decode_responses=True)
            pong = await r.ping()
            await r.aclose()
            return bool(pong)
        except Exception:
            logger.warning("health.redis.unreachable", exc_info=True)
            return False

    async def check_broker(self, broker_name: str) -> bool:
        """Return ``True`` if the named broker API is reachable.

        Currently performs a lightweight connectivity test.  Concrete
        implementations can be extended per broker.
        """
        try:
            if broker_name == "alpaca":
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("https://paper-api.alpaca.markets/v2/clock")
                    return resp.status_code < 500
            elif broker_name == "binance":
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("https://api.binance.com/api/v3/ping")
                    return resp.status_code == 200
            else:
                logger.warning("health.broker.unknown", broker=broker_name)
                return False
        except Exception:
            logger.warning("health.broker.unreachable", broker=broker_name, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    async def check_all(self) -> dict[str, HealthStatus]:
        """Run all health checks and return a mapping of component -> status."""
        results: dict[str, HealthStatus] = {}

        db_ok = await self.check_database()
        results["database"] = HealthStatus.HEALTHY if db_ok else HealthStatus.UNHEALTHY

        redis_ok = await self.check_redis()
        results["redis"] = HealthStatus.HEALTHY if redis_ok else HealthStatus.UNHEALTHY

        for broker in ("alpaca", "binance"):
            broker_ok = await self.check_broker(broker)
            results[f"broker_{broker}"] = (
                HealthStatus.HEALTHY if broker_ok else HealthStatus.DEGRADED
            )

        # Overall status
        statuses = list(results.values())
        if all(s == HealthStatus.HEALTHY for s in statuses):
            results["overall"] = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            results["overall"] = HealthStatus.UNHEALTHY
        else:
            results["overall"] = HealthStatus.DEGRADED

        logger.info("health.check_all.complete", results={k: v.value for k, v in results.items()})
        return results
