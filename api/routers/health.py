"""Health check endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.dependencies import get_db, get_redis
from api.schemas.common import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Overall system health",
)
async def health_check(request: Request) -> HealthResponse:
    """Check the health of all core services (database, Redis, brokers)."""
    services: dict[str, str] = {}

    # Database
    try:
        factory = request.app.state.db_session_factory
        async with factory() as session:
            await session.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        services["database"] = "ok"
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        services["database"] = "down"

    # Redis
    try:
        redis_client = request.app.state.redis
        await redis_client.ping()
        services["redis"] = "ok"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        services["redis"] = "down"

    # Determine overall status
    statuses = list(services.values())
    if all(s == "ok" for s in statuses):
        overall = "ok"
    elif any(s == "ok" for s in statuses):
        overall = "degraded"
    else:
        overall = "down"

    return HealthResponse(status=overall, services=services)


@router.get(
    "/{service}",
    response_model=HealthResponse,
    summary="Individual service health",
)
async def service_health(service: str, request: Request) -> HealthResponse:
    """Check the health of a specific service by name."""
    services: dict[str, str] = {}

    if service == "database":
        try:
            factory = request.app.state.db_session_factory
            async with factory() as session:
                await session.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
            services["database"] = "ok"
        except Exception:
            services["database"] = "down"

    elif service == "redis":
        try:
            redis_client = request.app.state.redis
            await redis_client.ping()
            services["redis"] = "ok"
        except Exception:
            services["redis"] = "down"

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown service: {service}",
        )

    overall = "ok" if all(v == "ok" for v in services.values()) else "down"
    return HealthResponse(status=overall, services=services)
