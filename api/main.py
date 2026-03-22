"""FastAPI application entry point for the InvestAI platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from config.logging_config import setup_logging
from config.settings import get_settings
from core.events.base import EventBus
from core.models.base import get_async_session_factory

from api.routers import (
    audit,
    backtest,
    config_router,
    health,
    market_data,
    models,
    orders,
    portfolio,
    positions,
    predictions,
    risk,
)
from api.websockets.streams import router as ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    settings = get_settings()

    # Logging
    setup_logging(log_level="INFO", json_output=False)
    logger.info("Starting InvestAI API...")

    # Redis (optional -- degrades gracefully if unavailable)
    redis_client = None
    event_bus = None
    try:
        redis_client = aioredis.from_url(
            settings.redis_url, decode_responses=True
        )
        await redis_client.ping()
        event_bus = EventBus(redis_url=settings.redis_url)
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis unavailable, running without cache/events: %s", exc)
        redis_client = None

    app.state.redis = redis_client
    app.state.event_bus = event_bus

    # Database session factory
    try:
        app.state.db_session_factory = get_async_session_factory()
        logger.info("Database configured")
    except Exception as exc:
        logger.warning("Database configuration failed: %s", exc)
        app.state.db_session_factory = None

    # Settings
    app.state.settings = settings

    logger.info("InvestAI API started successfully")
    yield

    # Shutdown
    logger.info("Shutting down InvestAI API...")
    if event_bus is not None:
        await event_bus.close()
    if redis_client is not None:
        await redis_client.aclose()
    logger.info("InvestAI API shutdown complete")


app = FastAPI(
    title="InvestAI",
    version="0.1.0",
    description="AI-powered investment platform API",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS (permissive for development)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
API_V1 = "/api/v1"

app.include_router(health.router, prefix=API_V1, tags=["health"])
app.include_router(market_data.router, prefix=API_V1, tags=["market-data"])
app.include_router(orders.router, prefix=API_V1, tags=["orders"])
app.include_router(portfolio.router, prefix=API_V1, tags=["portfolio"])
app.include_router(positions.router, prefix=API_V1, tags=["positions"])
app.include_router(predictions.router, prefix=API_V1, tags=["predictions"])
app.include_router(risk.router, prefix=API_V1, tags=["risk"])
app.include_router(models.router, prefix=API_V1, tags=["models"])
app.include_router(config_router.router, prefix=API_V1, tags=["config"])
app.include_router(audit.router, prefix=API_V1, tags=["audit"])
app.include_router(backtest.router, prefix=API_V1, tags=["backtest"])
app.include_router(ws_router, prefix=API_V1, tags=["websockets"])


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    """Expose Prometheus metrics."""
    try:
        from prometheus_client import generate_latest

        return generate_latest().decode("utf-8")
    except ImportError:
        return "# prometheus_client not installed\n"
