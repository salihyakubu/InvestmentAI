"""FastAPI dependency injection functions."""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings, get_settings as _get_settings
from core.events.base import EventBus
from api.middleware.auth import JWTAuth

logger = logging.getLogger(__name__)


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session from the app-level session factory."""
    factory = request.app.state.db_session_factory
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_redis(request: Request):
    """Return the shared Redis client stored in app.state."""
    return request.app.state.redis


async def get_event_bus(request: Request) -> EventBus:
    """Return the shared EventBus stored in app.state."""
    return request.app.state.event_bus


async def get_settings() -> Settings:
    """Return application settings."""
    return _get_settings()


async def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Extract and validate the current user from the JWT bearer token.

    Returns a dict with ``user_id`` and ``role`` fields.
    Raises 401 if the token is missing or invalid.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix("Bearer ").strip()
    jwt_auth = JWTAuth(
        secret=settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    payload = jwt_auth.verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": UUID(payload["sub"]),
        "role": payload.get("role", "viewer"),
    }
