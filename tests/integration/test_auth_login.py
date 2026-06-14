"""API-layer tests for the login endpoint that issues JWT access tokens.

The rest of the API requires a JWT but nothing issued one before; this verifies
the new /auth/token endpoint end-to-end against a seeded SQLite user.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.dependencies import get_db
from api.main import app
from api.middleware.auth import JWTAuth, hash_password
from config.settings import get_settings
from core.models.base import AsyncBase
from core.models.users import User

_EMAIL = "trader@example.com"
_PASSWORD = "s3cret-pw"


@pytest.fixture
def client_with_user():
    # File-based SQLite + NullPool so connections opened in the setup loop and in
    # the TestClient's loop both see the same data (in-memory SQLite is
    # per-connection and would not be shared).
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(AsyncBase.metadata.create_all, tables=[User.__table__])
        async with factory() as s:
            s.add(
                User(
                    email=_EMAIL,
                    hashed_password=hash_password(_PASSWORD),
                    role="admin",
                    is_active=True,
                    created_at=datetime.now(UTC),
                )
            )
            await s.commit()

    asyncio.run(_setup())

    async def _override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())
        os.unlink(path)


def test_login_success_returns_usable_token(client_with_user) -> None:
    resp = client_with_user.post(
        "/api/v1/auth/token", json={"email": _EMAIL, "password": _PASSWORD}
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    # The issued token must be one get_current_user would accept.
    settings = get_settings()
    payload = JWTAuth(
        secret=settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    ).verify_token(token)
    assert payload is not None
    assert payload["role"] == "admin"
    uuid.UUID(payload["sub"])  # sub is a valid user id


def test_login_wrong_password_rejected(client_with_user) -> None:
    resp = client_with_user.post(
        "/api/v1/auth/token", json={"email": _EMAIL, "password": "wrong"}
    )
    assert resp.status_code == 401


def test_login_unknown_user_rejected(client_with_user) -> None:
    resp = client_with_user.post(
        "/api/v1/auth/token", json={"email": "nobody@example.com", "password": _PASSWORD}
    )
    assert resp.status_code == 401
