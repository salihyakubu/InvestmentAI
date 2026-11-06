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


@pytest.mark.filterwarnings("error::sqlalchemy.exc.SADeprecationWarning")
async def test_login_lookup_is_one_users_select_with_api_keys_unloaded(tmp_path) -> None:
    """Pin the login query's behaviour across the SQLAlchemy 2.0 -> 2.1 move.

    One SELECT, from users only: api_keys is never queried (the table does not
    even exist in this harness, so any attempt would fail loudly). Every User
    column arrives with the seeded values, and the api_keys relationship is
    left UNLOADED and refuses access. 2.0's noload() faked an empty collection
    there, which is exactly why 2.1 deprecates it; the replacement must refuse
    rather than lie. The deprecation warning is promoted to an error so the
    noload() path cannot come back silently.
    """
    from sqlalchemy import event, inspect
    from sqlalchemy.exc import InvalidRequestError

    from api.middleware.auth import verify_password
    from api.routers.auth import login_user_query

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}", poolclass=NullPool
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()))

    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=[User.__table__])
    seeded_at = datetime.now(UTC)
    async with factory() as s:
        s.add(
            User(
                email=_EMAIL,
                hashed_password=hash_password(_PASSWORD),
                role="admin",
                is_active=True,
                created_at=seeded_at,
            )
        )
        await s.commit()

    async with factory() as s:
        statements.clear()
        user = (await s.execute(login_user_query(_EMAIL))).scalar_one()

        assert len(statements) == 1, statements
        assert "FROM users" in statements[0]
        assert "api_keys" not in statements[0]

        assert isinstance(user.id, uuid.UUID)
        assert (user.email, user.role, user.is_active) == (_EMAIL, "admin", True)
        assert verify_password(_PASSWORD, user.hashed_password)
        assert user.created_at.replace(tzinfo=UTC) == seeded_at
        assert not inspect(user).unloaded - {"api_keys"}, "a column was left unloaded"

        assert "api_keys" in inspect(user).unloaded
        with pytest.raises(InvalidRequestError):
            _ = user.api_keys
        assert len(statements) == 1, "touching api_keys must not issue a query"

    await engine.dispose()
