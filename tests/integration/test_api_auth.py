"""API-layer tests: authentication is enforced on the money endpoints, and
order input validation is applied at the API boundary.

These are the first API-level tests; the review flagged that none existed and
that it was unclear whether auth was actually wired into the routes.

``get_db`` is overridden with a harmless stub: without a real database the app's
session factory is ``None`` and ``get_db`` would 500 before auth runs, masking
what we want to assert (in production the factory exists and auth gates first).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_db
from api.main import app

_VALID_MARKET_ORDER = {
    "symbol": "AAPL",
    "side": "buy",
    "order_type": "market",
    "quantity": "1",
}


async def _stub_db():
    yield None


@pytest.fixture(autouse=True)
def _override_db():
    app.dependency_overrides[get_db] = _stub_db
    yield
    app.dependency_overrides.clear()


def test_list_orders_requires_auth() -> None:
    with TestClient(app) as client:
        assert client.get("/api/v1/orders").status_code == 401


def test_create_order_requires_auth() -> None:
    with TestClient(app) as client:
        assert client.post("/api/v1/orders", json=_VALID_MARKET_ORDER).status_code == 401


def test_cancel_order_requires_auth() -> None:
    with TestClient(app) as client:
        assert client.delete(f"/api/v1/orders/{uuid.uuid4()}").status_code == 401


def test_create_order_rejects_invalid_body() -> None:
    """With auth satisfied, an invalid order body is rejected (422) by the
    OrderCreate validation added on the order path."""
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": uuid.uuid4(),
        "role": "admin",
    }
    with TestClient(app) as client:
        # limit order missing the required limit_price -> 422
        bad = {"symbol": "AAPL", "side": "buy", "order_type": "limit", "quantity": "1"}
        assert client.post("/api/v1/orders", json=bad).status_code == 422

        # non-positive quantity -> 422
        zero_qty = {**_VALID_MARKET_ORDER, "quantity": "0"}
        assert client.post("/api/v1/orders", json=zero_qty).status_code == 422
