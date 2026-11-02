"""Role enforcement on money-moving and configuration routes, authenticated
WebSockets, and attributable audit rows (GO_LIVE.md 2026-11-02, item 6).

Before this change one of 45 routes checked a role: any valid token (default
role ``viewer``) could place and cancel orders, which in live mode reach the
venue; the WebSocket channels broadcast fills and risk alerts to anyone; and the
kill switch left no audit row naming who pulled it.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.dependencies import (
    get_audit_logger,
    get_current_user,
    get_db,
    get_event_bus,
)
from api.main import app
from api.middleware.auth import JWTAuth
from api.websockets import streams
from config.settings import get_settings
from core.events.base import InProcessEventBus

_VALID_MARKET_ORDER = {"symbol": "AAPL", "side": "buy", "order_type": "market", "quantity": "1"}
_INVALID_ORDER = {"symbol": "AAPL", "side": "buy", "order_type": "limit", "quantity": "1"}


async def _stub_db():
    yield None


class _AuditSpy:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def log_action(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


@pytest.fixture(autouse=True)
def _overrides(monkeypatch: pytest.MonkeyPatch):
    app.dependency_overrides[get_db] = _stub_db
    # The auth window exists to drop idle anonymous sockets; on a loaded CI
    # box the test client can take longer than 5 s to send its first frame.
    monkeypatch.setattr(streams, "AUTH_TIMEOUT_SECONDS", 60.0)
    yield
    app.dependency_overrides.clear()


def _as(role: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: {"user_id": uuid.uuid4(), "role": role}


_PRIVILEGED_WRITES = [
    ("post", "/api/v1/orders", _INVALID_ORDER),
    ("delete", f"/api/v1/orders/{uuid.uuid4()}", None),
    ("put", "/api/v1/config/risk-rules", {}),
    ("put", "/api/v1/config/trading-mode", {"trading_mode": "live"}),
    ("post", "/api/v1/risk/circuit-breaker/reset", None),
    ("post", f"/api/v1/models/{uuid.uuid4()}/retrain", None),
    ("post", "/api/v1/portfolio/rebalance", None),
    ("post", "/api/v1/admin/control", {"action": "halt"}),
]


@pytest.mark.parametrize("method,path,body", _PRIVILEGED_WRITES)
def test_viewer_token_is_forbidden_on_every_privileged_write(method, path, body) -> None:
    _as("viewer")
    with TestClient(app) as client:
        response = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert response.status_code == 403, (method, path, response.text)


@pytest.mark.parametrize("method,path,body", _PRIVILEGED_WRITES)
def test_admin_token_passes_the_role_gate(method, path, body) -> None:
    """Admin is never 403; the route then does whatever it does (422 on the
    invalid body, 501 on a stub, 202 on control)."""
    _as("admin")
    app.dependency_overrides[get_event_bus] = lambda: InProcessEventBus()
    app.dependency_overrides[get_audit_logger] = lambda: _AuditSpy()
    # The stub DB makes routes that query it fail after the gate; that 500 is
    # the point (the gate was passed), so do not let the client re-raise it.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert response.status_code != 403, (method, path, response.text)
    assert response.status_code in (202, 404, 422, 500, 501), (method, path, response.status_code)


def test_trader_may_submit_orders_but_not_change_configuration() -> None:
    _as("trader")
    with TestClient(app) as client:
        assert client.post("/api/v1/orders", json=_INVALID_ORDER).status_code == 422  # past the gate
        assert client.put("/api/v1/config/trading-mode", json={}).status_code == 403
        assert client.post("/api/v1/admin/control", json={"action": "halt"}).status_code == 403


def test_kill_switch_writes_an_attributable_audit_row() -> None:
    actor = uuid.uuid4()
    app.dependency_overrides[get_current_user] = lambda: {"user_id": actor, "role": "admin"}
    bus = InProcessEventBus()
    spy = _AuditSpy()
    app.dependency_overrides[get_event_bus] = lambda: bus
    app.dependency_overrides[get_audit_logger] = lambda: spy
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/admin/control", json={"action": "halt", "reason": "drill"}
        )
    assert response.status_code == 202
    assert len(spy.rows) == 1
    row = spy.rows[0]
    assert row["action"] == "control.halt"
    assert row["user_id"] == actor
    assert row["ip_address"]  # the caller's address, not None
    assert row["details"] == {"reason": "drill"}
    assert any(e.event_type == "TradingControlEvent" for _, e in bus.history)


def _token(role: str = "viewer") -> str:
    settings = get_settings()
    return JWTAuth(
        secret=settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    ).create_access_token(user_id=str(uuid.uuid4()), role=role)


def test_websocket_without_valid_auth_message_is_closed_with_1008() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/prices") as ws:
            ws.send_json({"type": "auth", "token": "not-a-jwt"})
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == 1008
        with client.websocket_connect("/api/v1/ws/orders") as ws:
            ws.send_json({"type": "ping"})  # wrong first message
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == 1008


def test_websocket_with_valid_first_message_joins_the_channel() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/prices") as ws:
            ws.send_json({"type": "auth", "token": _token()})
            assert ws.receive_json() == {"type": "auth_ok", "channel": "prices"}


def test_token_in_the_url_is_ignored_not_honoured() -> None:
    """A token in the query string must not authenticate (it would be logged)."""
    with TestClient(app) as client:
        with client.websocket_connect(f"/api/v1/ws/prices?token={_token()}") as ws:
            ws.send_json({"type": "ping"})
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == 1008
