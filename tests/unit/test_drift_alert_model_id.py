"""Drift alerts must identify WHICH model drifted.

DriftDetectedEvent used to lack a model_id, so the operator alert could not
name the drifting model. The event now carries it and the monitoring alert
must surface it in the details dict.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.events.system_events import DriftDetectedEvent
from services.monitoring.service import MonitoringService


def _make_service() -> tuple[MonitoringService, MagicMock]:
    alert_manager = MagicMock()
    alert_manager.send_alert = AsyncMock()
    svc = MonitoringService(
        event_bus=MagicMock(),
        settings=MagicMock(),
        metrics=MagicMock(),
        health_checker=MagicMock(),
        alert_manager=alert_manager,
        audit_logger=MagicMock(),
    )
    return svc, alert_manager


@pytest.mark.asyncio
async def test_drift_alert_includes_model_id() -> None:
    svc, alert_manager = _make_service()

    await svc._handle_system_event(
        DriftDetectedEvent(
            source_service="continuous_learning",
            drift_type="prediction",
            score=0.08,
            threshold=0.05,
            model_id="ensemble-v3",
        )
    )

    alert_manager.send_alert.assert_awaited_once()
    details = alert_manager.send_alert.await_args.kwargs["details"]
    assert details["model_id"] == "ensemble-v3"


@pytest.mark.asyncio
async def test_drift_alert_defaults_empty_model_id() -> None:
    """Platform-wide (data) drift events carry no model_id; the alert must
    still go out with an empty string rather than fail."""
    svc, alert_manager = _make_service()

    await svc._handle_system_event(
        DriftDetectedEvent(
            source_service="continuous_learning",
            drift_type="data",
            score=0.3,
            threshold=0.2,
        )
    )

    details = alert_manager.send_alert.await_args.kwargs["details"]
    assert details["model_id"] == ""
