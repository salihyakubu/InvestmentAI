"""CONTROL-stream log hygiene in the execution engine.

Every consumer group on the shared CONTROL stream sees every control event, so
actions owned by other services (e.g. ``retrain`` -> continuous learning) must
not be logged as 'control_unknown_action' warnings — runbook readers would
misread a successful retrain request as a failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import Settings
from core.events.base import InProcessEventBus
from core.events.order_events import TradingControlEvent
from services.execution.service import ExecutionEngineService


class _RecordingLogger:
    """Captures (level, event) pairs independent of global structlog config."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def debug(self, event: str, **kwargs: Any) -> None:
        self.calls.append(("debug", event))

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append(("info", event))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.calls.append(("warning", event))

    def error(self, event: str, **kwargs: Any) -> None:
        self.calls.append(("error", event))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.calls.append(("exception", event))


def _engine(settings: Settings) -> ExecutionEngineService:
    return ExecutionEngineService(
        event_bus=InProcessEventBus(), settings=settings, brokers={}
    )


@pytest.mark.asyncio
async def test_foreign_action_logs_debug_not_warning(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'retrain' is owned by continuous learning: debug, never a warning."""
    recorder = _RecordingLogger()
    monkeypatch.setattr("services.execution.service.logger", recorder)
    engine = _engine(mock_settings)

    await engine._handle_control(
        TradingControlEvent(action="retrain", source_service="test")
    )

    events = {(level, event) for level, event in recorder.calls}
    assert ("debug", "control_foreign_action") in events
    assert all(event != "control_unknown_action" for _level, event in recorder.calls)
    # And the foreign action must not trip the kill switch.
    assert engine.halted is False


@pytest.mark.asyncio
async def test_control_command_received_is_debug_for_foreign_actions(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt line itself must not warn for foreign actions either."""
    recorder = _RecordingLogger()
    monkeypatch.setattr("services.execution.service.logger", recorder)
    engine = _engine(mock_settings)

    await engine._handle_control(
        TradingControlEvent(action="retrain", source_service="test")
    )

    assert ("debug", "control_command_received") in recorder.calls
    warnings = [event for level, event in recorder.calls if level == "warning"]
    assert warnings == []  # a retrain pass leaves the WARNING channel silent


@pytest.mark.asyncio
async def test_control_command_received_stays_warning_for_operator_actions(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """halt is operator-critical: receipt must stay at WARNING."""
    recorder = _RecordingLogger()
    monkeypatch.setattr("services.execution.service.logger", recorder)
    engine = _engine(mock_settings)

    await engine._handle_control(
        TradingControlEvent(action="halt", source_service="test")
    )

    assert ("warning", "control_command_received") in recorder.calls
    assert engine.halted is True


@pytest.mark.asyncio
async def test_genuinely_unknown_action_still_warns(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr("services.execution.service.logger", recorder)
    engine = _engine(mock_settings)

    await engine._handle_control(
        TradingControlEvent(action="self_destruct", source_service="test")
    )

    assert ("warning", "control_unknown_action") in recorder.calls
    assert all(event != "control_foreign_action" for _level, event in recorder.calls)
