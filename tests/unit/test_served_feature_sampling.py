"""Serve-time feature sampling: complete, ordered, hashed -- or nothing.

A partial vector cannot be replayed through a challenger, so completeness is
a hard gate; the ordering hash lets a future replay refuse rows from a
different pipeline generation instead of silently mis-aligning columns.
"""

from __future__ import annotations

from datetime import UTC, datetime

from core.events.base import InProcessEventBus
from services.prediction.service import PredictionService


class _StubServer:
    feature_names = ["alpha", "beta", "gamma"]
    active_model_types = ["xgboost"]


def _service() -> PredictionService:
    return PredictionService(
        event_bus=InProcessEventBus(),  # type: ignore[arg-type]
        model_server=_StubServer(),  # type: ignore[arg-type]
    )


def test_complete_vector_is_ordered_rounded_and_hashed(monkeypatch) -> None:
    svc = _service()
    monkeypatch.setattr(svc, "_should_persist_features", lambda now: True)
    features = {"gamma": 3.14159265, "alpha": 1.0000004, "beta": -2.5}
    vector, digest = svc._sampled_features(features)
    # Trained column order, NOT dict/alphabetical-insertion order.
    assert vector == [1.0, -2.5, 3.14159]
    assert isinstance(digest, str) and len(digest) == 16
    # The hash identifies the ORDERING: same names, same hash, every time.
    assert digest == svc._sampled_features(features)[1]


def test_incomplete_vector_is_refused_entirely(monkeypatch) -> None:
    """Half a vector is worse than none: it cannot be replayed."""
    svc = _service()
    monkeypatch.setattr(svc, "_should_persist_features", lambda now: True)
    vector, digest = svc._sampled_features({"alpha": 1.0, "beta": 2.0})
    assert vector is None and digest is None


def test_unsampled_minutes_attach_nothing() -> None:
    svc = _service()
    assert svc._should_persist_features(
        datetime(2026, 8, 9, 12, 15, tzinfo=UTC)
    ) is True
    assert svc._should_persist_features(
        datetime(2026, 8, 9, 12, 17, tzinfo=UTC)
    ) is False


def test_no_model_server_means_no_features(monkeypatch) -> None:
    svc = PredictionService(event_bus=InProcessEventBus())  # type: ignore[arg-type]
    monkeypatch.setattr(svc, "_should_persist_features", lambda now: True)
    assert svc._sampled_features({"alpha": 1.0}) == (None, None)
