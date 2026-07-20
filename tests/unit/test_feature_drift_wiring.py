"""Feature (data) drift wiring in the continuous-learning service.

DataDriftDetector.detect_feature_drift used to have zero callers. The service
now buffers live FeaturesReadyEvent rows PER SYMBOL (bounded float32 deques,
sorted-name column order) and, inside the evaluation cycle, compares each
symbol with >= 500 buffered rows against the training-side reference
distributions in ``model_artifacts/feature_reference.npz``:

  - "feature_names": 1-D str array, sorted-name order.
  - one 2-D float array per symbol, keyed by the raw symbol string
    (slashes like "BTC/USDT" round-trip through np.savez/np.load),
    shape (n_rows, n_features) aligned with "feature_names".

No reference file -> the check is silently skipped. Drifting symbols publish
``DriftDetectedEvent(drift_type="data", model_id="features:<symbol>")`` on the
SYSTEM stream; the retrainer is marked at most once per cycle across all
symbols.
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import services.continuous_learning.service as cl_service
from config.settings import Settings
from core.events.base import InProcessEventBus
from core.events.signal_events import FeaturesReadyEvent
from services.continuous_learning.drift_detector import DataDriftDetector, DriftReport
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.continuous_learning.service import ContinuousLearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(artifact_dir: Path) -> Settings:
    return Settings(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
        model_artifact_path=str(artifact_dir),
    )


def _make_service(
    artifact_dir: Path,
    drift_detector: DataDriftDetector | MagicMock | None = None,
) -> tuple[ContinuousLearningService, InProcessEventBus, MagicMock]:
    bus = InProcessEventBus()
    retrainer = MagicMock()
    retrainer.should_retrain.return_value = False
    svc = ContinuousLearningService(
        event_bus=bus,  # type: ignore[arg-type]
        settings=_settings(artifact_dir),
        evaluator=MagicMock(spec=ModelEvaluator),
        retrainer=retrainer,
        drift_detector=drift_detector or MagicMock(spec=DataDriftDetector),
        feedback_loop=MagicMock(spec=TradingFeedbackLoop),
    )
    return svc, bus, retrainer


def _features_event(
    symbol: str = "BTC/USDT", feature_vector: dict[str, float] | None = None
) -> FeaturesReadyEvent:
    return FeaturesReadyEvent(
        symbol=symbol,
        feature_vector=feature_vector or {"rsi": 50.0, "atr": 1.5},
        source_service="test",
    )


def _write_reference(
    artifact_dir: Path, names: list[str], refs: dict[str, np.ndarray]
) -> None:
    np.savez(
        artifact_dir / "feature_reference.npz",
        feature_names=np.array(names),
        **refs,
    )


def _fill_buffer(
    svc: ContinuousLearningService, symbol: str, rows: np.ndarray
) -> None:
    """Inject pre-built live rows, bypassing the (slower) event handler."""
    buffer: deque[np.ndarray] = deque(maxlen=cl_service._FEATURE_BUFFER_MAXLEN)
    for row in rows.astype(np.float32):
        buffer.append(row)
    svc._feature_buffers[symbol] = buffer


def _data_drift_events(bus: InProcessEventBus) -> list[object]:
    return [
        e
        for _stream, e in bus.history
        if e.event_type == "DriftDetectedEvent"
        and getattr(e, "drift_type", "") == "data"
    ]


# ---------------------------------------------------------------------------
# (a) live-side buffering: sorted-name order, float32 rows, bounded deque
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_rows_use_sorted_name_order(tmp_path: Path) -> None:
    svc, _bus, _retrainer = _make_service(tmp_path)

    # Keys deliberately out of order; the row must follow sorted names.
    await svc.handle_features_ready(
        _features_event(feature_vector={"zeta": 3.0, "alpha": 1.0, "mid": 2.0})
    )

    assert svc._feature_names == ["alpha", "mid", "zeta"]
    (row,) = svc._feature_buffers["BTC/USDT"]
    assert isinstance(row, np.ndarray)
    assert row.dtype == np.float32
    np.testing.assert_allclose(row, [1.0, 2.0, 3.0])


@pytest.mark.asyncio
async def test_buffer_respects_maxlen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cl_service, "_FEATURE_BUFFER_MAXLEN", 5)
    svc, _bus, _retrainer = _make_service(tmp_path)

    for i in range(8):
        await svc.handle_features_ready(
            _features_event(feature_vector={"a": float(i), "b": 0.0})
        )

    buffer = svc._feature_buffers["BTC/USDT"]
    assert buffer.maxlen == 5
    assert len(buffer) == 5
    # Oldest rows evicted: values 0-2 gone, 3-7 kept in arrival order.
    assert [float(r[0]) for r in buffer] == [3.0, 4.0, 5.0, 6.0, 7.0]


@pytest.mark.asyncio
async def test_buffers_are_per_symbol(tmp_path: Path) -> None:
    svc, _bus, _retrainer = _make_service(tmp_path)

    await svc.handle_features_ready(_features_event(symbol="BTC/USDT"))
    await svc.handle_features_ready(_features_event(symbol="AAPL"))
    await svc.handle_features_ready(_features_event(symbol="AAPL"))

    assert len(svc._feature_buffers["BTC/USDT"]) == 1
    assert len(svc._feature_buffers["AAPL"]) == 2


@pytest.mark.asyncio
async def test_mismatched_schema_and_bad_rows_skipped(tmp_path: Path) -> None:
    svc, _bus, _retrainer = _make_service(tmp_path)

    await svc.handle_features_ready(
        _features_event(feature_vector={"a": 1.0, "b": 2.0})
    )
    # Different key set: must not poison the pinned column alignment.
    await svc.handle_features_ready(
        _features_event(feature_vector={"a": 1.0, "c": 2.0})
    )
    # Non-finite values would corrupt PSI percentile binning.
    await svc.handle_features_ready(
        _features_event(feature_vector={"a": float("nan"), "b": 2.0})
    )
    # Missing symbol / missing vector are ignored, not crashes.
    await svc.handle_features_ready(
        FeaturesReadyEvent(symbol="", feature_vector={"a": 1.0}, source_service="t")
    )

    assert svc._feature_names == ["a", "b"]
    assert len(svc._feature_buffers["BTC/USDT"]) == 1


@pytest.mark.asyncio
async def test_start_subscribes_features_stream(tmp_path: Path) -> None:
    """End-to-end over the bus: start() wires handle_features_ready to
    features.ready under the continuous-learning consumer group."""
    svc, bus, _retrainer = _make_service(tmp_path)

    await svc.start()
    try:
        # start() registers subscriptions inside freshly created tasks; yield
        # once so the in-process bus actually has the handler before publish.
        await asyncio.sleep(0)
        await bus.publish("features.ready", _features_event())
        assert len(svc._feature_buffers["BTC/USDT"]) == 1
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# (b) reference side: absent file -> silently skipped; schema guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_reference_file_skips_check_without_crash(tmp_path: Path) -> None:
    detector = MagicMock(spec=DataDriftDetector)
    svc, bus, retrainer = _make_service(tmp_path, drift_detector=detector)

    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", np.zeros((600, 2)))

    await svc._run_evaluation_cycle()

    detector.detect_feature_drift.assert_not_called()
    assert _data_drift_events(bus) == []
    retrainer.mark_drift.assert_not_called()


@pytest.mark.asyncio
async def test_below_min_rows_not_checked(tmp_path: Path) -> None:
    detector = MagicMock(spec=DataDriftDetector)
    svc, _bus, _retrainer = _make_service(tmp_path, drift_detector=detector)

    _write_reference(tmp_path, ["a", "b"], {"BTC/USDT": np.zeros((1000, 2))})
    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", np.zeros((499, 2)))

    await svc._run_evaluation_cycle()

    detector.detect_feature_drift.assert_not_called()


@pytest.mark.asyncio
async def test_reference_schema_mismatch_skips_check(tmp_path: Path) -> None:
    """A stale reference (different feature set) must not be compared
    column-wise against live rows."""
    detector = MagicMock(spec=DataDriftDetector)
    svc, bus, _retrainer = _make_service(tmp_path, drift_detector=detector)

    _write_reference(tmp_path, ["x", "y"], {"BTC/USDT": np.zeros((1000, 2))})
    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", np.zeros((600, 2)))

    await svc._run_evaluation_cycle()

    detector.detect_feature_drift.assert_not_called()
    assert _data_drift_events(bus) == []


@pytest.mark.asyncio
async def test_symbol_without_reference_array_skipped(tmp_path: Path) -> None:
    detector = MagicMock(spec=DataDriftDetector)
    svc, _bus, _retrainer = _make_service(tmp_path, drift_detector=detector)

    _write_reference(tmp_path, ["a", "b"], {"AAPL": np.zeros((1000, 2))})
    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", np.zeros((600, 2)))  # no "BTC/USDT" reference

    await svc._run_evaluation_cycle()

    detector.detect_feature_drift.assert_not_called()


# ---------------------------------------------------------------------------
# (c) drift end-to-end with the REAL detector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drifted_live_features_publish_data_drift_event(
    tmp_path: Path,
) -> None:
    svc, bus, retrainer = _make_service(tmp_path, drift_detector=DataDriftDetector())
    svc._tracked_predictions["model-a"] = []

    rng = np.random.default_rng(7)
    # Reference at N(0,1); live shifted to N(5,1) -- unambiguous PSI drift.
    # The "BTC/USDT" key also pins the slash-in-npz-key round-trip.
    _write_reference(
        tmp_path, ["f0", "f1", "f2"], {"BTC/USDT": rng.normal(0, 1, (1000, 3))}
    )
    svc._feature_names = ["f0", "f1", "f2"]
    _fill_buffer(svc, "BTC/USDT", rng.normal(5, 1, (600, 3)))

    await svc._run_evaluation_cycle()

    (event,) = _data_drift_events(bus)
    assert event.drift_type == "data"  # type: ignore[attr-defined]
    assert event.model_id == "features:BTC/USDT"  # type: ignore[attr-defined]
    assert event.score > event.threshold  # type: ignore[attr-defined]
    retrainer.mark_drift.assert_called_once_with("model-a")


@pytest.mark.asyncio
async def test_stable_live_features_publish_nothing(tmp_path: Path) -> None:
    svc, bus, retrainer = _make_service(tmp_path, drift_detector=DataDriftDetector())

    rng = np.random.default_rng(7)
    _write_reference(
        tmp_path, ["f0", "f1", "f2"], {"BTC/USDT": rng.normal(0, 1, (1000, 3))}
    )
    svc._feature_names = ["f0", "f1", "f2"]
    _fill_buffer(svc, "BTC/USDT", rng.normal(0, 1, (600, 3)))

    await svc._run_evaluation_cycle()

    assert _data_drift_events(bus) == []
    retrainer.mark_drift.assert_not_called()


@pytest.mark.asyncio
async def test_mark_drift_once_per_cycle_across_symbols(tmp_path: Path) -> None:
    """Two drifting symbols -> two data-drift events (monitoring needs the
    per-symbol detail) but only ONE mark_drift per tracked model."""
    detector = MagicMock(spec=DataDriftDetector)
    detector.detect_feature_drift.return_value = DriftReport(
        is_drifting=True, drift_score=0.5, threshold=0.2
    )
    svc, bus, retrainer = _make_service(tmp_path, drift_detector=detector)
    svc._tracked_predictions["model-a"] = []

    _write_reference(
        tmp_path,
        ["a", "b"],
        {"BTC/USDT": np.zeros((1000, 2)), "AAPL": np.zeros((1000, 2))},
    )
    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", np.zeros((600, 2)))
    _fill_buffer(svc, "AAPL", np.zeros((600, 2)))

    await svc._run_evaluation_cycle()

    events = _data_drift_events(bus)
    assert {getattr(e, "model_id", "") for e in events} == {
        "features:BTC/USDT",
        "features:AAPL",
    }
    retrainer.mark_drift.assert_called_once_with("model-a")


@pytest.mark.asyncio
async def test_rewritten_reference_reloaded_by_mtime(tmp_path: Path) -> None:
    """A retrain that rewrites the .npz must be picked up on the next cycle."""
    svc, bus, _retrainer = _make_service(tmp_path, drift_detector=DataDriftDetector())

    rng = np.random.default_rng(7)
    live = rng.normal(5, 1, (600, 2))
    svc._feature_names = ["a", "b"]
    _fill_buffer(svc, "BTC/USDT", live)

    # First reference matches the live distribution -> no drift.
    _write_reference(tmp_path, ["a", "b"], {"BTC/USDT": rng.normal(5, 1, (1000, 2))})
    await svc._run_evaluation_cycle()
    assert _data_drift_events(bus) == []

    # Rewritten reference far from live -> drift on the next cycle. Force a
    # distinct mtime; coarse filesystem timestamps could hide the rewrite.
    _write_reference(tmp_path, ["a", "b"], {"BTC/USDT": rng.normal(0, 1, (1000, 2))})
    ref_path = tmp_path / "feature_reference.npz"
    stat = ref_path.stat()
    import os

    os.utime(ref_path, (stat.st_atime, stat.st_mtime + 10))

    await svc._run_evaluation_cycle()
    assert len(_data_drift_events(bus)) == 1
