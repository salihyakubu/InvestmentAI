"""The live-transfer gate is wired INTO promotion, not beside it.

Era 4's lesson: a challenger that passes validation can still be worse live.
Pinned here: the gate's refusal blocks register+promote even when validation
passes, its decision rides the retrain result either way, and a pass
proceeds to promotion exactly once.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import services.continuous_learning.live_replay as live_replay
from services.continuous_learning.live_replay import GateDecision
from services.continuous_learning.retrainer import AutoRetrainer


class _TrainResult:
    def to_metrics(self) -> dict[str, float]:
        return {"val_accuracy": 0.9}


def _retrainer() -> AutoRetrainer:
    trainer = MagicMock()
    trainer.train_model.return_value = (_TrainResult(), object())
    registry = MagicMock()
    champion_meta = MagicMock()
    champion_meta.metrics = {"val_accuracy": 0.5}
    registry.get_active.return_value = (object(), champion_meta)
    registry.register.return_value = ("xgboost-v2", 2)
    retrainer = AutoRetrainer(
        trainer=trainer,
        registry=registry,
        settings=MagicMock(),
        session_factory=MagicMock(),
    )
    return retrainer


def _data() -> dict:
    X = np.zeros((4, 2))
    y = np.zeros(4)
    return {
        "X_train": X, "y_train": y, "returns_train": None,
        "X_val": X, "y_val": y, "returns_val": None,
        "feature_names": ["a", "b"],
    }


@pytest.mark.asyncio
async def test_gate_refusal_blocks_a_validation_passing_challenger(monkeypatch) -> None:
    refusal = GateDecision(
        promote=False, reason="challenger_does_not_beat_champion_on_live_rows",
        challenger_ic=-0.01, champion_ic=0.02, n_rows=5000, span_days=20.0,
    )

    async def _fake_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return refusal

    monkeypatch.setattr(live_replay, "live_transfer_gate", _fake_gate)
    retrainer = _retrainer()
    result = await retrainer._retrain_member("ensemble:xgboost", "xgboost", **_data())

    assert result["skipped"] is True
    assert result["reason"] == refusal.reason
    assert result["live_replay"]["champion_ic"] == 0.02
    retrainer._registry.register.assert_not_called()
    retrainer._registry.promote.assert_not_called()


@pytest.mark.asyncio
async def test_gate_error_fails_closed(monkeypatch) -> None:
    """A gate that cannot judge must refuse, not wave the challenger
    through: blind promotion is the measured failure mode."""

    async def _broken_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(live_replay, "live_transfer_gate", _broken_gate)
    retrainer = _retrainer()
    result = await retrainer._retrain_member("ensemble:xgboost", "xgboost", **_data())

    assert result["skipped"] is True
    assert result["reason"] == "live_transfer_gate_error"
    retrainer._registry.promote.assert_not_called()


@pytest.mark.asyncio
async def test_gate_pass_promotes_and_records_the_decision(monkeypatch) -> None:
    verdict = GateDecision(
        promote=True, reason="challenger_beats_champion_on_live_rows",
        challenger_ic=0.04, champion_ic=0.01, n_rows=5000, span_days=20.0,
    )

    async def _fake_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return verdict

    monkeypatch.setattr(live_replay, "live_transfer_gate", _fake_gate)
    retrainer = _retrainer()

    async def _noop(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return None

    monkeypatch.setattr(retrainer, "_mirror_metadata_to_db", _noop)
    monkeypatch.setattr(retrainer, "_persist_artifacts_to_db", _noop)
    result = await retrainer._retrain_member("ensemble:xgboost", "xgboost", **_data())

    assert "skipped" not in result
    assert result["live_replay"]["promote"] is True
    retrainer._registry.promote.assert_called_once()
