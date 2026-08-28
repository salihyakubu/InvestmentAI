"""The live-transfer gate is wired INTO promotion, not beside it.

Era 4's lesson: a challenger that passes validation can still be worse
live. Pinned here: the gate's refusal blocks register+promote even when
validation passes, a data-side refusal skips the expensive training run
entirely, gate errors fail closed, the decision rides the retrain result
either way, and a pass proceeds to promotion exactly once.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import services.continuous_learning.live_replay as live_replay
from services.continuous_learning.live_replay import GateDecision, ReplayData
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
    return AutoRetrainer(
        trainer=trainer,
        registry=registry,
        settings=MagicMock(),
        session_factory=MagicMock(),
    )


def _replay_data() -> ReplayData:
    return ReplayData(X=np.zeros((4, 2)), y=np.array([0.1, -0.2, 0.3, 0.0]), span_days=15.0)


def _member_kwargs() -> dict:
    X = np.zeros((4, 2))
    y = np.zeros(4)
    return {
        "X_train": X, "y_train": y, "returns_train": None,
        "X_val": X, "y_val": y, "returns_val": None,
        "feature_names": ["a", "b"],
        "replay_data": _replay_data(),
    }


@pytest.mark.asyncio
async def test_gate_refusal_blocks_a_validation_passing_challenger(monkeypatch) -> None:
    refusal = GateDecision(
        promote=False, reason="challenger_does_not_beat_champion_on_live_rows",
        challenger_ic=-0.01, champion_ic=0.02, n_rows=5000, span_days=15.0,
    )

    async def _fake_decide(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return refusal

    monkeypatch.setattr(live_replay, "decide", _fake_decide)
    retrainer = _retrainer()
    result = await retrainer._retrain_member(
        "ensemble:xgboost", "xgboost", **_member_kwargs()
    )

    assert result["skipped"] is True
    assert result["reason"] == refusal.reason
    assert result["live_replay"]["champion_ic"] == 0.02
    retrainer._registry.register.assert_not_called()
    retrainer._registry.promote.assert_not_called()


@pytest.mark.asyncio
async def test_gate_error_fails_closed(monkeypatch) -> None:
    """A gate that cannot judge must refuse, not wave the challenger
    through: blind promotion is the measured failure mode."""

    async def _broken_decide(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("scorer blew up")

    monkeypatch.setattr(live_replay, "decide", _broken_decide)
    retrainer = _retrainer()
    result = await retrainer._retrain_member(
        "ensemble:xgboost", "xgboost", **_member_kwargs()
    )

    assert result["skipped"] is True
    assert result["reason"] == "live_transfer_gate_error"
    retrainer._registry.promote.assert_not_called()


@pytest.mark.asyncio
async def test_gate_pass_promotes_and_records_the_decision(monkeypatch) -> None:
    verdict = GateDecision(
        promote=True, reason="challenger_beats_champion_on_live_rows",
        challenger_ic=0.04, champion_ic=0.01, n_rows=5000, span_days=15.0,
    )

    async def _fake_decide(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return verdict

    monkeypatch.setattr(live_replay, "decide", _fake_decide)
    retrainer = _retrainer()

    async def _noop(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return None

    monkeypatch.setattr(retrainer, "_mirror_metadata_to_db", _noop)
    monkeypatch.setattr(retrainer, "_persist_artifacts_to_db", _noop)
    result = await retrainer._retrain_member(
        "ensemble:xgboost", "xgboost", **_member_kwargs()
    )

    assert "skipped" not in result
    assert result["live_replay"]["promote"] is True
    retrainer._registry.promote.assert_called_once()


@pytest.mark.asyncio
async def test_data_side_refusal_skips_training_entirely(monkeypatch) -> None:
    """A deterministic refusal (floors, hash, boundary) is known BEFORE any
    model exists; paying a full hyperopt run to feed a gate that must refuse
    was a review finding. The trainer must never be invoked."""
    refusal = GateDecision(
        promote=False, reason="insufficient_live_rows",
        challenger_ic=None, champion_ic=None, n_rows=120, span_days=1.0,
    )

    async def _fake_prepare(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return refusal

    monkeypatch.setattr(live_replay, "prepare_replay", _fake_prepare)
    retrainer = _retrainer()

    X = np.zeros((30, 2))
    y = np.zeros(30)

    async def _fake_load(model_id, end=None):  # noqa: ANN001, ANN202, ARG001
        return X[:24], y[:24], None, X[24:], y[24:], None, ["a", "b"]

    monkeypatch.setattr(retrainer, "_load_training_data", _fake_load)
    result = await retrainer.retrain("ensemble:xgboost")

    assert result["skipped"] is True
    assert result["reason"] == "insufficient_live_rows"
    assert result["members"][0]["live_replay"]["n_rows"] == 120
    retrainer._trainer.train_model.assert_not_called()
    retrainer._registry.promote.assert_not_called()
