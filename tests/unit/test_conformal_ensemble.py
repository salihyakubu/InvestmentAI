"""Conformal gating in EnsemblePredictor, layered ON TOP of the agreement vote.

A non-flat winner must be conformally supportable by every agreeing member:
each agreeing member's conformal set must be {winner} or {winner, "flat"}.
Members without conformal state (old artifacts, torch models) are vote-only
and never block, so mixed ensembles keep working.

Stub predictors are used so these tests need neither xgboost nor lightgbm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult
from services.prediction.models.ensemble import EnsemblePredictor

LONG_PROBS = {"long": 0.7, "flat": 0.2, "short": 0.1}
FLAT_PROBS = {"long": 0.4, "flat": 0.5, "short": 0.1}


class _FixedPredictor(BasePredictor):
    """Returns a canned output; model_type set per instance."""

    def __init__(self, model_type: str, output: PredictionOutput) -> None:
        self.model_type = model_type
        self._output = output

    def predict(self, features: np.ndarray) -> PredictionOutput:
        return self._output

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        return [self._output]

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:  # noqa: ANN001
        raise NotImplementedError

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def get_feature_importance(self) -> dict[str, float]:
        return {}


def _out(
    direction: str,
    probs: dict[str, float],
    conformal_set: list[str] | None = None,
) -> PredictionOutput:
    metadata = (
        {"conformal_set": sorted(conformal_set), "conformal_alpha": 0.10}
        if conformal_set is not None
        else {}
    )
    return PredictionOutput(
        direction=direction,
        confidence=probs[direction],
        expected_return=0.01 if direction == "long" else 0.0,
        probabilities=probs,
        metadata=metadata,
    )


def _predict(*outputs: PredictionOutput, min_agreement: int = 1) -> PredictionOutput:
    # "catboost" deliberately included: any non-sequence model_type must work
    # (the next-phase CatBoost predictor rides the same contract).
    types = ["xgboost", "lightgbm", "catboost"]
    models: list[BasePredictor] = [
        _FixedPredictor(t, o) for t, o in zip(types, outputs)
    ]
    ensemble = EnsemblePredictor(models=models, min_agreement=min_agreement)
    return ensemble.predict(features_flat=np.zeros(3))


def test_vote_passes_when_every_agreeing_member_supports_it() -> None:
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long"]),
        _out("long", LONG_PROBS, conformal_set=["flat", "long"]),
        min_agreement=2,
    )
    assert result.direction == "long"
    assert result.metadata["conformal_gated"] is False


def test_gate_degrades_to_flat_when_member_set_excludes_winner() -> None:
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long"]),
        _out("long", LONG_PROBS, conformal_set=["flat", "short"]),  # excludes long
        min_agreement=2,
    )
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is True
    # Confidence degrades to the combined flat probability; the combined
    # probability distribution itself is untouched.
    assert result.confidence == result.probabilities["flat"]
    assert result.probabilities["long"] > result.probabilities["flat"]


def test_gate_degrades_on_ambiguous_set_containing_other_direction() -> None:
    # The set contains the winner but also the OTHER non-flat class: not
    # singleton-or-{winner, flat}, so the member cannot support the signal.
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long", "short"]),
        _out("long", LONG_PROBS, conformal_set=["long"]),
        min_agreement=2,
    )
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is True


def test_empty_conformal_set_blocks() -> None:
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long"]),
        _out("long", LONG_PROBS, conformal_set=[]),  # abstains entirely
        min_agreement=2,
    )
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is True


def test_members_without_conformal_state_are_non_blocking() -> None:
    # Old artifacts / torch models publish no conformal_set at all.
    result = _predict(
        _out("long", LONG_PROBS),
        _out("long", LONG_PROBS),
        min_agreement=2,
    )
    assert result.direction == "long"
    assert result.metadata["conformal_gated"] is False


def test_mixed_ensemble_gates_only_on_members_with_state() -> None:
    result = _predict(
        _out("long", LONG_PROBS),  # no conformal state: vote-only
        _out("long", LONG_PROBS, conformal_set=["flat"]),  # blocks
        min_agreement=2,
    )
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is True


def test_non_agreeing_member_never_blocks() -> None:
    # The flat-voting member's hostile set is irrelevant: only members that
    # agreed with the winning direction are consulted.
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long"]),
        _out("flat", FLAT_PROBS, conformal_set=["short"]),
        min_agreement=1,
    )
    assert result.direction == "long"  # combined long prob (0.55) still wins
    assert result.metadata["conformal_gated"] is False


def test_min_agreement_vote_still_enforced_before_gate() -> None:
    result = _predict(
        _out("long", LONG_PROBS, conformal_set=["long"]),
        _out("flat", FLAT_PROBS, conformal_set=["flat"]),
        min_agreement=2,
    )
    # Vote fails (only 1 of 2 agree): flat via the ORIGINAL agreement check,
    # not the conformal gate.
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is False


def test_flat_vote_is_never_gated() -> None:
    result = _predict(
        _out("flat", FLAT_PROBS, conformal_set=["short"]),
        _out("flat", FLAT_PROBS, conformal_set=["long"]),
        min_agreement=2,
    )
    assert result.direction == "flat"
    assert result.metadata["conformal_gated"] is False
