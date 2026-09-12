"""Accuracy on imbalanced labels is not skill, and the gate must know it.

Production, 2026-09-12: nine model versions climbed val_accuracy 0.505 ->
0.594 while serving 373,267 flat predictions against 7 long and 0 short.
0.594 was exactly the majority-class base rate. Raw accuracy rewarded the
collapse; these metrics cannot be fooled by it, and the promotion gate now
turns on the one that can't.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.prediction.training.skill_metrics import (
    balanced_accuracy,
    chance_level,
    majority_class_rate,
)


def test_a_constant_predictor_scores_chance_however_skewed_the_labels() -> None:
    """The property the whole change rests on: raw accuracy rewards
    collapse, balanced accuracy refuses to."""
    y = np.array([1] * 594 + [0] * 203 + [2] * 203)
    always_flat = np.ones_like(y)
    assert majority_class_rate(y) == pytest.approx(0.594)  # what v4 scored
    assert balanced_accuracy(y, always_flat) == pytest.approx(1 / 3)
    assert chance_level(y) == pytest.approx(1 / 3)


def test_it_stays_chance_even_at_extreme_imbalance() -> None:
    y = np.array([1] * 9900 + [0] * 50 + [2] * 50)
    assert majority_class_rate(y) == pytest.approx(0.99)
    assert balanced_accuracy(y, np.ones_like(y)) == pytest.approx(1 / 3)


def test_a_perfect_classifier_scores_one() -> None:
    y = np.array([0, 1, 2] * 50)
    assert balanced_accuracy(y, y.copy()) == pytest.approx(1.0)


def test_partial_skill_lands_between_chance_and_one() -> None:
    y = np.array([1] * 600 + [0] * 200 + [2] * 200)
    pred = y.copy()
    pred[800:900] = 1  # miss half the longs
    score = balanced_accuracy(y, pred)
    assert 1 / 3 < score < 1.0


def test_empty_inputs_return_none_not_a_number() -> None:
    empty = np.array([])
    assert majority_class_rate(empty) is None
    assert balanced_accuracy(empty, empty) is None
    assert chance_level(empty) is None


# ---------------------------------------------------------------------------
# The gate, driven by metrics a real training run produces
# ---------------------------------------------------------------------------


def test_the_collapsed_champion_is_refused_by_its_own_recorded_skill() -> None:
    """v4's shape: 0.594 raw accuracy, chance balanced accuracy. Under the
    old gate this was a champion; it must now fail to qualify at all."""
    from services.continuous_learning.retrainer import AutoRetrainer

    collapsed = {"val_accuracy": 0.5939, "majority_class_rate": 0.5939,
                 "balanced_accuracy": 1 / 3}
    assert not AutoRetrainer._validate_new_model(collapsed, {})


def test_a_below_chance_challenger_is_refused_however_high_its_accuracy() -> None:
    from services.continuous_learning.retrainer import AutoRetrainer

    flattering = {"val_accuracy": 0.85, "majority_class_rate": 0.86,
                  "balanced_accuracy": 0.30}
    assert not AutoRetrainer._validate_new_model(flattering, {})


def test_real_skill_beats_a_legacy_champion_with_lower_raw_accuracy() -> None:
    """The trap the fix escapes: a model that attempts direction scores
    LOWER raw accuracy than one that always guesses the majority."""
    from services.continuous_learning.retrainer import AutoRetrainer

    legacy_collapsed_champion = {"val_accuracy": 0.5939}  # no skill recorded
    learner = {"val_accuracy": 0.5100, "majority_class_rate": 0.5939,
               "balanced_accuracy": 0.42}
    assert learner["val_accuracy"] < legacy_collapsed_champion["val_accuracy"]
    assert AutoRetrainer._validate_new_model(learner, legacy_collapsed_champion)


def test_skill_must_not_regress_against_a_skill_reporting_champion() -> None:
    from services.continuous_learning.retrainer import AutoRetrainer

    champion = {"val_accuracy": 0.52, "balanced_accuracy": 0.44}
    worse = {"val_accuracy": 0.61, "balanced_accuracy": 0.40}
    better = {"val_accuracy": 0.49, "balanced_accuracy": 0.46}
    assert not AutoRetrainer._validate_new_model(worse, champion)
    assert AutoRetrainer._validate_new_model(better, champion)


def test_a_real_trained_model_records_served_pipeline_skill() -> None:
    """Not a hand-written metrics dict: train the actual predictor and
    assert the recorded skill describes the SERVED (calibrated) pipeline,
    which is the model production runs."""
    from services.prediction.models.xgboost_model import XGBoostPredictor

    rng = np.random.default_rng(0)
    n = 4000
    x0 = rng.normal(0, 1, n)
    y = np.ones(n, dtype=int)
    y[x0 > 0.8] = 2
    y[x0 < -0.8] = 0
    X = np.column_stack([x0 + rng.normal(0, 0.4, n)] + [rng.normal(0, 1, n) for _ in range(5)])
    s = int(n * 0.8)
    model = XGBoostPredictor(feature_names=[f"f{i}" for i in range(6)])
    result = model.train(X[:s], y[:s], X[s:], y[s:])

    assert result.majority_class_rate is not None
    assert result.balanced_accuracy is not None
    # A learnable signal must clear chance on the served pipeline.
    assert result.balanced_accuracy > 1 / 3
    metrics = result.to_metrics()
    assert "balanced_accuracy" in metrics
    assert "accuracy_over_base_rate" not in metrics  # the retracted metric
