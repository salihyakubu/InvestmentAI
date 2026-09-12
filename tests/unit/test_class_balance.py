"""The classifier must not collapse to the majority class.

Production, 2026-09-12: nine model versions climbed val_accuracy 0.505 ->
0.594 while serving 373,267 flat predictions against 7 long and 0 short.
0.594 was exactly the flat base rate -- the ensemble had learned to say
"nothing will happen" and never changed its mind, because an unweighted
multiclass loss over flat-heavy labels is minimised by a constant
predictor. These tests pin the fix and, more importantly, pin the failure
so it can never return silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.prediction.training.class_balance import (
    balanced_sample_weights,
    majority_class_rate,
)

# ---------------------------------------------------------------------------
# The weighting itself
# ---------------------------------------------------------------------------


def test_every_class_ends_up_with_equal_total_weight() -> None:
    """That is the whole definition of balanced: the loss must not care
    which class is more common."""
    y = np.array([1] * 594 + [0] * 203 + [2] * 203)
    w = balanced_sample_weights(y)
    totals = [w[y == c].sum() for c in (0, 1, 2)]
    assert max(totals) - min(totals) < 1e-9


def test_effective_sample_size_is_preserved() -> None:
    """Mean weight 1.0, so regularisation strengths and early-stopping
    thresholds tuned on unweighted data keep their meaning."""
    y = np.array([1] * 900 + [0] * 50 + [2] * 50)
    assert balanced_sample_weights(y).mean() == pytest.approx(1.0)


def test_rarer_classes_are_weighted_up() -> None:
    y = np.array([1] * 900 + [0] * 100)
    w = balanced_sample_weights(y)
    assert w[y == 0][0] > w[y == 1][0]


def test_a_balanced_label_set_is_left_alone() -> None:
    y = np.array([0, 1, 2] * 100)
    assert np.allclose(balanced_sample_weights(y), 1.0)


def test_degenerate_inputs_do_not_invent_weights() -> None:
    assert balanced_sample_weights(np.array([])).size == 0
    single = np.array([1] * 50)
    assert np.allclose(balanced_sample_weights(single), 1.0)


# ---------------------------------------------------------------------------
# The honesty metric
# ---------------------------------------------------------------------------


def test_majority_class_rate_is_the_constant_predictors_score() -> None:
    y = np.array([1] * 594 + [0] * 203 + [2] * 203)
    # The exact number production reported as "val_accuracy" for v4.
    assert majority_class_rate(y) == pytest.approx(0.594)
    assert majority_class_rate(np.array([])) is None


# ---------------------------------------------------------------------------
# The behaviour that actually matters, on a real classifier
# ---------------------------------------------------------------------------


def _flat_heavy_but_learnable(seed: int = 0):
    """Direction IS predictable from feature 0, but 60% of labels are flat --
    the production shape. A competent learner must find the direction; a
    loss-minimising one is tempted to answer 'flat' forever."""
    rng = np.random.default_rng(seed)
    n = 3000
    x0 = rng.normal(0, 1, n)
    y = np.ones(n, dtype=int)  # flat
    y[x0 > 1.0] = 2  # long  ~16%
    y[x0 < -1.0] = 0  # short ~16%
    # Pad the flat class to ~60% with pure-noise rows.
    pad = rng.normal(0, 0.3, 1200)
    X = np.column_stack([np.concatenate([x0, pad]), rng.normal(0, 1, n + 1200)])
    y = np.concatenate([y, np.ones(1200, dtype=int)])
    return X, y


def test_unweighted_training_collapses_and_weighted_training_does_not() -> None:
    """The regression that matters. Without balancing, the classifier is
    permitted to become a constant function; with it, it must express every
    class it was shown."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _flat_heavy_but_learnable()
    base = majority_class_rate(y)
    assert base > 0.55  # the fixture really is flat-heavy

    unweighted = HistGradientBoostingClassifier(
        max_iter=40, max_depth=2, learning_rate=0.05, random_state=0
    ).fit(X, y)
    weighted = HistGradientBoostingClassifier(
        max_iter=40, max_depth=2, learning_rate=0.05, random_state=0
    ).fit(X, y, sample_weight=balanced_sample_weights(y))

    unweighted_directional = float(np.mean(unweighted.predict(X) != 1))
    weighted_directional = float(np.mean(weighted.predict(X) != 1))

    # The fix must materially increase how often the model commits.
    assert weighted_directional > unweighted_directional
    # And it must express BOTH directions, not trade one collapse for another.
    predicted = set(np.unique(weighted.predict(X)).tolist())
    assert {0, 2}.issubset(predicted)


def test_accuracy_over_base_rate_is_what_the_registry_records() -> None:
    """A model scoring its own base rate has learned nothing, and the
    recorded metric must say so rather than reading as 59% skill."""
    from services.prediction.models.base import TrainResult

    collapsed = TrainResult(
        train_loss=0.87, val_loss=0.94, train_accuracy=0.5931,
        val_accuracy=0.5939, epochs_trained=10, majority_class_rate=0.5939,
    )
    metrics = collapsed.to_metrics()
    assert metrics["majority_class_rate"] == 0.5939
    assert metrics["accuracy_over_base_rate"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The gate must not reward the pathology
# ---------------------------------------------------------------------------


def test_a_collapsed_champion_cannot_block_a_learning_challenger() -> None:
    """The trap this whole change exists to escape: champion v4 scored
    0.5939 by always guessing the majority class, and a model that attempts
    direction necessarily scores LOWER raw accuracy. Comparing raw accuracy
    would let the constant predictor hold the throne forever."""
    from services.continuous_learning.retrainer import AutoRetrainer

    collapsed_champion = {"val_accuracy": 0.5939, "majority_class_rate": 0.5939,
                          "accuracy_over_base_rate": 0.0}
    learning_challenger = {"val_accuracy": 0.5600, "majority_class_rate": 0.5400,
                           "accuracy_over_base_rate": 0.0200}
    assert learning_challenger["val_accuracy"] < collapsed_champion["val_accuracy"]
    assert AutoRetrainer._validate_new_model(learning_challenger, collapsed_champion)


def test_a_challenger_with_no_skill_is_still_refused() -> None:
    from services.continuous_learning.retrainer import AutoRetrainer

    champion = {"val_accuracy": 0.55, "majority_class_rate": 0.52,
                "accuracy_over_base_rate": 0.03}
    no_better = {"val_accuracy": 0.62, "majority_class_rate": 0.61,
                 "accuracy_over_base_rate": 0.01}
    # Higher raw accuracy, LESS skill -> refused.
    assert not AutoRetrainer._validate_new_model(no_better, champion)


def test_a_legacy_champion_without_skill_metrics_defers_to_learning() -> None:
    """v4 records no base rate. Raw accuracy cannot arbitrate against it, so
    the challenger need only have learned something; the live-transfer gate
    (conjunctive, class-balance immune) casts the deciding vote."""
    from services.continuous_learning.retrainer import AutoRetrainer

    legacy = {"val_accuracy": 0.5939}
    learner = {"val_accuracy": 0.5600, "majority_class_rate": 0.5400,
               "accuracy_over_base_rate": 0.0200}
    assert AutoRetrainer._validate_new_model(learner, legacy)
    guesser = {"val_accuracy": 0.5939, "majority_class_rate": 0.5939,
               "accuracy_over_base_rate": 0.0}
    assert not AutoRetrainer._validate_new_model(guesser, legacy)


def test_the_absolute_floor_still_applies() -> None:
    from services.continuous_learning.retrainer import AutoRetrainer

    garbage = {"val_accuracy": 0.20, "majority_class_rate": 0.10,
               "accuracy_over_base_rate": 0.10}
    assert not AutoRetrainer._validate_new_model(garbage, {})
