"""Skill metrics that an imbalanced-label classifier cannot fake.

Measured on production 2026-09-12: nine model versions climbed
val_accuracy 0.505 -> 0.594 while serving 373,267 flat predictions against
7 long and 0 short. 0.594 was exactly the base rate of the majority class.
Raw accuracy on imbalanced labels is not a skill measure -- a constant
predictor maximises it -- so it must never again be the number a promotion
turns on.

* :func:`majority_class_rate` is what a constant predictor scores. Recorded
  beside val_accuracy, it turns "59.4% accurate" into "59.4% accurate
  against a 59.4% base rate", which reads as what it is.
* :func:`balanced_accuracy` (macro-averaged recall) is the honest measure:
  a constant predictor scores 1/n_classes no matter how skewed the labels,
  so only a model that actually separates classes can beat chance.

Note on what these revealed: with balanced accuracy in hand, the platform's
own classifier scores ~1/3 -- chance -- on production-shaped data whether
or not the loss is class-weighted. The collapse to "flat" is therefore not
a weighting bug; it is the model correctly reporting that it has no signal
to express (GO_LIVE 2026-09-12 retraction).
"""

from __future__ import annotations

import numpy as np

__all__ = ["balanced_accuracy", "chance_level", "majority_class_rate"]


def majority_class_rate(y: np.ndarray) -> float | None:
    """Accuracy a constant predictor would achieve on *y*.

    ``None`` for an empty input -- the honest answer when there is no
    distribution to describe.
    """
    labels = np.asarray(y)
    if labels.size == 0:
        return None
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / labels.size)


def chance_level(y: np.ndarray) -> float | None:
    """Balanced accuracy of a constant (or random) predictor: 1/n_classes."""
    labels = np.asarray(y)
    if labels.size == 0:
        return None
    return 1.0 / float(np.unique(labels).size)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Macro-averaged recall: mean per-class recall over the classes present.

    Immune to label imbalance by construction -- always guessing the
    majority class scores 1/n_classes, not the majority's share. ``None``
    for an empty input.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.size == 0:
        return None
    recalls = []
    for cls in np.unique(truth):
        mask = truth == cls
        recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls))
