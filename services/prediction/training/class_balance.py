"""Class balancing for the directional classifier.

Measured on production 2026-09-12: nine model versions climbed validation
accuracy 0.505 -> 0.594 while serving 373,267 flat predictions against 7
long and 0 short. 0.594 was not skill -- it was the base rate of the
majority (flat) class. An unweighted multiclass loss over flat-heavy
triple-barrier labels is minimised by a constant predictor, so the
ensemble learned to say "nothing will happen" and never changed its mind.

These helpers make that failure both impossible to repeat silently and
visible when it happens:

* :func:`balanced_sample_weights` weights each sample by the inverse of
  its class frequency, so every class contributes equally to the loss.
  Weights are normalised to mean 1.0, which preserves the effective sample
  size -- regularisation strengths and early-stopping thresholds tuned on
  unweighted data keep their meaning.
* :func:`majority_class_rate` is the accuracy a constant predictor would
  score. Reported beside val_accuracy in every TrainResult, it turns "59.4%
  accurate" into "59.4% accurate against a 59.4% base rate", which reads
  as what it is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["balanced_sample_weights", "majority_class_rate"]


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency.

    Normalised to mean 1.0. An empty or single-class input yields all-ones:
    there is nothing to balance, and inventing weights would silently
    rescale the loss.
    """
    labels = np.asarray(y)
    if labels.size == 0:
        return np.ones(0, dtype=float)

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        return np.ones(labels.size, dtype=float)

    # weight(class) proportional to 1 / count(class)
    weight_of = {cls: labels.size / (classes.size * cnt) for cls, cnt in zip(classes, counts, strict=True)}
    weights = np.array([weight_of[cls] for cls in labels], dtype=float)
    mean = weights.mean()
    return weights / mean if mean > 0 else np.ones(labels.size, dtype=float)


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
