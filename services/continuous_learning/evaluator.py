"""Live model evaluation -- compares predictions against actual outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceReport:
    """Summary of a model's live prediction performance."""

    model_id: str
    lookback_days: int
    sample_size: int
    accuracy: float
    precision_per_class: dict[str, float] = field(default_factory=dict)
    recall_per_class: dict[str, float] = field(default_factory=dict)
    f1_per_class: dict[str, float] = field(default_factory=dict)
    directional_accuracy: float = 0.0
    confidence_calibration: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class ModelEvaluator:
    """Evaluates model performance against realised market outcomes.

    Prediction records are accumulated via :meth:`record_prediction`
    and :meth:`record_outcome`.  Evaluation is triggered on demand by
    :meth:`evaluate_live_performance`.
    """

    def __init__(self) -> None:
        # model_id -> list of {prediction_id, symbol, predicted, confidence, timestamp}
        self._predictions: dict[str, list[dict[str, Any]]] = {}
        # prediction_id -> {actual, actual_return}
        self._outcomes: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        prediction_id: str,
        model_id: str,
        symbol: str,
        predicted: str,
        confidence: float,
    ) -> None:
        """Store a prediction for later evaluation."""
        self._predictions.setdefault(model_id, []).append(
            {
                "prediction_id": prediction_id,
                "symbol": symbol,
                "predicted": predicted,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def record_outcome(
        self,
        prediction_id: str,
        actual: str,
        actual_return: float = 0.0,
    ) -> None:
        """Record the realised outcome for a prediction."""
        self._outcomes[prediction_id] = {
            "actual": actual,
            "actual_return": actual_return,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_live_performance(
        self,
        model_id: str,
        lookback_days: int = 30,
    ) -> PerformanceReport:
        """Evaluate a model's performance over the lookback window.

        Metrics computed:
        - Accuracy (overall fraction correct).
        - Precision / recall / F1 per class.
        - Directional accuracy (predicted up/down matches actual).
        - Confidence calibration (actual accuracy per confidence bucket).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        preds = self._predictions.get(model_id, [])
        recent = [p for p in preds if p["timestamp"] >= cutoff]

        # Match predictions with outcomes
        matched: list[dict[str, Any]] = []
        for p in recent:
            outcome = self._outcomes.get(p["prediction_id"])
            if outcome is not None:
                matched.append({**p, **outcome})

        if not matched:
            return PerformanceReport(
                model_id=model_id,
                lookback_days=lookback_days,
                sample_size=0,
                accuracy=0.0,
            )

        # Overall accuracy
        correct = sum(1 for m in matched if m["predicted"] == m["actual"])
        accuracy = correct / len(matched)

        # Per-class metrics
        classes = set(m["predicted"] for m in matched) | set(m["actual"] for m in matched)
        precision_per_class: dict[str, float] = {}
        recall_per_class: dict[str, float] = {}
        f1_per_class: dict[str, float] = {}

        for cls in classes:
            tp = sum(1 for m in matched if m["predicted"] == cls and m["actual"] == cls)
            fp = sum(1 for m in matched if m["predicted"] == cls and m["actual"] != cls)
            fn = sum(1 for m in matched if m["predicted"] != cls and m["actual"] == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precision_per_class[cls] = round(precision, 4)
            recall_per_class[cls] = round(recall, 4)
            f1_per_class[cls] = round(f1, 4)

        # Directional accuracy
        directional_matches = sum(
            1
            for m in matched
            if (m["predicted"] in ("long", "up") and m.get("actual_return", 0) > 0)
            or (m["predicted"] in ("short", "down") and m.get("actual_return", 0) < 0)
            or (m["predicted"] in ("flat", "neutral") and abs(m.get("actual_return", 0)) < 0.001)
        )
        directional_accuracy = directional_matches / len(matched) if matched else 0.0

        # Confidence calibration (bucket predictions by confidence decile)
        calibration = self._compute_calibration(matched)

        report = PerformanceReport(
            model_id=model_id,
            lookback_days=lookback_days,
            sample_size=len(matched),
            accuracy=round(accuracy, 4),
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_per_class=f1_per_class,
            directional_accuracy=round(directional_accuracy, 4),
            confidence_calibration=calibration,
        )

        logger.info(
            "evaluator.performance",
            model_id=model_id,
            accuracy=accuracy,
            directional_accuracy=directional_accuracy,
            sample_size=len(matched),
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration(matched: list[dict[str, Any]]) -> dict[str, float]:
        """Bin predictions by confidence and compute actual accuracy per bin."""
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration: dict[str, float] = {}

        for low, high in bins:
            bucket = [m for m in matched if low <= m.get("confidence", 0) < high]
            if bucket:
                acc = sum(1 for m in bucket if m["predicted"] == m["actual"]) / len(bucket)
                calibration[f"{low:.1f}-{high:.1f}"] = round(acc, 4)
            else:
                calibration[f"{low:.1f}-{high:.1f}"] = 0.0

        return calibration
