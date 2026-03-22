"""Data and model drift detection using PSI and KS tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from scipy.stats import ks_2samp

logger = structlog.get_logger(__name__)

# PSI threshold: values above 0.2 indicate significant drift.
_PSI_THRESHOLD = 0.2


@dataclass
class DriftReport:
    """Result of a drift detection check."""

    is_drifting: bool
    drift_score: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


class DataDriftDetector:
    """Detects distributional shift in input features.

    Uses Population Stability Index (PSI) as the primary metric and a
    two-sample Kolmogorov-Smirnov test as a secondary confirmation.
    """

    def __init__(self, psi_threshold: float = _PSI_THRESHOLD, n_bins: int = 10) -> None:
        self._psi_threshold = psi_threshold
        self._n_bins = n_bins

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect_feature_drift(
        self,
        current_features: np.ndarray,
        historical_features: np.ndarray,
    ) -> DriftReport:
        """Compare current vs. historical feature distributions.

        Parameters:
            current_features: 2-D array ``(n_samples, n_features)``.
            historical_features: 2-D array ``(n_samples, n_features)``.

        Returns:
            A :class:`DriftReport` summarising whether drift is present.
        """
        if current_features.ndim == 1:
            current_features = current_features.reshape(-1, 1)
        if historical_features.ndim == 1:
            historical_features = historical_features.reshape(-1, 1)

        n_features = current_features.shape[1]
        psi_scores: list[float] = []
        ks_results: list[dict[str, Any]] = []
        drifted_features: list[int] = []

        for i in range(n_features):
            current_col = current_features[:, i]
            hist_col = historical_features[:, i]

            psi = self._compute_psi(hist_col, current_col)
            psi_scores.append(psi)

            ks_stat, ks_pvalue = ks_2samp(hist_col, current_col)
            ks_results.append({"statistic": float(ks_stat), "p_value": float(ks_pvalue)})

            if psi > self._psi_threshold:
                drifted_features.append(i)

        mean_psi = float(np.mean(psi_scores))
        is_drifting = mean_psi > self._psi_threshold or len(drifted_features) > n_features * 0.3

        report = DriftReport(
            is_drifting=is_drifting,
            drift_score=mean_psi,
            threshold=self._psi_threshold,
            details={
                "per_feature_psi": psi_scores,
                "ks_results": ks_results,
                "drifted_feature_indices": drifted_features,
                "drifted_feature_count": len(drifted_features),
                "total_features": n_features,
            },
        )

        logger.info(
            "drift.data.checked",
            is_drifting=is_drifting,
            mean_psi=mean_psi,
            drifted_count=len(drifted_features),
        )
        return report

    # ------------------------------------------------------------------
    # PSI calculation
    # ------------------------------------------------------------------

    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-6

        # Use percentile-based bins derived from the expected distribution.
        breakpoints = np.percentile(expected, np.linspace(0, 100, self._n_bins + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

        expected_pct = expected_counts / expected_counts.sum() + eps
        actual_pct = actual_counts / actual_counts.sum() + eps

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi


class ModelDriftDetector:
    """Detects accuracy degradation by comparing rolling prediction windows."""

    def __init__(self, accuracy_drop_threshold: float = 0.05) -> None:
        self._threshold = accuracy_drop_threshold

    def detect_accuracy_drift(
        self,
        recent_predictions: list[dict[str, Any]],
        older_predictions: list[dict[str, Any]],
    ) -> DriftReport:
        """Compare accuracy between recent and older prediction windows.

        Each prediction dict must contain ``predicted`` and ``actual`` keys.

        Parameters:
            recent_predictions: Most recent prediction outcomes.
            older_predictions: Earlier prediction outcomes for baseline.

        Returns:
            :class:`DriftReport` indicating whether model accuracy has degraded.
        """
        if not recent_predictions or not older_predictions:
            return DriftReport(
                is_drifting=False,
                drift_score=0.0,
                threshold=self._threshold,
                details={"reason": "insufficient_data"},
            )

        recent_accuracy = self._compute_accuracy(recent_predictions)
        older_accuracy = self._compute_accuracy(older_predictions)
        accuracy_drop = older_accuracy - recent_accuracy

        is_drifting = accuracy_drop > self._threshold

        report = DriftReport(
            is_drifting=is_drifting,
            drift_score=accuracy_drop,
            threshold=self._threshold,
            details={
                "recent_accuracy": recent_accuracy,
                "older_accuracy": older_accuracy,
                "accuracy_drop": accuracy_drop,
                "recent_sample_size": len(recent_predictions),
                "older_sample_size": len(older_predictions),
            },
        )

        logger.info(
            "drift.model.checked",
            is_drifting=is_drifting,
            accuracy_drop=accuracy_drop,
            recent_acc=recent_accuracy,
            older_acc=older_accuracy,
        )
        return report

    @staticmethod
    def _compute_accuracy(predictions: list[dict[str, Any]]) -> float:
        """Compute fraction of correct predictions."""
        if not predictions:
            return 0.0
        correct = sum(1 for p in predictions if p.get("predicted") == p.get("actual"))
        return correct / len(predictions)
