"""Trading feedback loop -- connects realised P&L back to prediction quality."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TradingFeedbackLoop:
    """Tracks prediction outcomes and computes model-level performance metrics.

    This closes the loop between prediction quality and actual trading
    results so that ensemble weights can be adjusted dynamically.
    """

    def __init__(self) -> None:
        # prediction_id -> outcome record
        self._outcomes: dict[str, dict[str, Any]] = {}
        # model_id -> list of prediction_ids
        self._model_predictions: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        prediction_id: str,
        actual_return: float,
        trade_pnl: float,
    ) -> None:
        """Record a realised outcome for a prediction.

        Parameters:
            prediction_id: Identifier matching a previously issued prediction.
            actual_return: Realised price return for the predicted asset.
            trade_pnl: Actual P&L from the trade (after slippage / commissions).
        """
        self._outcomes[prediction_id] = {
            "actual_return": actual_return,
            "trade_pnl": trade_pnl,
            "recorded_at": datetime.now(timezone.utc),
        }
        logger.debug(
            "feedback.outcome_recorded",
            prediction_id=prediction_id,
            actual_return=actual_return,
            trade_pnl=trade_pnl,
        )

    def register_prediction(self, prediction_id: str, model_id: str) -> None:
        """Associate a prediction_id with its originating model."""
        self._model_predictions[model_id].append(prediction_id)

    # ------------------------------------------------------------------
    # Model metrics
    # ------------------------------------------------------------------

    def compute_model_metrics(
        self,
        model_id: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Compute aggregate performance metrics for a model.

        Returns:
            Dictionary with total_pnl, avg_return, win_rate, sharpe estimate,
            and sample_size.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        pred_ids = self._model_predictions.get(model_id, [])

        pnl_values: list[float] = []
        returns: list[float] = []

        for pid in pred_ids:
            outcome = self._outcomes.get(pid)
            if outcome is None:
                continue
            if outcome["recorded_at"] < cutoff:
                continue
            pnl_values.append(outcome["trade_pnl"])
            returns.append(outcome["actual_return"])

        if not pnl_values:
            return {
                "model_id": model_id,
                "total_pnl": 0.0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "sample_size": 0,
            }

        arr_pnl = np.array(pnl_values)
        arr_ret = np.array(returns)
        wins = int(np.sum(arr_pnl > 0))
        sharpe = float(arr_ret.mean() / arr_ret.std()) if arr_ret.std() > 0 else 0.0

        metrics = {
            "model_id": model_id,
            "total_pnl": float(arr_pnl.sum()),
            "avg_return": float(arr_ret.mean()),
            "win_rate": wins / len(pnl_values),
            "sharpe": sharpe,
            "sample_size": len(pnl_values),
        }

        logger.info("feedback.model_metrics", **metrics)
        return metrics

    # ------------------------------------------------------------------
    # Ensemble weight adjustment
    # ------------------------------------------------------------------

    def update_ensemble_weights(
        self,
        model_metrics: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Compute new ensemble weights proportional to model performance.

        Parameters:
            model_metrics: Mapping of ``model_id -> metrics dict`` where each
                metrics dict contains at least ``sharpe`` and ``win_rate``.

        Returns:
            Mapping of ``model_id -> weight`` (sums to 1.0).
        """
        scores: dict[str, float] = {}

        for model_id, metrics in model_metrics.items():
            sharpe = metrics.get("sharpe", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            # Combined score: weighted sum of Sharpe ratio and win rate
            score = max(0.0, 0.6 * sharpe + 0.4 * win_rate)
            scores[model_id] = score

        total = sum(scores.values())
        if total <= 0:
            # Equal weights as fallback
            n = len(model_metrics)
            return {m: 1.0 / n for m in model_metrics} if n > 0 else {}

        weights = {m: s / total for m, s in scores.items()}

        logger.info(
            "feedback.ensemble_weights_updated",
            weights=weights,
        )
        return weights
