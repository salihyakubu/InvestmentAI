"""Markowitz mean-variance portfolio optimizer."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from services.portfolio.optimizers.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer(BaseOptimizer):
    """Classic Markowitz mean-variance optimizer using SLSQP.

    Maximises the Sharpe ratio by default, or minimises variance when a
    target return is supplied via constraints.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        max_position_pct: float = 0.10,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_position_pct = max_position_pct

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: dict | None = None,
    ) -> OptimizationResult:
        n = len(symbols)
        constraints = constraints or {}

        max_weight = constraints.get("max_weight", self.max_position_pct)
        min_weight = constraints.get("min_weight", 0.0)
        target_return = constraints.get("target_return")

        bounds = tuple((min_weight, max_weight) for _ in range(n))
        init_weights = np.full(n, 1.0 / n)

        # Weights must sum to 1.
        eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        scipy_constraints: list[dict] = [eq_constraint]

        if target_return is not None:
            # Minimise variance for a given return target.
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: w @ expected_returns - target_return,
                }
            )
            objective = self._portfolio_variance
        else:
            # Maximise Sharpe ratio (minimise negative Sharpe).
            objective = self._neg_sharpe

        result = minimize(
            objective,
            init_weights,
            args=(expected_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("Optimisation did not converge: %s", result.message)

        weights = result.x
        # Clip tiny negatives from numerical noise and re-normalise.
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        port_return = float(weights @ expected_returns)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = (
            (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        )

        return OptimizationResult(
            weights={s: float(w) for s, w in zip(symbols, weights)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="mean_variance",
        )

    # ------------------------------------------------------------------
    # Objective helpers
    # ------------------------------------------------------------------

    def _neg_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        port_return = weights @ expected_returns
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        if port_vol < 1e-12:
            return 0.0
        return -float((port_return - self.risk_free_rate) / port_vol)

    @staticmethod
    def _portfolio_variance(
        weights: np.ndarray,
        expected_returns: np.ndarray,  # noqa: ARG004
        cov_matrix: np.ndarray,
    ) -> float:
        return float(weights @ cov_matrix @ weights)
