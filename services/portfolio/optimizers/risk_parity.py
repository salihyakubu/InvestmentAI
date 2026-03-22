"""Risk-parity portfolio optimizer."""

from __future__ import annotations

import logging

import numpy as np

from services.portfolio.optimizers.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class RiskParityOptimizer(BaseOptimizer):
    """Risk-parity optimizer: each asset contributes equally to total
    portfolio risk.

    Uses an iterative Newton-type method to solve for weights such that
    marginal risk contributions are equalised.
    """

    def __init__(
        self,
        max_iterations: int = 500,
        tolerance: float = 1e-10,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: dict | None = None,
    ) -> OptimizationResult:
        n = len(symbols)
        weights = np.full(n, 1.0 / n)

        for iteration in range(self.max_iterations):
            mrc = self.marginal_risk_contribution(weights, cov_matrix)
            risk_contributions = weights * mrc
            total_risk = risk_contributions.sum()

            if total_risk < 1e-16:
                break

            target_rc = total_risk / n

            # Newton step: adjust weights proportional to deviation from
            # equal risk contribution.
            rc_deviation = risk_contributions - target_rc
            if np.max(np.abs(rc_deviation)) < self.tolerance:
                logger.debug(
                    "Risk parity converged after %d iterations.", iteration + 1
                )
                break

            # Update weights: reduce those over-contributing, increase
            # those under-contributing.
            adjustment = rc_deviation / (mrc + 1e-16)
            weights -= 0.5 * adjustment
            weights = np.maximum(weights, 1e-8)
            weights /= weights.sum()
        else:
            logger.warning(
                "Risk parity did not converge within %d iterations.",
                self.max_iterations,
            )

        weights /= weights.sum()

        port_return = float(weights @ expected_returns)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = port_return / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights={s: float(w) for s, w in zip(symbols, weights)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="risk_parity",
        )

    @staticmethod
    def marginal_risk_contribution(
        weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute marginal risk contribution for each asset.

        MRC_i = (Sigma @ w)_i / sqrt(w^T Sigma w)
        """
        portfolio_var = weights @ cov_matrix @ weights
        if portfolio_var < 1e-16:
            return np.zeros_like(weights)
        portfolio_vol = np.sqrt(portfolio_var)
        return (cov_matrix @ weights) / portfolio_vol
