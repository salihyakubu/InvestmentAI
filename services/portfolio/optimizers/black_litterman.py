"""Black-Litterman portfolio optimizer."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from services.portfolio.optimizers.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman model combining market equilibrium with investor views.

    Views typically originate from ML prediction events.  When no strong
    views are available the optimizer falls back to market-cap weights.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        max_position_pct: float = 0.10,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.max_position_pct = max_position_pct

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: dict | None = None,
    ) -> OptimizationResult:
        constraints = constraints or {}
        n = len(symbols)

        market_weights = constraints.get("market_weights")
        if market_weights is None:
            market_weights = np.full(n, 1.0 / n)
        else:
            market_weights = np.asarray(market_weights, dtype=float)

        views = constraints.get("views")  # P @ mu = Q style
        view_confidences = constraints.get("view_confidences")

        # 1. Equilibrium (implied) returns.
        equilibrium = self.compute_equilibrium_returns(
            cov_matrix, market_weights, self.risk_aversion
        )

        # 2. If views are available compute posterior, else use equilibrium.
        if views is not None and view_confidences is not None:
            P = np.asarray(views["P"], dtype=float)
            Q = np.asarray(views["Q"], dtype=float)
            confidences = np.asarray(view_confidences, dtype=float)
            posterior_returns = self.compute_posterior(
                equilibrium, cov_matrix, P, Q, confidences
            )
        else:
            logger.info(
                "No strong views provided; falling back to equilibrium weights."
            )
            posterior_returns = equilibrium

        # 3. Optimise using posterior returns (max Sharpe via SLSQP).
        max_weight = constraints.get("max_weight", self.max_position_pct)
        min_weight = constraints.get("min_weight", 0.0)
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        init_weights = np.full(n, 1.0 / n)

        result = minimize(
            self._neg_sharpe,
            init_weights,
            args=(posterior_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(
                "BL optimisation did not converge: %s – using market weights.",
                result.message,
            )
            weights = market_weights.copy()
        else:
            weights = result.x

        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        port_return = float(weights @ posterior_returns)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = (
            (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        )

        return OptimizationResult(
            weights={s: float(w) for s, w in zip(symbols, weights)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="black_litterman",
        )

    # ------------------------------------------------------------------
    # Black-Litterman internals
    # ------------------------------------------------------------------

    @staticmethod
    def compute_equilibrium_returns(
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_aversion: float = 2.5,
    ) -> np.ndarray:
        """Compute implied equilibrium excess returns: Pi = delta * Sigma * w_mkt."""
        return risk_aversion * (cov_matrix @ market_weights)

    def compute_posterior(
        self,
        equilibrium_returns: np.ndarray,
        cov_matrix: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """Combine equilibrium returns with views to get posterior returns.

        Args:
            equilibrium_returns: Pi vector from ``compute_equilibrium_returns``.
            cov_matrix: N x N covariance matrix.
            P: K x N pick matrix (K views, N assets).
            Q: K-length vector of view returns.
            confidences: K-length vector of confidence levels (0, 1].

        Returns:
            Posterior expected returns (N-length array).
        """
        tau = self.tau
        tau_sigma = tau * cov_matrix

        # Omega: diagonal uncertainty matrix for views.
        # Higher confidence -> lower uncertainty.
        omega_diag = (1.0 - confidences) / (confidences + 1e-12) * np.diag(
            P @ tau_sigma @ P.T
        )
        Omega = np.diag(omega_diag)

        # Posterior precision and mean.
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        Omega_inv = np.linalg.inv(Omega + np.eye(len(Q)) * 1e-12)

        posterior_precision = tau_sigma_inv + P.T @ Omega_inv @ P
        posterior_mean = np.linalg.solve(
            posterior_precision,
            tau_sigma_inv @ equilibrium_returns + P.T @ Omega_inv @ Q,
        )

        return posterior_mean

    # ------------------------------------------------------------------
    # Objective helper
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
