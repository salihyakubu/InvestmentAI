"""Value at Risk (VaR) and Conditional VaR calculations."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class VaRCalculator:
    """Compute Value at Risk using multiple methodologies.

    All methods return *dollar* VaR amounts (positive numbers representing
    potential loss).
    """

    # ------------------------------------------------------------------
    # Historical VaR
    # ------------------------------------------------------------------

    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 1.0,
    ) -> float:
        """Compute historical VaR from an array of portfolio returns.

        Parameters
        ----------
        returns:
            1-D array of historical portfolio returns (e.g. daily log or
            simple returns).
        confidence:
            Confidence level (e.g. 0.95 for 95 % VaR).
        portfolio_value:
            Current portfolio value used to convert percentage VaR to
            dollar VaR.

        Returns
        -------
        float
            Dollar VaR (positive number).
        """
        if len(returns) == 0:
            return 0.0

        percentile = (1.0 - confidence) * 100.0
        var_pct = float(np.percentile(returns, percentile))

        # VaR is a loss amount, so negate the left-tail return
        return abs(var_pct) * portfolio_value

    # ------------------------------------------------------------------
    # Parametric (variance-covariance) VaR
    # ------------------------------------------------------------------

    @staticmethod
    def parametric_var(
        portfolio_value: float,
        mean_return: float,
        std_return: float,
        confidence: float = 0.95,
    ) -> float:
        """Compute parametric VaR assuming normally distributed returns.

        Parameters
        ----------
        portfolio_value:
            Current dollar value of the portfolio.
        mean_return:
            Expected return over the holding period.
        std_return:
            Standard deviation of returns over the holding period.
        confidence:
            Confidence level.

        Returns
        -------
        float
            Dollar VaR (positive number).
        """
        if std_return <= 0.0:
            return 0.0

        z_score = stats.norm.ppf(1.0 - confidence)  # negative
        var_pct = -(mean_return + z_score * std_return)

        return max(var_pct, 0.0) * portfolio_value

    # ------------------------------------------------------------------
    # Monte Carlo VaR
    # ------------------------------------------------------------------

    @staticmethod
    def monte_carlo_var(
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        portfolio_value: float,
        n_simulations: int = 10_000,
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """Compute Monte Carlo VaR using Cholesky-decomposed correlated sims.

        Parameters
        ----------
        returns_matrix:
            (T, N) array of historical returns for N assets over T periods.
        weights:
            (N,) array of portfolio weights summing to 1.
        portfolio_value:
            Current dollar value of the portfolio.
        n_simulations:
            Number of simulated portfolio return paths.
        confidence:
            Confidence level.
        holding_period:
            Number of periods to simulate forward.

        Returns
        -------
        float
            Dollar VaR (positive number).
        """
        if returns_matrix.size == 0 or len(weights) == 0:
            return 0.0

        n_assets = returns_matrix.shape[1] if returns_matrix.ndim > 1 else 1

        if n_assets == 1:
            returns_matrix = returns_matrix.reshape(-1, 1)
            weights = np.array([1.0])

        # Estimate mean and covariance from historical data
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        # Ensure covariance matrix is positive semi-definite by adding a
        # small regularisation term if needed.
        try:
            cholesky = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small diagonal for numerical stability
            cov_matrix += np.eye(n_assets) * 1e-8
            cholesky = np.linalg.cholesky(cov_matrix)

        # Generate correlated random returns
        rng = np.random.default_rng(seed=42)
        uncorrelated = rng.standard_normal((n_simulations, n_assets))
        correlated = uncorrelated @ cholesky.T

        # Scale to match historical distribution
        simulated_returns = correlated * np.sqrt(holding_period) + mean_returns * holding_period

        # Portfolio returns
        portfolio_returns = simulated_returns @ weights

        # VaR at the given confidence level
        percentile = (1.0 - confidence) * 100.0
        var_pct = float(np.percentile(portfolio_returns, percentile))

        return abs(min(var_pct, 0.0)) * portfolio_value

    # ------------------------------------------------------------------
    # CVaR (Expected Shortfall)
    # ------------------------------------------------------------------

    @staticmethod
    def cvar(
        returns: np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 1.0,
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall).

        CVaR is the expected loss *given* that the loss exceeds the VaR
        threshold.

        Parameters
        ----------
        returns:
            1-D array of historical portfolio returns.
        confidence:
            Confidence level.
        portfolio_value:
            Current portfolio value.

        Returns
        -------
        float
            Dollar CVaR (positive number).
        """
        if len(returns) == 0:
            return 0.0

        percentile = (1.0 - confidence) * 100.0
        var_threshold = float(np.percentile(returns, percentile))

        # Average of returns that fall below the VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        cvar_pct = float(np.mean(tail_returns))
        return abs(cvar_pct) * portfolio_value
