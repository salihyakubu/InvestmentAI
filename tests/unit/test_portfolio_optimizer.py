"""Unit tests for portfolio optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from services.portfolio.optimizers.black_litterman import BlackLittermanOptimizer
from services.portfolio.optimizers.mean_variance import MeanVarianceOptimizer
from services.portfolio.optimizers.risk_parity import RiskParityOptimizer


@pytest.fixture
def symbols() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL", "AMZN"]


@pytest.fixture
def expected_returns() -> np.ndarray:
    return np.array([0.12, 0.10, 0.14, 0.11])


@pytest.fixture
def cov_matrix() -> np.ndarray:
    """Return a realistic covariance matrix for 4 assets."""
    # Correlation matrix.
    corr = np.array(
        [
            [1.00, 0.65, 0.55, 0.50],
            [0.65, 1.00, 0.60, 0.55],
            [0.55, 0.60, 1.00, 0.45],
            [0.50, 0.55, 0.45, 1.00],
        ]
    )
    vols = np.array([0.20, 0.18, 0.25, 0.22])
    cov = np.outer(vols, vols) * corr
    return cov


class TestMeanVarianceOptimizer:
    """Tests for the Markowitz mean-variance optimizer."""

    def test_mean_variance_weights_sum_to_one(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> None:
        """Optimal weights must sum to 1.0."""
        opt = MeanVarianceOptimizer(max_position_pct=0.50)
        result = opt.optimize(expected_returns, cov_matrix, symbols)

        total = sum(result.weights.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_mean_variance_respects_bounds(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> None:
        """No individual weight should exceed the configured max."""
        max_pct = 0.40
        opt = MeanVarianceOptimizer(max_position_pct=max_pct)
        result = opt.optimize(expected_returns, cov_matrix, symbols)

        for symbol, weight in result.weights.items():
            assert weight <= max_pct + 1e-6, (
                f"{symbol} weight {weight:.4f} exceeds max {max_pct}"
            )
            assert weight >= -1e-6, f"{symbol} weight {weight:.4f} is negative"


class TestRiskParityOptimizer:
    """Tests for the risk parity optimizer."""

    def test_risk_parity_equal_contribution(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> None:
        """Risk contributions should be approximately equal across assets."""
        opt = RiskParityOptimizer(max_iterations=1000, tolerance=1e-10)
        result = opt.optimize(expected_returns, cov_matrix, symbols)

        weights = np.array([result.weights[s] for s in symbols])

        # Compute marginal risk contributions.
        mrc = RiskParityOptimizer.marginal_risk_contribution(weights, cov_matrix)
        risk_contributions = weights * mrc
        total_risk = risk_contributions.sum()

        if total_risk > 1e-12:
            rc_pcts = risk_contributions / total_risk
            # Each asset should contribute approximately 1/n of total risk.
            target = 1.0 / len(symbols)
            for i, s in enumerate(symbols):
                assert pytest.approx(rc_pcts[i], abs=0.05) == target, (
                    f"{s} risk contribution {rc_pcts[i]:.4f} != {target:.4f}"
                )


class TestBlackLittermanOptimizer:
    """Tests for the Black-Litterman optimizer."""

    def test_black_litterman_with_no_views(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> None:
        """With no views, BL should fall back to equilibrium-implied
        weights that still sum to 1.0."""
        opt = BlackLittermanOptimizer(max_position_pct=0.50)
        result = opt.optimize(
            expected_returns,
            cov_matrix,
            symbols,
            constraints=None,
        )

        total = sum(result.weights.values())
        assert pytest.approx(total, abs=1e-6) == 1.0
        assert result.method == "black_litterman"

        # All weights should be non-negative.
        for s, w in result.weights.items():
            assert w >= -1e-6, f"{s} weight {w:.4f} is negative"
