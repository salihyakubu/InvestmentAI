"""Base optimizer interface for portfolio construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""

    weights: dict[str, float]  # symbol -> weight (0-1)
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str


class BaseOptimizer(ABC):
    """Abstract base class for all portfolio optimizers."""

    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: list[str],
        constraints: dict | None = None,
    ) -> OptimizationResult:
        """Compute optimal portfolio weights.

        Args:
            expected_returns: Array of expected returns per asset.
            cov_matrix: Covariance matrix of asset returns.
            symbols: List of asset symbols matching array indices.
            constraints: Optional dict of constraint parameters.

        Returns:
            OptimizationResult with optimal weights and statistics.
        """
        ...
