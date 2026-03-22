"""Portfolio correlation and concentration monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationViolation:
    """A pair of assets whose correlation exceeds the allowed limit."""

    symbol_a: str
    symbol_b: str
    correlation: float
    limit: float


class CorrelationMonitor:
    """Monitor pairwise asset correlations and portfolio concentration."""

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def compute_correlation_matrix(
        returns_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute the pairwise correlation matrix from return series.

        Parameters
        ----------
        returns_dict:
            Mapping of symbol -> 1-D numpy array of returns. All arrays
            must have the same length.

        Returns
        -------
        np.ndarray
            (N, N) correlation matrix ordered by sorted symbol names.
        """
        if not returns_dict:
            return np.array([]).reshape(0, 0)

        symbols = sorted(returns_dict.keys())
        n = len(symbols)

        if n == 1:
            return np.array([[1.0]])

        # Stack into (T, N) matrix
        try:
            returns_matrix = np.column_stack(
                [returns_dict[s] for s in symbols]
            )
        except ValueError:
            logger.error("Return series have mismatched lengths.")
            return np.eye(n)

        if returns_matrix.shape[0] < 2:
            logger.warning("Fewer than 2 observations; returning identity matrix.")
            return np.eye(n)

        # np.corrcoef returns (N, N) when given (T, N) with rowvar=False
        corr = np.corrcoef(returns_matrix, rowvar=False)

        # Handle NaN values (e.g. zero-variance series)
        corr = np.nan_to_num(corr, nan=0.0)

        return corr

    # ------------------------------------------------------------------
    # Pairwise limit checks
    # ------------------------------------------------------------------

    @staticmethod
    def check_pairwise_limits(
        corr_matrix: np.ndarray,
        symbols: list[str],
        max_correlation: float = 0.85,
    ) -> list[CorrelationViolation]:
        """Check all pairwise correlations against a maximum threshold.

        Parameters
        ----------
        corr_matrix:
            (N, N) correlation matrix.
        symbols:
            List of symbol names matching the matrix ordering.
        max_correlation:
            Maximum allowed absolute correlation.

        Returns
        -------
        list[CorrelationViolation]
            Pairs that exceed the limit.
        """
        violations: list[CorrelationViolation] = []
        n = len(symbols)

        if corr_matrix.size == 0 or n < 2:
            return violations

        for i in range(n):
            for j in range(i + 1, n):
                abs_corr = abs(float(corr_matrix[i, j]))
                if abs_corr > max_correlation:
                    violations.append(
                        CorrelationViolation(
                            symbol_a=symbols[i],
                            symbol_b=symbols[j],
                            correlation=float(corr_matrix[i, j]),
                            limit=max_correlation,
                        )
                    )

        if violations:
            logger.warning(
                "%d pairwise correlation violations (limit=%.2f).",
                len(violations),
                max_correlation,
            )

        return violations

    # ------------------------------------------------------------------
    # Portfolio concentration
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_concentration(
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute portfolio concentration metrics.

        Parameters
        ----------
        weights:
            Mapping of symbol -> portfolio weight (should sum to ~1.0).

        Returns
        -------
        dict
            ``hhi`` -- Herfindahl-Hirschman Index (0 = diversified, 1 = concentrated).
            ``max_weight`` -- largest single position weight.
            ``effective_positions`` -- 1 / HHI, the effective number of
            equally-weighted positions.
        """
        if not weights:
            return {
                "hhi": 0.0,
                "max_weight": 0.0,
                "effective_positions": 0.0,
            }

        w = np.array(list(weights.values()), dtype=float)

        # Filter out zero / negative weights
        w = w[w > 0.0]

        if len(w) == 0:
            return {
                "hhi": 0.0,
                "max_weight": 0.0,
                "effective_positions": 0.0,
            }

        # Normalise so weights sum to 1
        total = w.sum()
        if total > 0.0:
            w = w / total

        hhi = float(np.sum(w**2))
        max_weight = float(np.max(w))
        effective_positions = 1.0 / hhi if hhi > 0.0 else 0.0

        return {
            "hhi": hhi,
            "max_weight": max_weight,
            "effective_positions": effective_positions,
        }
