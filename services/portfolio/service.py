"""Portfolio optimisation service – main entry point."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.settings import Settings
from core.events.base import Event, EventBus
from core.events.risk_events import RebalanceRequestEvent
from core.events.signal_events import PredictionReadyEvent
from services.portfolio.allocation import Rebalancer, TargetAllocation
from services.portfolio.constraints import PortfolioConstraints
from services.portfolio.optimizers.base import BaseOptimizer
from services.portfolio.optimizers.black_litterman import BlackLittermanOptimizer
from services.portfolio.optimizers.mean_variance import MeanVarianceOptimizer
from services.portfolio.optimizers.risk_parity import RiskParityOptimizer

logger = logging.getLogger(__name__)

_OPTIMIZERS: dict[str, type[BaseOptimizer]] = {
    "mean_variance": MeanVarianceOptimizer,
    "risk_parity": RiskParityOptimizer,
    "black_litterman": BlackLittermanOptimizer,
}


class PortfolioOptimizerService:
    """Orchestrates portfolio optimisation and rebalancing.

    Subscribes to :class:`PredictionReadyEvent` on the event bus,
    accumulates predictions, and triggers rebalancing when all symbols
    have been scored.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        optimizer_type: str = "mean_variance",
    ) -> None:
        self.event_bus = event_bus
        self.settings = settings
        self.rebalancer = Rebalancer(min_trade_notional=1.0)

        self.constraints = PortfolioConstraints(
            max_weight=settings.max_position_pct,
            max_sector_weight=settings.max_sector_pct,
            max_asset_class_weight=settings.max_asset_class_pct,
            max_positions=settings.max_portfolio_positions,
        )

        optimizer_cls = _OPTIMIZERS.get(optimizer_type)
        if optimizer_cls is None:
            raise ValueError(
                f"Unknown optimizer type '{optimizer_type}'. "
                f"Available: {list(_OPTIMIZERS.keys())}"
            )
        self.optimizer: BaseOptimizer = optimizer_cls(
            **self._optimizer_kwargs(optimizer_type)
        )

        # Buffer predictions for the current optimisation cycle.
        self._pending_predictions: dict[str, PredictionReadyEvent] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the prediction stream on the event bus."""
        await self.event_bus.subscribe(
            stream="predictions",
            group="portfolio_optimizer",
            consumer="optimizer-1",
            handler=self._on_prediction,
        )
        logger.info("PortfolioOptimizerService started (optimizer=%s).", type(self.optimizer).__name__)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_prediction(self, event: Event) -> None:
        """Handle an incoming prediction event."""
        prediction = PredictionReadyEvent.model_validate(event.model_dump())
        self._pending_predictions[prediction.symbol] = prediction
        logger.debug(
            "Received prediction for %s (confidence=%.2f).",
            prediction.symbol,
            prediction.confidence,
        )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        predictions: dict[str, PredictionReadyEvent],
        current_portfolio: dict[str, float],
    ) -> TargetAllocation:
        """Run the optimizer using buffered predictions.

        Args:
            predictions: symbol -> PredictionReadyEvent.
            current_portfolio: symbol -> current dollar position value.

        Returns:
            TargetAllocation with optimised weights.
        """
        symbols = sorted(predictions.keys())
        n = len(symbols)

        if n == 0:
            return TargetAllocation(weights={})

        expected_returns, cov_matrix = self._compute_returns_and_cov(
            symbols, predictions
        )

        constraint_dict = self.constraints.to_optimizer_dict()
        result = self.optimizer.optimize(
            expected_returns, cov_matrix, symbols, constraint_dict
        )

        constrained_weights = self._apply_constraints(result.weights)

        logger.info(
            "Optimisation complete: method=%s sharpe=%.4f vol=%.4f",
            result.method,
            result.sharpe_ratio,
            result.expected_volatility,
        )

        return TargetAllocation(weights=constrained_weights)

    def _compute_returns_and_cov(
        self,
        symbols: list[str],
        predictions: dict[str, PredictionReadyEvent],
        lookback_days: int = 60,  # noqa: ARG002 – reserved for historical data integration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build expected-return vector and covariance matrix.

        Currently uses prediction-derived expected returns and constructs
        a synthetic covariance matrix.  When historical price data is
        available this should be replaced with sample covariance
        estimated over *lookback_days* of daily returns.
        """
        n = len(symbols)

        # Expected returns from predictions.
        expected_returns = np.array(
            [
                predictions[s].expected_return
                if predictions[s].expected_return is not None
                else 0.0
                for s in symbols
            ]
        )

        # Synthetic covariance: uncorrelated assets with vol proportional
        # to |expected_return|.  Replace with historical sample covariance
        # once a data layer is in place.
        vols = np.maximum(np.abs(expected_returns) * 2, 0.01)
        cov_matrix = np.diag(vols**2)

        return expected_returns, cov_matrix

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Enforce hard constraints on the weight vector.

        Clips individual weights, enforces position count limits, and
        re-normalises.
        """
        max_w = self.constraints.max_weight
        min_w = self.constraints.min_weight
        max_pos = self.constraints.max_positions

        # Clip per-position bounds.
        clipped = {s: min(max(w, min_w), max_w) for s, w in weights.items()}

        # Enforce max positions: keep the top-N by weight.
        if len(clipped) > max_pos:
            sorted_symbols = sorted(clipped, key=lambda s: clipped[s], reverse=True)
            clipped = {s: clipped[s] for s in sorted_symbols[:max_pos]}

        # Remove negligible weights.
        clipped = {s: w for s, w in clipped.items() if w > 1e-6}

        # Re-normalise to sum to 1.
        total = sum(clipped.values())
        if total > 0:
            clipped = {s: w / total for s, w in clipped.items()}

        return clipped

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    async def trigger_rebalance(self, target: TargetAllocation) -> None:
        """Publish a :class:`RebalanceRequestEvent` to the event bus."""
        event = RebalanceRequestEvent(
            target_allocations=target.weights,
            source_service="portfolio_optimizer",
        )
        await self.event_bus.publish("rebalance", event)
        logger.info("Published RebalanceRequestEvent for %d symbols.", len(target.weights))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _optimizer_kwargs(self, optimizer_type: str) -> dict[str, Any]:
        """Build constructor kwargs for the chosen optimizer."""
        common: dict[str, Any] = {"max_position_pct": self.settings.max_position_pct}
        if optimizer_type in ("mean_variance", "black_litterman"):
            common["risk_free_rate"] = 0.0
        if optimizer_type == "black_litterman":
            common["risk_aversion"] = 2.5
            common["tau"] = 0.05
        return common
