"""Portfolio optimisation service – main entry point."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from config.settings import Settings
from core.enums import SignalDirection
from core.events.base import Event, EventBus
from core.events.risk_events import RebalanceRequestEvent
from core.events.signal_events import PredictionReadyEvent
from core.events.streams import PREDICTIONS_READY, REBALANCE
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

# Real-time tick stream published by data ingestion; the literal is owned
# there (kept in sync with execution and liquidation, which also consume it).
_MARKET_PRICES_STREAM = "market.prices"

# Minimum seconds between published rebalances, so a burst of per-symbol
# predictions produces one coherent target instead of one event per symbol.
_REBALANCE_COOLDOWN = 300.0

# Predictions older than this cannot qualify for a rebalance: the model scored
# a market state that no longer exists.
_PREDICTION_MAX_AGE = 120.0


class PortfolioOptimizerService:
    """Orchestrates portfolio optimisation and rebalancing.

    Subscribes to :class:`PredictionReadyEvent` on the ``predictions.ready``
    stream, buffers the latest prediction per symbol, and publishes a
    :class:`RebalanceRequestEvent` once at least one fresh, confident signal
    exists and the rebalance cooldown has elapsed.

    LONG-ONLY by design: the optimizer stack produces long weights, so only
    ``direction == "long"`` predictions qualify; ``short``/``flat`` signals
    simply do not qualify. A held symbol whose prediction stops qualifying
    exits via an explicit weight-0.0 allocation in the next published
    rebalance -- the risk manager never touches symbols absent from
    ``target_allocations``, so exits must be spelled out.
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

        # Buffer predictions for the current optimisation cycle. Keyed by
        # symbol (latest wins) so both dicts stay bounded by symbol count.
        self._pending_predictions: dict[str, PredictionReadyEvent] = {}
        self._prediction_received_at: dict[str, float] = {}  # monotonic clock

        # Latest tick per symbol, forwarded to risk as reference prices --
        # the risk manager has no market data and skips sizing without one.
        self._last_prices: dict[str, float] = {}

        # Cooldown clock (monotonic) and the held book: symbols we last
        # published with non-zero weight and have not yet exited.
        self._last_rebalance_at: float | None = None
        self._last_target_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the prediction and market-price streams."""
        await self.event_bus.subscribe(
            stream=PREDICTIONS_READY,
            group="portfolio_optimizer",
            consumer="optimizer-1",
            handler=self._on_prediction,
        )
        await self.event_bus.subscribe(
            stream=_MARKET_PRICES_STREAM,
            group="portfolio_optimizer",
            consumer="optimizer-1",
            handler=self._on_price_update,
        )
        logger.info("PortfolioOptimizerService started (optimizer=%s).", type(self.optimizer).__name__)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_prediction(self, event: Event) -> None:
        """Buffer an incoming prediction, then evaluate the rebalance gate."""
        prediction = PredictionReadyEvent.model_validate(event.model_dump())
        self._pending_predictions[prediction.symbol] = prediction
        self._prediction_received_at[prediction.symbol] = time.monotonic()
        logger.debug(
            "Received prediction for %s (confidence=%.2f).",
            prediction.symbol,
            prediction.confidence,
        )
        await self._maybe_rebalance()

    async def _on_price_update(self, event: Event) -> None:
        """Track the latest positive price per symbol (bounded: one entry each)."""
        if event.event_type != "PriceUpdateEvent":
            return
        symbol = getattr(event, "symbol", "") or event.payload.get("symbol", "")
        price_raw = getattr(event, "price", None) or event.payload.get("price")
        if not symbol or price_raw is None:
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return
        if price > 0.0:
            self._last_prices[str(symbol)] = price

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

        # Scale down if over-invested, but never scale up past the per-position
        # cap: the risk manager rejects any target_weight above
        # max_position_pct, so re-normalising to a full 1.0 with fewer than
        # max_positions names would guarantee downstream rejection of every
        # order. The shortfall stays in cash.
        total = sum(clipped.values())
        if total > 1.0:
            clipped = {s: w / total for s, w in clipped.items()}

        return clipped

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _qualifying_predictions(self, now: float) -> dict[str, PredictionReadyEvent]:
        """Return the buffered predictions eligible for optimisation.

        Long-only: a prediction qualifies when it is fresh, its direction is
        ``long``, and its confidence clears the configured floor.
        """
        min_confidence = self.settings.min_prediction_confidence
        qualifying: dict[str, PredictionReadyEvent] = {}
        for symbol, prediction in self._pending_predictions.items():
            received_at = self._prediction_received_at.get(symbol)
            if received_at is None or now - received_at > _PREDICTION_MAX_AGE:
                continue
            if prediction.direction != SignalDirection.LONG:
                continue
            if prediction.confidence < min_confidence:
                continue
            qualifying[symbol] = prediction
        return qualifying

    async def _maybe_rebalance(self) -> None:
        """Publish a rebalance if the cooldown has elapsed and signals qualify."""
        now = time.monotonic()
        if (
            self._last_rebalance_at is not None
            and now - self._last_rebalance_at < _REBALANCE_COOLDOWN
        ):
            return

        qualifying = self._qualifying_predictions(now)
        if not qualifying:
            # Hold. Deliberately NOT rotating to cash on every uncertain
            # minute -- that would thrash the book; risk-based exits are the
            # liquidation manager's and the kill switch's job.
            return

        # Risk skips sizing any symbol without a positive reference price, so
        # a cycle where no qualifying symbol is priced could only publish a
        # no-op event.
        if not any(self._last_prices.get(s, 0.0) > 0.0 for s in qualifying):
            logger.warning(
                "Rebalance skipped: no reference price for any of %s.",
                sorted(qualifying),
            )
            return

        # current_portfolio is unused by the optimizer stack; sizing against
        # actual positions happens downstream in the risk manager.
        target = self.optimize(qualifying, {})

        # Exits must be computed AFTER optimisation: _apply_constraints and
        # TargetAllocation both strip near-zero weights, so a weight-0.0 exit
        # added earlier would silently disappear from the payload.
        exit_symbols = self._last_target_symbols - set(target.weights)

        priced_targets = {
            s: w for s, w in target.weights.items() if self._last_prices.get(s, 0.0) > 0.0
        }
        priced_exits = {s for s in exit_symbols if self._last_prices.get(s, 0.0) > 0.0}
        unpriced = (set(target.weights) - set(priced_targets)) | (exit_symbols - priced_exits)
        if unpriced:
            # Risk cannot size these (targets and exits alike); an unpriced
            # exit stays in the held book and is retried next cycle.
            logger.warning(
                "Omitting %s from rebalance: no reference price.", sorted(unpriced)
            )
        if not priced_targets and not priced_exits:
            logger.warning("Rebalance skipped: nothing publishable has a price.")
            return

        reference_prices = {
            s: self._last_prices[s] for s in (*priced_targets, *sorted(priced_exits))
        }
        await self.trigger_rebalance(
            TargetAllocation(weights=priced_targets),
            reference_prices=reference_prices,
            exits=sorted(priced_exits),
        )

        self._last_rebalance_at = now
        # Held book: previously held symbols stay tracked unless their exit
        # was actually published; newly published targets are added.
        self._last_target_symbols = (
            self._last_target_symbols | set(priced_targets)
        ) - priced_exits

    async def trigger_rebalance(
        self,
        target: TargetAllocation,
        reference_prices: dict[str, float] | None = None,
        exits: Iterable[str] = (),
    ) -> None:
        """Publish a :class:`RebalanceRequestEvent` to the event bus.

        *exits* ride along as explicit weight-0.0 allocations (they cannot
        live inside *target*, which drops zero weights): the risk manager
        only acts on symbols present in ``target_allocations``.
        """
        allocations: dict[str, float] = dict(target.weights)
        for symbol in exits:
            allocations.setdefault(symbol, 0.0)
        if not allocations:
            logger.info("Rebalance not published: empty allocation set.")
            return
        event = RebalanceRequestEvent(
            target_allocations=allocations,
            reference_prices=dict(reference_prices or {}),
            source_service="portfolio_optimizer",
        )
        await self.event_bus.publish(REBALANCE, event)
        logger.info(
            "Published RebalanceRequestEvent: %d targets, %d exits.",
            len(target.weights),
            len(allocations) - len(target.weights),
        )

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
