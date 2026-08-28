"""Phase 2: the live-transfer promotion gate (registered 2026-08-28, amended
same day before first decision -- the embargo).

Era 4 proved the promotion criterion did not transfer: a challenger that
passed the validation-fold gate served a significantly NEGATIVE live IC for
twelve days. This module is the registered fix: before promotion, a
challenger is REPLAYED over actually-served feature vectors (Phase 2a,
persisted with their outcomes) and must beat the champion replayed over the
IDENTICAL rows.

THE EMBARGO (amendment, 2026-08-28): review of the first build found the
replay window equal to the challenger's own training window -- the
challenger would have been scored in-sample and won by memorization, the
registered intent inverted. Now the challenger's training data must END
``EMBARGO_DAYS`` before now, and replay rows are drawn EXCLUSIVELY from
after that training end: out-of-sample for challenger and champion alike.

Fixed by the registration + amendment:

* Rows: complete served vectors with resolved outcomes, ordering hash equal
  to the challenger's (computed by the SAME shared function the serving
  path uses), predicted_at strictly AFTER the challenger's training end.
  Non-finite vectors or outcomes are dropped at fetch.
* Floors: >= MIN_SPAN_DAYS between first and last row AND >= MIN_ROWS rows;
  below either the gate REFUSES and no promotion occurs. A missing or
  > SCAN_DAYS-stale training boundary refuses (fail closed).
* Score: Spearman IC of replayed expected_return vs stored actual_return,
  identical row set for both models. Criterion: challenger > champion,
  strict. No champion artifact -> the replay leg passes vacuously.
* A degenerate OUTCOME series refuses as its own named condition -- never
  misattributed to either model's regressor.

The data check (``prepare_replay``) is separable from the model comparison
(``decide``) so a deterministic refusal is discovered BEFORE the expensive
training run, not after it. Nothing here trades.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.predictions import Prediction as PredictionRow
from services.prediction.feature_hash import feature_ordering_hash

if TYPE_CHECKING:
    from services.prediction.models.base import BasePredictor

logger = structlog.get_logger(__name__)

# The challenger's training data must end this many days before now; the
# replay window is exactly the embargoed span (training_end, now].
EMBARGO_DAYS = 15
# A training boundary older than this is stale: the replay window would
# stretch beyond the registered trailing month.
SCAN_DAYS = 30
MIN_SPAN_DAYS = 14.0
MIN_ROWS = 2_000

__all__ = [
    "EMBARGO_DAYS",
    "MIN_ROWS",
    "MIN_SPAN_DAYS",
    "GateDecision",
    "ReplayData",
    "decide",
    "feature_ordering_hash",
    "fetch_replay_rows",
    "live_transfer_gate",
    "prepare_replay",
    "replay_ic",
]


@dataclass(frozen=True)
class GateDecision:
    """One promotion decision, with everything the record needs."""

    promote: bool
    reason: str
    challenger_ic: float | None
    champion_ic: float | None
    n_rows: int
    span_days: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "challenger_ic": self.challenger_ic,
            "champion_ic": self.champion_ic,
            "n_rows": self.n_rows,
            "span_days": self.span_days,
        }


@dataclass(frozen=True)
class ReplayData:
    """Embargoed replay rows that cleared every data floor."""

    X: np.ndarray
    y: np.ndarray
    span_days: float


async def fetch_replay_rows(
    session_factory: async_sessionmaker[AsyncSession],
    expected_hash: str,
    training_end: datetime,
) -> tuple[np.ndarray, np.ndarray, float]:
    """(X, y, span_days) from served vectors strictly AFTER *training_end*.

    The strict lower bound IS the embargo: any row at or before the
    challenger's training end is potentially its own training data and must
    never score it. Complete, hash-matched, finite, outcome-resolved rows
    only; a vector whose length disagrees with the first accepted row is
    dropped (it cannot be the same generation, whatever its hash claims).
    """
    async with session_factory() as session:
        db_rows = (
            await session.execute(
                select(
                    PredictionRow.predicted_at,
                    PredictionRow.features_used,
                    PredictionRow.actual_return,
                )
                .where(
                    PredictionRow.features_used.is_not(None),
                    PredictionRow.actual_return.is_not(None),
                    PredictionRow.predicted_at > training_end,
                )
                .order_by(PredictionRow.predicted_at)
            )
        ).all()

    vectors: list[list[float]] = []
    outcomes: list[float] = []
    stamps: list[datetime] = []
    width: int | None = None
    for when, payload, actual in db_rows:
        if not isinstance(payload, dict):
            continue
        vector, digest = payload.get("v"), payload.get("h")
        if not vector or digest != expected_hash:
            continue
        if width is None:
            width = len(vector)
        if len(vector) != width:
            continue
        values = [float(x) for x in vector]
        outcome = float(actual)
        if not np.isfinite(outcome) or not all(np.isfinite(v) for v in values):
            continue  # a poisoned row must not decide promotions
        vectors.append(values)
        outcomes.append(outcome)
        stamps.append(when if when.tzinfo else when.replace(tzinfo=UTC))

    if not vectors:
        return np.empty((0, 0)), np.empty(0), 0.0
    span = (max(stamps) - min(stamps)).total_seconds() / 86_400.0
    return np.asarray(vectors, dtype=float), np.asarray(outcomes, dtype=float), span


async def prepare_replay(
    session_factory: async_sessionmaker[AsyncSession],
    feature_names: list[str] | None,
    training_end: datetime | None,
    now: datetime | None = None,
) -> ReplayData | GateDecision:
    """Every data-side check, runnable BEFORE any training cost is paid.

    Returns :class:`ReplayData` when the gate can judge, else the refusal
    :class:`GateDecision` (promote=False) explaining exactly why not.
    """
    now = now or datetime.now(UTC)

    def _refuse(reason: str, n_rows: int = 0, span: float | None = None) -> GateDecision:
        return GateDecision(
            promote=False, reason=reason, challenger_ic=None, champion_ic=None,
            n_rows=n_rows, span_days=None if span is None else round(span, 2),
        )

    if training_end is None:
        return _refuse("unknown_training_boundary")
    if training_end.tzinfo is None:
        training_end = training_end.replace(tzinfo=UTC)
    if (now - training_end).total_seconds() > SCAN_DAYS * 86_400:
        return _refuse("stale_training_boundary")
    if not feature_names:
        return _refuse("no_feature_names_for_hash")

    expected_hash = feature_ordering_hash(feature_names)
    X, y, span = await fetch_replay_rows(session_factory, expected_hash, training_end)
    if X.shape[0] == 0:
        return _refuse(f"no_rows_for_feature_hash_{expected_hash}")
    if X.shape[0] < MIN_ROWS or span < MIN_SPAN_DAYS:
        return _refuse("insufficient_live_rows", n_rows=int(X.shape[0]), span=span)
    if float(np.std(y)) == 0.0:
        # A property of the stored outcomes, never of either model.
        return _refuse("degenerate_outcomes", n_rows=int(X.shape[0]), span=span)
    return ReplayData(X=X, y=y, span_days=round(span, 2))


def replay_ic(model: BasePredictor, X: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman IC of the model's replayed expected_return against outcomes.

    None when the replayed scores are degenerate (constant or non-finite): a
    model with no ordering has no IC, and fabricating 0.0 would let it tie
    rather than fail the strict criterion.
    """
    from scipy import stats

    outputs = model.predict_batch(X)
    scores = np.asarray([o.expected_return for o in outputs], dtype=float)
    if not np.isfinite(scores).all() or np.std(scores) == 0:
        return None
    rho = stats.spearmanr(scores, y)
    return None if np.isnan(rho.statistic) else float(rho.statistic)


async def decide(
    data: ReplayData,
    challenger: BasePredictor,
    champion: BasePredictor | None,
) -> GateDecision:
    """The registered comparison over prepared, embargoed rows."""
    n_rows, span = int(data.X.shape[0]), data.span_days

    if champion is None:
        # Registered edge: nothing to beat -- the replay leg passes
        # vacuously; the validation gate upstream still applies.
        challenger_ic = await asyncio.to_thread(replay_ic, challenger, data.X, data.y)
        return GateDecision(
            promote=True, reason="no_champion_vacuous_pass",
            challenger_ic=challenger_ic, champion_ic=None,
            n_rows=n_rows, span_days=span,
        )

    challenger_ic = await asyncio.to_thread(replay_ic, challenger, data.X, data.y)
    champion_ic = await asyncio.to_thread(replay_ic, champion, data.X, data.y)
    if challenger_ic is None:
        return GateDecision(
            promote=False, reason="degenerate_challenger_scores",
            challenger_ic=None, champion_ic=champion_ic,
            n_rows=n_rows, span_days=span,
        )
    if champion_ic is None:
        # A champion with no ordering on live data cannot be "beaten" under
        # the strict paired criterion; refuse rather than invent a tie.
        return GateDecision(
            promote=False, reason="degenerate_champion_scores",
            challenger_ic=challenger_ic, champion_ic=None,
            n_rows=n_rows, span_days=span,
        )

    promote = challenger_ic > champion_ic
    return GateDecision(
        promote=promote,
        reason="challenger_beats_champion_on_live_rows"
        if promote
        else "challenger_does_not_beat_champion_on_live_rows",
        challenger_ic=round(challenger_ic, 6),
        champion_ic=round(champion_ic, 6),
        n_rows=n_rows,
        span_days=span,
    )


async def live_transfer_gate(
    session_factory: async_sessionmaker[AsyncSession],
    challenger: BasePredictor,
    champion: BasePredictor | None,
    feature_names: list[str] | None,
    training_end: datetime | None,
    now: datetime | None = None,
) -> GateDecision:
    """One-call form: prepare the embargoed rows, then decide."""
    prepared = await prepare_replay(session_factory, feature_names, training_end, now)
    if isinstance(prepared, GateDecision):
        return prepared
    return await decide(prepared, challenger, champion)
