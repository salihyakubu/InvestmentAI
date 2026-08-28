"""Phase 2: the live-transfer promotion gate (registered 2026-08-28).

Era 4 proved the promotion criterion did not transfer: a challenger that
passed the validation-fold gate served a significantly NEGATIVE live IC for
twelve days. The first feature-drift reading explains why that can happen --
the served input distribution walks far from any fixed training fold within
weeks. This module is the registered fix: before promotion, a challenger is
REPLAYED over the champion's actually-served feature vectors (Phase 2a,
persisted with their outcomes) and must beat the champion replayed over the
IDENTICAL rows.

Everything below is fixed by the registration (GO_LIVE.md 2026-08-28):

* Data: trailing SCAN_DAYS of predictions carrying complete feature vectors
  and resolved outcomes, whose ordering hash equals the challenger's --
  computed by the SAME shared function the serving path uses.
* Floors: >= MIN_SPAN_DAYS between first and last row AND >= MIN_ROWS rows.
  Below either floor the gate REFUSES and no promotion occurs: the
  validation-only promotion is the measured failure mode, so a blind
  promotion is worse than none.
* Score: Spearman IC of replayed expected_return vs stored actual_return,
  identical row set for both models. Criterion: challenger > champion,
  strict. No champion artifact -> the replay leg passes vacuously.
* Hash mismatch (zero matching rows) -> loud refusal: serving and training
  share one feature pipeline, so a mismatch is an inconsistency.

Nothing here trades; the gate only decides whether a trained challenger may
replace its champion in serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

SCAN_DAYS = 30
MIN_SPAN_DAYS = 14.0
MIN_ROWS = 2_000

__all__ = [
    "MIN_ROWS",
    "MIN_SPAN_DAYS",
    "GateDecision",
    "feature_ordering_hash",
    "fetch_replay_rows",
    "live_transfer_gate",
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


async def fetch_replay_rows(
    session_factory: async_sessionmaker[AsyncSession],
    expected_hash: str,
    now: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """(X, y, span_days) from stored served vectors matching *expected_hash*.

    Only complete, hash-matched, outcome-resolved rows enter; a vector whose
    length disagrees with the first accepted row is dropped (it cannot be
    the same generation, whatever its hash claims).
    """
    now = now or datetime.now(UTC)
    since = now - timedelta(days=SCAN_DAYS)
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
                    PredictionRow.predicted_at >= since,
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
        vectors.append([float(x) for x in vector])
        outcomes.append(float(actual))
        stamps.append(when if when.tzinfo else when.replace(tzinfo=UTC))

    if not vectors:
        return np.empty((0, 0)), np.empty(0), 0.0
    span = (max(stamps) - min(stamps)).total_seconds() / 86_400.0
    return np.asarray(vectors, dtype=float), np.asarray(outcomes, dtype=float), span


def replay_ic(model: BasePredictor, X: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman IC of the model's replayed expected_return against outcomes.

    None when the replayed scores are degenerate (constant): a model with no
    ordering has no IC, and fabricating 0.0 would let it tie rather than
    fail the strict criterion.
    """
    from scipy import stats

    outputs = model.predict_batch(X)
    scores = np.asarray([o.expected_return for o in outputs], dtype=float)
    if not np.isfinite(scores).all() or np.std(scores) == 0 or np.std(y) == 0:
        return None
    rho = stats.spearmanr(scores, y)
    return None if np.isnan(rho.statistic) else float(rho.statistic)


async def live_transfer_gate(
    session_factory: async_sessionmaker[AsyncSession],
    challenger: BasePredictor,
    champion: BasePredictor | None,
    feature_names: list[str] | None,
    now: datetime | None = None,
) -> GateDecision:
    """The registered decision: may *challenger* replace *champion*?"""
    import asyncio

    if not feature_names:
        return GateDecision(
            promote=False, reason="no_feature_names_for_hash",
            challenger_ic=None, champion_ic=None, n_rows=0, span_days=None,
        )
    expected_hash = feature_ordering_hash(feature_names)
    X, y, span = await fetch_replay_rows(session_factory, expected_hash, now)

    if X.shape[0] == 0:
        return GateDecision(
            promote=False,
            reason=f"no_rows_for_feature_hash_{expected_hash}",
            challenger_ic=None, champion_ic=None, n_rows=0, span_days=None,
        )
    if X.shape[0] < MIN_ROWS or span < MIN_SPAN_DAYS:
        return GateDecision(
            promote=False, reason="insufficient_live_rows",
            challenger_ic=None, champion_ic=None,
            n_rows=int(X.shape[0]), span_days=round(span, 2),
        )

    if champion is None:
        # Registered edge: nothing to beat -- the replay leg passes
        # vacuously; the validation gate upstream still applies.
        challenger_ic = await asyncio.to_thread(replay_ic, challenger, X, y)
        return GateDecision(
            promote=True, reason="no_champion_vacuous_pass",
            challenger_ic=challenger_ic, champion_ic=None,
            n_rows=int(X.shape[0]), span_days=round(span, 2),
        )

    challenger_ic = await asyncio.to_thread(replay_ic, challenger, X, y)
    champion_ic = await asyncio.to_thread(replay_ic, champion, X, y)
    if challenger_ic is None:
        return GateDecision(
            promote=False, reason="degenerate_challenger_scores",
            challenger_ic=None, champion_ic=champion_ic,
            n_rows=int(X.shape[0]), span_days=round(span, 2),
        )
    if champion_ic is None:
        # A champion with no ordering on live data cannot be "beaten" under
        # the strict paired criterion; refuse rather than invent a tie.
        return GateDecision(
            promote=False, reason="degenerate_champion_scores",
            challenger_ic=challenger_ic, champion_ic=None,
            n_rows=int(X.shape[0]), span_days=round(span, 2),
        )

    promote = challenger_ic > champion_ic
    return GateDecision(
        promote=promote,
        reason="challenger_beats_champion_on_live_rows"
        if promote
        else "challenger_does_not_beat_champion_on_live_rows",
        challenger_ic=round(challenger_ic, 6),
        champion_ic=round(champion_ic, 6),
        n_rows=int(X.shape[0]),
        span_days=round(span, 2),
    )
