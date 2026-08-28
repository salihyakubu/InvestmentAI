"""Walk-forward research endpoints.

Read-only view over the ``factor_watch`` adjudication record: the live
quarterly IC of every registered hypothesis, and how far each is through the
registered bar (four consecutive positive quarters on its own unseen data).
The API reports; it does not judge early.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from core.models.research import FactorWatchObservation
from services.research.funding_watch import (
    HORIZON_STAMPS,
    WATCHED_FACTORS,
    quarterly_rollup,
)

router = APIRouter(prefix="/research")

# The registered bar, restated where the API serves it so the UI cannot
# drift from the runbook.
REQUIRED_CONSECUTIVE_POSITIVE_QUARTERS = 4

_HYPOTHESES = {
    "funding_carry_24h_plus": (
        "Since mid-2025, perpetuals with HIGH funding outperform "
        "cross-sectionally (registered 2026-07-31)"
    ),
    "reversal_8h_minus": (
        "Fading the last 8h move predicts relative return "
        "(sign from the spot study; registered 2026-08-02)"
    ),
    "momentum_72h_minus": (
        "Fading the 3-day move predicts relative return "
        "(sign from the spot study; registered 2026-08-02)"
    ),
    "low_vol_7d_minus": (
        "Quiet contracts outperform volatile ones "
        "(sign from the spot study; registered 2026-08-02)"
    ),
    "momentum_24h_minus": (
        "Fading the 24h move predicts relative return "
        "(sign from the spot study; registered 2026-08-07)"
    ),
    "volume_surge_minus": (
        "Fading volume spikes predicts relative return "
        "(sign from the spot study; registered 2026-08-07)"
    ),
    "funding_delta_minus": (
        "Rising funding marks building crowding; fade it "
        "(economic prior only -- weakest provenance; registered 2026-08-07)"
    ),
    "blend_registered_v1": (
        "The frozen equal-weight blend of the four prior registered factors "
        "beats its best component (registered 2026-08-07)"
    ),
    "oi_buildup_24h_minus": (
        "Fading rapid 24h open-interest growth predicts relative return "
        "(economic prior only; registered 2026-08-09)"
    ),
    "crowded_longs_minus": (
        "Fading the account-count long/short crowd predicts relative return "
        "(economic prior only; registered 2026-08-09)"
    ),
    "taker_buying_24h_minus": (
        "Fading aggressive taker buying, lagged one stamp for causality, "
        "predicts relative return (economic prior only; registered 2026-08-09)"
    ),
    "near_high_fade_minus": (
        "Contracts at their own 7-day high underperform -- the testable "
        "kernel of a viral indicator ad (external claim, weakest "
        "provenance; registered 2026-08-28)"
    ),
}


@router.get(
    "/funding-watch",
    summary="Live walk-forward record of every registered hypothesis",
)
async def get_funding_watch(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    result = await db.execute(
        select(FactorWatchObservation).order_by(FactorWatchObservation.time)
    )
    rows = result.scalars().all()
    by_factor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_factor.setdefault(row.factor, []).append(
            {"time": row.time, "ic": row.ic, "n_symbols": row.n_symbols}
        )

    factors: list[dict[str, Any]] = []
    for watched in WATCHED_FACTORS:
        observations = by_factor.get(watched.name, [])
        quarters = quarterly_rollup(observations)
        streak = 0
        for quarter in reversed(quarters):
            if quarter["positive"]:
                streak += 1
            else:
                break
        factors.append(
            {
                "factor": watched.name,
                "hypothesis": _HYPOTHESES.get(watched.name, ""),
                "unseen_from": watched.unseen_from,
                "observations_recorded": len(observations),
                "quarters": [
                    {
                        "quarter": q["quarter"],
                        "n": q["n"],
                        "mean_ic": round(q["mean_ic"], 5),
                        "t_stat": round(q["t_stat"], 2),
                        "positive": q["positive"],
                    }
                    for q in quarters
                ],
                "current_positive_streak": streak,
            }
        )

    return {
        "horizon_hours": HORIZON_STAMPS * 8,
        "required_consecutive_positive_quarters": REQUIRED_CONSECUTIVE_POSITIVE_QUARTERS,
        "factors": factors,
        "verdict": (
            "watching -- no trading proposal may cite any factor until its "
            "registered bar is met on complete quarters of unseen data"
        ),
    }
