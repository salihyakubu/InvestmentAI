"""ML model management endpoints."""

from __future__ import annotations

import time as _time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import extract, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse
from core.models.ml_models import ModelMetadata
from core.models.predictions import Prediction

router = APIRouter(prefix="/models")

# Learning metrics scan the full resolved history; the inputs move on the
# resolution cadence (minutes), so a short cache keeps the scan off the
# 5-second dashboard poll.
_LEARNING_TTL_SECONDS = 300.0
_learning_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# Drift never needs rows older than this: the reference is the current
# generation's FIRST days and the subject is the last days, both within.
_DRIFT_SCAN_DAYS = 60


@router.get(
    "/learning-metrics",
    summary="Does the learning loop learn? Live signal quality by model era",
)
async def get_learning_metrics(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Live out-of-sample quality of the served signal, era by era.

    De-overlapping happens AT FETCH TIME: predictions are emitted per minute
    against a 5-minute horizon, so sampling minute %% 5 == 0 yields
    independent observations and cuts the scan 5x in one stroke.
    """
    cached = _learning_cache.get("all")
    now = _time.monotonic()
    if cached is not None and now - cached[0] < _LEARNING_TTL_SECONDS:
        return cached[1]

    from services.continuous_learning.learning_metrics import (
        ResolvedPrediction,
        build_report,
    )

    rows = (
        await db.execute(
            select(
                Prediction.predicted_at,
                Prediction.symbol,
                Prediction.expected_return,
                Prediction.confidence,
                Prediction.direction,
                Prediction.actual_return,
                Prediction.actual_direction,
            )
            .where(
                Prediction.actual_return.is_not(None),
                Prediction.expected_return.is_not(None),
                extract("minute", Prediction.predicted_at) % 5 == 0,
            )
            .order_by(Prediction.predicted_at)
        )
    ).all()
    resolved = [
        ResolvedPrediction(
            predicted_at=r[0],
            symbol=r[1],
            expected_return=float(r[2]),
            confidence=float(r[3] or 0.0),
            direction=str(r[4]),
            actual_return=float(r[5]),
            actual_direction=str(r[6] or ""),
        )
        for r in rows
    ]

    promo_rows = (
        await db.execute(select(ModelMetadata.created_at))
    ).scalars().all()
    report = build_report(resolved, list(promo_rows))

    # Throughput/health over the last 7 days, from the FULL (unsampled) set.
    from datetime import UTC, datetime, timedelta

    since = datetime.now(UTC) - timedelta(days=7)
    health = (
        await db.execute(
            select(func.count()).select_from(Prediction).where(
                Prediction.predicted_at >= since
            )
        )
    ).scalar()

    feature_rows = (
        await db.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.features_used.is_not(None))
        )
    ).scalar()

    payload: dict[str, Any] = {
        "feature_rows_persisted": int(feature_rows or 0),
        "eras": [
            {
                "label": e.label,
                "start": e.start.isoformat(),
                "end": e.end.isoformat() if e.end else None,
                "days": e.days,
                "n": e.n,
                "mean_ic": None if e.mean_ic is None else round(e.mean_ic, 5),
                "ic_t_stat": None if e.ic_t_stat is None else round(e.ic_t_stat, 2),
                "mean_brier": None if e.mean_brier is None else round(e.mean_brier, 4),
                "sign_agreement": None
                if e.sign_agreement is None
                else round(e.sign_agreement, 4),
            }
            for e in report.eras
        ],
        "daily": [
            {
                "day": d.day.isoformat(),
                "n": d.n,
                "ic": None if d.ic is None else round(d.ic, 5),
                "sign_agreement": None
                if d.sign_agreement is None
                else round(d.sign_agreement, 4),
                "abstention_rate": round(d.abstention_rate, 4),
                "brier_flat": None if d.brier_flat is None else round(d.brier_flat, 4),
            }
            for d in report.daily[-14:]
        ],
        "calibration": report.calibration,
        "predictions_last_7d": int(health or 0),
        "observations_used": len(resolved),
        "notes": report.notes,
    }
    _learning_cache["all"] = (now, payload)
    return payload


@router.get(
    "/feature-drift",
    summary="Are the models seeing the world they were judged on?",
)
async def get_feature_drift(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """PSI per served-feature index: recent days vs the platform's own
    earlier serving history (see feature_drift module docstring for the
    honest limitations -- this brackets train-serve skew, it is not it)."""
    cached = _learning_cache.get("drift")
    now_mono = _time.monotonic()
    if cached is not None and now_mono - cached[0] < _LEARNING_TTL_SECONDS:
        return cached[1]

    from datetime import UTC, datetime, timedelta

    from services.continuous_learning.feature_drift import (
        PSI_MODERATE,
        PSI_SIGNIFICANT,
        build_report,
    )

    # Bounded scan: the report only ever needs the current generation's
    # first reference days plus the recent window, both inside this horizon.
    scan_since = datetime.now(UTC) - timedelta(days=_DRIFT_SCAN_DAYS)
    db_rows = (
        await db.execute(
            select(
                Prediction.predicted_at, Prediction.symbol, Prediction.features_used
            )
            .where(
                Prediction.features_used.is_not(None),
                Prediction.predicted_at >= scan_since,
            )
            .order_by(Prediction.predicted_at)
        )
    ).all()
    rows: list[tuple[datetime, str, str, list[float]]] = []
    for when, symbol, payload in db_rows:
        if not isinstance(payload, dict):
            continue
        vector, digest = payload.get("v"), payload.get("h")
        if not vector or not digest:
            continue
        when = when if when.tzinfo else when.replace(tzinfo=UTC)
        rows.append((when, str(symbol), str(digest), [float(x) for x in vector]))

    report = build_report(rows, datetime.now(UTC))
    measured = [f for f in report.features if f.psi is not None]
    top = sorted(measured, key=lambda f: f.psi or 0.0, reverse=True)[:5]
    notes = [
        *report.notes,
        f"Scan bounded to the last {_DRIFT_SCAN_DAYS} days of persisted rows; "
        "a generation older than that re-anchors its reference to the oldest "
        "retained window.",
    ]
    payload_out: dict[str, Any] = {
        "computable": report.computable,
        "reason": report.reason,
        "generation_hash": report.generation_hash,
        "n_reference": report.n_reference,
        "n_recent": report.n_recent,
        "n_symbols_measured": report.n_symbols_measured,
        "reference_start": (
            report.reference_start.isoformat() if report.reference_start else None
        ),
        "reference_end": (
            report.reference_end.isoformat() if report.reference_end else None
        ),
        "recent_start": (
            report.recent_start.isoformat() if report.recent_start else None
        ),
        "n_features": len(report.features),
        "n_unmeasurable": len(report.features) - len(measured),
        "share_significant": (
            round(
                sum(1 for f in measured if (f.psi or 0.0) > PSI_SIGNIFICANT)
                / len(measured),
                4,
            )
            if measured
            else None
        ),
        "top_drifted": [
            {
                "index": f.index,
                "psi": round(f.psi, 4),
                "psi_max": None if f.psi_max is None else round(f.psi_max, 4),
                "n_symbols": f.n_symbols,
            }
            for f in top
            if f.psi is not None
        ],
        "worst": (
            None
            if report.worst is None
            else {
                "symbol": report.worst[0],
                "index": report.worst[1],
                "psi": round(report.worst[2], 4),
            }
        ),
        "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
        "notes": notes,
    }
    _learning_cache["drift"] = (now_mono, payload_out)
    return payload_out


@router.get(
    "",
    summary="List all models",
)
async def list_models(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Return all registered ML models (latest version of each)."""
    stmt = select(ModelMetadata).order_by(
        ModelMetadata.model_name, ModelMetadata.version.desc()
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    # Group by model name and return latest version
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.model_name not in seen:
            seen[row.model_name] = {
                "id": str(row.id),
                "model_name": row.model_name,
                "model_type": row.model_type,
                "version": row.version,
                "is_active": row.is_active,
                "trained_at": row.trained_at.isoformat() if row.trained_at else None,
                # The dashboard's ML page renders these; without them it only
                # has zeros to show.
                "validation_metrics": row.validation_metrics or {},
                "feature_importance": row.feature_importance or {},
            }

    return list(seen.values())


@router.get(
    "/{model_id}/versions",
    summary="Model version history",
)
async def list_model_versions(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Return all versions of a specific model."""
    # First get the model name from the given ID
    stmt = select(ModelMetadata).where(ModelMetadata.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Now get all versions with that name
    stmt = (
        select(ModelMetadata)
        .where(ModelMetadata.model_name == model.model_name)
        .order_by(ModelMetadata.version.desc())
    )
    result = await db.execute(stmt)
    versions = result.scalars().all()

    return [
        {
            "id": str(v.id),
            "version": v.version,
            "is_active": v.is_active,
            "trained_at": v.trained_at.isoformat() if v.trained_at else None,
            "training_metrics": v.training_metrics,
            "validation_metrics": v.validation_metrics,
        }
        for v in versions
    ]


@router.get(
    "/{model_id}/performance",
    summary="Model performance metrics",
)
async def get_model_performance(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return performance metrics for a specific model version."""
    stmt = select(ModelMetadata).where(ModelMetadata.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    return {
        "id": str(model.id),
        "model_name": model.model_name,
        "version": model.version,
        "training_metrics": model.training_metrics,
        "validation_metrics": model.validation_metrics,
        "feature_importance": model.feature_importance,
        "training_data_start": (
            model.training_data_start.isoformat()
            if model.training_data_start
            else None
        ),
        "training_data_end": (
            model.training_data_end.isoformat()
            if model.training_data_end
            else None
        ),
    }


@router.post(
    "/{model_id}/retrain",
    response_model=SuccessResponse,
    summary="Trigger model retraining",
)
async def trigger_retrain(
    model_id: uuid.UUID,
    _user: dict[str, Any] = Depends(get_current_user),
) -> SuccessResponse:
    """Trigger retraining for a specific model.

    The actual training is handled asynchronously by the ML service.
    """
    # TODO: Publish retrain event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model retraining service not yet implemented",
    )
