"""ML model management endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.ml_models import ModelMetadata
from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse

router = APIRouter(prefix="/models")


@router.get(
    "",
    summary="List all models",
)
async def list_models(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """Return all registered ML models (latest version of each)."""
    stmt = select(ModelMetadata).order_by(
        ModelMetadata.model_name, ModelMetadata.version.desc()
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    # Group by model name and return latest version
    seen: dict[str, dict] = {}
    for row in rows:
        if row.model_name not in seen:
            seen[row.model_name] = {
                "id": str(row.id),
                "model_name": row.model_name,
                "model_type": row.model_type,
                "version": row.version,
                "is_active": row.is_active,
                "trained_at": row.trained_at.isoformat() if row.trained_at else None,
            }

    return list(seen.values())


@router.get(
    "/{model_id}/versions",
    summary="Model version history",
)
async def list_model_versions(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[dict]:
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
    _user: dict = Depends(get_current_user),
) -> dict:
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
    _user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Trigger retraining for a specific model.

    The actual training is handled asynchronously by the ML service.
    """
    # TODO: Publish retrain event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model retraining service not yet implemented",
    )
