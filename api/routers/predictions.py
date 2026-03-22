"""Prediction endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_current_user
from api.schemas.predictions import PredictionSchema

router = APIRouter(prefix="/predictions")


@router.get(
    "/latest",
    response_model=list[PredictionSchema],
    summary="Latest predictions",
)
async def get_latest_predictions(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    _user: dict = Depends(get_current_user),
) -> list[PredictionSchema]:
    """Return the most recent predictions, optionally filtered by symbol.

    Will be connected to the ML prediction service once implemented.
    """
    # TODO: Query predictions table once model is created
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Prediction service not yet implemented",
    )


@router.get(
    "/history",
    response_model=list[PredictionSchema],
    summary="Historical predictions",
)
async def get_prediction_history(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _user: dict = Depends(get_current_user),
) -> list[PredictionSchema]:
    """Return historical predictions with optional filtering."""
    # TODO: Query predictions table once model is created
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Prediction service not yet implemented",
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionSchema,
    summary="Prediction detail",
)
async def get_prediction(
    prediction_id: uuid.UUID,
    _user: dict = Depends(get_current_user),
) -> PredictionSchema:
    """Return a single prediction by ID."""
    # TODO: Query predictions table once model is created
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Prediction service not yet implemented",
    )
