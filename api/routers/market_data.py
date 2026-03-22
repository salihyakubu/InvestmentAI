"""Market data endpoints."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import TimeFrame
from core.models.market_data import OHLCVRecord
from api.dependencies import get_db
from api.schemas.market_data import BarSchema, LatestPriceResponse

router = APIRouter(prefix="/market-data")


@router.get(
    "/{symbol}/bars",
    response_model=list[BarSchema],
    summary="Query historical OHLCV bars",
)
async def get_bars(
    symbol: str,
    timeframe: TimeFrame = Query(default=TimeFrame.D1),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10000),
    db: AsyncSession = Depends(get_db),
) -> list[BarSchema]:
    """Return historical OHLCV bars for a given symbol and timeframe."""
    stmt = (
        select(OHLCVRecord)
        .where(
            OHLCVRecord.symbol == symbol,
            OHLCVRecord.timeframe == timeframe.value,
        )
        .order_by(OHLCVRecord.time.desc())
        .limit(limit)
    )

    if start is not None:
        stmt = stmt.where(OHLCVRecord.time >= start)
    if end is not None:
        stmt = stmt.where(OHLCVRecord.time <= end)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        BarSchema(
            time=row.time,
            symbol=row.symbol,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        for row in rows
    ]


@router.get(
    "/{symbol}/latest",
    response_model=LatestPriceResponse,
    summary="Latest price for a symbol",
)
async def get_latest_price(
    symbol: str,
    db: AsyncSession = Depends(get_db),
) -> LatestPriceResponse:
    """Return the most recent price for a symbol."""
    stmt = (
        select(OHLCVRecord)
        .where(OHLCVRecord.symbol == symbol)
        .order_by(OHLCVRecord.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for symbol: {symbol}",
        )

    return LatestPriceResponse(
        symbol=row.symbol,
        price=row.close,
        timestamp=row.time,
    )
