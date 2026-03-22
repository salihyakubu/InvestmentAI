"""OHLCV market data model (TimescaleDB hypertable)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase


class OHLCVRecord(AsyncBase):
    """Maps to the ``ohlcv`` TimescaleDB hypertable."""

    __tablename__ = "ohlcv"

    # TimescaleDB hypertables don't have a traditional PK; we use a composite
    # of (time, symbol, timeframe) as the logical identity.
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(
        String(20), primary_key=True, nullable=False
    )
    timeframe: Mapped[str] = mapped_column(
        String(5), primary_key=True, nullable=False, server_default="1d"
    )

    asset_class: Mapped[str] = mapped_column(String(10), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<OHLCVRecord symbol={self.symbol!r} time={self.time} "
            f"tf={self.timeframe!r} close={self.close}>"
        )
