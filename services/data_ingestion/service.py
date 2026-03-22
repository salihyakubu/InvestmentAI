"""Data ingestion service -- orchestrates providers, normalisation, and storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import Settings, get_settings
from core.enums import AssetClass, TimeFrame
from core.events.base import Event, EventBus
from core.events.market_events import BarCloseEvent, PriceUpdateEvent
from core.models.base import get_async_session_factory
from core.models.market_data import OHLCVRecord
from services.data_ingestion.normalizer import NormalizationError, normalize_bar
from services.data_ingestion.providers.alpaca_provider import AlpacaDataProvider
from services.data_ingestion.providers.base import BaseDataProvider, RawBar
from services.data_ingestion.providers.ccxt_provider import CCXTDataProvider
from services.data_ingestion.providers.yahoo_provider import YahooDataProvider
from services.data_ingestion.validator import detect_gaps, validate_bar

logger = structlog.get_logger(__name__)

# Redis Streams topic for market bar events.
MARKET_BARS_STREAM = "market.bars"
MARKET_PRICES_STREAM = "market.prices"


class DataIngestionService:
    """Central coordinator for market-data ingestion.

    Responsibilities:
    - Register and manage data providers per asset class.
    - Fetch historical bars and persist them to the database.
    - Subscribe to real-time streams and publish events via the EventBus.
    - Apply normalisation and validation before storage.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings or get_settings()
        self._providers: dict[AssetClass, BaseDataProvider] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._running = False

        self._register_providers()

    # ------------------------------------------------------------------
    # Provider registration
    # ------------------------------------------------------------------

    def _register_providers(self) -> None:
        """Instantiate and register the configured providers."""
        alpaca_key = self._settings.alpaca_api_key.get_secret_value()
        alpaca_secret = self._settings.alpaca_secret_key.get_secret_value()

        if alpaca_key and alpaca_secret:
            self._providers[AssetClass.STOCK] = AlpacaDataProvider(
                api_key=alpaca_key,
                secret_key=alpaca_secret,
            )
            logger.info("ingestion.provider.registered", provider="alpaca")
        else:
            # Fall back to Yahoo for historical-only stock data.
            self._providers[AssetClass.STOCK] = YahooDataProvider()
            logger.info(
                "ingestion.provider.registered",
                provider="yahoo",
                note="alpaca credentials not set; using yahoo fallback",
            )

        binance_key = self._settings.binance_api_key.get_secret_value()
        binance_secret = self._settings.binance_secret_key.get_secret_value()

        # Only pass credentials if they're real (not placeholders).
        if binance_key and not binance_key.startswith("your_"):
            crypto_kwargs = {"api_key": binance_key, "secret_key": binance_secret}
        else:
            crypto_kwargs = {}  # Use unauthenticated public access

        self._providers[AssetClass.CRYPTO] = CCXTDataProvider(**crypto_kwargs)
        logger.info("ingestion.provider.registered", provider="ccxt_binance")

    def get_provider(self, asset_class: AssetClass) -> BaseDataProvider:
        """Return the provider registered for *asset_class*."""
        provider = self._providers.get(asset_class)
        if provider is None:
            raise ValueError(f"No provider registered for {asset_class}")
        return provider

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the real-time ingestion loops for all asset classes."""
        if self._running:
            logger.warning("ingestion.already_running")
            return

        self._running = True
        logger.info("ingestion.starting")

        # Launch a real-time loop per asset class with configured symbols.
        stock_symbols = self._settings.active_symbols_stocks
        crypto_symbols = self._settings.active_symbols_crypto

        if stock_symbols:
            task = asyncio.create_task(
                self._run_realtime_loop(AssetClass.STOCK, stock_symbols),
                name="ingestion-realtime-stock",
            )
            self._tasks.append(task)

        if crypto_symbols:
            task = asyncio.create_task(
                self._run_realtime_loop(AssetClass.CRYPTO, crypto_symbols),
                name="ingestion-realtime-crypto",
            )
            self._tasks.append(task)

        logger.info(
            "ingestion.started",
            stock_symbols=stock_symbols,
            crypto_symbols=crypto_symbols,
        )

    async def stop(self) -> None:
        """Gracefully stop all ingestion tasks and provider streams."""
        self._running = False
        logger.info("ingestion.stopping")

        # Unsubscribe all providers.
        for asset_class, provider in self._providers.items():
            try:
                await provider.unsubscribe()
            except Exception:
                logger.exception(
                    "ingestion.unsubscribe_error",
                    provider=provider.name,
                )

        # Cancel background tasks.
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("ingestion.stopped")

    # ------------------------------------------------------------------
    # Historical ingestion
    # ------------------------------------------------------------------

    async def ingest_historical(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: AssetClass,
    ) -> int:
        """Fetch and store historical bars for the given symbols.

        Returns the total number of bars persisted.
        """
        provider = self.get_provider(asset_class)
        total_stored = 0

        for symbol in symbols:
            try:
                raw_bars = await provider.fetch_historical_bars(
                    symbol, timeframe, start, end
                )
                if not raw_bars:
                    logger.info(
                        "ingestion.historical.no_data",
                        symbol=symbol,
                        provider=provider.name,
                    )
                    continue

                normalised = self._normalize_bars(
                    raw_bars, asset_class, provider.name, timeframe
                )
                validated = self._validate_bars(normalised)

                # Detect gaps and log them (informational).
                gaps = detect_gaps(validated, timeframe)
                if gaps:
                    logger.warning(
                        "ingestion.historical.gaps",
                        symbol=symbol,
                        gap_count=len(gaps),
                    )

                count = await self._store_bars(validated)
                total_stored += count

                logger.info(
                    "ingestion.historical.symbol_done",
                    symbol=symbol,
                    bars_stored=count,
                )
            except Exception:
                logger.exception(
                    "ingestion.historical.symbol_error",
                    symbol=symbol,
                    provider=provider.name,
                )

        logger.info(
            "ingestion.historical.complete",
            total_bars=total_stored,
            symbols=symbols,
        )
        return total_stored

    # ------------------------------------------------------------------
    # Real-time loop
    # ------------------------------------------------------------------

    async def _run_realtime_loop(
        self,
        asset_class: AssetClass,
        symbols: list[str],
    ) -> None:
        """Subscribe to the provider and relay bars as events."""
        provider = self.get_provider(asset_class)

        async def _on_bar(raw_bar: RawBar) -> None:
            try:
                bar_dict = normalize_bar(
                    raw_bar, asset_class, provider.name, TimeFrame.M1
                )
                is_valid, errors = validate_bar(bar_dict)
                if not is_valid:
                    logger.warning(
                        "ingestion.realtime.invalid_bar",
                        symbol=raw_bar.symbol,
                        errors=errors,
                    )
                    return

                # Persist to database.
                await self._store_bars([bar_dict])

                # Publish BarCloseEvent.
                bar_event = BarCloseEvent(
                    source_service="data_ingestion",
                    symbol=bar_dict["symbol"],
                    timeframe=bar_dict["timeframe"],
                    open=float(bar_dict["open"]),
                    high=float(bar_dict["high"]),
                    low=float(bar_dict["low"]),
                    close=float(bar_dict["close"]),
                    volume=float(bar_dict["volume"]),
                    vwap=float(bar_dict["vwap"]) if bar_dict.get("vwap") else None,
                    trade_count=bar_dict.get("trade_count"),
                    bar_time=bar_dict["time"],
                )
                await self._event_bus.publish(MARKET_BARS_STREAM, bar_event)

                # Publish PriceUpdateEvent (latest close as current price).
                price_event = PriceUpdateEvent(
                    source_service="data_ingestion",
                    symbol=bar_dict["symbol"],
                    price=float(bar_dict["close"]),
                    volume=float(bar_dict["volume"]),
                    market_timestamp=bar_dict["time"],
                )
                await self._event_bus.publish(MARKET_PRICES_STREAM, price_event)

            except NormalizationError as exc:
                logger.warning(
                    "ingestion.realtime.normalization_error",
                    symbol=raw_bar.symbol,
                    error=str(exc),
                )
            except Exception:
                logger.exception(
                    "ingestion.realtime.processing_error",
                    symbol=raw_bar.symbol,
                )

        try:
            await provider.subscribe_realtime(symbols, _on_bar)
            logger.info(
                "ingestion.realtime.subscribed",
                asset_class=str(asset_class),
                symbols=symbols,
            )
            # Keep task alive until cancelled.
            while self._running:
                await asyncio.sleep(1)
        except NotImplementedError:
            logger.info(
                "ingestion.realtime.not_supported",
                provider=provider.name,
                asset_class=str(asset_class),
            )
        except asyncio.CancelledError:
            logger.info(
                "ingestion.realtime.cancelled",
                asset_class=str(asset_class),
            )
        except Exception:
            logger.exception(
                "ingestion.realtime.fatal",
                asset_class=str(asset_class),
            )

    # ------------------------------------------------------------------
    # Normalisation & validation helpers
    # ------------------------------------------------------------------

    def _normalize_bars(
        self,
        raw_bars: list[RawBar],
        asset_class: AssetClass,
        source: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
        """Normalise a batch of raw bars, dropping any that fail."""
        normalised: list[dict[str, Any]] = []
        for raw in raw_bars:
            try:
                normalised.append(normalize_bar(raw, asset_class, source, timeframe))
            except NormalizationError as exc:
                logger.warning(
                    "ingestion.normalize_error",
                    symbol=raw.symbol,
                    error=str(exc),
                )
        return normalised

    @staticmethod
    def _validate_bars(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only bars that pass validation."""
        valid: list[dict[str, Any]] = []
        for bar in bars:
            is_ok, _ = validate_bar(bar)
            if is_ok:
                valid.append(bar)
        return valid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _store_bars(self, bars: list[dict[str, Any]]) -> int:
        """Bulk upsert bars into the ``ohlcv`` table.

        Uses PostgreSQL ``ON CONFLICT DO NOTHING`` so duplicate bars
        (same time/symbol/timeframe) are silently skipped.

        Returns the number of rows actually inserted.
        """
        if not bars:
            return 0

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            try:
                stmt = pg_insert(OHLCVRecord).values(bars)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["time", "symbol", "timeframe"],
                )
                result = await session.execute(stmt)
                await session.commit()
                inserted = result.rowcount if result.rowcount else 0
                logger.debug("ingestion.store.inserted", count=inserted)
                return inserted
            except Exception:
                await session.rollback()
                logger.exception("ingestion.store.error", bar_count=len(bars))
                raise

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, bool]:
        """Return health status for every registered provider."""
        results: dict[str, bool] = {}
        for asset_class, provider in self._providers.items():
            try:
                results[provider.name] = await provider.health_check()
            except Exception:
                results[provider.name] = False
        return results
