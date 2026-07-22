"""Alpaca Markets data provider for US equities."""

from __future__ import annotations

import asyncio
import contextlib
import random
import threading
from datetime import UTC, datetime
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.enums import AssetClass, TimeFrame
from services.data_ingestion.providers.base import BaseDataProvider, RawBar, RealtimeCallback

logger = structlog.get_logger(__name__)

# Known-transient websocket connect failures. 'connection limit exceeded'
# happens on every deploy while the previous container still holds the
# stream connection (~13s drain window) — retry with backoff, no traceback.
_TRANSIENT_STREAM_ERRORS = ("connection limit exceeded",)

# Reconnect backoff bounds (seconds).
_STREAM_BACKOFF_BASE_S = 1.0
_STREAM_BACKOFF_CAP_S = 30.0

# Stream-thread shutdown budgets (seconds).
_THREAD_READY_TIMEOUT_S = 5.0
_STOP_WS_TIMEOUT_S = 2.0
_GRACEFUL_STOP_TIMEOUT_S = 6.0  # SDK _consume polls recv with a 5s timeout
_CANCEL_STOP_TIMEOUT_S = 2.0
_THREAD_JOIN_TIMEOUT_S = 15.0


def _is_transient_stream_error(exc: BaseException) -> bool:
    """True for connect errors that resolve on their own (deploy drain)."""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_STREAM_ERRORS)


def _stream_backoff_delay(
    attempt: int,
    *,
    base: float = _STREAM_BACKOFF_BASE_S,
    cap: float = _STREAM_BACKOFF_CAP_S,
    rand: float = 0.5,
) -> float:
    """Delay before reconnect *attempt* (0-based): capped exponential + jitter.

    *rand* is a uniform draw in [0, 1) mapped to a 0.5-1.5x multiplier so
    concurrent reconnects don't synchronise; the cap holds after jitter.
    """
    delay = base * (2.0 ** min(attempt, 32))
    return min(cap, delay * (0.5 + rand))


async def _wait_for_stop(stop: asyncio.Event, delay: float) -> bool:
    """Sleep up to *delay* seconds; return True early if *stop* is set.

    Used for backoff sleeps on the stream thread's loop so shutdown never
    waits out a 30s backoff window.
    """
    if stop.is_set():
        return True
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        return False
    return True


# Mapping from our TimeFrame enum values to Alpaca TimeFrame objects.
# Resolved lazily to avoid import-time failures when alpaca-py is absent.
_ALPACA_TF_MAP: dict[str, Any] | None = None


def _get_alpaca_tf_map() -> dict[str, Any]:
    global _ALPACA_TF_MAP
    if _ALPACA_TF_MAP is None:
        from alpaca.data.timeframe import TimeFrame as AlpacaTF
        from alpaca.data.timeframe import TimeFrameUnit

        _ALPACA_TF_MAP = {
            TimeFrame.M1: AlpacaTF(1, TimeFrameUnit.Minute),
            TimeFrame.M5: AlpacaTF(5, TimeFrameUnit.Minute),
            TimeFrame.M15: AlpacaTF(15, TimeFrameUnit.Minute),
            TimeFrame.H1: AlpacaTF(1, TimeFrameUnit.Hour),
            TimeFrame.H4: AlpacaTF(4, TimeFrameUnit.Hour),
            TimeFrame.D1: AlpacaTF(1, TimeFrameUnit.Day),
            TimeFrame.W1: AlpacaTF(1, TimeFrameUnit.Week),
        }
    return _ALPACA_TF_MAP


class AlpacaDataProvider(BaseDataProvider):
    """Fetches US equity data via the Alpaca Markets v2 API.

    Uses ``alpaca-py`` for both historical REST queries and real-time
    WebSocket streams.

    Stream lifecycle (single-loop discipline)
    -----------------------------------------
    alpaca-py's ``DataStream`` binds its loop inside ``_run_forever`` and its
    ``run()`` calls ``asyncio.run`` internally, so the stream must live
    entirely on ONE event loop. A dedicated daemon thread creates its own
    loop and both constructs and drives the stream there. Bars are handed
    back to the worker's main loop with ``call_soon_threadsafe`` onto an
    ``asyncio.Queue`` owned by the main loop; a consumer task on the main
    loop awaits the service callback, so DB sessions and the event bus are
    only ever touched from the loop that owns them.
    """

    name = "alpaca"
    asset_class = AssetClass.STOCK

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._running = False
        self._symbols: list[str] = []
        # Main-loop side (owned by the loop that called subscribe_realtime).
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._bar_queue: asyncio.Queue[RawBar] | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        # Stream-thread side (created fresh per subscribe cycle).
        self._thread: threading.Thread | None = None
        self._thread_ready = threading.Event()
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._thread_stop: asyncio.Event | None = None
        self._stream: Any | None = None

    # ------------------------------------------------------------------
    # Historical
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[RawBar]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        tf_map = _get_alpaca_tf_map()
        alpaca_tf = tf_map.get(timeframe)
        if alpaca_tf is None:
            raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")

        client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start.astimezone(UTC),
            end=end.astimezone(UTC),
        )

        # The SDK call is synchronous; run in executor to keep the loop free.
        loop = asyncio.get_running_loop()
        bar_set = await loop.run_in_executor(None, client.get_stock_bars, request)

        raw_bars: list[RawBar] = []
        bars = bar_set[symbol] if symbol in bar_set else []
        for bar in bars:
            raw_bars.append(
                RawBar(
                    time=bar.timestamp.replace(tzinfo=UTC)
                    if bar.timestamp.tzinfo is None
                    else bar.timestamp,
                    symbol=symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    vwap=float(bar.vwap) if bar.vwap is not None else None,
                    trade_count=int(bar.trade_count) if bar.trade_count is not None else None,
                )
            )

        logger.info(
            "alpaca.historical.fetched",
            symbol=symbol,
            timeframe=timeframe,
            bar_count=len(raw_bars),
        )
        return raw_bars

    # ------------------------------------------------------------------
    # Real-time
    # ------------------------------------------------------------------

    async def subscribe_realtime(
        self,
        symbols: list[str],
        callback: RealtimeCallback,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            await self.unsubscribe()

        self._symbols = list(symbols)
        self._main_loop = asyncio.get_running_loop()
        self._bar_queue = asyncio.Queue()
        self._consumer_task = asyncio.create_task(
            self._consume_bars(self._bar_queue, callback),
            name="alpaca-bar-consumer",
        )
        self._running = True
        self._thread_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._stream_thread_main,
            name="alpaca-stream",
            daemon=True,
        )
        self._thread.start()
        # Wait for the thread to publish its loop so a subsequent
        # unsubscribe can always signal it (never blocks the main loop).
        ready = await asyncio.to_thread(self._thread_ready.wait, _THREAD_READY_TIMEOUT_S)
        if not ready:
            logger.warning("alpaca.realtime.stream_thread_slow_start")
        logger.info("alpaca.realtime.subscribed", symbols=self._symbols)

    async def unsubscribe(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The thread publishes its loop before running the supervisor;
            # wait for that so the stop signal cannot race a just-started
            # thread. join() runs in an executor to keep the main loop free.
            await asyncio.to_thread(self._thread_ready.wait, _THREAD_READY_TIMEOUT_S)
            loop = self._thread_loop
            stop = self._thread_stop
            if loop is not None and stop is not None:
                try:
                    loop.call_soon_threadsafe(stop.set)
                except RuntimeError:
                    pass  # thread loop already closed on its way out
            await asyncio.to_thread(thread.join, _THREAD_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning("alpaca.realtime.thread_join_timeout")
        self._thread = None
        self._thread_loop = None
        self._thread_stop = None
        self._stream = None
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
            self._consumer_task = None
        self._bar_queue = None
        self._main_loop = None
        logger.info("alpaca.realtime.unsubscribed")

    # -- main-loop side -------------------------------------------------

    async def _consume_bars(
        self,
        queue: asyncio.Queue[RawBar],
        callback: RealtimeCallback,
    ) -> None:
        """Deliver marshalled bars to the service callback on the main loop.

        This task is the ONLY place the service callback runs, so DB
        sessions and the event bus are always driven by the loop that owns
        them. Callback errors are logged and never kill the consumer.
        """
        while True:
            raw = await queue.get()
            try:
                await callback(raw)
            except Exception:
                logger.exception("alpaca.realtime.callback_error", symbol=raw.symbol)

    # -- stream-thread side ---------------------------------------------

    def _create_stream(self) -> Any:
        """Build the SDK stream. Called on the stream thread's own loop.

        Kept as a tiny factory so tests can substitute a fake stream class
        without any network access.
        """
        from alpaca.data.live import StockDataStream

        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

    def _stream_thread_main(self) -> None:
        """Thread target: owns a private event loop for the whole stream.

        The DataStream is constructed AND driven inside this loop —
        construction on the same loop matters because the SDK binds
        internal state at ``__init__``/first-await and ``_run_forever``
        captures ``asyncio.get_running_loop()``.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._thread_loop = loop
        # Created while this loop is current; binds to it on first await.
        self._thread_stop = asyncio.Event()
        self._thread_ready.set()
        try:
            loop.run_until_complete(self._stream_supervisor(self._thread_stop))
        except BaseException:
            # The supervisor is designed never to raise; this is the
            # never-die-silently backstop.
            logger.exception("alpaca.realtime.stream_thread_error")
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                logger.debug("alpaca.realtime.loop_cleanup_error", exc_info=True)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    async def _stream_supervisor(self, stop: asyncio.Event) -> None:
        """Own the DataStream lifecycle on the stream thread's loop.

        Constructs the stream, drives its ``_run_forever`` coroutine, and
        reconnects with capped jittered backoff if the SDK loop ever
        returns or raises (e.g. 'insufficient subscription'), so the
        thread never dies silently. Returns only when *stop* is set.
        """
        attempt = 0
        while not stop.is_set():
            try:
                stream = self._create_stream()
                self._stream = stream
                self._install_stream_backoff(stream, stop)
                stream.subscribe_bars(self._on_stream_bar, *self._symbols)
            except Exception:
                logger.exception("alpaca.realtime.stream_setup_error")
                delay = _stream_backoff_delay(attempt, rand=random.random())
                attempt += 1
                if await _wait_for_stop(stop, delay):
                    return
                continue

            run_task: asyncio.Task[Any] = asyncio.ensure_future(stream._run_forever())
            stop_task: asyncio.Task[Any] = asyncio.ensure_future(stop.wait())
            try:
                await asyncio.wait(
                    {run_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                stop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stop_task

            if stop.is_set():
                await self._shutdown_stream(stream, run_task)
                return

            # The SDK loop ended on its own. Log a single line (the SDK
            # already logged specifics) and reconnect with a fresh stream.
            error: BaseException | None = None
            if not run_task.cancelled():
                error = run_task.exception()
            delay = _stream_backoff_delay(attempt, rand=random.random())
            attempt += 1
            logger.warning(
                "alpaca.realtime.stream_ended",
                error=str(error) if error is not None else None,
                attempt=attempt,
                retry_in_s=round(delay, 1),
            )
            if await _wait_for_stop(stop, delay):
                return

    async def _shutdown_stream(self, stream: Any, run_task: asyncio.Task[Any]) -> None:
        """Gracefully stop the SDK stream, then cancel whatever remains."""
        try:
            await asyncio.wait_for(stream.stop_ws(), timeout=_STOP_WS_TIMEOUT_S)
        except Exception:
            logger.debug("alpaca.realtime.stop_ws_error", exc_info=True)
        _done, pending = await asyncio.wait({run_task}, timeout=_GRACEFUL_STOP_TIMEOUT_S)
        if pending:
            run_task.cancel()
            await asyncio.wait({run_task}, timeout=_CANCEL_STOP_TIMEOUT_S)
        if run_task.done() and not run_task.cancelled():
            _ = run_task.exception()  # retrieve so the loop never warns
        try:
            await asyncio.wait_for(stream.close(), timeout=_STOP_WS_TIMEOUT_S)
        except Exception:
            logger.debug("alpaca.realtime.stream_close_error", exc_info=True)

    async def _on_stream_bar(self, bar: Any) -> None:
        """SDK bar handler — runs on the stream thread's loop.

        Converts the SDK bar and hands it to the main loop thread-safely.
        Never awaits main-loop resources from this thread: that exact
        pattern caused 'Future attached to a different loop' storms and
        poisoned DB sessions in production.
        """
        try:
            raw = RawBar(
                time=bar.timestamp.replace(tzinfo=UTC)
                if bar.timestamp.tzinfo is None
                else bar.timestamp,
                symbol=bar.symbol,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if bar.vwap is not None else None,
                trade_count=int(bar.trade_count) if bar.trade_count is not None else None,
            )
        except Exception:
            logger.exception("alpaca.realtime.bar_parse_error")
            return
        main_loop = self._main_loop
        queue = self._bar_queue
        if main_loop is None or queue is None or main_loop.is_closed():
            return
        try:
            main_loop.call_soon_threadsafe(queue.put_nowait, raw)
        except RuntimeError:
            # Main loop shut down mid-flight; drop the bar.
            logger.debug("alpaca.realtime.main_loop_closed")

    def _install_stream_backoff(self, stream: Any, stop: asyncio.Event) -> None:
        """Wrap the SDK's connect step with jittered exponential backoff.

        alpaca-py's ``_run_forever`` retries ``_start_ws`` immediately
        (``asyncio.sleep(0)``) and logs a full traceback per attempt, so
        any recurring connect failure — transient 'connection limit
        exceeded' during deploy drain, or a persistent auth error on dead
        credentials — storms at 2-3 tracebacks/sec. The SDK exposes no
        backoff hook; wrapping ``_start_ws`` keeps its loop intact while
        ALL connect failures retry here with capped backoff:

        * transient errors: single-line warning per attempt;
        * non-transient errors: traceback logged once per distinct error,
          single-line warnings thereafter — retried under the same cap
          instead of propagating, so the SDK's immediate-retry loop never
          storms and the stream thread never dies.

        Success resets the backoff: ``attempt`` is local, and the SDK
        calls ``_start_ws`` afresh per reconnect. *stop* (bound to the
        stream thread's loop) aborts backoff sleeps promptly on shutdown.
        """
        original_start_ws = stream._start_ws

        async def _start_ws_with_backoff() -> None:
            attempt = 0
            seen_errors: set[str] = set()
            while True:
                try:
                    await original_start_ws()
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Drop the half-open socket so retries don't hold extra
                    # connections against the account limit.
                    try:
                        await stream.close()
                    except Exception:
                        logger.debug("alpaca.realtime.backoff_close_error", exc_info=True)
                    delay = _stream_backoff_delay(attempt, rand=random.random())
                    attempt += 1
                    if _is_transient_stream_error(exc):
                        logger.warning(
                            "alpaca.realtime.transient_connect_error",
                            error=str(exc),
                            attempt=attempt,
                            retry_in_s=round(delay, 1),
                        )
                    else:
                        signature = f"{type(exc).__name__}: {exc}"
                        if signature not in seen_errors:
                            seen_errors.add(signature)
                            logger.error(
                                "alpaca.realtime.connect_error",
                                error=signature,
                                attempt=attempt,
                                retry_in_s=round(delay, 1),
                                exc_info=True,
                            )
                        else:
                            logger.warning(
                                "alpaca.realtime.connect_error_retry",
                                error=signature,
                                attempt=attempt,
                                retry_in_s=round(delay, 1),
                            )
                    if await _wait_for_stop(stop, delay) or not self._running:
                        raise asyncio.CancelledError from None

        stream._start_ws = _start_ws_with_backoff

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest

            client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            # A lightweight call to verify credentials and connectivity.
            # get_stock_latest_bar takes a request object, not a bare symbol.
            request = StockLatestBarRequest(symbol_or_symbols="AAPL")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: client.get_stock_latest_bar(request))
            return True
        except Exception:
            logger.warning("alpaca.health_check.failed", exc_info=True)
            return False
