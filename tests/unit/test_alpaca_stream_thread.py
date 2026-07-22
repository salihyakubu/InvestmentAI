"""Single-loop discipline for the Alpaca stream thread.

Production defect (seen twice, with dead AND valid credentials): the
DataStream was constructed on the worker's main event loop but driven via
``asyncio.run`` inside ``asyncio.to_thread``, and bar callbacks (DB
sessions, event bus) executed on the stream's loop — RuntimeError
'attached to a different loop' storms, flooded logs, and poisoned DB
sessions. These tests pin the redesigned lifecycle with a fake DataStream
(no network): construction and run share one thread and one loop, bars are
marshalled to the main loop, stop joins the thread cleanly, and repeated
start/stop cycles work.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import services.data_ingestion.providers.alpaca_provider as alpaca_mod
from services.data_ingestion.providers.alpaca_provider import AlpacaDataProvider
from services.data_ingestion.providers.base import RawBar


def _sdk_bar(symbol: str = "AAPL", close: float = 101.5) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=datetime(2026, 7, 22, 14, 30, tzinfo=UTC),
        symbol=symbol,
        open=100.0,
        high=102.0,
        low=99.5,
        close=close,
        volume=1_000.0,
        vwap=100.9,
        trade_count=42,
    )


class FakeDataStream:
    """Stands in for alpaca-py's StockDataStream; captures loop identity.

    ``__init__`` calls ``asyncio.get_running_loop()`` so construction
    outside a running loop fails loudly — exactly the discipline the
    provider must uphold.
    """

    def __init__(self, bars_to_emit: list[SimpleNamespace] | None = None) -> None:
        self.init_thread = threading.current_thread()
        self.init_loop = asyncio.get_running_loop()
        self.run_thread: threading.Thread | None = None
        self.run_loop: asyncio.AbstractEventLoop | None = None
        self.handler_thread: threading.Thread | None = None
        self.bar_handler: Any = None
        self.subscribed_symbols: tuple[str, ...] = ()
        self.stop_ws_calls = 0
        self.close_calls = 0
        self._bars_to_emit = list(bars_to_emit or [])
        self._stopped = asyncio.Event()

    async def _start_ws(self) -> None:  # replaced by the backoff wrapper
        pass

    def subscribe_bars(self, handler: Any, *symbols: str) -> None:
        self.bar_handler = handler
        self.subscribed_symbols = symbols

    async def _run_forever(self) -> None:
        self.run_thread = threading.current_thread()
        self.run_loop = asyncio.get_running_loop()
        for bar in self._bars_to_emit:
            self.handler_thread = threading.current_thread()
            await self.bar_handler(bar)
        await self._stopped.wait()

    async def stop_ws(self) -> None:
        self.stop_ws_calls += 1
        self._stopped.set()

    async def close(self) -> None:
        self.close_calls += 1


def _make_provider(
    streams: list[FakeDataStream],
    bars: list[SimpleNamespace] | None = None,
    stream_cls: type[FakeDataStream] = FakeDataStream,
) -> AlpacaDataProvider:
    provider = AlpacaDataProvider(api_key="k", secret_key="s")

    def _create() -> FakeDataStream:
        stream = stream_cls(bars_to_emit=bars)
        streams.append(stream)
        return stream

    provider._create_stream = _create  # type: ignore[method-assign]
    return provider


async def _eventually(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            pytest.fail("condition not met within timeout")
        await asyncio.sleep(0.01)


async def _noop_callback(_raw: RawBar) -> None:
    pass


@pytest.mark.asyncio
async def test_stream_constructed_and_run_on_same_thread_and_loop() -> None:
    streams: list[FakeDataStream] = []
    provider = _make_provider(streams)

    await provider.subscribe_realtime(["AAPL", "MSFT"], _noop_callback)
    try:
        await _eventually(lambda: streams and streams[0].run_loop is not None)
        stream = streams[0]
        # Construction and run share ONE thread and ONE loop...
        assert stream.init_thread is stream.run_thread
        assert stream.init_loop is stream.run_loop
        # ...and that thread/loop is NOT the worker's main one.
        assert stream.init_thread is not threading.current_thread()
        assert stream.init_loop is not asyncio.get_running_loop()
        assert stream.subscribed_symbols == ("AAPL", "MSFT")
    finally:
        await provider.unsubscribe()


@pytest.mark.asyncio
async def test_bars_are_delivered_on_the_main_loop() -> None:
    streams: list[FakeDataStream] = []
    received: list[tuple[RawBar, asyncio.AbstractEventLoop, threading.Thread]] = []

    async def _callback(raw: RawBar) -> None:
        received.append((raw, asyncio.get_running_loop(), threading.current_thread()))

    provider = _make_provider(streams, bars=[_sdk_bar()])
    await provider.subscribe_realtime(["AAPL"], _callback)
    try:
        await _eventually(lambda: received)
        raw, loop, thread = received[0]
        # The service callback runs on the main loop, in the main thread.
        assert loop is asyncio.get_running_loop()
        assert thread is threading.current_thread()
        # The SDK handler itself fired on the stream thread — the bar was
        # marshalled across, not awaited cross-loop.
        assert streams[0].handler_thread is not None
        assert streams[0].handler_thread is not threading.current_thread()
        # Payload survives the crossing intact.
        assert raw.symbol == "AAPL"
        assert raw.close == 101.5
        assert raw.vwap == 100.9
        assert raw.trade_count == 42
        assert raw.time.tzinfo is not None
    finally:
        await provider.unsubscribe()


@pytest.mark.asyncio
async def test_unsubscribe_joins_thread_and_supports_restart() -> None:
    streams: list[FakeDataStream] = []
    provider = _make_provider(streams)

    for cycle in range(2):
        await provider.subscribe_realtime(["AAPL"], _noop_callback)
        await _eventually(
            lambda: len(streams) == cycle + 1 and streams[cycle].run_loop is not None
        )
        thread = provider._thread
        assert thread is not None and thread.is_alive()

        await provider.unsubscribe()

        assert not thread.is_alive()  # joined, not leaked
        assert provider._thread is None
        assert provider._consumer_task is None
        assert provider._bar_queue is None
        assert streams[cycle].stop_ws_calls >= 1  # graceful stop signalled

    # Each cycle built a fresh stream on a fresh loop.
    assert len(streams) == 2
    assert streams[0].run_loop is not streams[1].run_loop


@pytest.mark.asyncio
async def test_unsubscribe_without_subscribe_is_noop() -> None:
    provider = AlpacaDataProvider(api_key="k", secret_key="s")
    await provider.unsubscribe()  # must not raise or hang


@pytest.mark.asyncio
async def test_supervisor_reconnects_when_stream_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SDK loop returns/raises on its own, the thread reconnects
    with a NEW stream on the SAME thread loop instead of dying silently."""
    monkeypatch.setattr(alpaca_mod, "_stream_backoff_delay", lambda *a, **k: 0.0)
    streams: list[FakeDataStream] = []

    class EndingStream(FakeDataStream):
        async def _run_forever(self) -> None:
            self.run_thread = threading.current_thread()
            self.run_loop = asyncio.get_running_loop()
            if len(streams) == 1:
                raise ValueError("insufficient subscription")
            await self._stopped.wait()

    provider = _make_provider(streams, stream_cls=EndingStream)
    await provider.subscribe_realtime(["AAPL"], _noop_callback)
    try:
        await _eventually(lambda: len(streams) >= 2 and streams[1].run_loop is not None)
        # Replacement stream still honours single-loop discipline.
        assert streams[1].init_loop is streams[1].run_loop
        assert streams[1].run_loop is streams[0].run_loop  # same thread loop
    finally:
        await provider.unsubscribe()
