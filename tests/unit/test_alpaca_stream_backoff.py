"""Alpaca websocket reconnect backoff.

alpaca-py's internal run loop retries the connect step immediately and logs a
full traceback per attempt, so any recurring connect failure — the
deploy-drain 'connection limit exceeded' window or a persistent auth error on
dead credentials — storms at 2-3 tracebacks/sec. The provider wraps the
connect step with jittered exponential backoff; these tests pin the delay
math (pure function, no network) and the wrapper's retry behaviour for both
transient and non-transient errors.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import services.data_ingestion.providers.alpaca_provider as alpaca_mod
from services.data_ingestion.providers.alpaca_provider import (
    _STREAM_BACKOFF_CAP_S,
    AlpacaDataProvider,
    _is_transient_stream_error,
    _stream_backoff_delay,
)

# ---------------------------------------------------------------------------
# Delay math
# ---------------------------------------------------------------------------


def test_backoff_doubles_from_base_then_caps() -> None:
    """rand=0.5 is the neutral 1.0x multiplier: 1, 2, 4, 8, 16, 30, 30..."""
    delays = [_stream_backoff_delay(attempt, rand=0.5) for attempt in range(8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


def test_backoff_jitter_spans_half_to_full_range() -> None:
    # rand=0 -> 0.5x, rand->1 -> 1.5x (still capped).
    assert _stream_backoff_delay(0, rand=0.0) == 0.5
    assert _stream_backoff_delay(0, rand=1.0) == 1.5
    assert _stream_backoff_delay(2, rand=0.0) == 2.0  # 4s * 0.5


def test_backoff_never_exceeds_cap_even_with_max_jitter() -> None:
    for attempt in range(64):
        assert _stream_backoff_delay(attempt, rand=1.0) <= _STREAM_BACKOFF_CAP_S


def test_backoff_huge_attempt_does_not_overflow() -> None:
    """Hours of retries must not overflow 2.0**attempt into OverflowError."""
    assert _stream_backoff_delay(10_000, rand=0.5) == _STREAM_BACKOFF_CAP_S


def test_transient_error_detection() -> None:
    assert _is_transient_stream_error(ValueError("connection limit exceeded (429)"))
    assert _is_transient_stream_error(ValueError("Connection Limit Exceeded"))
    assert not _is_transient_stream_error(ValueError("auth failed"))
    assert not _is_transient_stream_error(RuntimeError("boom"))


# ---------------------------------------------------------------------------
# Wrapper behaviour (fake stream, zero delays, no network)
# ---------------------------------------------------------------------------


class _FakeStream:
    """Stands in for alpaca-py's DataStream: a _start_ws attr and close()."""

    def __init__(self, failures: list[Exception]) -> None:
        self._failures = failures
        self.start_calls = 0
        self.close_calls = 0
        self._start_ws = self._real_start_ws

    async def _real_start_ws(self) -> None:
        self.start_calls += 1
        if self._failures:
            raise self._failures.pop(0)

    async def close(self) -> None:
        self.close_calls += 1


class _Recorder:
    """Captures structlog-style calls as (level, event, kwargs) dicts."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def warning(self, event: str, **kwargs: Any) -> None:
        self._records.append({"level": "warning", "event": event, **kwargs})

    def error(self, event: str, **kwargs: Any) -> None:
        self._records.append({"level": "error", "event": event, **kwargs})

    def debug(self, event: str, **kwargs: Any) -> None:
        pass


def _provider_with_wrapped_stream(
    monkeypatch: pytest.MonkeyPatch,
    failures: list[Exception],
    delay_attempts: list[int] | None = None,
) -> tuple[AlpacaDataProvider, _FakeStream, asyncio.Event, list[dict[str, Any]]]:
    def _zero_delay(attempt: int, **_kwargs: Any) -> float:
        if delay_attempts is not None:
            delay_attempts.append(attempt)
        return 0.0

    monkeypatch.setattr(alpaca_mod, "_stream_backoff_delay", _zero_delay)
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(alpaca_mod, "logger", _Recorder(records))
    provider = AlpacaDataProvider(api_key="k", secret_key="s")
    provider._running = True
    stop = asyncio.Event()
    stream = _FakeStream(failures)
    provider._install_stream_backoff(stream, stop)
    return provider, stream, stop, records


@pytest.mark.asyncio
async def test_transient_failures_retry_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[Exception] = [
        ValueError("connection limit exceeded"),
        ValueError("connection limit exceeded"),
    ]
    _provider, stream, _stop, records = _provider_with_wrapped_stream(monkeypatch, failures)

    await stream._start_ws()

    assert stream.start_calls == 3  # two transient failures, then success
    assert stream.close_calls == 2  # half-open socket dropped before each retry
    assert [r["event"] for r in records] == ["alpaca.realtime.transient_connect_error"] * 2
    assert all(r["level"] == "warning" for r in records)
    # Single-line warning: the error is a string field, never a traceback.
    assert all("exc_info" not in r for r in records)


@pytest.mark.asyncio
async def test_backoff_attempt_sequence_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper feeds 0-based increasing attempts into the delay curve."""
    failures: list[Exception] = [
        ValueError("connection limit exceeded"),
        ValueError("connection limit exceeded"),
        ValueError("connection limit exceeded"),
    ]
    delay_attempts: list[int] = []
    _provider, stream, _stop, _records = _provider_with_wrapped_stream(
        monkeypatch, failures, delay_attempts=delay_attempts
    )

    await stream._start_ws()

    assert delay_attempts == [0, 1, 2]
    # A fresh _start_ws call (SDK reconnect after success) resets the backoff.
    stream._failures = [ValueError("connection limit exceeded")]
    await stream._start_ws()
    assert delay_attempts == [0, 1, 2, 0]


@pytest.mark.asyncio
async def test_non_transient_error_retries_with_backoff_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead credentials must not storm tracebacks or kill the stream thread.

    Non-transient connect errors are retried under the same capped backoff:
    the traceback is logged once per distinct error, single-line warnings
    thereafter — never propagated into the SDK's immediate-retry loop.
    """
    failures: list[Exception] = [
        ValueError("auth failed"),
        ValueError("auth failed"),
        ValueError("auth failed"),
    ]
    _provider, stream, _stop, records = _provider_with_wrapped_stream(monkeypatch, failures)

    await stream._start_ws()

    assert stream.start_calls == 4  # three failures, then success
    assert stream.close_calls == 3
    events = [(r["level"], r["event"]) for r in records]
    assert events == [
        ("error", "alpaca.realtime.connect_error"),
        ("warning", "alpaca.realtime.connect_error_retry"),
        ("warning", "alpaca.realtime.connect_error_retry"),
    ]
    # Traceback exactly once (first occurrence); retries are single-line.
    assert records[0].get("exc_info") is True
    assert all("exc_info" not in r for r in records[1:])


@pytest.mark.asyncio
async def test_stop_event_aborts_backoff_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown during a backoff sleep cancels promptly instead of retrying."""
    _provider, stream, stop, _records = _provider_with_wrapped_stream(
        monkeypatch, [ValueError("connection limit exceeded")]
    )
    stop.set()

    with pytest.raises(asyncio.CancelledError):
        await stream._start_ws()

    assert stream.start_calls == 1  # no further connect attempts after stop
