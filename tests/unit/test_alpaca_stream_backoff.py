"""Alpaca websocket reconnect backoff.

alpaca-py's internal run loop retries the connect step immediately and logs a
full traceback per attempt, so the deploy-drain 'connection limit exceeded'
window storms at 2-3 tracebacks/sec. The provider wraps the connect step with
jittered exponential backoff; these tests pin the delay math (pure function,
no network) and the wrapper's retry/raise behaviour.
"""

from __future__ import annotations

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


def _provider_with_wrapped_stream(
    monkeypatch: pytest.MonkeyPatch, failures: list[Exception]
) -> tuple[AlpacaDataProvider, _FakeStream, list[dict[str, Any]]]:
    monkeypatch.setattr(alpaca_mod, "_stream_backoff_delay", lambda *a, **k: 0.0)
    warnings: list[dict[str, Any]] = []

    class _Recorder:
        def warning(self, event: str, **kwargs: Any) -> None:
            warnings.append({"event": event, **kwargs})

        def debug(self, event: str, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(alpaca_mod, "logger", _Recorder())
    provider = AlpacaDataProvider(api_key="k", secret_key="s")
    provider._running = True
    stream = _FakeStream(failures)
    provider._install_stream_backoff(stream)
    return provider, stream, warnings


@pytest.mark.asyncio
async def test_transient_failures_retry_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[Exception] = [
        ValueError("connection limit exceeded"),
        ValueError("connection limit exceeded"),
    ]
    _provider, stream, warnings = _provider_with_wrapped_stream(monkeypatch, failures)

    await stream._start_ws()

    assert stream.start_calls == 3  # two transient failures, then success
    assert stream.close_calls == 2  # half-open socket dropped before each retry
    assert [w["event"] for w in warnings] == ["alpaca.realtime.transient_connect_error"] * 2
    # Single-line warning: the error is a string field, never a traceback.
    assert all("exc_info" not in w for w in warnings)


@pytest.mark.asyncio
async def test_non_transient_error_propagates_to_sdk_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider, stream, warnings = _provider_with_wrapped_stream(
        monkeypatch, [ValueError("auth failed")]
    )

    with pytest.raises(ValueError, match="auth failed"):
        await stream._start_ws()

    assert stream.start_calls == 1
    assert warnings == []  # no backoff warning for non-transient errors
