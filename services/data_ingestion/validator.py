"""Data-quality validation for normalised OHLCV bars."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from core.enums import TimeFrame

logger = structlog.get_logger(__name__)

# Expected durations (in seconds) for each canonical timeframe.
_TF_SECONDS: dict[str, int] = {
    TimeFrame.M1: 60,
    TimeFrame.M5: 300,
    TimeFrame.M15: 900,
    TimeFrame.H1: 3_600,
    TimeFrame.H4: 14_400,
    TimeFrame.D1: 86_400,
    TimeFrame.W1: 604_800,
}


def validate_bar(bar: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a normalised bar dict for data-quality issues.

    Returns ``(is_valid, errors)`` where *errors* is a list of human-readable
    descriptions of every issue found.  A bar with one or more errors is
    considered invalid but the caller decides whether to drop or quarantine it.
    """
    errors: list[str] = []

    open_p = bar.get("open")
    high_p = bar.get("high")
    low_p = bar.get("low")
    close_p = bar.get("close")
    volume = bar.get("volume")
    bar_time = bar.get("time")

    # ---- Price positivity ----
    for name, value in [("open", open_p), ("high", high_p), ("low", low_p), ("close", close_p)]:
        if value is not None and value <= 0:
            errors.append(f"{name} must be positive, got {value}")

    # ---- OHLC consistency ----
    if high_p is not None and low_p is not None:
        if high_p < low_p:
            errors.append(f"high ({high_p}) < low ({low_p})")

    if high_p is not None:
        if open_p is not None and high_p < open_p:
            errors.append(f"high ({high_p}) < open ({open_p})")
        if close_p is not None and high_p < close_p:
            errors.append(f"high ({high_p}) < close ({close_p})")

    if low_p is not None:
        if open_p is not None and low_p > open_p:
            errors.append(f"low ({low_p}) > open ({open_p})")
        if close_p is not None and low_p > close_p:
            errors.append(f"low ({low_p}) > close ({close_p})")

    # ---- Volume non-negative ----
    if volume is not None and volume < 0:
        errors.append(f"volume must be non-negative, got {volume}")

    # ---- No future timestamps ----
    if bar_time is not None:
        now_utc = datetime.now(timezone.utc)
        if isinstance(bar_time, datetime) and bar_time > now_utc:
            errors.append(f"bar timestamp {bar_time.isoformat()} is in the future")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(
            "validator.bar_invalid",
            symbol=bar.get("symbol"),
            time=str(bar_time),
            errors=errors,
        )
    return is_valid, errors


def detect_gaps(
    bars: list[dict[str, Any]],
    timeframe: str,
) -> list[tuple[datetime, datetime]]:
    """Identify missing bar intervals in a sorted sequence.

    Parameters
    ----------
    bars:
        A list of normalised bar dicts, assumed sorted ascending by ``time``.
    timeframe:
        One of the canonical :class:`core.enums.TimeFrame` values.

    Returns
    -------
    list[tuple[datetime, datetime]]
        Each element is ``(gap_start, gap_end)`` representing a contiguous
        stretch of missing bars.  An empty list means no gaps were detected.
    """
    expected_delta = _TF_SECONDS.get(timeframe)
    if expected_delta is None:
        logger.warning("validator.detect_gaps.unknown_timeframe", timeframe=timeframe)
        return []

    if len(bars) < 2:
        return []

    gaps: list[tuple[datetime, datetime]] = []
    # Allow up to 50% tolerance for market-hours gaps in daily data.
    tolerance = expected_delta * 1.5

    for i in range(1, len(bars)):
        prev_time: datetime = bars[i - 1]["time"]
        curr_time: datetime = bars[i]["time"]

        delta_seconds = (curr_time - prev_time).total_seconds()
        if delta_seconds > tolerance:
            gaps.append((prev_time, curr_time))

    if gaps:
        logger.info(
            "validator.gaps_detected",
            timeframe=timeframe,
            gap_count=len(gaps),
        )
    return gaps
