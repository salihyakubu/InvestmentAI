"""Time-Weighted Average Price (TWAP) execution algorithm."""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TWAPSlice:
    """A single time slice in a TWAP schedule."""

    time_offset_seconds: float  # seconds from algorithm start
    quantity: Decimal


class TWAPAlgorithm:
    """Splits a large order into equal time slices with random jitter.

    The order is divided into *num_slices* pieces spread evenly over
    *duration_minutes*, with a small random jitter (+/- 10% of interval)
    applied to each slice time to reduce predictability.
    """

    def __init__(
        self,
        total_quantity: Decimal,
        duration_minutes: int,
        num_slices: int = 10,
    ) -> None:
        if num_slices < 1:
            raise ValueError("num_slices must be >= 1")
        if duration_minutes < 1:
            raise ValueError("duration_minutes must be >= 1")

        self._total_quantity = total_quantity
        self._duration_seconds = duration_minutes * 60
        self._num_slices = num_slices

    def generate_schedule(self) -> list[TWAPSlice]:
        """Generate the TWAP execution schedule.

        Returns a list of (time_offset, quantity) slices ordered by time.
        """
        interval = self._duration_seconds / self._num_slices
        base_qty = self._total_quantity / Decimal(str(self._num_slices))
        remainder = self._total_quantity - base_qty * self._num_slices

        slices: list[TWAPSlice] = []
        for i in range(self._num_slices):
            # Base time offset for this slice.
            base_time = interval * i

            # Add random jitter: +/- 10% of interval.
            jitter = random.uniform(-0.10, 0.10) * interval
            time_offset = max(0.0, base_time + jitter)

            # Last slice picks up any rounding remainder.
            qty = base_qty + (remainder if i == self._num_slices - 1 else Decimal("0"))

            slices.append(TWAPSlice(time_offset_seconds=time_offset, quantity=qty))

        # Sort by time in case jitter reordered adjacent slices.
        slices.sort(key=lambda s: s.time_offset_seconds)

        logger.info(
            "twap_schedule_generated",
            total_quantity=str(self._total_quantity),
            num_slices=self._num_slices,
            duration_seconds=self._duration_seconds,
        )
        return slices
