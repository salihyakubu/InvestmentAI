"""Volume-Weighted Average Price (VWAP) execution algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)

# Default intraday volume profile (fraction of daily volume per half-hour).
# U-shaped: more volume at open and close.
DEFAULT_VOLUME_PROFILE: dict[str, float] = {
    "09:30": 0.08,
    "10:00": 0.06,
    "10:30": 0.05,
    "11:00": 0.04,
    "11:30": 0.04,
    "12:00": 0.03,
    "12:30": 0.03,
    "13:00": 0.04,
    "13:30": 0.04,
    "14:00": 0.05,
    "14:30": 0.06,
    "15:00": 0.07,
    "15:30": 0.09,
    "16:00": 0.00,  # market close (no new orders)
}


@dataclass
class VWAPSlice:
    """A single time slice in a VWAP schedule."""

    time_bucket: str  # e.g. "09:30", "10:00"
    time_offset_seconds: float  # seconds from algorithm start
    quantity: Decimal
    volume_weight: float


class VWAPAlgorithm:
    """Splits a large order proportionally to a historical volume profile.

    Higher-volume periods (typically market open and close) receive
    larger slices; quieter midday periods receive smaller slices.
    """

    def __init__(
        self,
        total_quantity: Decimal,
        volume_profile: dict[str, float] | None = None,
    ) -> None:
        self._total_quantity = total_quantity
        self._profile = volume_profile or DEFAULT_VOLUME_PROFILE

    def generate_schedule(self) -> list[VWAPSlice]:
        """Generate the VWAP execution schedule.

        Returns a list of slices weighted by the volume profile.
        """
        # Normalise profile weights so they sum to 1.0.
        total_weight = sum(self._profile.values())
        if total_weight <= 0:
            raise ValueError("Volume profile weights must sum to > 0")

        normalised = {
            bucket: weight / total_weight
            for bucket, weight in self._profile.items()
            if weight > 0
        }

        slices: list[VWAPSlice] = []
        cumulative_qty = Decimal("0")
        bucket_list = sorted(normalised.keys())

        for idx, bucket in enumerate(bucket_list):
            weight = normalised[bucket]

            if idx == len(bucket_list) - 1:
                # Last slice gets the remainder to avoid rounding drift.
                qty = self._total_quantity - cumulative_qty
            else:
                qty = (self._total_quantity * Decimal(str(weight))).quantize(
                    Decimal("0.00000001")
                )
                cumulative_qty += qty

            # Convert bucket time to offset from market open (09:30).
            offset_seconds = self._bucket_to_offset(bucket)

            slices.append(
                VWAPSlice(
                    time_bucket=bucket,
                    time_offset_seconds=offset_seconds,
                    quantity=qty,
                    volume_weight=weight,
                )
            )

        logger.info(
            "vwap_schedule_generated",
            total_quantity=str(self._total_quantity),
            num_slices=len(slices),
        )
        return slices

    @staticmethod
    def _bucket_to_offset(bucket: str) -> float:
        """Convert a time bucket string (HH:MM) to seconds from 09:30."""
        parts = bucket.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        market_open_minutes = 9 * 60 + 30
        bucket_minutes = hours * 60 + minutes
        return float((bucket_minutes - market_open_minutes) * 60)
