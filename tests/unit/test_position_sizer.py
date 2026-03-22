"""Unit tests for position sizing strategies."""

from __future__ import annotations

from decimal import Decimal

import pytest

from services.risk.position_sizer import (
    FixedFractionalSizer,
    KellyCriterionSizer,
)


class TestFixedFractionalSizer:
    """Tests for fixed fractional position sizing."""

    def test_fixed_fractional(self) -> None:
        """Size should be exactly equity * fraction."""
        sizer = FixedFractionalSizer(fraction=0.02)
        equity = Decimal("50000")
        size = sizer.size(equity=equity)

        expected = Decimal("50000") * Decimal("0.02")
        assert size == expected


class TestKellyCriterionSizer:
    """Tests for Kelly criterion position sizing."""

    def test_kelly_positive_edge(self) -> None:
        """With a positive edge, Kelly should return a positive size."""
        sizer = KellyCriterionSizer(kelly_fraction=0.25)
        size = sizer.size(
            win_rate=0.60,
            win_loss_ratio=1.5,
            equity=Decimal("10000"),
            max_pct=0.20,
        )

        # Kelly pct = 0.60 - (0.40 / 1.5) = 0.3333
        # Quarter Kelly = 0.0833
        # Not capped (0.0833 < 0.20)
        assert size > Decimal("0")

        # Verify the exact value.
        kelly_pct = 0.60 - (0.40 / 1.5)
        adjusted = kelly_pct * 0.25
        expected = Decimal("10000") * Decimal(str(adjusted))
        assert pytest.approx(float(size), rel=1e-6) == float(expected)

    def test_kelly_negative_edge_returns_zero(self) -> None:
        """With a negative edge, Kelly should return zero (no bet)."""
        sizer = KellyCriterionSizer(kelly_fraction=0.25)
        size = sizer.size(
            win_rate=0.30,
            win_loss_ratio=0.8,
            equity=Decimal("10000"),
            max_pct=0.20,
        )

        # Kelly pct = 0.30 - (0.70 / 0.8) = 0.30 - 0.875 = -0.575
        # Negative -> floored at zero.
        assert size == Decimal("0")

    def test_kelly_capped_at_max(self) -> None:
        """When Kelly fraction exceeds max_pct, it should be capped."""
        sizer = KellyCriterionSizer(kelly_fraction=1.0)  # Full Kelly
        size = sizer.size(
            win_rate=0.70,
            win_loss_ratio=2.0,
            equity=Decimal("10000"),
            max_pct=0.05,
        )

        # Full Kelly pct = 0.70 - (0.30 / 2.0) = 0.55
        # Capped at 0.05
        expected = Decimal("10000") * Decimal("0.05")
        assert size == expected
