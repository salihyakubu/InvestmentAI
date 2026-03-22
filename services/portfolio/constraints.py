"""Portfolio constraint definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioConstraints:
    """Configurable constraints applied during portfolio optimisation.

    Defaults mirror the micro-account risk rules defined in
    ``config.settings.Settings``.
    """

    # Per-position weight bounds.
    min_weight: float = 0.0
    max_weight: float = 0.10

    # Sector / asset-class concentration limits.
    max_sector_weight: float = 0.30
    max_asset_class_weight: float = 0.70

    # Position count limits.
    min_positions: int = 1
    max_positions: int = 10

    # Turnover constraint (fraction of portfolio value per rebalance).
    turnover_limit: float = 1.0  # 1.0 = unconstrained

    # Optional per-symbol overrides: symbol -> (min_weight, max_weight).
    symbol_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_optimizer_dict(self) -> dict:
        """Return a flat dict consumable by optimizer ``constraints`` param."""
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "max_sector_weight": self.max_sector_weight,
            "max_asset_class_weight": self.max_asset_class_weight,
            "min_positions": self.min_positions,
            "max_positions": self.max_positions,
            "turnover_limit": self.turnover_limit,
        }
