"""Feature registry for tracking metadata about computed features.

The registry is a lightweight catalogue that maps feature names to
their compute functions, categories, and human-readable descriptions.
It is used by the service layer to discover available features and by
monitoring / documentation tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a single registered feature."""

    name: str
    compute_fn: Callable[..., Any]
    category: str
    description: str = ""


class FeatureRegistry:
    """In-memory feature catalogue."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureMeta] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        compute_fn: Callable[..., Any],
        category: str,
        description: str = "",
    ) -> None:
        """Register a feature.

        Parameters
        ----------
        name        : unique feature name (e.g. ``"rsi_14"``)
        compute_fn  : callable that produces this feature's value
        category    : grouping key (``"technical"``, ``"fundamental"``, etc.)
        description : optional human-readable explanation
        """
        self._features[name] = FeatureMeta(
            name=name,
            compute_fn=compute_fn,
            category=category,
            description=description,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> FeatureMeta | None:
        """Look up a single feature by name."""
        return self._features.get(name)

    def get_all(self) -> list[FeatureMeta]:
        """Return metadata for every registered feature."""
        return list(self._features.values())

    def get_by_category(self, category: str) -> list[FeatureMeta]:
        """Return features belonging to *category*."""
        return [f for f in self._features.values() if f.category == category]

    def names(self) -> list[str]:
        """Return all registered feature names."""
        return list(self._features.keys())

    def categories(self) -> list[str]:
        """Return the distinct set of registered categories."""
        return sorted({f.category for f in self._features.values()})

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features
