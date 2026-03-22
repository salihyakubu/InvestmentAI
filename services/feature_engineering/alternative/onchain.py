"""On-chain metrics for crypto assets.

Placeholder implementation that returns neutral/zero values.  In
production these methods would connect to blockchain data providers
such as Glassnode, IntoTheBlock, or direct node RPC endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """Collects on-chain analytics for a given crypto asset."""

    symbol: str

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_active_addresses(self, lookback_days: int = 7) -> dict:
        """Return daily active-address counts.

        In production this would query a blockchain indexer.
        """
        logger.debug(
            "on-chain active_addresses placeholder for %s (lookback=%d)",
            self.symbol,
            lookback_days,
        )
        return {
            "symbol": self.symbol,
            "metric": "active_addresses",
            "value": 0,
            "change_pct": 0.0,
            "lookback_days": lookback_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_exchange_flows(self) -> dict:
        """Return net exchange inflow/outflow.

        Positive net_flow means coins are flowing *into* exchanges
        (potential sell pressure); negative means outflow (accumulation).
        """
        logger.debug("on-chain exchange_flows placeholder for %s", self.symbol)
        return {
            "symbol": self.symbol,
            "metric": "exchange_flows",
            "inflow": 0.0,
            "outflow": 0.0,
            "net_flow": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_mvrv_ratio(self) -> dict:
        """Return the Market-Value-to-Realised-Value ratio.

        MVRV > 3.5 historically indicates overvaluation; < 1.0 indicates
        undervaluation.
        """
        logger.debug("on-chain mvrv_ratio placeholder for %s", self.symbol)
        return {
            "symbol": self.symbol,
            "metric": "mvrv_ratio",
            "value": 1.0,  # neutral default
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_all_metrics(self) -> dict[str, float]:
        """Convenience method that fetches all on-chain features at once.

        Returns a flat dict suitable for merging into a feature vector.
        """
        active = await self.get_active_addresses()
        flows = await self.get_exchange_flows()
        mvrv = await self.get_mvrv_ratio()

        return {
            "onchain_active_addresses": float(active["value"]),
            "onchain_active_addresses_change": float(active["change_pct"]),
            "onchain_exchange_net_flow": float(flows["net_flow"]),
            "onchain_mvrv_ratio": float(mvrv["value"]),
        }
