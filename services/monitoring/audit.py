"""Audit logging -- persists every significant action and decision trace."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from core.models.audit import AuditLog
from core.models.base import get_async_session_factory

logger = structlog.get_logger(__name__)


class AuditLogger:
    """Writes structured audit records to the ``audit_logs`` table.

    Every trade decision is recorded with a full trace from prediction
    through risk check, sizing, routing, and execution.
    """

    # ------------------------------------------------------------------
    # Generic action logging
    # ------------------------------------------------------------------

    async def log_action(
        self,
        service: str,
        action: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
        decision_trace: dict[str, Any] | None = None,
    ) -> None:
        """Persist a single audit log entry.

        Parameters:
            service: Name of the originating service (e.g. ``execution``).
            action: Short verb describing the action (e.g. ``order.created``).
            entity_type: Category of affected entity (e.g. ``order``).
            entity_id: Identifier of the affected entity.
            details: Arbitrary structured payload.
            decision_trace: Optional full decision trace (prediction -> execution).
        """
        entry = AuditLog(
            timestamp=datetime.now(timezone.utc),
            service=service,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details or {},
            decision_trace=decision_trace,
        )

        session_factory = get_async_session_factory()
        async with session_factory() as session:
            try:
                session.add(entry)
                await session.commit()
                logger.debug(
                    "audit.logged",
                    service=service,
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                )
            except Exception:
                await session.rollback()
                logger.exception(
                    "audit.log_failed",
                    service=service,
                    action=action,
                )

    # ------------------------------------------------------------------
    # Trade-specific decision trace
    # ------------------------------------------------------------------

    async def log_trade_decision(
        self,
        order: dict[str, Any],
        prediction: dict[str, Any],
        risk_check: dict[str, Any],
        final_decision: str,
    ) -> None:
        """Log the full decision chain for a trade.

        The trace captures every stage of the pipeline so that any trade
        can be reconstructed and audited after the fact:

        prediction -> risk check -> sizing -> routing -> execution
        """
        decision_trace = {
            "prediction": prediction,
            "risk_check": risk_check,
            "final_decision": final_decision,
        }

        await self.log_action(
            service="trading_pipeline",
            action="trade.decision",
            entity_type="order",
            entity_id=order.get("order_id", ""),
            details={
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "quantity": order.get("quantity", 0),
                "order_type": order.get("order_type", ""),
            },
            decision_trace=decision_trace,
        )
