"""Durable orders and safety state -- what a restart must not forget

Revision ID: 0011_durable_safety
Revises: 0010_p_flat_raw
Create Date: 2026-11-02

The 2026-11-02 fleet review (GO_LIVE.md same date) found that order state and
the safety flags lived only in process memory:

* ``orders.broker_name`` / ``orders.correlation_id``: an Order row is now
  written BEFORE the broker is called, with the adapter it was routed to and
  its risk correlation id, so a restarted worker can ask the venue what became
  of an in-flight order and a redelivered RiskApprovedEvent is skipped instead
  of re-submitted.
* ``fills.external_fill_id`` UNIQUE: a redelivered OrderFilledEvent cannot
  write the same fill twice.
* ``safety_state``: one row per trading mode holding the halt flag, breaker
  state, daily baseline and all-time peak, read before any service starts.

Guarded with column/table checks because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database everything already exists.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0011_durable_safety"
down_revision = "0010_p_flat_raw"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table):
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def _has_table(table: str) -> bool:
    return bool(sa.inspect(op.get_bind()).has_table(table))


def upgrade() -> None:
    order_cols = _columns("orders")
    if order_cols and "broker_name" not in order_cols:
        op.add_column("orders", sa.Column("broker_name", sa.String(20), nullable=True))
    if order_cols and "correlation_id" not in order_cols:
        op.add_column(
            "orders", sa.Column("correlation_id", sa.String(100), nullable=True)
        )
        op.create_index("ix_orders_correlation_id", "orders", ["correlation_id"])

    if _has_table("fills"):
        existing = {
            idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes("fills")
        }
        if "uq_fills_external_fill_id" not in existing:
            op.create_index(
                "uq_fills_external_fill_id",
                "fills",
                ["external_fill_id"],
                unique=True,
            )

    if not _has_table("safety_state"):
        op.create_table(
            "safety_state",
            sa.Column("trading_mode", sa.String(20), primary_key=True),
            sa.Column(
                "halted", sa.Boolean(), nullable=False, server_default=sa.text("false")
            ),
            sa.Column("halt_reason", sa.String(200), nullable=True),
            sa.Column(
                "breaker_state", sa.String(20), nullable=False, server_default="closed"
            ),
            sa.Column("breaker_tripped_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("breaker_reason", sa.String(500), nullable=True),
            sa.Column("day_start_equity", sa.Numeric(20, 8), nullable=True),
            sa.Column("peak_equity", sa.Numeric(20, 8), nullable=True),
            sa.Column("session_date", sa.Date(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    if _has_table("safety_state"):
        op.drop_table("safety_state")
    if _has_table("fills"):
        existing = {
            idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes("fills")
        }
        if "uq_fills_external_fill_id" in existing:
            op.drop_index("uq_fills_external_fill_id", table_name="fills")
    order_cols = _columns("orders")
    if "correlation_id" in order_cols:
        op.drop_index("ix_orders_correlation_id", table_name="orders")
        op.drop_column("orders", "correlation_id")
    if "broker_name" in order_cols:
        op.drop_column("orders", "broker_name")
