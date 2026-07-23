"""aux_market_state -- slow-moving market-regime metrics for as-of replay

Revision ID: 0005_aux_market
Revises: 0004_pred_outcomes
Create Date: 2026-07-23

Stores timestamped market-regime observations (crypto funding rate, Crypto
Fear & Greed index, VIX close, SPY daily return) so the feature pipeline can
look them up AS-OF any past bar time. The live ingestion service upserts the
latest values here and mirrors them into an in-memory snapshot; training replay
reads this table through a HistoricalAuxProvider to reproduce, without
look-ahead, the value the live snapshot held at each historical window-end.

Guarded with has_table because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database this table already exists by the
time this migration runs, and on pre-existing deployments it does not.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005_aux_market"
down_revision = "0004_pred_outcomes"
branch_labels = None
depends_on = None

_TABLE = "aux_market_state"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        return  # fresh database: 0001's metadata.create_all already made it
    op.create_table(
        _TABLE,
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False, server_default=""),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("time", "metric", "symbol"),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        op.drop_table(_TABLE)
