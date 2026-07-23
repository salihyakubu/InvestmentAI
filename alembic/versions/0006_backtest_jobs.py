"""backtest_jobs -- background backtest jobs behind the dashboard's page

Revision ID: 0006_backtest_jobs
Revises: 0005_aux_market
Create Date: 2026-07-23

The Backtesting page was a dead 501 stub; backtests now run as background
jobs in the API process with this table as the job's source of truth (the UI
polls status and fetches the JSON result on completion).

Guarded with has_table because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database this table already exists by
the time this migration runs.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0006_backtest_jobs"
down_revision = "0005_aux_market"
branch_labels = None
depends_on = None

_TABLE = "backtest_jobs"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        return  # fresh database: 0001's metadata.create_all already made it
    op.create_table(
        _TABLE,
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        op.drop_table(_TABLE)
