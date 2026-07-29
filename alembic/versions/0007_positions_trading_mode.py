"""positions.trading_mode -- scope the durable position book to its account

Revision ID: 0007_positions_mode
Revises: 0006_backtest_jobs
Create Date: 2026-07-28

The paper broker's book is now checkpointed into ``positions`` so a restart
stops rebasing equity to initial capital. ``portfolio_snapshots`` already
carries ``trading_mode`` in its primary key; ``positions`` did not, so a live
worker sharing this database would restore the paper book onto the live
broker (and the reverse). The column closes that hole before live is ever
switched on.

Existing rows predate the checkpoint writer and belong to the paper soak, so
they backfill to 'paper'.

Guarded with a column check because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database the column is already present.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007_positions_mode"
down_revision = "0006_backtest_jobs"
branch_labels = None
depends_on = None

_TABLE = "positions"
_COLUMN = "trading_mode"


def _has_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(_TABLE):
        return True  # nothing to alter
    return any(c["name"] == _COLUMN for c in inspector.get_columns(_TABLE))


def upgrade() -> None:
    if _has_column():
        return
    op.add_column(
        _TABLE,
        sa.Column(
            _COLUMN,
            sa.String(20),
            nullable=False,
            server_default="paper",
        ),
    )
    op.create_index(
        "ix_positions_trading_mode_open",
        _TABLE,
        [_COLUMN, "closed_at"],
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(_TABLE):
        return
    if any(c["name"] == _COLUMN for c in inspector.get_columns(_TABLE)):
        op.drop_index("ix_positions_trading_mode_open", table_name=_TABLE)
        op.drop_column(_TABLE, _COLUMN)
