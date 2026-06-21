"""timescale hypertables + retention policies

Revision ID: 0002_timescale
Revises: 0001_baseline
Create Date: 2026-06-21

Converts the time-series tables to TimescaleDB hypertables and adds retention
policies -- the setup that used to live in infrastructure/docker/postgres/init.sql,
now expressed as a proper, version-controlled migration.

Guarded on the timescaledb extension being *available*: on plain PostgreSQL (or
SQLite in tests) the whole migration is a safe no-op, so the baseline schema from
0001 stays fully usable without TimescaleDB.

Eligible tables -- their time column must be part of the primary key, which is
what create_hypertable requires:
  - ohlcv               (time)        retention 5y
  - portfolio_snapshots (time)        no retention (keep full equity history)
  - risk_metrics        (time)        retention 3y
  - audit_logs          (timestamp)   retention 7y  [PK widened to (id, timestamp)]

predictions is intentionally left a plain table: its ORM model uses a simple id
primary key so orders.prediction_id can reference it (a regular table cannot
FK-reference a hypertable), and prediction volume is low enough that partitioning
is optional.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_timescale"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

# (table, time_column, retention_interval | None). Table/column names are fixed
# literals from the ORM, not user input.
_HYPERTABLES: list[tuple[str, str, str | None]] = [
    ("ohlcv", "time", "5 years"),
    ("portfolio_snapshots", "time", None),
    ("risk_metrics", "time", "3 years"),
    ("audit_logs", "timestamp", "7 years"),
]


def _timescale_available(conn: sa.engine.Connection) -> bool:
    """True only on PostgreSQL with the timescaledb extension installed."""
    if conn.dialect.name != "postgresql":
        return False
    return bool(
        conn.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).scalar()
    )


def upgrade() -> None:
    conn = op.get_bind()
    if not _timescale_available(conn):
        return  # plain PostgreSQL / SQLite -> nothing to do

    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))

    # audit_logs needs its partitioning column inside the primary key before it
    # can become a hypertable. It is append-only and nothing FK-references it, so
    # widening the PK is safe.
    conn.execute(sa.text("ALTER TABLE audit_logs DROP CONSTRAINT IF EXISTS audit_logs_pkey"))
    conn.execute(sa.text("ALTER TABLE audit_logs ADD PRIMARY KEY (id, timestamp)"))

    for table, time_col, retention in _HYPERTABLES:
        # No migrate_data: the baseline tables are empty, and migrate_data=>TRUE
        # cannot run inside a transaction block.
        conn.execute(
            sa.text(
                f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE)"
            )
        )
        if retention is not None:
            conn.execute(
                sa.text(
                    f"SELECT add_retention_policy('{table}', INTERVAL '{retention}', "
                    "if_not_exists => TRUE)"
                )
            )


def downgrade() -> None:
    conn = op.get_bind()
    if not _timescale_available(conn):
        return
    # Retention policies are reversible. The hypertable conversion itself is
    # forward-only -- to fully revert, drop and recreate the tables via 0001's
    # downgrade.
    for table, _time_col, retention in _HYPERTABLES:
        if retention is not None:
            conn.execute(
                sa.text(f"SELECT remove_retention_policy('{table}', if_exists => TRUE)")
            )
