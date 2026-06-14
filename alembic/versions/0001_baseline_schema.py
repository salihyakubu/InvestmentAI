"""baseline schema -- create all ORM tables

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-14

Creates the full ORM schema (the single source of truth) via
``metadata.create_all``. This supersedes
infrastructure/docker/postgres/init.sql, which is kept only as a local
TimescaleDB convenience.

TimescaleDB hypertable conversions and retention policies (see init.sql) are
intentionally NOT applied here so this migration stays portable to plain
PostgreSQL. Add them in a follow-up migration guarded on the timescaledb
extension when deploying to TimescaleDB.
"""

from __future__ import annotations

import core.models  # noqa: F401  -- import registers every table on the metadata
from alembic import op
from core.models.base import AsyncBase

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    AsyncBase.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    AsyncBase.metadata.drop_all(bind=op.get_bind())
