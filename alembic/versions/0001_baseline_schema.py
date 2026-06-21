"""baseline schema -- create all ORM tables

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-14

Creates the full ORM schema (the single source of truth) via
``metadata.create_all``. The old hand-written
infrastructure/docker/postgres/init.sql has been retired in favour of this
migration chain.

TimescaleDB hypertable conversions and retention policies are intentionally NOT
applied here so this migration stays portable to plain PostgreSQL. They live in
0002_timescale_hypertables, guarded on the timescaledb extension being available.
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
