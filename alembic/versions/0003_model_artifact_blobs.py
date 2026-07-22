"""model_artifact_blobs -- durable mirror for cloud-retrained model artifacts

Revision ID: 0003_model_blobs
Revises: 0002_timescale
Create Date: 2026-07-21

The deploy target (Railway) has no volumes: model versions promoted by the
cloud retrainer live only on the container filesystem, so the next deploy
silently reverts serving to the repo's champion artifacts while model_metadata
still advertises the lost version as active. This table mirrors every file of
a promoted version (a few MB of joblib files per version) so worker startup
can replay them onto disk before models are loaded.

Guarded with has_table because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database this table already exists by
the time this migration runs, and on pre-existing deployments it does not.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003_model_blobs"
down_revision = "0002_timescale"
branch_labels = None
depends_on = None

_TABLE = "model_artifact_blobs"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        return  # fresh database: 0001's metadata.create_all already made it
    op.create_table(
        _TABLE,
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_name", "version", "filename"),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(_TABLE):
        op.drop_table(_TABLE)
