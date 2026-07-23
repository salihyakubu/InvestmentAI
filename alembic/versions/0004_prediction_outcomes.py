"""predictions -- durable outcome columns for learning-state rehydration

Revision ID: 0004_pred_outcomes
Revises: 0003_model_blobs
Create Date: 2026-07-23

The continuous-learning loop kept tracked predictions, resolved outcomes, and
live-accuracy history in memory only, so every deploy reset the evaluator's
window and the drift detector's >=1000-resolved-outcome gate to zero. These
columns let the outcome resolver write each realised outcome back to its
predictions row, and let the service rehydrate recent resolved rows on
startup so drift/evaluation statistics survive deploys.

Guarded per-column with the inspector because 0001 creates the whole ORM
schema via ``metadata.create_all`` -- on a fresh database the predictions
table already has these columns by the time this migration runs; on
pre-existing deployments it does not.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_pred_outcomes"
down_revision = "0003_model_blobs"
branch_labels = None
depends_on = None

_TABLE = "predictions"
_INDEX = "ix_predictions_event_id"
_COLUMNS = {
    "event_id": sa.Column("event_id", sa.String(length=64), nullable=True),
    "actual_direction": sa.Column("actual_direction", sa.String(length=10), nullable=True),
    "actual_return": sa.Column("actual_return", sa.Float(), nullable=True),
    "resolved_at": sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {c["name"] for c in inspector.get_columns(_TABLE)}
    for name, column in _COLUMNS.items():
        if name not in existing:
            op.add_column(_TABLE, column)

    index_names = {ix["name"] for ix in inspector.get_indexes(_TABLE)}
    if _INDEX not in index_names:
        op.create_index(_INDEX, _TABLE, ["event_id"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    index_names = {ix["name"] for ix in inspector.get_indexes(_TABLE)}
    if _INDEX in index_names:
        op.drop_index(_INDEX, table_name=_TABLE)
    existing = {c["name"] for c in inspector.get_columns(_TABLE)}
    for name in _COLUMNS:
        if name in existing:
            op.drop_column(_TABLE, name)
