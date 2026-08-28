"""predictions.p_flat_raw -- the raw series must never be unlosable again

Revision ID: 0010_p_flat_raw
Revises: 0009_features_null
Create Date: 2026-08-28

Phase 1's live-calibration verdict (GO_LIVE 2026-08-28) is undiagnosable in
hindsight because the ensemble's raw pre-calibration p(flat) lived only in
in-memory metadata: once the layer transformed the served probabilities, the
raw stated series was gone. This nullable column receives the raw value on
every persisted prediction so no future calibration layer can repeat that
loss. Equals the served flat probability whenever no layer is active.

Guarded with a column check because 0001 creates the whole ORM schema via
``metadata.create_all`` -- on a fresh database the column already exists.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_p_flat_raw"
down_revision = "0009_features_null"
branch_labels = None
depends_on = None

_TABLE = "predictions"
_COLUMN = "p_flat_raw"


def _has_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(_TABLE):
        return True  # nothing to alter
    return any(c["name"] == _COLUMN for c in inspector.get_columns(_TABLE))


def upgrade() -> None:
    if _has_column():
        return
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
