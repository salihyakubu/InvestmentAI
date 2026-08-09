"""features_used: JSON 'null' -> SQL NULL

Revision ID: 0009_features_null
Revises: 0008_factor_watch
Create Date: 2026-08-09

Since PR #76 the prediction writer passed features_used=None explicitly for
unsampled predictions, and SQLAlchemy's JSON default (none_as_null=False)
serialized that as JSON 'null' rather than SQL NULL. Consequence: the
feature_rows_persisted counter (COUNT WHERE features_used IS NOT NULL) was
inflated ~5x -- 4,078 "feature-bearing" rows on 2026-08-09 where only 829
carried vectors. The model now declares JSONB(none_as_null=True); this
migration repairs the stored representation. A JSON null carries zero
information, so rewriting it to SQL NULL corrects storage semantics without
touching any recorded measurement.
"""

from __future__ import annotations

from alembic import op

revision = "0009_features_null"
down_revision = "0008_factor_watch"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":  # jsonb comparison is PG-only
        return
    op.execute(
        "UPDATE predictions SET features_used = NULL "
        "WHERE features_used = 'null'::jsonb"
    )


def downgrade() -> None:
    # Irreversible by design: the JSON nulls were a serialization accident,
    # not data.
    pass
