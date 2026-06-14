"""ORM schema coherence.

Guards against the previously-broken state where ``orders.prediction_id``
referenced a ``predictions`` table that had no ORM model -- which made the whole
metadata unresolvable (NoReferencedTableError on create_all / sorted_tables) and
meant ``alembic upgrade head`` could never build a consistent schema.
"""

from __future__ import annotations

import core.models  # noqa: F401  -- registers all tables on the metadata
from core.models.base import AsyncBase
from core.models.orders import Order
from core.models.predictions import Prediction


def test_predictions_table_registered() -> None:
    assert "predictions" in AsyncBase.metadata.tables
    assert Prediction.__tablename__ == "predictions"


def test_metadata_foreign_keys_resolve() -> None:
    # sorted_tables forces resolution of every FK dependency; this raised
    # NoReferencedTableError before the Prediction model existed.
    names = {t.name for t in AsyncBase.metadata.sorted_tables}
    assert "predictions" in names
    assert "orders" in names


def test_orders_prediction_fk_targets_predictions() -> None:
    fk_targets = {fk.column.table.name for fk in Order.__table__.foreign_keys}
    assert "predictions" in fk_targets  # orders.prediction_id -> predictions.id
