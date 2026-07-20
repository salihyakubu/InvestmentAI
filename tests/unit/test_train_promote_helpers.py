"""train_and_promote helpers: overfit-guarded gate + per-symbol feature reference.

The .npz feature-reference writer must reproduce build_dataset's per-symbol
row arithmetic exactly (the continuous-learning drift check compares live
feature rows against these blocks per symbol), and the promotion gate must
reject both sub-floor accuracy and train/val overfit gaps.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_and_promote import (
    FEATURE_REFERENCE_FILENAME,
    MAX_TRAIN_VAL_GAP,
    MODEL_TYPES,
    gate_failures,
    per_symbol_val_rows,
    write_feature_reference,
)
from services.prediction.training.dataset_builder import (
    MIN_BARS,
    MIN_VAL_ACCURACY,
    build_dataset,
)

# ---------------------------------------------------------------------------
# Gate: accuracy floor + overfit guard
# ---------------------------------------------------------------------------


def _result(train_acc: float, val_acc: float) -> SimpleNamespace:
    return SimpleNamespace(train_accuracy=train_acc, val_accuracy=val_acc)


def test_model_types_include_catboost() -> None:
    assert MODEL_TYPES == ("xgboost", "lightgbm", "catboost")


def test_gate_passes_honest_models() -> None:
    results = {t: _result(0.55, 0.45) for t in MODEL_TYPES}
    assert gate_failures(results) == []


def test_gate_rejects_below_accuracy_floor() -> None:
    results = {"xgboost": _result(0.40, MIN_VAL_ACCURACY - 0.01)}
    failures = gate_failures(results)
    assert len(failures) == 1
    assert "xgboost" in failures[0] and "floor" in failures[0]


def test_gate_rejects_overfit_gap_even_above_floor() -> None:
    # Beats the floor but memorizes: train 0.99 vs val 0.40 -> gap 0.59 > 0.35.
    results = {"catboost": _result(0.99, 0.40)}
    failures = gate_failures(results)
    assert len(failures) == 1
    assert "gap" in failures[0]

    # Exactly at the limit is allowed (guard is strict-greater).
    assert gate_failures({"m": _result(0.40 + MAX_TRAIN_VAL_GAP, 0.40)}) == []


def test_gate_reports_both_failures_for_one_model() -> None:
    failures = gate_failures({"m": _result(0.95, 0.20)})
    assert len(failures) == 2


# ---------------------------------------------------------------------------
# Per-symbol validation rows + .npz feature reference
# ---------------------------------------------------------------------------


def _make_cols(n: int, seed: int, start: float = 100.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, size=n)
    close = start * np.cumprod(1.0 + rets)
    high = close * (1.0 + rng.uniform(0.0001, 0.004, size=n))
    low = close * (1.0 - rng.uniform(0.0001, 0.004, size=n))
    open_ = np.concatenate(([start], close[:-1]))
    volume = rng.uniform(1_000.0, 5_000.0, size=n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


@pytest.fixture(scope="module")
def built():
    stride = 5
    per_symbol_cols = {
        "BTC/USDT": _make_cols(MIN_BARS + 105, seed=1),
        "AAPL": _make_cols(MIN_BARS + 60, seed=2),
        "TINY": _make_cols(MIN_BARS - 1, seed=3),  # skipped by build_dataset
    }
    splits = build_dataset(per_symbol_cols, stride=stride)
    return per_symbol_cols, splits, stride


def test_per_symbol_val_rows_reconstruct_pooled_split(built) -> None:
    per_symbol_cols, (_, _, _, va_X, _, _, names), stride = built
    rows = per_symbol_val_rows(per_symbol_cols, va_X, stride)

    # Only symbols above MIN_BARS contribute, in insertion order.
    assert list(rows) == ["BTC/USDT", "AAPL"]
    for arr in rows.values():
        assert arr.dtype == np.float32
        assert arr.shape[1] == len(names)

    # Concatenated per-symbol blocks are exactly the pooled validation rows.
    stacked = np.concatenate(list(rows.values()))
    assert stacked.shape[0] == len(va_X)
    np.testing.assert_allclose(stacked, va_X.astype(np.float32), rtol=1e-6)


def test_per_symbol_val_rows_detects_contract_drift(built) -> None:
    per_symbol_cols, (_, _, _, va_X, _, _, _), stride = built
    with pytest.raises(ValueError, match="reconstruction"):
        per_symbol_val_rows(per_symbol_cols, va_X[:-1], stride)


def test_write_feature_reference_npz_schema(built, tmp_path: Path) -> None:
    per_symbol_cols, (_, _, _, va_X, _, _, names), stride = built
    rows = per_symbol_val_rows(per_symbol_cols, va_X, stride)

    path = write_feature_reference(tmp_path, names, rows)
    assert path == tmp_path / FEATURE_REFERENCE_FILENAME
    assert path.exists()

    with np.load(str(path)) as data:
        assert [str(n) for n in data["feature_names"]] == names
        assert sorted(names) == names  # dataset builder's sorted-name convention
        # Raw symbol keys round-trip, including the slash.
        symbol_keys = [k for k in data.files if k != "feature_names"]
        assert sorted(symbol_keys) == sorted(rows)
        for sym in symbol_keys:
            block = np.asarray(data[sym])
            assert block.ndim == 2 and block.shape[1] == len(names)
            np.testing.assert_allclose(block, rows[sym])
