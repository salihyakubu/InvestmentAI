"""Purged, embargoed chronological K-fold splitting (purged_chrono_folds)."""

from __future__ import annotations

import numpy as np
import pytest

from services.prediction.training.dataset_builder import HORIZON
from services.prediction.training.walk_forward import default_embargo, purged_chrono_folds


def test_default_embargo_formula() -> None:
    # ceil(HORIZON / stride) + 1
    assert default_embargo() == 2  # stride defaults to HORIZON
    assert default_embargo(HORIZON) == 2
    assert default_embargo(1) == HORIZON + 1
    assert default_embargo(2) == int(np.ceil(HORIZON / 2)) + 1


def test_purged_folds_small_case_exact_indices() -> None:
    # n=20, 4 folds -> val blocks [0..4] [5..9] [10..14] [15..19], embargo=2.
    folds = purged_chrono_folds(20, 4, embargo=2)
    assert len(folds) == 4

    expected = [
        # (train, val): train excludes [val_start - 2, val_end + 2]
        (list(range(7, 20)), list(range(0, 5))),
        ([0, 1, 2] + list(range(12, 20)), list(range(5, 10))),
        (list(range(0, 8)) + [17, 18, 19], list(range(10, 15))),
        (list(range(0, 13)), list(range(15, 20))),
    ]
    for (train_idx, val_idx), (exp_train, exp_val) in zip(folds, expected):
        np.testing.assert_array_equal(train_idx, exp_train)
        np.testing.assert_array_equal(val_idx, exp_val)


def test_purged_folds_exclusion_verified_index_by_index() -> None:
    n, emb = 30, 3
    folds = purged_chrono_folds(n, 5, embargo=emb)
    all_val: list[int] = []
    for train_idx, val_idx in folds:
        lo, hi = int(val_idx[0]), int(val_idx[-1])
        for t in train_idx:
            # every train sample sits strictly outside the embargoed val zone
            assert t < lo - emb or t > hi + emb
        # and every sample outside that zone IS in train (nothing over-dropped)
        outside = [i for i in range(n) if i < lo - emb or i > hi + emb]
        np.testing.assert_array_equal(train_idx, outside)
        assert np.all(np.diff(val_idx) == 1)  # val block is contiguous
        all_val.extend(int(v) for v in val_idx)
    # val blocks partition the sample range chronologically
    assert all_val == list(range(n))


def test_purged_folds_zero_embargo_excludes_only_val() -> None:
    for train_idx, val_idx in purged_chrono_folds(12, 3, embargo=0):
        combined = np.sort(np.concatenate([train_idx, val_idx]))
        np.testing.assert_array_equal(combined, np.arange(12))


def test_purged_folds_default_embargo_applied() -> None:
    # None -> default_embargo() == 2 at the default stride.
    got = purged_chrono_folds(20, 4)
    exp = purged_chrono_folds(20, 4, embargo=default_embargo())
    for (g_tr, g_va), (e_tr, e_va) in zip(got, exp):
        np.testing.assert_array_equal(g_tr, e_tr)
        np.testing.assert_array_equal(g_va, e_va)


def test_purged_folds_uneven_blocks_stay_chronological() -> None:
    folds = purged_chrono_folds(10, 3, embargo=1)
    val_sizes = [len(v) for _, v in folds]
    assert val_sizes == [4, 3, 3]  # np.array_split front-loads the remainder
    assert [int(v[0]) for _, v in folds] == [0, 4, 7]


def test_purged_folds_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        purged_chrono_folds(10, 1)
    with pytest.raises(ValueError, match="n_samples"):
        purged_chrono_folds(3, 4)
    with pytest.raises(ValueError, match="embargo"):
        purged_chrono_folds(10, 2, embargo=-1)
    # embargo so wide every candidate train sample is excluded for some fold
    with pytest.raises(ValueError, match="no training samples"):
        purged_chrono_folds(10, 2, embargo=10)
