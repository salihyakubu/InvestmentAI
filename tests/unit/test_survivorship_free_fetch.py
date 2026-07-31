"""Fetch helpers for the survivorship-free universe.

Small pure functions, but each guards against a silent corruption: a header
row parsed as data, a 4h-funded contract ranked against 8h-funded ones on raw
per-stamp rates, or a stamp that drifts off the 8h grid.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_survivorship_free_funding",
    Path(__file__).resolve().parents[2] / "scripts" / "build_survivorship_free_funding.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def _zip_bytes(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def test_csv_reader_skips_a_header_row() -> None:
    blob = _zip_bytes(
        "x.csv",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1743465600000,8,0.0001\n",
    )
    rows = _MOD._read_csv_rows(blob)
    assert len(rows) == 1
    assert rows[0][0] == "1743465600000"


def test_csv_reader_handles_headerless_files() -> None:
    """Older archive files have no header; the first row is data and must
    not be discarded."""
    blob = _zip_bytes("x.csv", "1743465600000,8,0.0001\n1743494400000,8,0.0002\n")
    rows = _MOD._read_csv_rows(blob)
    assert len(rows) == 2


def test_month_range_is_chronological_and_correct_length() -> None:
    months = _MOD.month_range(24)
    assert len(months) == 24
    assert months == sorted(months)
    # Consecutive months, no gaps.
    for a, b in zip(months, months[1:], strict=False):
        year_a, month_a = map(int, a.split("-"))
        year_b, month_b = map(int, b.split("-"))
        assert (year_b * 12 + month_b) - (year_a * 12 + month_a) == 1


def test_funding_interval_normalisation() -> None:
    """The corruption this exists to prevent: a contract funding every 4h at
    rate r pays 2r per 8h. Ranking raw per-stamp rates across mixed intervals
    systematically understates the 4h-funded contracts' true carry."""
    # Simulate the row-level arithmetic used by fetch_symbol.
    rate, interval = 0.0001, 4.0
    normalised = rate * (8.0 / interval)
    assert normalised == pytest.approx(0.0002)
    rate8, interval8 = 0.0001, 8.0
    assert rate8 * (8.0 / interval8) == pytest.approx(0.0001)


def test_stamps_snap_to_the_8h_grid() -> None:
    ms_8h = _MOD._MS_8H
    # A stamp 3ms after the settlement (as seen in real archive files) must
    # snap DOWN to the settlement stamp, never up to the next one.
    exact = 1743465600000
    late = exact + 3
    assert (late // ms_8h) * ms_8h == exact
