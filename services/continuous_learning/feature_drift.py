"""Feature drift: are the models seeing the world they were judged on?

The learning-metrics instrument found promotions that do not transfer (era 4
live IC significantly negative) and recorded two candidate explanations --
regime shift vs genuine degradation -- with no way to tell them apart. This
instrument supplies the missing measurement: Population Stability Index per
served feature, computed from the vectors persisted since Phase 2a (PR #76).

Two design rules exist because the adversarial review of this very module
proved the naive versions wrong (2026-08-09):

* PER SYMBOL, never pooled. The served vector carries raw price-scale
  features (SMA/ATR/MACD in price units), so BTC rows and ADA rows sit
  orders of magnitude apart; a pooled "distribution" is a mixture whose PSI
  measures symbol row-share, not input drift. Each symbol is compared only
  against its own history; the per-feature number is the mean across
  measurably drifting symbols, with the worst (symbol, feature) called out.
* FIXED EARLY REFERENCE. The reference is the first REFERENCE_DAYS of the
  current pipeline generation, not "everything before the recent window" --
  an expanding reference absorbs the drifted past and attenuates real drift
  (measured: a 2-sigma/30-day drift scores ~0.8 expanding vs ~2.8 fixed).

Honest limitations, stated where the numbers are made:

* The reference is the platform's OWN early serving distribution, NOT the
  training distribution -- training-time feature distributions were never
  persisted. Drift here means "the inputs moved relative to the start of
  the persisted record", which brackets but does not equal train-serve skew.
* Only rows whose ordering hash matches the most recent generation are
  used: after a pipeline change, comparing column i across generations
  would be silent misalignment, and the reference re-anchors to the new
  generation's own first days.
* Features are reported by INDEX (with the generation hash for context):
  Phase 2a deliberately stored the hash rather than the names.
* Below the data floor every metric is None -- absent, never 0.00.

PSI conventions (industry-standard thresholds): < 0.1 stable, 0.1-0.25
moderate, > 0.25 significant. Quantile bins are taken from the REFERENCE
window; a feature too degenerate to bin (near-constant) is reported as
unmeasurable rather than fabricated as stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

# Reference: the first days of the current generation's persisted record.
REFERENCE_DAYS = 7
# The recent window under test.
RECENT_DAYS = 2
# Minimum generation span before drift is computable at all.
MIN_TOTAL_DAYS = 5
# Each side of each SYMBOL needs enough vectors for decile bins to mean
# anything; symbols below the floor are excluded, never padded.
MIN_VECTORS_PER_SIDE = 300
_BINS = 10
# Laplace-style smoothing so an empty bin cannot produce an infinite PSI.
_EPS = 1e-4

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


@dataclass(frozen=True)
class FeatureDrift:
    index: int
    psi: float | None  # mean across measured symbols; None: unmeasurable
    psi_max: float | None
    n_symbols: int  # symbols this feature was measurable for


@dataclass(frozen=True)
class DriftReport:
    computable: bool
    reason: str | None
    generation_hash: str | None
    n_reference: int
    n_recent: int
    n_symbols_measured: int
    reference_start: datetime | None
    reference_end: datetime | None
    recent_start: datetime | None
    features: list[FeatureDrift]
    worst: tuple[str, int, float] | None  # (symbol, feature index, psi)
    notes: list[str]


def psi(reference: np.ndarray, recent: np.ndarray, bins: int = _BINS) -> float | None:
    """PSI of *recent* against *reference*, bins from reference deciles.

    Returns None when the reference is too degenerate to form at least two
    distinct bin edges -- a near-constant feature has no distribution to be
    stable OR drifted; fabricating 0.0 would overstate what we know.
    """
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:  # fewer than two real bins
        return None
    # Open the outer edges so recent values beyond the reference range count
    # in the tail bins instead of vanishing.
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, edges)
    rec_counts, _ = np.histogram(recent, edges)
    p = ref_counts / ref_counts.sum() + _EPS
    q = rec_counts / rec_counts.sum() + _EPS
    return float(np.sum((q - p) * np.log(q / p)))


def _notes() -> list[str]:
    return [
        "Reference is the platform's own EARLY serving history (the first "
        f"{REFERENCE_DAYS} days of the current pipeline generation), not the "
        "training distribution -- training-time distributions were never "
        "persisted; drift brackets train-serve skew, it does not equal it.",
        "PSI is computed PER SYMBOL against that symbol's own early window, "
        "then averaged: a pooled cross-symbol distribution would measure "
        "symbol row-share, not input drift (price-scale features differ by "
        "orders of magnitude across symbols).",
        "Rows from older pipeline generations (mismatched ordering hash) are "
        "excluded: cross-generation column comparison would be silent "
        "misalignment.",
        f"PSI thresholds: <{PSI_MODERATE} stable, "
        f"{PSI_MODERATE}-{PSI_SIGNIFICANT} moderate, "
        f">{PSI_SIGNIFICANT} significant. A null PSI means the feature was "
        "too degenerate to bin, never that it is stable.",
    ]


def _refusal(
    reason: str,
    generation_hash: str | None = None,
    n_reference: int = 0,
    n_recent: int = 0,
) -> DriftReport:
    return DriftReport(
        computable=False, reason=reason, generation_hash=generation_hash,
        n_reference=n_reference, n_recent=n_recent, n_symbols_measured=0,
        reference_start=None, reference_end=None, recent_start=None,
        features=[], worst=None, notes=_notes(),
    )


def build_report(
    rows: list[tuple[datetime, str, str, list[float]]],
    now: datetime,
) -> DriftReport:
    """Drift report from persisted (predicted_at, symbol, hash, vector) rows.

    Only the most recent generation's rows are used. Reference = that
    generation's first REFERENCE_DAYS (clipped so it can never overlap the
    recent window); recent = the last RECENT_DAYS. Rows between the windows
    are deliberately ignored -- they are neither baseline nor subject.
    """
    if not rows:
        return _refusal("no persisted feature vectors yet")

    latest_hash = max(rows, key=lambda r: r[0])[2]
    current = [(t, s, v) for t, s, h, v in rows if h == latest_hash]
    dims = {len(v) for _, _, v in current}
    if len(dims) != 1:
        # Same hash but ragged vectors: refuse loudly, never pad.
        return _refusal(
            "inconsistent vector lengths within one generation", latest_hash
        )
    n_features = dims.pop()

    gen_start = min(t for t, _, _ in current)
    gen_end = max(t for t, _, _ in current)
    span_days = (gen_end - gen_start).total_seconds() / 86_400.0
    recent_split = now - timedelta(days=RECENT_DAYS)
    reference_end = min(gen_start + timedelta(days=REFERENCE_DAYS), recent_split)

    reference = [(t, s, v) for t, s, v in current if t < reference_end]
    recent = [(t, s, v) for t, s, v in current if t >= recent_split]
    if span_days < MIN_TOTAL_DAYS:
        return _refusal(
            f"insufficient history: {span_days:.1f} days persisted for the "
            f"current generation; needs >= {MIN_TOTAL_DAYS}",
            latest_hash, len(reference), len(recent),
        )

    # Per symbol, both sides must clear the floor; below it the symbol is
    # excluded (never padded, never pooled with others).
    symbols = sorted({s for _, s, _ in current})
    per_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol in symbols:
        ref_vecs = [v for _, s, v in reference if s == symbol]
        rec_vecs = [v for _, s, v in recent if s == symbol]
        if len(ref_vecs) >= MIN_VECTORS_PER_SIDE and len(rec_vecs) >= MIN_VECTORS_PER_SIDE:
            per_symbol[symbol] = (
                np.asarray(ref_vecs, dtype=float),
                np.asarray(rec_vecs, dtype=float),
            )
    if not per_symbol:
        return _refusal(
            f"no symbol has >= {MIN_VECTORS_PER_SIDE} vectors on both sides "
            f"(reference: first {REFERENCE_DAYS}d of the generation; recent: "
            f"last {RECENT_DAYS}d)",
            latest_hash, len(reference), len(recent),
        )

    features: list[FeatureDrift] = []
    worst: tuple[str, int, float] | None = None
    for i in range(n_features):
        values: list[float] = []
        for symbol, (ref_m, rec_m) in per_symbol.items():
            value = psi(ref_m[:, i], rec_m[:, i])
            if value is None:
                continue
            values.append(value)
            if worst is None or value > worst[2]:
                worst = (symbol, i, value)
        features.append(
            FeatureDrift(
                index=i,
                psi=float(np.mean(values)) if values else None,
                psi_max=float(np.max(values)) if values else None,
                n_symbols=len(values),
            )
        )

    return DriftReport(
        computable=True, reason=None, generation_hash=latest_hash,
        n_reference=len(reference), n_recent=len(recent),
        n_symbols_measured=len(per_symbol),
        reference_start=gen_start, reference_end=reference_end,
        recent_start=recent_split,
        features=features, worst=worst, notes=_notes(),
    )
