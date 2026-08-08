"""Does the learning loop actually learn? Metrics that answer honestly.

The platform retrains weekly, gates champions, and has resolved 130k+ live
prediction outcomes -- but nothing measured whether any of that IMPROVES the
served signal. These metrics close that gap, from stored predictions alone.

Design constraints learned the hard way in this repo:

* ``model_version`` on predictions is constant (the ensemble always stamps
  v1), so "v3 vs v4" cannot be read off rows. Promotions ARE dated in
  ``model_metadata``, so the honest comparison is BY ERA: live signal
  quality in the periods between promotion events. Any future retrain
  automatically opens a new era -- no code change.
* Sign agreement EXCLUDES zero-return outcomes. Coarse-tick symbols leave
  the price literally unchanged in up to a third of windows; sign(0)
  matches nothing, and including those bars once manufactured a fake
  "inversion" finding (retracted in PR #59). The guard is now structural.
* The stored ``confidence`` is p(flat) for gated predictions (~100% of
  them), so calibration is measured on exactly the event that probability
  describes: DID the price stay inside the deadband? Brier score plus
  reliability bins, restricted to flattened predictions.
* Predictions overlap 5x (per-minute emissions, 5-minute horizon).
  Statistics are computed on a de-overlapped subsample; the API achieves
  this at fetch time by sampling minute % 5 == 0.

A metric the data cannot support is None -- never zero. Zero IC is a
measurement; None is an absence. The UI renders the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import numpy as np

# Below this many resolved, de-overlapped observations a day is not judged.
MIN_DAILY_OBSERVATIONS = 50
# Reliability bins for p(flat) calibration.
CALIBRATION_BINS = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0))


@dataclass(frozen=True)
class ResolvedPrediction:
    """One resolved, de-overlapped serving prediction."""

    predicted_at: datetime
    symbol: str
    expected_return: float
    confidence: float
    direction: str
    actual_return: float
    actual_direction: str


@dataclass
class DailyQuality:
    day: date
    n: int
    ic: float | None
    sign_agreement: float | None
    abstention_rate: float
    brier_flat: float | None


@dataclass
class EraSummary:
    """Live signal quality between two promotion events."""

    label: str
    start: date
    end: date | None
    days: int
    n: int
    mean_ic: float | None
    ic_t_stat: float | None
    mean_brier: float | None
    sign_agreement: float | None


@dataclass
class LearningReport:
    daily: list[DailyQuality] = field(default_factory=list)
    eras: list[EraSummary] = field(default_factory=list)
    calibration: list[dict[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    from scipy import stats

    rho = stats.spearmanr(a, b)
    return None if np.isnan(rho.statistic) else float(rho.statistic)


def daily_quality(rows: list[ResolvedPrediction]) -> list[DailyQuality]:
    """Per-UTC-day signal quality over de-overlapped resolved predictions."""
    by_day: dict[date, list[ResolvedPrediction]] = {}
    for row in rows:
        by_day.setdefault(row.predicted_at.date(), []).append(row)

    out: list[DailyQuality] = []
    for day in sorted(by_day):
        bucket = by_day[day]
        n = len(bucket)
        abstention = sum(1 for r in bucket if r.direction == "flat") / n
        if n < MIN_DAILY_OBSERVATIONS:
            out.append(
                DailyQuality(
                    day=day, n=n, ic=None, sign_agreement=None,
                    abstention_rate=abstention, brier_flat=None,
                )
            )
            continue
        expected = np.array([r.expected_return for r in bucket])
        actual = np.array([r.actual_return for r in bucket])
        ic = _spearman(expected, actual)
        # Zero-return outcomes are excluded from AGREEMENT: sign(0) matches
        # nothing and coarse ticks would fake degradation (see PR #59).
        moved = actual != 0.0
        agreement: float | None = None
        if moved.sum() >= MIN_DAILY_OBSERVATIONS // 2:
            centred = expected[moved] - np.median(expected[moved])
            agreement = float((np.sign(centred) == np.sign(actual[moved])).mean())
        # Calibration only where confidence MEANS p(flat): gated predictions.
        gated = [r for r in bucket if r.direction == "flat"]
        brier: float | None = None
        if len(gated) >= MIN_DAILY_OBSERVATIONS // 2:
            forecast = np.array([r.confidence for r in gated])
            outcome = np.array(
                [1.0 if r.actual_direction == "flat" else 0.0 for r in gated]
            )
            brier = float(np.mean((forecast - outcome) ** 2))
        out.append(
            DailyQuality(
                day=day, n=n, ic=ic, sign_agreement=agreement,
                abstention_rate=abstention, brier_flat=brier,
            )
        )
    return out


def era_boundaries(promotion_times: list[datetime]) -> list[tuple[str, date]]:
    """Distinct promotion DATES become era starts, labelled generationally.

    Multiple models promoted the same day (an ensemble refresh) are one
    era boundary, not several.
    """
    days = sorted({t.astimezone(UTC).date() for t in promotion_times})
    return [(f"era {i + 1} (from {d.isoformat()})", d) for i, d in enumerate(days)]


def era_summaries(
    daily: list[DailyQuality], boundaries: list[tuple[str, date]]
) -> list[EraSummary]:
    """Aggregate daily quality between promotion events.

    THE learning question: if later eras do not show better live IC or
    calibration than earlier ones, retraining is churning, not learning --
    and that is a finding, not an insult.
    """
    if not boundaries:
        return []
    out: list[EraSummary] = []
    for i, (label, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else None
        in_era = [
            d for d in daily
            if d.day >= start and (end is None or d.day < end)
        ]
        ics = np.array([d.ic for d in in_era if d.ic is not None])
        briers = [d.brier_flat for d in in_era if d.brier_flat is not None]
        agreements = [d.sign_agreement for d in in_era if d.sign_agreement is not None]
        mean_ic = float(ics.mean()) if ics.size else None
        t_stat: float | None = None
        if ics.size > 2 and ics.std(ddof=1) > 0:
            t_stat = float(ics.mean() / ics.std(ddof=1) * np.sqrt(ics.size))
        out.append(
            EraSummary(
                label=label,
                start=start,
                end=end,
                days=len(in_era),
                n=sum(d.n for d in in_era),
                mean_ic=mean_ic,
                ic_t_stat=t_stat,
                mean_brier=float(np.mean(briers)) if briers else None,
                sign_agreement=float(np.mean(agreements)) if agreements else None,
            )
        )
    return out


def calibration_bins(rows: list[ResolvedPrediction]) -> list[dict[str, float]]:
    """Reliability table: stated p(flat) vs realized flat frequency.

    A calibrated gate has forecast ~= frequency in every bin. Restricted to
    gated (flattened) predictions, where confidence IS p(flat).
    """
    gated = [r for r in rows if r.direction == "flat"]
    out: list[dict[str, float]] = []
    for lo, hi in CALIBRATION_BINS:
        members = [r for r in gated if lo <= r.confidence < hi]
        if len(members) < MIN_DAILY_OBSERVATIONS:
            continue
        forecast = float(np.mean([r.confidence for r in members]))
        realized = float(
            np.mean([1.0 if r.actual_direction == "flat" else 0.0 for r in members])
        )
        out.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": float(len(members)),
                "mean_forecast": forecast,
                "realized_flat_rate": realized,
                "gap": realized - forecast,
            }
        )
    return out


def build_report(
    rows: list[ResolvedPrediction], promotion_times: list[datetime]
) -> LearningReport:
    daily = daily_quality(rows)
    eras = era_summaries(daily, era_boundaries(promotion_times))
    report = LearningReport(
        daily=daily,
        eras=eras,
        calibration=calibration_bins(rows),
        notes=[
            "IC is Spearman(expected_return, actual_return) on de-overlapped "
            "predictions; sign agreement excludes zero-return outcomes "
            "(coarse-tick guard, PR #59).",
            "Eras are promotion-dated segments; predictions do not carry a "
            "usable model version, so this is the only honest live "
            "version-over-version comparison.",
            "confidence is p(flat) for gated predictions; calibration is "
            "measured on exactly that event.",
            "None means the data cannot support the metric; it is never 0.",
        ],
    )
    return report
