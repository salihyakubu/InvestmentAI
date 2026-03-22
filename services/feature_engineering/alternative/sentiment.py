"""Sentiment analysis for financial text.

Uses keyword-based scoring as a lightweight baseline.  In production this
would be replaced by a FinBERT or similar transformer model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Keyword lexicon
# ---------------------------------------------------------------------------

_BULLISH_WORDS: set[str] = {
    "bullish", "buy", "long", "upgrade", "upside", "growth", "beat",
    "strong", "surge", "rally", "breakout", "outperform", "positive",
    "profit", "gain", "higher", "recovery", "momentum", "accumulate",
    "boost", "optimistic", "innovative", "record", "exceeds", "soar",
    "boom", "dividend", "earnings", "revenue", "expand",
}

_BEARISH_WORDS: set[str] = {
    "bearish", "sell", "short", "downgrade", "downside", "decline",
    "miss", "weak", "crash", "plunge", "breakdown", "underperform",
    "negative", "loss", "lower", "recession", "risk", "distribute",
    "cut", "pessimistic", "overvalued", "warning", "debt", "default",
    "fraud", "investigation", "layoff", "bankrupt", "lawsuit", "drop",
}

_INTENSIFIERS: set[str] = {
    "very", "extremely", "significantly", "strongly", "massively",
    "sharply", "dramatically", "hugely",
}

_NEGATORS: set[str] = {
    "not", "no", "never", "neither", "nor", "without", "hardly",
    "barely", "scarcely", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "won't", "wouldn't", "can't",
    "cannot", "couldn't", "shouldn't",
}

_WORD_RE = re.compile(r"[a-z']+")


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------


@dataclass
class SentimentAnalyzer:
    """Keyword-based financial sentiment scorer."""

    bullish_words: set[str] = field(default_factory=lambda: set(_BULLISH_WORDS))
    bearish_words: set[str] = field(default_factory=lambda: set(_BEARISH_WORDS))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> float:
        """Score a single piece of text.

        Returns a float in [-1, 1] where -1 is maximally bearish and
        +1 is maximally bullish.
        """
        tokens = _WORD_RE.findall(text.lower())
        if not tokens:
            return 0.0

        score = 0.0
        negate = False
        intensify = False

        for tok in tokens:
            if tok in _NEGATORS:
                negate = True
                continue
            if tok in _INTENSIFIERS:
                intensify = True
                continue

            raw = 0.0
            if tok in self.bullish_words:
                raw = 1.0
            elif tok in self.bearish_words:
                raw = -1.0

            if raw != 0.0:
                if intensify:
                    raw *= 1.5
                if negate:
                    raw *= -1.0
                score += raw

            negate = False
            intensify = False

        # Normalise to [-1, 1] using tanh-style squashing.
        if len(tokens) > 0:
            normalised = score / max(len(tokens) ** 0.5, 1.0)
            return max(-1.0, min(1.0, normalised))
        return 0.0

    @staticmethod
    def aggregate_sentiment(scores: list[float]) -> float:
        """Aggregate multiple sentiment scores into one.

        Uses a volume-weighted mean (uniform weight here).
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def get_market_sentiment(symbol: str) -> dict:
        """Return a placeholder market-sentiment summary for *symbol*.

        In production this would query news APIs, social-media feeds,
        and analyst reports.
        """
        return {
            "symbol": symbol,
            "score": 0.0,
            "volume": 0,
            "sources": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
