"""The feature-ordering hash: one definition, shared by writer and replayer.

Phase 2a stamps every persisted feature vector with this hash of the
column-name ordering; the Phase 2 live-transfer gate refuses any row whose
hash differs from the challenger's. Serving and replay MUST compute it
identically -- a silent divergence would misalign columns, so the function
lives here and nowhere else.
"""

from __future__ import annotations

import hashlib


def feature_ordering_hash(feature_names: list[str]) -> str:
    """16-hex digest of the comma-joined feature-name ordering."""
    return hashlib.sha256(",".join(feature_names).encode()).hexdigest()[:16]
