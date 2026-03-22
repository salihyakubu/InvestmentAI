"""Model registry for versioned storage and promotion of trained models."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.prediction.models.base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata record for a registered model version."""

    model_id: str
    model_name: str
    model_type: str
    version: int
    is_active: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Filesystem + JSON-based model registry.

    Artefacts are saved under ``artifact_base/<model_name>/v<version>/``.
    A ``registry.json`` file in ``artifact_base`` tracks all metadata.
    If a *db_session* is provided it is reserved for future database
    integration but not used in the current implementation.
    """

    _REGISTRY_FILE = "registry.json"

    def __init__(self, artifact_base: Path, db_session: Any = None) -> None:
        self.artifact_base = artifact_base
        self.artifact_base.mkdir(parents=True, exist_ok=True)
        self._db_session = db_session
        self._registry_path = self.artifact_base / self._REGISTRY_FILE
        self._entries: list[ModelMetadata] = self._load_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: BasePredictor,
        model_name: str,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Register a new model version.

        Returns:
            ``(model_id, version)``
        """
        existing_versions = [e.version for e in self._entries if e.model_name == model_name]
        version = max(existing_versions, default=0) + 1
        model_id = str(uuid.uuid4())

        artifact_dir = self.artifact_base / model_name / f"v{version}"
        model.save(artifact_dir)

        meta = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model.model_type,
            version=version,
            is_active=False,
            metrics=metrics,
            artifact_path=str(artifact_dir),
            created_at=datetime.now(timezone.utc).isoformat(),
            extra=extra or {},
        )
        self._entries.append(meta)
        self._save_registry()

        logger.info("Registered %s v%d (id=%s)", model_name, version, model_id)
        return model_id, version

    def promote(self, model_id: str, version: int | None = None) -> None:
        """Promote a model version to active, deactivating all other
        versions of the same model name.
        """
        target: ModelMetadata | None = None
        for entry in self._entries:
            if entry.model_id == model_id:
                if version is None or entry.version == version:
                    target = entry
                    break

        if target is None:
            raise ValueError(f"Model id={model_id} version={version} not found")

        # Deactivate all versions of this model name
        for entry in self._entries:
            if entry.model_name == target.model_name:
                entry.is_active = False

        target.is_active = True
        self._save_registry()
        logger.info("Promoted %s v%d to active", target.model_name, target.version)

    def get_active(self, model_type: str) -> tuple[BasePredictor, ModelMetadata]:
        """Load and return the active model for the given model type.

        Raises:
            ValueError: if no active model exists.
        """
        for entry in self._entries:
            if entry.model_type == model_type and entry.is_active:
                model = self._load_model(entry)
                return model, entry

        raise ValueError(f"No active model found for type '{model_type}'")

    def list_versions(self, model_name: str) -> list[ModelMetadata]:
        return [e for e in self._entries if e.model_name == model_name]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, meta: ModelMetadata) -> BasePredictor:
        """Instantiate and load a model from its saved artefacts."""
        from services.prediction.models.lightgbm_model import LightGBMPredictor
        from services.prediction.models.lstm_model import LSTMPredictor
        from services.prediction.models.transformer_model import TransformerPredictor
        from services.prediction.models.xgboost_model import XGBoostPredictor

        model_classes: dict[str, type[BasePredictor]] = {
            "xgboost": XGBoostPredictor,
            "lightgbm": LightGBMPredictor,
            "lstm": LSTMPredictor,
            "transformer": TransformerPredictor,
        }
        cls = model_classes.get(meta.model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{meta.model_type}'")

        model = cls()
        model.load(Path(meta.artifact_path))
        return model

    def _load_registry(self) -> list[ModelMetadata]:
        if not self._registry_path.exists():
            return []
        try:
            data = json.loads(self._registry_path.read_text())
            return [ModelMetadata(**entry) for entry in data]
        except Exception:
            logger.exception("Failed to load registry, starting fresh")
            return []

    def _save_registry(self) -> None:
        data = [asdict(e) for e in self._entries]
        self._registry_path.write_text(json.dumps(data, indent=2))
