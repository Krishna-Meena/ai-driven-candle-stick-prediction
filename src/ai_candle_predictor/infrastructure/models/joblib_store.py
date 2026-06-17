from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger

log = get_logger(__name__)


class JoblibStore(ModelStore):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.models_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: BaseEstimator,
        symbol: str,
        label: str = "",
    ) -> Path:
        safe = symbol.replace("^", "_").replace(".", "_")
        parts = [safe]
        if label:
            parts.append(label)
        filename = "_".join(parts) + ".joblib"
        path = self._base_dir / filename

        joblib.dump(model, path)
        log.info("model saved", path=str(path))
        return path

    def load(self, path: Path) -> BaseEstimator:
        model: BaseEstimator = joblib.load(path)
        log.info("model loaded", path=str(path))
        return model
