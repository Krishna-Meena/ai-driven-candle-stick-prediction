from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class RegistryEntry:
    symbol: str
    model_type: str
    label: str
    filename: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    support: int
    training_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ModelRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = registry_path or settings.models_dir / "registry.json"
        self._entries: list[RegistryEntry] = []
        self._load()

    def register(self, entry: RegistryEntry) -> None:
        self._entries.append(entry)
        self._save()
        log.info(
            "model registered",
            symbol=entry.symbol,
            model_type=entry.model_type,
            label=entry.label,
        )

    def list_models(self) -> list[RegistryEntry]:
        return list(self._entries)

    def list_by_symbol(self, symbol: str) -> list[RegistryEntry]:
        return [e for e in self._entries if e.symbol == symbol]

    def get_latest(self, symbol: str, model_type: str) -> RegistryEntry | None:
        matching = [e for e in self._entries if e.symbol == symbol and e.model_type == model_type]
        return matching[-1] if matching else None

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [RegistryEntry(**item) for item in raw]
        except Exception:
            log.warning("failed to load model registry, starting fresh", path=str(self._path))
            self._entries = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = [asdict(e) for e in self._entries]
        self._path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
