from __future__ import annotations

from datetime import datetime

from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.label import Label
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.labeling.forward_returns import generate_labels_raw

log = get_logger(__name__)


def generate_labels_for_symbol(
    symbol: Symbol,
    storage: StorageAdapter,
    label_store: LabelStore,
    horizon: int = 5,
    threshold: float = 0.005,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, int]:
    log.info(
        "generating labels",
        symbol=symbol.value,
        horizon=horizon,
        threshold=threshold,
    )

    candles = storage.load(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if not candles:
        log.warning("no candles available for labeling", symbol=symbol.value)
        return {"total": 0, "up": 0, "down": 0, "excluded": 0}

    all_samples = generate_labels_raw(candles, horizon=horizon, threshold=threshold)
    samples = [s for s in all_samples if s.label != Label.EXCLUDED]
    excluded = len(all_samples) - len(samples)

    saved = label_store.save(samples)

    up = sum(1 for s in samples if s.label == Label.UP)
    down = sum(1 for s in samples if s.label == Label.DOWN)

    log.info(
        "labels complete",
        symbol=symbol.value,
        total=len(all_samples),
        up=up,
        down=down,
        excluded=excluded,
        saved=saved,
    )

    return {
        "total": len(all_samples),
        "up": up,
        "down": down,
        "excluded": excluded,
    }
