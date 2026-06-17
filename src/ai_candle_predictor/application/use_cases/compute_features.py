from __future__ import annotations

from datetime import datetime

import pandas as pd

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.features.computations import compute_all

log = get_logger(__name__)


def compute_features(
    symbol: Symbol,
    storage: StorageAdapter,
    feature_store: FeatureStore,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> int:
    log.info("computing features", symbol=symbol.value)

    candles = storage.load(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if not candles:
        log.warning("no candles available for feature computation", symbol=symbol.value)
        return 0

    records = []
    for c in candles:
        records.append(
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
        )

    df = pd.DataFrame(records)
    ts_index = pd.to_datetime([c.timestamp for c in candles])
    if isinstance(ts_index.dtype, pd.DatetimeTZDtype):
        ts_index = ts_index.tz_convert("UTC").tz_localize(None)
    df.index = ts_index

    features = compute_all(symbol.value, df)

    saved = feature_store.save(features)
    log.info(
        "features computed",
        symbol=symbol.value,
        candles=len(candles),
        features=saved,
    )
    return saved
