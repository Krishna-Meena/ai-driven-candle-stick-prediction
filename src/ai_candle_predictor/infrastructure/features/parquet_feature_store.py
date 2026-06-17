from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.indicators import IndicatorType, IndicatorValue
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

_FEATURE_COLUMNS = [
    "symbol",
    "timestamp",
    "indicator",
    "value",
]


class ParquetFeatureStore(FeatureStore):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.data_processed_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: Symbol) -> Path:
        safe = symbol.value.replace("^", "_").replace(".", "_")
        return self._base_dir / f"{safe}_features.parquet"

    def save(self, features: Sequence[IndicatorValue]) -> int:
        if not features:
            return 0

        symbol_str = features[0].symbol
        path = self._file_path(Symbol(symbol_str))

        records = [
            {
                "symbol": f.symbol,
                "timestamp": f.timestamp,
                "indicator": f.indicator.name,
                "value": f.value,
            }
            for f in features
        ]
        new_df = pd.DataFrame(records, columns=_FEATURE_COLUMNS)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "timestamp", "indicator"]
            ).sort_values("timestamp")
        else:
            combined = new_df

        combined.to_parquet(path, index=False)
        log.info("features saved", path=str(path), rows=len(combined))
        return len(features)

    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[IndicatorValue]:
        path = self._file_path(symbol)
        if not path.exists():
            log.warning("features not found", path=str(path))
            return []

        df = pd.read_parquet(path)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        result: list[IndicatorValue] = []
        for _, row in df.iterrows():
            try:
                result.append(
                    IndicatorValue(
                        symbol=str(row["symbol"]),
                        timestamp=row["timestamp"].to_pydatetime(),
                        indicator=IndicatorType[row["indicator"]],
                        value=float(row["value"]),
                    )
                )
            except (KeyError, ValueError):
                continue

        return result
