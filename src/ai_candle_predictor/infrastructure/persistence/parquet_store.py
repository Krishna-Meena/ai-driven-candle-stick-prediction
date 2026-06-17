from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

PARQUET_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
]


class ParquetStore(StorageAdapter):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.data_raw_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: Symbol) -> Path:
        safe = symbol.value.replace("^", "_").replace(".", "_")
        return self._base_dir / f"{safe}.parquet"

    def save(self, symbol: Symbol, candles: Sequence[CandleStick]) -> str:
        path = self._file_path(symbol)
        new_df = self._candles_to_frame(candles)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "timestamp"]).sort_values(
                "timestamp"
            )
        else:
            combined = new_df

        combined.to_parquet(path, index=False)
        log.info("parquet saved", path=str(path), rows=len(combined))
        return str(path)

    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[CandleStick]:
        path = self._file_path(symbol)
        if not path.exists():
            log.warning("parquet not found", path=str(path))
            return []

        df = pd.read_parquet(path)

        if start_date:
            df = df[pd.to_datetime(df["timestamp"]) >= pd.Timestamp(start_date)]
        if end_date:
            df = df[pd.to_datetime(df["timestamp"]) <= pd.Timestamp(end_date)]

        return [
            CandleStick(
                symbol=row["symbol"],
                timestamp=row["timestamp"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                adjusted_close=(
                    float(row["adjusted_close"]) if pd.notna(row["adjusted_close"]) else None
                ),
            )
            for _, row in df.iterrows()
        ]

    def _candles_to_frame(self, candles: Sequence[CandleStick]) -> pd.DataFrame:
        records = []
        for c in candles:
            records.append(
                {
                    "symbol": c.symbol,
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "adjusted_close": c.adjusted_close,
                }
            )
        return pd.DataFrame(records, columns=PARQUET_COLUMNS)
