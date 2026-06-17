from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.label import Label, LabeledSample
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

_COLUMNS = [
    "symbol",
    "timestamp",
    "label",
    "forward_return",
    "horizon",
    "close",
]


class ParquetLabelStore(LabelStore):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.data_interim_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: Symbol) -> Path:
        safe = symbol.value.replace("^", "_").replace(".", "_")
        return self._base_dir / f"{safe}_labels.parquet"

    def save(self, samples: Sequence[LabeledSample]) -> int:
        if not samples:
            return 0

        symbol_str = samples[0].symbol
        path = self._file_path(Symbol(symbol_str))

        records = [
            {
                "symbol": s.symbol,
                "timestamp": s.timestamp,
                "label": s.label.name,
                "forward_return": s.forward_return,
                "horizon": s.horizon,
                "close": s.close,
            }
            for s in samples
        ]
        new_df = pd.DataFrame(records, columns=_COLUMNS)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "timestamp", "horizon"]
            ).sort_values("timestamp")
        else:
            combined = new_df

        combined.to_parquet(path, index=False)
        log.info("labels saved", path=str(path), rows=len(combined))
        return len(samples)

    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[LabeledSample]:
        path = self._file_path(symbol)
        if not path.exists():
            return []

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        result: list[LabeledSample] = []
        for _, row in df.iterrows():
            result.append(
                LabeledSample(
                    symbol=str(row["symbol"]),
                    timestamp=row["timestamp"].to_pydatetime(),
                    label=Label[row["label"]],
                    forward_return=float(row["forward_return"]),
                    horizon=int(row["horizon"]),
                    close=float(row["close"]),
                )
            )
        return result
