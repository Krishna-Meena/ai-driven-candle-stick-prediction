from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CandlePrediction:
    timestamp: datetime
    close: float
    predicted_direction: int
    confidence: float
    actual_return: float | None = None
    actual_direction: int | None = None
    is_correct: bool | None = None


@dataclass(frozen=True)
class PredictionResult:
    symbol: str
    model_label: str
    start_date: datetime
    end_date: datetime
    horizon: int
    total_candles: int
    predictions: list[CandlePrediction] = field(default_factory=list)
