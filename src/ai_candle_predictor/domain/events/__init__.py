from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DataIngested:
    symbol: str
    start_date: datetime
    end_date: datetime
    candle_count: int
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class DataValidationFailed:
    symbol: str
    reason: str
    row: dict[str, Any] | None = None
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class DataStored:
    symbol: str
    path: str
    row_count: int
    occurred_at: datetime = field(default_factory=datetime.utcnow)
