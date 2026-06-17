from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str
    start_date: datetime
    end_date: datetime | None = None


@dataclass(frozen=True)
class IngestionResult:
    symbol: str
    rows_fetched: int
    rows_valid: int
    rows_rejected: int
    storage_path: str | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
