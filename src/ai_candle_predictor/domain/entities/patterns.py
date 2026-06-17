from __future__ import annotations

from datetime import datetime
from enum import Enum, auto

from pydantic.dataclasses import dataclass


class CandlePattern(Enum):
    NONE = auto()
    BULLISH = auto()
    BEARISH = auto()
    DOJI = auto()
    HAMMER = auto()
    SHOOTING_STAR = auto()
    BULLISH_ENGULFING = auto()
    BEARISH_ENGULFING = auto()


@dataclass(frozen=True)
class PatternMatch:
    pattern: CandlePattern
    timestamp: datetime
    symbol: str
    confidence: float = 1.0
    description: str = ""
