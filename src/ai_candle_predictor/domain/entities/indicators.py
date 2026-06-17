from __future__ import annotations

from datetime import datetime
from enum import Enum, auto

from pydantic.dataclasses import dataclass


class IndicatorType(Enum):
    SMA = auto()
    EMA = auto()
    RSI = auto()
    MACD = auto()
    MACD_SIGNAL = auto()
    MACD_HISTOGRAM = auto()
    BOLLINGER_UPPER = auto()
    BOLLINGER_MIDDLE = auto()
    BOLLINGER_LOWER = auto()
    BOLLINGER_PERCENT_B = auto()
    BOLLINGER_BANDWIDTH = auto()
    ATR = auto()
    ADX = auto()
    ADX_PLUS_DI = auto()
    ADX_MINUS_DI = auto()
    STOCHASTIC_K = auto()
    STOCHASTIC_D = auto()


@dataclass(frozen=True)
class IndicatorValue:
    symbol: str
    timestamp: datetime
    indicator: IndicatorType
    value: float
