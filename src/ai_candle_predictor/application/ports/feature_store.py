from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from ai_candle_predictor.domain.entities.indicators import IndicatorValue
from ai_candle_predictor.domain.value_objects.symbol import Symbol


class FeatureStore(ABC):
    @abstractmethod
    def save(self, features: Sequence[IndicatorValue]) -> int: ...

    @abstractmethod
    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[IndicatorValue]: ...
