from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.value_objects.symbol import Symbol


class StorageAdapter(ABC):
    @abstractmethod
    def save(self, symbol: Symbol, candles: Sequence[CandleStick]) -> str: ...

    @abstractmethod
    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[CandleStick]: ...
