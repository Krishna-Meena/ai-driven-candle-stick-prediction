from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.value_objects.symbol import Symbol


class DataProvider(ABC):
    @abstractmethod
    def fetch_historical(
        self,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> Sequence[CandleStick]: ...
