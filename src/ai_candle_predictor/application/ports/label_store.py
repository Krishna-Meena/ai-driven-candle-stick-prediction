from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from ai_candle_predictor.domain.entities.label import LabeledSample
from ai_candle_predictor.domain.value_objects.symbol import Symbol


class LabelStore(ABC):
    @abstractmethod
    def save(self, samples: Sequence[LabeledSample]) -> int: ...

    @abstractmethod
    def load(
        self,
        symbol: Symbol,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[LabeledSample]: ...
