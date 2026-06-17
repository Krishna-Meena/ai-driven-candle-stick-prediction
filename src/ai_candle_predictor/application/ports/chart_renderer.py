from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_candle_predictor.application.dto.chart import ChartRequest
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.patterns import PatternMatch


class ChartRenderer(ABC):
    @abstractmethod
    def render_to_bytes(
        self,
        request: ChartRequest,
        candles: Sequence[CandleStick],
        patterns: Sequence[PatternMatch],
    ) -> bytes: ...
