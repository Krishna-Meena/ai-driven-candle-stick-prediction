from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ai_candle_predictor.domain.entities.patterns import CandlePattern


@dataclass(frozen=True)
class ChartConfig:
    width: int = 1200
    height: int = 700
    dpi: int = 150
    style: str = "charles"
    highlight_patterns: tuple[CandlePattern, ...] = field(
        default_factory=lambda: (
            CandlePattern.DOJI,
            CandlePattern.HAMMER,
            CandlePattern.SHOOTING_STAR,
            CandlePattern.BULLISH_ENGULFING,
            CandlePattern.BEARISH_ENGULFING,
        )
    )


@dataclass(frozen=True)
class ChartRequest:
    symbol: str
    start_date: datetime
    end_date: datetime
    title: str = ""
    config: ChartConfig = field(default_factory=ChartConfig)


@dataclass(frozen=True)
class RenderedChart:
    symbol: str
    format: str
    width: int
    height: int
    size_bytes: int
    path: Path | None = None
