from datetime import datetime

from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=dict(validate_assignment=True, extra="forbid"))
class CandleStick:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float | None = None

    def __post_init__(self) -> None:
        if self.open < 0:
            raise ValueError(f"open must be >= 0, got {self.open}")
        if self.high < 0:
            raise ValueError(f"high must be >= 0, got {self.high}")
        if self.low < 0:
            raise ValueError(f"low must be >= 0, got {self.low}")
        if self.close < 0:
            raise ValueError(f"close must be >= 0, got {self.close}")
        if self.volume < 0:
            raise ValueError(f"volume must be >= 0, got {self.volume}")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) must be >= max(open={self.open}, close={self.close})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) must be <= min(open={self.open}, close={self.close})"
            )
        if self.adjusted_close is not None and self.adjusted_close < 0:
            raise ValueError(f"adjusted_close must be >= 0, got {self.adjusted_close}")

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def return_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return ((self.close - self.open) / self.open) * 100.0
