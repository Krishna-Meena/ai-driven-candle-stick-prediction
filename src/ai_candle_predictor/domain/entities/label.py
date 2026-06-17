from __future__ import annotations

from datetime import datetime
from enum import Enum, auto

from pydantic.dataclasses import dataclass


class Label(Enum):
    UP = auto()
    DOWN = auto()
    EXCLUDED = auto()

    @property
    def as_int(self) -> int | None:
        if self == Label.UP:
            return 1
        if self == Label.DOWN:
            return 0
        return None


@dataclass(frozen=True)
class LabeledSample:
    symbol: str
    timestamp: datetime
    label: Label
    forward_return: float
    horizon: int
    close: float
