from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.base import BaseEstimator


class ModelStore(ABC):
    @abstractmethod
    def save(
        self,
        model: BaseEstimator,
        symbol: str,
        label: str = "",
    ) -> Path: ...

    @abstractmethod
    def load(self, path: Path) -> BaseEstimator: ...
