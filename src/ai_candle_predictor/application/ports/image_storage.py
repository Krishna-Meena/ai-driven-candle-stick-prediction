from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageStorage(ABC):
    @abstractmethod
    def save(self, symbol: str, image_bytes: bytes, fmt: str = "png") -> Path: ...

    @abstractmethod
    def load(self, path: Path) -> bytes: ...
