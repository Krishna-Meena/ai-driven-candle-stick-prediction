from __future__ import annotations

from pathlib import Path

from ai_candle_predictor.application.ports.image_storage import ImageStorage
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger

log = get_logger(__name__)


class ImageStore(ImageStorage):
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or settings.reports_dir / "charts"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, symbol: str, image_bytes: bytes, fmt: str = "png") -> Path:
        symbol_dir = self._base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(symbol_dir.glob(f"*.{fmt}"))
        counter = len(existing) + 1
        path = symbol_dir / f"{symbol}_{counter:04d}.{fmt}"

        path.write_bytes(image_bytes)
        log.info("chart image saved", path=str(path), size_bytes=len(image_bytes))
        return path

    def load(self, path: Path) -> bytes:
        return path.read_bytes()
