from __future__ import annotations

from ai_candle_predictor.application.dto.chart import ChartRequest, RenderedChart
from ai_candle_predictor.application.ports.chart_renderer import ChartRenderer
from ai_candle_predictor.application.ports.image_storage import ImageStorage
from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.visualization.pattern_detector import detect_patterns

log = get_logger(__name__)


def generate_chart(
    request: ChartRequest,
    storage: StorageAdapter,
    renderer: ChartRenderer,
    image_storage: ImageStorage,
) -> RenderedChart:
    symbol = Symbol(request.symbol)

    log.info(
        "generating chart",
        symbol=request.symbol,
        start=request.start_date.isoformat(),
        end=request.end_date.isoformat(),
    )

    candles = storage.load(
        symbol=symbol,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    if not candles:
        log.warning("no candles found for chart", symbol=request.symbol)
        return RenderedChart(
            symbol=request.symbol,
            format="png",
            width=request.config.width,
            height=request.config.height,
            size_bytes=0,
        )

    patterns = detect_patterns(candles)
    log.info("patterns detected", symbol=request.symbol, count=len(patterns))

    image_bytes = renderer.render_to_bytes(request, candles, patterns)
    path = image_storage.save(request.symbol, image_bytes)

    return RenderedChart(
        symbol=request.symbol,
        format="png",
        width=request.config.width,
        height=request.config.height,
        size_bytes=len(image_bytes),
        path=path,
    )
