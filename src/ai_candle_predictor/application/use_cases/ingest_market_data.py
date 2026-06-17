from ai_candle_predictor.application.dto.market_data import (
    IngestionResult,
    MarketDataRequest,
)
from ai_candle_predictor.application.ports.data_provider import DataProvider
from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.events import DataStored, DataValidationFailed
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)


def ingest_market_data(
    request: MarketDataRequest,
    provider: DataProvider,
    storage: StorageAdapter,
) -> IngestionResult:
    symbol = Symbol(request.symbol)

    log.info("fetching market data", symbol=symbol.value, start=request.start_date.isoformat())

    raw_candles = provider.fetch_historical(
        symbol=symbol,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    valid_candles: list[CandleStick] = []
    errors: list[str] = []

    for candle in raw_candles:
        try:
            validated = CandleStick(
                symbol=candle.symbol,
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                adjusted_close=candle.adjusted_close,
            )
            valid_candles.append(validated)
        except ValueError as exc:
            errors.append(str(exc))
            DataValidationFailed(
                symbol=symbol.value,
                reason=str(exc),
            )

    if not valid_candles:
        log.error("no valid candles after validation", symbol=symbol.value)
        return IngestionResult(
            symbol=symbol.value,
            rows_fetched=len(raw_candles),
            rows_valid=0,
            rows_rejected=len(errors),
            errors=errors,
        )

    storage_path = storage.save(symbol, valid_candles)

    DataStored(
        symbol=symbol.value,
        path=storage_path,
        row_count=len(valid_candles),
    )

    log.info(
        "market data ingested",
        symbol=symbol.value,
        valid=len(valid_candles),
        rejected=len(errors),
        path=storage_path,
    )

    return IngestionResult(
        symbol=symbol.value,
        rows_fetched=len(raw_candles),
        rows_valid=len(valid_candles),
        rows_rejected=len(errors),
        storage_path=storage_path,
        errors=errors,
    )
