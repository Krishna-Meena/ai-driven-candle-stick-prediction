from collections.abc import Sequence
from datetime import datetime

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from ai_candle_predictor.application.ports.data_provider import DataProvider
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Adj Close": "adjusted_close",
}

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


class YahooProvider(DataProvider):
    def fetch_historical(
        self,
        symbol: Symbol,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> Sequence[CandleStick]:
        log.info(
            "yahoo fetch",
            symbol=symbol.value,
            start=start_date.isoformat(),
        )
        df = self._download(symbol.value, start_date, end_date)
        return self._to_candles(df, symbol.value)

    @retry(
        stop=stop_after_attempt(settings.yfinance_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _download(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None,
    ) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            log.warning("empty response from yfinance", symbol=ticker)
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            log.error("missing columns in yfinance response", symbol=ticker, missing=missing)
            return pd.DataFrame()

        return df

    def _to_candles(self, df: pd.DataFrame, ticker: str) -> list[CandleStick]:
        if df.empty:
            return []

        df = df.rename(columns=COLUMN_MAP)
        candles: list[CandleStick] = []

        for idx, row in df.iterrows():
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            candles.append(
                CandleStick(
                    symbol=ticker,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    adjusted_close=(
                        float(row["adjusted_close"]) if "adjusted_close" in row else None
                    ),
                )
            )

        return candles
