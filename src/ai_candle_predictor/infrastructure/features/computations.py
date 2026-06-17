from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandas_ta as ta

from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.indicators import IndicatorType, IndicatorValue

log = get_logger(__name__)


def compute_all(symbol: str, df: pd.DataFrame) -> list[IndicatorValue]:
    if df.empty:
        return []

    features: list[IndicatorValue] = []
    append = features.append

    timestamps = df.index
    _sma(df, symbol, timestamps, append)
    _ema(df, symbol, timestamps, append)
    _rsi(df, symbol, timestamps, append)
    _macd(df, symbol, timestamps, append)
    _bollinger(df, symbol, timestamps, append)
    _atr(df, symbol, timestamps, append)
    _adx(df, symbol, timestamps, append)
    _stochastic(df, symbol, timestamps, append)

    log.debug("all indicators computed", symbol=symbol, count=len(features))
    return features


def _emit(
    series: pd.Series,
    symbol: str,
    timestamps: pd.Index,
    indicator: IndicatorType,
    append: Callable[[IndicatorValue], None],
) -> None:
    for ts, val in zip(timestamps, series.reindex(timestamps), strict=False):
        if pd.notna(val):
            append(
                IndicatorValue(
                    symbol=symbol,
                    timestamp=ts,
                    indicator=indicator,
                    value=float(val),
                )
            )


def _sma(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    for window in (10, 20, 50, 200):
        result = ta.sma(df["close"], length=window)
        if result is not None:
            _emit(result, symbol, timestamps, IndicatorType.SMA, append)


def _ema(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    for window in (12, 26, 50):
        result = ta.ema(df["close"], length=window)
        if result is not None:
            _emit(result, symbol, timestamps, IndicatorType.EMA, append)


def _rsi(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    result = ta.rsi(df["close"], length=14)
    if result is not None:
        _emit(result, symbol, timestamps, IndicatorType.RSI, append)


def _macd(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    macd_df = ta.macd(df["close"])
    if macd_df is None or macd_df.empty:
        return

    for col, indicator in (
        ("MACD_12_26_9", IndicatorType.MACD),
        ("MACDs_12_26_9", IndicatorType.MACD_SIGNAL),
        ("MACDh_12_26_9", IndicatorType.MACD_HISTOGRAM),
    ):
        if col in macd_df.columns:
            _emit(macd_df[col], symbol, timestamps, indicator, append)


def _bollinger(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    bbands_df = ta.bbands(df["close"], length=20, std=2)  # type: ignore[arg-type]
    if bbands_df is None or bbands_df.empty:
        return

    for col, indicator in (
        ("BBL_20_2.0_2.0", IndicatorType.BOLLINGER_LOWER),
        ("BBM_20_2.0_2.0", IndicatorType.BOLLINGER_MIDDLE),
        ("BBU_20_2.0_2.0", IndicatorType.BOLLINGER_UPPER),
        ("BBB_20_2.0_2.0", IndicatorType.BOLLINGER_BANDWIDTH),
        ("BBP_20_2.0_2.0", IndicatorType.BOLLINGER_PERCENT_B),
    ):
        if col in bbands_df.columns:
            _emit(bbands_df[col], symbol, timestamps, indicator, append)


def _atr(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    result = ta.atr(df["high"], df["low"], df["close"], length=14)
    if result is not None:
        _emit(result, symbol, timestamps, IndicatorType.ATR, append)


def _adx(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is None or adx_df.empty:
        return

    for col, indicator in (
        ("ADX_14", IndicatorType.ADX),
        ("DMP_14", IndicatorType.ADX_PLUS_DI),
        ("DMN_14", IndicatorType.ADX_MINUS_DI),
    ):
        if col in adx_df.columns:
            _emit(adx_df[col], symbol, timestamps, indicator, append)


def _stochastic(
    df: pd.DataFrame,
    symbol: str,
    timestamps: pd.Index,
    append: Callable[[IndicatorValue], None],
) -> None:
    stoch_df = ta.stoch(df["high"], df["low"], df["close"])
    if stoch_df is None or stoch_df.empty:
        return

    for col, indicator in (
        ("STOCHk_14_3_3", IndicatorType.STOCHASTIC_K),
        ("STOCHd_14_3_3", IndicatorType.STOCHASTIC_D),
    ):
        if col in stoch_df.columns:
            _emit(stoch_df[col], symbol, timestamps, indicator, append)
