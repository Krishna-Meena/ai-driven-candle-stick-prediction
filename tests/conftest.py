from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.indicators import IndicatorType, IndicatorValue
from ai_candle_predictor.domain.entities.label import Label, LabeledSample
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.entities.patterns import CandlePattern, PatternMatch

# ── Symbols (as strings, matching entity schema) ───────────────────────────


@pytest.fixture
def symbol_btc_str() -> str:
    return "BTC-USD"


@pytest.fixture
def symbol_eth_str() -> str:
    return "ETH-USD"


# ── Candles ─────────────────────────────────────────────────────────────────


@pytest.fixture
def candle_ohlc() -> tuple[float, float, float, float, float]:
    return (50000.0, 51000.0, 49500.0, 50500.0, 1000.0)


@pytest.fixture
def candle(
    symbol_btc_str: str, candle_ohlc: tuple[float, float, float, float, float]
) -> CandleStick:
    ts = datetime(2024, 1, 15, 12, 0, 0)
    return CandleStick(
        symbol=symbol_btc_str,
        timestamp=ts,
        open=candle_ohlc[0],
        high=candle_ohlc[1],
        low=candle_ohlc[2],
        close=candle_ohlc[3],
        volume=int(candle_ohlc[4]),
    )


@pytest.fixture
def candle_doji(symbol_btc_str: str) -> CandleStick:
    return CandleStick(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 13, 0, 0),
        open=100.0,
        high=102.0,
        low=98.0,
        close=100.5,
        volume=500,
    )


@pytest.fixture
def candle_bullish(symbol_btc_str: str) -> CandleStick:
    return CandleStick(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 14, 0, 0),
        open=100.0,
        high=110.0,
        low=99.0,
        close=108.0,
        volume=1000,
    )


@pytest.fixture
def candle_bearish(symbol_btc_str: str) -> CandleStick:
    return CandleStick(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 15, 0, 0),
        open=108.0,
        high=109.0,
        low=95.0,
        close=97.0,
        volume=800,
    )


@pytest.fixture
def candle_list(symbol_btc_str: str) -> list[CandleStick]:
    candles = []
    base = datetime(2024, 1, 10, 0, 0, 0)
    for i in range(20):
        candles.append(
            CandleStick(
                symbol=symbol_btc_str,
                timestamp=base + timedelta(hours=i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=int(500 + i * 10),
            )
        )
    return candles


# ── OHLCV DataFrame ─────────────────────────────────────────────────────────


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(hours=i) for i in range(n)]
    opens = 50000 + rng.normal(0, 100, n).cumsum()
    closes = opens + rng.normal(0, 20, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 50, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 50, n))
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.abs(rng.normal(1000, 200, n)),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ── Indicators ──────────────────────────────────────────────────────────────


@pytest.fixture
def indicator_value(symbol_btc_str: str) -> IndicatorValue:
    return IndicatorValue(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        indicator=IndicatorType.RSI,
        value=65.5,
    )


@pytest.fixture
def indicator_list(symbol_btc_str: str) -> list[IndicatorValue]:
    base = datetime(2024, 1, 10, 0, 0, 0)
    result = []
    for i, ind in enumerate(IndicatorType):
        for j in range(5):
            result.append(
                IndicatorValue(
                    symbol=symbol_btc_str,
                    timestamp=base + timedelta(hours=j),
                    indicator=ind,
                    value=float(50 + i * 10 + j),
                )
            )
    return result


# ── Labels ──────────────────────────────────────────────────────────────────


@pytest.fixture
def labeled_sample(symbol_btc_str: str) -> LabeledSample:
    return LabeledSample(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        label=Label.UP,
        forward_return=0.015,
        horizon=5,
        close=50500.0,
    )


@pytest.fixture
def label_list(symbol_btc_str: str) -> list[LabeledSample]:
    base = datetime(2024, 1, 10, 0, 0, 0)
    return [
        LabeledSample(
            symbol=symbol_btc_str,
            timestamp=base + timedelta(hours=i),
            label=Label.UP if i % 2 == 0 else Label.DOWN,
            forward_return=0.01 if i % 2 == 0 else -0.01,
            horizon=5,
            close=100.0 + i,
        )
        for i in range(20)
    ]


# ── Metrics ─────────────────────────────────────────────────────────────────


@pytest.fixture
def perfect_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=1.0,
        support=100,
    )


@pytest.fixture
def random_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        roc_auc=0.5,
        support=100,
    )


# ── Patterns ────────────────────────────────────────────────────────────────


@pytest.fixture
def pattern_match(symbol_btc_str: str) -> PatternMatch:
    return PatternMatch(
        symbol=symbol_btc_str,
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        pattern=CandlePattern.DOJI,
        confidence=0.85,
    )


# ── Temp directories ────────────────────────────────────────────────────────


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def temp_reports_dir(tmp_path: Path) -> Path:
    d = tmp_path / "reports"
    d.mkdir(parents=True)
    return d
