from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_candle_predictor.application.use_cases.compute_features import compute_features
from ai_candle_predictor.application.use_cases.generate_labels import (
    generate_labels_for_symbol,
)
from ai_candle_predictor.application.use_cases.train_baseline import train_baseline
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.indicators import IndicatorType, IndicatorValue
from ai_candle_predictor.domain.entities.label import Label, LabeledSample
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.features.computations import compute_all
from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
    ParquetFeatureStore,
)
from ai_candle_predictor.infrastructure.labeling.forward_returns import (
    generate_labels,
    generate_labels_raw,
)
from ai_candle_predictor.infrastructure.labeling.parquet_label_store import (
    ParquetLabelStore,
)
from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_synthetic_candles(n: int, symbol: str = "BTC-USD", seed: int = 42) -> list[CandleStick]:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1)
    prices = 50000 + rng.normal(0, 100, n).cumsum()
    candles = []
    for i in range(n):
        open_ = float(prices[i])
        close_ = float(open_ + rng.normal(0, 20))
        high_ = float(max(open_, close_) + abs(rng.normal(0, 50)))
        low_ = float(min(open_, close_) - abs(rng.normal(0, 50)))
        candles.append(
            CandleStick(
                symbol=symbol,
                timestamp=base + timedelta(hours=i),
                open=open_,
                high=high_,
                low=low_,
                close=close_,
                volume=int(abs(rng.integers(500, 1500))),
            )
        )
    return candles


def _candles_to_df(candles: list[CandleStick]) -> pd.DataFrame:
    dates = [c.timestamp for c in candles]
    return pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ── Feature computation ───────────────────────────────────────────────────────


class TestFeatureComputation:
    def test_compute_all_returns_indicators(self, ohlcv_df: pd.DataFrame) -> None:
        features = compute_all("BTC-USD", ohlcv_df)
        assert len(features) > 0
        assert all(isinstance(f, IndicatorValue) for f in features)
        assert all(f.symbol == "BTC-USD" for f in features)

    def test_compute_all_contains_expected_indicator_types(self, ohlcv_df: pd.DataFrame) -> None:
        features = compute_all("BTC-USD", ohlcv_df)
        types = {f.indicator for f in features}
        assert IndicatorType.SMA in types
        assert IndicatorType.RSI in types
        assert IndicatorType.MACD in types
        assert IndicatorType.BOLLINGER_UPPER in types

    def test_compute_all_empty_df(self) -> None:
        df = pd.DataFrame({"close": []})
        assert compute_all("BTC-USD", df) == []

    def test_compute_values_are_finite(self, ohlcv_df: pd.DataFrame) -> None:
        features = compute_all("BTC-USD", ohlcv_df)
        for f in features:
            assert np.isfinite(f.value)


# ── Label generation ──────────────────────────────────────────────────────────


class TestLabelGeneration:
    def test_generate_labels_returns_samples(self, candle_list: list[CandleStick]) -> None:
        samples = generate_labels(candle_list, horizon=3, threshold=0.005)
        assert len(samples) > 0
        assert all(isinstance(s, LabeledSample) for s in samples)
        assert all(s.symbol == candle_list[0].symbol for s in samples)

    def test_generate_labels_excludes_neutral(self) -> None:
        candles = _make_synthetic_candles(50)
        samples = generate_labels(candles, horizon=5, threshold=0.1)
        # With high threshold, many should be excluded
        for s in samples:
            assert s.label in (Label.UP, Label.DOWN)

    def test_generate_labels_raw_includes_all(self) -> None:
        candles = _make_synthetic_candles(50)
        filtered = generate_labels(candles, horizon=5, threshold=0.001)
        raw = generate_labels_raw(candles, horizon=5, threshold=0.001)
        assert len(raw) >= len(filtered)

    def test_generate_labels_insufficient_data(self) -> None:
        candles = _make_synthetic_candles(3)
        assert generate_labels(candles, horizon=10) == []
        assert generate_labels_raw(candles, horizon=10) == []

    def test_generate_labels_horizon_respected(self) -> None:
        candles = _make_synthetic_candles(100)
        samples = generate_labels_raw(candles, horizon=5, threshold=0.0)
        # All but last 5 should have labels (no nan forward return)
        assert len(samples) == 95
        assert all(s.horizon == 5 for s in samples)

    def test_generate_labels_consistency(self) -> None:
        candles = _make_synthetic_candles(100, seed=1)
        s1 = generate_labels_raw(candles, horizon=5, threshold=0.005)
        s2 = generate_labels_raw(candles, horizon=5, threshold=0.005)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2, strict=False):
            assert a.label == b.label
            assert a.forward_return == pytest.approx(b.forward_return)


# ── End-to-end pipeline: candles → features → labels → model ─────────────────


class TestEndToEndPipeline:
    @pytest.fixture
    def e2e_candles(self) -> list[CandleStick]:
        return _make_synthetic_candles(200, seed=42)

    @pytest.fixture
    def e2e_symbol(self, e2e_candles: list[CandleStick]) -> Symbol:
        return Symbol(e2e_candles[0].symbol)

    def test_full_pipeline(
        self,
        e2e_candles: list[CandleStick],
        e2e_symbol: Symbol,
        tmp_path: Path,
    ) -> None:
        candle_store = ParquetStore(base_dir=tmp_path / "candles")
        feat_store = ParquetFeatureStore(base_dir=tmp_path / "features")
        label_store = ParquetLabelStore(base_dir=tmp_path / "labels")
        model_store = JoblibStore(base_dir=tmp_path / "models")

        # Step 1: save candles
        candle_store.save(e2e_symbol, e2e_candles)
        loaded_candles = candle_store.load(e2e_symbol)
        assert len(loaded_candles) == len(e2e_candles)

        # Step 2: compute features
        feat_count = compute_features(e2e_symbol, candle_store, feat_store)
        assert feat_count > 0

        loaded_feats = feat_store.load(e2e_symbol)
        assert len(loaded_feats) > 0

        # Step 3: generate labels
        label_counts = generate_labels_for_symbol(
            e2e_symbol,
            candle_store,
            label_store,
            horizon=5,
            threshold=0.005,
        )
        assert label_counts["total"] > 0
        total = label_counts["total"]
        assert label_counts["up"] + label_counts["down"] + label_counts["excluded"] == total
        assert label_counts.get("excluded", 0) >= 0

        loaded_labels = label_store.load(e2e_symbol)
        assert len(loaded_labels) == label_counts["up"] + label_counts["down"]

        # Step 4: train baseline model
        pipeline, metrics, model_path = train_baseline(
            symbol=e2e_symbol,
            feature_store=feat_store,
            label_store=label_store,
            model_store=model_store,
            val_split=0.2,
            horizon=5,
            max_iter=1000,
            random_state=42,
        )
        assert metrics.accuracy > 0
        assert metrics.roc_auc > 0
        assert model_path.exists()
