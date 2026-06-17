from __future__ import annotations

from pathlib import Path

import pytest

from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.indicators import IndicatorValue
from ai_candle_predictor.domain.entities.label import LabeledSample
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
    ParquetFeatureStore,
)
from ai_candle_predictor.infrastructure.labeling.parquet_label_store import (
    ParquetLabelStore,
)
from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

# ── ParquetStore (candle persistence) ─────────────────────────────────────────


class TestParquetStore:
    def test_save_and_load(self, candle_list: list[CandleStick], tmp_path: Path) -> None:
        store = ParquetStore(base_dir=tmp_path)
        symbol = Symbol(candle_list[0].symbol)
        path = store.save(symbol, candle_list)
        assert Path(path).exists()

        loaded = store.load(symbol)
        assert len(loaded) == len(candle_list)
        assert loaded[0].symbol == candle_list[0].symbol

    def test_save_appends(self, candle_list: list[CandleStick], tmp_path: Path) -> None:
        store = ParquetStore(base_dir=tmp_path)
        symbol = Symbol(candle_list[0].symbol)
        store.save(symbol, candle_list[:5])
        store.save(symbol, candle_list[5:])
        loaded = store.load(symbol)
        assert len(loaded) == len(candle_list)

    def test_save_dedup(self, candle_list: list[CandleStick], tmp_path: Path) -> None:
        store = ParquetStore(base_dir=tmp_path)
        symbol = Symbol(candle_list[0].symbol)
        store.save(symbol, candle_list)
        store.save(symbol, candle_list)  # same data again
        loaded = store.load(symbol)
        assert len(loaded) == len(candle_list)

    def test_load_empty(self, tmp_path: Path) -> None:
        store = ParquetStore(base_dir=tmp_path)
        loaded = store.load(Symbol("NONEXISTENT"))
        assert loaded == []

    def test_load_with_date_range(self, candle_list: list[CandleStick], tmp_path: Path) -> None:
        store = ParquetStore(base_dir=tmp_path)
        symbol = Symbol(candle_list[0].symbol)
        store.save(symbol, candle_list)

        mid_time = candle_list[10].timestamp
        loaded = store.load(symbol, start_date=candle_list[0].timestamp, end_date=mid_time)
        assert 11 <= len(loaded) <= len(candle_list)
        for c in loaded:
            assert c.timestamp <= mid_time

    def test_roundtrip_preserves_fields(
        self, candle_list: list[CandleStick], tmp_path: Path
    ) -> None:
        store = ParquetStore(base_dir=tmp_path)
        symbol = Symbol(candle_list[0].symbol)
        store.save(symbol, candle_list)
        loaded = store.load(symbol)

        orig = candle_list[0]
        rt = loaded[0]
        assert rt.symbol == orig.symbol
        assert rt.timestamp == orig.timestamp
        assert rt.open == pytest.approx(orig.open)
        assert rt.close == pytest.approx(orig.close)
        assert rt.high == pytest.approx(orig.high)
        assert rt.low == pytest.approx(orig.low)
        assert rt.volume == orig.volume


# ── ParquetFeatureStore ───────────────────────────────────────────────────────


class TestParquetFeatureStore:
    def test_save_and_load(self, indicator_list: list[IndicatorValue], tmp_path: Path) -> None:
        store = ParquetFeatureStore(base_dir=tmp_path)
        symbol = Symbol(indicator_list[0].symbol)
        count = store.save(indicator_list)
        assert count == len(indicator_list)

        loaded = store.load(symbol)
        assert len(loaded) == len(indicator_list)

    def test_load_empty(self, tmp_path: Path) -> None:
        store = ParquetFeatureStore(base_dir=tmp_path)
        loaded = store.load(Symbol("NONEXISTENT"))
        assert loaded == []

    def test_save_appends(self, indicator_list: list[IndicatorValue], tmp_path: Path) -> None:
        store = ParquetFeatureStore(base_dir=tmp_path)
        symbol = Symbol(indicator_list[0].symbol)
        store.save(indicator_list[:10])
        store.save(indicator_list[10:])
        loaded = store.load(symbol)
        assert len(loaded) == len(indicator_list)

    def test_save_empty_returns_zero(self, tmp_path: Path) -> None:
        store = ParquetFeatureStore(base_dir=tmp_path)
        assert store.save([]) == 0

    def test_roundtrip_preserves_fields(
        self, indicator_list: list[IndicatorValue], tmp_path: Path
    ) -> None:
        store = ParquetFeatureStore(base_dir=tmp_path)
        symbol = Symbol(indicator_list[0].symbol)
        store.save(indicator_list)
        loaded = store.load(symbol)

        orig = indicator_list[0]
        rt = loaded[0]
        assert rt.symbol == orig.symbol
        assert rt.indicator == orig.indicator
        assert rt.value == pytest.approx(orig.value)


# ── ParquetLabelStore ─────────────────────────────────────────────────────────


class TestParquetLabelStore:
    def test_save_and_load(self, label_list: list[LabeledSample], tmp_path: Path) -> None:
        store = ParquetLabelStore(base_dir=tmp_path)
        symbol = Symbol(label_list[0].symbol)
        count = store.save(label_list)
        assert count == len(label_list)

        loaded = store.load(symbol)
        assert len(loaded) == len(label_list)

    def test_load_empty(self, tmp_path: Path) -> None:
        store = ParquetLabelStore(base_dir=tmp_path)
        loaded = store.load(Symbol("NONEXISTENT"))
        assert loaded == []

    def test_save_appends(self, label_list: list[LabeledSample], tmp_path: Path) -> None:
        store = ParquetLabelStore(base_dir=tmp_path)
        symbol = Symbol(label_list[0].symbol)
        store.save(label_list[:5])
        store.save(label_list[5:])
        loaded = store.load(symbol)
        assert len(loaded) == len(label_list)

    def test_save_empty_returns_zero(self, tmp_path: Path) -> None:
        store = ParquetLabelStore(base_dir=tmp_path)
        assert store.save([]) == 0

    def test_roundtrip_preserves_fields(
        self, label_list: list[LabeledSample], tmp_path: Path
    ) -> None:
        store = ParquetLabelStore(base_dir=tmp_path)
        symbol = Symbol(label_list[0].symbol)
        store.save(label_list)
        loaded = store.load(symbol)

        orig = label_list[0]
        rt = loaded[0]
        assert rt.symbol == orig.symbol
        assert rt.label == orig.label
        assert rt.forward_return == pytest.approx(orig.forward_return)
        assert rt.horizon == orig.horizon
        assert rt.close == pytest.approx(orig.close)
