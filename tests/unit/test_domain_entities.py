from __future__ import annotations

from datetime import datetime

import pytest

from ai_candle_predictor.domain.entities.indicators import IndicatorType, IndicatorValue
from ai_candle_predictor.domain.entities.label import Label, LabeledSample
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.entities.patterns import CandlePattern, PatternMatch

# ── IndicatorType ─────────────────────────────────────────────────────────────


class TestIndicatorType:
    def test_members(self) -> None:
        assert IndicatorType.SMA.value == 1
        assert IndicatorType.EMA.value == 2
        assert IndicatorType.RSI.value == 3
        assert IndicatorType.MACD.value == 4

    def test_bollinger_members(self) -> None:
        assert IndicatorType.BOLLINGER_UPPER is not None
        assert IndicatorType.BOLLINGER_MIDDLE is not None
        assert IndicatorType.BOLLINGER_LOWER is not None

    def test_adx_members(self) -> None:
        assert IndicatorType.ADX is not None
        assert IndicatorType.ADX_PLUS_DI is not None
        assert IndicatorType.ADX_MINUS_DI is not None

    def test_stochastic_members(self) -> None:
        assert IndicatorType.STOCHASTIC_K is not None
        assert IndicatorType.STOCHASTIC_D is not None

    def test_count(self) -> None:
        assert len(IndicatorType) == 17

    def test_from_string(self) -> None:
        assert IndicatorType["RSI"] == IndicatorType.RSI
        assert IndicatorType["MACD_HISTOGRAM"] == IndicatorType.MACD_HISTOGRAM

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(KeyError):
            IndicatorType["INVALID"]


# ── IndicatorValue ────────────────────────────────────────────────────────────


class TestIndicatorValueCreation:
    def test_create(self, indicator_value: IndicatorValue) -> None:
        assert indicator_value.symbol == "BTC-USD"
        assert indicator_value.indicator == IndicatorType.RSI
        assert indicator_value.value == pytest.approx(65.5)

    def test_immutable(self, indicator_value: IndicatorValue) -> None:
        with pytest.raises((TypeError, AttributeError)):
            indicator_value.value = 99.0

    def test_repr(self, indicator_value: IndicatorValue) -> None:
        r = repr(indicator_value)
        assert "IndicatorValue" in r
        assert "RSI" in r

    def test_frozen_hash(self) -> None:
        a = IndicatorValue(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            indicator=IndicatorType.SMA,
            value=75.0,
        )
        b = IndicatorValue(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            indicator=IndicatorType.SMA,
            value=75.0,
        )
        assert len({a, b}) == 1


# ── CandlePattern ─────────────────────────────────────────────────────────────


class TestCandlePattern:
    def test_members(self) -> None:
        assert CandlePattern.NONE.value == 1
        assert CandlePattern.BULLISH.value == 2
        assert CandlePattern.BEARISH.value == 3
        assert CandlePattern.DOJI.value == 4

    def test_count(self) -> None:
        assert len(CandlePattern) == 8

    def test_engulfing_members(self) -> None:
        assert CandlePattern.BULLISH_ENGULFING is not None
        assert CandlePattern.BEARISH_ENGULFING is not None

    def test_from_string(self) -> None:
        assert CandlePattern["HAMMER"] == CandlePattern.HAMMER
        assert CandlePattern["SHOOTING_STAR"] == CandlePattern.SHOOTING_STAR


# ── PatternMatch ──────────────────────────────────────────────────────────────


class TestPatternMatchCreation:
    def test_create(self, pattern_match: PatternMatch) -> None:
        assert pattern_match.pattern == CandlePattern.DOJI
        assert pattern_match.symbol == "BTC-USD"
        assert pattern_match.confidence == pytest.approx(0.85)
        assert pattern_match.description == ""

    def test_create_with_description(self) -> None:
        pm = PatternMatch(
            symbol="ETH-USD",
            timestamp=datetime(2024, 1, 1),
            pattern=CandlePattern.BULLISH,
            confidence=0.9,
            description="Strong bullish signal",
        )
        assert pm.description == "Strong bullish signal"

    def test_default_description(self) -> None:
        pm = PatternMatch(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            pattern=CandlePattern.NONE,
        )
        assert pm.description == ""

    def test_default_confidence(self) -> None:
        pm = PatternMatch(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            pattern=CandlePattern.NONE,
        )
        assert pm.confidence == pytest.approx(1.0)

    def test_immutable(self, pattern_match: PatternMatch) -> None:
        with pytest.raises((TypeError, AttributeError)):
            pattern_match.pattern = CandlePattern.BULLISH


# ── Label ─────────────────────────────────────────────────────────────────────


class TestLabel:
    def test_members(self) -> None:
        assert Label.UP.value == 1
        assert Label.DOWN.value == 2
        assert Label.EXCLUDED.value == 3

    def test_as_int_up(self) -> None:
        assert Label.UP.as_int == 1

    def test_as_int_down(self) -> None:
        assert Label.DOWN.as_int == 0

    def test_as_int_excluded(self) -> None:
        assert Label.EXCLUDED.as_int is None


# ── LabeledSample ─────────────────────────────────────────────────────────────


class TestLabeledSampleCreation:
    def test_create(self, labeled_sample: LabeledSample) -> None:
        assert labeled_sample.symbol == "BTC-USD"
        assert labeled_sample.label == Label.UP
        assert labeled_sample.forward_return == pytest.approx(0.015)

    def test_down_sample(self) -> None:
        ls = LabeledSample(
            symbol="ETH-USD",
            timestamp=datetime(2024, 1, 1),
            label=Label.DOWN,
            forward_return=-0.02,
            horizon=5,
            close=100.0,
        )
        assert ls.label == Label.DOWN
        assert ls.forward_return == pytest.approx(-0.02)

    def test_excluded_sample(self) -> None:
        ls = LabeledSample(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            label=Label.EXCLUDED,
            forward_return=0.0,
            horizon=5,
            close=100.0,
        )
        assert ls.label == Label.EXCLUDED

    def test_immutable(self, labeled_sample: LabeledSample) -> None:
        with pytest.raises((TypeError, AttributeError)):
            labeled_sample.label = Label.DOWN


# ── ClassificationMetrics ─────────────────────────────────────────────────────


class TestClassificationMetricsCreation:
    def test_create(self, perfect_metrics: ClassificationMetrics) -> None:
        assert perfect_metrics.accuracy == pytest.approx(1.0)
        assert perfect_metrics.support == 100

    def test_create_random(self, random_metrics: ClassificationMetrics) -> None:
        assert random_metrics.accuracy == pytest.approx(0.5)
        assert random_metrics.roc_auc == pytest.approx(0.5)

    def test_immutable(self, perfect_metrics: ClassificationMetrics) -> None:
        with pytest.raises((TypeError, AttributeError)):
            perfect_metrics.accuracy = 0.0

    def test_repr(self, perfect_metrics: ClassificationMetrics) -> None:
        r = repr(perfect_metrics)
        assert "ClassificationMetrics" in r
        assert "1.0000" in r
        assert "support  =100" in r

    def test_list_metric(self, perfect_metrics: ClassificationMetrics) -> None:
        pm = perfect_metrics
        # All range-bound metrics should be between 0 and 1
        for attr in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= getattr(pm, attr) <= 1.0
        assert pm.support >= 0
