from __future__ import annotations

from datetime import datetime

import pytest

from ai_candle_predictor.domain.entities.candle import CandleStick


class TestCandleStickCreation:
    def test_valid_candle(self, candle: CandleStick) -> None:
        assert isinstance(candle.symbol, str)
        assert isinstance(candle.timestamp, datetime)
        assert candle.open == 50000.0
        assert candle.high == 51000.0
        assert candle.low == 49500.0
        assert candle.close == 50500.0
        assert candle.volume == 1000

    def test_is_bullish_true(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=99.0,
            close=105.0,
            volume=1000,
        )
        assert c.is_bullish is True
        assert c.is_bearish is False

    def test_is_bearish_true(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=105.0,
            high=106.0,
            low=95.0,
            close=100.0,
            volume=1000,
        )
        assert c.is_bearish is True
        assert c.is_bullish is False

    def test_is_bullish_equal(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000,
        )
        assert c.is_bullish is True
        assert c.is_bearish is False

    def test_body_size(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=108.0,
            volume=1000,
        )
        assert c.body_size == pytest.approx(8.0)

    def test_upper_wick(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )
        assert c.upper_wick == pytest.approx(5.0)

    def test_lower_wick(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000,
        )
        assert c.lower_wick == pytest.approx(5.0)

    def test_range(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )
        assert c.range == pytest.approx(20.0)

    def test_return_pct(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )
        assert c.return_pct == pytest.approx(5.0)

    def test_return_pct_zero_open(self) -> None:
        c = CandleStick(
            symbol="BTC-USD",
            timestamp=datetime(2024, 1, 1),
            open=0.0,
            high=10.0,
            low=0.0,
            close=5.0,
            volume=1000,
        )
        assert c.return_pct == pytest.approx(0.0)

    def test_immutable(self, candle: CandleStick) -> None:
        with pytest.raises((TypeError, AttributeError)):
            candle.open = 99999.0

    def test_repr(self, candle: CandleStick) -> None:
        r = repr(candle)
        assert "CandleStick" in r
        assert "BTC-USD" in r


class TestCandleStickValidation:
    def test_negative_open_raises(self) -> None:
        with pytest.raises(ValueError, match="open must be >= 0"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=-1.0,
                high=10.0,
                low=1.0,
                close=5.0,
                volume=1000,
            )

    def test_negative_high_raises(self) -> None:
        with pytest.raises(ValueError, match="high must be >= 0"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=1.0,
                high=-1.0,
                low=1.0,
                close=5.0,
                volume=1000,
            )

    def test_negative_low_raises(self) -> None:
        with pytest.raises(ValueError, match="low must be >= 0"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=1.0,
                high=10.0,
                low=-1.0,
                close=5.0,
                volume=1000,
            )

    def test_negative_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close must be >= 0"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=1.0,
                high=10.0,
                low=1.0,
                close=-5.0,
                volume=1000,
            )

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="volume must be >= 0"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=-100,
            )

    def test_high_lt_max_open_close_raises(self) -> None:
        with pytest.raises(ValueError, match="high.*>=.*max"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=90.0,
                low=95.0,
                close=105.0,
                volume=1000,
            )

    def test_low_gt_min_open_close_raises(self) -> None:
        with pytest.raises(ValueError, match="low.*<=.*min"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=110.0,
                low=102.0,
                close=101.0,
                volume=1000,
            )

    def test_negative_adjusted_close_raises(self) -> None:
        with pytest.raises(ValueError, match="adjusted_close"):
            CandleStick(
                symbol="BTC-USD",
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=1000,
                adjusted_close=-1.0,
            )


class TestCandleStickEquality:
    def test_equal_candles(self) -> None:
        ts = datetime(2024, 1, 15, 12, 0, 0)
        a = CandleStick(
            symbol="BTC-USD",
            timestamp=ts,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1000,
        )
        b = CandleStick(
            symbol="BTC-USD",
            timestamp=ts,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1000,
        )
        assert a == b

    def test_different_symbol_not_equal(self) -> None:
        ts = datetime(2024, 1, 15, 12, 0, 0)
        a = CandleStick(
            symbol="BTC-USD",
            timestamp=ts,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1000,
        )
        b = CandleStick(
            symbol="ETH-USD",
            timestamp=ts,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1000,
        )
        assert a != b
