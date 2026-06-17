from __future__ import annotations

from datetime import datetime

import pytest

from ai_candle_predictor.domain.events import DataIngested, DataStored, DataValidationFailed


class TestDataIngested:
    def test_create(self) -> None:
        event = DataIngested(
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            candle_count=1000,
        )
        assert event.symbol == "BTC-USD"
        assert event.candle_count == 1000

    def test_occurred_at_set(self) -> None:
        event = DataIngested(
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            candle_count=1000,
        )
        assert isinstance(event.occurred_at, datetime)

    def test_immutable(self) -> None:
        event = DataIngested(
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            candle_count=1000,
        )
        with pytest.raises((TypeError, AttributeError)):
            event.candle_count = 999  # type: ignore[misc]


class TestDataValidationFailed:
    def test_create(self) -> None:
        event = DataValidationFailed(
            symbol="ETH-USD",
            reason="missing close price",
        )
        assert event.symbol == "ETH-USD"
        assert event.reason == "missing close price"
        assert event.row is None

    def test_create_with_row(self) -> None:
        event = DataValidationFailed(
            symbol="BTC-USD",
            reason="negative price",
            row={"open": -1.0, "close": 100.0},
        )
        assert event.row == {"open": -1.0, "close": 100.0}

    def test_occurred_at_set(self) -> None:
        event = DataValidationFailed(symbol="AAPL", reason="bad data")
        assert isinstance(event.occurred_at, datetime)


class TestDataStored:
    def test_create(self) -> None:
        event = DataStored(symbol="BTC-USD", path="/data/btc.parquet", row_count=500)
        assert event.path == "/data/btc.parquet"
        assert event.row_count == 500

    def test_occurred_at_set(self) -> None:
        event = DataStored(symbol="ETH-USD", path="/data/eth.parquet", row_count=300)
        assert isinstance(event.occurred_at, datetime)

    def test_immutable(self) -> None:
        event = DataStored(symbol="BTC-USD", path="/data/btc.parquet", row_count=500)
        with pytest.raises((TypeError, AttributeError)):
            event.row_count = 999  # type: ignore[misc]
