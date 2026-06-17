from __future__ import annotations

from datetime import date, datetime

import pytest

from ai_candle_predictor.common.date_utils import ensure_date


class TestEnsureDate:
    def test_none_returns_none(self) -> None:
        assert ensure_date(None) is None

    def test_none_with_fallback(self) -> None:
        assert ensure_date(None, fallback=date(2024, 1, 1)) == date(2024, 1, 1)

    def test_datetime_date(self) -> None:
        d = date(2023, 6, 15)
        assert ensure_date(d) is d

    def test_datetime_datetime(self) -> None:
        dt = datetime(2023, 6, 15, 10, 30, 0)
        assert ensure_date(dt) == date(2023, 6, 15)

    def test_iso_string(self) -> None:
        assert ensure_date("2024-01-15") == date(2024, 1, 15)

    def test_iso_string_with_time(self) -> None:
        assert ensure_date("2024-01-15T14:30:00") == date(2024, 1, 15)

    def test_invalid_string_returns_fallback(self) -> None:
        assert ensure_date("not-a-date", fallback=date(2020, 1, 1)) == date(2020, 1, 1)

    def test_pandas_timestamp(self) -> None:
        pd = pytest.importorskip("pandas")
        ts = pd.Timestamp("2023-12-25")
        assert ensure_date(ts) == date(2023, 12, 25)

    def test_pandas_timestamp_with_tz(self) -> None:
        pd = pytest.importorskip("pandas")
        ts = pd.Timestamp("2023-12-25", tz="UTC")
        assert ensure_date(ts) == date(2023, 12, 25)

    def test_numpy_datetime64(self) -> None:
        np = pytest.importorskip("numpy")
        dt64 = np.datetime64("2023-06-15")
        assert ensure_date(dt64) == date(2023, 6, 15)

    def test_integer_returns_fallback(self) -> None:
        assert ensure_date(42, fallback=date(2020, 1, 1)) == date(2020, 1, 1)

    def test_empty_string_returns_fallback(self) -> None:
        assert ensure_date("", fallback=date(2020, 1, 1)) == date(2020, 1, 1)
