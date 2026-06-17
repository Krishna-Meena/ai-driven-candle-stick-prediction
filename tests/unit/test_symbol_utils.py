from __future__ import annotations

import pytest

from ai_candle_predictor.common.symbol_utils import (
    ALLOWED_TICKERS,
    DISALLOWED_TICKERS,
    is_valid_ticker,
    normalize_symbol,
)


class TestIsValidTicker:
    @pytest.mark.parametrize("ticker", ALLOWED_TICKERS)
    def test_allowed_tickers(self, ticker: str) -> None:
        assert is_valid_ticker(ticker), f"{ticker!r} should be valid"

    @pytest.mark.parametrize("ticker", DISALLOWED_TICKERS)
    def test_disallowed_tickers(self, ticker: str) -> None:
        assert not is_valid_ticker(ticker), f"{ticker!r} should be invalid"

    def test_empty_string(self) -> None:
        assert not is_valid_ticker("")

    def test_lowercase_rejected(self) -> None:
        assert not is_valid_ticker("btc-usd")

    def test_underscore_rejected(self) -> None:
        assert not is_valid_ticker("_NSEI")
        assert not is_valid_ticker("NSEI_")
        assert not is_valid_ticker("BAD_SYMBOL")

    def test_caret_allowed(self) -> None:
        assert is_valid_ticker("^NSEI")
        assert is_valid_ticker("^SPX")

    def test_dot_allowed(self) -> None:
        assert is_valid_ticker("RELIANCE.NS")
        assert is_valid_ticker("TCS.NS")

    def test_hyphen_allowed(self) -> None:
        assert is_valid_ticker("BTC-USD")
        assert is_valid_ticker("ETH-USD")

    def test_equal_allowed(self) -> None:
        assert is_valid_ticker("ABC=DEF")


class TestNormalizeSymbol:
    def test_replaces_caret(self) -> None:
        assert normalize_symbol("^NSEI") == "_NSEI"

    def test_replaces_dot(self) -> None:
        assert normalize_symbol("RELIANCE.NS") == "RELIANCE_NS"

    def test_unchanged(self) -> None:
        assert normalize_symbol("BTC-USD") == "BTC-USD"
        assert normalize_symbol("ETH-USD") == "ETH-USD"

    def test_both_replacements(self) -> None:
        assert normalize_symbol("^.NS") == "__NS"
