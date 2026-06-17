from __future__ import annotations

import pytest

from ai_candle_predictor.domain.value_objects.symbol import Symbol


class TestSymbolCreation:
    def test_valid_symbol(self) -> None:
        s = Symbol("BTC-USD")
        assert s.value == "BTC-USD"

    def test_index_symbol(self) -> None:
        s = Symbol("^NSEI")
        assert s.value == "^NSEI"

    def test_nse_equity(self) -> None:
        s = Symbol("RELIANCE.NS")
        assert s.value == "RELIANCE.NS"

    def test_forex_pair(self) -> None:
        s = Symbol("EURUSD=X")
        assert s.value == "EURUSD=X"

    def test_crypto_with_usdt(self) -> None:
        s = Symbol("ETH-USDT")
        assert s.value == "ETH-USDT"

    def test_immutable(self) -> None:
        s = Symbol("BTC-USD")
        with pytest.raises((TypeError, AttributeError)):
            s.value = "ETH-USD"

    def test_repr(self) -> None:
        s = Symbol("BTC-USD")
        r = repr(s)
        assert "Symbol" in r
        assert "BTC-USD" in r

    def test_str(self) -> None:
        s = Symbol("BTC-USD")
        assert str(s) == "BTC-USD"


class TestSymbolValidation:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Symbol("")

    def test_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Symbol("   ")

    def test_lowercase_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol must match"):
            Symbol("btc-usd")

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol must match"):
            Symbol("BTC@USD")

    def test_spaces_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol must match"):
            Symbol("BTC USD")


class TestSymbolProperties:
    def test_base_quote_crypto(self) -> None:
        s = Symbol("BTC-USD")
        assert s.base == "BTC"
        assert s.quote == "USD"

    def test_base_quote_no_dash(self) -> None:
        s = Symbol("RELIANCE.NS")
        assert s.base is None
        assert s.quote is None

    def test_base_quote_index(self) -> None:
        s = Symbol("^NSEI")
        assert s.base is None
        assert s.quote is None

    def test_is_crypto_true(self) -> None:
        assert Symbol("BTC-USD").is_crypto is True
        assert Symbol("ETH-USDT").is_crypto is True

    def test_is_crypto_false(self) -> None:
        assert Symbol("RELIANCE.NS").is_crypto is False
        assert Symbol("^NSEI").is_crypto is False
        assert Symbol("AAPL").is_crypto is False

    def test_is_index_true(self) -> None:
        assert Symbol("^NSEI").is_index is True
        assert Symbol("^GSPC").is_index is True

    def test_is_index_false(self) -> None:
        assert Symbol("BTC-USD").is_index is False
        assert Symbol("AAPL").is_index is False


class TestSymbolEquality:
    def test_equal(self) -> None:
        assert Symbol("BTC-USD") == Symbol("BTC-USD")

    def test_not_equal(self) -> None:
        assert Symbol("BTC-USD") != Symbol("ETH-USD")

    def test_hashable(self) -> None:
        s = {Symbol("BTC-USD"), Symbol("BTC-USD")}
        assert len(s) == 1
