from __future__ import annotations

import re

TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")

ALLOWED_TICKERS: list[str] = [
    "BTC-USD",
    "ETH-USD",
    "^NSEI",
    "RELIANCE.NS",
]

DISALLOWED_TICKERS: list[str] = [
    "_NSEI",
]


def normalize_symbol(value: str) -> str:
    """Convert a symbol to a filesystem-safe filename part.

    ``^`` and ``.`` are replaced with ``_`` so they work cross-platform.
    """
    return value.replace("^", "_").replace(".", "_")


def is_valid_ticker(value: str) -> bool:
    """Check whether a string is a valid ticker per the domain Symbol regex."""
    return bool(TICKER_PATTERN.match(value))
