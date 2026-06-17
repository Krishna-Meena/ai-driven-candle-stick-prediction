from __future__ import annotations

from datetime import date, datetime
from typing import Any


def ensure_date(
    value: Any,
    fallback: date | None = None,
) -> date | None:
    """Convert a date-like value to ``datetime.date``.

    Supports:
    - ``datetime.datetime`` / ``datetime.date``
    - ``pandas.Timestamp``
    - ``numpy.datetime64``
    - ISO-format strings (e.g. ``"2024-01-15"``)
    - ``None`` (returns *fallback*)

    Returns ``None`` (or *fallback*) when conversion is not possible.
    """
    if value is None:
        return fallback

    if isinstance(value, datetime):
        return date(value.year, value.month, value.day)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except (ValueError, TypeError):
            return fallback

    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return date(value.year, value.month, value.day)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.datetime64):
            ts = pd.Timestamp(value)
            return date(ts.year, ts.month, ts.day)
    except ImportError:
        pass

    return fallback
