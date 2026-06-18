from __future__ import annotations

import pytest

from ai_candle_predictor.infrastructure.explainability.shap_analyzer import get_shap_position


class TestGetShapPosition:
    def test_exact_match_int_position(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        pos = get_shap_position(idx, "2024-01-02", n_samples=3)
        assert pos == 1

    def test_first_element(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        pos = get_shap_position(idx, "2024-01-01", n_samples=3)
        assert pos == 0

    def test_last_element(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        pos = get_shap_position(idx, "2024-01-03", n_samples=3)
        assert pos == 2

    def test_large_feature_count_matches_shap_count(self) -> None:
        import pandas as pd

        n = 1104
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        idx = pd.DatetimeIndex(dates)
        pos = get_shap_position(idx, dates[500], n_samples=n)
        assert pos == 500

    def test_small_shap_count_raises_index_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        with pytest.raises(IndexError, match="out of bounds"):
            get_shap_position(idx, "2024-01-03", n_samples=2)

    def test_nonexistent_timestamp_raises_key_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        with pytest.raises(KeyError, match="2024-01-04"):
            get_shap_position(idx, "2024-01-04", n_samples=3)

    def test_duplicate_timestamps_uses_first_position(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
        pos = get_shap_position(idx, "2024-01-01", n_samples=3)
        assert pos == 0

    def test_rangeindex_support(self) -> None:
        import pandas as pd

        idx = pd.RangeIndex(0, 100)
        pos = get_shap_position(idx, 50, n_samples=100)
        assert pos == 50

    def test_string_index_support(self) -> None:
        import pandas as pd

        idx = pd.Index(["a", "b", "c"])
        pos = get_shap_position(idx, "b", n_samples=3)
        assert pos == 1

    def test_negative_position_raises_index_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        with pytest.raises(IndexError, match="out of bounds"):
            # n_samples=0 forces pos >= 0 check to fail even for pos=0
            get_shap_position(idx, "2024-01-01", n_samples=0)

    def test_empty_index_raises_key_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex([])
        with pytest.raises(KeyError):
            get_shap_position(idx, "2024-01-01", n_samples=0)
