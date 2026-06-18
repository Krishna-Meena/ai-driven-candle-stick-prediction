from __future__ import annotations

import numpy as np
import pytest

from ai_candle_predictor.common.feature_utils import ensure_2d_features


class TestEnsure2dFeatures:
    def test_2d_ndarray_passthrough(self) -> None:
        X = np.ones((10, 5))
        result = ensure_2d_features(X)
        assert result is X or np.array_equal(result, X)
        assert result.shape == (10, 5)

    def test_single_sample_dataframe(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = ensure_2d_features(df)
        assert result.shape == (1, 3)

    def test_single_row_from_iloc_double_brackets(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        row = df.iloc[[0]]
        result = ensure_2d_features(row)
        assert result.shape == (1, 2)

    def test_iloc_single_bracket_raises(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        series = df.iloc[0]
        with pytest.raises(ValueError, match="1-D"):
            ensure_2d_features(series)

    def test_series_raises(self) -> None:
        import pandas as pd

        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="1-D"):
            ensure_2d_features(s)

    def test_1d_ndarray_raises(self) -> None:
        X = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="1-D"):
            ensure_2d_features(X)

    def test_3d_ndarray_raises(self) -> None:
        X = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="3 dimensions"):
            ensure_2d_features(X)

    def test_zero_rows_raises(self) -> None:
        X = np.ones((0, 5))
        with pytest.raises(ValueError, match="zero rows"):
            ensure_2d_features(X)

    def test_zero_features_raises(self) -> None:
        X = np.ones((3, 0))
        with pytest.raises(ValueError, match="zero features"):
            ensure_2d_features(X)

    def test_list_of_lists(self) -> None:
        X = [[1, 2], [3, 4], [5, 6]]
        result = ensure_2d_features(X)
        assert result.shape == (3, 2)

    def test_flat_list_raises(self) -> None:
        X = [1, 2, 3]
        with pytest.raises(ValueError, match="1-D"):
            ensure_2d_features(X)

    def test_custom_name_in_error(self) -> None:
        X = np.array([1, 2])
        with pytest.raises(ValueError, match="my_data"):
            ensure_2d_features(X, name="my_data")
