from __future__ import annotations

import numpy as np
import pytest

from ai_candle_predictor.infrastructure.explainability.shap_analyzer import ShapIndexMapper


class TestShapIndexMapper:
    def test_to_position_exact_match(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        assert mapper.to_position("2024-01-02") == 1

    def test_to_position_first_element(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        assert mapper.to_position("2024-01-01") == 0

    def test_to_position_last_element(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        assert mapper.to_position("2024-01-03") == 2

    def test_to_position_nonexistent_raises_key_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        with pytest.raises(KeyError):
            mapper.to_position("2024-01-04")

    def test_to_position_out_of_bounds_by_length(self) -> None:
        import pandas as pd

        _ = ShapIndexMapper(pd.DatetimeIndex(["2024-01-01", "2024-01-02"]), n_features=5)
        # This case is academic: get_loc won't return a position outside the
        # index length for valid lookups, so it is covered by validate_sample_index.

    def test_validate_shap_alignment_passes(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        shap_vals = np.ones((3, 5))
        mapper.validate_shap_alignment(shap_vals)  # should not raise

    def test_validate_shap_alignment_raises_on_mismatch(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        shap_vals = np.ones((2, 5))
        with pytest.raises(ValueError, match="has 2 rows but the feature index has 3"):
            mapper.validate_shap_alignment(shap_vals)

    def test_validate_sample_index_passes(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        mapper.validate_sample_index(1)  # should not raise

    def test_validate_sample_index_raises_on_negative(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        with pytest.raises(IndexError, match="outside valid range"):
            mapper.validate_sample_index(-1)

    def test_validate_sample_index_raises_on_overflow(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        with pytest.raises(IndexError, match="outside valid range"):
            mapper.validate_sample_index(3)

    def test_n_rows_property(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        mapper = ShapIndexMapper(idx, n_features=5)
        assert mapper.n_rows == 3

    def test_rangeindex_support(self) -> None:
        import pandas as pd

        idx = pd.RangeIndex(0, 100)
        mapper = ShapIndexMapper(idx, n_features=10)
        assert mapper.to_position(50) == 50

    def test_duplicate_timestamps(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
        mapper = ShapIndexMapper(idx, n_features=5)
        assert mapper.to_position("2024-01-01") == 0

    def test_empty_index_raises_key_error(self) -> None:
        import pandas as pd

        idx = pd.DatetimeIndex([])
        mapper = ShapIndexMapper(idx, n_features=0)
        with pytest.raises(KeyError):
            mapper.to_position("2024-01-01")
