from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def ensure_2d_features(x: Any, name: str = "X") -> NDArray[Any]:
    """Validate that *X* is a 2-D feature matrix and return it as an ndarray.

    Args:
        X: Input data – numpy array, pandas DataFrame, or array-like.
        name: A friendly label used in error messages (default ``"X"``).

    Returns:
        *X* converted to a 2-D :class:`numpy.ndarray`.

    Raises:
        ValueError: If *X* is not 2-D or has zero rows / zero features.
    """
    arr = np.asarray(x)

    if arr.ndim == 2:
        if arr.shape[0] == 0:
            msg = f"{name} has zero rows (shape {arr.shape})"
            raise ValueError(msg)
        if arr.shape[1] == 0:
            msg = f"{name} has zero features (shape {arr.shape})"
            raise ValueError(msg)
        return arr

    if arr.ndim == 1:
        msg = (
            f"{name} is 1-D (shape {arr.shape}) but a 2-D array was expected. "
            f"This usually means a single row was selected with "
            f"``X.iloc[-1]`` (returns a Series) instead of "
            f"``X.iloc[[-1]]`` (returns a DataFrame). "
            f"Use ``X[[col]]`` instead of ``X[col]`` to keep 2-D shape."
        )
        raise ValueError(msg)

    msg = (
        f"{name} has {arr.ndim} dimensions (shape {arr.shape}). "
        f"Expected 2 dimensions (n_samples, n_features)."
    )
    raise ValueError(msg)
