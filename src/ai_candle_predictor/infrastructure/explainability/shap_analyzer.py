from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pandas import Index as PdIndex
from sklearn.pipeline import Pipeline

from ai_candle_predictor.application.ports.image_storage import ImageStorage
from ai_candle_predictor.common.feature_utils import ensure_2d_features
from ai_candle_predictor.common.logging import get_logger

matplotlib.use("Agg")

log = get_logger(__name__)

try:
    import shap  # noqa: F401

    _SHAP_AVAILABLE = True
except ModuleNotFoundError:
    _SHAP_AVAILABLE = False


def check_shap() -> None:
    """Raise RuntimeError with install instructions if SHAP is unavailable."""
    if not _SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed.  Run: uv add shap")


# ---------------------------------------------------------------------------
# ShapIndexMapper — validate and convert DataFrame index → positional index
# ---------------------------------------------------------------------------


class ShapIndexMapper:
    """Maps DataFrame index labels to 0-based positional indices for SHAP arrays.

    Validates alignment between the feature matrix, its index, and SHAP output
    at every step so that index-vs-position mismatches are caught early.
    """

    def __init__(self, feature_index: PdIndex, n_features: int) -> None:
        self._index = feature_index
        self._n_features = n_features  # number of columns (used for shape checks)

    # -- Public API -------------------------------------------------------

    def to_position(self, timestamp: Any) -> int:
        """Convert a DataFrame index label to a 0-based positional index.

        Raises
        ------
        KeyError
            *timestamp* is not present in the stored index.
        IndexError
            Resolved position is outside the valid range.
        """
        raw: Any = self._index.get_loc(timestamp)

        if isinstance(raw, slice):
            pos: int = raw.start
        elif isinstance(raw, (np.ndarray, list)):
            pos = int(raw[0])
        else:
            pos = int(raw)

        if not 0 <= pos < len(self._index):
            raise IndexError(
                f"ShapIndexMapper: converted position {pos} is outside "
                f"[0, {len(self._index)}) for index of length {len(self._index)}. "
                f"Timestamp={timestamp!r}, index_dtype={self._index.dtype}."
            )
        return pos

    def validate_shap_alignment(
        self, shap_values: NDArray[Any], label: str = "shap_values"
    ) -> None:
        """Assert that *shap_values* has the same first dimension as the feature index."""
        n_expected = len(self._index)
        n_actual = shap_values.shape[0]
        if n_actual != n_expected:
            raise ValueError(
                f"{label} has {n_actual} rows but the feature index has {n_expected}. "
                "This means the SHAP computation used a different dataset than "
                "the one the index was built from. Check that the same X was passed "
                "to both shap_analysis() (or explain_single_sample()) and the mapper."
            )

    def validate_sample_index(self, position: int) -> None:
        """Assert that *position* is a valid index into the feature matrix."""
        n = len(self._index)
        if not 0 <= position < n:
            raise IndexError(
                f"Sample position {position} is outside valid range [0, {n}) "
                f"for feature index of length {n}."
            )

    @property
    def n_rows(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shap_analysis(
    symbol: str,
    pipeline: Pipeline,
    X: Any,
    feature_names: list[str],
    image_storage: ImageStorage,
    max_samples: int = 100,
    mapper: ShapIndexMapper | None = None,
) -> dict[str, object]:
    check_shap()
    import shap

    X_arr = ensure_2d_features(X, name="X")
    log.info(
        "shap analysis started",
        symbol=symbol,
        samples=X_arr.shape[0],
        features=X_arr.shape[1],
        X_type=type(X).__name__,
        X_dtype=str(X_arr.dtype),
    )

    X_bg = _select_background(X_arr, max_samples)
    explainer, X_bg_used = _build_explainer(pipeline, X_bg)

    log.debug(
        "running shap_values",
        X_shape=X_arr.shape,
        X_ndim=X_arr.ndim,
        explainer_type=type(explainer).__name__,
    )
    shap_vals = explainer.shap_values(X_arr)
    log.debug("shap_values returned", shap_type=type(shap_vals).__name__)

    vals, base = _normalize_shap_output(shap_vals)
    log.debug("normalized shap output", vals_shape=vals.shape, base_shape=base.shape)

    if mapper is not None:
        mapper.validate_shap_alignment(vals, label="shap_analysis output")

    bar_explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=X_arr,
        feature_names=feature_names,
    )

    summary_path = _save_beeswarm(vals, X_bg_used, feature_names, symbol, image_storage)
    bar_path = _save_bar_plot(bar_explanation, symbol, image_storage)

    mean_abs = np.mean(np.abs(vals), axis=0)
    ranked_idx = np.argsort(mean_abs)[::-1]
    ranking: dict[str, float] = {}
    for idx in ranked_idx:
        ranking[feature_names[idx]] = round(float(mean_abs[idx]), 6)

    local = _local_explanations(bar_explanation, feature_names, symbol, image_storage)

    return {
        "summary_plot": summary_path,
        "bar_plot": bar_path,
        "global_ranking": ranking,
        "local_explanations": local,
        "samples_analyzed": int(vals.shape[0]),
    }


def explain_single_sample(
    symbol: str,
    pipeline: Pipeline,
    X: Any,
    feature_names: list[str],
    sample_index: int,
    image_storage: ImageStorage,
    background_samples: int = 100,
    mapper: ShapIndexMapper | None = None,
) -> dict[str, object]:
    check_shap()

    X_arr = ensure_2d_features(X, name="X")
    log.debug(
        "explain_single_sample started",
        X_shape=X_arr.shape,
        X_ndim=X_arr.ndim,
        X_type=type(X).__name__,
        sample_index=sample_index,
    )

    if mapper is not None:
        mapper.validate_sample_index(sample_index)

    X_bg = _select_background(X_arr, background_samples)
    explainer, X_bg_used = _build_explainer(pipeline, X_bg)

    log.debug(
        "running shap_values for local explanation",
        X_shape=X_arr.shape,
        explainer_type=type(explainer).__name__,
    )
    shap_vals = explainer.shap_values(X_arr)
    log.debug("shap_values returned", shap_type=type(shap_vals).__name__)

    vals, base = _normalize_shap_output(shap_vals)
    log.debug("normalized shap output", vals_shape=vals.shape, base_shape=base.shape)

    if mapper is not None:
        mapper.validate_shap_alignment(vals, label="explain_single_sample output")

    sample_values = vals[sample_index : sample_index + 1]
    sample_base = float(base[sample_index] if isinstance(base, np.ndarray) else base)
    pred = float(sample_values.sum() + sample_base)

    sample_explanation = shap.Explanation(
        values=sample_values,
        base_values=sample_base,
        data=X_arr[sample_index : sample_index + 1],
        feature_names=feature_names,
    )

    waterfall_path = _save_waterfall(
        sample_explanation, feature_names, sample_index, symbol, image_storage
    )
    force_html = _render_force_html(sample_explanation, feature_names)

    vals_i = sample_values[0]
    top_idx = np.argsort(np.abs(vals_i))[::-1][:10]
    top_features = [
        {"feature": feature_names[j], "shap_value": round(float(vals_i[j]), 6)} for j in top_idx
    ]

    return {
        "base_value": sample_base,
        "prediction": pred,
        "waterfall_plot": str(waterfall_path),
        "force_html": force_html,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_shap_output(shap_values: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Convert any SHAP explainer output to a uniform (values, base_values) pair.

    Handles:
    * ``shap.Explanation`` objects (SHAP >= 0.42)
    * ``numpy.ndarray`` (older SHAP or raw return)
    * ``list`` of arrays or Explanations (multi-class / multi-output)

    Returns
    -------
    values : ndarray, shape ``(n_samples, n_features)``
    base_values : ndarray, shape ``(n_samples,)``
    """
    import shap

    log.debug("normalizing SHAP output", input_type=type(shap_values).__name__)

    if isinstance(shap_values, shap.Explanation):
        vals = np.asarray(shap_values.values, dtype=np.float64)
        base = np.asarray(shap_values.base_values, dtype=np.float64)
    elif isinstance(shap_values, list):
        log.debug("SHAP output is a list", length=len(shap_values))
        extracted: list[NDArray[np.float64]] = []
        extracted_base: list[NDArray[np.float64]] = []
        for item in shap_values:
            if isinstance(item, shap.Explanation):
                extracted.append(np.asarray(item.values, dtype=np.float64))
                extracted_base.append(np.asarray(item.base_values, dtype=np.float64))
            else:
                extracted.append(np.asarray(item, dtype=np.float64))
                extracted_base.append(np.zeros(extracted[-1].shape[0]))
        vals = np.stack(extracted, axis=0)
        base = np.stack(extracted_base, axis=0)
    elif isinstance(shap_values, np.ndarray):
        vals = np.asarray(shap_values, dtype=np.float64)
        base = np.zeros(vals.shape[0], dtype=np.float64)
    else:
        raise TypeError(f"unexpected SHAP output type: {type(shap_values)}")

    vals, base = _extract_binary(vals, base)
    return vals, base


def _extract_binary(vals: NDArray[Any], base: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract the positive-class (class 1) values from a binary classifier output.

    *Values* and *base* are already ndarrays at this point.
    """
    if vals.ndim == 3:
        log.debug("extracting class 1 from 3-D values", shape=vals.shape)
        vals = vals[:, :, 1]
    if base.ndim == 2:
        log.debug("extracting class 1 from 2-D base_values", shape=base.shape)
        base = base[:, 1]

    return vals, base


def _select_background(X: NDArray[Any], max_samples: int = 100) -> NDArray[Any]:
    """Subsample *X* to at most *max_samples* rows, preserving 2-D shape."""
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        bg = X[idx]
        log.debug("background subsampled", original=len(X), selected=len(bg), shape=bg.shape)
        return bg
    log.debug("background full dataset used", rows=len(X), shape=X.shape)
    return X


def _build_explainer(pipeline: Pipeline, X_bg: Any) -> tuple[Any, Any]:
    import shap

    clf = pipeline.named_steps["classifier"]
    clf_name = clf.__class__.__name__

    if clf_name in ("RandomForestClassifier", "XGBClassifier"):
        log.info("using TreeExplainer", model=clf_name)
        return shap.TreeExplainer(clf), X_bg

    if clf_name == "LogisticRegression":
        log.info("using LinearExplainer", model=clf_name)
        if "scaler" in pipeline.named_steps:
            X_scaled = pipeline.named_steps["scaler"].transform(X_bg)
        else:
            X_scaled = X_bg
        return shap.LinearExplainer(clf, X_scaled), X_scaled

    raise ValueError(f"unsupported model type: {clf_name}")


# ---------------------------------------------------------------------------
# Plotting helpers (receive ``shap.Explanation`` objects)
# ---------------------------------------------------------------------------


def _save_waterfall(
    explanation: Any,
    _feature_names: list[str],
    _sample_index: int,
    symbol: str,
    image_storage: ImageStorage,
) -> Path:
    import shap

    shap.plots.waterfall(explanation[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return path


def _render_force_html(explanation: Any, feature_names: list[str]) -> str:
    import shap

    html = shap.force_plot(
        base_value=explanation.base_values,
        shap_values=explanation.values,
        features=explanation.data,
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    ).html()
    return str(html)


def _save_beeswarm(
    shap_vals: Any,
    X_bg: Any,
    feature_names: list[str],
    symbol: str,
    image_storage: ImageStorage,
) -> str:
    import shap

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_bg, feature_names=feature_names, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return str(path)


def _save_bar_plot(explanation: Any, symbol: str, image_storage: ImageStorage) -> str:
    import shap

    shap.plots.bar(explanation, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return str(path)


def _local_explanations(
    explanation: Any,
    feature_names: list[str],
    symbol: str,
    image_storage: ImageStorage,
) -> list[dict[str, object]]:
    import shap

    local = []
    n_local = min(3, len(explanation.values))

    for i in range(n_local):
        shap.plots.waterfall(explanation[i], show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        path = image_storage.save(symbol, buf.getvalue())

        vals_i = explanation.values[i]
        top_idx = np.argsort(np.abs(vals_i))[::-1][:5]
        contributors = [
            {"feature": feature_names[j], "shap_value": round(float(vals_i[j]), 6)} for j in top_idx
        ]

        local.append(
            {
                "sample_idx": int(i),
                "base_value": round(
                    float(
                        explanation.base_values[i]
                        if isinstance(explanation.base_values, np.ndarray)
                        else explanation.base_values
                    ),
                    6,
                ),
                "waterfall_plot": str(path),
                "top_contributors": contributors,
            }
        )

    return local
