from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from ai_candle_predictor.application.ports.image_storage import ImageStorage
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


def shap_analysis(
    symbol: str,
    pipeline: Pipeline,
    X: Any,
    feature_names: list[str],
    image_storage: ImageStorage,
    max_samples: int = 100,
) -> dict[str, object]:
    check_shap()
    import shap

    log.info(
        "shap analysis started",
        symbol=symbol,
        samples=len(X),
        features=len(feature_names),
    )

    X_bg = _select_background(X, max_samples)
    explainer, X_bg_used = _build_explainer(pipeline, X_bg)

    shap_vals = explainer.shap_values(X)
    vals, base = _extract_binary(shap_vals)

    bar_explanation = shap.Explanation(
        values=shap_vals,
        base_values=base,
        data=X,
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
        "samples_analyzed": len(X),
    }


def explain_single_sample(
    symbol: str,
    pipeline: Pipeline,
    X: Any,
    feature_names: list[str],
    sample_index: int,
    image_storage: ImageStorage,
    background_samples: int = 100,
) -> dict[str, object]:
    check_shap()

    X_bg = _select_background(X, background_samples)
    explainer, X_bg_used = _build_explainer(pipeline, X_bg)

    shap_vals = explainer.shap_values(X)

    sample = shap_vals[sample_index : sample_index + 1]
    base_val = float(
        sample.base_values[0] if isinstance(sample.base_values, np.ndarray) else sample.base_values
    )
    prediction = float(sample.values.sum() + base_val)

    waterfall_path = _save_waterfall(sample, feature_names, sample_index, symbol, image_storage)
    force_html = _render_force_html(sample, feature_names)

    vals_i = sample.values[0]
    top_idx = np.argsort(np.abs(vals_i))[::-1][:10]
    top_features = [
        {"feature": feature_names[j], "shap_value": round(float(vals_i[j]), 6)} for j in top_idx
    ]

    return {
        "base_value": base_val,
        "prediction": prediction,
        "waterfall_plot": str(waterfall_path),
        "force_html": force_html,
        "top_features": top_features,
    }


def _select_background(X: Any, max_samples: int = 100) -> Any | None:
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        return rng.choice(len(X), max_samples, replace=False)
    return None


def _save_waterfall(
    shap_values: Any,
    _feature_names: list[str],
    _sample_index: int,
    symbol: str,
    image_storage: ImageStorage,
) -> Path:
    import shap

    shap.plots.waterfall(shap_values[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return path


def _render_force_html(
    shap_values: Any,
    feature_names: list[str],
) -> str:
    import shap

    html = shap.force_plot(
        base_value=shap_values.base_values,
        shap_values=shap_values.values,
        features=shap_values.data,
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    ).html()
    return str(html)


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


def _extract_binary(shap_values: Any) -> tuple[Any, Any]:
    vals = shap_values.values
    base = shap_values.base_values

    if isinstance(vals, list):
        vals = vals[1]
    if isinstance(vals, np.ndarray) and vals.ndim == 3:
        vals = vals[:, :, 1]

    if isinstance(base, list):
        base = base[1]
    if isinstance(base, np.ndarray) and base.ndim == 2:
        base = base[:, 1]

    return np.asarray(vals, dtype=np.float64), base


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


def _save_bar_plot(
    explanation: Any,
    symbol: str,
    image_storage: ImageStorage,
) -> str:
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
