from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from ai_candle_predictor.application.ports.image_storage import ImageStorage
from ai_candle_predictor.common.logging import get_logger

matplotlib.use("Agg")

log = get_logger(__name__)


def shap_analysis(
    symbol: str,
    pipeline: Pipeline,
    X: Any,
    feature_names: list[str],
    image_storage: ImageStorage,
    max_samples: int = 100,
) -> dict[str, object]:
    log.info(
        "shap analysis started",
        symbol=symbol,
        samples=len(X),
        features=len(feature_names),
    )

    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    log.info("computing shap values", background_samples=len(X_bg))
    explainer, X_explain = _build_explainer(pipeline, X_bg)
    shap_values = explainer(X_explain)

    shap_vals, base_val = _extract_binary(shap_values)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)

    global_ranking = {
        name: round(float(abs_val), 6)
        for name, abs_val in sorted(
            zip(feature_names, mean_abs, strict=True), key=lambda x: x[1], reverse=True
        )
    }
    global_direction = {
        name: round(float(val), 6) for name, val in zip(feature_names, mean_shap, strict=True)
    }

    bar_explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=X_bg,
        feature_names=feature_names,
    )

    beeswarm_path = _save_beeswarm(shap_vals, X_bg, feature_names, symbol, image_storage)
    bar_path = _save_bar_plot(bar_explanation, symbol, image_storage)
    local_explanations = _local_explanations(bar_explanation, feature_names, symbol, image_storage)

    top5 = list(global_ranking.keys())[:5]
    log.info(
        "shap analysis complete",
        top_features=top5,
        beeswarm=str(beeswarm_path),
        bar=str(bar_path),
    )

    return {
        "symbol": symbol,
        "model_type": pipeline.named_steps["classifier"].__class__.__name__,
        "samples_analyzed": len(X_bg),
        "global_ranking": global_ranking,
        "global_direction": global_direction,
        "base_value": (
            round(float(base_val), 6)
            if isinstance(base_val, (int, float, np.floating))
            else round(float(np.asarray(base_val).mean()), 6)
        ),
        "summary_plot": str(beeswarm_path),
        "bar_plot": str(bar_path),
        "local_explanations": local_explanations,
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
    log.info("explaining single sample", symbol=symbol, sample_index=sample_index)

    if sample_index < 0 or sample_index >= len(X):
        raise ValueError(f"sample_index {sample_index} out of range (0\u2013{len(X) - 1})")

    bg_idx = _select_background(X, background_samples)
    X_bg = X[bg_idx] if bg_idx is not None else X
    explainer, X_bg_used = _build_explainer(pipeline, X_bg)

    X_sample = X[sample_index : sample_index + 1]
    shap_values = explainer(X_sample)

    shap_vals, base_val = _extract_binary(shap_values)
    vals_i = shap_vals[0]
    base_i = base_val[0] if isinstance(base_val, np.ndarray) else base_val

    waterfall_path = _save_waterfall(
        shap_values, feature_names, sample_index, symbol, image_storage
    )

    force_html = _render_force_html(shap_values, feature_names)

    top_idx = np.argsort(np.abs(vals_i))[::-1][:10]
    top_features = [
        {"feature": feature_names[j], "shap_value": round(float(vals_i[j]), 6)} for j in top_idx
    ]

    log.info("single sample explanation complete", sample_index=sample_index)
    return {
        "sample_index": sample_index,
        "base_value": round(float(base_i), 6),
        "prediction": round(float(base_i + float(np.sum(vals_i))), 6),
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
    shap_values: shap.Explanation,
    _feature_names: list[str],
    _sample_index: int,
    symbol: str,
    image_storage: ImageStorage,
) -> Path:
    shap.plots.waterfall(shap_values[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return path


def _render_force_html(
    shap_values: shap.Explanation,
    feature_names: list[str],
) -> str:
    html = shap.force_plot(
        base_value=shap_values.base_values,
        shap_values=shap_values.values,
        features=shap_values.data,
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    ).html()
    return str(html)


def _build_explainer(pipeline: Pipeline, X_bg: Any) -> tuple[shap.Explainer, Any]:
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


def _extract_binary(
    shap_values: shap.Explanation,
) -> tuple[Any, Any]:
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
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_bg, feature_names=feature_names, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return str(path)


def _save_bar_plot(
    explanation: shap.Explanation,
    symbol: str,
    image_storage: ImageStorage,
) -> str:
    shap.plots.bar(explanation, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    path = image_storage.save(symbol, buf.getvalue())
    return str(path)


def _local_explanations(
    explanation: shap.Explanation,
    feature_names: list[str],
    symbol: str,
    image_storage: ImageStorage,
) -> list[dict[str, object]]:
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
                "prediction": round(
                    (
                        float(np.sum(vals_i) + explanation.base_values[i])
                        if isinstance(explanation.base_values, np.ndarray)
                        else float(np.sum(vals_i) + explanation.base_values)
                    ),
                    6,
                ),
                "top_contributors": contributors,
                "waterfall_plot": str(path),
            }
        )

    return local
