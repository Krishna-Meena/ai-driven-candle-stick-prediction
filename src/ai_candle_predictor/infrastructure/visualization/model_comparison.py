from __future__ import annotations

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.application.use_cases.train_baseline import train_baseline
from ai_candle_predictor.application.use_cases.train_random_forest import train_random_forest
from ai_candle_predictor.application.use_cases.train_xgboost import train_xgboost
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

MODEL_KEYS = ("lr", "rf", "xgb")


def compare_models(
    symbol: Symbol,
    feature_store: FeatureStore,
    label_store: LabelStore,
    model_store: ModelStore,
    val_split: float = 0.2,
    horizon: int = 5,
) -> dict[str, object]:
    log.info("model comparison started", symbol=symbol.value)

    _, lr_metrics, _ = train_baseline(
        symbol=symbol,
        feature_store=feature_store,
        label_store=label_store,
        model_store=model_store,
        val_split=val_split,
        horizon=horizon,
    )

    _, rf_metrics, _, rf_importances = train_random_forest(
        symbol=symbol,
        feature_store=feature_store,
        label_store=label_store,
        model_store=model_store,
        val_split=val_split,
        horizon=horizon,
    )

    _, xgb_metrics, _, xgb_importances = train_xgboost(
        symbol=symbol,
        feature_store=feature_store,
        label_store=label_store,
        model_store=model_store,
        val_split=val_split,
        horizon=horizon,
    )

    results = _build_comparison(
        lr_metrics, rf_metrics, xgb_metrics, rf_importances, xgb_importances
    )
    _log_table(results)

    log.info("model comparison complete", symbol=symbol.value)
    return results


def _build_comparison(
    lr: ClassificationMetrics,
    rf: ClassificationMetrics,
    xgb: ClassificationMetrics,
    rf_importances: dict[str, float],
    xgb_importances: dict[str, float],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        rows[metric] = {
            "lr": round(getattr(lr, metric), 4),
            "rf": round(getattr(rf, metric), 4),
            "xgb": round(getattr(xgb, metric), 4),
        }
    rows["support"] = {"lr": lr.support, "rf": rf.support, "xgb": xgb.support}

    for _metric, values in rows.items():
        if not isinstance(values, dict):
            continue
        if _metric == "support":
            continue
        candidates = {
            k: v for k, v in values.items() if k in MODEL_KEYS and isinstance(v, (int, float))
        }
        if not candidates:
            continue
        best_key = max(candidates, key=candidates.__getitem__)
        values["winner"] = best_key

    rows["top_features_rf"] = list(rf_importances.keys())[:5]
    rows["top_features_xgb"] = list(xgb_importances.keys())[:5]

    return rows


def _log_table(results: dict[str, object]) -> None:
    header = f"{'Metric':<20} {'LR':<12} {'RF':<12} {'XGB':<12} {'Winner':<8}"
    sep = "-" * len(header)
    lines = [header, sep]

    for key, value in results.items():
        if isinstance(value, dict):
            lr_val = value.get("lr", "")
            rf_val = value.get("rf", "")
            xgb_val = value.get("xgb", "")
            winner = value.get("winner", "")
            lines.append(
                f"{key:<20} {str(lr_val):<12} {str(rf_val):<12} {str(xgb_val):<12} {winner:<8}"
            )
        else:
            lines.append(f"{key:<20} {value}")

    log.info("\n" + "\n".join(lines))
