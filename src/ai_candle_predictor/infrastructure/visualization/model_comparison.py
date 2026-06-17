from __future__ import annotations

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.application.use_cases.train_baseline import train_baseline
from ai_candle_predictor.application.use_cases.train_random_forest import train_random_forest
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)


def compare_models(
    symbol: Symbol,
    feature_store: FeatureStore,
    label_store: LabelStore,
    model_store: ModelStore,
    val_split: float = 0.2,
    horizon: int = 5,
) -> dict[str, object]:
    log.info("model comparison started", symbol=symbol.value)

    lr_pipeline, lr_metrics, lr_path = train_baseline(
        symbol=symbol,
        feature_store=feature_store,
        label_store=label_store,
        model_store=model_store,
        val_split=val_split,
        horizon=horizon,
    )

    rf_pipeline, rf_metrics, rf_path, rf_importances = train_random_forest(
        symbol=symbol,
        feature_store=feature_store,
        label_store=label_store,
        model_store=model_store,
        val_split=val_split,
        horizon=horizon,
    )

    results = _build_comparison(lr_metrics, rf_metrics, rf_importances)
    _log_table(results)

    log.info("model comparison complete", symbol=symbol.value)
    return results


def _build_comparison(
    lr: ClassificationMetrics,
    rf: ClassificationMetrics,
    rf_importances: dict[str, float],
) -> dict[str, object]:
    rows: dict[str, object] = {
        "accuracy": {"lr": round(lr.accuracy, 4), "rf": round(rf.accuracy, 4)},
        "precision": {"lr": round(lr.precision, 4), "rf": round(rf.precision, 4)},
        "recall": {"lr": round(lr.recall, 4), "rf": round(rf.recall, 4)},
        "f1": {"lr": round(lr.f1, 4), "rf": round(rf.f1, 4)},
        "roc_auc": {"lr": round(lr.roc_auc, 4), "rf": round(rf.roc_auc, 4)},
        "support": {"lr": lr.support, "rf": rf.support},
    }

    for _metric, values in rows.items():
        if isinstance(values, dict) and "lr" in values and "rf" in values:
            lv = values["lr"]
            rv = values["rf"]
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                if lv > rv:
                    values["winner"] = "lr"
                elif rv > lv:
                    values["winner"] = "rf"
                else:
                    values["winner"] = "tie"

    top_features = list(rf_importances.keys())[:5]
    rows["top_features_rf"] = top_features

    return rows


def _log_table(results: dict[str, object]) -> None:
    header = f"{'Metric':<20} {'LR':<12} {'RF':<12} {'Winner':<8}"
    sep = "-" * len(header)
    lines = [header, sep]

    for key, value in results.items():
        if isinstance(value, dict):
            lr_val = value.get("lr", "")
            rf_val = value.get("rf", "")
            winner = value.get("winner", "")
            lines.append(f"{key:<20} {str(lr_val):<12} {str(rf_val):<12} {winner:<8}")
        else:
            lines.append(f"{key:<20} {value}")

    log.info("\n" + "\n".join(lines))
