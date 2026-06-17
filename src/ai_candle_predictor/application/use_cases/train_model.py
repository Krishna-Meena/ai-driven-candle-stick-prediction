from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.application.use_cases.train_baseline import train_baseline
from ai_candle_predictor.application.use_cases.train_random_forest import (
    train_random_forest,
)
from ai_candle_predictor.application.use_cases.train_xgboost import train_xgboost
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.models.model_registry import (
    ModelRegistry,
    RegistryEntry,
)

ProgressCallback = Callable[[int, str], None]
_TRAINERS = ("lr", "rf", "xgb")


def _noop(progress: int, message: str) -> None:
    pass


def train_model(
    symbol: Symbol,
    model_type: str,
    feature_store: FeatureStore,
    label_store: LabelStore,
    model_store: ModelStore,
    val_split: float = 0.2,
    horizon: int = 5,
    on_progress: ProgressCallback | None = None,
    **hyperparams: Any,
) -> tuple[Path, ClassificationMetrics]:
    progress = on_progress or _noop

    mt = model_type.lower()
    if mt not in _TRAINERS:
        raise ValueError(f"unknown model type '{model_type}'; choose from {_TRAINERS}")

    progress(10, f"Loading features for {symbol.value}...")
    features = feature_store.load(symbol)
    progress(25, f"Loading labels for {symbol.value}...")
    labels = label_store.load(symbol)

    if not features:
        raise ValueError(f"no features found for {symbol.value}")
    if not labels:
        raise ValueError(f"no labels found for {symbol.value}")

    progress(40, "Training model...")

    if mt == "lr":
        c_val = float(hyperparams.get("C", 1.0))
        max_iter = int(hyperparams.get("max_iter", 5000))
        pipeline, metrics, path = train_baseline(
            symbol=symbol,
            feature_store=feature_store,
            label_store=label_store,
            model_store=model_store,
            val_split=val_split,
            horizon=horizon,
            C=c_val,
            max_iter=max_iter,
        )
        model_label = f"baseline_C{c_val}"
    elif mt == "rf":
        n_estimators = int(hyperparams.get("n_estimators", 300))
        max_depth = int(hyperparams.get("max_depth", 10))
        min_samples_leaf = int(hyperparams.get("min_samples_leaf", 5))
        pipeline, metrics, path, _ = train_random_forest(
            symbol=symbol,
            feature_store=feature_store,
            label_store=label_store,
            model_store=model_store,
            val_split=val_split,
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        model_label = f"rf_n{n_estimators}_d{max_depth}"
    else:
        n_estimators = int(hyperparams.get("n_estimators", 300))
        max_depth = int(hyperparams.get("max_depth", 6))
        learning_rate = float(hyperparams.get("learning_rate", 0.05))
        pipeline, metrics, path, _ = train_xgboost(
            symbol=symbol,
            feature_store=feature_store,
            label_store=label_store,
            model_store=model_store,
            val_split=val_split,
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        model_label = f"xgb_n{n_estimators}_d{max_depth}_lr{learning_rate}"

    progress(80, "Evaluating model...")
    progress(90, "Saving model...")

    safe = symbol.value.replace("^", "_").replace(".", "_")
    filename = f"{safe}_{model_label}.joblib"

    registry = ModelRegistry()
    registry.register(
        RegistryEntry(
            symbol=symbol.value,
            model_type=mt.upper(),
            label=model_label,
            filename=filename,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            roc_auc=metrics.roc_auc,
            support=metrics.support,
        )
    )

    progress(100, "Training complete!")
    return path, metrics
