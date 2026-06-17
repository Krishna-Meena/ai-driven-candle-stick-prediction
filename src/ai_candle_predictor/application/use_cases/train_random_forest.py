from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.indicators import IndicatorValue
from ai_candle_predictor.domain.entities.label import LabeledSample
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)


def train_random_forest(
    symbol: Symbol,
    feature_store: FeatureStore,
    label_store: LabelStore,
    model_store: ModelStore,
    val_split: float = 0.2,
    horizon: int = 5,
    n_estimators: int = 300,
    max_depth: int = 10,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    class_weight: str = "balanced",
    random_state: int = 42,
) -> tuple[Pipeline, ClassificationMetrics, Path, dict[str, float]]:
    log.info("starting random forest training", symbol=symbol.value)

    features = feature_store.load(symbol)
    if not features:
        raise ValueError(f"no features found for {symbol.value}")

    labels = label_store.load(symbol)
    if not labels:
        raise ValueError(f"no labels found for {symbol.value}")

    feature_df = _pivot_features(features)
    label_df = _pivot_labels(labels)

    merged = feature_df.merge(label_df, left_index=True, right_index=True, how="inner")
    if merged.empty:
        raise ValueError("no aligned feature-label pairs after merge")

    log.info(
        "training matrix assembled",
        rows=len(merged),
        features=len(feature_df.columns),
    )

    X = merged.drop(columns=["label", "forward_return"]).values
    y = merged["label"].values
    feature_names = list(feature_df.columns)

    split_idx = int(len(merged) * (1 - val_split))
    purge_idx = max(0, split_idx - horizon)

    X_train, X_val = X[:purge_idx], X[split_idx:]
    y_train, y_val = y[:purge_idx], y[split_idx:]

    if len(set(y_train)) < 2:
        raise ValueError("training set has only one class after split")

    log.info(
        "train/val split",
        train=len(X_train),
        val=len(X_val),
        purged=split_idx - purge_idx,
        train_pos=float(y_train.mean()),
        val_pos=float(y_val.mean()),
    )

    pipeline = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight=class_weight,
                    oob_score=True,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_val, y_pred),
        precision=precision_score(y_val, y_pred, zero_division=0),
        recall=recall_score(y_val, y_pred, zero_division=0),
        f1=f1_score(y_val, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.0,
        support=len(y_val),
    )

    rf_model: RandomForestClassifier = pipeline.named_steps["classifier"]
    importances = dict(
        sorted(
            zip(feature_names, rf_model.feature_importances_, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    log.info(
        "random forest training complete",
        oob_score=rf_model.oob_score_,
        metrics=str(metrics).replace("\n", " "),
    )
    log.info("top 5 features", features=list(importances.keys())[:5])

    path = model_store.save(
        pipeline,
        symbol.value,
        label=f"rf_n{n_estimators}_d{max_depth}",
    )
    log.info("model persisted", path=str(path))

    return pipeline, metrics, path, importances


def _pivot_features(features: Sequence[IndicatorValue]) -> pd.DataFrame:
    records = [
        {
            "timestamp": f.timestamp,
            "indicator": f.indicator.name,
            "value": f.value,
        }
        for f in features
    ]
    df = pd.DataFrame(records)
    df = _to_naive_timestamps(df)
    pivoted = df.pivot_table(
        index="timestamp",
        columns="indicator",
        values="value",
        aggfunc="first",
    )
    pivoted = pivoted.ffill().bfill().dropna(how="all")
    return pivoted


def _pivot_labels(labels: Sequence[LabeledSample]) -> pd.DataFrame:
    records = [
        {
            "timestamp": lbl.timestamp,
            "label": lbl.label.as_int,
            "forward_return": lbl.forward_return,
        }
        for lbl in labels
    ]
    df = pd.DataFrame(records)
    df = _to_naive_timestamps(df)
    df = df.set_index("timestamp")
    return df


def _to_naive_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    col = "timestamp"
    if col not in df:
        return df
    ts = pd.to_datetime(df[col])
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df[col] = ts
    return df
