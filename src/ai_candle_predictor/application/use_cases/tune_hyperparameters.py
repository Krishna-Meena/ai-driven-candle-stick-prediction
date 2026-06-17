from __future__ import annotations

from collections.abc import Sequence

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.indicators import IndicatorValue
from ai_candle_predictor.domain.entities.label import LabeledSample
from ai_candle_predictor.domain.entities.metrics import ClassificationMetrics
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)


def tune_hyperparameters(
    symbol: Symbol,
    feature_store: FeatureStore,
    label_store: LabelStore,
    model_store: ModelStore,
    model_type: str = "rf",
    n_trials: int = 50,
    val_split: float = 0.2,
    horizon: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    log.info(
        "hyperparameter tuning started",
        symbol=symbol.value,
        model=model_type,
        trials=n_trials,
    )

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

    log.info("data prepared", rows=len(merged), features=len(feature_df.columns))

    X = merged.drop(columns=["label", "forward_return"]).values
    y = merged["label"].values

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
    )

    def objective(trial: optuna.Trial) -> float:
        if model_type == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = Pipeline([("classifier", RandomForestClassifier(**params))])
            model.fit(X_train, y_train)

        elif model_type == "xgb":
            pos_count = int(y_train.sum())
            neg_count = int(len(y_train) - pos_count)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "early_stopping_rounds": 20,
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            model = Pipeline([("classifier", XGBClassifier(**params))])
            model.fit(
                X_train,
                y_train,
                classifier__eval_set=[(X_val, y_val)],
                classifier__verbose=False,
            )

        else:
            raise ValueError(f"unknown model_type: {model_type}")

        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.0
        return auc

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{symbol.value}_{model_type}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    log.info(
        "tuning complete",
        best_value=round(best_value, 4),
        best_params=best_params,
        n_trials=n_trials,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in completed if t.value is not None]
    if values:
        log.info(
            "trial distribution",
            min=round(min(values), 4),
            max=round(max(values), 4),
            mean=round(sum(values) / len(values), 4),
            std=round(
                (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5,
                4,
            ),
        )

    if model_type == "rf":
        rf_params = {k: v for k, v in best_params.items()}
        rf_params.update(
            {
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            }
        )
        final_model = Pipeline([("classifier", RandomForestClassifier(**rf_params))])
        final_model.fit(X_train, y_train)
        model_label = f"rf_tuned_{n_trials}trials"

    elif model_type == "xgb":
        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        xgb_params = {k: v for k, v in best_params.items()}
        xgb_params.update(
            {
                "scale_pos_weight": neg_count / pos_count if pos_count > 0 else 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
        )
        final_model = Pipeline([("classifier", XGBClassifier(**xgb_params))])
        final_model.fit(X_train, y_train)
        model_label = f"xgb_tuned_{n_trials}trials"

    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_val, y_pred),
        precision=precision_score(y_val, y_pred, zero_division=0),
        recall=recall_score(y_val, y_pred, zero_division=0),
        f1=f1_score(y_val, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.0,
        support=len(y_val),
    )

    path = model_store.save(final_model, symbol.value, label=model_label)

    results: dict[str, object] = {
        "model_type": model_type,
        "best_params": best_params,
        "best_trial_auc": round(best_value, 4),
        "metrics": str(metrics).replace("\n", " "),
        "model_path": str(path),
        "n_trials": n_trials,
        "completed_trials": len(completed),
    }

    log.info(
        "tuned model trained",
        metrics=str(metrics).replace("\n", " "),
        path=str(path),
    )

    return results


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
