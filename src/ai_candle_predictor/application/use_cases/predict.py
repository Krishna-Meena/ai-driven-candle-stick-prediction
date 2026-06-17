from __future__ import annotations

from datetime import datetime
from typing import Any

from ai_candle_predictor.application.dto.prediction import CandlePrediction, PredictionResult
from ai_candle_predictor.application.ports.feature_store import FeatureStore
from ai_candle_predictor.application.ports.label_store import LabelStore
from ai_candle_predictor.application.ports.model_store import ModelStore
from ai_candle_predictor.application.ports.storage_adapter import StorageAdapter
from ai_candle_predictor.application.use_cases.train_baseline import _pivot_features
from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)


def predict_range(
    symbol: Symbol,
    model_store: ModelStore,
    feature_store: FeatureStore,
    label_store: LabelStore,
    candle_store: StorageAdapter,
    start_date: datetime,
    end_date: datetime,
    model_label: str = "",
    horizon: int = 5,
) -> PredictionResult:
    log.info(
        "predicting range",
        symbol=symbol.value,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        model=model_label,
    )

    model = _load_model(model_store, symbol, model_label)

    features = feature_store.load(symbol, start_date=start_date, end_date=end_date)
    if not features:
        log.warning("no features found for date range", symbol=symbol.value)
        return PredictionResult(
            symbol=symbol.value,
            model_label=model_label,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            total_candles=0,
        )

    feature_df = _pivot_features(features)
    feature_df = feature_df.sort_index()
    x = feature_df.values

    y_prob = model.predict_proba(x)[:, 1]
    y_pred = model.predict(x)

    candles = candle_store.load(symbol, start_date=start_date, end_date=end_date)
    close_map: dict[datetime, float] = {}
    for c in candles:
        ts = _naive_ts(c.timestamp)
        close_map[ts] = c.close

    labels = label_store.load(symbol, start_date=start_date, end_date=end_date)
    actual_map: dict[datetime, tuple[float, int | None]] = {}
    for lbl in labels:
        ts = _naive_ts(lbl.timestamp)
        actual_map[ts] = (lbl.forward_return, lbl.label.as_int)

    predictions: list[CandlePrediction] = []
    for i, ts in enumerate(feature_df.index):
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        close_val = close_map.get(ts_dt)
        if close_val is None:
            continue

        pred_dir = int(y_pred[i])
        conf = float(y_prob[i])

        actual_ret: float | None = None
        actual_dir: int | None = None
        correct: bool | None = None

        if ts_dt in actual_map:
            actual_ret, actual_dir = actual_map[ts_dt]
            correct = bool(pred_dir == actual_dir)

        predictions.append(
            CandlePrediction(
                timestamp=ts_dt,
                close=close_val,
                predicted_direction=pred_dir,
                confidence=conf,
                actual_return=actual_ret,
                actual_direction=actual_dir,
                is_correct=correct,
            )
        )

    result = PredictionResult(
        symbol=symbol.value,
        model_label=model_label,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        total_candles=len(predictions),
        predictions=predictions,
    )

    correct_count = sum(1 for p in predictions if p.is_correct is True)
    total_labeled = sum(1 for p in predictions if p.is_correct is not None)
    accuracy = correct_count / total_labeled if total_labeled > 0 else 0.0
    log.info(
        "prediction complete",
        total=len(predictions),
        labeled=total_labeled,
        correct=correct_count,
        accuracy=f"{accuracy:.4f}",
    )

    return result


def _load_model(
    model_store: ModelStore,
    symbol: Symbol,
    model_label: str,
) -> Any:
    safe = symbol.value.replace("^", "_").replace(".", "_")
    parts = [safe]
    if model_label:
        parts.append(model_label)
    filename = "_".join(parts) + ".joblib"
    path = settings.models_dir / filename
    return model_store.load(path)


def _naive_ts(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt
