from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)

app = FastAPI(
    title="AI-Driven Candlestick Prediction Platform API",
    description="Production-grade REST API exposing data ingestion, feature engineering, labeling, prediction, and backtesting services.",
    version="0.1.0",
)

# Enable CORS for cross-origin integration (e.g. custom dashboards, trading bots)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schema DTOs ──────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    description: str
    timestamp: datetime


class SymbolStatus(BaseModel):
    symbol: str
    raw_rows: int
    feature_rows: int
    label_rows: int
    models_count: int


class IngestRequest(BaseModel):
    symbol: str = Field(..., example="BTC-USD")
    start_date: str | None = Field(None, example="2024-01-01", description="Format YYYY-MM-DD")
    end_date: str | None = Field(None, example="2024-06-01", description="Format YYYY-MM-DD")


class IngestResponse(BaseModel):
    symbol: str
    rows_fetched: int
    rows_valid: int
    rows_rejected: int
    storage_path: str | None
    errors: list[str]


class RunPipelineRequest(BaseModel):
    symbol: str = Field(..., example="BTC-USD")


class RunPipelineResponse(BaseModel):
    symbol: str
    success: bool
    message: str


class LabelRequest(BaseModel):
    symbol: str = Field(..., example="BTC-USD")
    horizon: int = Field(5, description="Prediction horizon in candles")
    threshold: float = Field(0.005, description="Minimum price movement threshold")


class LabelResponse(BaseModel):
    symbol: str
    total_samples: int
    up_samples: int
    down_samples: int
    excluded_samples: int


class PredictRequest(BaseModel):
    symbol: str = Field(..., example="BTC-USD")
    model_label: str = Field("", example="Random Forest", description="Model label to load")
    start_date: str | None = Field(None, example="2024-01-01")
    end_date: str | None = Field(None, example="2024-06-01")
    horizon: int = Field(5, description="Prediction horizon")


class PredictResponse(BaseModel):
    symbol: str
    model_label: str
    start_date: datetime
    end_date: datetime
    total_candles: int
    accuracy: float | None = None
    predictions: list[dict[str, Any]]


class BacktestRequest(BaseModel):
    symbol: str = Field(..., example="BTC-USD")
    model_label: str = Field("", example="Random Forest")
    start_date: str | None = Field(None, example="2024-01-01")
    end_date: str | None = Field(None, example="2024-06-01")
    initial_capital: float = Field(10000.0, description="Initial investment capital")


class BacktestTradeDTO(BaseModel):
    entry_date: datetime
    exit_date: datetime
    side: str
    entry_price: float
    exit_price: float
    return_pct: float
    won: bool


class BacktestResponse(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    initial_capital: float
    final_equity: float
    strategy_return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    trades: list[BacktestTradeDTO]


# ── REST API Endpoints ──────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check API server health and system information."""
    return HealthResponse(
        status="OK",
        version="0.1.0",
        description=app.description,
        timestamp=datetime.now(),
    )


@app.get("/symbols", response_model=list[SymbolStatus])
def list_symbols() -> list[SymbolStatus]:
    """Get status breakdown and count of data items for all tracked symbols."""
    from ai_candle_predictor.common.symbol_utils import normalize_symbol
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore

    result: list[SymbolStatus] = []

    fs = ParquetFeatureStore()
    ls = ParquetLabelStore()

    for sym in settings.default_symbols:
        safe = normalize_symbol(sym)
        raw_path = settings.data_raw_dir / f"{safe}.parquet"

        # Calculate rows
        raw_rows = 0
        if raw_path.exists():
            import pandas as pd

            try:
                raw_rows = len(pd.read_parquet(raw_path))
            except Exception:
                pass

        feat_rows = 0
        try:
            feats = fs.load(Symbol(sym))
            feat_rows = len(feats) if feats else 0
        except Exception:
            pass

        lbl_rows = 0
        try:
            labels = ls.load(Symbol(sym))
            lbl_rows = len(labels) if labels else 0
        except Exception:
            pass

        models_count = 0
        if settings.models_dir.exists():
            models_count = len(list(settings.models_dir.glob(f"{safe}_*.joblib")))

        result.append(
            SymbolStatus(
                symbol=sym,
                raw_rows=raw_rows,
                feature_rows=feat_rows,
                label_rows=lbl_rows,
                models_count=models_count,
            )
        )
    return result


@app.post("/pipeline/ingest", response_model=IngestResponse)
def trigger_ingest(payload: IngestRequest) -> IngestResponse:
    """Trigger market data ingestion from Yahoo Finance."""
    from ai_candle_predictor.application.dto.market_data import MarketDataRequest
    from ai_candle_predictor.application.use_cases.ingest_market_data import ingest_market_data
    from ai_candle_predictor.infrastructure.data.yahoo_provider import YahooProvider
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = (
        datetime.strptime(payload.start_date, "%Y-%m-%d")
        if payload.start_date
        else datetime.strptime(settings.default_start_date, "%Y-%m-%d")
    )
    end = datetime.strptime(payload.end_date, "%Y-%m-%d") if payload.end_date else datetime.now()

    try:
        req = MarketDataRequest(symbol=payload.symbol, start_date=start, end_date=end)
        provider = YahooProvider()
        storage = ParquetStore()
        res = ingest_market_data(req, provider, storage)

        return IngestResponse(
            symbol=res.symbol,
            rows_fetched=res.rows_fetched,
            rows_valid=res.rows_valid,
            rows_rejected=res.rows_rejected,
            storage_path=res.storage_path,
            errors=res.errors,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest data: {e}",
        ) from e


@app.post("/pipeline/features", response_model=RunPipelineResponse)
def trigger_features(payload: RunPipelineRequest) -> RunPipelineResponse:
    """Trigger technical indicators (features) computation."""
    from ai_candle_predictor.application.use_cases.compute_features import (
        compute_features as comp_feat,
    )
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    try:
        storage = ParquetStore()
        feature_store = ParquetFeatureStore()
        count = comp_feat(Symbol(payload.symbol), storage, feature_store)
        return RunPipelineResponse(
            symbol=payload.symbol,
            success=True,
            message=f"Computed and saved {count} feature rows successfully.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute features: {e}",
        ) from e


@app.post("/pipeline/labels", response_model=LabelResponse)
def trigger_labels(payload: LabelRequest) -> LabelResponse:
    """Trigger label generation using forward returns."""
    from ai_candle_predictor.application.use_cases.generate_labels import generate_labels_for_symbol
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    try:
        storage = ParquetStore()
        label_store = ParquetLabelStore()
        stats = generate_labels_for_symbol(
            Symbol(payload.symbol),
            storage,
            label_store,
            horizon=payload.horizon,
            threshold=payload.threshold,
        )
        return LabelResponse(
            symbol=payload.symbol,
            total_samples=stats["total"],
            up_samples=stats["up"],
            down_samples=stats["down"],
            excluded_samples=stats["excluded"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate labels: {e}",
        ) from e


@app.get("/models")
def list_models() -> list[dict[str, Any]]:
    """List registered models and their performance metrics from the registry."""
    from ai_candle_predictor.infrastructure.models.model_registry import ModelRegistry

    try:
        registry = ModelRegistry()
        models = registry.list_models()
        return [
            {
                "symbol": m.symbol,
                "model_type": m.model_type,
                "label": m.label,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
                "support": m.support,
                "filename": m.filename,
                "training_date": m.training_date,
            }
            for m in models
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query model registry: {e}",
        ) from e


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Generate candlestick predictions and direction confidence scoring."""
    from ai_candle_predictor.application.use_cases.predict import predict_range
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = (
        datetime.strptime(payload.start_date, "%Y-%m-%d")
        if payload.start_date
        else datetime.now() - timedelta(days=30)
    )
    end = datetime.strptime(payload.end_date, "%Y-%m-%d") if payload.end_date else datetime.now()

    try:
        model_store = JoblibStore()
        feature_store = ParquetFeatureStore()
        label_store = ParquetLabelStore()
        candle_store = ParquetStore()

        result = predict_range(
            symbol=Symbol(payload.symbol),
            model_store=model_store,
            feature_store=feature_store,
            label_store=label_store,
            candle_store=candle_store,
            start_date=start,
            end_date=end,
            model_label=payload.model_label,
            horizon=payload.horizon,
        )

        preds_dto = []
        correct = 0
        labeled = 0
        for p in result.predictions:
            preds_dto.append(
                {
                    "timestamp": p.timestamp,
                    "close": p.close,
                    "predicted_direction": p.predicted_direction,
                    "confidence": p.confidence,
                    "actual_return": p.actual_return,
                    "actual_direction": p.actual_direction,
                    "is_correct": p.is_correct,
                }
            )
            if p.is_correct is not None:
                labeled += 1
                if p.is_correct:
                    correct += 1

        accuracy = correct / labeled if labeled > 0 else None

        return PredictResponse(
            symbol=result.symbol,
            model_label=result.model_label,
            start_date=result.start_date,
            end_date=result.end_date,
            total_candles=result.total_candles,
            accuracy=accuracy,
            predictions=preds_dto,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run predictions: {e}",
        ) from e


@app.post("/backtest", response_model=BacktestResponse)
def backtest(payload: BacktestRequest) -> BacktestResponse:
    """Run interactive backtesting strategy on range predictions."""
    from ai_candle_predictor.application.use_cases.backtest import run_backtest
    from ai_candle_predictor.application.use_cases.predict import predict_range
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = (
        datetime.strptime(payload.start_date, "%Y-%m-%d")
        if payload.start_date
        else datetime.now() - timedelta(days=90)
    )
    end = datetime.strptime(payload.end_date, "%Y-%m-%d") if payload.end_date else datetime.now()

    try:
        model_store = JoblibStore()
        feature_store = ParquetFeatureStore()
        label_store = ParquetLabelStore()
        candle_store = ParquetStore()

        pred_result = predict_range(
            symbol=Symbol(payload.symbol),
            model_store=model_store,
            feature_store=feature_store,
            label_store=label_store,
            candle_store=candle_store,
            start_date=start,
            end_date=end,
            model_label=payload.model_label,
        )

        if not pred_result.predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No prediction data generated for backtesting range. Make sure data is ingested.",
            )

        res = run_backtest(pred_result.predictions, initial_capital=payload.initial_capital)

        trades_dto = [
            BacktestTradeDTO(
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                return_pct=t.return_pct,
                won=t.won,
            )
            for t in res.trades
        ]

        return BacktestResponse(
            symbol=payload.symbol,
            start_date=res.start_date,
            end_date=res.end_date,
            total_trades=res.total_trades,
            winning_trades=res.winning_trades,
            losing_trades=res.losing_trades,
            win_rate=res.win_rate,
            initial_capital=res.initial_capital,
            final_equity=res.final_equity,
            strategy_return_pct=res.strategy_return_pct,
            buy_hold_return_pct=res.buy_hold_return_pct,
            sharpe_ratio=res.sharpe_ratio,
            max_drawdown_pct=res.max_drawdown_pct,
            trades=trades_dto,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run backtest: {e}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
