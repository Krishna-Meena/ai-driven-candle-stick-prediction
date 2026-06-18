from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ai_candle_predictor.presentation.api.main import app

client = TestClient(app)


def test_api_health() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "version" in data
    assert "timestamp" in data


@patch("ai_candle_predictor.presentation.api.main.settings")
def test_api_symbols(mock_settings: MagicMock) -> None:
    # Set default symbols to a controlled list
    mock_settings.default_symbols = ["BTC-USD"]
    mock_settings.data_raw_dir = MagicMock()
    mock_settings.models_dir = MagicMock()

    response = client.get("/symbols")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTC-USD"
    assert "raw_rows" in data[0]
    assert "feature_rows" in data[0]


@patch("ai_candle_predictor.application.use_cases.ingest_market_data.ingest_market_data")
def test_api_trigger_ingest(mock_ingest: MagicMock) -> None:
    from ai_candle_predictor.application.dto.market_data import IngestionResult

    mock_ingest.return_value = IngestionResult(
        symbol="BTC-USD",
        rows_fetched=100,
        rows_valid=90,
        rows_rejected=10,
        storage_path="mock.parquet",
        errors=[],
    )

    payload = {"symbol": "BTC-USD", "start_date": "2024-01-01", "end_date": "2024-06-01"}
    response = client.post("/pipeline/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert data["rows_fetched"] == 100
    assert data["rows_valid"] == 90


@patch("ai_candle_predictor.application.use_cases.compute_features.compute_features")
def test_api_trigger_features(mock_compute: MagicMock) -> None:
    mock_compute.return_value = 150

    payload = {"symbol": "BTC-USD"}
    response = client.post("/pipeline/features", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert data["success"] is True
    assert "150" in data["message"]


@patch("ai_candle_predictor.application.use_cases.generate_labels.generate_labels_for_symbol")
def test_api_trigger_labels(mock_labels: MagicMock) -> None:
    mock_labels.return_value = {"total": 100, "up": 45, "down": 45, "excluded": 10}

    payload = {"symbol": "BTC-USD", "horizon": 5, "threshold": 0.005}
    response = client.post("/pipeline/labels", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert data["total_samples"] == 100
    assert data["up_samples"] == 45
    assert data["down_samples"] == 45


@patch("ai_candle_predictor.infrastructure.models.model_registry.ModelRegistry.list_models")
def test_api_list_models(mock_list: MagicMock) -> None:
    from ai_candle_predictor.infrastructure.models.model_registry import RegistryEntry

    mock_list.return_value = [
        RegistryEntry(
            symbol="BTC-USD",
            model_type="Random Forest",
            label="RF",
            accuracy=0.85,
            precision=0.84,
            recall=0.86,
            f1=0.85,
            roc_auc=0.91,
            support=100,
            filename="BTC-USD_RF.joblib",
            training_date="2026-06-17T12:00:00",
        )
    ]

    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTC-USD"
    assert data[0]["accuracy"] == 0.85
    assert data[0]["roc_auc"] == 0.91


@patch("ai_candle_predictor.application.use_cases.predict.predict_range")
def test_api_predict(mock_predict: MagicMock) -> None:
    from ai_candle_predictor.application.dto.prediction import CandlePrediction, PredictionResult

    predictions = [
        CandlePrediction(
            timestamp=datetime(2024, 1, 1),
            close=50000.0,
            predicted_direction=1,
            confidence=0.8,
            actual_return=0.01,
            actual_direction=1,
            is_correct=True,
        )
    ]

    mock_predict.return_value = PredictionResult(
        symbol="BTC-USD",
        model_label="Random Forest",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        horizon=5,
        total_candles=1,
        predictions=predictions,
    )

    payload = {
        "symbol": "BTC-USD",
        "model_label": "Random Forest",
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert data["accuracy"] == 1.0
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["close"] == 50000.0


@patch("ai_candle_predictor.application.use_cases.predict.predict_range")
@patch("ai_candle_predictor.application.use_cases.backtest.run_backtest")
def test_api_backtest(mock_backtest: MagicMock, mock_predict: MagicMock) -> None:
    from ai_candle_predictor.application.dto.backtest import BacktestResult
    from ai_candle_predictor.application.dto.prediction import CandlePrediction, PredictionResult

    predictions = [
        CandlePrediction(
            timestamp=datetime(2024, 1, 1),
            close=50000.0,
            predicted_direction=1,
            confidence=0.8,
            actual_return=0.01,
            actual_direction=1,
            is_correct=True,
        )
    ]

    mock_predict.return_value = PredictionResult(
        symbol="BTC-USD",
        model_label="Random Forest",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        horizon=5,
        total_candles=1,
        predictions=predictions,
    )

    mock_backtest.return_value = BacktestResult(
        symbol="BTC-USD",
        model_label="Random Forest",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=1.0,
        total_return_pct=10.0,
        strategy_return_pct=10.0,
        buy_hold_return_pct=5.0,
        sharpe_ratio=2.5,
        max_drawdown_pct=1.0,
        trades=[],
        initial_capital=10000.0,
        final_equity=11000.0,
    )

    payload = {
        "symbol": "BTC-USD",
        "model_label": "Random Forest",
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "initial_capital": 10000.0,
    }
    response = client.post("/backtest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert data["win_rate"] == 1.0
    assert data["strategy_return_pct"] == 10.0
    assert data["max_drawdown_pct"] == 1.0
