from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_candle_predictor.presentation.cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI-Driven Candlestick" in result.stdout


@patch("ai_candle_predictor.application.use_cases.ingest_market_data.ingest_market_data")
def test_cli_ingest(mock_ingest: MagicMock) -> None:
    from ai_candle_predictor.application.dto.market_data import IngestionResult

    mock_ingest.return_value = IngestionResult(
        symbol="BTC-USD",
        rows_fetched=100,
        rows_valid=95,
        rows_rejected=5,
        storage_path="mock_path.parquet",
        errors=["Price < 0"],
    )

    result = runner.invoke(app, ["ingest", "-s", "BTC-USD", "--start-date", "2024-01-01"])
    assert result.exit_code == 0
    assert "Ingesting historical data" in result.stdout
    assert "Rows fetched: 100" in result.stdout
    assert "Rows valid:   95" in result.stdout
    assert "Storage path: mock_path.parquet" in result.stdout


@patch("ai_candle_predictor.application.use_cases.compute_features.compute_features")
def test_cli_compute_features(mock_compute: MagicMock) -> None:
    mock_compute.return_value = 100

    result = runner.invoke(app, ["compute-features", "-s", "BTC-USD"])
    assert result.exit_code == 0
    assert "Computing technical indicators" in result.stdout
    assert "Computed and saved 100 feature rows" in result.stdout


@patch("ai_candle_predictor.application.use_cases.generate_labels.generate_labels_for_symbol")
def test_cli_generate_labels(mock_labels: MagicMock) -> None:
    mock_labels.return_value = {"total": 100, "up": 45, "down": 45, "excluded": 10}

    result = runner.invoke(app, ["generate-labels", "-s", "BTC-USD"])
    assert result.exit_code == 0
    assert "Generating labels for BTC-USD" in result.stdout
    assert "Total samples processed: 100" in result.stdout
    assert "UP labels generated:     45" in result.stdout


@patch("ai_candle_predictor.application.use_cases.predict.predict_range")
def test_cli_predict(mock_predict: MagicMock) -> None:
    from datetime import datetime

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

    result = runner.invoke(app, ["predict", "-s", "BTC-USD", "-m", "Random Forest"])
    assert result.exit_code == 0
    assert "Predicting candlestick directions" in result.stdout
    assert "Total candles predicted: 1" in result.stdout
    assert "Classification accuracy:       100.00%" in result.stdout


@patch("ai_candle_predictor.application.use_cases.predict.predict_range")
@patch("ai_candle_predictor.application.use_cases.backtest.run_backtest")
def test_cli_backtest(mock_backtest: MagicMock, mock_predict: MagicMock) -> None:
    from datetime import datetime

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

    result = runner.invoke(app, ["backtest", "-s", "BTC-USD", "-m", "Random Forest"])
    assert result.exit_code == 0
    assert "Simulating trading strategy" in result.stdout
    assert "Strategy Win Rate:  100.00%" in result.stdout
    assert "Strategy Return:    +10.00%" in result.stdout
