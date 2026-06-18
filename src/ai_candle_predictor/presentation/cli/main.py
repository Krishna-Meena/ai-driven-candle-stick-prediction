from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import typer

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.value_objects.symbol import Symbol

log = get_logger(__name__)
app = typer.Typer(
    name="ai-candle-predictor",
    help="AI-Driven Candlestick Prediction Platform Command Line Interface",
    no_args_is_help=True,
)


@app.command()
def ingest(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Asset ticker, e.g., BTC-USD"),
    start_date: str = typer.Option(
        settings.default_start_date,
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to today.",
    ),
) -> None:
    """Ingest historical OHLCV data from market data provider."""
    from ai_candle_predictor.application.dto.market_data import MarketDataRequest
    from ai_candle_predictor.application.use_cases.ingest_market_data import ingest_market_data
    from ai_candle_predictor.infrastructure.data.yahoo_provider import YahooProvider
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    typer.echo(f"Ingesting historical data for {symbol} from {start.date()} to {end.date()}...")

    try:
        request = MarketDataRequest(symbol=symbol, start_date=start, end_date=end)
        provider = YahooProvider()
        storage = ParquetStore()
        result = ingest_market_data(request, provider, storage)

        typer.echo(typer.style("Ingestion complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  Rows fetched: {result.rows_fetched}")
        typer.echo(f"  Rows valid:   {result.rows_valid}")
        typer.echo(f"  Rows rejected:{result.rows_rejected}")
        if result.storage_path:
            typer.echo(f"  Storage path: {result.storage_path}")
        if result.errors:
            typer.echo(
                typer.style(f"  Errors encountered: {len(result.errors)}", fg=typer.colors.RED)
            )
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def compute_features(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Asset ticker, e.g., BTC-USD"),
) -> None:
    """Compute technical indicators (features) for ingested market data."""
    from ai_candle_predictor.application.use_cases.compute_features import (
        compute_features as comp_feat,
    )
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    typer.echo(f"Computing technical indicators for {symbol}...")

    try:
        storage = ParquetStore()
        feature_store = ParquetFeatureStore()
        saved_count = comp_feat(Symbol(symbol), storage, feature_store)

        typer.echo(typer.style("Feature computation complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  Computed and saved {saved_count} feature rows.")
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def generate_labels(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Asset ticker, e.g., BTC-USD"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="Prediction horizon in candles"),
    threshold: float = typer.Option(
        0.005, "--threshold", "-t", help="Minimum price movement threshold"
    ),
) -> None:
    """Generate forward returns labels for training and evaluation."""
    from ai_candle_predictor.application.use_cases.generate_labels import generate_labels_for_symbol
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    typer.echo(f"Generating labels for {symbol} (Horizon: {horizon}, Threshold: {threshold})...")

    try:
        storage = ParquetStore()
        label_store = ParquetLabelStore()
        stats = generate_labels_for_symbol(
            Symbol(symbol), storage, label_store, horizon=horizon, threshold=threshold
        )

        typer.echo(typer.style("Label generation complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  Total samples processed: {stats['total']}")
        typer.echo(f"  UP labels generated:     {stats['up']}")
        typer.echo(f"  DOWN labels generated:   {stats['down']}")
        typer.echo(f"  Excluded samples:        {stats['excluded']}")
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def predict(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Asset ticker, e.g., BTC-USD"),
    model_label: str = typer.Option(
        "", "--model", "-m", help="Model type label, e.g. Random Forest, XGBoost"
    ),
    start_date: str = typer.Option(
        (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "--start-date",
        help="Start date for prediction window (YYYY-MM-DD)",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date for prediction window (YYYY-MM-DD). Defaults to today.",
    ),
    horizon: int = typer.Option(5, "--horizon", help="Prediction horizon in candles"),
) -> None:
    """Run model inference on a specific range of candles."""
    from ai_candle_predictor.application.use_cases.predict import predict_range
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    typer.echo(
        f"Predicting candlestick directions for {symbol} using '{model_label or 'latest'}' model..."
    )

    try:
        model_store = JoblibStore()
        feature_store = ParquetFeatureStore()
        label_store = ParquetLabelStore()
        candle_store = ParquetStore()

        result = predict_range(
            symbol=Symbol(symbol),
            model_store=model_store,
            feature_store=feature_store,
            label_store=label_store,
            candle_store=candle_store,
            start_date=start,
            end_date=end,
            model_label=model_label,
            horizon=horizon,
        )

        typer.echo(typer.style("Prediction analysis complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  Total candles predicted: {result.total_candles}")

        if result.predictions:
            correct = sum(1 for p in result.predictions if p.is_correct is True)
            labeled = sum(1 for p in result.predictions if p.is_correct is not None)
            accuracy = (correct / labeled) * 100 if labeled > 0 else 0.0

            typer.echo(f"  Correct / Labeled predictions: {correct} / {labeled}")
            typer.echo(f"  Classification accuracy:       {accuracy:.2f}%")

            # Print sample of predictions
            typer.echo("\nSample Predictions:")
            typer.echo(
                f"{'Timestamp':<20} | {'Close':<10} | {'Pred Dir':<8} | {'Conf':<6} | {'Actual Ret':<10} | {'Correct':<8}"
            )
            typer.echo("-" * 75)
            for pred in result.predictions[:5]:
                corr_str = str(pred.is_correct) if pred.is_correct is not None else "N/A"
                ret_str = f"{pred.actual_return:.2%}" if pred.actual_return is not None else "N/A"
                dir_str = "UP" if pred.predicted_direction == 1 else "DOWN"
                typer.echo(
                    f"{pred.timestamp.strftime('%Y-%m-%d %H:%M'):<20} | "
                    f"${pred.close:<9.2f} | "
                    f"{dir_str:<8} | "
                    f"{pred.confidence:<5.1%} | "
                    f"{ret_str:<10} | "
                    f"{corr_str:<8}"
                )
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def backtest(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Asset ticker, e.g., BTC-USD"),
    model_label: str = typer.Option(
        "", "--model", "-m", help="Model type label, e.g. Random Forest, XGBoost"
    ),
    start_date: str = typer.Option(
        (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        "--start-date",
        help="Start date for backtesting (YYYY-MM-DD)",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date for backtesting (YYYY-MM-DD). Defaults to today.",
    ),
    capital: float = typer.Option(10000.0, "--capital", "-c", help="Initial investment capital"),
) -> None:
    """Run quantitative strategy backtesting on predictions."""
    from ai_candle_predictor.application.use_cases.backtest import run_backtest
    from ai_candle_predictor.application.use_cases.predict import predict_range
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import ParquetLabelStore
    from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    typer.echo(f"Simulating trading strategy on {symbol} using '{model_label}' predictions...")

    try:
        model_store = JoblibStore()
        feature_store = ParquetFeatureStore()
        label_store = ParquetLabelStore()
        candle_store = ParquetStore()

        pred_result = predict_range(
            symbol=Symbol(symbol),
            model_store=model_store,
            feature_store=feature_store,
            label_store=label_store,
            candle_store=candle_store,
            start_date=start,
            end_date=end,
            model_label=model_label,
        )

        if not pred_result.predictions:
            typer.echo(
                typer.style(
                    "No predictions generated for backtesting range. Make sure data is ingested/labeled.",
                    fg=typer.colors.YELLOW,
                )
            )
            raise typer.Exit(code=0)

        result = run_backtest(pred_result.predictions, initial_capital=capital)

        typer.echo(typer.style("Backtest complete!", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  Trading Period:     {result.start_date.date()} to {result.end_date.date()}")
        typer.echo(f"  Total Trades:       {result.total_trades}")
        typer.echo(f"  Winning / Losing:   {result.winning_trades} / {result.losing_trades}")
        typer.echo(f"  Strategy Win Rate:  {result.win_rate:.2%}")
        typer.echo(f"  Initial Capital:    ${result.initial_capital:,.2f}")
        typer.echo(f"  Final Equity:       ${result.final_equity:,.2f}")

        strat_color = typer.colors.GREEN if result.strategy_return_pct >= 0 else typer.colors.RED
        bh_color = typer.colors.GREEN if result.buy_hold_return_pct >= 0 else typer.colors.RED

        typer.echo(
            "  Strategy Return:    "
            + typer.style(f"{result.strategy_return_pct:+.2f}%", fg=strat_color, bold=True)
        )
        typer.echo(
            "  Buy & Hold Return:  "
            + typer.style(f"{result.buy_hold_return_pct:+.2f}%", fg=bh_color)
        )
        typer.echo(f"  Max Drawdown:       {result.max_drawdown_pct:.2f}%")
        typer.echo(f"  Sharpe Ratio:       {result.sharpe_ratio:.4f}")
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run server on"),
    reload: bool = typer.Option(False, "--reload", help="Enable code hot-reloading"),
) -> None:
    """Start the FastAPI application REST API server."""
    import uvicorn

    typer.echo(f"Launching FastAPI web service on http://{host}:{port}...")
    try:
        uvicorn.run(
            "ai_candle_predictor.presentation.api.main:app", host=host, port=port, reload=reload
        )
    except Exception as e:
        typer.echo(typer.style(f"Server error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
