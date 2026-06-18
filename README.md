# AI-Driven Candlestick Prediction Platform

![Python](https://img.shields.io/badge/python-3.13-blue)
![UV](https://img.shields.io/badge/packaging-uv-ffd242)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-black-000000)
![Tests](https://img.shields.io/badge/tests-169%20passed-brightgreen)

Production-grade machine learning platform that predicts financial market candlestick
patterns using technical indicators, ensemble models (Logistic Regression, Random Forest,
XGBoost), and hyperparameter tuning — all delivered through a professional Streamlit dashboard.

Built with **Clean Architecture** and **strict dependency inversion**.

---

## Live Demo

> *Demo GIF coming soon 

```
Home → Market Overview → Training Center → Predictions → Model Comparison → Backtesting → About
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi Asset Support** | Trade across multiple symbols simultaneously |
| **Technical Indicators** | 8+ indicators (RSI, MACD, BB, SMA, EMA, ATR, etc.) via `pandas-ta` |
| **3 Model Architectures** | Logistic Regression, Random Forest, XGBoost |
| **Hyperparameter Tuning** | Optuna-backed Bayesian optimization |
| **Backtesting Engine** | Strategy simulation with equity curves, trade logs, 5 KPI metrics |
| **Date Range Predictions** | Historical date range prediction with confidence scoring |
| **Model Registry** | Persistent JSON-backed metadata store |
| **Model Training Center** | Interactive training with live progress logs |
| **Executive Dashboard** | 5 KPI cards, asset grid, model leaderboard, activity feed |
| **Dark Professional Theme** | Bloomberg/TradingView-inspired Black + Gold UI |
| **Docker Support** | Containerized deployment ready |
| **CI/CD Pipeline** | GitHub Actions: lint, test, build in 3 parallel jobs |

---

## Quickstart

```bash
git clone https://github.com/Krishna-Meena/ai-driven-candle-stick-prediction.git
cd ai-driven-candle-stick-prediction

# Install all dependencies (including dev tools)
uv sync --group dev

# Launch the dashboard
uv run streamlit run src/ai_candle_predictor/presentation/dashboard/app.py
```

Opens at `http://localhost:8501` with a premium Black + Gold institutional theme.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Executive dashboard — 5 KPI cards, asset market snapshot grid, model leaderboard, recent training runs, quick-action buttons |
| **Data Pipeline** | Per-asset pipeline cards showing ingest/compute/label status with batch operations |
| **Market Overview** | Interactive OHLCV candlestick chart + volume, range presets (1M/3M/6M/YTD/1Y/All), zoom, expanded data table |
| **Predictions** | Date range predictor, confidence gauge (SVG radial), KPI metrics, distribution histogram, prediction table |
| **Model Comparison** | On-demand LR/RF/XGB training, radar chart, performance leaderboard, top-10 feature importance bars |
| **Model Insights** | Feature importance from native model weights, model metrics from registry (ROC-AUC, Precision, Recall, F1) |
| **Training Center** | Asset/model selectors, configurable hyperparams, live progress bar + log, instant metrics display |
| **Backtesting** | Strategy simulation, 5 KPI metrics, 3-chart tabs (equity curve, drawdown, returns), trade log |
| **About** | Architecture diagram, pipeline steps, tech stack table, system info |

### Theme

- **Black + Gold institutional theme** — `#050505` background, `#F5C542` primary gold
- **Glassmorphism cards** with hover effects, custom sidebar with logo
- **Inter font** for modern typography
- **Custom Plotly template** `black_gold` — all 14+ charts use the same palette

---

## Architecture

```
src/
├── domain/          Core entities, value objects, domain events (zero deps)
├── application/     Use cases, ports, DTOs (orchestration layer)
├── infrastructure/  Data providers, persistence, model serving, visualization
├── presentation/    Streamlit dashboard, CLI, API stubs
└── common/          Config (pydantic-settings), logging (structlog), exceptions
```

**Clean Architecture** — outer layers depend on inner layers, never the reverse.
Domain has zero external dependencies. Use cases depend only on abstractions (ports).

---

## Pipeline

```
Data Ingestion (yfinance) → ParquetStore
    ↓
Feature Engineering (8 indicators via pandas-ta) → ParquetFeatureStore
    ↓
Label Engineering (forward returns, horizon=5) → ParquetLabelStore
    ↓
Model Training (LR / RF / XGBoost) → JoblibStore + ModelRegistry
    ↓
Prediction (predict_range) → CandlePrediction[]
    ↓
Model Insights (feature importance, metrics) → Dashboard
```

---

## Screenshots

> *Add screenshots here — capture each dashboard page and save to `docs/screenshots/`.*

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Packaging | UV |
| Data | Pandas, NumPy, PyArrow, pandas-ta |
| ML | Scikit-learn, XGBoost, Optuna |
| Visualization | Plotly, Matplotlib, mplfinance |
| Dashboard | Streamlit (custom Black + Gold theme) |
| Config | pydantic-settings |
| Logging | structlog |
| Quality | Ruff, Black, Mypy |
| Testing | Pytest, Pytest-cov, Pytest-xdist (169 tests) |
| CI/CD | GitHub Actions (3 parallel jobs) |

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run all 169 tests
uv run pytest

# Quality gates
uv run ruff check .
uv run black --check .
uv run mypy src/ tests/
```

All four gates must pass before commit.

---

## Future Improvements

- **SHAP Explainability** (planned) — model-agnostic feature attribution
- **Deep Learning Models** — LSTM, Transformer architectures
- **Real-time Data** — WebSocket streaming via CCXT
- **REST API** — FastAPI serving layer for model predictions
- **Docker Compose** — Multi-service deployment with Redis/Celery

---

## Project Impact

This project demonstrates production-grade ML engineering skills:

- **Clean Architecture** — strict dependency inversion, testable design
- **Full ML Pipeline** — ingestion → features → labels → training → prediction → evaluation
- **Professional UI** — institutional-grade Streamlit dashboard with custom theming
- **Test Coverage** — 169 unit/integration/e2e tests with 4 quality gates
- **CI/CD** — automated lint, test, build pipeline

---

## License

MIT — Built by Krishna S Meena
