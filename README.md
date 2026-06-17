# AI-Driven Candlestick Prediction Platform

Production-grade machine learning platform for predicting financial market
candlestick patterns using technical indicators, ensemble models, and deep
learning.

![Python](https://img.shields.io/badge/python-3.13-blue)
![UV](https://img.shields.io/badge/packaging-uv-ffd242)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-black-000000)

---

## Architecture

```
src/
├── domain/          Core entities & business rules (zero deps)
├── application/     Use cases & orchestration
├── infrastructure/  Data providers, DB, model serving
├── presentation/    Streamlit dashboard + CLI + API stubs
└── common/          Config, logging, exceptions
```

Clean Architecture with strict dependency inversion — outer layers depend on
inner layers, never the reverse.

---

## Quickstart

```bash
# Clone & enter
git clone https://github.com/yourusername/ai-candle-predictor.git
cd ai-candle-predictor

# Create virtual env & install all deps
uv sync --group dev

# Launch the Streamlit dashboard
uv run streamlit run src/ai_candle_predictor/presentation/dashboard/app.py
```

The dashboard opens at `http://localhost:8501` with a professional dark theme.

---

## Dashboard

### Pages

| Page | Description |
|------|-------------|
| **Home** | KPI panels (assets tracked, feature rows, labeled samples, trained models), asset overview cards, quick navigation |
| **Market Overview** | Interactive OHLCV candlestick chart with volume, range presets (1M/3M/6M/YTD/1Y/All), zoom + range slider, expanded raw data table |
| **Predictions** | Date range selector, confidence gauge (SVG radial), KPI metrics, confidence distribution histogram, full per-candle prediction table |
| **Model Comparison** | On-demand LR/RF/XGB training with radar chart, performance leaderboard with highlights, top-10 feature importance bar charts per model type |
| **Explainability** | SHAP visualization gallery (tabs per image), on-demand SHAP analysis with global feature ranking bar chart and sorted importance table |
| **About** | Tabbed view: Architecture diagram, Pipeline steps, Tech Stack table, System Info metrics |

### Theme

- **Dark professional theme** via `.streamlit/config.toml` (base dark, teal accent `#00d4aa`)
- **Inter font** for modern typography
- **CSS customizations**: KPI panels with gradient backgrounds and hover effects, asset cards, badge styling, leaderboard rows, gauge containers
- **Interactive Plotly charts** with `plotly_dark` template, rangesliders, hover analytics

---

## Project Structure

```
ai-candle-predictor/
├── .streamlit/            Streamlit theme configuration
├── config/                Environment-specific configuration files
├── data/                  Data storage (raw, processed, external)
├── docs/                  Architecture Decision Records and API docs
├── models/                Serialized model artifacts and registry
├── notebooks/             EDA and experimentation notebooks
├── reports/               Generated metrics, figures, SHAP charts
├── scripts/               One-shot DevOps and utility scripts
├── src/                   Application source code
│   ├── domain/            Entities, value objects, domain events
│   ├── application/       Use cases, ports, DTOs
│   ├── infrastructure/    Data ingestion, persistence, model serving
│   ├── presentation/      Streamlit dashboard, CLI, API stubs
│   └── common/            Configuration, logging, exceptions
├── tests/                 Unit, integration, and end-to-end tests
└── pyproject.toml         Single source of truth for deps & tooling
```

---

## Pipeline

```
Data Ingestion (yfinance) → ParquetStore
    ↓
Feature Engineering (8 indicators via pandas-ta) → ParquetFeatureStore
    ↓
Label Engineering (forward returns, horizon=5) → ParquetLabelStore
    ↓
Model Training (LR / RF / XGBoost) → JoblibStore
    ↓
Prediction (predict_range use case) → CandlePrediction[]
    ↓
Explainability (SHAP global + local) → ImageStore
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Packaging | UV |
| Data | Pandas, NumPy, PyArrow, pandas-ta |
| ML | Scikit-learn, XGBoost, Optuna, SHAP |
| Visualization | Plotly, Matplotlib, mplfinance |
| Dashboard | Streamlit (custom dark theme) |
| Config | pydantic-settings |
| Logging | structlog |
| Quality | Ruff, Black, Mypy |
| Testing | Pytest, Pytest-cov, Pytest-xdist (114 tests) |

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (114 total)
pytest

# Run linter
ruff check src/ tests/

# Format code
black src/ tests/

# Type check
mypy src/
```

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- **lint**: ruff check + black --check + mypy
- **test**: pytest -n auto with coverage on Python 3.13
- **build**: uv build via hatchling, wheel artifact uploaded

---

## License

MIT
