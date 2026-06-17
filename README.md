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
├── presentation/    FastAPI + Streamlit + CLI
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

# Activate
.venv\Scripts\activate

# Run lint & type check
ruff check src/
mypy src/
```

---

## Project Structure

```
ai-candle-predictor/
├── config/               Environment-specific configuration files
├── data/                 Data storage (raw, processed, external)
├── docs/                 Architecture Decision Records and API docs
├── models/               Serialized model artifacts and registry
├── notebooks/            EDA and experimentation notebooks
├── reports/              Generated metrics, figures, and exports
├── scripts/              One-shot DevOps and utility scripts
├── src/                  Application source code
│   ├── domain/           Entities, value objects, domain events
│   ├── application/      Use cases, ports, DTOs
│   ├── infrastructure/   Data ingestion, persistence, model serving
│   ├── presentation/     FastAPI routes, Streamlit pages, CLI commands
│   └── common/           Configuration, logging, exceptions
├── tests/                Unit, integration, and end-to-end tests
└── pyproject.toml        Single source of truth for deps & tooling
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Packaging | UV |
| Data | Pandas, NumPy, Polars |
| ML | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch, PyTorch Forecasting |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Database | PostgreSQL, SQLAlchemy, Alembic |
| Task Queue | Celery, Redis |
| Experiment Tracking | MLflow |
| Quality | Ruff, Black, Mypy, Pre-commit |
| Testing | Pytest, Pytest-cov, Pytest-xdist |

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Format code
black src/ tests/

# Type check
mypy src/
```

---

## License

MIT
