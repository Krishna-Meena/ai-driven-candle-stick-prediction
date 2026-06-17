# Release v0.2.0 — Premium Analytics Dashboard

> 17 June 2026

## Highlights

- **Complete UI overhaul** — Streamlit dashboard transformed into an institutional-grade analytics platform with dark professional theme, Inter typography, and custom CSS components.
- **Historical Date Range Prediction** — New `predict_range()` use case enables predictions over any date range with per-candle confidence scores, forward return comparison, and correct/incorrect flags.
- **On-demand Model Comparison** — Train LR, RF, and XGBoost directly from the UI with radar charts, performance leaderboards, and feature importance bar charts.
- **Interactive Charts** — Plotly candlestick charts with range sliders, zoom controls, and date-range presets (1M/3M/6M/YTD/1Y/All).
- **CI/CD Pipeline** — GitHub Actions workflow automates lint, test, and build on every push/PR to master.
- **114 Passing Tests** — Complete test suite: unit (Symbol, CandleStick, domain entities, events), integration (Parquet stores), and E2E (full pipeline regression).

## What's New

### Dashboard (Phase 16C)

| Component | Detail |
|-----------|--------|
| Theme | `.streamlit/config.toml` with dark base, teal accent `#00d4aa`, Inter font |
| Home | KPI panels (assets, features, labels, models), asset overview cards with price + change %, nav cards with hover effects |
| Market Overview | 6-column metrics row, range presets, Plotly candlestick + volume with rangeslider |
| Predictions | SVG radial confidence gauge, color-coded UP/DOWN/correct/incorrect badges |
| Model Comparison | On-demand training with spinner, 5-metric radar chart, leaderboard with highlighted winners, top-10 feature importances per model |
| Explainability | Tabbed SHAP image gallery, on-demand SHAP analysis with global feature ranking |
| About | Tabbed layout: Architecture, Pipeline, Tech Stack, System Info |

### Prediction System (Phase 16A)

- **New file:** `application/dto/prediction.py` — `CandlePrediction` and `PredictionResult` DTOs
- **New file:** `application/use_cases/predict.py` — `predict_range()` loads model + features, pivots, predicts, aligns with actuals
- **Dashboard integration:** Date range pickers, per-candle results table with confidence, actual return, correct/incorrect flags

### CI/CD (Phase 14)

- **New file:** `.github/workflows/ci.yml` — 3 parallel jobs: lint (ruff + black + mypy), test (pytest with xdist + coverage), build (uv build)

### Test Suite (Phase 12)

- `tests/conftest.py` — 28 shared fixtures
- `tests/unit/test_candle.py` — 22 tests (creation, properties, validation, equality)
- `tests/unit/test_symbol.py` — 23 tests (creation, validation, properties, equality)
- `tests/unit/test_domain_entities.py` — 33 tests (indicators, patterns, label, metrics)
- `tests/unit/test_events.py` — 9 tests (domain events)
- `tests/integration/test_parquet_stores.py` — 16 tests (ParquetStore, ParquetFeatureStore, ParquetLabelStore)
- `tests/e2e/test_pipeline.py` — 11 tests (feature computation, label generation, full pipeline)

## Files Changed (this release)

```
 .github/workflows/ci.yml                          |  95 +++
 .streamlit/config.toml                             |  14 +
 README.md                                          | 155 +++--
 reports/release-notes-v0.2.0.md                    |  79 +++
 src/.../application/dto/prediction.py               |  31 +
 src/.../application/use_cases/predict.py             | 143 ++++
 src/.../common/config/settings.py                    |   4 +-
 src/.../presentation/dashboard/app.py                | 691 ++++++++--------
 tests/conftest.py                                   | 258 +++++++
 tests/e2e/test_pipeline.py                          | 214 ++++++
 tests/integration/test_parquet_stores.py             | 131 ++++
 tests/unit/test_candle.py                           | 283 +++++++
 tests/unit/test_domain_entities.py                  | 216 ++++++
 tests/unit/test_events.py                           |  77 ++
 tests/unit/test_symbol.py                           | 110 +++
 15 files changed, 2219 insertions(+), 282 deletions(-)
```

## How to Run

```bash
uv sync --group dev
uv run streamlit run src/ai_candle_predictor/presentation/dashboard/app.py
```

Opens at `http://localhost:8501`.
