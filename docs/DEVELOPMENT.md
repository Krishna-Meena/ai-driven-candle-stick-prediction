# Developer and Contribution Guide

This document is designed for developers, quantitative engineers, and researchers who wish to extend or modify the **AI-Driven Candlestick Prediction Platform**.

---

## Development Standards & Quality Gates

To maintain production-grade standards, the codebase enforces strict style, typing, and architectural boundaries. Before submitting any changes or pull requests, you must pass all quality gates locally.

### 1. Code Style and Formatting (Black)
Ensure that all code conforms to the project's formatting rules:
```bash
# Check for formatting violations
uv run black --check src/ tests/

# Auto-format files
uv run black src/ tests/
```

### 2. Linting and Code Style Checks (Ruff)
Ruff is used to identify syntax issues, unused imports, programming errors, and code style deviations:
```bash
# Run the linter
uv run ruff check src/ tests/

# Automatically fix correctable lint warnings
uv run ruff check src/ tests/ --fix
```

### 3. Type Checking (Mypy)
Strict type definitions are enforced throughout the core application:
```bash
uv run mypy src/ tests/
```

### 4. Running the Test Suite (Pytest)
Ensure all existing unit, integration, and E2E regression tests pass:
```bash
# Run the full test suite
uv run pytest

# Run tests with terminal coverage report
uv run pytest --cov=src/ai_candle_predictor --cov-report=term-missing
```

---

## How to Extend the Platform

### Adding a New Technical Indicator (Feature)
To introduce a new feature (technical indicator) to the machine learning pipeline:

1. **Update Domain Layer Types**: Add your indicator type to the `IndicatorType` enum in [indicators.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/domain/entities/indicators.py).
2. **Implement Calculation**: In [computations.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/infrastructure/features/computations.py), modify the `compute_all` function to calculate the indicator value using `pandas-ta` or pandas.
3. **Map and Output**: Map the calculated DataFrame column back into `IndicatorValue` objects within `compute_all`.
4. **Test**: Add a unit test in `tests/unit/test_feature_utils.py` or `tests/e2e/test_pipeline.py` to assert the feature is computed.

### Adding a New Machine Learning Model
To introduce a new classification model (e.g. LightGBM, Support Vector Classifiers):

1. **Update Train Use Case**: In [train_model.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/application/use_cases/train_model.py) or by creating a new use case file like `train_lgbm.py`, define the training pipeline layout.
2. **Define Classifier & Hyperparameters**: Set up the scikit-learn standard pipeline (e.g., standard scaling followed by classifier instantiation).
3. **Support Registry**: Ensure you register the model in the JSON-backed registry and save the resulting joblib binary via the `ModelStore`.
4. **Update Dashboard Views**: Integrate the new model option into the Streamlit dashboard dropdown selectors in [app.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/presentation/dashboard/app.py).

---

## Logging Conventions
The application uses structured JSON logging via `structlog`.
Avoid standard `print` statements in core use cases. Use the structured logger to attach metadata:

```python
from ai_candle_predictor.common.logging import get_logger

log = get_logger(__name__)

# Correct usage with context variables
log.info("completed model evaluation", symbol=symbol.value, accuracy=0.854)
log.error("data provider rate-limited", provider="yahoo", status_code=429)
```
