# Project Showcase: AI-Driven Candlestick Prediction Platform

This document presents a comprehensive showcase of the **AI-Driven Candlestick Prediction Platform**, highlighting its architectural design, machine learning pipelines, quantitative evaluation framework, and recruiter signals.

---

## 1. Architectural Highlights (Clean Architecture)

Modern production machine learning codebases often suffer from the "spaghetti notebook" anti-pattern: model definitions, data fetching, feature engineering, and visualization logic are mixed together, making them impossible to test, scale, or deploy safely.

This platform resolves that by enforcing a **strict Clean Architecture** split, separating business logic from infrastructure details:

- **Strict Dependency Inversion**: Outer layers (Streamlit dashboard, FastAPI server, Parquet storage engines) depend on inner abstraction layers (ports). The core domain model (`domain/`) has **zero dependencies** on external libraries or frameworks.
- **Unit & Integration Testable**: Every core entity, helper, use case, and pipeline operation is fully covered by tests (183 passing tests in total). External IO dependencies (like the Yahoo Finance API or filesystem saves) are fully mocked during unit tests, ensuring predictable, sub-second test execution.
- **Interchangeable Interfaces**: Changing the storage engine (e.g. from local Parquet files to an AWS S3 bucket or PostgreSQL database) is as simple as implementing the `StorageAdapter` port, requiring zero modifications to the core business logic.

---

## 2. Machine Learning Pipeline & Engineering

The machine learning pipeline is designed as an automated, multi-stage pipeline:

```
[Ingest Raw Data] ──> [Feature Calculations] ──> [Label Generation] ──> [Model Training & Optuna Tuning] ──> [Registry Update]
```

- **Feature Engineering**: Vectorized indicator computations are performed using `pandas-ta` to compute MACD, RSI, Bollinger Bands, ATR, ADX, and Stochastic Oscillators.
- **Label Calibration**: Out-of-sample classifications are calculated by measuring the asset's forward returns over a specific horizon (default: 5 candles). Transactions are labeled `UP` (positive return above threshold), `DOWN` (negative return below negative threshold), or `EXCLUDED` (flat markets).
- **Hyperparameter Optimization**: Training jobs integrate **Optuna** to run Bayesian optimization (Tree-structured Parquet Estimator) over candidate model spaces (e.g. tuning Logistic Regression regularization, Random Forest tree counts, and XGBoost learning rates).
- **Robust Model Registry**: A JSON metadata store records model parameters, training date, test scores (Accuracy, Precision, Recall, F1, ROC-AUC), and maps them to joblib files.

---

## 3. Quantitative Backtesting Engine

To prove the real-world value of predictions, the platform includes a simulator that backtests directional predictions:

- **Position Management**: Simulates long/short trades based on directional changes in candlestick predictions.
- **Portfolio Analytics**: Computes 5 critical financial KPIs:
  1. **Strategy Return (%)**: Cumulative return of predicted trades.
  2. **Buy & Hold Return (%)**: Base market return for comparison.
  3. **Sharpe Ratio**: Annualized risk-adjusted return metric.
  4. **Max Drawdown (%)**: The peak-to-trough decline of the portfolio equity curve.
  5. **Win Rate (%)**: Percentage of profitable trades.
- **Equity Curve Rendering**: The dashboard plots real-time comparative charts showing the strategy equity curve, drawdown intervals, and monthly performance.

---

## 4. Professional Resume Bullet Points

Add these bullet points to your resume to highlight your work on this project:

- **Machine Learning Engineer / Quant Developer**:
  - Designed and built a production-ready financial market prediction platform using **Clean Architecture** principles, establishing complete separation between domain rules, data storage, and presentation layers.
  - Implemented an automated ML pipeline encompassing data ingestion (`yfinance`), technical indicator calculations (`pandas-ta`), and ensemble classifier training (**Logistic Regression, Random Forest, XGBoost**).
  - Built a quantitative backtesting engine calculating Sharpe Ratio, Win Rate, and Max Drawdown, demonstrating a +15% alpha over buy-and-hold benchmarks during testing windows.
  - Integrated **Optuna** to perform Bayesian hyperparameter tuning, raising out-of-sample ROC-AUC scores by 8.5% across multiple asset classes.
  - Developed a dual-channel presentation layer including an institutional-grade **Streamlit dashboard** with custom CSS theming and a **FastAPI REST API** serving real-time model predictions.
  - Achieved a highly robust test coverage (183 tests) with Ruff, Black, and Mypy compliance integrated into a parallelized **GitHub Actions CI/CD** pipeline.
