# Troubleshooting Guide

This guide compiles common issues, error messages, and their solutions when deploying or developing on the **AI-Driven Candlestick Prediction Platform**.

---

## 1. Data Ingestion Errors

### Issue: yfinance API Rate Limiting or Timed Out
- **Symptom**: Ingestion logs show HTTP warnings, or requests return empty DataFrames with logs like `Failed to fetch from Yahoo Provider`.
- **Cause**: Yahoo Finance blocks/throttles IP addresses that query too many tickers or too many historical date ranges in rapid succession.
- **Solution**:
  1. Reduce the frequency of ingestion calls.
  2. Implement an exponential backoff. The platform already uses `tenacity` for retries, but if rate-limiting persists, consider switching the IP address or running the task using a rotating proxy list.
  3. Narrow down the date range window.

### Issue: No Data Saved to Parquet or empty directory
- **Symptom**: Directory `data/raw/` remains empty after running ingestion, or use cases raise `ValueError: no candles available`.
- **Cause**: The ticker symbol requested is invalid, not listed on Yahoo Finance, or the date range requested is invalid (e.g. start date is in the future).
- **Solution**:
  1. Verify the ticker symbol exists on Yahoo Finance (e.g. `BTC-USD`, `ETH-USD`, `RELIANCE.NS`, `^NSEI`).
  2. Ensure date range inputs are formatted as `YYYY-MM-DD` and represent a historical window where data exists.

---

## 2. Feature & Label Pipeline Errors

### Issue: `ValueError: Feature computation failed...`
- **Symptom**: Compute features step terminates with errors about missing DataFrame columns or index mismatch.
- **Cause**: Calculated features depend on technical indicators that require a minimum lookback window (e.g. 50-day SMA, 14-day RSI). If the raw dataset contains fewer candles than the minimum lookback, calculations will produce only NaN values.
- **Solution**:
  1. Ingest a larger historical date range (at least 100+ candles).
  2. Ensure that you run the `ingest` command before running `compute-features`.

---

## 3. Model Training & Deserialization Errors

### Issue: Model file missing or registry key mismatch
- **Symptom**: Predictions or dashboard tabs return `no model registered` or throw errors like `FileNotFoundError: ...joblib`.
- **Cause**: The model registry file `registry.json` refers to a model version or file name that has been deleted from the `models/` directory, or training was not completed.
- **Solution**:
  1. Check the contents of your model directory (default `models/`) for the presence of the `.joblib` model binary.
  2. If the registry is corrupted, delete `models/registry.json` and retrain the model. The registry will auto-recreate and self-heal during the next training run.

### Issue: `ModuleNotFoundError: No module named 'fastapi'` (or similar)
- **Symptom**: Serving commands fail on startup.
- **Cause**: Dependencies for web-serving (`api`) or deep learning (`dl`) are declared as optional package extras and are not installed in the basic environment.
- **Solution**:
  - Re-sync your environment with the appropriate extras flag:
    ```bash
    uv sync --all-extras
    ```

---

## 4. UI / Dashboard Display Issues

### Issue: Dashboards show "No Data Ingested Yet"
- **Symptom**: Opening `http://localhost:8501` displays gray warning boxes or empty layouts.
- **Cause**: The app is querying directories `data/raw` or `models/` which have not yet been populated.
- **Solution**:
  - In the Streamlit sidebar, click **Data Pipeline**, select a symbol, and click **Ingest Raw Data**.
  - Once raw data is ingested, click **Compute Features** and **Generate Labels**.
  - Navigate to **Training Center** to train at least one model.
