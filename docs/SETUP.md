# Setup and Installation Guide

This guide will walk you through setting up the **AI-Driven Candlestick Prediction Platform** on your local machine.

---

## Prerequisites

Before starting, ensure you have the following installed:
1. **Python 3.13** or higher.
2. **uv** (Astral's lightning-fast Python package installer and manager).
   - Install `uv` on macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Install `uv` on Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. **Docker** (optional, if running inside containers).

---

## Local Development Installation

### 1. Clone the Repository
Clone the codebase to your local workstation:
```bash
git clone https://github.com/Krishna-Meena/ai-driven-candle-stick-prediction.git
cd ai-driven-candle-stick-prediction
```

### 2. Synchronize Virtual Environment
Initialize the environment and install dependencies. The platform uses a modular dependency tree:

- **Development Setup (Recommended)** - Installs all test engines, code checkers, and core libraries:
  ```bash
  uv sync --group dev
  ```

- **Production Setup with All Extras** - Installs FastAPI services, SHAP explainers, deep learning utilities, database libraries, and celery tasks:
  ```bash
  uv sync --all-extras
  ```

- **Combined Development & All Extras** (recruiter & local tester setup):
  ```bash
  uv sync --all-extras --group dev
  ```

This command automatically creates a virtual environment in `.venv/` and pins/installs all dependencies listed in `uv.lock`.

### 3. Activate Virtual Environment
- **macOS/Linux**: `source .venv/bin/activate`
- **Windows (CMD)**: `.venv\Scripts\activate.bat`
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`

---

## Running the Application Components

Once installed, you can launch the following three interfaces:

### A. The Interactive Analytics Dashboard (Streamlit)
Launch the Bloomberg-style analytics dashboard:
```bash
uv run streamlit run src/ai_candle_predictor/presentation/dashboard/app.py
```
This will open the application in your browser at `http://localhost:8501`.

### B. The REST API Web Server (FastAPI)
Launch the API server to query data, register models, and fetch predictions programmatically:
```bash
# Using uv CLI entrypoint
uv run ai-candle-predictor serve --port 8000

# Or directly running the uvicorn server
uv run uvicorn ai_candle_predictor.presentation.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

### C. The Command Line Interface (Typer CLI)
Execute pipeline steps, query models, and test backtests directly from your terminal:
```bash
# Show usage and available commands
uv run ai-candle-predictor --help

# Ingest historical data for BTC
uv run ai-candle-predictor ingest --symbol BTC-USD --start-date 2024-01-01

# Generate technical indicator features
uv run ai-candle-predictor compute-features --symbol BTC-USD

# Run predictions and see classifier accuracy
uv run ai-candle-predictor predict --symbol BTC-USD --model "Random Forest"

# Run a backtest strategy simulation
uv run ai-candle-predictor backtest --symbol BTC-USD --capital 25000
```

---

## Running with Docker

You can run the entire platform in a containerized environment.

### 1. Build the Docker Image
```bash
docker build -t ai-candle-predictor .
```

### 2. Run the Streamlit Dashboard
```bash
docker run -p 8501:8501 ai-candle-predictor
```
Open `http://localhost:8501` to view the dashboard.

### 3. Run the FastAPI REST API
```bash
docker run -p 8000:8000 ai-candle-predictor uvicorn ai_candle_predictor.presentation.api.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` to view the API Swagger docs.
