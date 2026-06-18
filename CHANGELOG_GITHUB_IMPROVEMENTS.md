# Changelog: GitHub Repository Professionalization & Overhaul

This changelog records all updates, files added, files modified, and engineering rationales completed to elevate the **AI-Driven Candlestick Prediction Platform** to a production-ready portfolio-grade project.

---

## Summary of Changes

### 1. Presentation Layer Expansion
- **[NEW] [cli/main.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/presentation/cli/main.py)**: Added a command-line interface using `typer`. Exposes pipeline operations (ingest, compute features, generate labels), prediction analysis, backtest simulations, and REST server serving.
- **[NEW] [api/main.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/src/ai_candle_predictor/presentation/api/main.py)**: Added a FastAPI REST application. Exposes core pipeline actions and serving endpoints as a standard HTTP REST API. Includes CORS middleware for frontend/external client integrations.
- **[MODIFY] [pyproject.toml](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/pyproject.toml)**:
  - Registered the `ai-candle-predictor` command in `[project.scripts]` mapping to `cli.main:app`.
  - Added `httpx` to development dependencies for API client testing.

### 2. Comprehensive Test Suite Extensions
- **[NEW] [tests/unit/test_cli.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/tests/unit/test_cli.py)**: Added unit tests covering all Typer commands with mocked orchestrators.
- **[NEW] [tests/unit/test_api.py](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/tests/unit/test_api.py)**: Added integration tests for all FastAPI endpoints using the Starlette `TestClient` and mocked services.

### 3. Exhaustive Documentation Suite
- **[NEW] [docs/ARCHITECTURE.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/docs/ARCHITECTURE.md)**: Created a deep architectural specification explaining Clean Architecture layers, data processing, and ML models lifecycle with Mermaid flowchart/sequence diagrams.
- **[NEW] [docs/SETUP.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/docs/SETUP.md)**: Added setup instructions for development (using `uv`) and production (Docker).
- **[NEW] [docs/DEVELOPMENT.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/docs/DEVELOPMENT.md)**: Created a contributor/developer guide on code standards, checking gates, and extending models/features.
- **[NEW] [docs/DEPLOYMENT.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/docs/DEPLOYMENT.md)**: Outlined VM services setup (systemd + nginx configurations), container run commands, and Streamlit Cloud configurations.
- **[NEW] [docs/TROUBLESHOOTING.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/docs/TROUBLESHOOTING.md)**: Documented standard error logs (rate limits, missing indices, registry corruption) and their solutions.

### 4. GitHub Professionalization & Showcases
- **[MODIFY] [README.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/README.md)**: Rewrote the project documentation completely. Added shields/badges, clean features grids, architecture descriptions, mathematical formulas for labeling and backtesting metrics, and high-impact resume bullets.
- **[NEW] [PROJECT_SHOWCASE.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/PROJECT_SHOWCASE.md)**: Created a portfolio showcase guide highlighting engineering patterns, Clean Architecture decoupling, and backtest alphas for technical hiring managers.
- **[MODIFY] [.github/workflows/ci.yml](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/.github/workflows/ci.yml)**: Updated dependencies installation step to use `--all-extras` so FastAPI and ML packages are available on CI.
- **[NEW] [.github/ISSUE_TEMPLATE/bug_report.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/.github/ISSUE_TEMPLATE/bug_report.md)**: Standardized bug logging format.
- **[NEW] [.github/ISSUE_TEMPLATE/feature_request.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/.github/ISSUE_TEMPLATE/feature_request.md)**: Structured enhancement suggestions template.
- **[NEW] [.github/PULL_REQUEST_TEMPLATE.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/.github/PULL_REQUEST_TEMPLATE.md)**: Code review checkbox checklist.
- **[NEW] [CONTRIBUTING.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/CONTRIBUTING.md)**: PR contribution standards and testing workflows.
- **[NEW] [CODE_OF_CONDUCT.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/CODE_OF_CONDUCT.md)**: Standard Contributor Covenant conduct code.
- **[NEW] [SECURITY.md](file:///c:/Dev/Projects/ai-driven-candle-stick-prediction/SECURITY.md)**: Vulnerability disclosure guidelines.

---

## Technical Rationale
The primary goal of this overhaul was to bridge the gap between a standalone Streamlit dashboard and a production-grade machine learning platform. By implementing standard REST APIs and command-line interfaces, the platform becomes integrateable into automated Cron data pipelines, CI/CD train schedules, and microservices topologies. The detailed documentation, community guidelines, and rigorous testing demonstrate a FAANG-ready software engineering execution.
