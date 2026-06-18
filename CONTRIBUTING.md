# Contributing Guidelines

We welcome contributions to the **AI-Driven Candlestick Prediction Platform**! Whether you are writing code, fixing typos in the documentation, or filing issue bug reports, your help is highly appreciated.

---

## How to Contribute

### 1. Reporting Bugs & Requesting Features
- Search existing issues to ensure your topic hasn't already been discussed.
- Use the provided **Bug Report** or **Feature Request** issue templates under the [Issues](https://github.com/Krishna-Meena/ai-driven-candle-stick-prediction/issues) tab.
- Be as detailed as possible, providing OS details, configurations, and logs.

### 2. Submitting Pull Requests (PR)
- Fork the repository and create your branch from `master`.
- Keep changes concise, focused, and well-tested.
- Ensure your commits use clear, conventional message structures (e.g. `feat: add SVM classifier`, `fix: correct MACD calculation bounds`).
- Run quality checks locally before submitting (see below).

---

## Local Development Workflow

1. Install prerequisites: Python 3.13 and `uv`.
2. Sync dependencies:
   ```bash
   uv sync --all-extras --group dev
   ```
3. Create your feature branch: `git checkout -b feature/your-feature`
4. Write code, add tests under `tests/unit` or `tests/integration`.
5. Run code verification locally:
   ```bash
   uv run black .
   # Run linter
   uv run ruff check . --fix
   # Run type check
   uv run mypy src/ tests/
   # Run tests
   uv run pytest
   ```
6. Open a Pull Request on GitHub, filling out the Pull Request template checklist.
