from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.domain.value_objects.symbol import Symbol
from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
    ParquetFeatureStore,
)
from ai_candle_predictor.infrastructure.labeling.parquet_label_store import (
    ParquetLabelStore,
)

st.set_page_config(
    page_title="AI Candle Predictor",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #888; margin-top: 0; }
    .metric-card {
        background: #1a1a2e; border-radius: 12px; padding: 20px;
        text-align: center; border: 1px solid #333; margin-bottom: 16px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .nav-card {
        background: #16213e; border-radius: 10px; padding: 16px;
        border: 1px solid #2a2a4a; cursor: pointer;
        transition: border-color 0.2s;
    }
    .nav-card:hover { border-color: #00d4aa; }
    .nav-card-title { font-size: 1rem; font-weight: 600; }
    .nav-card-desc { font-size: 0.8rem; color: #888; }
    .stApp { background: #0f0f23; }
    .stButton>button { border-radius: 8px; }
    footer { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

MODEL_DIR = settings.models_dir
RAW_DIR = settings.data_raw_dir
SYMBOLS = settings.default_symbols

if "symbol" not in st.session_state:
    st.session_state.symbol = "BTC-USD"


@st.cache_data(ttl=60)
def _count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    import pandas as pd

    df = pd.read_parquet(path)
    return len(df)


@st.cache_data(ttl=60)
def _load_feature_count(symbol: str) -> int:
    fs = ParquetFeatureStore()
    feats = fs.load(Symbol(symbol))
    return len(feats) if feats else 0


@st.cache_data(ttl=60)
def _load_label_count(symbol: str) -> int:
    ls = ParquetLabelStore()
    labels = ls.load(Symbol(symbol))
    return len(labels) if labels else 0


@st.cache_data(ttl=60)
def _count_models() -> int:
    if not MODEL_DIR.exists():
        return 0
    return len(list(MODEL_DIR.glob("*.joblib")))


@st.cache_data(ttl=60)
def _list_models() -> list[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(str(p.name) for p in MODEL_DIR.glob("*.joblib"))


@st.cache_data(ttl=60)
def _list_symbols_with_data() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_DIR.glob("*.parquet"))


# ── Page functions ──────────────────────────────────────────────────────────


def page_home() -> None:
    st.markdown('<p class="main-header">🕯️ AI Candle Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Production-grade candlestick prediction platform</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    symbols_with_data = _list_symbols_with_data()
    model_count = _count_models()
    feature_total = sum(_load_feature_count(s) for s in symbols_with_data)
    label_total = sum(_load_label_count(s) for s in symbols_with_data)

    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            f"{len(symbols_with_data)}</div>"
            '<div class="metric-label">Symbols Ingested</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            f"{feature_total:,}</div>"
            '<div class="metric-label">Feature Rows</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            f"{label_total:,}</div>"
            '<div class="metric-label">Labeled Samples</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            f"{model_count}</div>"
            '<div class="metric-label">Trained Models</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Quick Navigation")
    nav_cols = st.columns(3)
    nav_items = [
        ("📊", "Market Overview", "Explore OHLCV data and volume patterns"),
        ("🤖", "Predictions", "View model predictions and trade signals"),
        ("📈", "Model Comparison", "Compare LR, RF, and XGBoost side by side"),
        ("🔬", "Explainability", "SHAP feature importance and local explanations"),
        ("ℹ️", "About", "Project architecture, tech stack, and version info"),
    ]
    for i, (icon, title, desc) in enumerate(nav_items):
        col = nav_cols[i % 3]
        with col:
            st.markdown(
                f'<div class="nav-card">'
                f'<div class="nav-card-title">{icon} {title}</div>'
                f'<div class="nav-card-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    if symbols_with_data:
        st.markdown("### Available Symbols")
        st.text(", ".join(symbols_with_data))

    st.caption(f"Data directory: {RAW_DIR}")


def page_market_overview() -> None:
    st.markdown('<p class="main-header">📊 Market Overview</p>', unsafe_allow_html=True)
    st.divider()

    symbol = st.selectbox("Select Symbol", SYMBOLS, key="market_symbol")
    raw_path = RAW_DIR / f"{symbol}.parquet"

    if not raw_path.exists():
        st.warning(f"No data found for {symbol}. Run data ingestion first.")
        return

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.read_parquet(raw_path)
    df = df.sort_index()
    rows = _count_parquet_rows(raw_path)

    ts0 = df.index[0]
    ts1 = df.index[-1]
    start = ts0.strftime("%Y-%m-%d") if hasattr(ts0, "strftime") else str(ts0)
    end = ts1.strftime("%Y-%m-%d") if hasattr(ts1, "strftime") else str(ts1)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Rows", f"{rows:,}")
    with mc2:
        st.metric("From", start)
    with mc3:
        st.metric("To", end)
    with mc4:
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        pct = ((last_close - prev_close) / prev_close) * 100
        st.metric("Last Close", f"${last_close:,.2f}", f"{pct:+.2f}%")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(0,212,170,0.3)"),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Raw Data")
    with st.expander("Show raw data"):
        st.dataframe(df.tail(100), use_container_width=True)


def page_predictions() -> None:
    st.markdown('<p class="main-header">🤖 Predictions</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found. Train a model first.")
        return

    symbol = st.selectbox("Symbol", SYMBOLS, key="pred_symbol")
    model_name = st.selectbox("Model", models, key="pred_model")

    try:
        import joblib

        pipeline = joblib.load(MODEL_DIR / model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    fs = ParquetFeatureStore()
    ls = ParquetLabelStore()
    sym = Symbol(symbol)

    features = fs.load(sym)
    labels = ls.load(sym)
    if not features or not labels:
        st.warning("No features or labels for this symbol.")
        return

    from ai_candle_predictor.application.use_cases.train_baseline import (
        _pivot_features,
        _pivot_labels,
    )

    feature_df = _pivot_features(features)
    label_df = _pivot_labels(labels)
    merged = feature_df.merge(label_df, left_index=True, right_index=True, how="inner")

    X = merged.drop(columns=["label", "forward_return"]).values
    y = merged["label"].values

    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    results = merged[["forward_return"]].copy()
    results["probability"] = y_prob
    results["prediction"] = y_pred
    results["actual"] = y

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predictions Made", len(results))
    with col2:
        correct = int((results["prediction"] == results["actual"]).sum())
        acc = correct / len(results) * 100
        st.metric("Accuracy (in-sample)", f"{acc:.1f}%")

    import plotly.graph_objects as go

    fig_prob = go.Figure()
    pos = results[results["actual"] == 1]["probability"]
    neg = results[results["actual"] == 0]["probability"]
    fig_prob.add_trace(go.Histogram(x=pos, name="Up (actual)", opacity=0.7, marker_color="#00d4aa"))
    fig_prob.add_trace(
        go.Histogram(x=neg, name="Down (actual)", opacity=0.7, marker_color="#ff6b6b")
    )
    fig_prob.update_layout(
        barmode="overlay",
        template="plotly_dark",
        title="Predicted Probability Distribution by Actual Class",
        xaxis_title="P(UP)",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Recent Predictions")
    styled = results.tail(50).style.format({"probability": "{:.4f}", "forward_return": "{:.6f}"})
    st.dataframe(styled, use_container_width=True)


def page_model_comparison() -> None:
    st.markdown('<p class="main-header">📈 Model Comparison</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found.")
        return

    import pandas as pd

    rows_list = [{"Model": m, "Label": m.replace(".joblib", "")} for m in models]
    st.markdown("### Trained Models")
    st.dataframe(pd.DataFrame(rows_list).style.hide(axis="index"), use_container_width=True)

    feature_total = _load_feature_count(st.session_state.symbol)
    label_total = _load_label_count(st.session_state.symbol)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Features", f"{feature_total:,}")
    with col2:
        st.metric("Available Labels", f"{label_total:,}")

    st.info(
        "Run the model comparison pipeline to see side-by-side metrics:\n\n"
        '`python -c "from ai_candle_predictor.infrastructure.visualization'
        '.model_comparison import compare_models; ..."`'
    )


def page_explainability() -> None:
    st.markdown('<p class="main-header">🔬 Explainability</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found.")
        return

    symbol = st.selectbox("Symbol", SYMBOLS, key="explain_symbol")
    st.selectbox("Model", models, key="explain_model")

    charts_dir = settings.reports_dir / "charts" / symbol
    images = sorted(charts_dir.glob("*.png")) if charts_dir.exists() else []

    st.markdown("### SHAP Visualizations")

    shap_images = [p for p in images if "0002" in p.stem or "0003" in p.stem]
    if shap_images:
        for img in shap_images:
            st.image(str(img), use_container_width=True)
    else:
        st.info(
            "No SHAP plots found. Run SHAP analysis first:\n\n"
            '`python -c "from ai_candle_predictor.infrastructure.explainability'
            '.shap_analyzer import shap_analysis; ..."`'
        )

    st.markdown("### Feature Ranking")
    st.info("SHAP global feature ranking will appear here after analysis.")


def page_about() -> None:
    st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("""
        **AI Candle Predictor** is a production-grade platform for financial
        candlestick prediction using machine learning.

        ### Architecture
        - **Clean Architecture** with strict dependency inversion
        - **Domain layer** — entities, value objects, events (zero deps)
        - **Application layer** — ports, DTOs, use cases
        - **Infrastructure layer** — data providers, persistence, models, viz

        ### Pipeline
        1. **Data Ingestion** — yfinance → Parquet (OHLCV)
        2. **Feature Engineering** — 8 technical indicators via pandas-ta
        3. **Label Engineering** — forward returns → binary UP/DOWN
        4. **Model Training** — LogisticRegression, RandomForest, XGBoost
        5. **Hyperparameter Tuning** — Optuna (TPE sampler)
        6. **Explainability** — SHAP (global + local explanations)

        ### Tech Stack
        - **Language:** Python 3.13
        - **Package Manager:** UV
        - **ML:** scikit-learn, XGBoost, Optuna, SHAP
        - **Data:** pandas, numpy, pyarrow, pandas-ta
        - **Visualization:** Plotly, Matplotlib
        - **Dashboard:** Streamlit
        - **Config:** pydantic-settings
        - **Logging:** structlog
        """)

    st.markdown("### System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Directory", str(MODEL_DIR))
        st.metric("Raw Data Directory", str(RAW_DIR))
    with col2:
        import platform

        st.metric("Python", platform.python_version())
        st.metric("Platform", platform.platform())


# ── Navigation ──────────────────────────────────────────────────────────────

nav = st.navigation(
    [
        st.Page(page_home, title="Home", icon="🏠", default=True),
        st.Page(page_market_overview, title="Market Overview", icon="📊"),
        st.Page(page_predictions, title="Predictions", icon="🤖"),
        st.Page(page_model_comparison, title="Model Comparison", icon="📈"),
        st.Page(page_explainability, title="Explainability", icon="🔬"),
        st.Page(page_about, title="About", icon="ℹ️"),
    ]
)

nav.run()
