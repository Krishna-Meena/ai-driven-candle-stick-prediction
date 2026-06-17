from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    page_title=settings.dashboard_title,
    page_icon="\U0001f5ee",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .stApp { background: #0f0f23; }
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0; color: #e0e0e0; }
    .sub-header { font-size: 1rem; color: #888; margin-top: 0; }
    .kpi-panel {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; padding: 24px; text-align: center;
        border: 1px solid #2a2a4a; margin-bottom: 24px;
        transition: border-color 0.2s, transform 0.2s;
    }
    .kpi-panel:hover { border-color: #00d4aa; transform: translateY(-2px); }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #00d4aa; line-height: 1.2; }
    .kpi-label {
        font-size: 0.8rem; color: #888; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 6px;
    }
    .kpi-delta { font-size: 0.85rem; margin-top: 4px; }
    .kpi-delta.up { color: #00d4aa; }
    .kpi-delta.down { color: #ff6b6b; }
    .asset-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 16px; border: 1px solid #2a2a4a;
        margin-bottom: 12px; transition: border-color 0.2s;
    }
    .asset-card:hover { border-color: #00d4aa; }
    .asset-symbol { font-size: 1.1rem; font-weight: 600; color: #e0e0e0; }
    .asset-price { font-size: 1.4rem; font-weight: 700; color: #fff; }
    .asset-change { font-size: 0.9rem; }
    .asset-change.pos { color: #00d4aa; }
    .asset-change.neg { color: #ff6b6b; }
    .nav-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 18px; border: 1px solid #2a2a4a;
        cursor: pointer; transition: border-color 0.2s, transform 0.2s; margin-bottom: 12px;
    }
    .nav-card:hover { border-color: #00d4aa; transform: translateY(-2px); }
    .nav-card-title { font-size: 1rem; font-weight: 600; color: #e0e0e0; }
    .nav-card-desc { font-size: 0.78rem; color: #888; margin-top: 2px; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .badge-up { background: rgba(0,212,170,0.15); color: #00d4aa; }
    .badge-down { background: rgba(255,107,107,0.15); color: #ff6b6b; }
    .badge-na { background: rgba(136,136,136,0.15); color: #888; }
    .badge-correct { background: rgba(0,212,170,0.15); color: #00d4aa; }
    .badge-incorrect { background: rgba(255,107,107,0.15); color: #ff6b6b; }
    .gauge-container {
        display: flex; justify-content: center; align-items: center;
        padding: 12px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 16px; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] { background: rgba(0,212,170,0.15); }
    .leaderboard-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; border-bottom: 1px solid #1e1e3a; font-size: 0.85rem;
    }
    .leaderboard-row:last-child { border-bottom: none; }
    .leaderboard-rank { font-weight: 700; color: #888; width: 24px; }
    .leaderboard-name { flex: 1; color: #e0e0e0; }
    .leaderboard-score { font-weight: 600; color: #00d4aa; min-width: 40px; text-align: right; }
    footer { display: none; }
    hr { border-color: #2a2a4a; margin: 24px 0; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 16px; border: 1px solid #2a2a4a;
    }
    div[data-testid="stMetric"] > div:first-child {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.3px;
    }
    div[data-testid="stMetric"] > div:nth-child(2) { font-size: 1.5rem; font-weight: 700; }
    </style>""",
    unsafe_allow_html=True,
)

MODEL_DIR = settings.models_dir
RAW_DIR = settings.data_raw_dir
SYMBOLS = settings.default_symbols
TITLE = settings.dashboard_title

if "symbol" not in st.session_state:
    st.session_state.symbol = SYMBOLS[0]

# ── Cached helpers ──────────────────────────────────────────────────────────


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


@st.cache_data(ttl=60)
def _load_candle_data(symbol: str) -> Any:
    path = RAW_DIR / f"{symbol}.parquet"
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_parquet(path)


# ── Page: Home ───────────────────────────────────────────────────────────────


def page_home() -> None:
    st.markdown(f'<p class="main-header">{TITLE}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Institutional-grade candlestick prediction platform</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    symbols_with_data = _list_symbols_with_data()
    model_count = _count_models()
    feature_total = sum(_load_feature_count(s) for s in symbols_with_data)
    label_total = sum(_load_label_count(s) for s in symbols_with_data)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-panel"><div class="kpi-value">{len(symbols_with_data)}</div>'
            '<div class="kpi-label">Assets Tracked</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-panel"><div class="kpi-value">{feature_total:,}</div>'
            '<div class="kpi-label">Feature Rows</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-panel"><div class="kpi-value">{label_total:,}</div>'
            '<div class="kpi-label">Labeled Samples</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-panel"><div class="kpi-value">{model_count}</div>'
            '<div class="kpi-label">Trained Models</div></div>',
            unsafe_allow_html=True,
        )

    if symbols_with_data:
        st.markdown("### Asset Overview")
        cols = st.columns(len(symbols_with_data))
        for ci, sym in enumerate(symbols_with_data):
            df = _load_candle_data(sym)
            if df is not None and len(df) > 0:
                last_c = float(df["close"].iloc[-1])
                prev_c = float(df["close"].iloc[-2]) if len(df) > 1 else last_c
                chg = ((last_c - prev_c) / prev_c) * 100
                cls = "pos" if chg >= 0 else "neg"
                with cols[ci]:
                    st.markdown(
                        f'<div class="asset-card">'
                        f'<div class="asset-symbol">{sym}</div>'
                        f'<div class="asset-price">${last_c:,.2f}</div>'
                        f'<div class="asset-change {cls}">{"+" if chg >= 0 else ""}{chg:.2f}%</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("### Quick Navigation")
    nav_items = [
        ("\U0001f4ca", "Market Overview", "Interactive OHLCV charts with zoom and range selectors"),
        ("\U0001f916", "Predictions", "Date-range prediction with confidence gauges"),
        ("\U0001f4c8", "Model Comparison", "Radar charts, leaderboards, feature importance"),
        ("\U0001f52c", "Explainability", "SHAP global rankings and local explanations"),
        ("\U0001f3af", "Training Center", "Interactive model training with live logs"),
        ("\U0001f4ca", "Backtesting", "Quant strategy backtest with equity curve and metrics"),
        ("\u2139\ufe0f", "About", "Architecture, pipeline, tech stack, system info"),
    ]
    for i in range(0, len(nav_items), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(nav_items):
                icon, title, desc = nav_items[i + j]
                with cols[j]:
                    st.markdown(
                        f'<div class="nav-card">'
                        f'<div class="nav-card-title">{icon} {title}</div>'
                        f'<div class="nav-card-desc">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )

    st.caption(f"Data directory: {RAW_DIR}")


# ── Page: Market Overview ────────────────────────────────────────────────────


def page_market_overview() -> None:
    st.markdown('<p class="main-header">\U0001f4ca Market Overview</p>', unsafe_allow_html=True)
    st.divider()

    symbol = st.selectbox("Select Asset", SYMBOLS, key="market_symbol")
    df = _load_candle_data(symbol)
    if df is None:
        st.warning(f"No data for {symbol}. Run data ingestion first.")
        return

    import pandas as pd

    df = df.sort_index()
    rows = len(df)
    ts0 = df.index[0]
    ts1 = df.index[-1]
    last_c = float(df["close"].iloc[-1])
    prev_c = float(df["close"].iloc[-2]) if rows > 1 else last_c
    chg_pct = ((last_c - prev_c) / prev_c) * 100
    avg_vol = float(df["volume"].mean())
    hi = float(df["high"].max())
    lo = float(df["low"].min())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Close", f"${last_c:,.2f}", f"{chg_pct:+.2f}%")
    with k2:
        st.metric("High", f"${hi:,.2f}")
    with k3:
        st.metric("Low", f"${lo:,.2f}")
    with k4:
        st.metric("Avg Volume", f"{avg_vol:,.0f}")
    with k5:
        start_str = ts0.strftime("%Y-%m-%d") if hasattr(ts0, "strftime") else str(ts0)
        st.metric("From", start_str)
    with k6:
        end_str = ts1.strftime("%Y-%m-%d") if hasattr(ts1, "strftime") else str(ts1)
        st.metric("To", end_str)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    preset = st.radio(
        "Range",
        ["1M", "3M", "6M", "YTD", "1Y", "All"],
        index=5,
        horizontal=True,
        key="market_range",
    )
    now = ts1
    if preset == "1M":
        mask = df.index >= now - pd.Timedelta(days=30)
    elif preset == "3M":
        mask = df.index >= now - pd.Timedelta(days=90)
    elif preset == "6M":
        mask = df.index >= now - pd.Timedelta(days=180)
    elif preset == "YTD":
        mask = df.index >= pd.Timestamp(year=now.year, month=1, day=1)
    elif preset == "1Y":
        mask = df.index >= now - pd.Timedelta(days=365)
    else:
        mask = slice(None)
    dff = df[mask] if not isinstance(mask, slice) else df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Candlestick(
            x=dff.index,
            open=dff["open"],
            high=dff["high"],
            low=dff["low"],
            close=dff["close"],
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dff.index,
            y=dff["volume"],
            name="Volume",
            marker_color="rgba(0,212,170,0.3)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        xaxis2=dict(rangeslider=dict(visible=False)),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Raw Data Table"):
        st.dataframe(dff.tail(100), width="stretch")


# ── Page: Predictions ────────────────────────────────────────────────────────


def page_predictions() -> None:
    st.markdown('<p class="main-header">\U0001f916 Predictions</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found. Train a model first.")
        return

    symbol = st.selectbox("Symbol", SYMBOLS, key="pred_symbol")
    model_name = st.selectbox("Model", models, key="pred_model")

    df = _load_candle_data(symbol)
    if df is None:
        st.warning("No raw data. Run ingestion first.")
        return

    avail_start = df.index[0]
    avail_end = df.index[-1]

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start", value=avail_start, min_value=avail_start, max_value=avail_end, key="pred_start"
        )
    with c2:
        end_date = st.date_input(
            "End", value=avail_end, min_value=avail_start, max_value=avail_end, key="pred_end"
        )

    if start_date >= end_date:
        st.error("Start must be before end.")
        return

    model_label = model_name.replace(".joblib", "").replace(f"{symbol}_", "", 1)

    from datetime import datetime

    from ai_candle_predictor.application.use_cases.predict import predict_range
    from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    ps = ParquetStore()
    fs = ParquetFeatureStore()
    ls = ParquetLabelStore()
    ms = JoblibStore()
    sym = Symbol(symbol)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    with st.spinner("Running predictions..."):
        result = predict_range(
            symbol=sym,
            model_store=ms,
            feature_store=fs,
            label_store=ls,
            candle_store=ps,
            start_date=start_dt,
            end_date=end_dt,
            model_label=model_label,
        )

    if not result.predictions:
        st.warning("No predictions returned for the selected range.")
        return

    import pandas as pd

    df_display = pd.DataFrame(
        [
            {
                "date": p.timestamp,
                "close": p.close,
                "pred_direction": p.predicted_direction,
                "predicted": ("UP" if p.predicted_direction == 1 else "DOWN"),
                "confidence": p.confidence,
                "actual_direction": p.actual_direction,
                "actual": (
                    "UP"
                    if p.actual_direction == 1
                    else "DOWN" if p.actual_direction == 0 else "N/A"
                ),
                "correct": (
                    "\u2713"
                    if p.is_correct is True
                    else "\u2717" if p.is_correct is False else "N/A"
                ),
                "is_correct": p.is_correct,
            }
            for p in result.predictions
        ]
    )

    labeled_df = df_display.dropna(subset=["actual_direction"])
    labeled = len(labeled_df)
    correct_count = int(labeled_df["is_correct"].sum())
    win_rate = correct_count / labeled * 100 if labeled > 0 else 0.0

    if labeled > 0:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        y_true = labeled_df["actual_direction"].astype(int)
        y_pred_labeled = labeled_df["pred_direction"].astype(int)
        prec = precision_score(y_true, y_pred_labeled, zero_division=0)
        rec = recall_score(y_true, y_pred_labeled, zero_division=0)
        f1 = f1_score(y_true, y_pred_labeled, zero_division=0)
        acc_sk = accuracy_score(y_true, y_pred_labeled)
    else:
        prec = rec = f1 = acc_sk = 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with k2:
        st.metric("Accuracy", f"{acc_sk:.4f}")
    with k3:
        st.metric("Precision", f"{prec:.4f}")
    with k4:
        st.metric("Recall", f"{rec:.4f}")
    with k5:
        st.metric("F1 Score", f"{f1:.4f}")
    with k6:
        st.metric("Predictions", labeled)

    if labeled > 0:
        import plotly.graph_objects as go

        chart_tabs = st.tabs(["Timeline", "Confidence", "Accuracy Over Range", "Distribution"])

        with chart_tabs[0]:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_display["date"],
                    y=df_display["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="#888", width=1),
                )
            )
            up_mask = df_display["predicted"] == "UP"
            down_mask = df_display["predicted"] == "DOWN"
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_display.loc[up_mask, "date"],
                    y=df_display.loc[up_mask, "close"],
                    mode="markers",
                    name="Predicted UP",
                    marker=dict(color="#00d4aa", size=6, symbol="triangle-up"),
                )
            )
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_display.loc[down_mask, "date"],
                    y=df_display.loc[down_mask, "close"],
                    mode="markers",
                    name="Predicted DOWN",
                    marker=dict(color="#ff6b6b", size=6, symbol="triangle-down"),
                )
            )
            fig_timeline.update_layout(
                template="plotly_dark",
                title="Prediction Timeline",
                xaxis_title="Date",
                yaxis_title="Price",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_timeline, width="stretch")

        with chart_tabs[1]:
            fig_conf = go.Figure()
            fig_conf.add_trace(
                go.Scatter(
                    x=df_display["date"],
                    y=df_display["confidence"],
                    mode="lines+markers",
                    name="Confidence",
                    line=dict(color="#00d4aa", width=2),
                    marker=dict(
                        color=df_display["confidence"],
                        colorscale="tealgrn",
                        size=5,
                        showscale=True,
                        colorbar=dict(title="P(UP)"),
                    ),
                )
            )
            fig_conf.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#888",
                annotation_text="Random (0.5)",
            )
            fig_conf.update_layout(
                template="plotly_dark",
                title="Confidence Over Time",
                xaxis_title="Date",
                yaxis_title="P(UP)",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_conf, width="stretch")

        with chart_tabs[2]:
            labeled_sorted = labeled_df.sort_values("date").reset_index(drop=True)
            labeled_sorted["cumulative_accuracy"] = labeled_sorted["is_correct"].expanding().mean()
            window = min(30, len(labeled_sorted))
            labeled_sorted["rolling_accuracy"] = (
                labeled_sorted["is_correct"].rolling(window, min_periods=1).mean()
            )
            fig_acc = go.Figure()
            fig_acc.add_trace(
                go.Scatter(
                    x=labeled_sorted["date"],
                    y=labeled_sorted["cumulative_accuracy"],
                    mode="lines",
                    name="Cumulative",
                    line=dict(color="#00d4aa", width=2),
                )
            )
            fig_acc.add_trace(
                go.Scatter(
                    x=labeled_sorted["date"],
                    y=labeled_sorted["rolling_accuracy"],
                    mode="lines",
                    name=f"Rolling ({window})",
                    line=dict(color="#ffd700", width=2, dash="dot"),
                )
            )
            fig_acc.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#888",
                annotation_text="Random (0.5)",
            )
            fig_acc.update_layout(
                template="plotly_dark",
                title="Accuracy Over Range",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_acc, width="stretch")

        with chart_tabs[3]:
            pos_conf = labeled_df[labeled_df["actual"] == "UP"]["confidence"]
            neg_conf = labeled_df[labeled_df["actual"] == "DOWN"]["confidence"]
            fig_dist = go.Figure()
            if not pos_conf.empty:
                fig_dist.add_trace(
                    go.Histogram(
                        x=pos_conf, name="Up (actual)", opacity=0.7, marker_color="#00d4aa"
                    )
                )
            if not neg_conf.empty:
                fig_dist.add_trace(
                    go.Histogram(
                        x=neg_conf, name="Down (actual)", opacity=0.7, marker_color="#ff6b6b"
                    )
                )
            fig_dist.update_layout(
                barmode="overlay",
                template="plotly_dark",
                title="Confidence Distribution by Actual Class",
                xaxis_title="P(UP)",
                yaxis_title="Count",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_dist, width="stretch")

    st.markdown("### Prediction Table")
    display_cols = ["date", "actual", "predicted", "confidence", "correct"]
    styled = (
        df_display[display_cols]
        .style.format({"confidence": "{:.4f}"})
        .rename(
            columns={
                "date": "Date",
                "actual": "Actual",
                "predicted": "Predicted",
                "confidence": "Confidence",
                "correct": "Correct/Incorrect",
            }
        )
    )
    st.dataframe(styled, width="stretch", height=500)


# ── Page: Model Comparison ───────────────────────────────────────────────────


def page_model_comparison() -> None:
    st.markdown('<p class="main-header">\U0001f4c8 Model Comparison</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found.")
        return

    import pandas as pd

    symbol = st.selectbox("Symbol", SYMBOLS, key="compare_symbol")

    rows_list = [{"Model": m.replace(".joblib", "").replace(f"{symbol}_", "", 1)} for m in models]
    st.markdown("### Available Models")
    st.dataframe(pd.DataFrame(rows_list).style.hide(axis="index"), width="stretch")

    feature_count = _load_feature_count(symbol)
    label_count = _load_label_count(symbol)
    k1, k2 = st.columns(2)
    with k1:
        st.metric("Features Available", f"{feature_count:,}")
    with k2:
        st.metric("Labels Available", f"{label_count:,}")

    if feature_count == 0 or label_count == 0:
        st.warning("Compute features and labels first.")
        return

    if st.button("Run Full Comparison (LR + RF + XGB)", type="primary"):
        from ai_candle_predictor.application.ports.feature_store import FeatureStore
        from ai_candle_predictor.application.ports.label_store import LabelStore
        from ai_candle_predictor.application.ports.model_store import ModelStore
        from ai_candle_predictor.application.use_cases.train_baseline import train_baseline
        from ai_candle_predictor.application.use_cases.train_random_forest import (
            train_random_forest,
        )
        from ai_candle_predictor.application.use_cases.train_xgboost import train_xgboost
        from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore

        fs: FeatureStore = ParquetFeatureStore()
        ls: LabelStore = ParquetLabelStore()
        ms: ModelStore = JoblibStore()
        sym = Symbol(symbol)

        with st.spinner("Training LR baseline..."):
            _, lr_m, _ = train_baseline(
                symbol=sym, feature_store=fs, label_store=ls, model_store=ms
            )
        with st.spinner("Training Random Forest..."):
            _, rf_m, _, rf_imp = train_random_forest(
                symbol=sym, feature_store=fs, label_store=ls, model_store=ms
            )
        with st.spinner("Training XGBoost..."):
            _, xgb_m, _, xgb_imp = train_xgboost(
                symbol=sym, feature_store=fs, label_store=ls, model_store=ms
            )

        st.success("Training complete!")

        metrics = {
            "Accuracy": {"LR": lr_m.accuracy, "RF": rf_m.accuracy, "XGB": xgb_m.accuracy},
            "Precision": {"LR": lr_m.precision, "RF": rf_m.precision, "XGB": xgb_m.precision},
            "Recall": {"LR": lr_m.recall, "RF": rf_m.recall, "XGB": xgb_m.recall},
            "F1 Score": {"LR": lr_m.f1, "RF": rf_m.f1, "XGB": xgb_m.f1},
            "ROC-AUC": {"LR": lr_m.roc_auc, "RF": rf_m.roc_auc, "XGB": xgb_m.roc_auc},
        }

        st.markdown("### Performance Leaderboard")
        summary_rows = []
        for metric_name, vals in metrics.items():
            best_model = max(vals, key=lambda k: vals[k])
            for mk, mv in vals.items():
                summary_rows.append(
                    {
                        "Metric": metric_name,
                        "Model": mk,
                        "Score": round(mv, 4),
                        "Rank": 1 if mk == best_model else 2,
                    }
                )
        leaderboard_df = pd.DataFrame(summary_rows).sort_values(["Metric", "Rank"])
        st.dataframe(
            leaderboard_df.pivot(index="Metric", columns="Model", values="Score")
            .style.format("{:.4f}")
            .highlight_max(color="rgba(0,212,170,0.2)", axis=1),
            width="stretch",
        )

        import plotly.graph_objects as go

        categories = list(metrics.keys())
        fig_radar = go.Figure()
        colors = {"LR": "#00d4aa", "RF": "#ffd700", "XGB": "#ff6b6b"}
        for mk in ("LR", "RF", "XGB"):
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[metrics[c][mk] for c in categories],
                    theta=categories,
                    fill="toself",
                    name=mk,
                    line_color=colors[mk],
                    opacity=0.7,
                )
            )
        fig_radar.update_layout(
            template="plotly_dark",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, width="stretch")

        if rf_imp:
            st.markdown("### Top Features — Random Forest")
            rf_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in rf_imp.items()]).head(
                10
            )
            fig_rf = go.Figure(
                go.Bar(
                    x=rf_df["Importance"],
                    y=rf_df["Feature"],
                    orientation="h",
                    marker_color="#ffd700",
                )
            )
            fig_rf.update_layout(
                template="plotly_dark",
                title="Top 10 Features (RF)",
                xaxis_title="Importance",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_rf, width="stretch")

        if xgb_imp:
            st.markdown("### Top Features — XGBoost")
            xgb_df = pd.DataFrame(
                [{"Feature": k, "Importance": v} for k, v in xgb_imp.items()]
            ).head(10)
            fig_xgb = go.Figure(
                go.Bar(
                    x=xgb_df["Importance"],
                    y=xgb_df["Feature"],
                    orientation="h",
                    marker_color="#ff6b6b",
                )
            )
            fig_xgb.update_layout(
                template="plotly_dark",
                title="Top 10 Features (XGB)",
                xaxis_title="Importance",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_xgb, width="stretch")


# ── Page: Explainability ─────────────────────────────────────────────────────


def page_explainability() -> None:
    st.markdown('<p class="main-header">\U0001f52c Explainability</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found.")
        return

    symbol = st.selectbox("Symbol", SYMBOLS, key="explain_symbol")
    model_name = st.selectbox("Model", models, key="explain_model")

    glbl, lcl = st.tabs(["Global Explanations", "Local Explanations"])

    with glbl:
        _explain_global(symbol, model_name)

    with lcl:
        _explain_local(symbol, model_name)


def _explain_global(symbol: str, model_name: str) -> None:
    import pandas as pd

    if st.button("Run SHAP Analysis", type="primary"):
        from ai_candle_predictor.infrastructure.explainability.shap_analyzer import (
            shap_analysis,
        )
        from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
        from ai_candle_predictor.infrastructure.visualization.image_store import ImageStore

        try:
            ms = JoblibStore()
            model_label = model_name.replace(".joblib", "").replace(f"{symbol}_", "", 1)
            safe = symbol.replace("^", "_").replace(".", "_")
            fname = f"{safe}_{model_label}.joblib"
            pipeline = ms.load(settings.models_dir / fname)

            feat_store = ParquetFeatureStore()
            feats = feat_store.load(Symbol(symbol))
            if not feats:
                st.error("No features available.")
                return

            from ai_candle_predictor.application.use_cases.train_baseline import (
                _pivot_features,
            )

            fdf = _pivot_features(feats).sort_index()
            feature_names = list(fdf.columns)
            X = fdf.values

            img_store = ImageStore()
            with st.spinner("Computing SHAP values..."):
                result = shap_analysis(
                    symbol=symbol,
                    pipeline=pipeline,
                    X=X,
                    feature_names=feature_names,
                    image_storage=img_store,
                )

            st.success(f"SHAP analysis complete ({result.get('samples_analyzed', 0)} samples)")

            st.markdown("### SHAP Summary")
            summary_path = result.get("summary_plot")
            if summary_path:
                st.image(str(summary_path), width="stretch")

            st.markdown("### Feature Importance (bar)")
            bar_path = result.get("bar_plot")
            if bar_path:
                st.image(str(bar_path), width="stretch")

            st.markdown("### Global Feature Ranking")
            ranking = result.get("global_ranking", {})
            if ranking:
                assert isinstance(ranking, dict)
                ranking_df = pd.DataFrame(
                    [{"Feature": k, "Mean |SHAP|": v} for k, v in ranking.items()]
                )
                st.dataframe(
                    ranking_df.style.format({"Mean |SHAP|": "{:.6f}"}),
                    width="stretch",
                )

                import plotly.graph_objects as go

                fig = go.Figure()
                top10 = list(ranking.items())[:10]
                fig.add_trace(
                    go.Bar(
                        x=[v for _, v in top10],
                        y=[k for k, _ in top10],
                        orientation="h",
                        marker_color="#00d4aa",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Top 10 Features by Mean |SHAP|",
                    xaxis_title="Mean |SHAP Value|",
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, width="stretch")

            st.session_state.shap_pipeline = pipeline
            st.session_state.shap_X = X
            st.session_state.shap_feature_names = feature_names
            st.session_state.shap_symbol = symbol

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")


def _explain_local(symbol: str, _model_name: str) -> None:
    import pandas as pd

    pipeline = st.session_state.get("shap_pipeline")
    X = st.session_state.get("shap_X")
    feature_names = st.session_state.get("shap_feature_names")
    shap_symbol = st.session_state.get("shap_symbol")

    if pipeline is None or X is None or feature_names is None:
        st.info("Run SHAP analysis first from the Global Explanations tab.")
        return

    if shap_symbol != symbol:
        st.info("Switch to the symbol used in the last SHAP analysis.")
        return

    from ai_candle_predictor.domain.value_objects.symbol import Symbol
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    ps = ParquetStore()
    sym = Symbol(symbol)
    full_data = list(ps.load(sym))
    if not full_data:
        st.warning("No candle data for date mapping.")
        return

    dates = sorted({c.timestamp for c in full_data})
    if len(dates) > len(X):
        dates = dates[-len(X) :]

    if not dates:
        st.warning("No dates available.")
        return

    available_range = st.selectbox(
        "Select Date",
        dates,
        format_func=lambda d: d.strftime("%Y-%m-%d %H:%M"),
        key="explain_date",
    )

    if available_range is None:
        return

    sample_idx = len(dates) - 1 - dates[::-1].index(available_range)

    if st.button("Explain Prediction", type="primary"):
        from ai_candle_predictor.infrastructure.explainability.shap_analyzer import (
            explain_single_sample,
        )
        from ai_candle_predictor.infrastructure.visualization.image_store import (
            ImageStore,
        )

        img_store = ImageStore()
        with st.spinner("Computing local explanation..."):
            try:
                local = explain_single_sample(
                    symbol=symbol,
                    pipeline=pipeline,
                    X=X,
                    feature_names=feature_names,
                    sample_index=sample_idx,
                    image_storage=img_store,
                )
            except Exception as e:
                st.error(f"Local explanation failed: {e}")
                return

        st.success(f"Explanation for {available_range.strftime('%Y-%m-%d %H:%M')}")

        bv = local["base_value"]
        pred = local["prediction"]
        assert isinstance(bv, (int, float))
        assert isinstance(pred, (int, float))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Value", f"{bv:.6f}")
        with col2:
            st.metric("Prediction (log-odds)", f"{pred:.6f}")

        st.markdown("### Waterfall Plot")
        waterfall = local.get("waterfall_plot")
        if waterfall:
            assert isinstance(waterfall, str)
            st.image(waterfall, width="stretch")

        st.markdown("### Force Plot")
        force_html = local.get("force_html", "")
        if force_html:
            assert isinstance(force_html, str)
            st.components.v1.html(force_html, height=200, scrolling=True)

        st.markdown("### Top 10 Contributing Features")
        top = local.get("top_features", [])
        if top:
            assert isinstance(top, list)
            top_df = pd.DataFrame(top)
            st.dataframe(
                top_df.style.format({"shap_value": "{:+.6f}"}),
                width="stretch",
            )

            import plotly.graph_objects as go

            fig = go.Figure()
            vals = [t["shap_value"] for t in top]
            names = [t["feature"] for t in top]
            colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in vals]
            fig.add_trace(
                go.Bar(
                    x=vals,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig.update_layout(
                template="plotly_dark",
                title="Top 10 SHAP Contributors",
                xaxis_title="SHAP Value",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, width="stretch")


# ── Page: Training Center ──────────────────────────────────────────────────────


def page_training_center() -> None:
    st.markdown('<p class="main-header">\U0001f3af Training Center</p>', unsafe_allow_html=True)
    st.divider()

    symbols_with_data = _list_symbols_with_data()
    if not symbols_with_data:
        st.warning("No data found. Run data ingestion first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Select Asset", symbols_with_data, key="train_symbol")
    with col2:
        model_type = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Random Forest", "XGBoost"],
            key="train_model",
        )

    mt_map = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb"}

    feature_count = _load_feature_count(symbol)
    label_count = _load_label_count(symbol)

    k1, k2 = st.columns(2)
    with k1:
        st.metric("Features Available", f"{feature_count:,}" if feature_count else "0")
    with k2:
        st.metric("Labels Available", f"{label_count:,}" if label_count else "0")

    if feature_count == 0 or label_count == 0:
        st.warning("Compute features and labels first.")
        return

    hp: dict[str, object] = {}
    with st.expander("Hyperparameters"):
        if model_type == "Logistic Regression":
            hp["C"] = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01, key="hp_lr_c")
            hp["max_iter"] = st.number_input(
                "Max iterations", 100, 20000, 5000, 100, key="hp_lr_iter"
            )
        elif model_type == "Random Forest":
            hp["n_estimators"] = st.slider("Trees", 50, 1000, 300, 50, key="hp_rf_n")
            hp["max_depth"] = st.slider("Max depth", 3, 50, 10, 1, key="hp_rf_d")
            hp["min_samples_leaf"] = st.slider("Min samples leaf", 1, 50, 5, 1, key="hp_rf_leaf")
        else:
            hp["n_estimators"] = st.slider("Trees", 50, 1000, 300, 50, key="hp_xgb_n")
            hp["max_depth"] = st.slider("Max depth", 3, 30, 6, 1, key="hp_xgb_d")
            hp["learning_rate"] = st.slider(
                "Learning rate", 0.001, 0.5, 0.05, 0.001, key="hp_xgb_lr"
            )

    if st.button("Train Model", type="primary"):
        st.cache_data.clear()

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.code("", language="text", height=200)

        log_lines: list[str] = []

        def on_progress(pct: int, msg: str) -> None:
            log_lines.append(msg)
            progress_bar.progress(pct)
            status_text.text(msg)
            log_area.code("\n".join(log_lines), language="text")

        from ai_candle_predictor.application.use_cases.train_model import train_model
        from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore

        fs = ParquetFeatureStore()
        ls = ParquetLabelStore()
        ms = JoblibStore()
        sym = Symbol(symbol)

        try:
            path, metrics = train_model(
                symbol=sym,
                model_type=mt_map[model_type],
                feature_store=fs,
                label_store=ls,
                model_store=ms,
                on_progress=on_progress,
                **hp,  # type: ignore[arg-type]
            )

            st.success(f"Training complete! Model saved to **{path.name}**")

            st.markdown("### Performance Metrics")
            mk1, mk2, mk3, mk4, mk5 = st.columns(5)
            with mk1:
                st.metric("Accuracy", f"{metrics.accuracy:.4f}")
            with mk2:
                st.metric("Precision", f"{metrics.precision:.4f}")
            with mk3:
                st.metric("Recall", f"{metrics.recall:.4f}")
            with mk4:
                st.metric("F1 Score", f"{metrics.f1:.4f}")
            with mk5:
                st.metric("ROC-AUC", f"{metrics.roc_auc:.4f}")
            st.caption(f"Validation samples: {metrics.support}")
        except Exception as e:
            st.error(f"Training failed: {e}")


# ── Page: Backtesting ─────────────────────────────────────────────────────────


def page_backtesting() -> None:
    st.markdown('<p class="main-header">\U0001f4ca Backtesting</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found. Train a model first.")
        return

    symbol = st.selectbox("Symbol", SYMBOLS, key="bt_symbol")
    model_name = st.selectbox("Model", models, key="bt_model")

    df = _load_candle_data(symbol)
    if df is None:
        st.warning("No raw data. Run ingestion first.")
        return

    avail_start = df.index[0]
    avail_end = df.index[-1]

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start", value=avail_start, min_value=avail_start, max_value=avail_end, key="bt_start"
        )
    with c2:
        end_date = st.date_input(
            "End", value=avail_end, min_value=avail_start, max_value=avail_end, key="bt_end"
        )

    if start_date >= end_date:
        st.error("Start must be before end.")
        return

    model_label = model_name.replace(".joblib", "").replace(f"{symbol}_", "", 1)

    if st.button("Run Backtest", type="primary"):
        from datetime import datetime

        import pandas as pd

        from ai_candle_predictor.application.use_cases.backtest import run_backtest
        from ai_candle_predictor.application.use_cases.predict import predict_range
        from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore
        from ai_candle_predictor.infrastructure.persistence.parquet_store import (
            ParquetStore,
        )

        ps = ParquetStore()
        fs = ParquetFeatureStore()
        ls = ParquetLabelStore()
        ms = JoblibStore()
        sym = Symbol(symbol)
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        with st.spinner("Running predictions..."):
            result = predict_range(
                symbol=sym,
                model_store=ms,
                feature_store=fs,
                label_store=ls,
                candle_store=ps,
                start_date=start_dt,
                end_date=end_dt,
                model_label=model_label,
            )

        if not result.predictions:
            st.warning("No predictions returned for the selected range.")
            return

        with st.spinner("Running backtest..."):
            bt = run_backtest(result.predictions, initial_capital=10000.0)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Win Rate", f"{bt.win_rate:.1%}")
        with k2:
            st.metric("Total Return", f"{bt.total_return_pct:+.2f}%")
        with k3:
            st.metric("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}")
        with k4:
            st.metric("Max Drawdown", f"{bt.max_drawdown_pct:.1f}%")
        with k5:
            st.metric("Trades", bt.total_trades)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Final Equity", f"${bt.final_equity:,.2f}")
        with c2:
            st.metric("Buy & Hold Return", f"{bt.buy_hold_return_pct:+.2f}%")

        import plotly.graph_objects as go

        chart_tabs = st.tabs(["Portfolio Growth", "Trade Distribution", "Monthly Returns"])

        with chart_tabs[0]:
            eq_df = pd.DataFrame({"date": bt.equity_dates, "equity": bt.equity_values})
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["date"],
                    y=eq_df["equity"],
                    mode="lines",
                    name="Strategy",
                    line=dict(color="#00d4aa", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,170,0.1)",
                )
            )
            bh_equity = [bt.initial_capital * (1 + bt.buy_hold_return_pct / 100)] * len(eq_df)
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["date"],
                    y=bh_equity,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#888", width=1, dash="dot"),
                )
            )
            fig_eq.add_hline(
                y=bt.initial_capital,
                line_dash="dash",
                line_color="#555",
                annotation_text="Initial Capital",
            )
            fig_eq.update_layout(
                template="plotly_dark",
                title="Portfolio Growth",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_eq, width="stretch")

        with chart_tabs[1]:
            if bt.trades:
                trades_df = pd.DataFrame(
                    [
                        {
                            "return_pct": t.return_pct,
                            "won": t.won,
                            "side": t.side,
                        }
                        for t in bt.trades
                    ]
                )
                colors = ["#00d4aa" if w else "#ff6b6b" for w in trades_df["won"]]
                fig_td = go.Figure()
                fig_td.add_trace(
                    go.Bar(
                        x=list(range(len(trades_df))),
                        y=trades_df["return_pct"],
                        marker_color=colors,
                        name="Trade Returns",
                        hovertemplate="Trade %{x}<br>Return: %{y:+.2f}%<extra></extra>",
                    )
                )
                fig_td.add_hline(
                    y=0,
                    line_color="#555",
                    line_width=1,
                )
                fig_td.update_layout(
                    template="plotly_dark",
                    title=f"Trade Distribution ({bt.winning_trades}W / {bt.losing_trades}L)",
                    xaxis_title="Trade #",
                    yaxis_title="Return (%)",
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_td, width="stretch")
            else:
                st.info("No trades were opened.")

        with chart_tabs[2]:
            if bt.monthly_returns:
                months = sorted(bt.monthly_returns.keys())
                m_rets = [bt.monthly_returns[m] for m in months]
                m_colors = ["#00d4aa" if r >= 0 else "#ff6b6b" for r in m_rets]
                fig_mr = go.Figure()
                fig_mr.add_trace(
                    go.Bar(
                        x=months,
                        y=m_rets,
                        marker_color=m_colors,
                        name="Monthly Return",
                        hovertemplate="%{x}<br>Return: %{y:+.2f}%<extra></extra>",
                    )
                )
                fig_mr.add_hline(
                    y=0,
                    line_color="#555",
                    line_width=1,
                )
                fig_mr.update_layout(
                    template="plotly_dark",
                    title="Monthly Returns",
                    xaxis_title="Month",
                    yaxis_title="Return (%)",
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_mr, width="stretch")
            else:
                st.info("No monthly return data.")

        if bt.trades:
            st.markdown("### Trade Log")
            tl_df = pd.DataFrame(
                [
                    {
                        "entry": t.entry_date,
                        "exit": t.exit_date,
                        "side": t.side,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "return_pct": f"{t.return_pct:+.2f}%",
                        "won": "\u2713" if t.won else "\u2717",
                    }
                    for t in bt.trades
                ]
            )
            st.dataframe(tl_df, width="stretch", height=400)


# ── Page: About ───────────────────────────────────────────────────────────────


def page_about() -> None:
    st.markdown('<p class="main-header">\u2139\ufe0f About</p>', unsafe_allow_html=True)
    st.divider()

    tabs = st.tabs(["Architecture", "Pipeline", "Tech Stack", "System Info"])

    with tabs[0]:
        st.markdown("""
        **AI Candle Predictor** is a production-grade platform for financial
        candlestick prediction using machine learning, built with
        **Clean Architecture** principles.

        ```
        src/
        \u251c\u2500\u2500 domain/          Core entities & business rules (zero deps)
        \u251c\u2500\u2500 application/     Ports, DTOs, use cases
        \u251c\u2500\u2500 infrastructure/  Data providers, persistence, models, viz
        \u251c\u2500\u2500 presentation/    Streamlit dashboard, CLI, API stubs
        \u2514\u2500\u2500 common/          Config, logging, exceptions
        ```

        - **Domain** has zero external dependencies
        - **Application** defines interfaces (ports) that infrastructure implements
        - **Presentation** depends only on application layer
        """)

    with tabs[1]:
        st.markdown("""
        ### Pipeline Steps

        1. **Data Ingestion** — yfinance \u2192 Parquet (OHLCV)
        2. **Feature Engineering** — 8 technical indicators via `pandas-ta`
           (SMA, EMA, RSI, MACD, Bollinger, ATR, ADX, Stochastic)
        3. **Label Engineering** — Forward returns \u2192 binary UP/DOWN (horizon=5, threshold=0.5%)
        4. **Model Training** — LogisticRegression, RandomForest, XGBoost
        5. **Hyperparameter Tuning** — Optuna (TPE sampler, 50+ trials)
        6. **Explainability** — SHAP (global rankings + local waterfall plots)
        """)

    with tabs[2]:
        st.markdown("""
        | Category | Tools |
        |---|---|
        | **Language** | Python 3.13 |
        | **Package Manager** | UV |
        | **Data** | pandas, numpy, pyarrow, pandas-ta |
        | **ML** | scikit-learn, XGBoost, Optuna, SHAP |
        | **Visualization** | Plotly, Matplotlib, mplfinance |
        | **Dashboard** | Streamlit (custom dark theme) |
        | **Config** | pydantic-settings |
        | **Logging** | structlog |
        | **Quality** | Ruff, Black, Mypy |
        | **Testing** | Pytest (114 unit + integration + e2e) |
        """)

    with tabs[3]:
        import platform

        k1, k2 = st.columns(2)
        with k1:
            st.metric("Models Directory", str(MODEL_DIR))
            st.metric("Raw Data Directory", str(RAW_DIR))
        with k2:
            st.metric("Python", platform.python_version())
            st.metric("Platform", platform.platform())
            st.metric("Dashboard", settings.dashboard_title)


# ── Navigation ────────────────────────────────────────────────────────────────

nav = st.navigation(
    [
        st.Page(page_home, title="Home", icon="\U0001f3e0", default=True),
        st.Page(page_market_overview, title="Market Overview", icon="\U0001f4ca"),
        st.Page(page_predictions, title="Predictions", icon="\U0001f916"),
        st.Page(page_model_comparison, title="Model Comparison", icon="\U0001f4c8"),
        st.Page(page_explainability, title="Explainability", icon="\U0001f52c"),
        st.Page(page_training_center, title="Training Center", icon="\U0001f3af"),
        st.Page(page_backtesting, title="Backtesting", icon="\U0001f4ca"),
        st.Page(page_about, title="About", icon="\u2139\ufe0f"),
    ]
)

nav.run()
