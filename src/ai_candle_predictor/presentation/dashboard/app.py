from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

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

P = ParamSpec("P")
R = TypeVar("R")


def _typed_cache_data(*, ttl: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return cast(Callable[[Callable[P, R]], Callable[P, R]], st.cache_data(ttl=ttl))


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


@_typed_cache_data(ttl=60)
def _count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    import pandas as pd

    df = pd.read_parquet(path)
    return len(df)


@_typed_cache_data(ttl=60)
def _load_feature_count(symbol: str) -> int:
    fs = ParquetFeatureStore()
    feats = fs.load(Symbol(symbol))
    return len(feats) if feats else 0


@_typed_cache_data(ttl=60)
def _load_label_count(symbol: str) -> int:
    ls = ParquetLabelStore()
    labels = ls.load(Symbol(symbol))
    return len(labels) if labels else 0


@_typed_cache_data(ttl=60)
def _count_models() -> int:
    if not MODEL_DIR.exists():
        return 0
    return len(list(MODEL_DIR.glob("*.joblib")))


@_typed_cache_data(ttl=60)
def _list_models() -> list[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(str(p.name) for p in MODEL_DIR.glob("*.joblib"))


@_typed_cache_data(ttl=60)
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
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Data Table"):
        st.dataframe(dff.tail(100), use_container_width=True)


# ── Page: Predictions ────────────────────────────────────────────────────────


def page_predictions() -> None:
    st.markdown('<p class="main-header">\U0001f916 Predictions</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found. Train a model first.")
        return

    import pandas as pd

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

    import plotly.graph_objects as go

    labeled = sum(1 for p in result.predictions if p.is_correct is not None)
    correct = sum(1 for p in result.predictions if p.is_correct is True)
    acc = correct / labeled * 100 if labeled > 0 else 0.0
    up_pred = sum(1 for p in result.predictions if p.predicted_direction == 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Candles", result.total_candles)
    with k2:
        st.metric("Labeled", labeled)
    with k3:
        st.metric("Accuracy", f"{acc:.1f}%")
    with k4:
        st.metric("Predicted UP", up_pred)

    if labeled > 0:
        correct_pct = acc / 100.0
        gauge_color = (
            "#00d4aa" if correct_pct >= 0.6 else "#ffd700" if correct_pct >= 0.4 else "#ff6b6b"
        )
        st.markdown(
            f"""<div class="gauge-container">
            <svg width="160" height="100" viewBox="0 0 160 100">
              <path d="M 20 90 A 60 60 0 0 1 140 90" fill="none" stroke="#2a2a4a" stroke-width="12"
                    stroke-linecap="round"/>
              <path d="M 20 90 A 60 60 0 0 1 140 90" fill="none"
                    stroke="{gauge_color}" stroke-width="12"
                    stroke-linecap="round"
                    stroke-dasharray="{188.5 * correct_pct} 188.5"/>
              <text x="80" y="80" text-anchor="middle" font-size="24" font-weight="700"
                    fill="{gauge_color}" font-family="Inter, sans-serif">{acc:.0f}%</text>
            </svg></div>""",
            unsafe_allow_html=True,
        )

        df_display = pd.DataFrame(
            [
                {
                    "timestamp": p.timestamp,
                    "close": p.close,
                    "prediction": ("UP" if p.predicted_direction == 1 else "DOWN"),
                    "confidence": p.confidence,
                    "actual_return": p.actual_return,
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
                }
                for p in result.predictions
            ]
        )

    pos_conf = df_display[df_display["actual"] == "UP"]["confidence"]
    neg_conf = df_display[df_display["actual"] == "DOWN"]["confidence"]

    if not pos_conf.empty or not neg_conf.empty:
        fig_prob = go.Figure()
        if not pos_conf.empty:
            fig_prob.add_trace(
                go.Histogram(x=pos_conf, name="Up (actual)", opacity=0.7, marker_color="#00d4aa")
            )
        if not neg_conf.empty:
            fig_prob.add_trace(
                go.Histogram(x=neg_conf, name="Down (actual)", opacity=0.7, marker_color="#ff6b6b")
            )
        fig_prob.update_layout(
            barmode="overlay",
            template="plotly_dark",
            title="Confidence Distribution by Actual Class",
            xaxis_title="P(UP)",
            yaxis_title="Count",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Predictions by Candle")
    styled = df_display.style.format(
        {"close": "${:,.2f}", "confidence": "{:.4f}", "actual_return": "{:+.6f}"}
    )
    st.dataframe(styled, use_container_width=True, height=500)


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
    st.dataframe(pd.DataFrame(rows_list).style.hide(axis="index"), use_container_width=True)

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
            use_container_width=True,
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
        st.plotly_chart(fig_radar, use_container_width=True)

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
            st.plotly_chart(fig_rf, use_container_width=True)

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
            st.plotly_chart(fig_xgb, use_container_width=True)


# ── Page: Explainability ─────────────────────────────────────────────────────


def page_explainability() -> None:
    st.markdown('<p class="main-header">\U0001f52c Explainability</p>', unsafe_allow_html=True)
    st.divider()

    models = _list_models()
    if not models:
        st.warning("No trained models found.")
        return

    import pandas as pd

    symbol = st.selectbox("Symbol", SYMBOLS, key="explain_symbol")
    model_name = st.selectbox("Model", models, key="explain_model")

    charts_dir = settings.reports_dir / "charts" / symbol
    images = (
        sorted(charts_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        if charts_dir.exists()
        else []
    )

    if images:
        st.markdown("### SHAP Visualizations")
        tabs = st.tabs([p.name for p in images])
        for ti, img in enumerate(images):
            with tabs[ti]:
                st.image(str(img), use_container_width=True)

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

                st.markdown("### Global Feature Ranking")
                ranking = result.get("global_ranking", {})
                if ranking:
                    assert isinstance(ranking, dict)
                    ranking_df = pd.DataFrame(
                        [{"Feature": k, "Mean |SHAP|": v} for k, v in ranking.items()]
                    )
                    st.dataframe(
                        ranking_df.style.format({"Mean |SHAP|": "{:.6f}"}),
                        use_container_width=True,
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
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
    else:
        st.info("No SHAP plots found. Click 'Run SHAP Analysis' to generate them.")


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
        st.Page(page_about, title="About", icon="\u2139\ufe0f"),
    ]
)

nav.run()
