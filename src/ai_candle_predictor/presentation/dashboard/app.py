from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_candle_predictor.common.config.settings import settings
from ai_candle_predictor.common.date_utils import ensure_date
from ai_candle_predictor.common.symbol_utils import normalize_symbol
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
SYMBOL_DISPLAY: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "^NSEI": "Nifty 50",
    "RELIANCE.NS": "Reliance",
}


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
    result: list[str] = []
    for sym in SYMBOLS:
        safe = normalize_symbol(sym)
        if (RAW_DIR / f"{safe}.parquet").exists():
            result.append(sym)
    return result


@st.cache_data(ttl=60)
def _asset_model_count(symbol: str) -> int:
    if not MODEL_DIR.exists():
        return 0
    safe = normalize_symbol(symbol)
    return len(list(MODEL_DIR.glob(f"{safe}_*.joblib")))


@st.cache_data(ttl=60)
def _pipeline_status(symbol: str) -> dict[str, int]:
    safe = normalize_symbol(symbol)
    raw = _count_parquet_rows(RAW_DIR / f"{safe}.parquet")
    feats = _load_feature_count(symbol)
    labels = _load_label_count(symbol)
    models = _asset_model_count(symbol)
    return {"raw": raw, "features": feats, "labels": labels, "models": models}


@st.cache_data(ttl=60)
def _load_candle_data(symbol: str) -> Any:
    safe = normalize_symbol(symbol)
    path = RAW_DIR / f"{safe}.parquet"
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_parquet(path)


# ── Cached helpers (executive dashboard) ──────────────────────────────


@st.cache_data(ttl=120)
def _load_registry_entries() -> list[Any]:
    """Return RegistryEntry objects (cast to Any for Streamlit caching)."""
    from ai_candle_predictor.infrastructure.models.model_registry import ModelRegistry

    return ModelRegistry().list_models()


@st.cache_data(ttl=120)
def _best_accuracy() -> float:
    entries = _load_registry_entries()
    if not entries:
        return 0.0
    return float(max(e.accuracy for e in entries))


@st.cache_data(ttl=120)
def _pipeline_coverage() -> float:
    """Percentage of default symbols with data in all pipeline stages."""
    if not SYMBOLS:
        return 0.0
    complete = 0
    for sym in SYMBOLS:
        status = _pipeline_status(sym)
        if status["raw"] > 0 and status["features"] > 0 and status["labels"] > 0:
            complete += 1
    return complete / len(SYMBOLS) * 100


@st.cache_data(ttl=120)
def _latest_prediction_summary() -> dict[str, Any]:
    """Attempt a live prediction using the most recently registered model."""
    entries = _load_registry_entries()
    if not entries:
        return {"status": "no model", "direction": "N/A", "confidence": 0.0}
    latest = entries[-1]
    sym_str = latest.symbol
    safe = normalize_symbol(sym_str)
    fname = f"{safe}_{latest.label}.joblib"
    model_path = settings.models_dir / fname
    if not model_path.exists():
        return {"status": "file missing", "direction": "N/A", "confidence": 0.0, "symbol": sym_str}

    try:
        from ai_candle_predictor.application.use_cases.train_baseline import (
            _pivot_features,
        )
        from ai_candle_predictor.infrastructure.models.joblib_store import JoblibStore

        ms = JoblibStore()
        pipeline = ms.load(model_path)

        fs = ParquetFeatureStore()
        feats = fs.load(Symbol(sym_str))
        if not feats:
            return {
                "status": "no features",
                "direction": "N/A",
                "confidence": 0.0,
                "symbol": sym_str,
            }

        fdf = _pivot_features(feats).sort_index()
        if fdf.empty:
            return {
                "status": "empty features",
                "direction": "N/A",
                "confidence": 0.0,
                "symbol": sym_str,
            }

        X_latest = fdf.iloc[[-1]].values
        y_prob = pipeline.predict_proba(X_latest)[0, 1]
        direction = "UP" if y_prob >= 0.5 else "DOWN"
        return {
            "status": "ok",
            "direction": direction,
            "confidence": float(y_prob),
            "symbol": sym_str,
            "date": str(fdf.index[-1]),
        }
    except Exception:
        return {"status": "error", "direction": "N/A", "confidence": 0.0, "symbol": sym_str}


@st.cache_data(ttl=120)
def _global_top_features(limit: int = 10) -> list[dict[str, Any]]:
    """Aggregate top feature names from the feature store of the first symbol with data."""
    for sym in SYMBOLS:
        feats = _load_feature_count(sym)
        if feats > 0:
            try:
                from ai_candle_predictor.application.use_cases.train_baseline import (
                    _pivot_features,
                )

                fs = ParquetFeatureStore()
                fdf = _pivot_features(fs.load(Symbol(sym))).sort_index()
                if not fdf.empty:
                    names = list(fdf.columns)[:limit]
                    return [
                        {"rank": i + 1, "feature": n, "symbol": sym} for i, n in enumerate(names)
                    ]
            except Exception:
                pass
        break
    return []


# ── Page: Home (Executive Dashboard) ────────────────────────────────


def _exec_kpi(value: str, label: str, delta: str | None = None) -> None:
    cls = ' class="up"' if delta and not delta.startswith("-") else ""
    delta_html = f'<div class="kpi-delta{cls}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-panel">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"{delta_html}</div>",
        unsafe_allow_html=True,
    )


def _asset_mini_card(symbol: str) -> None:
    df = _load_candle_data(symbol)
    if df is None or len(df) < 2:
        st.markdown(
            f'<div class="asset-card"><div class="asset-symbol">'
            f"{SYMBOL_DISPLAY.get(symbol, symbol)}</div>"
            f'<div style="color:#888;font-size:0.8rem;">No data</div></div>',
            unsafe_allow_html=True,
        )
        return
    last_c = float(df["close"].iloc[-1])
    prev_c = float(df["close"].iloc[-2])
    chg = ((last_c - prev_c) / prev_c) * 100
    cls = "pos" if chg >= 0 else "neg"
    chg_sign = "+" if chg >= 0 else ""
    status = _pipeline_status(symbol)
    raw_ok = status["raw"] > 0
    feat_ok = status["features"] > 0
    lbl_ok = status["labels"] > 0
    pct_change = f"{chg_sign}{chg:.2f}%"
    st.markdown(
        f'<div class="asset-card" style="padding:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span class="asset-symbol" style="font-size:0.95rem;">'
        f"{SYMBOL_DISPLAY.get(symbol, symbol)}</span>"
        f'<span class="asset-change {cls}" style="font-size:0.8rem;">{pct_change}</span>'
        f"</div>"
        f'<div class="asset-price" style="font-size:1.2rem;">${last_c:,.2f}</div>'
        f'<div style="font-size:0.7rem;color:#888;margin-top:6px;">'
        f'<span>{"🟢" if raw_ok else "⚪"} {status["raw"]:,}r</span> '
        f'<span>{"🟢" if feat_ok else "⚪"} {status["features"]:,}f</span> '
        f'<span>{"🟢" if lbl_ok else "⚪"} {status["labels"]:,}l</span> '
        f'<span>🧠 {status["models"]}m</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def page_home() -> None:
    st.markdown(f'<p class="main-header">{TITLE}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        "Executive Dashboard \u2014 Institutional-grade candlestick prediction platform"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    symbols_with_data = _list_symbols_with_data()
    registry = _load_registry_entries()
    model_count = _count_models()
    best_acc = _best_accuracy()
    coverage = _pipeline_coverage()
    pred = _latest_prediction_summary()

    # ── Top KPI Row ────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _exec_kpi(str(len(symbols_with_data)), "Assets Tracked", f"/ {len(SYMBOLS)} total")
    with k2:
        _exec_kpi(str(model_count), "Models Trained")
    with k3:
        _exec_kpi(f"{best_acc:.4f}" if best_acc > 0 else "—", "Best Accuracy")
    with k4:
        if pred["status"] == "ok":
            sym = pred.get("symbol", "")
            _exec_kpi(pred["direction"], f"Latest: {sym}", f"{pred['confidence']:.1%}")
        else:
            _exec_kpi("—", "Latest Prediction")
    with k5:
        _exec_kpi(f"{coverage:.0f}%", "Pipeline Coverage")

    # ── Quick action buttons ──────────────────────────────────────
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        if st.button("📥 Run Pipeline", use_container_width=True):
            st.switch_page(st.Page(page_data_pipeline, title="Data Pipeline"))
    with ac2:
        if st.button("🎯 Train Model", use_container_width=True):
            st.switch_page(st.Page(page_training_center, title="Training Center"))
    with ac3:
        if st.button("📊 Compare Models", use_container_width=True):
            st.switch_page(st.Page(page_model_comparison, title="Model Comparison"))
    with ac4:
        if st.button("🔍 Explain", use_container_width=True):
            st.switch_page(st.Page(page_explainability, title="Explainability"))

    st.divider()

    # ── Two-column: Market Snapshot + Model Leaderboard ──────────────
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("##### Market Snapshot")
        if symbols_with_data:
            cols = st.columns(2)
            for i, sym in enumerate(symbols_with_data):
                with cols[i % 2]:
                    _asset_mini_card(sym)
        else:
            st.info("No data ingested yet. Visit **Data Pipeline** to get started.")

    with col_right:
        st.markdown("##### Model Leaderboard")
        if registry:
            sorted_reg = sorted(registry, key=lambda r: r.roc_auc, reverse=True)[:6]
            import pandas as pd

            def _highlight_top(row: pd.Series) -> list[str]:
                hl = "background: rgba(0,212,170,0.08)"
                return [hl if row["Rank"] == 1 else "" for _ in row]

            lb_df = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Symbol": r.symbol,
                        "Type": r.model_type,
                        "Acc": f"{r.accuracy:.4f}",
                        "AUC": f"{r.roc_auc:.4f}",
                        "F1": f"{r.f1:.4f}",
                    }
                    for i, r in enumerate(sorted_reg)
                ]
            )
            st.dataframe(
                lb_df.style.hide(axis="index").apply(_highlight_top, axis=1),
                width="stretch",
                height=220,
            )
        else:
            st.info("No trained models yet.")

    st.divider()

    # ── Two-column: Latest Model Performance + Recent Activity ────────
    bot_left, bot_right = st.columns([1.4, 1])

    with bot_left:
        st.markdown("##### Latest Model Performance")
        if registry:
            latest = registry[-1]
            import plotly.graph_objects as go

            metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            metrics_vals = [
                latest.accuracy,
                latest.precision,
                latest.recall,
                latest.f1,
                latest.roc_auc,
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_vals,
                    marker_color=["#00d4aa", "#ffd700", "#ff6b6b", "#4fc3f7", "#ab47bc"],
                    text=[f"{v:.4f}" for v in metrics_vals],
                    textposition="outside",
                )
            )
            fig_title = (
                f"{latest.symbol} \u2013 " f"{latest.model_type} ({latest.training_date[:10]})"
            )
            fig.update_layout(
                template="plotly_dark",
                title=fig_title,
                yaxis=dict(range=[0, 1], title="Score"),
                margin=dict(l=0, r=0, t=36, b=0),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

            st.caption(
                f"Support: {latest.support:,} · "
                f"Label: {latest.label} · "
                f"File: {latest.filename}"
            )
        else:
            st.info("No training runs available.")

    with bot_right:
        st.markdown("##### Recent Activity Feed")
        if registry:
            recent = registry[-6:][::-1]
            for entry in recent:
                date_str = entry.training_date[:19].replace("T", " ")
                st.markdown(
                    f'<div style="padding:6px 8px;border-left:3px solid #00d4aa;'
                    f'margin-bottom:6px;font-size:0.8rem;">'
                    f'<span style="color:#00d4aa;font-weight:600;">'
                    f"{entry.model_type}</span> "
                    f'<span style="color:#e0e0e0;">on {entry.symbol}</span><br>'
                    f'<span style="color:#888;">AUC {entry.roc_auc:.4f} · '
                    f"Acc {entry.accuracy:.4f} · {date_str}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No activity yet.")

    st.divider()

    # ── Bottom row: Top Features + Latest Training Runs ──────────────
    b1, b2 = st.columns([1, 1.4])

    with b1:
        st.markdown("##### Top Features")
        top_feats = _global_top_features(8)
        if top_feats:
            import pandas as pd

            ft_df = pd.DataFrame(top_feats)
            st.dataframe(
                ft_df.style.hide(axis="index"),
                width="stretch",
                height=220,
            )
        else:
            st.info("Compute features to see the top indicators used by models.")

    with b2:
        st.markdown("##### Latest Training Runs")
        if registry:
            recent_reg = registry[-5:][::-1]
            import pandas as pd

            runs_df = pd.DataFrame(
                [
                    {
                        "Date": r.training_date[:10],
                        "Symbol": r.symbol,
                        "Type": r.model_type,
                        "Accuracy": f"{r.accuracy:.4f}",
                        "ROC-AUC": f"{r.roc_auc:.4f}",
                        "Support": r.support,
                    }
                    for r in recent_reg
                ]
            )
            st.dataframe(
                runs_df.style.hide(axis="index"),
                width="stretch",
                height=220,
            )
        else:
            st.info("Train models to populate the training history.")

    st.caption(f"Data: {RAW_DIR} · Registry: {settings.models_dir / 'registry.json'}")


# ── Page: Data Pipeline ────────────────────────────────────────────────────────


def _run_ingest(symbol: str) -> str | None:
    from datetime import datetime

    from ai_candle_predictor.application.dto.market_data import MarketDataRequest
    from ai_candle_predictor.application.use_cases.ingest_market_data import (
        ingest_market_data,
    )
    from ai_candle_predictor.infrastructure.data.yahoo_provider import YahooProvider
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    start = datetime.strptime(settings.default_start_date, "%Y-%m-%d")
    request = MarketDataRequest(symbol=symbol, start_date=start, end_date=datetime.now())
    provider = YahooProvider()
    storage = ParquetStore()
    result = ingest_market_data(request, provider, storage)
    return f"Ingested {result.rows_valid:,} rows for {symbol}"


def _run_features(symbol: str) -> str | None:
    from ai_candle_predictor.application.use_cases.compute_features import compute_features
    from ai_candle_predictor.infrastructure.features.parquet_feature_store import (
        ParquetFeatureStore,
    )
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    fs = ParquetFeatureStore()
    storage = ParquetStore()
    count = compute_features(Symbol(symbol), storage, fs)
    return f"Computed {count:,} feature rows for {symbol}"


def _run_labels(symbol: str) -> str | None:
    from ai_candle_predictor.application.use_cases.generate_labels import (
        generate_labels_for_symbol,
    )
    from ai_candle_predictor.infrastructure.labeling.parquet_label_store import (
        ParquetLabelStore,
    )
    from ai_candle_predictor.infrastructure.persistence.parquet_store import ParquetStore

    ls = ParquetLabelStore()
    storage = ParquetStore()
    stats = generate_labels_for_symbol(Symbol(symbol), storage, ls)
    return (
        f"Generated {stats['total']:,} labels "
        f"({stats['up']} UP / {stats['down']} DOWN) for {symbol}"
    )


def _asset_pipeline_card(symbol: str) -> None:
    display = SYMBOL_DISPLAY.get(symbol, symbol)
    status = _pipeline_status(symbol)

    st.markdown(
        f'<div class="asset-card" style="padding:20px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div><span class="asset-symbol">{display}</span>'
        f'<span style="font-size:0.75rem;color:#888;margin-left:8px;">{symbol}</span></div>'
        f"</div>"
        f'<div style="display:flex;gap:24px;margin:12px 0;font-size:0.85rem;">'
        f'<div>{"✅" if status["raw"] > 0 else "⬜"} Raw: {status["raw"]:,}</div>'
        f'<div>{"✅" if status["features"] > 0 else "⬜"} Features: {status["features"]:,}</div>'
        f'<div>{"✅" if status["labels"] > 0 else "⬜"} Labels: {status["labels"]:,}</div>'
        f"<div>Models: {status['models']}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ingest_btn = st.button("📥 Ingest Data", key=f"pipe_ingest_{symbol}")
    with c2:
        feat_btn = st.button("⚙️ Compute Features", key=f"pipe_feat_{symbol}")
    with c3:
        label_btn = st.button("🏷️ Generate Labels", key=f"pipe_label_{symbol}")

    msg_placeholder = st.empty()
    for btn, label, fn in [
        (ingest_btn, "Ingesting", _run_ingest),
        (feat_btn, "Computing features", _run_features),
        (label_btn, "Generating labels", _run_labels),
    ]:
        if btn:
            with st.spinner(f"{label} for {display}..."):
                try:
                    st.cache_data.clear()
                    result = fn(symbol)
                    msg_placeholder.success(result)
                except Exception as e:
                    msg_placeholder.error(f"Failed: {e}")


def page_data_pipeline() -> None:
    st.markdown('<p class="main-header">\U0001f4e6 Data Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ingest, compute features, and generate labels for all assets</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    top1, top2, top3 = st.columns(3)
    with top1:
        batch_ingest = st.button("📥 Ingest All Assets", type="primary", use_container_width=True)
    with top2:
        batch_feat = st.button("⚙️ Features All Assets", type="primary", use_container_width=True)
    with top3:
        batch_label = st.button("🏷️ Labels All Assets", type="primary", use_container_width=True)

    batch_placeholder = st.empty()
    if batch_ingest or batch_feat or batch_label:
        action = _run_ingest if batch_ingest else (_run_features if batch_feat else _run_labels)
        action_label = (
            "Ingesting"
            if batch_ingest
            else ("Computing features" if batch_feat else "Generating labels")
        )
        for sym in SYMBOLS:
            display = SYMBOL_DISPLAY.get(sym, sym)
            with st.spinner(f"{action_label} {display}..."):
                try:
                    st.cache_data.clear()
                    result = action(sym)
                    batch_placeholder.info(result)
                except Exception as e:
                    batch_placeholder.error(f"{display}: {e}")
                    break
        st.success("Batch operation complete!")
        st.rerun()

    stepper = st.checkbox("Expand per-asset controls", value=False)
    if stepper:
        for sym in SYMBOLS:
            _asset_pipeline_card(sym)


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

    if "timestamp" not in df.columns or df.empty:
        st.warning("Dataset has no timestamp column or is empty.")
        return

    rows = len(df)
    ts0 = pd.to_datetime(df["timestamp"].iloc[0])
    ts1 = pd.to_datetime(df["timestamp"].iloc[-1])
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
    ts_col = pd.to_datetime(df["timestamp"])
    if preset == "1M":
        mask = ts_col >= now - pd.Timedelta(days=30)
    elif preset == "3M":
        mask = ts_col >= now - pd.Timedelta(days=90)
    elif preset == "6M":
        mask = ts_col >= now - pd.Timedelta(days=180)
    elif preset == "YTD":
        mask = ts_col >= pd.Timestamp(year=now.year, month=1, day=1)
    elif preset == "1Y":
        mask = ts_col >= now - pd.Timedelta(days=365)
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
    if df is None or df.empty or "timestamp" not in df.columns:
        st.warning("No historical data available for selected asset.")
        return

    avail_start = ensure_date(df["timestamp"].iloc[0]) or date.today()
    avail_end = ensure_date(df["timestamp"].iloc[-1]) or date.today()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start",
            value=avail_start,
            min_value=avail_start,
            max_value=avail_end,
            key="pred_start",
        )
    with c2:
        end_date = st.date_input(
            "End",
            value=avail_end,
            min_value=avail_start,
            max_value=avail_end,
            key="pred_end",
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
        .rename(
            columns={
                "date": "Date",
                "actual": "Actual",
                "predicted": "Predicted",
                "confidence": "Confidence",
                "correct": "Correct/Incorrect",
            }
        )
        .style.format({"Confidence": "{:.4f}"})
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
    if df is None or df.empty or "timestamp" not in df.columns:
        st.warning("No historical data available for selected asset.")
        return

    avail_start = ensure_date(df["timestamp"].iloc[0]) or date.today()
    avail_end = ensure_date(df["timestamp"].iloc[-1]) or date.today()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start",
            value=avail_start,
            min_value=avail_start,
            max_value=avail_end,
            key="bt_start",
        )
    with c2:
        end_date = st.date_input(
            "End",
            value=avail_end,
            min_value=avail_start,
            max_value=avail_end,
            key="bt_end",
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
        st.Page(page_data_pipeline, title="Data Pipeline", icon="\U0001f4e6"),
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
