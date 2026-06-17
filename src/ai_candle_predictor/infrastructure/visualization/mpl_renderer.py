from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from io import BytesIO

import matplotlib
import mplfinance as mpf
import pandas as pd

from ai_candle_predictor.application.dto.chart import ChartRequest
from ai_candle_predictor.application.ports.chart_renderer import ChartRenderer
from ai_candle_predictor.common.logging import get_logger
from ai_candle_predictor.domain.entities.candle import CandleStick
from ai_candle_predictor.domain.entities.patterns import CandlePattern, PatternMatch

matplotlib.use("Agg")

log = get_logger(__name__)

_MARKER_MAP = {
    CandlePattern.BULLISH_ENGULFING: ("^", "green", 200, "Bullish Engulfing"),
    CandlePattern.BEARISH_ENGULFING: ("v", "red", 200, "Bearish Engulfing"),
    CandlePattern.DOJI: ("o", "orange", 150, "Doji"),
    CandlePattern.HAMMER: ("s", "blue", 150, "Hammer"),
    CandlePattern.SHOOTING_STAR: ("v", "purple", 150, "Shooting Star"),
}


class MplRenderer(ChartRenderer):
    MAX_CANDLES = 200

    def render_to_bytes(
        self,
        request: ChartRequest,
        candles: Sequence[CandleStick],
        patterns: Sequence[PatternMatch],
    ) -> bytes:
        df = self._build_frame(candles)
        if len(df) > self.MAX_CANDLES:
            df = df.tail(self.MAX_CANDLES)

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=request.config.style,
            title=request.title or f"{request.symbol} Candlestick Chart",
            volume=True,
            figsize=(
                request.config.width / request.config.dpi,
                request.config.height / request.config.dpi,
            ),
            returnfig=True,
        )

        ax_main = axes[0]
        self._annotate_patterns(ax_main, df, patterns)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=request.config.dpi, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    def _annotate_patterns(
        self,
        ax: matplotlib.axes.Axes,
        df: pd.DataFrame,
        patterns: Sequence[PatternMatch],
    ) -> None:
        pattern_by_date: dict[date, PatternMatch] = {}
        for p in patterns:
            d = p.timestamp.date() if hasattr(p.timestamp, "date") else p.timestamp
            pattern_by_date[d] = p

        for idx, (date_val, row) in enumerate(df.iterrows()):
            d = date_val.date() if hasattr(date_val, "date") else date_val
            match = pattern_by_date.get(d)
            if match is None:
                continue

            marker_info = _MARKER_MAP.get(match.pattern)
            if marker_info is None:
                continue

            marker, color, markersize, _ = marker_info
            y_pos = row["High"] * 1.02

            ax.scatter(
                idx,
                y_pos,
                marker=marker,
                color=color,
                s=markersize,
                zorder=5,
                label=marker_info[3] if idx == 0 else "",
            )

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)

    def _build_frame(self, candles: Sequence[CandleStick]) -> pd.DataFrame:
        records = []
        for c in candles:
            records.append(
                {
                    "Date": c.timestamp,
                    "Open": c.open,
                    "High": c.high,
                    "Low": c.low,
                    "Close": c.close,
                    "Volume": c.volume,
                }
            )
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df
