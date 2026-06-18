from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Theme:
    # Backgrounds
    background: str = "#050505"
    surface: str = "#0A0A0A"
    card: str = "#121212"
    border: str = "#252525"

    # Gold accents
    primary_gold: str = "#F5C542"
    secondary_gold: str = "#D4AF37"

    # Text
    text: str = "#FFFFFF"
    text_secondary: str = "#B3B3B3"
    text_muted: str = "#666666"

    # Semantic
    success: str = "#00C853"
    danger: str = "#FF5252"

    # Chart colors for Plotly traces
    chart_colors: tuple[str, ...] = (
        "#F5C542",
        "#D4AF37",
        "#00C853",
        "#FF5252",
        "#B3B3B3",
        "#F5C542",
    )

    def plotly_layout(self) -> dict[str, Any]:
        return {
            "paper_bgcolor": self.background,
            "plot_bgcolor": self.background,
            "font": {"color": self.text},
            "title": {
                "font": {"color": self.text, "size": 14},
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {
                "gridcolor": self.border,
                "zerolinecolor": self.border,
                "title": {"font": {"color": self.text_secondary}},
                "tickfont": {"color": self.text_secondary},
            },
            "yaxis": {
                "gridcolor": self.border,
                "zerolinecolor": self.border,
                "title": {"font": {"color": self.text_secondary}},
                "tickfont": {"color": self.text_secondary},
            },
            "legend": {
                "font": {"color": self.text_secondary},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "hoverlabel": {
                "bgcolor": self.card,
                "font": {"color": self.text},
                "bordercolor": self.border,
            },
            "margin": {"l": 0, "r": 0, "t": 36, "b": 0},
            "colorway": list(self.chart_colors),
        }


THEME = Theme()
