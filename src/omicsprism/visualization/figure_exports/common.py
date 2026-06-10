"""Shared helpers for interactive figure data exporters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..static.base import PALETTE

def _rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string."""
    if hex_color.startswith("#") and len(hex_color) == 7:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def _base_style() -> dict[str, Any]:
    return {
        "palette": {
            "grid": PALETTE.get("grid_aux", "#e5e7eb"),
            "text": "#111827",
            "background": "#ffffff",
            "positive": "#dc2626",
            "negative": "#2563eb",
        },
        "font_family": "Arial, sans-serif",
        "font_size": 10,
        "marker_size": 42,
    }


def _base_plotly_config() -> dict[str, Any]:
    return {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "omicsprism_figure",
            "height": 800,
            "width": 1200,
            "scale": 2,
        },
    }


def _base_layout(title: str, style: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"family": style["font_family"], "size": 14, "color": style["palette"]["text"]}},
        "font": {"family": style["font_family"], "size": style["font_size"], "color": style["palette"]["text"]},
        "paper_bgcolor": style["palette"]["background"],
        "plot_bgcolor": style["palette"]["background"],
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
        "legend": {"font": {"size": 10}, "itemsizing": "constant"},
        "xaxis": {
            "gridcolor": style["palette"]["grid"],
            "zerolinecolor": style["palette"]["grid"],
            "linecolor": style["palette"]["text"],
        },
        "yaxis": {
            "gridcolor": style["palette"]["grid"],
            "zerolinecolor": style["palette"]["grid"],
            "linecolor": style["palette"]["text"],
        },
    }


def _json_matrix_values(matrix: pd.DataFrame) -> list[list[float | None]]:
    """Return matrix values with NaN/inf converted to JSON null."""
    values: list[list[float | None]] = []
    for row in matrix.to_numpy(dtype=float):
        values.append([
            float(value) if np.isfinite(value) else None
            for value in row
        ])
    return values


def _seaborn_vlag_colorscale() -> list[list[float | str]]:
    """Sample seaborn's vlag colormap used by the static heatmaps."""
    try:
        import seaborn as sns
        from matplotlib import colors

        cmap = sns.color_palette("vlag", as_cmap=True)
        return [
            [round(pos, 4), colors.to_hex(cmap(pos))]
            for pos in np.linspace(0.0, 1.0, 11)
        ]
    except Exception:
        return [
            [0.0, "#3f7f93"],
            [0.25, "#98b9c2"],
            [0.5, "#faf6f3"],
            [0.75, "#d59a91"],
            [1.0, "#b44a50"],
        ]


# ---------------------------------------------------------------------------
# PCA (F02-F09)
# ---------------------------------------------------------------------------
