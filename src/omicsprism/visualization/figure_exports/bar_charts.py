from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import (
    _global_secondary_group_color_map,
    _group_color_map,
    _group_marker_map,
    _ordered_unique_nonempty,
    _ordered_unique_with_order,
)
from .common import (
    _base_layout,
    _base_plotly_config,
    _base_style,
    _json_matrix_values,
    _rgba,
    _seaborn_vlag_colorscale,
)

def export_bar_charts(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export bar chart data for interactive page 'bar-charts'."""
    style = _base_style()
    engine = context.engine

    is_edgeweight = "edgeweight" in prefix_key.lower() or "f28" in prefix_key.lower()

    if is_edgeweight:
        # F28
        from ..static.association import _edge_module_dataframe
        plot_df = _edge_module_dataframe(engine)
        if plot_df.empty:
            return None

        module_order = plot_df.attrs.get("module_order", [])
        module_to_color = plot_df.attrs.get("module_to_color", {})

        box_data = []
        for mod in module_order:
            mod_df = plot_df[plot_df["Module"] == mod]
            box_data.append({
                "module": str(mod),
                "color": module_to_color.get(str(mod), "#9ca3af"),
                "positive": mod_df[mod_df["Direction"] == "positive"]["EdgeWeight"].dropna().tolist(),
                "negative": mod_df[mod_df["Direction"] == "negative"]["EdgeWeight"].dropna().tolist(),
            })
        view_type = "edgeweight"
    else:
        # F27: Direction Summary
        from ..static.association import _edge_module_dataframe
        plot_df = _edge_module_dataframe(engine)
        if plot_df.empty:
            return None

        module_order = plot_df.attrs.get("module_order", [])
        counts = plot_df.groupby(["Module", "Direction"]).size().unstack(fill_value=0)
        counts = counts.reindex(index=module_order, fill_value=0)

        bar_data = []
        for mod in counts.index:
            bar_data.append({
                "module": str(mod),
                "positive": int(counts.loc[mod, "positive"]) if "positive" in counts.columns else 0,
                "negative": int(counts.loc[mod, "negative"]) if "negative" in counts.columns else 0,
            })
        box_data = bar_data
        view_type = "direction"

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{view_type}_bar_chart",
        "title": f"{'EdgeWeight Distribution' if is_edgeweight else 'Association Direction Summary'} by Module",
        "chart_type": "bar" if not is_edgeweight else "box",
        "interactive_page_id": "bar-charts",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "bar_data": box_data,
        "default_state": {"view_type": view_type, "sort": "default"},
        "available_states": {"view_type": ["direction", "edgeweight"],
                            "sort": ["default", "total_desc", "positive_desc", "negative_desc"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Circos Network (F29, F30)
# ---------------------------------------------------------------------------

__all__ = [
    "export_bar_charts",
]
