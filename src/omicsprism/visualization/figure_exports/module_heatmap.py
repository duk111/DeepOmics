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

def export_module_heatmap(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export module eigengene heatmap data for interactive page 'module-heatmap'."""
    style = _base_style()
    engine = context.engine
    group_df = context.pca_group_df

    from ..static.module import (_coerce_module_eigengene_df, _module_order_from_summary,
                                _align_group_annotations_to_samples, _module_group_orders_and_colors,
                                _sort_module_eigengene_samples, _row_zscore)

    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return None

    module_order_list = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    eigengenes_df = eigengenes_df[[m for m in module_order_list if m in eigengenes_df.columns]]
    if eigengenes_df.empty:
        return None

    samples = eigengenes_df.index.astype(str).tolist()
    annotation = _align_group_annotations_to_samples(samples, group_df)
    if annotation.empty:
        return None

    group_orders = annotation["_group_table_order"].astype(int).tolist()
    orders_by_col, colors_by_col = _module_group_orders_and_colors(
        group_df,
        annotation["group1"].astype(str).tolist(),
        annotation["group2"].astype(str).tolist(),
        group_orders,
    )

    is_group2 = "group2" in prefix_key.lower()
    block_col = "group2" if is_group2 else "group1"
    top_col = "group1" if is_group2 else "group2"
    bottom_col = "group2" if is_group2 else "group1"

    eigengenes_df, annotation = _sort_module_eigengene_samples(eigengenes_df, annotation, block_group_col=block_col,
                                                                group_orders_by_col=orders_by_col)
    heatmap_df = _row_zscore(eigengenes_df.T)

    heatmap_values = heatmap_df.values.tolist()
    modules = heatmap_df.index.tolist()
    samples_sorted = heatmap_df.columns.tolist()

    sample_annotations = []
    for s in samples_sorted:
        if s in annotation.index:
            sample_annotations.append({
                "sample": s,
                "group1": str(annotation.loc[s, "group1"]),
                "group2": str(annotation.loc[s, "group2"]),
                "group1_color": colors_by_col.get("group1", {}).get(str(annotation.loc[s, "group1"]), "#d1d5db"),
                "group2_color": colors_by_col.get("group2", {}).get(str(annotation.loc[s, "group2"]), "#d1d5db"),
            })

    block_order = orders_by_col.get(block_col, [])
    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)

    return {
        "figure_id": f"module_heatmap_{'group2' if is_group2 else 'group1'}",
        "title": "Module Eigengene Heatmap",
        "chart_type": "heatmap",
        "interactive_page_id": "module-heatmap",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "data": [{"type": "heatmap", "z": heatmap_values, "x": samples_sorted, "y": modules,
                      "colorscale": "RdBu_r", "zmid": 0, "zmin": -1.5, "zmax": 1.5,
                      "colorbar": {"title": "z-score"}}],
            "layout": _base_layout("Module Eigengene Heatmap", style),
            "config": _base_plotly_config(),
            "sample_annotations": sample_annotations,
            "block_col": block_col,
            "block_order": block_order,
            "top_col": top_col,
            "bottom_col": bottom_col,
        },
        "default_state": {"block_by": block_col, "show_values": False, "cluster_rows": False},
        "available_states": {"block_by": ["group1", "group2"],
                            "color_scheme": ["RdBu_r", "vlag", "coolwarm"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Line Panels (F17, F18, F26)
# ---------------------------------------------------------------------------

__all__ = [
    "export_module_heatmap",
]
