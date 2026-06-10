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

def export_ridge(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export ridge distribution data for interactive page 'ridge'."""
    style = _base_style()
    engine = context.engine
    group_df = context.pca_group_df

    from ..static.module import (_coerce_module_eigengene_df, _module_order_from_summary,
                                _align_group_annotations_to_samples, _module_group_orders_and_colors, _row_zscore)

    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return None

    module_order_list = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order_list = [m for m in module_order_list if m in eigengenes_df.columns]
    if not module_order_list:
        return None

    zscore_df = _row_zscore(eigengenes_df.loc[:, module_order_list].T).T
    is_grouped = "group1" in prefix_key.lower()

    from scipy.stats import gaussian_kde

    x_grid = np.linspace(float(np.nanmin(zscore_df.values)), float(np.nanmax(zscore_df.values)), 200).tolist()

    ridges = []
    mod_color_map = {}
    # Get module colors
    from ..static.module import _module_color_map_from_results
    mod_color_map = _module_color_map_from_results(engine, module_order_list)

    group1_colors = {}
    group1_order = []
    if is_grouped:
        samples = zscore_df.index.astype(str).tolist()
        annotation = _align_group_annotations_to_samples(samples, group_df)
        if not annotation.empty:
            annotation = annotation.reindex(samples)
            group_orders = annotation["_group_table_order"].astype(int).tolist()
            orders_by_col, colors_by_col = _module_group_orders_and_colors(
                group_df,
                annotation["group1"].astype(str).tolist(),
                annotation["group2"].astype(str).tolist(),
                group_orders,
            )
            group1_order = orders_by_col.get("group1", [])
            group1_colors = colors_by_col.get("group1", {})

    for mod in reversed(module_order_list):
        ridge_data = {"module": str(mod), "color": mod_color_map.get(str(mod), "#9ca3af")}
        if is_grouped and group1_order:
            groups = []
            for g_name in group1_order:
                g_samples = annotation.index[annotation["group1"].astype(str).eq(g_name)].tolist()
                values = zscore_df.loc[[s for s in g_samples if s in zscore_df.index], mod].dropna().values
                if len(values) >= 2:
                    try:
                        density = gaussian_kde(values)(x_grid).tolist()
                    except Exception:
                        density = None
                    rug = values.tolist()
                else:
                    density = None
                    rug = values.tolist() if len(values) > 0 else []
                groups.append({
                    "group": str(g_name),
                    "color": group1_colors.get(str(g_name), "#9ca3af"),
                    "density": density,
                    "rug": rug,
                })
            ridge_data["groups"] = groups
        else:
            values = zscore_df[mod].dropna().values
            try:
                density = gaussian_kde(values)(x_grid).tolist() if len(values) >= 2 else None
            except Exception:
                density = None
            ridge_data["density"] = density
            ridge_data["rug"] = values.tolist()
        ridges.append(ridge_data)

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"module_ridge_{'grouped' if is_grouped else 'plain'}",
        "title": f"Module Eigengene Ridge {'by group1' if is_grouped else 'Distribution'}",
        "chart_type": "ridge",
        "interactive_page_id": "ridge",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "ridge_data": {"ridges": ridges, "x_grid": x_grid, "group1_order": group1_order, "group1_colors": group1_colors},
        "default_state": {"grouped": is_grouped, "bandwidth": 1.0, "fill_alpha": 0.42},
        "available_states": {"grouped": [False, True]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Bar Charts (F27, F28)
# ---------------------------------------------------------------------------

__all__ = [
    "export_ridge",
]
