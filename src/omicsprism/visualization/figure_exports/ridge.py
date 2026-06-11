from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.module import (
    _align_group_annotations_to_samples,
    _centered_point_offsets,
    _density_curve,
    _module_group_orders_and_colors,
    _prepare_module_ridge_matrix,
)
from .common import _base_style


def export_ridge(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export F20 grouped module eigengene ridge data for the interactive page 'ridge'."""
    if "group1" not in prefix_key.lower():
        return None

    engine = context.engine
    group_df = context.pca_group_df
    zscore_df, module_order = _prepare_module_ridge_matrix(engine)
    if zscore_df.empty or not module_order:
        return None

    sample_names = zscore_df.index.astype(str).tolist()
    annotation = _align_group_annotations_to_samples(sample_names, group_df)
    if annotation.empty:
        return None
    annotation = annotation.reindex(sample_names)

    group_orders = annotation["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation["group1"].astype(str).tolist(),
        annotation["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = group_orders_by_col.get("group1", [])
    group1_color_map = color_maps_by_col.get("group1", {})
    if not group1_order:
        return None

    finite_values = zscore_df.to_numpy(dtype=float, copy=False)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 2:
        return None
    x_min = float(np.nanmin(finite_values))
    x_max = float(np.nanmax(finite_values))
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    x_pad = max(0.25, 0.10 * (x_max - x_min))
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 256)

    group_offsets = _centered_point_offsets(len(group1_order), width=0.12)
    ridges: list[dict[str, Any]] = []
    n_modules = len(module_order)
    for row_idx, module_name in enumerate(module_order):
        y_base = float(n_modules - row_idx - 1)
        groups: list[dict[str, Any]] = []
        for group_idx, group1_name in enumerate(group1_order):
            group_samples = annotation.index[
                annotation["group1"].astype(str).eq(str(group1_name))
            ].astype(str).tolist()
            values = (
                pd.to_numeric(zscore_df.loc[group_samples, module_name], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            density = _density_curve(values, x_grid)
            groups.append({
                "group": str(group1_name),
                "color": group1_color_map.get(str(group1_name), "#9ca3af"),
                "density": density.tolist() if density is not None else None,
                "rug": values.tolist(),
                "rug_offset": float(group_offsets[group_idx]),
            })
        ridges.append({
            "module": str(module_name),
            "y_base": y_base,
            "groups": groups,
        })

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": "module_ridge_grouped",
        "title": "Module Eigengene Ridge Distribution by group1",
        "chart_type": "ridge",
        "interactive_page_id": "ridge",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "ridge_data": {
            "ridges": ridges,
            "x_grid": x_grid.tolist(),
            "module_order": module_order,
            "group1_order": group1_order,
            "group1_colors": group1_color_map,
            "ridge_height": 0.72,
            "rug_height": 0.072,
        },
        "default_state": {
            "visible_groups": group1_order,
        },
        "available_states": {
            "visible_groups": group1_order,
        },
        "style": _base_style(),
    }


__all__ = ["export_ridge"]
