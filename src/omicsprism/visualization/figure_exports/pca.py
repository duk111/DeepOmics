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

PLOTLY_MARKER_SYMBOLS = {
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "D": "diamond",
    "P": "cross",
    "X": "x",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "p": "pentagon",
    "H": "hexagon",
    "8": "hexagon2",
    "d": "diamond-tall",
    "*": "star",
}


def export_pca_page(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export unified PCA data for interactive page 'pca'."""
    style = _base_style()

    datasets: dict[str, Any] = {}
    for source, title, pca_result in (
        ("transcriptome", "Transcriptome PCA", context.transcriptome_pca_result),
        ("metabolome", "Metabolome PCA", context.metabolome_pca_result),
    ):
        dataset = _pca_dataset_payload(source, title, pca_result)
        if dataset is not None:
            datasets[source] = dataset

    if not datasets:
        return None

    component_counts = [
        len(dataset.get("var_exp", []))
        for dataset in datasets.values()
    ]
    max_components = min(5, max(component_counts or [2]))
    static_files = {
        "png": f"plots/{FIGURE_FILE_PREFIXES.get('transcriptome_pca', 'transcriptome_pca')}.png",
        "svg": f"plots/{FIGURE_FILE_PREFIXES.get('transcriptome_pca', 'transcriptome_pca')}.svg",
    }

    return {
        "figure_id": "pca",
        "title": "PCA Explorer",
        "chart_type": "scatter",
        "interactive_page_id": "pca",
        "static_files": static_files,
        "plotly_spec": {"datasets": datasets, "config": _base_plotly_config()},
        "default_state": {"source": "transcriptome", "color_by": "group1", "x_pc": 1, "y_pc": 2},
        "available_states": {
            "source": list(datasets.keys()),
            "color_by": ["group1", "group2"],
            "x_pc": list(range(1, max_components + 1)),
            "y_pc": list(range(1, max_components + 1)),
        },
        "style": style,
    }


def _pca_dataset_payload(source: str, title: str, pca_result: dict[str, object] | None) -> dict[str, Any] | None:
    if pca_result is None:
        return None

    coords_full = np.asarray(pca_result["coords"], dtype=float)
    var_exp_full = np.asarray(pca_result["var_exp"], dtype=float)
    n_components = min(5, coords_full.shape[1], var_exp_full.size)
    if n_components < 2:
        return None

    coords = coords_full[:, :n_components]
    var_exp = var_exp_full[:n_components]
    sample_names = [str(s) for s in pca_result["plot_sample_names"]]
    group_df = pca_result.get("plot_group_df")
    groups: list[dict[str, Any]] = []
    group_styles: dict[str, Any] = {"group1": {}, "group2": {}}

    if group_df is not None:
        group_df = group_df.copy().reset_index(drop=True)
        for col in ("group1", "group2"):
            if col in group_df.columns:
                group_df[col] = group_df[col].astype("string").fillna("").astype(str).str.strip()

        group_orders = group_df["_group_table_order"].tolist() if "_group_table_order" in group_df.columns else None

        if "group1" in group_df.columns:
            primary_groups = _ordered_unique_with_order(group_df["group1"].tolist(), group_orders)
            color_map = _group_color_map(group_df["group1"].tolist())
            marker_map = _group_marker_map(primary_groups)
            group_styles["group1"] = {
                "groups": primary_groups,
                "colors": color_map,
                "markers": {k: PLOTLY_MARKER_SYMBOLS.get(v, "circle") for k, v in marker_map.items()},
            }

        if "group2" in group_df.columns:
            subgroup_values = group_df["group2"].astype("string").fillna("").astype(str).str.strip()
            subgroup_values = subgroup_values.where(subgroup_values.ne(""), "Missing")
            secondary_groups, secondary_color_map = _global_secondary_group_color_map(
                subgroup_values.astype(str).tolist(),
                group_orders,
            )
            primary_groups = (
                _ordered_unique_with_order(group_df["group1"].tolist(), group_orders)
                if "group1" in group_df.columns
                else []
            )
            marker_map = _group_marker_map(primary_groups)
            group_styles["group2"] = {
                "groups": secondary_groups,
                "colors": secondary_color_map,
                "markers": {k: PLOTLY_MARKER_SYMBOLS.get(v, "circle") for k, v in marker_map.items()},
                "shape_by": "group1",
            }

        for idx, sample_name in enumerate(sample_names):
            row: dict[str, Any] = {"sample_id": sample_name}
            if idx < len(group_df):
                row["group1"] = str(group_df.at[idx, "group1"]) if "group1" in group_df.columns else ""
                row["group2"] = str(group_df.at[idx, "group2"]) if "group2" in group_df.columns else ""
            groups.append(row)
    else:
        groups = [{"sample_id": sample_name, "group1": "", "group2": ""} for sample_name in sample_names]

    return {
        "source": source,
        "title": title,
        "samples": sample_names,
        "coords": coords.tolist(),
        "var_exp": var_exp.tolist(),
        "groups": groups,
        "group_styles": group_styles,
    }


def export_pca_scatter(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Backward-compatible wrapper for the unified PCA page."""
    return export_pca_page(context, save_dir, prefix_key)


def export_pca_pairs(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Backward-compatible wrapper for the unified PCA page."""
    return export_pca_page(context, save_dir, prefix_key)


# ---------------------------------------------------------------------------
# Dendrogram (F01)
# ---------------------------------------------------------------------------

__all__ = [
    "export_pca_page",
    "export_pca_scatter",
    "export_pca_pairs",
]
