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

def export_violin_box(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export violin-box plot data for interactive page 'violin-box'."""
    style = _base_style()
    engine = context.engine
    group_df = context.pca_group_df

    is_kme = "kme" in prefix_key.lower()
    is_metabolite = "metabolite" in prefix_key.lower() and "group1_violin" in prefix_key.lower() and not is_kme

    group1_order = []
    group1_colors = []
    feature_data = []  # [{feature_name, groups: [{group, values}]}]

    if is_kme:
        # F22: Module kME Boxplot
        assignment = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
        if assignment.empty:
            return None
        plot_df = assignment[["Gene", "Module", "kME"]].copy()
        plot_df["kME"] = pd.to_numeric(plot_df["kME"], errors="coerce")
        plot_df = plot_df[plot_df["Module"].str.lower() != "grey"].dropna(subset=["kME"])
        modules = plot_df["Module"].drop_duplicates().tolist()

        for mod in modules:
            mod_data = plot_df[plot_df["Module"] == mod]
            feature_data.append({
                "feature": str(mod),
                "groups": [{"group": "All", "values": mod_data["kME"].dropna().tolist()}],
            })
        group1_order = ["All"]
        group1_colors = ["#9ca3af"]
        feature_type = "kme"
    else:
        # F14 / F21
        if is_metabolite:
            # F14: Top Metabolite Group1 Violin Box
            metab_df = engine.metabolomics_df()
            if metab_df.empty:
                return None

            from ..static.distribution import _top_metabolite_order, _align_exact_group1_to_samples
            from ..static.base import _group_color_map, _ordered_unique_with_order

            annotation = _align_exact_group1_to_samples(metab_df.index.astype(str).tolist(), group_df)
            if annotation.empty:
                return None

            shared = annotation.index.intersection(metab_df.index)
            metab_df = metab_df.reindex(shared)
            annotation = annotation.reindex(shared)

            group_orders = annotation["_group_table_order"].astype(int).tolist()
            group1_order = _ordered_unique_with_order(annotation["group1"].astype(str).tolist(), group_orders)
            group1_colors = [_group_color_map(group1_order).get(g, "#9ca3af") for g in group1_order]

            metabolites = _top_metabolite_order(engine, metab_df, 12)
            for met in metabolites:
                groups_data = []
                for g_name in group1_order:
                    samples = annotation.index[annotation["group1"].astype(str).eq(g_name)].tolist()
                    values = metab_df.loc[[s for s in samples if s in metab_df.index], met].dropna().tolist()
                    groups_data.append({"group": str(g_name), "values": values})
                feature_data.append({"feature": str(met), "groups": groups_data})
            feature_type = "metabolite"
        else:
            # F21: Module Eigengene Group1 Violin Box
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
            samples = zscore_df.index.astype(str).tolist()
            annotation = _align_group_annotations_to_samples(samples, group_df)
            if annotation.empty:
                return None
            annotation = annotation.reindex(samples)

            group_orders = annotation["_group_table_order"].astype(int).tolist()
            orders_by_col, colors_by_col = _module_group_orders_and_colors(
                group_df,
                annotation["group1"].astype(str).tolist(),
                annotation["group2"].astype(str).tolist(),
                group_orders,
            )
            group1_order = orders_by_col.get("group1", [])
            group1_color_map = colors_by_col.get("group1", {})
            group1_colors = [group1_color_map.get(g, "#9ca3af") for g in group1_order]

            for mod in module_order_list:
                groups_data = []
                for g_name in group1_order:
                    g_samples = annotation.index[annotation["group1"].astype(str).eq(g_name)].tolist()
                    values = zscore_df.loc[[s for s in g_samples if s in zscore_df.index], mod].dropna().tolist()
                    groups_data.append({"group": str(g_name), "values": values})
                feature_data.append({"feature": str(mod), "groups": groups_data})
            feature_type = "module-eigengene"

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{feature_type}_violin_box",
        "title": f"{'kME' if is_kme else 'Metabolite Abundance' if is_metabolite else 'Module Eigengene'} Distribution",
        "chart_type": "violin_box",
        "interactive_page_id": "violin-box",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "features": feature_data,
            "group_order": group1_order,
            "group_colors": group1_colors,
            "config": _base_plotly_config(),
        },
        "default_state": {"feature_type": feature_type, "chart_style": "violin+box+strip"},
        "available_states": {"feature_type": ["metabolite", "module-eigengene", "kme"],
                            "chart_style": ["violin+box+strip", "box", "violin", "strip"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Module Heatmap (F15, F16)
# ---------------------------------------------------------------------------

__all__ = [
    "export_violin_box",
]
