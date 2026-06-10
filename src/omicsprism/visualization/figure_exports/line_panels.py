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

def export_line_panels(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export line panel data for interactive page 'line-panels'."""
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
    group1_order = orders_by_col.get("group1", [])
    group2_order = orders_by_col.get("group2", [])
    if not group1_order or not group2_order:
        return None

    is_gene_zscore = "gene_zscore" in prefix_key.lower()
    is_trend = "trend" in prefix_key.lower()

    if is_gene_zscore or not is_trend:
        zscore_df = _row_zscore(eigengenes_df.loc[:, module_order_list].T)
        annotation = annotation.reindex(samples)

        # Build sample groups
        sample_groups = {}
        for g1 in group1_order:
            for g2 in group2_order:
                mask = (annotation["group1"].astype(str).eq(str(g1)) &
                        annotation["group2"].astype(str).eq(str(g2)))
                sample_groups[(str(g1), str(g2))] = annotation.index[mask].astype(str).tolist()

        panels = []
        for mod in module_order_list:
            mod_panels = []
            for g1 in group1_order:
                means = []
                scatters = []
                for g2_idx, g2 in enumerate(group2_order):
                    g_samples = [s for s in sample_groups.get((str(g1), str(g2)), [])
                                if s in zscore_df.columns]
                    if g_samples:
                        y_vals = zscore_df.loc[mod, g_samples].dropna().tolist()
                        scatters.append({"group2_idx": g2_idx, "group2": str(g2), "values": y_vals})
                        means.append({"x": g2_idx, "y": float(np.nanmean(y_vals))} if y_vals else {"x": g2_idx})
                    else:
                        means.append({"x": g2_idx})
                mod_panels.append({"group1": str(g1), "means": means, "scatters": scatters})
            panels.append({"module": str(mod), "panels": mod_panels})
        view_type = "module-zscore"
    else:
        # F26 trend panels
        from ..static.association import _module_top_metabolite_pairs
        metab_df = engine.metabolomics_df()
        pairs = _module_top_metabolite_pairs(engine)
        if pairs.empty or metab_df.empty:
            return None
        annotation = annotation.reindex(samples)

        valid_pairs = []
        for _, row in pairs.iterrows():
            mod = str(row["Module"])
            met = str(row["Metabolite"])
            if mod in eigengenes_df.columns and met in metab_df.columns:
                valid_pairs.append((mod, met))
        if not valid_pairs:
            return None

        module_z = _row_zscore(eigengenes_df.loc[:, [m for m, _ in valid_pairs]].T).T
        metab_z = _row_zscore(metab_df.loc[:, [m for _, m in valid_pairs]].T).T

        panels = []
        for mod, met in valid_pairs:
            panel_pairs = []
            for g1 in group1_order:
                mod_means = []
                met_means = []
                for g2_idx, g2 in enumerate(group2_order):
                    g_samples = annotation.index[
                        annotation["group1"].astype(str).eq(str(g1)) &
                        annotation["group2"].astype(str).eq(str(g2))
                    ].astype(str).tolist()
                    mod_val = module_z.loc[[s for s in g_samples if s in module_z.index], mod].mean()
                    met_val = metab_z.loc[[s for s in g_samples if s in metab_z.index], met].mean()
                    mod_means.append({"x": g2_idx, "y": float(mod_val) if pd.notna(mod_val) else None})
                    met_means.append({"x": g2_idx, "y": float(met_val) if pd.notna(met_val) else None})
                panel_pairs.append({"group1": str(g1), "mod_means": mod_means, "met_means": met_means,
                                    "metabolite": str(met)})
            panels.append({"module": str(mod), "metabolite": str(met), "panels": panel_pairs})
        view_type = "trend"

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{view_type}_line_panels",
        "title": f"{'Gene Z-score' if is_gene_zscore else 'Module-Metabolite Trend' if is_trend else 'Module Z-score'} Line Panels",
        "chart_type": "line_panels",
        "interactive_page_id": "line-panels",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "panels": panels,
            "group1_order": group1_order,
            "group2_order": group2_order,
            "config": _base_plotly_config(),
        },
        "default_state": {"view_type": view_type, "show_scatter": True, "unified_y": False},
        "available_states": {"view_type": ["module-zscore", "gene-zscore", "trend"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Ridge (F19, F20)
# ---------------------------------------------------------------------------

__all__ = [
    "export_line_panels",
]
