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

def export_scatter_panels(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export regression scatter panel data for interactive page 'scatter-panels'."""
    style = _base_style()
    engine = context.engine

    is_module = "module" in prefix_key.lower()

    gene_df = engine.gene_expression_df()
    metab_df = engine.metabolomics_df()
    if gene_df.empty or metab_df.empty:
        return None

    panels = []
    if is_module:
        # F25: Module Top Metabolite Regressions
        from ..static.regression import _module_top_metabolite_regression_rows, _module_annotation_maps
        pairs = _module_top_metabolite_regression_rows(engine)
        eigengenes_df = None
        if "module_eigengenes_df" in engine.ml_results:
            from ..static.module import _coerce_module_eigengene_df
            eigengenes_df = _coerce_module_eigengene_df(engine.ml_results["module_eigengenes_df"])
        _, _, module_to_color = _module_annotation_maps(engine)

        if pairs.empty or eigengenes_df is None or eigengenes_df.empty:
            return None

        shared = eigengenes_df.index.intersection(metab_df.index)
        for _, row in pairs.iterrows():
            mod = str(row["Module"])
            met = str(row["Metabolite"])
            if mod not in eigengenes_df.columns or met not in metab_df.columns:
                continue
            x = eigengenes_df.loc[shared, mod].dropna().values.tolist()
            y = metab_df.loc[shared, met].dropna().values.tolist()
            if len(x) < 2 or len(y) < 2:
                continue
            pd_x, pd_y = min(len(x), len(y)), min(len(x), len(y))
            rho = float(row["SpearmanRho"]) if pd.notna(row.get("SpearmanRho")) else 0.0
            panels.append({
                "title": f"{mod} vs {met}",
                "x": x[:pd_x], "y": y[:pd_x],
                "x_label": f"{mod} eigengene", "y_label": met,
                "color": module_to_color.get(mod, "#9ca3af"),
                "rho": rho,
            })
        panel_type = "module-metabolite"
    else:
        # F13: Top Gene-Metabolite Pairs
        from ..static.regression import _module_annotation_maps
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        if edge_df.empty:
            edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
        if edge_df.empty:
            return None

        top_n = getattr(context.cfg, "top_pairs_plot_n", 8)
        ranked = edge_df.sort_values(["EdgeWeight", "RRARank"], ascending=[False, True]).head(top_n)
        _, gene_to_color, _ = _module_annotation_maps(engine)

        for _, row in ranked.iterrows():
            gene = str(row["Gene"])
            met = str(row["Metabolite"])
            if gene not in gene_df.columns or met not in metab_df.columns:
                continue
            x = gene_df[gene].dropna().values.tolist()
            y = metab_df[met].dropna().values.tolist()
            if len(x) < 2 or len(y) < 2:
                continue
            pd_len = min(len(x), len(y))
            r_val = float(np.corrcoef(x[:pd_len], y[:pd_len])[0, 1]) if pd_len >= 3 else 0.0
            panels.append({
                "title": f"{gene} vs {met}",
                "x": x[:pd_len], "y": y[:pd_len],
                "x_label": gene, "y_label": met,
                "color": gene_to_color.get(gene, "#1f77b4"),
                "rho": r_val if np.isfinite(r_val) else 0.0,
            })
        panel_type = "gene-metabolite"

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{panel_type}_panels",
        "title": f"{'Module-Metabolite' if is_module else 'Gene-Metabolite'} Regression Panels",
        "chart_type": "scatter_panels",
        "interactive_page_id": "scatter-panels",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {"panels": panels, "config": _base_plotly_config()},
        "default_state": {"panel_type": panel_type, "page": 0, "per_page": 8, "show_ci": True},
        "available_states": {"panel_type": ["gene-metabolite", "module-metabolite"],
                            "sort_by": ["rho_desc", "rho_asc", "name"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Violin-Box (F14, F21, F22)
# ---------------------------------------------------------------------------

__all__ = [
    "export_scatter_panels",
]
