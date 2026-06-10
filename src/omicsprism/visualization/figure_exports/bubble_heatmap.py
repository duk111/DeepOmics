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

def export_bubble_heatmap(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export bubble heatmap data for interactive page 'bubble-heatmap'."""
    style = _base_style()
    engine = context.engine

    is_gene_level = "gene" in prefix_key.lower() and "module" not in prefix_key.lower()

    if is_gene_level:
        # F11: Gene-Metabolite Bubble Heatmap
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        if edge_df.empty:
            return None

        from ..static.association import _module_maps
        gene_to_module, gene_to_color, _, module_order = _module_maps(engine)

        plot_df = edge_df[["Gene", "Metabolite", "SpearmanRho", "EdgeWeight"]].copy()
        plot_df["Gene"] = plot_df["Gene"].astype(str).str.strip()
        plot_df["Metabolite"] = plot_df["Metabolite"].astype(str).str.strip()
        plot_df["SpearmanRho"] = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
        plot_df["EdgeWeight"] = pd.to_numeric(plot_df["EdgeWeight"], errors="coerce")
        plot_df = plot_df.loc[
            plot_df["Gene"].ne("")
            & plot_df["Metabolite"].ne("")
            & plot_df["SpearmanRho"].notna()
            & plot_df["EdgeWeight"].notna()
        ].copy()
        if plot_df.empty:
            return None

        plot_df["Module"] = plot_df["Gene"].map(gene_to_module).fillna("Unassigned")
        module_rank = {module: idx for idx, module in enumerate(module_order)}
        top_gene_df = (
            plot_df.groupby(["Gene", "Module"], sort=False)
            .agg(_BestEdgeWeight=("EdgeWeight", "max"), _EdgeCount=("Metabolite", "nunique"))
            .reset_index()
            .sort_values(["_BestEdgeWeight", "_EdgeCount", "Gene"], ascending=[False, False, True], kind="mergesort")
            .head(100)
            .assign(_ModuleRank=lambda df: df["Module"].map(module_rank).fillna(len(module_rank)).astype(int))
            .sort_values(
                ["_ModuleRank", "Module", "_BestEdgeWeight", "_EdgeCount", "Gene"],
                ascending=[True, True, False, False, True],
                kind="mergesort",
            )
        )
        gene_order = top_gene_df["Gene"].astype(str).tolist()
        plot_df = plot_df.loc[plot_df["Gene"].isin(gene_order)].copy()
        metabolite_order = (
            plot_df.groupby("Metabolite", sort=False)
            .agg(_BestEdgeWeight=("EdgeWeight", "max"), _EdgeCount=("Gene", "nunique"))
            .sort_values(["_EdgeCount", "_BestEdgeWeight"], ascending=[False, False])
            .index.astype(str)
            .tolist()
        )
        if not gene_order or not metabolite_order:
            return None

        gene_pos = {gene: idx for idx, gene in enumerate(gene_order)}
        metabolite_pos = {metabolite: idx for idx, metabolite in enumerate(metabolite_order)}
        plot_df["_y"] = plot_df["Gene"].map(gene_pos)
        plot_df["_x"] = plot_df["Metabolite"].map(metabolite_pos)
        plot_df = plot_df.sort_values(["_y", "_x"], ascending=[True, True], kind="mergesort")

        rows = []
        for _, row in plot_df.iterrows():
            gene = str(row["Gene"])
            rows.append({
                "gene": gene,
                "metabolite": str(row["Metabolite"]),
                "spearman_rho": float(row["SpearmanRho"]),
                "edge_weight": float(row["EdgeWeight"]),
                "module": str(row["Module"]),
                "module_color": gene_to_color.get(gene, "#d1d5db"),
            })

        level = "gene"
        y_label = "Gene"
        y_modules = [gene_to_module.get(gene, "Unassigned") for gene in gene_order]
        y_colors = [gene_to_color.get(gene, "#d1d5db") for gene in gene_order]
    else:
        # F24: Module-Metabolite Bubble Plot
        assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
        if assoc_df.empty:
            return None

        from ..static.association import _module_maps, _ordered_modules
        _, _, module_to_color, module_order = _module_maps(engine)

        plot_df = assoc_df[["Module", "Metabolite", "SpearmanRho"]].copy()
        if "FDR" in assoc_df.columns:
            plot_df["FDR"] = pd.to_numeric(assoc_df["FDR"], errors="coerce")
        plot_df["Module"] = plot_df["Module"].astype(str).str.strip()
        plot_df["Metabolite"] = plot_df["Metabolite"].astype(str).str.strip()
        plot_df["SpearmanRho"] = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
        plot_df = plot_df.loc[
            plot_df["Module"].ne("")
            & plot_df["Metabolite"].ne("")
            & plot_df["SpearmanRho"].notna()
            & (plot_df["Module"].str.lower() != "grey")
        ].copy()
        if plot_df.empty:
            return None

        gene_order = _ordered_modules(module_order, plot_df["Module"].astype(str).tolist())
        metabolite_order = (
            plot_df.assign(_AbsRho=plot_df["SpearmanRho"].abs())
            .groupby("Metabolite", sort=False)["_AbsRho"]
            .max()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        if not gene_order or not metabolite_order:
            return None

        module_pos = {module_name: idx for idx, module_name in enumerate(gene_order)}
        metabolite_pos = {metabolite: idx for idx, metabolite in enumerate(metabolite_order)}
        plot_df["_y"] = plot_df["Module"].map(module_pos)
        plot_df["_x"] = plot_df["Metabolite"].map(metabolite_pos)
        plot_df = plot_df.sort_values(["_y", "_x"], ascending=[True, True], kind="mergesort")

        fdr_values = pd.to_numeric(plot_df.get("FDR", pd.Series(np.nan, index=plot_df.index)), errors="coerce")
        neglog_fdr = -np.log10(fdr_values.clip(lower=1e-300))
        neglog_fdr = neglog_fdr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        use_fdr_size = float(neglog_fdr.max()) > 0

        rows = []
        for row_idx, row in plot_df.iterrows():
            module_name = str(row["Module"])
            fdr = float(row.get("FDR", np.nan)) if "FDR" in row.index else np.nan
            edge_weight = float(neglog_fdr.loc[row_idx]) if use_fdr_size else abs(float(row["SpearmanRho"]))
            rows.append({
                "gene": module_name,
                "metabolite": str(row["Metabolite"]),
                "spearman_rho": float(row["SpearmanRho"]),
                "edge_weight": edge_weight,
                "module": module_name,
                "module_color": module_to_color.get(module_name, "#9ca3af"),
                "fdr": fdr if np.isfinite(fdr) else None,
            })

        level = "module"
        y_label = "Module"
        y_modules = gene_order
        y_colors = [module_to_color.get(module_name, "#9ca3af") for module_name in gene_order]

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{level}_bubble_heatmap",
        "title": f"{'Gene' if is_gene_level else 'Module'}-Metabolite Bubble Heatmap",
        "chart_type": "bubble_heatmap",
        "interactive_page_id": "bubble-heatmap",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "data": rows,
            "y_order": gene_order,
            "x_order": metabolite_order,
            "y_modules": y_modules,
            "y_colors": y_colors,
            "y_label": y_label,
            "x_label": "Metabolite",
            "color_label": "Spearman rho",
            "size_label": "EdgeWeight" if is_gene_level else "-log10(FDR)",
            "config": _base_plotly_config(),
        },
        "default_state": {"level": level, "min_abs_rho": 0.0, "top_y": len(gene_order)},
        "available_states": {"level": ["gene", "module"],
                            "min_abs_rho": list(np.arange(0.0, 1.01, 0.1)),
                            "sort_x": ["max_rho_desc", "name_asc"],
                            "sort_y": ["module_then_rho_desc", "max_rho_desc", "name_asc"]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Correlation Heatmap (F12, F23)
# ---------------------------------------------------------------------------

__all__ = [
    "export_bubble_heatmap",
]
