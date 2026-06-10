"""Interactive figure data export for F12 (gene-metabolite correlation heatmap)
and F23 (module-metabolite association heatmap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import _gene_expression_df, _metabolomics_df
from ..static.correlation import (
    _correlation_matrix,
    _top_gene_order,
    _top_metabolite_order,
)
from ..static.module import (
    _module_order_from_summary,
    _significance_star,
)
from ..static.regression import _module_annotation_maps
from .common import (
    _base_layout,
    _base_plotly_config,
    _base_style,
    _json_matrix_values,
    _seaborn_vlag_colorscale,
)


def _coerce_engine_gene_expression_df(engine) -> pd.DataFrame:
    """Get gene expression DataFrame with cleaned numeric values."""
    if hasattr(engine, "gene_expression_df"):
        try:
            gene_df = engine.gene_expression_df()
        except Exception:
            gene_df = pd.DataFrame()
    else:
        gene_df = pd.DataFrame()

    if (not isinstance(gene_df, pd.DataFrame) or gene_df.empty) and hasattr(engine, "adata"):
        gene_df = _gene_expression_df(engine.adata)

    if not isinstance(gene_df, pd.DataFrame) or gene_df.empty:
        return pd.DataFrame()

    work = gene_df.copy(deep=False)
    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work.columns = pd.Index(work.columns.astype(str).str.strip(), name=work.columns.name)
    work = work.loc[work.index.astype(str).str.len() > 0, work.columns.astype(str).str.len() > 0].copy()
    work = work.loc[~work.index.duplicated(keep="first"), ~work.columns.duplicated(keep="first")].copy()
    return work.apply(pd.to_numeric, errors="coerce")


def _coerce_engine_metabolomics_df(engine) -> pd.DataFrame:
    """Get metabolomics DataFrame with cleaned numeric values."""
    if hasattr(engine, "metabolomics_df"):
        try:
            metab_df = engine.metabolomics_df()
        except Exception:
            metab_df = pd.DataFrame()
    else:
        metab_df = pd.DataFrame()

    if (not isinstance(metab_df, pd.DataFrame) or metab_df.empty) and hasattr(engine, "adata"):
        metab_df = _metabolomics_df(engine.adata)

    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return pd.DataFrame()

    work = metab_df.copy(deep=False)
    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work.columns = pd.Index(work.columns.astype(str).str.strip(), name=work.columns.name)
    work = work.loc[work.index.astype(str).str.len() > 0, work.columns.astype(str).str.len() > 0].copy()
    work = work.loc[~work.index.duplicated(keep="first"), ~work.columns.duplicated(keep="first")].copy()
    return work.apply(pd.to_numeric, errors="coerce")


def export_gene_metabolite_heatmap(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export F12 top gene-metabolite Spearman correlation heatmap data."""
    style = _base_style()
    engine = context.engine
    cfg = context.cfg

    gene_df = _coerce_engine_gene_expression_df(engine)
    metab_df = _coerce_engine_metabolomics_df(engine)
    if gene_df.empty or metab_df.empty:
        return None

    shared_samples = gene_df.index.intersection(metab_df.index, sort=False)
    if len(shared_samples) < 3:
        return None

    gene_df = gene_df.reindex(shared_samples)
    metab_df = metab_df.reindex(shared_samples)

    top_n = int(getattr(cfg, "top_key_genes_plot_n", 20))
    top_m = int(getattr(cfg, "support_plot_top_metabolites", 20))

    genes = _top_gene_order(engine, gene_df, top_n)
    metabolites = _top_metabolite_order(engine, metab_df, top_m)
    if not genes or not metabolites:
        return None

    spearman_df = _correlation_matrix(gene_df, metab_df, genes, metabolites, method="spearman")
    if spearman_df.empty:
        return None

    gene_to_module, gene_to_color, _module_to_color = _module_annotation_maps(engine)
    gene_colors = [gene_to_color.get(gene, "#d1d5db") for gene in genes]

    finite_values = spearman_df.to_numpy(dtype=float, copy=False).ravel()
    finite_values = finite_values[np.isfinite(finite_values)]
    vmax = max(0.25, float(np.nanmax(np.abs(finite_values))) if finite_values.size else 1.0)

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    z_values = _json_matrix_values(spearman_df)

    return {
        "figure_id": "top_gene_metabolite_correlation_heatmaps",
        "title": f"Top {len(genes)} Genes × Top {len(metabolites)} Metabolites",
        "chart_type": "heatmap",
        "interactive_page_id": "corr-heatmap",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "data": [
                {
                    "type": "heatmap",
                    "z": z_values,
                    "x": metabolites,
                    "y": genes,
                    "colorscale": _seaborn_vlag_colorscale(),
                    "zmid": 0.0,
                    "zmin": -vmax,
                    "zmax": vmax,
                    "colorbar": {"title": "Correlation"},
                    "hovertemplate": "Gene: %{y}<br>Metabolite: %{x}<br>Spearman ρ: %{z:.3f}<extra></extra>",
                }
            ],
            "layout": {
                **_base_layout(f"Top {len(genes)} Genes × Top {len(metabolites)} Metabolites", style),
                "xaxis": {"tickangle": 45},
                "yaxis": {"autorange": "reversed"},
            },
            "config": _base_plotly_config(),
            "y_colors": gene_colors,
        },
        "default_state": {
            "view": "gene-metabolite",
            "color_scheme": "vlag",
            "show_significance": True,
            "show_values": False,
        },
        "available_states": {
            "view": ["gene-metabolite", "module-metabolite"],
            "color_scheme": ["vlag", "RdBu_r", "RdBu", "RdYlBu_r", "coolwarm", "viridis"],
        },
        "style": style,
    }


def _get_module_metabolite_association_df(engine) -> pd.DataFrame:
    """Fetch module-metabolite association dataframe from engine."""
    for key in (
        "module_metabolite_association_df",
        "module_metabolite_assoc_df",
        "module_metabolite_association",
    ):
        df = getattr(engine, key, None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        df = engine.ml_results.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def export_module_metabolite_heatmap(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export F23 module-metabolite association heatmap data with significance stars."""
    style = _base_style()
    engine = context.engine

    assoc_df = _get_module_metabolite_association_df(engine)
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return None

    required_columns = {"Module", "Metabolite", "SpearmanRho"}
    if not required_columns.issubset(assoc_df.columns):
        return None

    plot_df = assoc_df.copy()
    plot_df["Module"] = plot_df["Module"].astype(str).str.strip()
    plot_df["Metabolite"] = plot_df["Metabolite"].astype(str).str.strip()
    plot_df["SpearmanRho"] = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
    if "FDR" in plot_df.columns:
        plot_df["FDR"] = pd.to_numeric(plot_df["FDR"], errors="coerce")
    if "PValue" in plot_df.columns:
        plot_df["PValue"] = pd.to_numeric(plot_df["PValue"], errors="coerce")

    non_grey_df = plot_df.loc[plot_df["Module"].str.lower() != "grey"].copy()
    if not non_grey_df.empty:
        plot_df = non_grey_df

    significance_column = "FDR" if ("FDR" in plot_df.columns and plot_df["FDR"].notna().any()) else "PValue"
    if significance_column not in plot_df.columns:
        plot_df["PValue"] = np.nan
        significance_column = "PValue"

    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    module_order: list[str] = []
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        available_modules = set(plot_df["Module"].astype(str).tolist())
        module_order = [
            str(module_name)
            for module_name in module_summary_df["Module"].astype(str).tolist()
            if str(module_name) in available_modules
        ]

    if not module_order:
        module_order = (
            plot_df.assign(
                _AbsRho=plot_df["SpearmanRho"].abs(),
                _SigRank=pd.to_numeric(plot_df[significance_column], errors="coerce").fillna(1.0),
            )
            .sort_values(
                ["_SigRank", "_AbsRho", "Module"],
                ascending=[True, False, True],
                kind="mergesort",
            )["Module"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

    metabolite_order = (
        plot_df.assign(
            _AbsRho=plot_df["SpearmanRho"].abs(),
            _SigRank=pd.to_numeric(plot_df[significance_column], errors="coerce").fillna(1.0),
        )
        .sort_values(
            ["_SigRank", "_AbsRho", "Metabolite"],
            ascending=[True, False, True],
            kind="mergesort",
        )["Metabolite"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    rho_matrix = (
        plot_df.pivot(index="Module", columns="Metabolite", values="SpearmanRho")
        .reindex(index=module_order, columns=metabolite_order)
    )
    sig_matrix = (
        plot_df.pivot(index="Module", columns="Metabolite", values=significance_column)
        .reindex(index=module_order, columns=metabolite_order)
    )

    if rho_matrix.empty:
        return None

    # Build annotation stars for cells with significant p-values
    annotation_rows: list[dict[str, Any]] = []
    customdata_values: list[list[float | None]] = []
    for row_idx, module_name in enumerate(module_order):
        row_customdata: list[float | None] = []
        for col_idx, metabolite_name in enumerate(metabolite_order):
            sig_val = sig_matrix.at[module_name, metabolite_name] if module_name in sig_matrix.index and metabolite_name in sig_matrix.columns else np.nan
            star = _significance_star(sig_val)
            if star:
                annotation_rows.append({"row": row_idx, "col": col_idx, "text": star})
            row_customdata.append(float(sig_val) if np.isfinite(float(sig_val)) else None)
        customdata_values.append(row_customdata)

    finite_rho = rho_matrix.to_numpy(dtype=float)
    finite_rho = finite_rho[np.isfinite(finite_rho)]
    vmax = float(np.nanmax(np.abs(finite_rho))) if finite_rho.size > 0 else 1.0
    vmax = max(vmax, 0.25)

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    z_values = _json_matrix_values(rho_matrix)

    metric_label = "FDR" if significance_column == "FDR" else "P value"

    return {
        "figure_id": "module_metabolite_association_heatmap",
        "title": "Module-Metabolite Association Heatmap",
        "chart_type": "heatmap",
        "interactive_page_id": "corr-heatmap",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "data": [
                {
                    "type": "heatmap",
                    "z": z_values,
                    "x": metabolite_order,
                    "y": module_order,
                    "colorscale": _seaborn_vlag_colorscale(),
                    "zmid": 0.0,
                    "zmin": -vmax,
                    "zmax": vmax,
                    "colorbar": {
                        "title": {
                            "text": f"Spearman rho<br><sup>{metric_label}<br>* < 0.05, ** < 0.01, *** < 0.001</sup>",
                        },
                    },
                    "customdata": customdata_values,
                    "hovertemplate": (
                        "Module: %{y}<br>Metabolite: %{x}<br>Spearman ρ: %{z:.3f}<br>"
                        + metric_label
                        + ": %{customdata:.2e}<extra></extra>"
                    ),
                }
            ],
            "layout": {
                **_base_layout("Module-Metabolite Association Heatmap", style),
                "xaxis": {"tickangle": 45},
                "yaxis": {"autorange": "reversed"},
            },
            "config": _base_plotly_config(),
            "annotations": annotation_rows,
        },
        "default_state": {
            "view": "module-metabolite",
            "color_scheme": "vlag",
            "show_significance": True,
            "show_values": False,
        },
        "available_states": {
            "view": ["gene-metabolite", "module-metabolite"],
            "color_scheme": ["vlag", "RdBu_r", "RdBu", "RdYlBu_r", "coolwarm", "viridis"],
        },
        "style": style,
    }


__all__ = [
    "export_gene_metabolite_heatmap",
    "export_module_metabolite_heatmap",
]
