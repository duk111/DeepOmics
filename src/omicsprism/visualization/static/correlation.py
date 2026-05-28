from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.patches import Patch

from .base import _gene_expression_df, _metabolomics_df, _save_figure
from .regression import _module_annotation_maps


def _top_gene_order(engine, gene_df: pd.DataFrame, top_n: int) -> list[str]:
    summary_df = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and "Gene" in summary_df.columns:
        genes = summary_df["Gene"].astype(str).str.strip().tolist()
    else:
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
            edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
        if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "Gene" in edge_df.columns:
            genes = edge_df["Gene"].astype(str).str.strip().drop_duplicates().tolist()
        else:
            genes = gene_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()

    available = set(gene_df.columns.astype(str).tolist())
    ordered: list[str] = []
    for gene in genes:
        if gene and gene in available and gene not in ordered:
            ordered.append(gene)
        if len(ordered) >= int(top_n):
            break
    return ordered


def _top_metabolite_order(engine, metab_df: pd.DataFrame, top_m: int) -> list[str]:
    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and "Metabolite" in summary_df.columns:
        metabolites = summary_df["Metabolite"].astype(str).str.strip().tolist()
    else:
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
            edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
        if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and "Metabolite" in edge_df.columns:
            metabolites = edge_df["Metabolite"].astype(str).str.strip().drop_duplicates().tolist()
        else:
            metabolites = metab_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()

    available = set(metab_df.columns.astype(str).tolist())
    ordered: list[str] = []
    for metabolite in metabolites:
        if metabolite and metabolite in available and metabolite not in ordered:
            ordered.append(metabolite)
        if len(ordered) >= int(top_m):
            break
    return ordered


def _correlation_matrix(
    gene_df: pd.DataFrame,
    metab_df: pd.DataFrame,
    genes: list[str],
    metabolites: list[str],
    *,
    method: str,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for gene in genes:
        gene_values = pd.to_numeric(gene_df[gene], errors="coerce")
        corr_values = metab_df.loc[:, metabolites].corrwith(gene_values, axis=0, method=method)
        rows.append(corr_values.rename(gene))
    if not rows:
        return pd.DataFrame(index=genes, columns=metabolites, dtype=float)
    return pd.DataFrame(rows).reindex(index=genes, columns=metabolites).astype(float)


def plot_top_gene_metabolite_correlation_heatmaps(
    engine,
    save_stem: str | Path,
    cfg,
    *,
    top_n: int | None = None,
    top_m: int | None = None,
) -> None:
    gene_df = engine.gene_expression_df() if hasattr(engine, "gene_expression_df") else _gene_expression_df(engine.adata)
    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else _metabolomics_df(engine.adata)
    if not isinstance(gene_df, pd.DataFrame) or gene_df.empty:
        return
    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return

    gene_df = gene_df.copy(deep=False)
    metab_df = metab_df.copy(deep=False)
    gene_df.index = pd.Index(gene_df.index.astype(str).str.strip())
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip())
    shared_samples = gene_df.index.intersection(metab_df.index, sort=False)
    if len(shared_samples) < 3:
        return

    gene_df = gene_df.reindex(shared_samples).apply(pd.to_numeric, errors="coerce")
    metab_df = metab_df.reindex(shared_samples).apply(pd.to_numeric, errors="coerce")

    genes = _top_gene_order(engine, gene_df, int(top_n or cfg.top_key_genes_plot_n))
    metabolites = _top_metabolite_order(engine, metab_df, int(top_m or cfg.support_plot_top_metabolites))
    if not genes or not metabolites:
        return

    pearson_df = _correlation_matrix(gene_df, metab_df, genes, metabolites, method="pearson")
    spearman_df = _correlation_matrix(gene_df, metab_df, genes, metabolites, method="spearman")
    if pearson_df.empty or spearman_df.empty:
        return

    gene_to_module, gene_to_color, _module_to_color = _module_annotation_maps(engine)
    gene_colors = [gene_to_color.get(gene, "#d1d5db") for gene in genes]
    gene_modules = [gene_to_module.get(gene, "Unassigned") for gene in genes]

    finite_values = np.concatenate(
        [
            pearson_df.to_numpy(dtype=float, copy=False).ravel(),
            spearman_df.to_numpy(dtype=float, copy=False).ravel(),
        ]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    vmax = max(0.25, float(np.nanmax(np.abs(finite_values))) if finite_values.size else 1.0)

    fig_width = max(10.5, min(30.0, 0.55 * max(1, len(metabolites)) * 2 + 4.2))
    fig_height = max(5.6, min(20.0, 0.33 * max(1, len(genes)) + 2.2))
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    gs = fig.add_gridspec(1, 3, width_ratios=[0.12, 1.0, 1.0], wspace=0.08)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_pearson = fig.add_subplot(gs[0, 1])
    ax_spearman = fig.add_subplot(gs[0, 2], sharey=ax_pearson)

    rgba = np.array([[colors.to_rgba(color) for color in gene_colors]], dtype=float).transpose((1, 0, 2))
    ax_strip.imshow(rgba, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks(np.arange(len(genes)))
    ax_strip.set_yticklabels([])
    ax_strip.set_ylabel("Gene module", fontsize=10)
    ax_strip.tick_params(axis="both", length=0)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    heatmap_kwargs = {
        "cmap": "vlag",
        "center": 0.0,
        "vmin": -vmax,
        "vmax": vmax,
        "linewidths": 0.35,
        "linecolor": "#f3f4f6",
        "mask": pearson_df.isna(),
    }
    sns.heatmap(
        pearson_df,
        ax=ax_pearson,
        cbar=False,
        **heatmap_kwargs,
    )
    heatmap_kwargs["mask"] = spearman_df.isna()
    sns.heatmap(
        spearman_df,
        ax=ax_spearman,
        cbar_kws={"label": "Correlation"},
        **heatmap_kwargs,
    )

    ax_pearson.set_title("Pearson")
    ax_spearman.set_title("Spearman")
    ax_pearson.set_xlabel("Metabolite")
    ax_spearman.set_xlabel("Metabolite")
    ax_pearson.set_ylabel("Top gene")
    ax_spearman.set_ylabel("")
    ax_pearson.set_yticklabels(ax_pearson.get_yticklabels(), rotation=0)
    ax_spearman.tick_params(axis="y", left=False, labelleft=False)
    for ax in (ax_pearson, ax_spearman):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    module_legend: list[Patch] = []
    seen_modules: set[str] = set()
    for module_name, color_value in zip(gene_modules, gene_colors):
        if module_name in seen_modules:
            continue
        seen_modules.add(module_name)
        module_legend.append(Patch(facecolor=color_value, edgecolor="none", label=module_name))
    if module_legend:
        legend_ncol = min(len(module_legend), max(1, int(fig_width // 1.35)))
        fig.legend(
            handles=module_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=legend_ncol,
            frameon=False,
            fontsize=9,
            handlelength=1.2,
            handleheight=0.7,
            columnspacing=0.9,
        )

    fig.suptitle(
        f"Top {len(genes)} Genes x Top {len(metabolites)} Metabolites",
        y=0.99,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.22)
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "plot_top_gene_metabolite_correlation_heatmaps",
]
