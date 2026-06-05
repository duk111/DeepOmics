from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .base import _metabolomics_df, _save_figure
from .module import (
    _align_group_annotations_to_samples,
    _coerce_module_eigengene_df,
    _module_group_orders_and_colors,
    _module_order_from_summary,
    _row_zscore,
)


def _optional_color(value, fallback: str = "#9ca3af") -> str:
    if pd.isna(value):
        return fallback
    try:
        return colors.to_hex(colors.to_rgba(str(value).strip()), keep_alpha=False)
    except ValueError:
        return fallback


def _module_maps(engine) -> tuple[dict[str, str], dict[str, str], dict[str, str], list[str]]:
    assignment_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())

    gene_to_module: dict[str, str] = {}
    module_to_color: dict[str, str] = {}
    module_order: list[str] = []

    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        module_order = module_summary_df["Module"].astype(str).str.strip().drop_duplicates().tolist()
        if "ModuleColorHex" in module_summary_df.columns:
            for _, row in module_summary_df.loc[:, ["Module", "ModuleColorHex"]].iterrows():
                module_to_color[str(row["Module"]).strip()] = _optional_color(row["ModuleColorHex"])

    if isinstance(assignment_df, pd.DataFrame) and not assignment_df.empty and {"Gene", "Module"}.issubset(assignment_df.columns):
        keep_cols = [column for column in ["Gene", "Module", "ModuleColorHex"] if column in assignment_df.columns]
        for _, row in assignment_df.loc[:, keep_cols].iterrows():
            gene = str(row["Gene"]).strip()
            module_name = str(row["Module"]).strip()
            if not gene or not module_name:
                continue
            gene_to_module[gene] = module_name
            if "ModuleColorHex" in row.index:
                module_to_color.setdefault(module_name, _optional_color(row["ModuleColorHex"]))

    all_modules = [module for module in module_order if module]
    for module_name in gene_to_module.values():
        if module_name and module_name not in all_modules:
            all_modules.append(module_name)
    fallback_colors = sns.color_palette("tab20", n_colors=max(1, len(all_modules))).as_hex()
    for idx, module_name in enumerate(all_modules):
        module_to_color.setdefault(str(module_name), fallback_colors[idx % len(fallback_colors)])

    gene_to_color = {
        gene: module_to_color.get(module_name, "#9ca3af")
        for gene, module_name in gene_to_module.items()
    }
    return gene_to_module, gene_to_color, module_to_color, module_order


def _ordered_modules(module_order: list[str], observed_modules: list[str]) -> list[str]:
    observed = [str(module).strip() for module in observed_modules if str(module).strip()]
    observed_set = set(observed)
    ordered = [module for module in module_order if module in observed_set and module.lower() != "grey"]
    for module in observed:
        if module.lower() != "grey" and module not in ordered:
            ordered.append(module)
    return ordered


def _metabolite_color_map(metabolites: list[str]) -> dict[str, str]:
    ordered = []
    seen: set[str] = set()
    for metabolite in metabolites:
        label = str(metabolite).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    palette = sns.color_palette("Set2", n_colors=max(1, len(ordered))).as_hex()
    return {metabolite: palette[idx % len(palette)] for idx, metabolite in enumerate(ordered)}


CIRCOS_METABOLITE_COLOR = "#c9ad85"


def plot_gene_metabolite_correlation_bubble_heatmap(engine, save_stem: str | Path, cfg) -> None:
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return
    required = {"Gene", "Metabolite", "SpearmanRho", "EdgeWeight"}
    if not required.issubset(edge_df.columns):
        return

    gene_to_module, gene_to_color, _module_to_color, module_order = _module_maps(engine)
    plot_df = edge_df.loc[:, ["Gene", "Metabolite", "SpearmanRho", "EdgeWeight"]].copy()
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
        return

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
        return

    gene_pos = {gene: idx for idx, gene in enumerate(gene_order)}
    metabolite_pos = {metabolite: idx for idx, metabolite in enumerate(metabolite_order)}
    plot_df["_x"] = plot_df["Metabolite"].map(metabolite_pos)
    plot_df["_y"] = plot_df["Gene"].map(gene_pos)

    fig_width = max(9.0, min(30.0, 0.45 * len(metabolite_order) + 4.8))
    fig_height = max(7.0, min(34.0, 0.18 * len(gene_order) + 3.2))
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    fig._skip_default_tight_layout = True
    gs = fig.add_gridspec(1, 3, width_ratios=[0.24, 0.024, 1.0], wspace=0.006)
    ax_labels = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1], sharey=ax_labels)
    ax = fig.add_subplot(gs[0, 2], sharey=ax_labels)

    gene_colors = [gene_to_color.get(gene, "#d1d5db") for gene in gene_order]
    rgba = np.array([[colors.to_rgba(color_value) for color_value in gene_colors]], dtype=float).transpose((1, 0, 2))
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(len(gene_order) - 0.4, -0.6)
    ax_labels.set_xticks([])
    ax_labels.set_yticks(np.arange(len(gene_order)))
    ax_labels.set_yticklabels(gene_order, fontsize=10)
    ax_labels.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, length=0, pad=1)
    ax_labels.set_ylabel("High-confidence gene", labelpad=34)
    for tick_label in ax_labels.get_yticklabels():
        tick_label.set_horizontalalignment("right")
    for spine in ax_labels.spines.values():
        spine.set_visible(False)

    ax_strip.imshow(rgba, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.tick_params(axis="y", left=False, labelleft=False)
    ax_strip.tick_params(axis="both", left=False, labelleft=False, length=0)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    module_boundaries: list[int] = []
    previous_module = None
    for idx, gene in enumerate(gene_order):
        module_name = str(gene_to_module.get(gene, "Unassigned"))
        if previous_module is not None and module_name != previous_module:
            module_boundaries.append(idx)
        previous_module = module_name

    edge_weights = plot_df["EdgeWeight"].to_numpy(dtype=float)
    min_size, max_size = 18.0, 190.0
    if np.nanmax(edge_weights) > np.nanmin(edge_weights):
        sizes = min_size + (edge_weights - np.nanmin(edge_weights)) / (np.nanmax(edge_weights) - np.nanmin(edge_weights)) * (max_size - min_size)
    else:
        sizes = np.full(edge_weights.shape, 80.0)

    scatter = ax.scatter(
        plot_df["_x"].to_numpy(dtype=float),
        plot_df["_y"].to_numpy(dtype=float),
        s=sizes,
        c=plot_df["SpearmanRho"].to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        edgecolors="#111827",
        linewidths=0.35,
        alpha=0.88,
    )
    ax.set_xlim(-0.6, len(metabolite_order) - 0.4)
    ax.set_ylim(len(gene_order) - 0.4, -0.6)
    ax.set_xticks(np.arange(len(metabolite_order)))
    ax.set_xticklabels(metabolite_order, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("Metabolite")
    ax.set_ylabel("")
    ax.set_title("High-Confidence Gene-Metabolite Correlation Bubble Heatmap")
    ax.grid(color="#e5e7eb", linewidth=0.42)
    ax.set_axisbelow(True)
    for boundary in module_boundaries:
        ax.axhline(boundary - 0.5, color="#9ca3af", linewidth=0.55, alpha=0.75)
        ax_strip.axhline(boundary - 0.5, color="#ffffff", linewidth=0.65)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.012)
    colorbar.set_label("Spearman rho")
    for value in (0.35, 0.65, 0.95):
        ax.scatter([], [], s=min_size + value * (max_size - min_size), color="#9ca3af", edgecolors="#111827", linewidths=0.35, label=f"{value:.2f}")
    ax.legend(title="EdgeWeight", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, labelspacing=1.0)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.94, bottom=0.23)
    _save_figure(fig, save_stem, cfg)


def plot_module_metabolite_bubble_plot(engine, save_stem: str | Path, cfg) -> None:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return
    if not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return

    _gene_to_module, _gene_to_color, module_to_color, module_order = _module_maps(engine)
    plot_df = assoc_df.copy()
    plot_df["Module"] = plot_df["Module"].astype(str).str.strip()
    plot_df["Metabolite"] = plot_df["Metabolite"].astype(str).str.strip()
    plot_df["SpearmanRho"] = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
    if "FDR" in plot_df.columns:
        plot_df["FDR"] = pd.to_numeric(plot_df["FDR"], errors="coerce")
    else:
        plot_df["FDR"] = np.nan
    plot_df = plot_df.loc[
        plot_df["Module"].ne("")
        & plot_df["Metabolite"].ne("")
        & plot_df["SpearmanRho"].notna()
        & (plot_df["Module"].str.lower() != "grey")
    ].copy()
    if plot_df.empty:
        return

    module_order = _ordered_modules(module_order, plot_df["Module"].astype(str).tolist())
    metabolite_order = (
        plot_df.assign(_AbsRho=plot_df["SpearmanRho"].abs())
        .groupby("Metabolite", sort=False)["_AbsRho"]
        .max()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    module_pos = {module: idx for idx, module in enumerate(module_order)}
    metabolite_pos = {metabolite: idx for idx, metabolite in enumerate(metabolite_order)}
    plot_df["_x"] = plot_df["Metabolite"].map(metabolite_pos)
    plot_df["_y"] = plot_df["Module"].map(module_pos)

    neglog_fdr = -np.log10(plot_df["FDR"].clip(lower=1e-300))
    neglog_fdr = neglog_fdr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(neglog_fdr.max()) > 0:
        sizes = 24.0 + (neglog_fdr / float(neglog_fdr.max())).to_numpy(dtype=float) * 210.0
    else:
        sizes = 24.0 + plot_df["SpearmanRho"].abs().to_numpy(dtype=float) * 210.0

    fig_width = max(9.0, min(30.0, 0.42 * len(metabolite_order) + 4.2))
    fig_height = max(4.8, min(16.0, 0.54 * len(module_order) + 2.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(
        plot_df["_x"].to_numpy(dtype=float),
        plot_df["_y"].to_numpy(dtype=float),
        s=sizes,
        c=plot_df["SpearmanRho"].to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        edgecolors="#111827",
        linewidths=0.35,
        alpha=0.86,
    )
    ax.set_xticks(np.arange(len(metabolite_order)))
    ax.set_xticklabels(metabolite_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(module_order)))
    ax.set_yticklabels(module_order)
    for tick_label in ax.get_yticklabels():
        tick_label.set_color("#111827")
    ax.set_xlabel("Metabolite")
    ax.set_ylabel("Module")
    ax.set_title("Module-Metabolite Association Bubble Plot")
    ax.grid(color="#e5e7eb", linewidth=0.42)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.012)
    colorbar.set_label("Spearman rho")
    _save_figure(fig, save_stem, cfg)


def _edge_module_dataframe(engine) -> pd.DataFrame:
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return pd.DataFrame()
    required = {"Gene", "Metabolite", "EdgeWeight"}
    if not required.issubset(edge_df.columns):
        return pd.DataFrame()
    gene_to_module, _gene_to_color, module_to_color, module_order = _module_maps(engine)
    plot_df = edge_df.copy()
    plot_df["Gene"] = plot_df["Gene"].astype(str).str.strip()
    plot_df["Module"] = plot_df["Gene"].map(gene_to_module).fillna("Unassigned")
    plot_df["EdgeWeight"] = pd.to_numeric(plot_df["EdgeWeight"], errors="coerce")
    if "Sign" in plot_df.columns:
        plot_df["Direction"] = plot_df["Sign"].astype(str).str.lower().map({"positive": "positive", "negative": "negative"})
    elif "SpearmanRho" in plot_df.columns:
        rho = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
        plot_df["Direction"] = np.where(rho >= 0, "positive", "negative")
    else:
        plot_df["Direction"] = "positive"
    plot_df["Direction"] = plot_df["Direction"].fillna("positive")
    plot_df = plot_df.loc[
        plot_df["Module"].astype(str).str.lower().ne("grey")
        & plot_df["EdgeWeight"].notna()
    ].copy()
    plot_df.attrs["module_order"] = _ordered_modules(module_order, plot_df["Module"].astype(str).tolist())
    plot_df.attrs["module_to_color"] = module_to_color
    return plot_df


def plot_association_direction_summary(engine, save_stem: str | Path, cfg) -> None:
    plot_df = _edge_module_dataframe(engine)
    if plot_df.empty:
        return
    module_order = plot_df.attrs.get("module_order", [])
    counts = (
        plot_df.groupby(["Module", "Direction"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=module_order, columns=["positive", "negative"], fill_value=0)
    )
    if counts.empty:
        return

    fig_width = max(7.2, min(18.0, 0.62 * len(module_order) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    x = np.arange(len(counts.index))
    pos = counts["positive"].to_numpy(dtype=float)
    neg = counts["negative"].to_numpy(dtype=float)
    ax.bar(x, pos, color="#e8a29a", label="positive", width=0.68)
    ax.bar(x, neg, bottom=pos, color="#8fb7df", label="negative", width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str).tolist(), rotation=35, ha="right")
    ax.set_ylabel("High-confidence edge count")
    ax.set_xlabel("Module")
    ax.set_title("Association Direction Summary by Module")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)
    fig.subplots_adjust(right=0.82)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.55)
    ax.set_axisbelow(True)
    _save_figure(fig, save_stem, cfg)


def plot_edgeweight_distribution_by_module(engine, save_stem: str | Path, cfg) -> None:
    plot_df = _edge_module_dataframe(engine)
    if plot_df.empty:
        return
    module_order = plot_df.attrs.get("module_order", [])
    module_to_color = plot_df.attrs.get("module_to_color", {})
    if not module_order:
        return
    palette = {module: module_to_color.get(str(module), "#9ca3af") for module in module_order}

    fig_width = max(7.2, min(18.0, 0.72 * len(module_order) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    sns.boxplot(
        data=plot_df,
        x="Module",
        y="EdgeWeight",
        hue="Module",
        order=module_order,
        hue_order=module_order,
        palette=palette,
        width=0.50,
        showfliers=False,
        dodge=False,
        legend=False,
        ax=ax,
    )
    sign_palette = {"positive": "#dc2626", "negative": "#2563eb"}
    sns.stripplot(
        data=plot_df,
        x="Module",
        y="EdgeWeight",
        hue="Direction",
        order=module_order,
        palette=sign_palette,
        size=3.0,
        jitter=0.20,
        alpha=0.52,
        linewidth=0,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles[:2],
            labels[:2],
            title="Direction",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )
    ax.set_xlabel("Module")
    ax.set_ylabel("EdgeWeight")
    ax.set_title("High-Confidence EdgeWeight Distribution by Module")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.subplots_adjust(right=0.82)
    _save_figure(fig, save_stem, cfg)


def _module_top_metabolite_pairs(engine) -> pd.DataFrame:
    summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and {"Module", "TopMetabolite"}.issubset(summary_df.columns):
        pairs = summary_df.loc[:, ["Module", "TopMetabolite"]].rename(columns={"TopMetabolite": "Metabolite"}).copy()
        pairs["Module"] = pairs["Module"].astype(str).str.strip()
        pairs["Metabolite"] = pairs["Metabolite"].astype(str).str.strip()
        pairs = pairs.loc[pairs["Module"].ne("") & pairs["Metabolite"].ne("") & (pairs["Module"].str.lower() != "grey")]
        if not pairs.empty:
            return pairs.drop_duplicates("Module", keep="first").reset_index(drop=True)

    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty or not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return pd.DataFrame()
    work = assoc_df.copy()
    work["SpearmanRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce")
    work["Module"] = work["Module"].astype(str).str.strip()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work = work.loc[work["Module"].ne("") & work["Metabolite"].ne("") & work["SpearmanRho"].notna() & (work["Module"].str.lower() != "grey")]
    return (
        work.assign(_AbsRho=work["SpearmanRho"].abs())
        .sort_values(["Module", "_AbsRho", "Metabolite"], ascending=[True, False, True])
        .drop_duplicates("Module", keep="first")
        .loc[:, ["Module", "Metabolite"]]
        .reset_index(drop=True)
    )


def plot_module_eigengene_metabolite_trend_panels(engine, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return
    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else _metabolomics_df(engine.adata)
    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return
    metab_df = metab_df.copy(deep=False)
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip())
    metab_df.columns = metab_df.columns.astype(str)

    pairs_df = _module_top_metabolite_pairs(engine)
    if pairs_df.empty:
        return
    shared_samples = eigengenes_df.index.intersection(metab_df.index, sort=False)
    if len(shared_samples) < 2:
        return
    eigengenes_df = eigengenes_df.reindex(shared_samples)
    metab_df = metab_df.reindex(shared_samples)

    annotation_df = _align_group_annotations_to_samples(shared_samples.astype(str).tolist(), group_df)
    if annotation_df.empty:
        return
    annotation_df = annotation_df.reindex(shared_samples.astype(str).tolist())
    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation_df["group1"].astype(str).tolist(),
        annotation_df["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = group_orders_by_col.get("group1", [])
    group2_order = group_orders_by_col.get("group2", [])
    group1_color_map = color_maps_by_col.get("group1", {})
    if not group1_order or not group2_order:
        return

    valid_pairs = []
    for row in pairs_df.itertuples(index=False):
        module_name = str(row.Module)
        metabolite = str(row.Metabolite)
        if module_name in eigengenes_df.columns and metabolite in metab_df.columns:
            valid_pairs.append((module_name, metabolite))
    if not valid_pairs:
        return

    _gene_to_module, _gene_to_color, module_to_color, _all_module_order = _module_maps(engine)

    trend_modules = list(dict.fromkeys(module for module, _ in valid_pairs))
    trend_metabolites = list(dict.fromkeys(metabolite for _, metabolite in valid_pairs))
    module_z = _row_zscore(eigengenes_df.loc[:, trend_modules].T).T
    metab_z = _row_zscore(metab_df.loc[:, trend_metabolites].T).T

    n_rows = len(valid_pairs)
    n_cols = len(group1_order)
    fig_width = max(7.0, min(24.0, 2.35 * n_cols + 3.2))
    fig_height = max(4.8, min(28.0, 1.25 * n_rows + 1.6))
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.50],
        wspace=0.18,
        hspace=0.22,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
    legend_ax = fig.add_subplot(grid[:, -1])
    legend_ax.axis("off")
    x_positions = np.arange(len(group2_order), dtype=float)

    for row_idx, (module_name, metabolite) in enumerate(valid_pairs):
        for col_idx, group1_name in enumerate(group1_order):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(str(group1_name), color=group1_color_map.get(str(group1_name), "#111827"), fontsize=10)
            module_color = module_to_color.get(str(module_name), "#111827")
            metabolite_color = CIRCOS_METABOLITE_COLOR
            point_color = group1_color_map.get(str(group1_name), "#9ca3af")
            module_line = []
            metab_line = []
            for group2_name in group2_order:
                samples = annotation_df.index[
                    annotation_df["group1"].astype(str).eq(str(group1_name))
                    & annotation_df["group2"].astype(str).eq(str(group2_name))
                ].astype(str).tolist()
                module_line.append(float(pd.to_numeric(module_z.loc[samples, module_name], errors="coerce").mean()) if samples else np.nan)
                metab_line.append(float(pd.to_numeric(metab_z.loc[samples, metabolite], errors="coerce").mean()) if samples else np.nan)
            ax.scatter(
                x_positions,
                module_line,
                s=15,
                marker="o",
                facecolor=point_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.88,
                zorder=3,
            )
            ax.scatter(
                x_positions,
                metab_line,
                s=15,
                marker="s",
                facecolor=point_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.88,
                zorder=3,
            )
            ax.plot(x_positions, module_line, color=module_color, linewidth=1.35, alpha=0.96, zorder=4, label="Module")
            ax.plot(x_positions, metab_line, color=metabolite_color, linewidth=1.35, alpha=0.96, zorder=4, label="Metabolite")
            ax.axhline(0, color="#9ca3af", linewidth=0.6, linestyle=(0, (4, 3)))
            ax.set_xlim(-0.35, len(group2_order) - 0.65)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(group2_order if row_idx == n_rows - 1 else [], rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
            if col_idx == 0:
                ax.set_ylabel(f"{module_name}\n{metabolite}", fontsize=8)
            else:
                ax.set_yticklabels([])
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", linewidth=1.35, marker="o", markersize=4.2, markerfacecolor="#d1d5db", markeredgecolor="white", markeredgewidth=0.35, label="Module eigengene"),
            Line2D([0], [0], color=CIRCOS_METABOLITE_COLOR, linewidth=1.35, marker="s", markersize=4.2, markerfacecolor="#d1d5db", markeredgecolor="white", markeredgewidth=0.35, label="Metabolite"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=legend_ax.transAxes,
        ncol=1,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.42,
        columnspacing=0.85,
        labelspacing=0.42,
        borderaxespad=0.0,
        fontsize=10.5,
    )

    fig.suptitle("Module Eigengene vs Top Metabolite Trends by group1", y=0.995, fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.93, wspace=0.18, hspace=0.22)
    _save_figure(fig, save_stem, cfg)


def plot_gene_module_metabolite_sankey(engine, save_stem: str | Path, cfg, *, top_genes_per_module: int = 5, top_metabolites_per_module: int = 3) -> None:
    assignment_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assignment_df, pd.DataFrame) or assignment_df.empty:
        return
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return
    if not {"Gene", "Module", "kME"}.issubset(assignment_df.columns):
        return
    if not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return

    _gene_to_module, _gene_to_color, module_to_color, module_order = _module_maps(engine)
    modules = _ordered_modules(module_order, assignment_df["Module"].astype(str).tolist())
    if not modules:
        return

    gene_rows = assignment_df.copy()
    gene_rows["Gene"] = gene_rows["Gene"].astype(str).str.strip()
    gene_rows["Module"] = gene_rows["Module"].astype(str).str.strip()
    gene_rows["kME"] = pd.to_numeric(gene_rows["kME"], errors="coerce")
    gene_rows = gene_rows.loc[gene_rows["Gene"].ne("") & gene_rows["Module"].isin(modules) & gene_rows["kME"].notna()].copy()
    gene_rows = (
        gene_rows.assign(_AbsKME=gene_rows["kME"].abs())
        .sort_values(["Module", "_AbsKME", "Gene"], ascending=[True, False, True])
        .groupby("Module", sort=False)
        .head(int(top_genes_per_module))
    )

    metab_rows = assoc_df.copy()
    metab_rows["Module"] = metab_rows["Module"].astype(str).str.strip()
    metab_rows["Metabolite"] = metab_rows["Metabolite"].astype(str).str.strip()
    metab_rows["SpearmanRho"] = pd.to_numeric(metab_rows["SpearmanRho"], errors="coerce")
    metab_rows = metab_rows.loc[metab_rows["Module"].isin(modules) & metab_rows["Metabolite"].ne("") & metab_rows["SpearmanRho"].notna()].copy()
    metab_rows = (
        metab_rows.assign(_AbsRho=metab_rows["SpearmanRho"].abs())
        .sort_values(["Module", "_AbsRho", "Metabolite"], ascending=[True, False, True])
        .groupby("Module", sort=False)
        .head(int(top_metabolites_per_module))
    )
    if gene_rows.empty or metab_rows.empty:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    labels: list[str] = []
    node_colors: list[str] = []
    node_index: dict[str, int] = {}

    def add_node(key: str, label: str, color_value: str) -> int:
        if key in node_index:
            return node_index[key]
        node_index[key] = len(labels)
        labels.append(label)
        node_colors.append(color_value)
        return node_index[key]

    for _, row in gene_rows.iterrows():
        add_node(f"gene:{row['Gene']}", str(row["Gene"]), "#94a3b8")
    for module_name in modules:
        if module_name in set(gene_rows["Module"]) or module_name in set(metab_rows["Module"]):
            add_node(f"module:{module_name}", str(module_name), module_to_color.get(module_name, "#9ca3af"))
    for _, row in metab_rows.iterrows():
        add_node(f"metab:{row['Metabolite']}", str(row["Metabolite"]), "#f59e0b")

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    for _, row in gene_rows.iterrows():
        module_color = module_to_color.get(str(row["Module"]), "#9ca3af")
        sources.append(add_node(f"gene:{row['Gene']}", str(row["Gene"]), "#94a3b8"))
        targets.append(add_node(f"module:{row['Module']}", str(row["Module"]), module_color))
        values.append(max(0.05, float(abs(row["kME"]))))
        link_colors.append(colors.to_rgba(module_color, alpha=0.38))
    for _, row in metab_rows.iterrows():
        module_color = module_to_color.get(str(row["Module"]), "#9ca3af")
        sources.append(add_node(f"module:{row['Module']}", str(row["Module"]), module_color))
        targets.append(add_node(f"metab:{row['Metabolite']}", str(row["Metabolite"]), "#f59e0b"))
        values.append(max(0.05, float(abs(row["SpearmanRho"]))))
        link_colors.append("rgba(220,38,38,0.34)" if float(row["SpearmanRho"]) >= 0 else "rgba(37,99,235,0.34)")

    link_colors = [
        color_value if isinstance(color_value, str) else f"rgba({int(color_value[0] * 255)},{int(color_value[1] * 255)},{int(color_value[2] * 255)},{color_value[3]:.3f})"
        for color_value in link_colors
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={"pad": 12, "thickness": 14, "line": {"color": "#111827", "width": 0.25}, "label": labels, "color": node_colors},
                link={"source": sources, "target": targets, "value": values, "color": link_colors},
            )
        ]
    )
    fig.update_layout(title_text="Top Hub Genes - Modules - Top Metabolites Sankey", font_size=10, margin={"l": 10, "r": 10, "t": 42, "b": 10})

    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_stem.with_suffix(".html")), include_plotlyjs="cdn")
    if getattr(cfg, "export_png", True):
        try:
            fig.write_image(str(save_stem.with_suffix(".png")), width=1600, height=900, scale=2)
        except Exception:
            pass
    if getattr(cfg, "export_svg", True):
        try:
            fig.write_image(str(save_stem.with_suffix(".svg")), width=1600, height=900)
        except Exception:
            pass


__all__ = [
    "plot_gene_metabolite_correlation_bubble_heatmap",
    "plot_module_metabolite_bubble_plot",
    "plot_association_direction_summary",
    "plot_module_eigengene_metabolite_trend_panels",
    "plot_edgeweight_distribution_by_module",
]
