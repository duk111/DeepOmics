from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors

from .base import _group_color_map, _metabolomics_df, _ordered_unique_with_order, _save_figure
from .module import (
    _align_group_annotations_to_samples,
    _coerce_module_eigengene_df,
    _module_group_orders_and_colors,
    _module_order_from_summary,
    _row_zscore,
)


def _plot_group1_violin_box_facets(
    plot_df: pd.DataFrame,
    *,
    feature_col: str,
    value_col: str,
    group1_order: list[str],
    group1_color_map: dict[str, str],
    title: str,
    y_label: str,
    save_stem: str | Path,
    cfg,
) -> None:
    if plot_df.empty or not group1_order:
        return

    features = plot_df[feature_col].astype(str).drop_duplicates().tolist()
    if not features:
        return

    n_features = len(features)
    n_cols = 1 if n_features == 1 else 2 if n_features <= 8 else 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig_width = max(6.4, min(18.0, 4.3 * n_cols))
    fig_height = max(4.2, min(26.0, 3.15 * n_rows + 0.75))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.ravel()

    palette = {group: group1_color_map.get(str(group), "#9ca3af") for group in group1_order}
    for ax, feature_name in zip(axes_flat, features):
        sub_df = plot_df.loc[plot_df[feature_col].astype(str).eq(str(feature_name))].copy()
        sub_df = sub_df.loc[sub_df["group1"].astype(str).isin(group1_order)]
        if sub_df.empty:
            ax.axis("off")
            continue

        sns.violinplot(
            data=sub_df,
            x="group1",
            y=value_col,
            hue="group1",
            order=group1_order,
            hue_order=group1_order,
            palette=palette,
            inner=None,
            cut=0,
            linewidth=0.8,
            saturation=0.82,
            dodge=False,
            legend=False,
            ax=ax,
        )
        sns.boxplot(
            data=sub_df,
            x="group1",
            y=value_col,
            order=group1_order,
            width=0.24,
            showcaps=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "#111827", "linewidth": 0.85, "alpha": 0.78},
            whiskerprops={"color": "#111827", "linewidth": 0.85},
            capprops={"color": "#111827", "linewidth": 0.85},
            medianprops={"color": "#111827", "linewidth": 1.15},
            ax=ax,
        )
        sns.stripplot(
            data=sub_df,
            x="group1",
            y=value_col,
            order=group1_order,
            color="#111827",
            size=2.5,
            jitter=0.16,
            alpha=0.48,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(str(feature_name), fontsize=10.4, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    for ax in axes_flat[n_features:]:
        ax.axis("off")

    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.965), h_pad=1.3, w_pad=1.0)
    _save_figure(fig, save_stem, cfg)


def plot_module_eigengene_group1_violin_box(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order = [module for module in module_order if module in eigengenes_df.columns]
    if not module_order:
        return

    zscore_df = _row_zscore(eigengenes_df.loc[:, module_order].T).T
    sample_names = zscore_df.index.astype(str).tolist()
    annotation_df = _align_group_annotations_to_samples(sample_names, group_df)
    if annotation_df.empty:
        return
    annotation_df = annotation_df.reindex(sample_names)

    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation_df["group1"].astype(str).tolist(),
        annotation_df["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = group_orders_by_col.get("group1", [])
    group1_color_map = color_maps_by_col.get("group1", {})
    if not group1_order:
        return

    rows: list[dict[str, object]] = []
    for module_name in module_order:
        values = pd.to_numeric(zscore_df[module_name], errors="coerce")
        for sample_id, value in values.items():
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "group1": str(annotation_df.loc[str(sample_id), "group1"]),
                    "Module": str(module_name),
                    "Value": float(value),
                }
            )
    plot_df = pd.DataFrame(rows)
    _plot_group1_violin_box_facets(
        plot_df,
        feature_col="Module",
        value_col="Value",
        group1_order=group1_order,
        group1_color_map=group1_color_map,
        title="Module Eigengene Distribution by group1",
        y_label="Module eigengene z-score",
        save_stem=save_stem,
        cfg=cfg,
    )


def _top_metabolite_order(engine, metab_df: pd.DataFrame, top_m: int) -> list[str]:
    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and "Metabolite" in summary_df.columns:
        candidates = summary_df["Metabolite"].astype(str).str.strip().tolist()
    else:
        candidates = metab_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()

    available = set(metab_df.columns.astype(str).tolist())
    ordered: list[str] = []
    for metabolite in candidates:
        if metabolite and metabolite in available and metabolite not in ordered:
            ordered.append(metabolite)
        if len(ordered) >= int(top_m):
            break
    return ordered


def _align_exact_group1_to_samples(sample_names: list[str], group_df: pd.DataFrame | None) -> pd.DataFrame:
    if group_df is None or group_df.empty or not {"sample_id", "group1"}.issubset(group_df.columns):
        return pd.DataFrame()
    work = group_df.copy()
    work["sample_id"] = work["sample_id"].astype(str).str.strip()
    work["group1"] = work["group1"].astype(str).str.strip()
    if "_group_table_order" not in work.columns:
        work["_group_table_order"] = np.arange(len(work), dtype=int)
    work = work.loc[work["sample_id"].ne("") & work["group1"].ne("")].copy()
    work = work.sort_values("_group_table_order", kind="mergesort").drop_duplicates("sample_id", keep="first")
    work = work.set_index("sample_id", drop=False)

    sample_index = pd.Index([str(sample).strip() for sample in sample_names], dtype=str)
    matched = sample_index.intersection(work.index, sort=False)
    if len(matched) == 0:
        return pd.DataFrame()
    return work.reindex(matched)


def plot_top_metabolite_group1_violin_box(
    engine,
    adata,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
    top_m: int | None = None,
) -> None:
    metab_df = _metabolomics_df(adata)
    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return
    metab_df = metab_df.copy(deep=False)
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip(), name=metab_df.index.name)
    metab_df.columns = metab_df.columns.astype(str)

    annotation_df = _align_exact_group1_to_samples(metab_df.index.astype(str).tolist(), group_df)
    if annotation_df.empty:
        return

    shared_samples = annotation_df.index.intersection(metab_df.index, sort=False)
    if len(shared_samples) < 2:
        return
    metab_df = metab_df.reindex(shared_samples).apply(pd.to_numeric, errors="coerce")
    annotation_df = annotation_df.reindex(shared_samples)

    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group1_order = _ordered_unique_with_order(annotation_df["group1"].astype(str).tolist(), group_orders)
    if not group1_order:
        return
    group1_color_map = _group_color_map(group1_order)

    metabolites = _top_metabolite_order(engine, metab_df, int(top_m or cfg.support_plot_top_metabolites))
    if not metabolites:
        return

    rows: list[dict[str, object]] = []
    for metabolite in metabolites:
        values = pd.to_numeric(metab_df[metabolite], errors="coerce")
        for sample_id, value in values.items():
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "group1": str(annotation_df.loc[str(sample_id), "group1"]),
                    "Metabolite": str(metabolite),
                    "Value": float(value),
                }
            )
    plot_df = pd.DataFrame(rows)
    _plot_group1_violin_box_facets(
        plot_df,
        feature_col="Metabolite",
        value_col="Value",
        group1_order=group1_order,
        group1_color_map=group1_color_map,
        title=f"Top {len(metabolites)} Metabolite Abundance by group1",
        y_label="Metabolite abundance z-score",
        save_stem=save_stem,
        cfg=cfg,
    )


def plot_module_kme_boxplot(
    engine,
    save_stem: str | Path,
    cfg,
) -> None:
    assignment_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    if not isinstance(assignment_df, pd.DataFrame) or assignment_df.empty:
        return
    if not {"Gene", "Module", "kME"}.issubset(assignment_df.columns):
        return

    plot_df = assignment_df.copy()
    plot_df["Gene"] = plot_df["Gene"].astype(str).str.strip()
    plot_df["Module"] = plot_df["Module"].astype(str).str.strip()
    plot_df["kME"] = pd.to_numeric(plot_df["kME"], errors="coerce")
    plot_df = plot_df.loc[plot_df["Gene"].ne("") & plot_df["Module"].ne("") & plot_df["kME"].notna()].copy()
    plot_df = plot_df.loc[plot_df["Module"].str.lower() != "grey"].copy()
    if plot_df.empty:
        return

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        plot_df["Module"].astype(str).drop_duplicates().tolist(),
    )
    module_order = [module for module in module_order if module in set(plot_df["Module"].astype(str))]
    if not module_order:
        module_order = (
            plot_df.groupby("Module", sort=False)["kME"]
            .median()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )

    module_color_map: dict[str, str] = {}
    if "ModuleColorHex" in plot_df.columns:
        color_rows = plot_df.loc[:, ["Module", "ModuleColorHex"]].drop_duplicates("Module", keep="first")
        for _, row in color_rows.iterrows():
            module_name = str(row["Module"]).strip()
            color_value = str(row["ModuleColorHex"]).strip()
            if not module_name or not color_value:
                continue
            try:
                module_color_map[module_name] = colors.to_hex(colors.to_rgba(color_value), keep_alpha=False)
            except ValueError:
                continue
    fallback_palette = sns.color_palette("tab20", n_colors=max(1, len(module_order))).as_hex()
    for idx, module_name in enumerate(module_order):
        module_color_map.setdefault(str(module_name), fallback_palette[idx % len(fallback_palette)])

    fig_width = max(7.0, min(18.0, 0.78 * max(1, len(module_order)) + 3.2))
    fig_height = 5.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    palette = {module: module_color_map.get(str(module), "#9ca3af") for module in module_order}

    sns.boxplot(
        data=plot_df,
        x="Module",
        y="kME",
        hue="Module",
        order=module_order,
        hue_order=module_order,
        palette=palette,
        width=0.48,
        showfliers=False,
        linewidth=0.95,
        dodge=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="Module",
        y="kME",
        order=module_order,
        color="#111827",
        size=2.7,
        jitter=0.20,
        alpha=0.42,
        linewidth=0,
        ax=ax,
    )
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_title("Intramodular Gene kME by Module")
    ax.set_xlabel("Module")
    ax.set_ylabel("kME")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "plot_module_eigengene_group1_violin_box",
    "plot_top_metabolite_group1_violin_box",
    "plot_module_kme_boxplot",
]
