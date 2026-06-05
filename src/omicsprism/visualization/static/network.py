from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MplPath

from .base import (
    PALETTE,
    _categorical_colors,
    _gene_expression_df,
    _group_color_map,
    _hue_wheel_color_series,
    _metabolomics_df,
    _ordered_unique_nonempty,
    _save_figure,
)
from .pca import _load_pca_group_table

def _build_signed_count_summary(edge_df: pd.DataFrame, node_column: str) -> pd.DataFrame:
    """Aggregate node-level weighted degree and direction bias from high-confidence edges."""
    if edge_df.empty:
        return pd.DataFrame(
            columns=[
                node_column,
                "WeightedDegree",
                "PositiveEdgeCount",
                "NegativeEdgeCount",
                "DirectionBias",
            ]
        )

    work = edge_df.loc[:, [node_column, "EdgeWeight", "Sign"]].copy()
    work[node_column] = work[node_column].astype(str)
    work["EdgeWeight"] = pd.to_numeric(work["EdgeWeight"], errors="coerce").fillna(0.0)

    summary = work.groupby(node_column, sort=False)["EdgeWeight"].sum().rename("WeightedDegree").to_frame()
    summary["PositiveEdgeCount"] = (
        work["Sign"].astype(str).str.lower().eq("positive").groupby(work[node_column], sort=False).sum().astype(int)
    )
    summary["NegativeEdgeCount"] = (
        work["Sign"].astype(str).str.lower().eq("negative").groupby(work[node_column], sort=False).sum().astype(int)
    )

    total_counts = summary["PositiveEdgeCount"] + summary["NegativeEdgeCount"]
    summary["DirectionBias"] = np.where(
        total_counts > 0,
        (summary["PositiveEdgeCount"] - summary["NegativeEdgeCount"]) / total_counts,
        0.0,
    )
    return summary.reset_index()


def _compute_standardized_feature_variability(feature_df: pd.DataFrame) -> pd.Series:
    """Compute per-feature SD from the current standardized matrix without re-z-scoring."""
    if feature_df.empty:
        return pd.Series(dtype=float)

    values = feature_df.to_numpy(dtype=float, copy=False)
    variability = np.nanstd(values, axis=0, ddof=0)
    variability = np.where(np.isfinite(variability), variability, 0.0)
    variability = np.where(variability > 0, variability, 0.0)
    return pd.Series(variability, index=feature_df.columns.astype(str), dtype=float)


def _prepare_circos_node_tables(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ordered gene/metabolite node tables from the high-confidence network."""
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    required_columns = {"Gene", "Metabolite", "EdgeWeight", "Sign", "ModelSupportCount"}
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty or not required_columns.issubset(edge_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edge_df = edge_df.copy()
    edge_df["Gene"] = edge_df["Gene"].astype(str)
    edge_df["Metabolite"] = edge_df["Metabolite"].astype(str)
    edge_df["EdgeWeight"] = pd.to_numeric(edge_df["EdgeWeight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    edge_df["ModelSupportCount"] = pd.to_numeric(edge_df["ModelSupportCount"], errors="coerce").fillna(0.0)
    edge_df["Sign"] = edge_df["Sign"].astype(str).str.lower()

    gene_df = _gene_expression_df(engine.adata)
    metab_df = _metabolomics_df(engine.adata)
    gene_mean_z = gene_df.mean(axis=0)
    metab_mean_z = metab_df.mean(axis=0)
    gene_variability = _compute_standardized_feature_variability(gene_df)
    metab_variability = _compute_standardized_feature_variability(metab_df)

    gene_summary = _build_signed_count_summary(edge_df, "Gene").rename(columns={"Gene": "Node"})
    gene_summary["NodeType"] = "gene"
    gene_summary["MeanZScore"] = gene_summary["Node"].map(gene_mean_z).fillna(0.0).astype(float)
    gene_summary["InterSampleVariability"] = gene_summary["Node"].map(gene_variability).fillna(0.0).astype(float)
    gene_summary["AbsDirectionBias"] = gene_summary["DirectionBias"].abs()
    gene_summary = gene_summary.sort_values(
        ["WeightedDegree", "AbsDirectionBias", "MeanZScore", "Node"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    metab_summary = _build_signed_count_summary(edge_df, "Metabolite").rename(columns={"Metabolite": "Node"})
    metab_summary["NodeType"] = "metabolite"
    metab_summary["MeanZScore"] = metab_summary["Node"].map(metab_mean_z).fillna(0.0).astype(float)
    metab_summary["InterSampleVariability"] = metab_summary["Node"].map(metab_variability).fillna(0.0).astype(float)
    metab_summary["AbsDirectionBias"] = metab_summary["DirectionBias"].abs()
    metab_summary = metab_summary.sort_values(
        ["WeightedDegree", "AbsDirectionBias", "MeanZScore", "Node"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return edge_df, gene_summary, metab_summary


def _build_circos_module_color_map(module_names: list[str]) -> dict[str, str]:
    ordered_modules = [str(name) for name in module_names if str(name).strip()]
    unique_modules = _ordered_unique_nonempty(ordered_modules)

    canonical_map = {
        "turquoise": "#40E0D0",
        "blue": "#1F77B4",
        "brown": "#8B4513",
        "yellow": "#FFD700",
        "green": "#2CA02C",
        "red": "#D62728",
        "black": "#000000",
        "pink": "#FFC0CB",
        "magenta": "#FF00FF",
        "purple": "#800080",
        "greenyellow": "#ADFF2F",
        "tan": "#D2B48C",
        "salmon": "#FA8072",
        "cyan": "#00FFFF",
        "midnightblue": "#191970",
        "lightcyan": "#E0FFFF",
        "royalblue": "#4169E1",
        "darkred": "#8B0000",
        "darkgreen": "#006400",
        "darkturquoise": "#00CED1",
        "darkgrey": "#A9A9A9",
        "orange": "#FFA500",
        "white": "#FFFFFF",
        "skyblue": "#87CEEB",
        "saddlebrown": "#8B4513",
        "steelblue": "#4682B4",
        "paleturquoise": "#AFEEEE",
        "violet": "#EE82EE",
        "darkorange": "#FF8C00",
        "darkmagenta": "#8B008B",
        "grey": "#E5E7EB",
    }

    fallback_palette = _categorical_colors(len(unique_modules))
    color_map: dict[str, str] = {}
    for idx, module_name in enumerate(unique_modules):
        key = str(module_name).strip().lower()
        color_map[module_name] = canonical_map.get(key, fallback_palette[idx % len(fallback_palette)] if fallback_palette else "#9ca3af")
    return color_map


def _attach_circos_module_annotations(engine, gene_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    work = gene_summary.copy()
    work["Module"] = "grey"
    work["ModuleColor"] = "#E5E7EB"
    work["ModuleSize"] = 0
    work["kME"] = np.nan
    work["IntramodularDegree"] = np.nan
    work["IsGrey"] = 1

    module_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    required_columns = {"Gene", "Module"}
    if isinstance(module_df, pd.DataFrame) and not module_df.empty and required_columns.issubset(module_df.columns):
        keep_cols = [
            "Gene",
            "Module",
            "ModuleColorHex",
            "ModuleSize",
            "kME",
            "IntramodularDegree",
            "IsGrey",
        ]
        module_keep = module_df.loc[:, [col for col in keep_cols if col in module_df.columns]].copy()
        module_keep = module_keep.rename(columns={"Gene": "Node"})
        module_keep["Node"] = module_keep["Node"].astype(str)
        module_keep["Module"] = module_keep["Module"].astype(str).replace("", "grey")
        work = work.merge(module_keep, on="Node", how="left", suffixes=("", "_Module"))

        work["Module"] = work.get("Module_Module", work["Module"]).fillna("grey").astype(str)
        work["ModuleSize"] = pd.to_numeric(work.get("ModuleSize_Module", 0), errors="coerce").fillna(0).astype(int)
        work["kME"] = pd.to_numeric(work.get("kME_Module", np.nan), errors="coerce").astype(float)
        work["IntramodularDegree"] = pd.to_numeric(work.get("IntramodularDegree_Module", np.nan), errors="coerce").astype(float)
        work["IsGrey"] = pd.to_numeric(work.get("IsGrey_Module", 1), errors="coerce").fillna(1).astype(int)

        if "ModuleColorHex_Module" in work.columns:
            work["ModuleColor"] = work["ModuleColorHex_Module"].fillna("#E5E7EB").astype(str)

        drop_cols = [col for col in work.columns if col.endswith("_Module")]
        if drop_cols:
            work = work.drop(columns=drop_cols)

    work["Module"] = work["Module"].fillna("grey").astype(str)
    work["IsGrey"] = (work["Module"].str.lower() == "grey").astype(int)

    non_grey = work.loc[work["IsGrey"] == 0, ["Module", "ModuleSize"]].drop_duplicates()
    module_order = non_grey.sort_values(
        ["ModuleSize", "Module"],
        ascending=[False, True],
        kind="mergesort",
    )["Module"].astype(str).tolist()
    if "grey" in work["Module"].astype(str).tolist():
        module_order.append("grey")

    module_color_map = _build_circos_module_color_map(module_order)
    missing_color_mask = work["ModuleColor"].isna() | work["ModuleColor"].astype(str).eq("")
    work.loc[missing_color_mask, "ModuleColor"] = work.loc[missing_color_mask, "Module"].map(module_color_map).fillna("#E5E7EB")
    work["ModuleColor"] = work["Module"].map(module_color_map).fillna(work["ModuleColor"]).fillna("#E5E7EB")

    work = work.sort_values(
        ["IsGrey", "ModuleSize", "Module", "kME", "IntramodularDegree", "WeightedDegree", "Node"],
        ascending=[True, False, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return work, module_color_map


def _compute_circos_layout(gene_nodes: list[str], metabolite_nodes: list[str]) -> dict[str, dict[str, float | str]]:
    """Assign compact angular positions with genes and metabolites in two consecutive sectors."""
    n_gene = len(gene_nodes)
    n_metabolite = len(metabolite_nodes)
    n_total = n_gene + n_metabolite
    if n_total == 0:
        return {}

    full_circle = 2.0 * np.pi
    mean_item_span = full_circle / float(n_total)
    item_gap = min(np.deg2rad(0.45), mean_item_span * 0.10)
    group_gap = max(np.deg2rad(7.0), item_gap * 8.0)

    total_gap = max(0, n_total - 2) * item_gap + 2.0 * group_gap
    if total_gap >= full_circle * 0.92:
        item_gap = mean_item_span * 0.04
        group_gap = max(np.deg2rad(4.0), item_gap * 6.0)
        total_gap = max(0, n_total - 2) * item_gap + 2.0 * group_gap

    item_width = (full_circle - total_gap) / float(n_total)
    if item_width <= 0:
        item_gap = 0.0
        group_gap = 0.0
        item_width = full_circle / float(n_total)

    layout: dict[str, dict[str, float | str]] = {}
    current_angle = np.pi * 0.76 + group_gap / 2.0

    def _assign(node_ids: list[str], node_type: str, after_group_gap: float) -> float:
        nonlocal current_angle
        for idx, node_id in enumerate(node_ids):
            theta_start = current_angle
            theta_end = theta_start + item_width
            layout[str(node_id)] = {
                "theta_start": theta_start,
                "theta_end": theta_end,
                "theta_mid": 0.5 * (theta_start + theta_end),
                "node_type": node_type,
            }
            current_angle = theta_end
            if idx < len(node_ids) - 1:
                current_angle += item_gap
            else:
                current_angle += after_group_gap
        return current_angle

    if n_gene > 0:
        _assign(gene_nodes, "gene", group_gap)
    if n_metabolite > 0:
        _assign(metabolite_nodes, "metabolite", group_gap)

    return layout


def _polar_to_xy(theta: float, radius: float) -> tuple[float, float]:
    return float(radius * np.cos(theta)), float(radius * np.sin(theta))


def _add_annular_segment(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    facecolor,
    edgecolor: str = "#ffffff",
    linewidth: float = 0.35,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    if r_outer <= r_inner:
        return

    patch = Wedge(
        center=(0.0, 0.0),
        r=float(r_outer),
        theta1=float(np.degrees(theta_start)),
        theta2=float(np.degrees(theta_end)),
        width=float(r_outer - r_inner),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    patch.set_zorder(zorder)
    ax.add_patch(patch)


def _add_circos_link(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    radius: float,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int = 0,
) -> None:
    start_xy = np.asarray(_polar_to_xy(theta_start, radius), dtype=float)
    end_xy = np.asarray(_polar_to_xy(theta_end, radius), dtype=float)

    path = MplPath(
        [
            tuple(start_xy),
            tuple(start_xy * 0.18),
            tuple(end_xy * 0.18),
            tuple(end_xy),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    patch.set_zorder(zorder)
    ax.add_patch(patch)


def _robust_abs_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    scale = float(np.nanpercentile(np.abs(arr), 95))
    return max(scale, 1e-6)


def _positive_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    scale = float(np.nanmax(arr))
    return max(scale, 1e-6)


def _prepare_group1_mean_track_data(feature_df: pd.DataFrame, group_df: pd.DataFrame | None) -> dict[str, object] | None:
    if feature_df.empty:
        return None

    feature_work = feature_df.copy()
    feature_work.index = feature_work.index.astype(str)

    if group_df is None or "sample_id" not in group_df.columns or "group1" not in group_df.columns:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
            "group1_color_map": {},
        }

    group_work = group_df.copy()
    group_work["sample_id"] = group_work["sample_id"].astype(str).str.strip()
    group_work["group1"] = group_work["group1"].astype(str).str.strip()
    group_work = group_work.loc[group_work["sample_id"].isin(feature_work.index)].copy()
    if group_work.empty:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
            "group1_color_map": {},
        }

    group_work = group_work.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id", drop=True)
    aligned_samples = [sample for sample in feature_work.index.tolist() if sample in group_work.index]
    if not aligned_samples:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
            "group1_color_map": {},
        }

    feature_work = feature_work.loc[aligned_samples].copy()
    aligned_group = group_work.reindex(aligned_samples).copy()
    group1_order = _ordered_unique_nonempty(aligned_group["group1"].tolist())
    if not group1_order:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
            "group1_color_map": {},
        }

    agg_input = feature_work.copy()
    agg_input["group1"] = aligned_group["group1"].astype(str).to_numpy()
    agg_df = agg_input.groupby("group1", sort=False)[feature_work.columns.astype(str).tolist()].mean()
    feature_to_values = {
        str(feature): [float(agg_df.loc[group_name, feature]) for group_name in group1_order if group_name in agg_df.index]
        for feature in feature_work.columns.astype(str).tolist()
    }
    flattened = [float(v) for values in feature_to_values.values() for v in values if np.isfinite(v)]
    return {
        "mode": "group1_mean",
        "feature_to_values": feature_to_values,
        "abs_scale": _robust_abs_scale(flattened),
        "group1_order": group1_order,
        "group1_color_map": _group_color_map(group1_order),
    }


def _draw_track_baseline(ax: plt.Axes, theta_start: float, theta_end: float, radius: float, *, color: str = "#d1d5db", linewidth: float = 0.18, alpha: float = 1.0, zorder: float = 2.7) -> None:
    n_points = 32
    thetas = np.linspace(theta_start, theta_end, n_points)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_group1_scatter_track(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    values: list[float],
    value_scale: float,
    random_state: int,
    group_names: list[str] | None = None,
    group_color_map: dict[str, str] | None = None,
    zorder: float = 3.1,
) -> None:
    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        r_inner,
        r_outer,
        facecolor="#fbfbfb",
        edgecolor="#eef2f7",
        linewidth=0.14,
        alpha=1.0,
        zorder=int(zorder),
    )
    r_mid = 0.5 * (r_inner + r_outer)
    _draw_track_baseline(ax, theta_start, theta_end, r_mid, color="#d1d5db", linewidth=0.18, alpha=0.9, zorder=zorder)

    if group_names is not None and len(group_names) == len(values):
        clean_entries = [
            (float(value), str(group_name))
            for value, group_name in zip(values, group_names)
            if np.isfinite(value)
        ]
    else:
        clean_entries = [(float(value), "") for value in values if np.isfinite(value)]

    if not clean_entries:
        return

    rng_seed = int(random_state + round(float(theta_start) * 1e6)) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)

    theta_width = float(theta_end - theta_start)
    span_scale = max(theta_width * 0.06, np.deg2rad(0.08))
    radial_half_span = 0.42 * (r_outer - r_inner)
    scale = max(float(value_scale), 1e-6)

    for value, group_name in clean_entries:
        theta = 0.5 * (theta_start + theta_end) + float(rng.uniform(-span_scale, span_scale))
        clipped = float(np.clip(value, -scale, scale))
        radius = r_mid + (clipped / scale) * radial_half_span
        point_color = "#6b7280"
        if group_color_map is not None and group_name:
            point_color = str(group_color_map.get(group_name, point_color))
        x, y = _polar_to_xy(theta, radius)
        ax.scatter([x], [y], s=5.0, c=[point_color], edgecolors="none", alpha=0.92, zorder=zorder + 0.15)


def _draw_mean_hist_track(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    value: float,
    value_scale: float,
    color: str = "#6b7280",
    zorder: float = 3.0,
) -> None:
    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        r_inner,
        r_outer,
        facecolor="#fbfbfb",
        edgecolor="#eef2f7",
        linewidth=0.14,
        alpha=1.0,
        zorder=int(zorder),
    )
    r_mid = 0.5 * (r_inner + r_outer)
    _draw_track_baseline(ax, theta_start, theta_end, r_mid, color="#d1d5db", linewidth=0.18, alpha=0.9, zorder=zorder)

    scale = max(float(value_scale), 1e-6)
    clipped = float(np.clip(value, -scale, scale))
    if clipped >= 0:
        bar_inner = r_mid
        bar_outer = r_mid + (clipped / scale) * (r_outer - r_mid)
    else:
        bar_inner = r_mid + (clipped / scale) * (r_mid - r_inner)
        bar_outer = r_mid

    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        bar_inner,
        bar_outer,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.88,
        zorder=int(zorder + 0.1),
    )


def _prepare_metabolite_module_core_map(engine) -> pd.Series:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if isinstance(assoc_df, pd.DataFrame) and not assoc_df.empty and {"Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        work = assoc_df.copy()
        work["Metabolite"] = work["Metabolite"].astype(str)
        work["AbsRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce").abs()
        best = work.groupby("Metabolite", sort=False)["AbsRho"].max()
        return best.astype(float)
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and {"Metabolite", "EdgeWeight"}.issubset(edge_df.columns):
        fallback = edge_df.groupby("Metabolite", sort=False)["EdgeWeight"].sum()
        return fallback.astype(float)
    return pd.Series(dtype=float)


def _prepare_module_legend_items(gene_summary: pd.DataFrame) -> list[tuple[str, str]]:
    if gene_summary.empty or "Module" not in gene_summary.columns:
        return []
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for row in gene_summary.loc[:, ["Module", "ModuleColor"]].drop_duplicates().itertuples(index=False):
        module_name = str(row.Module)
        if module_name in seen:
            continue
        seen.add(module_name)
        items.append((module_name, str(row.ModuleColor)))
    return items


def _add_corner_module_legend(
    ax: plt.Axes,
    legend_items: list[tuple[str, str]],
    *,
    x_left: float,
    y_top: float,
    row_height: float = 0.072,
    swatch_width: float = 0.12,
    swatch_height: float = 0.028,
) -> None:
    if not legend_items:
        return

    for idx, (module_name, module_color) in enumerate(legend_items):
        y = y_top - idx * row_height
        rect = plt.Rectangle(
            (x_left, y - 0.5 * swatch_height),
            swatch_width,
            swatch_height,
            facecolor=module_color,
            edgecolor="#9ca3af",
            linewidth=0.3,
            zorder=7,
        )
        ax.add_patch(rect)
        ax.text(
            x_left + swatch_width + 0.03,
            y,
            module_name,
            ha="left",
            va="center",
            fontsize=8.0,
            color="#374151",
            zorder=7,
        )


def _prepare_group1_legend_items(track_data: dict[str, object] | None) -> list[tuple[str, str]]:
    if track_data is None or str(track_data.get("mode", "")) != "group1_mean":
        return []

    group1_order = [str(value) for value in track_data.get("group1_order", []) if str(value).strip()]
    color_map = {str(key): str(value) for key, value in dict(track_data.get("group1_color_map", {})).items()}

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for group_name in group1_order:
        if group_name in seen:
            continue
        seen.add(group_name)
        items.append((group_name, color_map.get(group_name, "#6b7280")))
    return items


def _add_corner_group_legend(
    ax: plt.Axes,
    legend_items: list[tuple[str, str]],
    *,
    title: str,
    x_left: float,
    y_top: float,
    row_height: float = 0.072,
    marker_diameter: float = 0.026,
) -> None:
    if not legend_items:
        return

    title_y = y_top
    ax.text(
        x_left,
        title_y,
        title,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        zorder=7,
    )

    marker_radius = 0.5 * marker_diameter
    for idx, (group_name, group_color) in enumerate(legend_items):
        y = y_top - (idx + 1) * row_height
        circle = plt.Circle(
            (x_left + marker_radius, y),
            radius=marker_radius,
            facecolor=group_color,
            edgecolor="#9ca3af",
            linewidth=0.3,
            zorder=7,
        )
        ax.add_patch(circle)
        ax.text(
            x_left + marker_diameter + 0.03,
            y,
            group_name,
            ha="left",
            va="center",
            fontsize=8.0,
            color="#374151",
            zorder=7,
        )


def _compute_circos_outer_gap_theta(
    layout: dict[str, dict[str, float | str]],
    gene_nodes: list[str],
    metabolite_nodes: list[str],
) -> float:
    if not layout or not gene_nodes or not metabolite_nodes:
        return float(np.pi / 2.0)

    first_gene = str(gene_nodes[0])
    last_metabolite = str(metabolite_nodes[-1])
    if first_gene not in layout or last_metabolite not in layout:
        return float(np.pi / 2.0)

    gap_start = float(layout[last_metabolite]["theta_end"])
    gap_end = float(layout[first_gene]["theta_start"]) + 2.0 * np.pi
    return float((0.5 * (gap_start + gap_end)) % (2.0 * np.pi))


def _add_circos_track_number_labels(
    ax: plt.Axes,
    radii: dict[str, float],
    label_theta: float,
    *,
    fontsize: float = 8.5,
) -> None:
    track_radii = [
        0.5 * (radii["outer_strip_inner"] + radii["outer_strip_outer"]),
        0.5 * (radii["track_meanbar_inner"] + radii["track_meanbar_outer"]),
        0.5 * (radii["track_meanheat_inner"] + radii["track_meanheat_outer"]),
        0.5 * (radii["track_degree_inner"] + radii["track_degree_outer"]),
        0.5 * (radii["track_core_inner"] + radii["track_core_outer"]),
        0.5 * (radii["track_bias_inner"] + radii["track_bias_outer"]),
    ]

    x_shift = 0.024 if np.cos(label_theta) >= -0.05 else -0.024
    ha = "left" if x_shift >= 0 else "right"

    for idx, radius in enumerate(track_radii, start=1):
        x, y = _polar_to_xy(label_theta, radius)
        ax.text(
            x + x_shift,
            y,
            str(idx),
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#374151",
            zorder=7,
        )


def _add_track_annotation_legend(
    ax: plt.Axes,
    *,
    x_left: float,
    y_top: float,
    row_height: float = 0.072,
    label_width: float = 0.18,
) -> None:
    legend_rows = [
        ("track 1", "sector strip"),
        ("track 2", "group-wise mean"),
        ("track 3", "mean z-score heatmap"),
        ("track 4", "weighted degree"),
        ("track 5", "module/core strength"),
        ("track 6", "direction bias"),
    ]

    ax.text(
        x_left,
        y_top,
        "Track annotations",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        zorder=7,
    )

    for idx, (track_label, description) in enumerate(legend_rows):
        y = y_top - (idx + 1) * row_height
        ax.text(
            x_left,
            y,
            track_label,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="#374151",
            zorder=7,
        )
        ax.text(
            x_left + label_width,
            y,
            description,
            ha="left",
            va="center",
            fontsize=8.0,
            color="#6b7280",
            zorder=7,
        )


def plot_compressed_circos_network(engine, save_stem: str | Path, cfg) -> None:
    """Plot a compact static Circos figure using high-confidence nodes and edges."""
    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return

    gene_summary, _module_color_map = _attach_circos_module_annotations(engine, gene_summary)

    metabolite_module_core = _prepare_metabolite_module_core_map(engine)
    metabolite_summary = metabolite_summary.copy()
    metabolite_summary["Module"] = ""
    metabolite_summary["ModuleColor"] = "#c9ad85"
    metabolite_summary["ModuleCore"] = metabolite_summary["Node"].map(metabolite_module_core).astype(float)

    gene_summary["ModuleCore"] = pd.to_numeric(gene_summary.get("kME", np.nan), errors="coerce").abs()
    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return

    node_df = pd.concat([gene_summary, metabolite_summary], ignore_index=True)
    node_df["Node"] = node_df["Node"].astype(str)

    gene_mean_scale = _robust_abs_scale(gene_summary["MeanZScore"])
    metabolite_mean_scale = _robust_abs_scale(metabolite_summary["MeanZScore"])
    gene_degree_scale = _positive_scale(gene_summary["WeightedDegree"])
    metabolite_degree_scale = _positive_scale(metabolite_summary["WeightedDegree"])
    gene_core_scale = _positive_scale(gene_summary["ModuleCore"])
    metabolite_core_scale = _positive_scale(metabolite_summary["ModuleCore"])

    gene_mean_norm = colors.TwoSlopeNorm(vmin=-gene_mean_scale, vcenter=0.0, vmax=gene_mean_scale)
    metabolite_mean_norm = colors.TwoSlopeNorm(vmin=-metabolite_mean_scale, vcenter=0.0, vmax=metabolite_mean_scale)
    bias_norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    mean_cmap = plt.get_cmap("RdBu_r")
    bias_cmap = plt.get_cmap("RdBu_r")

    group_df = _load_pca_group_table(cfg)
    circos_track_adata = getattr(engine, "plot_adata", getattr(engine, "unaggregated_adata", engine.adata))
    gene_track_data = _prepare_group1_mean_track_data(_gene_expression_df(circos_track_adata), group_df)
    metabolite_track_data = _prepare_group1_mean_track_data(_metabolomics_df(circos_track_adata), group_df)

    radii = {
        "outer_strip_inner": 0.992,
        "outer_strip_outer": 1.035,
        "track_meanbar_inner": 0.86,
        "track_meanbar_outer": 0.975,
        "track_meanheat_inner": 0.795,
        "track_meanheat_outer": 0.85,
        "track_degree_inner": 0.685,
        "track_degree_outer": 0.775,
        "track_core_inner": 0.605,
        "track_core_outer": 0.675,
        "track_bias_inner": 0.53,
        "track_bias_outer": 0.58,
        "link_radius": 0.47,
    }

    fig, ax = plt.subplots(figsize=(11.7, 10.8))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    edge_ordered = edge_df.sort_values(
        ["EdgeWeight", "ModelSupportCount", "Gene", "Metabolite"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    support_min = float(edge_ordered["ModelSupportCount"].min()) if not edge_ordered.empty else 0.0
    support_max = float(edge_ordered["ModelSupportCount"].max()) if not edge_ordered.empty else 1.0

    for row in edge_ordered.itertuples(index=False):
        gene_id = str(row.Gene)
        metabolite_id = str(row.Metabolite)
        if gene_id not in layout or metabolite_id not in layout:
            continue

        edge_weight = float(np.clip(getattr(row, "EdgeWeight", 0.0), 0.0, None))
        line_width = 0.18 + 1.72 * np.sqrt(min(1.0, edge_weight))

        model_support = float(getattr(row, "ModelSupportCount", 0.0))
        if support_max > support_min:
            line_alpha = 0.05 + 0.30 * (model_support - support_min) / (support_max - support_min)
        else:
            line_alpha = 0.22 if support_max > 0 else 0.08

        line_color = PALETTE["edge_positive"] if str(row.Sign).lower() == "positive" else PALETTE["edge_negative"]
        _add_circos_link(
            ax,
            float(layout[gene_id]["theta_mid"]),
            float(layout[metabolite_id]["theta_mid"]),
            radii["link_radius"],
            color=line_color,
            linewidth=line_width,
            alpha=float(np.clip(line_alpha, 0.04, 0.92)),
            zorder=0,
        )

    for row in node_df.itertuples(index=False):
        node_id = str(row.Node)
        geometry = layout.get(node_id)
        if geometry is None:
            continue

        theta_start = float(geometry["theta_start"])
        theta_end = float(geometry["theta_end"])
        node_type = str(row.NodeType)
        mean_value = float(row.MeanZScore)
        degree_value = float(max(0.0, row.WeightedDegree))
        direction_bias = float(np.clip(row.DirectionBias, -1.0, 1.0))
        core_value = float(max(0.0, getattr(row, "ModuleCore", np.nan))) if pd.notna(getattr(row, "ModuleCore", np.nan)) else 0.0

        if node_type == "gene":
            outer_color = getattr(row, "ModuleColor", "#7db8ab")
            mean_norm = gene_mean_norm
            degree_scale = gene_degree_scale
            core_scale = gene_core_scale
            track_data = gene_track_data
            core_color = outer_color
        else:
            outer_color = "#c9ad85"
            mean_norm = metabolite_mean_norm
            degree_scale = metabolite_degree_scale
            core_scale = metabolite_core_scale
            track_data = metabolite_track_data
            core_color = "#8c6d46"

        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["outer_strip_inner"],
            radii["outer_strip_outer"],
            facecolor=outer_color,
            edgecolor="#ffffff",
            linewidth=0.45,
            alpha=1.0,
            zorder=4,
        )

        mean_color = mean_cmap(mean_norm(mean_value))
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_meanheat_inner"],
            radii["track_meanheat_outer"],
            facecolor=mean_color,
            edgecolor="#ffffff",
            linewidth=0.22,
            alpha=1.0,
            zorder=3.2,
        )

        track_values = track_data["feature_to_values"].get(node_id, []) if track_data is not None else []
        if track_data is not None and str(track_data.get("mode", "")) == "group1_mean":
            _draw_group1_scatter_track(
                ax,
                theta_start,
                theta_end,
                radii["track_meanbar_inner"],
                radii["track_meanbar_outer"],
                values=list(track_values),
                value_scale=float(track_data.get("abs_scale", 1.0)),
                random_state=int(getattr(cfg, "random_state", 42)),
                group_names=[str(name) for name in track_data.get("group1_order", [])],
                group_color_map={str(key): str(value) for key, value in dict(track_data.get("group1_color_map", {})).items()},
                zorder=3.45,
            )
        else:
            mean_value_for_bar = float(track_values[0]) if track_values else float(mean_value)
            _draw_mean_hist_track(
                ax,
                theta_start,
                theta_end,
                radii["track_meanbar_inner"],
                radii["track_meanbar_outer"],
                value=mean_value_for_bar,
                value_scale=float(track_data.get("abs_scale", 1.0) if track_data is not None else max(gene_mean_scale, metabolite_mean_scale)),
                color="#6b7280",
                zorder=3.45,
            )

        degree_outer = radii["track_degree_inner"] + (radii["track_degree_outer"] - radii["track_degree_inner"]) * min(
            1.0, degree_value / max(degree_scale, 1e-6)
        )
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_degree_inner"],
            degree_outer,
            facecolor="#4b5563",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=2.3,
        )

        core_outer = radii["track_core_inner"] + (radii["track_core_outer"] - radii["track_core_inner"]) * min(
            1.0, core_value / max(core_scale, 1e-6)
        )
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_core_inner"],
            core_outer,
            facecolor=core_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=1.8,
        )

        bias_color = bias_cmap(bias_norm(direction_bias))
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_bias_inner"],
            radii["track_bias_outer"],
            facecolor=bias_color,
            edgecolor="#ffffff",
            linewidth=0.22,
            alpha=1.0,
            zorder=1.0,
        )

    group_legend_items = _prepare_group1_legend_items(gene_track_data)
    _add_corner_group_legend(
        ax,
        group_legend_items,
        title="Track 2 group colors",
        x_left=-1.48,
        y_top=-0.08,
        row_height=0.072,
        marker_diameter=0.026,
    )

    legend_items = _prepare_module_legend_items(gene_summary)
    _add_corner_module_legend(
        ax,
        legend_items,
        x_left=-1.48,
        y_top=-0.46,
        row_height=0.072,
        swatch_width=0.11,
        swatch_height=0.026,
    )

    label_theta = _compute_circos_outer_gap_theta(layout, gene_nodes, metabolite_nodes)
    _add_circos_track_number_labels(ax, radii, label_theta)
    _add_track_annotation_legend(
        ax,
        x_left=-1.48,
        y_top=0.98,
        row_height=0.072,
        label_width=0.18,
    )

    outer_limit_x = 1.58
    outer_limit_y = 1.12
    ax.set_xlim(-outer_limit_x, 1.12)
    ax.set_ylim(-outer_limit_y, outer_limit_y)
    _save_figure(fig, save_stem, cfg)


def plot_floating_cnet_circos_network(engine, save_stem: str | Path, cfg) -> None:
    """Plot a high-confidence circular cnetplot-style network with non-overlapping circular nodes."""
    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return

    gene_summary, _module_color_map = _attach_circos_module_annotations(engine, gene_summary)
    gene_summary = gene_summary.copy()
    metabolite_summary = metabolite_summary.copy()

    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return

    ordered_nodes = gene_nodes + metabolite_nodes
    theta_series = pd.Series({node: float(layout[node]["theta_mid"]) for node in ordered_nodes})

    gene_summary["EdgeCount"] = (
        pd.to_numeric(gene_summary.get("PositiveEdgeCount", 0), errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(gene_summary.get("NegativeEdgeCount", 0), errors="coerce").fillna(0).astype(int)
    )
    metabolite_summary["EdgeCount"] = (
        pd.to_numeric(metabolite_summary.get("PositiveEdgeCount", 0), errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(metabolite_summary.get("NegativeEdgeCount", 0), errors="coerce").fillna(0).astype(int)
    )

    node_table = pd.concat([
        gene_summary.loc[:, ["Node", "NodeType", "EdgeCount", "ModuleColor"]],
        metabolite_summary.assign(ModuleColor="#c9ad85").loc[:, ["Node", "NodeType", "EdgeCount", "ModuleColor"]],
    ], ignore_index=True)
    node_table["Node"] = node_table["Node"].astype(str)
    node_table = node_table.set_index("Node").reindex(ordered_nodes).reset_index()

    theta_values = theta_series.reindex(ordered_nodes).to_numpy(dtype=float)
    n_nodes = len(theta_values)
    if n_nodes == 0:
        return

    wrapped = np.r_[theta_values, theta_values[0] + 2.0 * np.pi]
    theta_diffs = np.diff(wrapped)
    positive_diffs = theta_diffs[theta_diffs > 1e-6]
    min_theta_gap = float(np.min(positive_diffs)) if positive_diffs.size else (2.0 * np.pi)

    base_radius = 1.0
    min_center_distance = 2.0 * base_radius * np.sin(max(min_theta_gap, 1e-6) / 2.0)
    max_node_radius = float(np.clip(min_center_distance * 0.36, 0.012, 0.032))
    min_node_radius = float(np.clip(max_node_radius * 0.42, 0.006, max_node_radius * 0.72))

    edge_count_series = pd.to_numeric(node_table["EdgeCount"], errors="coerce").fillna(0).astype(float)
    edge_count_max = float(edge_count_series.max()) if len(edge_count_series) else 0.0
    edge_count_min = float(edge_count_series.min()) if len(edge_count_series) else 0.0
    if edge_count_max > edge_count_min:
        scaled = (edge_count_series - edge_count_min) / (edge_count_max - edge_count_min)
    else:
        scaled = pd.Series(np.ones(len(edge_count_series)), index=node_table.index, dtype=float)
    node_table["NodeRadius"] = (min_node_radius + scaled * (max_node_radius - min_node_radius)).astype(float)

    base_jitter = min(0.060, max(0.016, min_theta_gap * 0.12))
    jitter = base_jitter * np.sin(np.linspace(0.0, 3.2 * np.pi, n_nodes, endpoint=False) + 0.65)

    for _ in range(8):
        adjusted_radius = base_radius + jitter
        ok = True
        for idx in range(n_nodes):
            jdx = (idx + 1) % n_nodes
            xy1 = np.asarray(_polar_to_xy(float(theta_values[idx]), float(adjusted_radius[idx])), dtype=float)
            xy2 = np.asarray(_polar_to_xy(float(theta_values[jdx]), float(adjusted_radius[jdx])), dtype=float)
            center_distance = float(np.linalg.norm(xy2 - xy1))
            min_required = float(node_table["NodeRadius"].iloc[idx] + node_table["NodeRadius"].iloc[jdx] + 0.008)
            if center_distance < min_required:
                ok = False
                break
        if ok:
            break
        jitter *= 0.82

    node_table["Theta"] = theta_values
    node_table["RingRadius"] = (base_radius + jitter).astype(float)
    node_table["X"] = [
        _polar_to_xy(float(theta), float(radius))[0]
        for theta, radius in zip(node_table["Theta"], node_table["RingRadius"])
    ]
    node_table["Y"] = [
        _polar_to_xy(float(theta), float(radius))[1]
        for theta, radius in zip(node_table["Theta"], node_table["RingRadius"])
    ]

    metabolite_edge_colors = _hue_wheel_color_series(len(metabolite_nodes), hue_start=18.0, lightness=63.0, safety=0.92)
    metabolite_edge_color_map = {
        metabolite: metabolite_edge_colors[idx]
        for idx, metabolite in enumerate(metabolite_nodes)
    }

    node_xy = {
        str(row.Node): (float(row.X), float(row.Y), float(row.Theta), float(row.RingRadius), float(row.NodeRadius))
        for row in node_table.itertuples(index=False)
    }

    fig, ax = plt.subplots(figsize=(10.8, 10.2))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    edge_ordered = edge_df.sort_values(
        ["Metabolite", "Gene", "EdgeWeight"],
        ascending=[True, True, False],
        kind="mergesort",
    )

    for row in edge_ordered.itertuples(index=False):
        gene_id = str(row.Gene)
        metabolite_id = str(row.Metabolite)
        if gene_id not in node_xy or metabolite_id not in node_xy:
            continue
        theta_gene = node_xy[gene_id][2]
        theta_metabolite = node_xy[metabolite_id][2]
        edge_radius = min(node_xy[gene_id][3], node_xy[metabolite_id][3]) - 0.05
        _add_circos_link(
            ax,
            float(theta_gene),
            float(theta_metabolite),
            max(0.70, float(edge_radius)),
            color=metabolite_edge_color_map.get(metabolite_id, "#9ca3af"),
            linewidth=0.30,
            alpha=0.80,
            zorder=0,
        )

    for row in node_table.itertuples(index=False):
        circle = plt.Circle(
            (float(row.X), float(row.Y)),
            radius=float(row.NodeRadius),
            facecolor=str(row.ModuleColor) if pd.notna(row.ModuleColor) else "#9ca3af",
            edgecolor="#ffffff",
            linewidth=0.9,
            alpha=1.0,
            zorder=3,
        )
        ax.add_patch(circle)

    gene_handle = Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#9ca3af", markeredgecolor="#ffffff", markeredgewidth=0.9, label="Gene node")
    metabolite_handle = Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#c9ad85", markeredgecolor="#ffffff", markeredgewidth=0.9, label="Metabolite node")
    edge_handle = Line2D([0], [0], color="#6b7280", lw=0.9, label="Metabolite-colored edge")
    ax.legend(handles=[gene_handle, metabolite_handle, edge_handle], loc="upper right", frameon=False, fontsize=9.5)

    max_extent = 1.24 + float(node_table["NodeRadius"].max()) if not node_table.empty else 1.3
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "_build_circos_module_color_map",
    "plot_compressed_circos_network",
    "plot_floating_cnet_circos_network",
]
