from __future__ import annotations

from pathlib import Path

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2
from sklearn.decomposition import PCA

from .base import (
    PALETTE,
    _global_secondary_group_color_map,
    _group_color_map,
    _group_marker_map,
    _ordered_unique_nonempty,
    _ordered_unique_with_order,
    _save_figure,
    logger,
)

def _load_pca_group_table(cfg) -> pd.DataFrame | None:
    group_table_path = getattr(cfg, "group_table_path", None)
    if not group_table_path:
        return None

    group_table_path = Path(group_table_path)
    group_df = pd.read_csv(group_table_path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    normalized_columns = {
        str(column).replace("\ufeff", "").strip().lower(): column
        for column in group_df.columns
    }

    required_columns = {"sample_id", "group1", "group2"}
    missing_columns = sorted(required_columns.difference(normalized_columns))
    if missing_columns:
        raise ValueError(
            "PCA group table must contain columns: sample_id, group1, group2. "
            f"Missing: {missing_columns}"
        )

    rename_map = {
        normalized_columns["sample_id"]: "sample_id",
        normalized_columns["group1"]: "group1",
        normalized_columns["group2"]: "group2",
    }
    source_columns = [
        normalized_columns["sample_id"],
        normalized_columns["group1"],
        normalized_columns["group2"],
    ]

    group_df = group_df.loc[:, source_columns].rename(columns=rename_map).copy()

    group_df["sample_id"] = group_df["sample_id"].astype(str).str.strip()
    group_df["group1"] = group_df["group1"].astype("string").str.strip().replace("", pd.NA)
    group_df["group2"] = group_df["group2"].astype("string").str.strip().replace("", pd.NA)

    valid_mask = group_df["sample_id"].ne("") & group_df["group1"].notna() & group_df["group2"].notna()
    dropped_rows = int((~valid_mask).sum())
    if dropped_rows > 0:
        logger.warning(
            "Dropped %d invalid rows from PCA group table because sample_id, group1, or group2 was empty.",
            dropped_rows,
        )
        group_df = group_df.loc[valid_mask].copy()

    if group_df.empty:
        logger.warning(
            "PCA group table is empty after removing invalid rows; PCA plots will be generated without grouping."
        )
        return None

    duplicated_mask = group_df["sample_id"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_ids = group_df.loc[duplicated_mask, "sample_id"].astype(str).unique().tolist()
        raise ValueError(
            "PCA group table contains duplicated sample_id values: "
            f"{duplicated_ids[:5]}"
        )

    column_order = ["sample_id", "group1", "group2"]
    group_df = group_df.loc[:, column_order].copy()
    group_df["_group_table_order"] = np.arange(len(group_df), dtype=int)

    group_df.attrs["source_path"] = str(group_table_path)
    logger.info(
        "Loaded PCA group table from %s with %d samples across %d primary groups%s.",
        group_table_path,
        len(group_df),
        group_df["group1"].nunique(),
        f" and {group_df['group2'].nunique()} subgroup labels" if "group2" in group_df.columns else "",
    )
    return group_df


def _prepare_grouped_pca_inputs(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    group_df: pd.DataFrame | None,
) -> tuple[np.ndarray, list[str], pd.DataFrame | None]:
    values = np.asarray(matrix, dtype=np.float32)
    samples = [str(name).strip() for name in sample_names]

    if group_df is None or "sample_id" not in group_df.columns:
        return values, samples, None

    sample_index = pd.Index(samples, dtype=str, name="sample_id")
    group_table = group_df.copy()
    group_table["sample_id"] = group_table["sample_id"].astype(str).str.strip()
    group_table = group_table.set_index("sample_id", drop=True)

    matched_mask = sample_index.isin(group_table.index)
    missing_samples = sample_index[~matched_mask].tolist()
    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        suffix = " ..." if len(missing_samples) > 10 else ""
        logger.warning(
            "[%s] Skipped %d samples without group annotation in %s: %s%s",
            title,
            len(missing_samples),
            Path(group_df.attrs.get("source_path", "group_table")).name,
            preview,
            suffix,
        )

    unused_group_rows = group_table.index.difference(sample_index, sort=False).tolist()
    if unused_group_rows:
        logger.info(
            "[%s] Ignored %d group table entries not present in the current matrix.",
            title,
            len(unused_group_rows),
        )

    filtered_samples = sample_index[matched_mask].tolist()
    filtered_matrix = values[matched_mask]
    if len(filtered_samples) == 0:
        logger.warning("[%s] No overlapping samples were found between the matrix and PCA group table.", title)
        return filtered_matrix, filtered_samples, None

    grouped_plot_df = group_table.reindex(filtered_samples).reset_index()
    if len(grouped_plot_df.columns) <= 1:
        return filtered_matrix, filtered_samples, None
    return filtered_matrix, filtered_samples, grouped_plot_df


def _has_secondary_grouping(group_df: pd.DataFrame | None) -> bool:
    if group_df is None or "group2" not in group_df.columns:
        return False
    return group_df["group2"].notna().any()


def _try_add_confidence_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    *,
    confidence: float = 0.95,
) -> bool:
    if points.shape[0] < 3:
        return False

    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return False

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues <= 0):
        return False

    scale = float(np.sqrt(chi2.ppf(confidence, df=2)))
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.18,
        zorder=2,
    )
    ax.add_patch(ellipse)
    return True


def _try_add_convex_hull(ax: plt.Axes, points: np.ndarray, color: str) -> bool:
    if points.shape[0] < 3:
        return False

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return False

    hull_points = points[hull.vertices]
    polygon = Polygon(
        hull_points,
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.18,
        zorder=2,
        joinstyle="round",
    )
    ax.add_patch(polygon)
    return True


def _add_group_envelope(ax: plt.Axes, points: np.ndarray, color: str, fallback_radius: float) -> None:
    if points.shape[0] >= 3 and _try_add_confidence_ellipse(ax, points, color):
        return
    if points.shape[0] >= 3 and _try_add_convex_hull(ax, points, color):
        return

    if points.shape[0] == 2:
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=10.0,
            alpha=0.14,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.3,
            alpha=0.95,
            zorder=2,
            solid_capstyle="round",
        )
        return

    if points.shape[0] == 1:
        circle = plt.Circle(
            (float(points[0, 0]), float(points[0, 1])),
            radius=float(fallback_radius),
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.18,
            zorder=2,
        )
        ax.add_patch(circle)


def _compute_pca_result(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    cfg,
    *,
    group_df: pd.DataFrame | None = None,
    max_components: int = 10,
) -> dict[str, object] | None:
    plot_matrix, plot_sample_names, plot_group_df = _prepare_grouped_pca_inputs(
        matrix=np.asarray(matrix, dtype=np.float32),
        sample_names=sample_names,
        title=title,
        group_df=group_df,
    )

    if plot_matrix.shape[0] < 2 or plot_matrix.shape[1] < 2:
        logger.warning("[%s] PCA was skipped because fewer than 2 samples or features remained for plotting.", title)
        return None

    n_components = min(int(max_components), int(plot_matrix.shape[0]), int(plot_matrix.shape[1]))
    if n_components < 2:
        logger.warning("[%s] PCA was skipped because fewer than 2 principal components were available.", title)
        return None

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    coords = pca.fit_transform(plot_matrix)
    var_exp = pca.explained_variance_ratio_ * 100.0
    return {
        "coords": coords,
        "var_exp": var_exp,
        "plot_sample_names": plot_sample_names,
        "plot_group_df": plot_group_df,
        "n_components": int(n_components),
    }


def _plot_pca_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
    *,
    group_df: pd.DataFrame | None = None,
    primary_group_col: str = "group1",
    secondary_group_col: str | None = None,
    add_group_envelope: bool = True,
    pca_result: dict[str, object] | None = None,
) -> None:
    if pca_result is None:
        pca_result = _compute_pca_result(
            matrix=np.asarray(matrix, dtype=np.float32),
            sample_names=sample_names,
            title=title,
            cfg=cfg,
            group_df=group_df,
            max_components=2,
        )
    if pca_result is None:
        return

    coords = np.asarray(pca_result["coords"], dtype=float)
    var_exp = np.asarray(pca_result["var_exp"], dtype=float)
    if coords.shape[1] < 2 or var_exp.size < 2:
        logger.warning("[%s] PCA plot was skipped because fewer than 2 principal components were available.", title)
        return

    coords = coords[:, :2]
    var_exp = var_exp[:2]
    plot_sample_names = list(pca_result["plot_sample_names"])
    plot_group_df = pca_result.get("plot_group_df")

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    legend_anchor = (1.02, 1.0)
    legend_marker_size = 7
    legend_fontsize = 10
    legend_handlelength = 0.8
    legend_handletextpad = 0.5
    legend_columnspacing = 1.0
    legend_labelspacing = 0.5

    if plot_group_df is None or primary_group_col not in plot_group_df.columns:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=42,
            alpha=0.90,
            color=PALETTE["pca_scatter"],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    else:
        plot_group_df = plot_group_df.copy()
        plot_group_df[primary_group_col] = plot_group_df[primary_group_col].astype(str).str.strip()
        plot_group_df["PC1"] = coords[:, 0]
        plot_group_df["PC2"] = coords[:, 1]

        span = max(
            float(np.ptp(coords[:, 0])) if coords.shape[0] > 0 else 0.0,
            float(np.ptp(coords[:, 1])) if coords.shape[0] > 0 else 0.0,
            1.0,
        )
        fallback_radius = 0.035 * span
        group_orders = plot_group_df["_group_table_order"].tolist() if "_group_table_order" in plot_group_df.columns else None
        primary_groups = _ordered_unique_with_order(
            plot_group_df[primary_group_col].tolist(),
            group_orders,
        )
        marker_map = _group_marker_map(primary_groups)

        has_secondary = (
            secondary_group_col is not None
            and secondary_group_col in plot_group_df.columns
            and plot_group_df[secondary_group_col].notna().any()
        )

        if has_secondary:
            subgroup_values = (
                plot_group_df[secondary_group_col].astype("string").fillna("").astype(str).str.strip()
            )
            subgroup_values = subgroup_values.where(subgroup_values.ne(""), "Missing")
            plot_group_df["_subgroup_label"] = subgroup_values

            secondary_groups, color_map = _global_secondary_group_color_map(
                plot_group_df["_subgroup_label"].astype(str).tolist(),
                group_orders,
            )

            for subgroup_name in secondary_groups:
                subgroup_df = plot_group_df.loc[plot_group_df["_subgroup_label"].astype(str) == subgroup_name]
                for primary_group_name in primary_groups:
                    group_points_df = subgroup_df.loc[
                        subgroup_df[primary_group_col].astype(str) == primary_group_name
                    ]
                    if group_points_df.empty:
                        continue

                    group_points = group_points_df.loc[:, ["PC1", "PC2"]].to_numpy(
                        dtype=float,
                        copy=False,
                    )
                    ax.scatter(
                        group_points[:, 0],
                        group_points[:, 1],
                        s=42,
                        alpha=0.90,
                        color=color_map[subgroup_name],
                        marker=marker_map[primary_group_name],
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=3,
                    )

            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in secondary_groups
            ]
            shape_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]

            legend_handles = color_handles + shape_handles
            total_legend_items = len(legend_handles)
            legend_ncol = 1 if total_legend_items <= 10 else 2 if total_legend_items <= 20 else 3

            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=legend_anchor,
                borderaxespad=0.0,
                ncol=legend_ncol,
                handlelength=legend_handlelength,
                handletextpad=legend_handletextpad,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                fontsize=legend_fontsize,
            )
        else:
            color_map = _group_color_map(plot_group_df[primary_group_col].tolist())
            primary_groups = _ordered_unique_nonempty(plot_group_df[primary_group_col].tolist())

            for group_name in primary_groups:
                group_points_df = plot_group_df.loc[plot_group_df[primary_group_col].astype(str) == group_name]
                group_points = group_points_df.loc[:, ["PC1", "PC2"]].to_numpy(dtype=float, copy=False)
                group_color = color_map[group_name]

                if add_group_envelope:
                    _add_group_envelope(ax, group_points, group_color, fallback_radius)

                ax.scatter(
                    group_points[:, 0],
                    group_points[:, 1],
                    s=42,
                    alpha=0.90,
                    color=group_color,
                    marker=marker_map[group_name],
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                    label=group_name,
                )

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=legend_anchor,
                borderaxespad=0.0,
                handlelength=legend_handlelength,
                handletextpad=legend_handletextpad,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                fontsize=legend_fontsize,
            )

    if len(plot_sample_names) <= 20 and adjust_text is not None:
        texts = [
            ax.text(x, y, label, fontsize=8, alpha=0.90)
            for x, y, label in zip(coords[:, 0], coords[:, 1], plot_sample_names)
        ]
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grid_aux"], "lw": 0.5})

    ax.axhline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.4,
        colors="black",
    )
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    _save_figure(fig, save_stem, cfg)


def _plot_pca_pairs_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
    *,
    group_df: pd.DataFrame | None = None,
    primary_group_col: str = "group1",
    secondary_group_col: str | None = None,
    max_components: int = 10,
    pca_result: dict[str, object] | None = None,
) -> None:
    if pca_result is None:
        pca_result = _compute_pca_result(
            matrix=np.asarray(matrix, dtype=np.float32),
            sample_names=sample_names,
            title=title,
            cfg=cfg,
            group_df=group_df,
            max_components=max_components,
        )
    if pca_result is None:
        return

    coords = np.asarray(pca_result["coords"], dtype=float)
    var_exp = np.asarray(pca_result["var_exp"], dtype=float)
    n_components = min(int(max_components), int(coords.shape[1]), int(var_exp.size))
    if n_components < 2:
        logger.warning("[%s] PCA pairs plot was skipped because fewer than 2 principal components were available.", title)
        return

    coords = coords[:, :n_components]
    var_exp = var_exp[:n_components]
    plot_group_df = pca_result.get("plot_group_df")

    panel_size = 1.12
    fig_size = max(7.8, panel_size * n_components + 1.8)
    fig, axes = plt.subplots(n_components, n_components, figsize=(fig_size, fig_size))
    axes = np.asarray(axes, dtype=object)

    legend_handles: list[Line2D] = []
    ordered_groups: list[str] = []
    color_map: dict[str, str] = {}
    marker_map: dict[str, str] = {}
    secondary_groups: list[str] = []
    has_secondary = False

    if plot_group_df is not None and primary_group_col in plot_group_df.columns:
        plot_group_df = plot_group_df.copy()
        plot_group_df[primary_group_col] = plot_group_df[primary_group_col].astype(str).str.strip()
        group_orders = plot_group_df["_group_table_order"].tolist() if "_group_table_order" in plot_group_df.columns else None
        ordered_groups = _ordered_unique_with_order(
            plot_group_df[primary_group_col].tolist(),
            group_orders,
        )
        marker_map = _group_marker_map(ordered_groups)

        has_secondary = (
            secondary_group_col is not None
            and secondary_group_col in plot_group_df.columns
            and plot_group_df[secondary_group_col].notna().any()
        )

        if has_secondary:
            subgroup_values = (
                plot_group_df[secondary_group_col].astype("string").fillna("").astype(str).str.strip()
            )
            subgroup_values = subgroup_values.where(subgroup_values.ne(""), "Missing")
            plot_group_df["_subgroup_label"] = subgroup_values

            secondary_groups, color_map = _global_secondary_group_color_map(
                plot_group_df["_subgroup_label"].astype(str).tolist(),
                group_orders,
            )

            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=9.5,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in secondary_groups
            ]
            shape_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=9.5,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in ordered_groups
            ]
            legend_handles = color_handles + shape_handles
        else:
            color_map = _group_color_map(ordered_groups)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=9.5,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in ordered_groups
            ]

    for row_idx in range(n_components):
        for col_idx in range(n_components):
            ax = axes[row_idx, col_idx]

            if col_idx < row_idx:
                ax.axis("off")
                continue

            if col_idx == row_idx:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.56,
                    f"PC{row_idx + 1},\n{var_exp[row_idx]:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
                continue

            x = coords[:, col_idx]
            y = coords[:, row_idx]
            ax.axhline(0, color="#7a7a7a", linewidth=0.55, linestyle=(0, (6, 4)), zorder=1)
            ax.axvline(0, color="#7a7a7a", linewidth=0.55, linestyle=(0, (6, 4)), zorder=1)

            if ordered_groups:
                if has_secondary:
                    for subgroup_name in secondary_groups:
                        subgroup_df = plot_group_df.loc[plot_group_df["_subgroup_label"].astype(str) == subgroup_name]
                        for primary_group_name in ordered_groups:
                            group_points_df = subgroup_df.loc[
                                subgroup_df[primary_group_col].astype(str) == primary_group_name
                            ]
                            if group_points_df.empty:
                                continue
                            mask = group_points_df.index.to_numpy(dtype=int, copy=False)
                            ax.scatter(
                                x[mask],
                                y[mask],
                                s=10,
                                alpha=0.85,
                                color=color_map[subgroup_name],
                                marker=marker_map[primary_group_name],
                                edgecolors="white",
                                linewidths=0.45,
                                zorder=2,
                            )
                else:
                    for group_name in ordered_groups:
                        mask = plot_group_df[primary_group_col].astype(str).eq(group_name).to_numpy(dtype=bool, copy=False)
                        if not np.any(mask):
                            continue
                        ax.scatter(
                            x[mask],
                            y[mask],
                            s=9,
                            alpha=0.85,
                            color=color_map[group_name],
                            marker=marker_map.get(group_name, "o"),
                            edgecolors="white",
                            linewidths=0.45,
                            zorder=2,
                        )
            else:
                ax.scatter(
                    x,
                    y,
                    s=9,
                    alpha=0.85,
                    color=PALETTE["pca_scatter"],
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=2,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#6b7280")
            ax.set_facecolor("white")

    fig.suptitle(title, fontsize=23, fontweight="bold", y=0.985)
    if legend_handles:
        legend_ncol = 1 if len(legend_handles) <= 12 else 2 if len(legend_handles) <= 24 else 3
        fig.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.055, 0.06),
            ncol=legend_ncol,
            frameon=False,
            fontsize=13,
            handletextpad=0.55,
            columnspacing=1.15,
            labelspacing=0.62,
            borderaxespad=0.0,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.06,
        top=0.93,
        wspace=0.14,
        hspace=0.14,
    )
    _save_figure(fig, save_stem, cfg)


def plot_sample_dendrogram(adata, save_stem: str | Path, cfg) -> None:
    if adata.n_obs < 2:
        return

    from scipy.cluster.hierarchy import dendrogram, linkage

    fig_width = max(12.0, adata.n_obs * 0.24)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    linkage_matrix = linkage(np.asarray(adata.X, dtype=np.float32), method="average")
    color_threshold = 0.70 * float(linkage_matrix[:, 2].max())

    dendrogram(
        linkage_matrix,
        labels=adata.obs_names.tolist(),
        leaf_rotation=90,
        leaf_font_size=max(4, min(9, int(fig_width * 72 / max(1, adata.n_obs) * 0.42))),
        color_threshold=color_threshold,
        above_threshold_color="#aaaaaa",
        ax=ax,
    )
    ax.set_title("Sample Clustering Dendrogram")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Euclidean Distance")
    _save_figure(fig, save_stem, cfg)


def plot_transcriptome_pca(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col=None,
        add_group_envelope=True,
        pca_result=pca_result,
    )


def plot_metabolome_pca(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col=None,
        add_group_envelope=True,
        pca_result=pca_result,
    )


def plot_transcriptome_pca_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        add_group_envelope=False,
        pca_result=pca_result,
    )


def plot_metabolome_pca_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        add_group_envelope=False,
        pca_result=pca_result,
    )


def plot_transcriptome_pca_pairs(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    _plot_pca_pairs_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        pca_result=pca_result,
    )


def plot_metabolome_pca_pairs(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_pairs_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        pca_result=pca_result,
    )


def plot_transcriptome_pca_pairs_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    _plot_pca_pairs_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        pca_result=pca_result,
    )


def plot_metabolome_pca_pairs_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None, pca_result: dict[str, object] | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_pairs_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        pca_result=pca_result,
    )


__all__ = [
    "_load_pca_group_table",
    "_has_secondary_grouping",
    "_compute_pca_result",
    "plot_sample_dendrogram",
    "plot_transcriptome_pca",
    "plot_metabolome_pca",
    "plot_transcriptome_pca_subgroups",
    "plot_metabolome_pca_subgroups",
    "plot_transcriptome_pca_pairs",
    "plot_metabolome_pca_pairs",
    "plot_transcriptome_pca_pairs_subgroups",
    "plot_metabolome_pca_pairs_subgroups",
]
