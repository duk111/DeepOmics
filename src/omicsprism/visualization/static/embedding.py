from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
from .pca import _prepare_grouped_pca_inputs


def _metabolome_matrix_from_adata(adata) -> np.ndarray:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        return metab_df.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(metab_df, dtype=np.float32)


def _pca_prefilter_matrix(matrix: np.ndarray, cfg, max_components: int = 50) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = values.shape
    n_components = min(int(max_components), max(1, n_samples - 1), int(n_features))
    if n_components < 2 or n_features <= n_components:
        return values

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    return pca.fit_transform(values).astype(np.float32)


def _compute_embedding_result(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    cfg,
    *,
    method: str,
    group_df: pd.DataFrame | None = None,
) -> dict[str, object] | None:
    plot_matrix, plot_sample_names, plot_group_df = _prepare_grouped_pca_inputs(
        matrix=np.asarray(matrix, dtype=np.float32),
        sample_names=sample_names,
        title=title,
        group_df=group_df,
    )
    if plot_matrix.shape[0] < 3 or plot_matrix.shape[1] < 2:
        logger.warning("[%s] %s was skipped because fewer than 3 samples or 2 features remained.", title, method)
        return None

    reduced_matrix = _pca_prefilter_matrix(plot_matrix, cfg)
    method_key = str(method).lower()
    if method_key == "umap":
        try:
            import umap as umap_module
        except ImportError:
            logger.warning("[%s] UMAP was skipped because the optional umap-learn dependency is not installed.", title)
            return None

        n_neighbors = min(15, max(2, reduced_matrix.shape[0] - 1))
        reducer = umap_module.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=cfg.random_state,
        )
        coords = reducer.fit_transform(reduced_matrix)
        axis_prefix = "UMAP"
    elif method_key in {"tsne", "t-sne"}:
        perplexity = min(30.0, max(2.0, float((reduced_matrix.shape[0] - 1) // 3)))
        perplexity = min(perplexity, float(reduced_matrix.shape[0] - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=cfg.random_state,
        )
        coords = reducer.fit_transform(reduced_matrix)
        axis_prefix = "t-SNE"
    else:
        raise ValueError("method must be either 'umap' or 'tsne'.")

    return {
        "coords": np.asarray(coords, dtype=float),
        "plot_sample_names": plot_sample_names,
        "plot_group_df": plot_group_df,
        "axis_prefix": axis_prefix,
    }


def _plot_embedding_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
    *,
    method: str,
    group_df: pd.DataFrame | None = None,
    primary_group_col: str = "group1",
    secondary_group_col: str | None = None,
) -> None:
    result = _compute_embedding_result(
        matrix=matrix,
        sample_names=sample_names,
        title=title,
        cfg=cfg,
        method=method,
        group_df=group_df,
    )
    if result is None:
        return

    coords = np.asarray(result["coords"], dtype=float)
    if coords.shape[1] < 2:
        return

    plot_group_df = result.get("plot_group_df")
    axis_prefix = str(result.get("axis_prefix", method.upper()))

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    legend_handles: list[Line2D] = []

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
        group_orders = plot_group_df["_group_table_order"].tolist() if "_group_table_order" in plot_group_df.columns else None
        primary_groups = _ordered_unique_with_order(plot_group_df[primary_group_col].tolist(), group_orders)
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
                    mask = (
                        subgroup_df[primary_group_col].astype(str).eq(str(primary_group_name)).to_numpy(dtype=bool)
                    )
                    if not np.any(mask):
                        continue
                    group_points_df = subgroup_df.loc[mask]
                    positions = group_points_df.index.to_numpy(dtype=int, copy=False)
                    ax.scatter(
                        coords[positions, 0],
                        coords[positions, 1],
                        s=42,
                        alpha=0.90,
                        color=color_map[subgroup_name],
                        marker=marker_map[primary_group_name],
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=3,
                    )

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in secondary_groups
            ] + [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=7,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]
        else:
            primary_groups = _ordered_unique_nonempty(plot_group_df[primary_group_col].tolist())
            color_map = _group_color_map(primary_groups)
            marker_map = _group_marker_map(primary_groups)
            for group_name in primary_groups:
                mask = plot_group_df[primary_group_col].astype(str).eq(str(group_name)).to_numpy(dtype=bool)
                if not np.any(mask):
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=42,
                    alpha=0.90,
                    color=color_map[group_name],
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
                    markersize=7,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]

    if legend_handles:
        legend_ncol = 1 if len(legend_handles) <= 10 else 2 if len(legend_handles) <= 20 else 3
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=legend_ncol,
            handlelength=0.8,
            handletextpad=0.5,
            columnspacing=1.0,
            labelspacing=0.5,
            fontsize=10,
        )

    ax.axhline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.tick_params(axis="both", direction="out", length=5, width=1.2, colors="black")
    ax.set_title(title)
    ax.set_xlabel(f"{axis_prefix} 1")
    ax.set_ylabel(f"{axis_prefix} 2")
    _save_figure(fig, save_stem, cfg)


def plot_transcriptome_umap(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        np.asarray(adata.X, dtype=np.float32),
        adata.obs_names.astype(str).tolist(),
        "Transcriptome UMAP",
        save_stem,
        cfg,
        method="umap",
        group_df=group_df,
        primary_group_col="group1",
    )


def plot_transcriptome_umap_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        np.asarray(adata.X, dtype=np.float32),
        adata.obs_names.astype(str).tolist(),
        "Transcriptome UMAP",
        save_stem,
        cfg,
        method="umap",
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


def plot_metabolome_umap(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        _metabolome_matrix_from_adata(adata),
        adata.obs_names.astype(str).tolist(),
        "Metabolome UMAP",
        save_stem,
        cfg,
        method="umap",
        group_df=group_df,
        primary_group_col="group1",
    )


def plot_metabolome_umap_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        _metabolome_matrix_from_adata(adata),
        adata.obs_names.astype(str).tolist(),
        "Metabolome UMAP",
        save_stem,
        cfg,
        method="umap",
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


def plot_transcriptome_tsne(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        np.asarray(adata.X, dtype=np.float32),
        adata.obs_names.astype(str).tolist(),
        "Transcriptome t-SNE",
        save_stem,
        cfg,
        method="tsne",
        group_df=group_df,
        primary_group_col="group1",
    )


def plot_transcriptome_tsne_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        np.asarray(adata.X, dtype=np.float32),
        adata.obs_names.astype(str).tolist(),
        "Transcriptome t-SNE",
        save_stem,
        cfg,
        method="tsne",
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


def plot_metabolome_tsne(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        _metabolome_matrix_from_adata(adata),
        adata.obs_names.astype(str).tolist(),
        "Metabolome t-SNE",
        save_stem,
        cfg,
        method="tsne",
        group_df=group_df,
        primary_group_col="group1",
    )


def plot_metabolome_tsne_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_embedding_from_matrix(
        _metabolome_matrix_from_adata(adata),
        adata.obs_names.astype(str).tolist(),
        "Metabolome t-SNE",
        save_stem,
        cfg,
        method="tsne",
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


__all__ = [
    "plot_transcriptome_umap",
    "plot_transcriptome_umap_subgroups",
    "plot_metabolome_umap",
    "plot_metabolome_umap_subgroups",
    "plot_transcriptome_tsne",
    "plot_transcriptome_tsne_subgroups",
    "plot_metabolome_tsne",
    "plot_metabolome_tsne_subgroups",
]
