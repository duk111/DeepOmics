from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .base import (
    _gene_expression_df,
    _global_secondary_group_color_map,
    _group_color_map,
    _ordered_unique_with_order,
    _save_figure,
)

def _significance_star(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    if value <= 0.05:
        return "*"
    return ""


def _shared_sample_id_field(sample_ids: list[str], fallback: str) -> str:
    clean_ids = [str(sample_id).strip() for sample_id in sample_ids if str(sample_id).strip()]
    if not clean_ids:
        return str(fallback)
    if len(clean_ids) == 1:
        return clean_ids[0]

    split_tokens = [re.split(r"[^A-Za-z0-9]+", sample_id) for sample_id in clean_ids]
    min_len = min(len(tokens) for tokens in split_tokens)
    shared_tokens = []
    for idx in range(min_len):
        token = split_tokens[0][idx]
        if token and all(tokens[idx] == token for tokens in split_tokens[1:]):
            shared_tokens.append(token)
        else:
            break
    if shared_tokens:
        return "_".join(shared_tokens)

    common_prefix = clean_ids[0]
    for sample_id in clean_ids[1:]:
        while common_prefix and not sample_id.startswith(common_prefix):
            common_prefix = common_prefix[:-1]
    common_prefix = re.sub(r"[^A-Za-z0-9]+$", "", common_prefix.strip())
    if common_prefix:
        return common_prefix
    return str(fallback)


def _coerce_module_eigengene_df(module_eigengenes_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(module_eigengenes_df, pd.DataFrame) or module_eigengenes_df.empty:
        return pd.DataFrame()

    work = module_eigengenes_df.copy()
    sample_column = None
    for column in work.columns:
        normalized = str(column).replace("\ufeff", "").strip().lower()
        if normalized in {"sampleid", "sample_id", "sample"}:
            sample_column = column
            break

    if sample_column is not None:
        work = work.set_index(sample_column, drop=True)
    elif isinstance(work.index, pd.RangeIndex) and work.shape[1] > 1:
        first_column = work.columns[0]
        first_values = work[first_column].astype(str).str.strip()
        numeric_rest = work.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if first_values.ne("").any() and numeric_rest.notna().any().any():
            work = work.set_index(first_column, drop=True)

    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work = work.loc[work.index.astype(str).str.len() > 0].copy()
    work = work.loc[~work.index.duplicated(keep="first")].copy()

    numeric_df = work.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)].copy()
    numeric_df.columns = numeric_df.columns.astype(str)
    return numeric_df


def _build_group_annotation_candidates(group_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_df is None or group_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_columns = {"sample_id", "group1", "group2"}
    if not required_columns.issubset(group_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    keep_columns = [col for col in ["sample_id", "group1", "group2", "_group_table_order"] if col in group_df.columns]
    work = group_df.loc[:, keep_columns].copy()
    work["sample_id"] = work["sample_id"].astype(str).str.strip()
    work["group1"] = work["group1"].astype("string").str.strip().replace("", pd.NA)
    work["group2"] = work["group2"].astype("string").str.strip().replace("", pd.NA)
    if "_group_table_order" not in work.columns:
        work["_group_table_order"] = np.arange(len(work), dtype=int)
    work["_group_table_order"] = pd.to_numeric(work["_group_table_order"], errors="coerce")

    valid_mask = work["sample_id"].ne("") & work["group1"].notna() & work["group2"].notna()
    work = work.loc[valid_mask].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    exact = (
        work.sort_values("_group_table_order", kind="mergesort")
        .drop_duplicates(subset=["sample_id"], keep="first")
        .set_index("sample_id", drop=False)
    )

    aggregated_rows: list[dict[str, object]] = []
    for (group1, group2), sub_df in work.groupby(["group1", "group2"], sort=False):
        member_ids = sub_df["sample_id"].astype(str).tolist()
        fallback_id = f"{group1}_{group2}"
        sample_id = _shared_sample_id_field(member_ids, fallback=fallback_id)
        aggregated_rows.append(
            {
                "sample_id": sample_id,
                "group1": str(group1),
                "group2": str(group2),
                "_group_table_order": int(sub_df["_group_table_order"].min()),
            }
        )

    if not aggregated_rows:
        return exact, pd.DataFrame()

    output_ids = [str(row["sample_id"]) for row in aggregated_rows]
    if len(output_ids) != len(set(output_ids)):
        counts: dict[str, int] = {}
        for row in aggregated_rows:
            base_id = str(row["sample_id"])
            counts[base_id] = counts.get(base_id, 0) + 1
            if counts[base_id] > 1:
                row["sample_id"] = f"{base_id}_{counts[base_id]}"

    aggregated = pd.DataFrame(aggregated_rows).set_index("sample_id", drop=False)
    return exact, aggregated


def _align_group_annotations_to_samples(sample_names: list[str], group_df: pd.DataFrame | None) -> pd.DataFrame:
    exact, aggregated = _build_group_annotation_candidates(group_df)

    rows: list[dict[str, object]] = []
    missing_samples: list[str] = []
    for sample_idx, sample_name in enumerate(sample_names):
        sample_id = str(sample_name).strip()
        if not exact.empty and sample_id in exact.index:
            row = exact.loc[sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "group1": str(row["group1"]),
                    "group2": str(row["group2"]),
                    "_group_table_order": int(row["_group_table_order"]),
                    "_original_sample_order": sample_idx,
                    "_has_group_annotation": True,
                }
            )
        elif not aggregated.empty and sample_id in aggregated.index:
            row = aggregated.loc[sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "group1": str(row["group1"]),
                    "group2": str(row["group2"]),
                    "_group_table_order": int(row["_group_table_order"]),
                    "_original_sample_order": sample_idx,
                    "_has_group_annotation": True,
                }
            )
        else:
            missing_samples.append(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "group1": "Missing",
                    "group2": "Missing",
                    "_group_table_order": len(sample_names) + sample_idx,
                    "_original_sample_order": sample_idx,
                    "_has_group_annotation": False,
                }
            )

    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        suffix = " ..." if len(missing_samples) > 10 else ""
        logger.warning(
            "[Module eigengene heatmap] %d samples had no group annotation: %s%s",
            len(missing_samples),
            preview,
            suffix,
        )

    annotation_df = pd.DataFrame(rows)
    if annotation_df.empty:
        return annotation_df
    return annotation_df.set_index("sample_id", drop=False)


def _module_order_from_summary(module_summary_df: pd.DataFrame, available_modules: list[str]) -> list[str]:
    available_set = set(str(module) for module in available_modules)
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        ordered = [
            str(module_name)
            for module_name in module_summary_df["Module"].astype(str).tolist()
            if str(module_name) in available_set
        ]
    else:
        ordered = []

    for module_name in available_modules:
        module_name = str(module_name)
        if module_name not in ordered:
            ordered.append(module_name)
    return ordered


def _row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float, copy=True)
    row_mean = np.nanmean(values, axis=1, keepdims=True)
    row_std = np.nanstd(values, axis=1, ddof=0, keepdims=True)
    row_std = np.where(np.isfinite(row_std) & (row_std > 0.0), row_std, 1.0)
    z_values = (values - row_mean) / row_std
    z_values = np.where(np.isfinite(z_values), z_values, np.nan)
    return pd.DataFrame(z_values, index=df.index.copy(), columns=df.columns.copy())


def _add_group_annotation_bar(
    ax: plt.Axes,
    labels: list[str],
    color_map: dict[str, str],
    *,
    row_label: str | None = None,
    missing_color: str = "#d1d5db",
) -> None:
    if not labels:
        ax.axis("off")
        return

    rgba = np.array([[colors.to_rgba(color_map.get(str(label), missing_color)) for label in labels]], dtype=float)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    if row_label:
        ax.set_yticks([0])
        ax.set_yticklabels([row_label], fontsize=9)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_heatmap_group_separators(axes: list[plt.Axes], group_labels: list[str]) -> None:
    if len(group_labels) < 2:
        return

    for idx in range(1, len(group_labels)):
        if str(group_labels[idx]) == str(group_labels[idx - 1]):
            continue
        for ax in axes:
            ax.axvline(idx - 0.5, color="white", linewidth=4.0, zorder=5)


def _legend_column_count(n_items: int, fig_width: float) -> int:
    if n_items <= 0:
        return 1
    return min(n_items, max(1, int(fig_width // 0.78)))


def _group_color_orders_from_table(
    group_df: pd.DataFrame | None,
) -> tuple[list[str], dict[str, str], list[str], dict[str, str]]:
    exact, _ = _build_group_annotation_candidates(group_df)
    if exact.empty:
        return [], {}, [], {}

    color_source = exact.sort_values("_group_table_order", kind="mergesort")
    color_orders = color_source["_group_table_order"].astype(int).tolist()

    group1_order = _ordered_unique_with_order(color_source["group1"].astype(str).tolist(), color_orders)
    group1_color_map = _group_color_map(group1_order)

    group2_order, group2_color_map = _global_secondary_group_color_map(
        color_source["group2"].astype(str).tolist(),
        color_orders,
    )
    return group1_order, group1_color_map, group2_order, group2_color_map


def _module_group_orders_and_colors(
    group_df: pd.DataFrame | None,
    group1_labels: list[str],
    group2_labels: list[str],
    group_orders: list[int] | None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    used_group1_order = _ordered_unique_with_order(group1_labels, group_orders)
    used_group2_order = _ordered_unique_with_order(group2_labels, group_orders)
    full_group1_order, group1_color_map, full_group2_order, group2_color_map = _group_color_orders_from_table(group_df)

    if not full_group1_order:
        full_group1_order = used_group1_order
        group1_color_map = _group_color_map(full_group1_order)
    if not full_group2_order:
        full_group2_order, group2_color_map = _global_secondary_group_color_map(group2_labels, group_orders)

    group1_order = [group_name for group_name in full_group1_order if group_name in set(used_group1_order)]
    group2_order = [group_name for group_name in full_group2_order if group_name in set(used_group2_order)]
    for group_name in used_group1_order:
        if group_name not in group1_order:
            group1_order.append(group_name)
    for group_name in used_group2_order:
        if group_name not in group2_order:
            group2_order.append(group_name)

    if "Missing" in used_group1_order:
        group1_color_map["Missing"] = "#d1d5db"
    if "Missing" in used_group2_order:
        group2_color_map["Missing"] = "#d1d5db"

    return (
        {"group1": group1_order, "group2": group2_order},
        {"group1": group1_color_map, "group2": group2_color_map},
    )


def _sort_module_eigengene_samples(
    eigengenes_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    *,
    block_group_col: str,
    group_orders_by_col: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if annotation_df.empty:
        return eigengenes_df, annotation_df

    inner_group_col = "group2" if block_group_col == "group1" else "group1"
    block_order = {group_name: idx for idx, group_name in enumerate(group_orders_by_col.get(block_group_col, []))}
    inner_order = {group_name: idx for idx, group_name in enumerate(group_orders_by_col.get(inner_group_col, []))}

    sort_df = annotation_df.loc[:, ["_has_group_annotation", "_group_table_order", "_original_sample_order"]].copy()
    sort_df["_missing_rank"] = np.where(sort_df["_has_group_annotation"], 0, 1)
    sort_df["_block_rank"] = [
        block_order.get(str(label), len(block_order) + idx)
        for idx, label in enumerate(annotation_df[block_group_col].astype(str).tolist())
    ]
    sort_df["_inner_rank"] = [
        inner_order.get(str(label), len(inner_order) + idx)
        for idx, label in enumerate(annotation_df[inner_group_col].astype(str).tolist())
    ]

    sorted_samples = (
        sort_df.sort_values(
            ["_missing_rank", "_block_rank", "_inner_rank", "_group_table_order", "_original_sample_order"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        .index.astype(str)
        .tolist()
    )
    return eigengenes_df.reindex(sorted_samples), annotation_df.reindex(sorted_samples)


def _plot_module_eigengene_heatmap_variant(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
    *,
    top_group_col: str,
    bottom_group_col: str,
    block_group_col: str,
) -> None:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    eigengenes_df = eigengenes_df.loc[:, [module for module in module_order if module in eigengenes_df.columns]].copy()
    if eigengenes_df.empty:
        return

    sample_names = eigengenes_df.index.astype(str).tolist()
    annotation_df = _align_group_annotations_to_samples(sample_names, group_df)

    group_orders = annotation_df["_group_table_order"].astype(int).tolist() if not annotation_df.empty else None
    group1_labels_unsorted = annotation_df["group1"].astype(str).tolist() if not annotation_df.empty else ["Missing"] * eigengenes_df.shape[0]
    group2_labels_unsorted = annotation_df["group2"].astype(str).tolist() if not annotation_df.empty else ["Missing"] * eigengenes_df.shape[0]
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        group1_labels_unsorted,
        group2_labels_unsorted,
        group_orders,
    )

    eigengenes_df, annotation_df = _sort_module_eigengene_samples(
        eigengenes_df,
        annotation_df,
        block_group_col=block_group_col,
        group_orders_by_col=group_orders_by_col,
    )

    heatmap_df = _row_zscore(eigengenes_df.T)
    if heatmap_df.empty:
        return

    if not annotation_df.empty:
        labels_by_col = {
            "group1": annotation_df["group1"].astype(str).tolist(),
            "group2": annotation_df["group2"].astype(str).tolist(),
        }
    else:
        labels_by_col = {
            "group1": ["Missing"] * heatmap_df.shape[1],
            "group2": ["Missing"] * heatmap_df.shape[1],
        }

    top_order = group_orders_by_col.get(top_group_col, [])
    bottom_order = group_orders_by_col.get(bottom_group_col, [])
    top_color_map = color_maps_by_col.get(top_group_col, {})
    bottom_color_map = color_maps_by_col.get(bottom_group_col, {})

    n_modules, n_samples = heatmap_df.shape
    fig_width = max(8.5, min(28.0, 0.18 * max(1, n_samples) + 3.6))
    fig_height = max(5.8, min(18.0, 0.28 * max(1, n_modules) + 3.4))

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    heatmap_height = max(3.0, 0.28 * max(1, n_modules))
    legend_height = 0.72 if len(top_order) + len(bottom_order) <= 20 else 0.96
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[heatmap_height, 0.16, 0.16, 0.36, legend_height],
        hspace=0.08,
    )

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_top_group = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    ax_bottom_group = fig.add_subplot(gs[2, 0], sharex=ax_heatmap)
    cbar_grid = gs[3, 0].subgridspec(1, 3, width_ratios=[1.0, 0.42, 1.0])
    cbar_ax = fig.add_subplot(cbar_grid[0, 1])
    legend_ax = fig.add_subplot(gs[4, 0])
    legend_ax.axis("off")

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f3f4f6")
    heatmap_values = np.ma.masked_invalid(heatmap_df.to_numpy(dtype=float, copy=False))
    image = ax_heatmap.imshow(
        heatmap_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-1.5,
        vmax=1.5,
    )

    ax_heatmap.text(
        -0.065,
        1.025,
        "a",
        transform=ax_heatmap.transAxes,
        ha="left",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="black",
    )
    ax_heatmap.set_ylabel("Module")
    if n_modules <= 60:
        y_ticks = np.arange(n_modules)
    else:
        step = int(np.ceil(n_modules / 60.0))
        y_ticks = np.arange(0, n_modules, step)
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels([heatmap_df.index[idx] for idx in y_ticks], rotation=0)
    ax_heatmap.set_xticks(np.arange(n_samples) if n_samples <= 90 else [])
    ax_heatmap.set_xticklabels([])
    ax_heatmap.tick_params(axis="x", length=2.5, width=0.7, bottom=True, labelbottom=False)
    ax_heatmap.tick_params(axis="y", length=3.0, width=0.8)
    for spine in ax_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#6b7280")

    _add_group_annotation_bar(ax_top_group, labels_by_col[top_group_col], top_color_map)
    _add_group_annotation_bar(ax_bottom_group, labels_by_col[bottom_group_col], bottom_color_map)
    _add_heatmap_group_separators([ax_heatmap, ax_top_group, ax_bottom_group], labels_by_col[block_group_col])

    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-1.5, 0.0, 1.5])
    cbar.set_ticklabels(["< -1.5", "0", "> 1.5"])
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=0, labelsize=9)
    cbar_ax.text(
        -0.08,
        0.5,
        "z score",
        transform=cbar_ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )

    top_handles = [
        Patch(facecolor=top_color_map[group_name], edgecolor="none", label=group_name)
        for group_name in top_order
    ]
    bottom_handles = [
        Patch(facecolor=bottom_color_map[group_name], edgecolor="none", label=group_name)
        for group_name in bottom_order
    ]

    if top_handles:
        top_legend = legend_ax.legend(
            handles=top_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.80),
            ncol=_legend_column_count(len(top_handles), fig_width),
            frameon=False,
            handlelength=1.4,
            handleheight=0.8,
            handletextpad=0.45,
            columnspacing=0.9,
            labelspacing=0.45,
            borderaxespad=0.0,
            fontsize=9,
        )
        legend_ax.add_artist(top_legend)

    if bottom_handles:
        legend_ax.legend(
            handles=bottom_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.18),
            ncol=_legend_column_count(len(bottom_handles), fig_width),
            frameon=False,
            handlelength=1.4,
            handleheight=0.8,
            handletextpad=0.45,
            columnspacing=0.9,
            labelspacing=0.45,
            borderaxespad=0.0,
            fontsize=9,
        )

    left_margin = 0.14 if n_modules < 40 else 0.17
    fig.subplots_adjust(left=left_margin, right=0.985, top=0.985, bottom=0.045)
    _save_figure(fig, save_stem, cfg)


def plot_module_eigengene_heatmap(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    _plot_module_eigengene_heatmap_variant(
        engine,
        save_stem,
        cfg,
        group_df=group_df,
        top_group_col="group2",
        bottom_group_col="group1",
        block_group_col="group1",
    )


def plot_module_eigengene_heatmap_group2(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    _plot_module_eigengene_heatmap_variant(
        engine,
        save_stem,
        cfg,
        group_df=group_df,
        top_group_col="group1",
        bottom_group_col="group2",
        block_group_col="group2",
    )


def _add_group2_color_strip(
    ax: plt.Axes,
    group2_order: list[str],
    group2_color_map: dict[str, str],
    *,
    missing_color: str = "#d1d5db",
) -> None:
    if not group2_order:
        ax.axis("off")
        return

    rgba = np.array(
        [[colors.to_rgba(group2_color_map.get(str(group_name), missing_color)) for group_name in group2_order]],
        dtype=float,
    )
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_xlim(-0.5, len(group2_order) - 0.5)
    ax.set_xticks(np.arange(len(group2_order)))
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _centered_point_offsets(n_points: int, width: float = 0.14) -> np.ndarray:
    if n_points <= 1:
        return np.zeros(max(1, int(n_points)), dtype=float)
    return np.linspace(-float(width), float(width), int(n_points), dtype=float)


def plot_module_zscore_line_panels(
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
    eigengenes_df = eigengenes_df.loc[:, [module for module in module_order if module in eigengenes_df.columns]].copy()
    if eigengenes_df.empty:
        return

    zscore_df = _row_zscore(eigengenes_df.T)
    if zscore_df.empty:
        return
    module_color_map = _module_color_map_from_results(
        engine,
        zscore_df.index.astype(str).tolist(),
    )

    sample_names = zscore_df.columns.astype(str).tolist()
    annotation_df = _align_group_annotations_to_samples(sample_names, group_df)
    if annotation_df.empty:
        return

    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation_df["group1"].astype(str).tolist(),
        annotation_df["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = group_orders_by_col.get("group1", [])
    group2_order = group_orders_by_col.get("group2", [])
    if not group1_order or not group2_order:
        return

    group1_color_map = color_maps_by_col.get("group1", {})
    group2_color_map = color_maps_by_col.get("group2", {})

    annotation_df = annotation_df.reindex(sample_names)
    sample_groups: dict[tuple[str, str], list[str]] = {}
    for group1_name in group1_order:
        for group2_name in group2_order:
            mask = (
                annotation_df["group1"].astype(str).eq(str(group1_name))
                & annotation_df["group2"].astype(str).eq(str(group2_name))
            )
            sample_groups[(str(group1_name), str(group2_name))] = annotation_df.index[mask].astype(str).tolist()

    finite_values = zscore_df.to_numpy(dtype=float, copy=False)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return
    y_min = float(np.nanmin(finite_values))
    y_max = float(np.nanmax(finite_values))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    y_pad = max(0.15, 0.08 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    n_modules = len(zscore_df.index)
    n_group1 = len(group1_order)
    n_group2 = len(group2_order)
    panel_width = max(1.10, min(1.65, 0.045 * n_group2 + 0.85))
    panel_height = 0.76 if n_modules <= 10 else 0.62
    fig_width = max(6.2, min(18.0, panel_width * n_group1 + 1.35))
    fig_height = max(4.8, min(28.0, panel_height * n_modules + 1.45))

    legend_ncol = _legend_column_count(len(group2_order), fig_width)
    legend_rows = int(np.ceil(len(group2_order) / max(1, legend_ncol)))
    bottom_margin = max(0.12, min(0.24, 0.075 + 0.035 * legend_rows))

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    fig._png_dpi = 180
    gs = fig.add_gridspec(
        nrows=n_modules + 1,
        ncols=n_group1,
        height_ratios=[1.0] * n_modules + [0.12],
        hspace=0.12,
        wspace=0.18,
    )

    x_positions = np.arange(n_group2, dtype=float)
    for row_idx, module_name in enumerate(zscore_df.index.astype(str).tolist()):
        module_values = zscore_df.loc[module_name]
        for col_idx, group1_name in enumerate(group1_order):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if row_idx == 0:
                ax.set_title(str(group1_name), fontsize=10, fontweight="normal", pad=4)

            mean_x: list[float] = []
            mean_y: list[float] = []
            point_color = group1_color_map.get(str(group1_name), "#4b5563")
            for group2_idx, group2_name in enumerate(group2_order):
                grouped_samples = sample_groups.get((str(group1_name), str(group2_name)), [])
                if not grouped_samples:
                    continue

                y_values = pd.to_numeric(module_values.reindex(grouped_samples), errors="coerce").dropna()
                if y_values.empty:
                    continue

                y_array = y_values.to_numpy(dtype=float, copy=False)
                offsets = _centered_point_offsets(len(y_array))
                ax.scatter(
                    np.full(len(y_array), float(group2_idx), dtype=float) + offsets,
                    y_array,
                    s=15,
                    color=point_color,
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=0.88,
                    zorder=3,
                )
                mean_x.append(float(group2_idx))
                mean_y.append(float(np.nanmean(y_array)))

            if mean_x:
                ax.plot(
                    mean_x,
                    mean_y,
                    color=module_color_map.get(str(module_name), "#111827"),
                    linewidth=1.35,
                    alpha=0.96,
                    zorder=4,
                )

            ax.set_xlim(-0.5, n_group2 - 0.5)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=2.0, width=0.55, labelbottom=False)
            ax.tick_params(axis="y", length=2.5, width=0.6, labelsize=7)
            if col_idx > 0:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
            ax.grid(axis="x", color="#eef2f7", linewidth=0.42)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.65)
                spine.set_edgecolor("#6b7280")

            if col_idx == n_group1 - 1:
                ax.text(
                    1.035,
                    0.5,
                    str(module_name),
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=8,
                    rotation=270,
                )

    for col_idx in range(n_group1):
        strip_ax = fig.add_subplot(gs[n_modules, col_idx])
        _add_group2_color_strip(strip_ax, group2_order, group2_color_map)

    legend_handles = [
        Patch(facecolor=group2_color_map.get(str(group_name), "#d1d5db"), edgecolor="none", label=str(group_name))
        for group_name in group2_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=legend_ncol,
            frameon=False,
            handlelength=1.3,
            handleheight=0.75,
            handletextpad=0.42,
            columnspacing=0.85,
            labelspacing=0.42,
            borderaxespad=0.0,
            fontsize=9,
        )

    fig.text(0.028, 0.54, "z score", rotation=90, ha="center", va="center", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.965, bottom=bottom_margin)
    _save_figure(fig, save_stem, cfg)


def _coerce_engine_gene_expression_df(engine) -> pd.DataFrame:
    if hasattr(engine, "gene_expression_df"):
        try:
            expr_df = engine.gene_expression_df()
        except Exception:
            expr_df = pd.DataFrame()
    else:
        expr_df = pd.DataFrame()

    if (not isinstance(expr_df, pd.DataFrame) or expr_df.empty) and hasattr(engine, "adata"):
        expr_df = _gene_expression_df(engine.adata)

    if not isinstance(expr_df, pd.DataFrame) or expr_df.empty:
        return pd.DataFrame()

    work = expr_df.copy()
    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work.columns = pd.Index(work.columns.astype(str).str.strip(), name=work.columns.name)
    work = work.loc[work.index.astype(str).str.len() > 0, work.columns.astype(str).str.len() > 0].copy()
    work = work.loc[~work.index.duplicated(keep="first"), ~work.columns.duplicated(keep="first")].copy()
    return work.apply(pd.to_numeric, errors="coerce")


def _module_gene_map_from_assignment(
    gene_module_assignment_df: pd.DataFrame,
    module_order: list[str],
    available_genes: pd.Index,
) -> dict[str, list[str]]:
    if not isinstance(gene_module_assignment_df, pd.DataFrame) or gene_module_assignment_df.empty:
        return {}
    if not {"Gene", "Module"}.issubset(gene_module_assignment_df.columns):
        return {}

    available_gene_set = set(available_genes.astype(str).tolist())
    work = gene_module_assignment_df.loc[:, ["Gene", "Module"]].copy()
    work["Gene"] = work["Gene"].astype(str).str.strip()
    work["Module"] = work["Module"].astype(str).str.strip()
    work = work.loc[work["Gene"].ne("") & work["Module"].ne("") & work["Gene"].isin(available_gene_set)].copy()
    if work.empty:
        return {}

    work = work.drop_duplicates(subset=["Module", "Gene"], keep="first")
    module_gene_map: dict[str, list[str]] = {}
    for module_name in module_order:
        genes = work.loc[work["Module"].eq(str(module_name)), "Gene"].astype(str).tolist()
        if genes:
            module_gene_map[str(module_name)] = genes
    return module_gene_map


def plot_module_gene_zscore_line_panels(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return

    expr_df = _coerce_engine_gene_expression_df(engine)
    if expr_df.empty:
        return

    expr_sample_set = set(expr_df.index.astype(str).tolist())
    shared_samples = [sample for sample in eigengenes_df.index.astype(str).tolist() if sample in expr_sample_set]
    if not shared_samples:
        return
    eigengenes_df = eigengenes_df.reindex(shared_samples)
    expr_df = expr_df.reindex(shared_samples)

    raw_module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order = [module for module in raw_module_order if module in eigengenes_df.columns]
    if not module_order:
        return

    module_gene_map = _module_gene_map_from_assignment(
        engine.ml_results.get("gene_module_assignment_df", pd.DataFrame()),
        module_order,
        expr_df.columns,
    )
    module_order = [module for module in module_order if module in module_gene_map]
    if not module_order:
        return

    all_module_genes = []
    seen_genes: set[str] = set()
    for module_name in module_order:
        for gene_name in module_gene_map[module_name]:
            if gene_name in seen_genes:
                continue
            seen_genes.add(gene_name)
            all_module_genes.append(gene_name)
    if not all_module_genes:
        return

    module_zscore_df = _row_zscore(eigengenes_df.loc[:, module_order].T)
    gene_zscore_df = _row_zscore(expr_df.loc[:, all_module_genes].T)
    if module_zscore_df.empty or gene_zscore_df.empty:
        return
    module_color_map = _module_color_map_from_results(engine, module_order)

    sample_names = module_zscore_df.columns.astype(str).tolist()
    annotation_df = _align_group_annotations_to_samples(sample_names, group_df)
    if annotation_df.empty:
        return

    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation_df["group1"].astype(str).tolist(),
        annotation_df["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = group_orders_by_col.get("group1", [])
    group2_order = group_orders_by_col.get("group2", [])
    if not group1_order or not group2_order:
        return

    group2_color_map = color_maps_by_col.get("group2", {})

    annotation_df = annotation_df.reindex(sample_names)
    sample_groups: dict[tuple[str, str], list[str]] = {}
    for group1_name in group1_order:
        for group2_name in group2_order:
            mask = (
                annotation_df["group1"].astype(str).eq(str(group1_name))
                & annotation_df["group2"].astype(str).eq(str(group2_name))
            )
            sample_groups[(str(group1_name), str(group2_name))] = annotation_df.index[mask].astype(str).tolist()

    gene_line_cache: dict[tuple[str, str], pd.DataFrame] = {}
    module_line_cache: dict[tuple[str, str], np.ndarray] = {}
    finite_chunks: list[np.ndarray] = []

    for module_name in module_order:
        module_genes = [gene for gene in module_gene_map[module_name] if gene in gene_zscore_df.index]
        if not module_genes:
            continue
        module_gene_zscores = gene_zscore_df.loc[module_genes]
        module_zscores = module_zscore_df.loc[str(module_name)]
        for group1_name in group1_order:
            gene_lines = pd.DataFrame(index=module_genes, columns=group2_order, dtype=float)
            module_line = np.full(len(group2_order), np.nan, dtype=float)
            for group2_idx, group2_name in enumerate(group2_order):
                grouped_samples = [
                    sample
                    for sample in sample_groups.get((str(group1_name), str(group2_name)), [])
                    if sample in module_gene_zscores.columns
                ]
                if not grouped_samples:
                    continue
                gene_lines.loc[:, group2_name] = module_gene_zscores.loc[:, grouped_samples].mean(axis=1).to_numpy(dtype=float)
                module_line[group2_idx] = float(pd.to_numeric(module_zscores.reindex(grouped_samples), errors="coerce").mean())

            gene_line_cache[(str(module_name), str(group1_name))] = gene_lines
            module_line_cache[(str(module_name), str(group1_name))] = module_line

            gene_values = gene_lines.to_numpy(dtype=float, copy=False)
            gene_values = gene_values[np.isfinite(gene_values)]
            if gene_values.size:
                finite_chunks.append(gene_values)
            module_values = module_line[np.isfinite(module_line)]
            if module_values.size:
                finite_chunks.append(module_values)

    if not finite_chunks:
        return

    finite_values = np.concatenate(finite_chunks)
    y_min = float(np.nanmin(finite_values))
    y_max = float(np.nanmax(finite_values))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    y_pad = max(0.15, 0.08 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    n_modules = len(module_order)
    n_group1 = len(group1_order)
    n_group2 = len(group2_order)
    panel_width = max(1.10, min(1.65, 0.045 * n_group2 + 0.85))
    panel_height = 0.76 if n_modules <= 10 else 0.62
    fig_width = max(6.2, min(18.0, panel_width * n_group1 + 1.35))
    fig_height = max(4.8, min(28.0, panel_height * n_modules + 1.45))

    legend_ncol = _legend_column_count(len(group2_order), fig_width)
    legend_rows = int(np.ceil(len(group2_order) / max(1, legend_ncol)))
    bottom_margin = max(0.12, min(0.24, 0.075 + 0.035 * legend_rows))

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    fig._png_dpi = 180
    gs = fig.add_gridspec(
        nrows=n_modules + 1,
        ncols=n_group1,
        height_ratios=[1.0] * n_modules + [0.12],
        hspace=0.12,
        wspace=0.18,
    )

    x_positions = np.arange(n_group2, dtype=float)
    for row_idx, module_name in enumerate(module_order):
        for col_idx, group1_name in enumerate(group1_order):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if row_idx == 0:
                ax.set_title(str(group1_name), fontsize=10, fontweight="normal", pad=4)

            gene_lines = gene_line_cache.get((str(module_name), str(group1_name)), pd.DataFrame())
            if not gene_lines.empty:
                for line_values in gene_lines.to_numpy(dtype=float, copy=False):
                    if np.isfinite(line_values).sum() < 2:
                        continue
                    ax.plot(
                        x_positions,
                        line_values,
                        color="#cfd3d8",
                        linewidth=0.55,
                        alpha=0.50,
                        zorder=2,
                    )

            module_line = module_line_cache.get((str(module_name), str(group1_name)), np.array([], dtype=float))
            if module_line.size and np.isfinite(module_line).sum() >= 2:
                ax.plot(
                    x_positions,
                    module_line,
                    color=module_color_map.get(str(module_name), "#111827"),
                    linewidth=1.45,
                    alpha=0.97,
                    zorder=4,
                )

            ax.set_xlim(-0.5, n_group2 - 0.5)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=2.0, width=0.55, labelbottom=False)
            ax.tick_params(axis="y", length=2.5, width=0.6, labelsize=7)
            if col_idx > 0:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
            ax.grid(axis="x", color="#eef2f7", linewidth=0.42)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.65)
                spine.set_edgecolor("#6b7280")

            if col_idx == n_group1 - 1:
                ax.text(
                    1.035,
                    0.5,
                    str(module_name),
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=8,
                    rotation=270,
                )

    for col_idx in range(n_group1):
        strip_ax = fig.add_subplot(gs[n_modules, col_idx])
        _add_group2_color_strip(strip_ax, group2_order, group2_color_map)

    legend_handles = [
        Patch(facecolor=group2_color_map.get(str(group_name), "#d1d5db"), edgecolor="none", label=str(group_name))
        for group_name in group2_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=legend_ncol,
            frameon=False,
            handlelength=1.3,
            handleheight=0.75,
            handletextpad=0.42,
            columnspacing=0.85,
            labelspacing=0.42,
            borderaxespad=0.0,
            fontsize=9,
        )

    fig.text(0.028, 0.54, "z score", rotation=90, ha="center", va="center", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.965, bottom=bottom_margin)
    _save_figure(fig, save_stem, cfg)


def _module_color_map_from_results(engine, module_order: list[str]) -> dict[str, str]:
    module_color_map: dict[str, str] = {}
    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty:
        if {"Module", "ModuleColorHex"}.issubset(module_summary_df.columns):
            for _, row in module_summary_df.loc[:, ["Module", "ModuleColorHex"]].iterrows():
                module_name = str(row["Module"]).strip()
                color_value = str(row["ModuleColorHex"]).strip()
                if not module_name or not color_value:
                    continue
                try:
                    module_color_map[module_name] = colors.to_hex(colors.to_rgba(color_value), keep_alpha=False)
                except ValueError:
                    continue

    fallback_colors = sns.color_palette("tab20", n_colors=max(1, len(module_order))).as_hex()
    for idx, module_name in enumerate(module_order):
        module_color_map.setdefault(str(module_name), fallback_colors[idx % len(fallback_colors)])
    return module_color_map


def _density_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size < 2:
        return None
    if float(np.nanstd(clean_values)) <= 1e-9:
        return None
    try:
        density = gaussian_kde(clean_values)(x_grid)
    except (ValueError, np.linalg.LinAlgError):
        return None
    density = np.asarray(density, dtype=float)
    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0:
        return None
    return density / float(np.nanmax(density))


def _prepare_module_ridge_matrix(engine) -> tuple[pd.DataFrame, list[str]]:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return pd.DataFrame(), []

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order = [module for module in module_order if module in eigengenes_df.columns]
    if not module_order:
        return pd.DataFrame(), []

    eigengenes_df = eigengenes_df.loc[:, module_order].copy()
    zscore_df = _row_zscore(eigengenes_df.T).T
    return zscore_df, module_order


def plot_module_eigengene_ridge(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    zscore_df, module_order = _prepare_module_ridge_matrix(engine)
    if zscore_df.empty or not module_order:
        return

    finite_values = zscore_df.to_numpy(dtype=float, copy=False)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 2:
        return

    x_min = float(np.nanmin(finite_values))
    x_max = float(np.nanmax(finite_values))
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    x_pad = max(0.25, 0.10 * (x_max - x_min))
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 256)

    module_color_map = _module_color_map_from_results(engine, module_order)
    n_modules = len(module_order)
    fig_width = 9.2
    fig_height = max(4.8, min(22.0, 0.58 * n_modules + 1.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ridge_height = 0.78
    for row_idx, module_name in enumerate(module_order):
        y_base = float(n_modules - row_idx - 1)
        values = pd.to_numeric(zscore_df[module_name], errors="coerce").dropna().to_numpy(dtype=float)
        density = _density_curve(values, x_grid)
        color_value = module_color_map.get(str(module_name), "#9ca3af")
        if density is not None:
            y_values = y_base + density * ridge_height
            ax.fill_between(x_grid, y_base, y_values, color=color_value, alpha=0.42, linewidth=0)
            ax.plot(x_grid, y_values, color=color_value, linewidth=1.25)
        else:
            ax.hlines(y_base, x_grid[0], x_grid[-1], color=color_value, linewidth=1.0, alpha=0.8)
        if values.size:
            ax.vlines(
                values,
                y_base,
                y_base + ridge_height * 0.12,
                color=color_value,
                linewidth=0.55,
                alpha=0.35,
            )
        ax.axhline(y_base, color="#e5e7eb", linewidth=0.55, zorder=0)

    ax.set_yticks(np.arange(n_modules, dtype=float))
    ax.set_yticklabels(list(reversed(module_order)))
    ax.set_xlabel("Module eigengene z-score")
    ax.set_ylabel("Module")
    ax.set_title("Module Eigengene Ridge Distribution")
    ax.set_ylim(-0.35, n_modules - 1 + ridge_height + 0.20)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.55)
    ax.set_axisbelow(True)
    _save_figure(fig, save_stem, cfg)


def plot_module_eigengene_ridge_group1(
    engine,
    save_stem: str | Path,
    cfg,
    group_df: pd.DataFrame | None = None,
) -> None:
    zscore_df, module_order = _prepare_module_ridge_matrix(engine)
    if zscore_df.empty or not module_order:
        return

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

    finite_values = zscore_df.to_numpy(dtype=float, copy=False)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 2:
        return
    x_min = float(np.nanmin(finite_values))
    x_max = float(np.nanmax(finite_values))
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    x_pad = max(0.25, 0.10 * (x_max - x_min))
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 256)

    n_modules = len(module_order)
    fig_width = 9.6
    fig_height = max(4.8, min(22.0, 0.62 * n_modules + 1.9))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ridge_height = 0.72
    group_offsets = _centered_point_offsets(len(group1_order), width=0.12)
    for row_idx, module_name in enumerate(module_order):
        y_base = float(n_modules - row_idx - 1)
        for group_idx, group1_name in enumerate(group1_order):
            group_samples = annotation_df.index[
                annotation_df["group1"].astype(str).eq(str(group1_name))
            ].astype(str).tolist()
            values = pd.to_numeric(zscore_df.loc[group_samples, module_name], errors="coerce").dropna().to_numpy(dtype=float)
            density = _density_curve(values, x_grid)
            color_value = group1_color_map.get(str(group1_name), "#9ca3af")
            if density is not None:
                y_values = y_base + density * ridge_height
                ax.fill_between(x_grid, y_base, y_values, color=color_value, alpha=0.16, linewidth=0)
                ax.plot(x_grid, y_values, color=color_value, linewidth=1.0, alpha=0.92)
            if values.size:
                ax.vlines(
                    values,
                    y_base + group_offsets[group_idx],
                    y_base + group_offsets[group_idx] + ridge_height * 0.10,
                    color=color_value,
                    linewidth=0.55,
                    alpha=0.36,
                )
        ax.axhline(y_base, color="#e5e7eb", linewidth=0.55, zorder=0)

    legend_handles = [
        Line2D([0], [0], color=group1_color_map.get(str(group_name), "#9ca3af"), linewidth=2.0, label=str(group_name))
        for group_name in group1_order
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            bbox_transform=ax.transAxes,
            ncol=1,
            frameon=False,
            fontsize=10.5,
            handlelength=1.5,
            columnspacing=0.9,
            borderaxespad=0.0,
        )

    ax.set_yticks(np.arange(n_modules, dtype=float))
    ax.set_yticklabels(list(reversed(module_order)))
    ax.set_xlabel("Module eigengene z-score")
    ax.set_ylabel("Module")
    ax.set_title("Module Eigengene Ridge Distribution by group1")
    ax.set_ylim(-0.35, n_modules - 1 + ridge_height + 0.20)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.55)
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.76)
    _save_figure(fig, save_stem, cfg)


def plot_module_metabolite_association_heatmap(engine, save_stem: str | Path, cfg) -> None:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return

    required_columns = {"Module", "Metabolite", "SpearmanRho"}
    if not required_columns.issubset(assoc_df.columns):
        return

    plot_df = assoc_df.copy()
    plot_df["Module"] = plot_df["Module"].astype(str)
    plot_df["Metabolite"] = plot_df["Metabolite"].astype(str)
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
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        module_order = [
            str(module_name)
            for module_name in module_summary_df["Module"].astype(str).tolist()
            if str(module_name) in set(plot_df["Module"].tolist())
        ]
    else:
        module_order = []

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
        return

    annotation_matrix = (
        sig_matrix.map(_significance_star)
        if hasattr(sig_matrix, "map")
        else sig_matrix.applymap(_significance_star)
    )

    finite_rho = rho_matrix.to_numpy(dtype=float)
    finite_rho = finite_rho[np.isfinite(finite_rho)]
    vmax = float(np.nanmax(np.abs(finite_rho))) if finite_rho.size > 0 else 1.0
    vmax = max(vmax, 0.25)

    fig_width = max(9.0, min(28.0, 0.42 * max(1, rho_matrix.shape[1]) + 4.5))
    fig_height = max(4.5, min(18.0, 0.50 * max(1, rho_matrix.shape[0]) + 2.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        rho_matrix,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#f3f4f6",
        annot=annotation_matrix,
        fmt="",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Spearman rho"},
        mask=rho_matrix.isna(),
    )

    metric_label = "FDR" if significance_column == "FDR" else "P value"
    colorbar = ax.collections[0].colorbar if ax.collections else None
    if colorbar is not None:
        colorbar.ax.set_title(
            f"{metric_label}\n* < 0.05\n** < 0.01\n*** < 0.001",
            fontsize=9,
            fontweight="normal",
            pad=8,
        )
    ax.set_title("Module-Metabolite Association Heatmap", pad=8)
    ax.set_xlabel("Metabolite")
    ax.set_ylabel("Module")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "_coerce_module_eigengene_df",
    "_module_order_from_summary",
    "plot_module_eigengene_heatmap",
    "plot_module_eigengene_heatmap_group2",
    "plot_module_zscore_line_panels",
    "plot_module_gene_zscore_line_panels",
    "plot_module_eigengene_ridge",
    "plot_module_eigengene_ridge_group1",
    "plot_module_metabolite_association_heatmap",
]
