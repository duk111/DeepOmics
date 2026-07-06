from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _scale_numeric(values: pd.Series, min_size: float = 28.0, max_size: float = 220.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(min_size, index=values.index, dtype=float)

    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        return pd.Series((min_size + max_size) / 2.0, index=values.index, dtype=float)

    scaled = (numeric - low) / (high - low)
    return (min_size + scaled.fillna(0.0) * (max_size - min_size)).astype(float)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(100, 116, 139, {alpha})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _save_static_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if output_path.suffix.lower() != ".svg":
        fig.savefig(output_path.with_suffix(".svg"), format="svg")


def plot_differential_upset(
    sig_results: list[pd.DataFrame],
    *,
    feature_col: str,
    title: str,
    unit_label: str,
    output_path: Path,
    max_intersections: int = 30,
) -> None:
    """Plot contrast-overlap UpSet for significant differential features."""
    memberships: dict[str, set[str]] = {}
    for sig_df in sig_results:
        if sig_df.empty or feature_col not in sig_df.columns or "comparison" not in sig_df.columns:
            continue
        comparison = str(sig_df["comparison"].iloc[0])
        features = set(sig_df[feature_col].dropna().astype(str))
        if features:
            memberships[comparison] = features

    if not memberships:
        return

    comparisons = list(memberships)
    all_features = sorted(set().union(*memberships.values()))
    membership_df = pd.DataFrame(index=all_features)
    for comparison in comparisons:
        membership_df[comparison] = membership_df.index.isin(memberships[comparison])

    set_sizes = membership_df.sum(axis=0).astype(int)
    intersections = (
        membership_df.groupby(comparisons, sort=False)
        .size()
        .rename("Count")
        .reset_index()
    )
    intersections = intersections.loc[intersections[comparisons].any(axis=1)].copy()
    intersections["SupportCount"] = intersections[comparisons].sum(axis=1).astype(int)
    intersections["_Pattern"] = intersections[comparisons].astype(int).astype(str).agg("".join, axis=1)
    intersections = (
        intersections.sort_values(
            ["Count", "SupportCount", "_Pattern"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(max(1, int(max_intersections)))
        .drop(columns=["_Pattern"])
        .reset_index(drop=True)
    )
    if intersections.empty:
        return

    counts = intersections["Count"].astype(int).to_numpy()
    n_intersections = len(intersections)
    n_sets = len(comparisons)
    set_colors = plt.colormaps["tab20"](np.linspace(0.0, 1.0, max(n_sets, 1)))

    fig_width = float(np.clip(6.0 + 0.36 * n_intersections, 10.5, 18.0))
    fig_height = float(np.clip(4.8 + 0.22 * n_sets, 6.8, 12.5))
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.7, max(4.4, 0.30 * n_intersections)],
        height_ratios=[2.1, max(1.35, 0.16 * n_sets + 1.0)],
        wspace=0.08,
        hspace=0.04,
    )
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_bars = fig.add_subplot(grid[0, 1])
    ax_sets = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_bars)

    x_positions = np.arange(n_intersections)
    ax_bars.bar(x_positions, counts, color="#374151", width=0.72)
    ax_bars.set_ylabel(unit_label)
    ax_bars.set_title(title, pad=12)
    ax_bars.set_xlim(-0.5, n_intersections - 0.5)
    ax_bars.set_xticks([])
    ax_bars.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax_bars.spines["bottom"].set_visible(False)
    ymax = max(int(counts.max()), 1)
    ax_bars.set_ylim(0, ymax * 1.18)
    for idx, value in enumerate(counts):
        if n_intersections <= 32 or idx % 2 == 0:
            ax_bars.text(idx, value + ymax * 0.025, str(int(value)), ha="center", va="bottom", fontsize=8)

    y_positions = np.arange(n_sets)
    set_values = set_sizes.reindex(comparisons).fillna(0).astype(int).to_numpy()
    ax_sets.barh(y_positions, set_values, color=set_colors[:n_sets], height=0.62)
    ax_sets.set_yticks(y_positions, labels=comparisons)
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Set size")
    ax_sets.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_sets.set_xlim(0, max(int(set_values.max()), 1) * 1.22)
    for y_pos, value in zip(y_positions, set_values):
        ax_sets.text(value + max(int(set_values.max()), 1) * 0.025, y_pos, str(int(value)), va="center", fontsize=8)

    active_matrix = intersections[comparisons].to_numpy(dtype=bool).T
    for row_idx in range(n_sets):
        if row_idx % 2 == 0:
            ax_matrix.axhspan(row_idx - 0.5, row_idx + 0.5, color="#f8fafc", zorder=0)
            ax_sets.axhspan(row_idx - 0.5, row_idx + 0.5, color="#f8fafc", zorder=0)

    for col_idx in range(n_intersections):
        active_rows = np.flatnonzero(active_matrix[:, col_idx])
        if active_rows.size:
            ax_matrix.plot(
                [col_idx, col_idx],
                [active_rows.min(), active_rows.max()],
                color="#111827",
                linewidth=1.1,
                solid_capstyle="round",
                zorder=2,
            )
        ax_matrix.scatter(
            np.repeat(col_idx, n_sets),
            y_positions,
            s=46,
            color="#d1d5db",
            edgecolors="none",
            zorder=1,
        )
        if active_rows.size:
            ax_matrix.scatter(
                np.repeat(col_idx, active_rows.size),
                active_rows,
                s=58,
                color=[set_colors[row_idx] for row_idx in active_rows],
                edgecolors="#111827",
                linewidths=0.35,
                zorder=3,
            )

    ax_matrix.set_ylim(-0.5, n_sets - 0.5)
    ax_matrix.invert_yaxis()
    ax_matrix.set_yticks([])
    ax_matrix.set_xticks([])
    ax_matrix.set_xlabel(f"Top {n_intersections} intersections")
    ax_matrix.spines["top"].set_visible(False)

    ax_summary.axis("off")
    summary_text = (
        f"Union significant {unit_label.lower()}: {len(all_features):,}\n"
        f"Contrasts with hits: {n_sets:,}\n"
        f"Displayed intersections: {n_intersections:,}"
    )
    ax_summary.text(
        0.0,
        0.96,
        summary_text,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.55,
        color="#111827",
    )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.14)
    _save_static_figure(fig, output_path)
    plt.close(fig)


def plot_dem_joint_scatter(
    results_df: pd.DataFrame,
    *,
    comparison: str,
    output_path: Path,
    x_axis: str,
    y_axis: str,
    size_by: str,
    vip_cutoff: float,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> None:
    """Plot DEM VIP-log2FC-padj joint scatter for a single contrast."""
    required = {"log2FoldChange", "vip", "padj_bh", "dem_status"}
    if results_df.empty or not required.issubset(results_df.columns):
        return

    plot_df = results_df.loc[:, ["log2FoldChange", "vip", "padj_bh", "dem_status"]].copy()
    finite_padj = plot_df.loc[plot_df["padj_bh"].notna() & (plot_df["padj_bh"] > 0), "padj_bh"]
    min_positive_padj = float(finite_padj.min()) if not finite_padj.empty else 1e-300
    plot_df["neg_log10_padj"] = -np.log10(plot_df["padj_bh"].fillna(1.0).clip(lower=min_positive_padj))

    axis_specs = {
        "vip": ("vip", "OPLS-DA VIP"),
        "log2fc": ("log2FoldChange", "log2 Fold Change"),
        "padj": ("padj_bh", "Adjusted P value"),
        "neg_log10_padj": ("neg_log10_padj", "-log10 adjusted P value"),
    }
    if x_axis not in axis_specs:
        raise ValueError("x_axis must be one of 'vip', 'log2fc', 'padj', or 'neg_log10_padj'.")
    if y_axis not in axis_specs:
        raise ValueError("y_axis must be one of 'vip', 'log2fc', 'padj', or 'neg_log10_padj'.")

    x_column, x_label = axis_specs[x_axis]
    y_column, y_label = axis_specs[y_axis]

    if size_by == "vip":
        sizes = _scale_numeric(plot_df["vip"], min_size=24.0, max_size=230.0)
        size_label = "Point size: VIP"
        size_title = "VIP"
        size_values = plot_df["vip"]
    elif size_by == "padj":
        sizes = _scale_numeric(plot_df["neg_log10_padj"], min_size=24.0, max_size=230.0)
        size_label = "Point size: -log10 adjusted P value"
        size_title = "-log10 adj P"
        size_values = plot_df["neg_log10_padj"]
    else:
        raise ValueError("size_by must be either 'vip' or 'padj'.")

    colors = {
        "Down": "#2B6CB0",
        "Non-significant": "#B8B8B8",
        "Up": "#C53030",
    }
    order = ["Non-significant", "Down", "Up"]

    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    for status in order:
        sub = plot_df.loc[plot_df["dem_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_column],
            sub[y_column],
            s=sizes.loc[sub.index],
            c=colors[status],
            alpha=0.78 if status != "Non-significant" else 0.42,
            edgecolors="white",
            linewidths=0.35,
            label=f"{status} (n={len(sub)})",
        )

    if x_axis == "log2fc":
        ax.axvline(log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
        ax.axvline(-log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif x_axis == "vip":
        ax.axvline(vip_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif x_axis == "padj":
        ax.axvline(padj_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif x_axis == "neg_log10_padj":
        ax.axvline(-np.log10(padj_cutoff), color="#606060", linestyle="--", linewidth=0.9)

    if y_axis == "log2fc":
        ax.axhline(log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
        ax.axhline(-log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif y_axis == "vip":
        ax.axhline(vip_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif y_axis == "padj":
        ax.axhline(padj_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    elif y_axis == "neg_log10_padj":
        ax.axhline(-np.log10(padj_cutoff), color="#606060", linestyle="--", linewidth=0.9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"DEM Joint Scatter: {comparison}")

    stats_order = ["Up", "Non-significant", "Down"]
    stats_handles = []
    for status in stats_order:
        count = int((plot_df["dem_status"] == status).sum())
        stats_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=colors[status],
                markeredgecolor="none",
                alpha=0.9 if status != "Non-significant" else 0.55,
                label=f"{status}: {count}",
            )
        )
    stats_legend = ax.legend(
        handles=stats_handles,
        title="Statistics",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(stats_legend)

    finite_size_values = size_values[np.isfinite(size_values)]
    size_handles = []
    if not finite_size_values.empty:
        legend_values = np.quantile(finite_size_values.to_numpy(dtype=float), [0.0, 0.5, 1.0])
        legend_values = np.unique(np.round(legend_values, 2))
        value_min = float(finite_size_values.min())
        value_max = float(finite_size_values.max())
        for value in legend_values:
            if np.isclose(value_min, value_max):
                point_size = (24.0 + 230.0) / 2.0
            else:
                point_size = 24.0 + ((float(value) - value_min) / (value_max - value_min)) * (230.0 - 24.0)
            marker_size = float(np.sqrt(max(point_size, 1.0)))
            size_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=marker_size,
                    markerfacecolor="white",
                    markeredgecolor="#374151",
                    markeredgewidth=0.9,
                    label=f"{value:g}",
                )
            )
    if size_handles:
        ax.legend(
            handles=size_handles,
            title=size_title,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.62),
            borderaxespad=0.0,
        )

    ax.text(
        1.02,
        0.18,
        f"{size_label}\npadj cutoff: {padj_cutoff:g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#374151",
    )
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    fig.subplots_adjust(left=0.12, right=0.74, top=0.90, bottom=0.14)
    _save_static_figure(fig, output_path)
    plt.close(fig)


def plot_differential_sankey(
    counts_df: pd.DataFrame,
    contrasts: list[dict],
    *,
    same_fields: list[str],
    same_field_orders: dict[str, list[str]] | None = None,
    tested_level_order: list[str] | None = None,
    tested_level_count: int,
    title: str,
    output_html: Path,
    output_png: Path,
) -> None:
    """Plot differential feature-count Sankey following the user-provided same_fields order."""
    if not same_fields or counts_df.empty or not contrasts:
        return

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Sankey output requires Plotly. Install the project dependencies with: "
            "python -m pip install -e ."
        ) from exc

    counts_by_comparison = {
        str(row["comparison"]): row
        for _, row in counts_df.iterrows()
        if "comparison" in counts_df.columns
    }
    if not counts_by_comparison:
        return

    node_indices: dict[str, int] = {}
    node_labels: list[str] = []
    node_layers: list[int] = []
    node_order_values: list[int] = []
    link_values: dict[tuple[int, int], float] = {}

    def get_node(key: str, label: str, layer: int, order_value: int = 0) -> int:
        if key not in node_indices:
            node_indices[key] = len(node_labels)
            node_labels.append(label)
            node_layers.append(layer)
            node_order_values.append(order_value)
        return node_indices[key]

    def add_link(
        source_key: str,
        source_label: str,
        source_layer: int,
        source_order: int,
        target_key: str,
        target_label: str,
        target_layer: int,
        target_order: int,
        value: float,
    ) -> None:
        if not np.isfinite(value) or value <= 0:
            return
        source = get_node(source_key, source_label, source_layer, source_order)
        target = get_node(target_key, target_label, target_layer, target_order)
        link_values[(source, target)] = link_values.get((source, target), 0.0) + float(value)

    include_tested_layer = tested_level_count > 1
    same_field_orders = same_field_orders or {}
    tested_level_order = tested_level_order or []
    order_maps = {
        field: {str(value): idx for idx, value in enumerate(values)}
        for field, values in same_field_orders.items()
    }
    tested_order_map = {str(value): idx for idx, value in enumerate(tested_level_order)}
    direction_layer = len(same_fields) + (1 if include_tested_layer else 0)
    for contrast in contrasts:
        comparison = str(contrast["name"])
        count_row = counts_by_comparison.get(comparison)
        if count_row is None:
            continue

        up_count = float(count_row.get("up_count", 0) or 0)
        down_count = float(count_row.get("down_count", 0) or 0)
        significant_count = up_count + down_count
        if significant_count <= 0:
            continue

        same_values = tuple(contrast.get("same_values", ()))
        if len(same_values) != len(same_fields):
            continue

        previous_key = ""
        previous_label = ""
        previous_layer = 0
        previous_order = 0
        for layer_idx, (field, value) in enumerate(zip(same_fields, same_values)):
            value_label = str(value)
            current_key = f"same:{field}={value_label}"
            current_label = value_label
            current_order = order_maps.get(field, {}).get(value_label, 10_000)
            if previous_key:
                add_link(
                    previous_key,
                    previous_label,
                    previous_layer,
                    previous_order,
                    current_key,
                    current_label,
                    layer_idx,
                    current_order,
                    significant_count,
                )
            else:
                get_node(current_key, current_label, layer_idx, current_order)
            previous_key = current_key
            previous_label = current_label
            previous_layer = layer_idx
            previous_order = current_order

        if not previous_key:
            continue

        terminal_source_key = previous_key
        terminal_source_label = previous_label
        terminal_source_layer = previous_layer
        terminal_source_order = previous_order
        if include_tested_layer:
            tested_label = f"{contrast['tested_level']}_vs_{contrast['reference_level']}"
            tested_key = f"{previous_key}|comparison={tested_label}"
            tested_order = tested_order_map.get(str(contrast["tested_level"]), 10_000)
            tested_layer = len(same_fields)
            add_link(
                previous_key,
                previous_label,
                previous_layer,
                previous_order,
                tested_key,
                tested_label,
                tested_layer,
                tested_order,
                significant_count,
            )
            terminal_source_key = tested_key
            terminal_source_label = tested_label
            terminal_source_layer = tested_layer
            terminal_source_order = tested_order

        add_link(
            terminal_source_key,
            terminal_source_label,
            terminal_source_layer,
            terminal_source_order,
            "direction:Up",
            "Up",
            direction_layer,
            0,
            up_count,
        )
        add_link(
            terminal_source_key,
            terminal_source_label,
            terminal_source_layer,
            terminal_source_order,
            "direction:Down",
            "Down",
            direction_layer,
            1,
            down_count,
        )

    if not link_values:
        return

    sources = [source for source, _ in link_values]
    targets = [target for _, target in link_values]
    values = list(link_values.values())
    layer_palette = [
        "#4C78A8",
        "#59A14F",
        "#F28E2B",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#9C755F",
        "#BAB0AC",
    ]
    node_colors = []
    for label, layer in zip(node_labels, node_layers):
        if label == "Up":
            node_colors.append("#C53030")
        elif label == "Down":
            node_colors.append("#2B6CB0")
        else:
            node_colors.append(layer_palette[layer % len(layer_palette)])
    link_colors = [_hex_to_rgba(node_colors[source], alpha=0.26) for source in sources]

    max_layer = max(node_layers) if node_layers else 0
    node_x: list[float] = []
    node_y: list[float] = []
    for idx, layer in enumerate(node_layers):
        layer_indices = [i for i, layer_value in enumerate(node_layers) if layer_value == layer]
        ordered_layer_indices = sorted(
            layer_indices,
            key=lambda i: (node_order_values[i], node_labels[i]),
        )
        rank = ordered_layer_indices.index(idx)
        n_in_layer = len(ordered_layer_indices)
        node_x.append(0.02 + 0.96 * (layer / max(max_layer, 1)))
        node_y.append(0.5 if n_in_layer == 1 else 0.02 + 0.96 * (rank / (n_in_layer - 1)))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node={
                    "pad": 18,
                    "thickness": 16,
                    "line": {"color": "rgba(55, 65, 81, 0.35)", "width": 0.5},
                    "label": node_labels,
                    "color": node_colors,
                    "x": node_x,
                    "y": node_y,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        font={"size": 12},
        width=1200,
        height=max(650, 80 * len(same_fields) + 430),
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    fig.write_image(output_png, scale=2)


__all__ = ["plot_dem_joint_scatter", "plot_differential_sankey", "plot_differential_upset"]
