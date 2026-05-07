from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import _save_figure, logger


EVIDENCE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("In_PCC", "PCC", "#4c78a8"),
    ("In_Spearman", "Spearman", "#59a14f"),
    ("In_MI", "MI", "#f28e2b"),
    ("ElasticNetSelected", "ElasticNet", "#b07aa1"),
    ("XGBoostSelected", "XGBoost", "#e15759"),
)


def _coerce_evidence_edge_table(gene_scores_df: pd.DataFrame) -> pd.DataFrame:
    evidence_columns = [column for column, _, _ in EVIDENCE_SPECS]
    required_columns = {"Metabolite", "Gene", *evidence_columns}
    missing_columns = sorted(required_columns.difference(gene_scores_df.columns))
    if missing_columns:
        logger.warning(
            "Association evidence UpSet plot was skipped because required columns are missing: %s",
            ", ".join(missing_columns),
        )
        return pd.DataFrame(columns=["Metabolite", "Gene", *evidence_columns])

    work = gene_scores_df.loc[:, ["Metabolite", "Gene", *evidence_columns]].copy()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work["Gene"] = work["Gene"].astype(str).str.strip()
    work = work.loc[work["Metabolite"].ne("") & work["Gene"].ne("")].copy()
    if work.empty:
        return pd.DataFrame(columns=["Metabolite", "Gene", *evidence_columns])

    for column in evidence_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0).gt(0)

    edge_table = (
        work.groupby(["Metabolite", "Gene"], sort=False, as_index=False)[evidence_columns]
        .max()
        .reset_index(drop=True)
    )
    return edge_table.loc[edge_table[evidence_columns].any(axis=1)].reset_index(drop=True)


def _build_evidence_intersection_table(
    gene_scores_df: pd.DataFrame,
    *,
    max_intersections: int = 30,
) -> tuple[pd.DataFrame, pd.Series, int]:
    edge_table = _coerce_evidence_edge_table(gene_scores_df)
    evidence_columns = [column for column, _, _ in EVIDENCE_SPECS]
    if edge_table.empty:
        empty_counts = pd.Series(0, index=pd.Index(evidence_columns, name="Evidence"), dtype=int)
        return pd.DataFrame(columns=[*evidence_columns, "Count", "SupportCount"]), empty_counts, 0

    set_sizes = edge_table[evidence_columns].sum(axis=0).astype(int)
    intersections = (
        edge_table.groupby(evidence_columns, sort=False)
        .size()
        .rename("Count")
        .reset_index()
    )
    intersections = intersections.loc[intersections[evidence_columns].any(axis=1)].copy()
    intersections["SupportCount"] = intersections[evidence_columns].sum(axis=1).astype(int)
    intersections["_Pattern"] = intersections[evidence_columns].astype(int).astype(str).agg("".join, axis=1)
    intersections = intersections.sort_values(
        ["Count", "SupportCount", "_Pattern"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(max(1, int(max_intersections)))
    intersections = intersections.drop(columns=["_Pattern"]).reset_index(drop=True)
    return intersections, set_sizes, int(len(edge_table))


def plot_association_evidence_upset(engine, save_stem: str | Path, cfg) -> None:
    """Plot global evidence overlap for gene-metabolite candidate edges."""
    gene_scores_df = engine.ml_results.get("gene_scores_df", pd.DataFrame())
    if not isinstance(gene_scores_df, pd.DataFrame) or gene_scores_df.empty:
        logger.warning("Association evidence UpSet plot was skipped because gene_scores_df is empty.")
        return

    max_intersections = int(getattr(cfg, "upset_plot_top_intersections", 30))
    intersections, set_sizes, n_edges = _build_evidence_intersection_table(
        gene_scores_df,
        max_intersections=max_intersections,
    )
    if intersections.empty:
        logger.warning("Association evidence UpSet plot was skipped because no evidence-positive edges were available.")
        return

    evidence_columns = [column for column, _, _ in EVIDENCE_SPECS]
    evidence_labels = [label for _, label, _ in EVIDENCE_SPECS]
    evidence_colors = [color for _, _, color in EVIDENCE_SPECS]
    counts = intersections["Count"].astype(int).to_numpy()
    n_intersections = len(intersections)
    n_sets = len(EVIDENCE_SPECS)

    fig_width = float(np.clip(6.0 + 0.34 * n_intersections, 10.5, 18.0))
    fig = plt.figure(figsize=(fig_width, 7.2))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.55, max(4.2, 0.30 * n_intersections)],
        height_ratios=[2.15, 1.35],
        wspace=0.08,
        hspace=0.04,
    )
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_bars = fig.add_subplot(grid[0, 1])
    ax_sets = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_bars)

    x_positions = np.arange(n_intersections)
    ax_bars.bar(x_positions, counts, color="#374151", width=0.72)
    ax_bars.set_ylabel("Candidate edges")
    ax_bars.set_title("Association Evidence Overlap", pad=12)
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
    set_values = set_sizes.reindex(evidence_columns).fillna(0).astype(int).to_numpy()
    ax_sets.barh(y_positions, set_values, color=evidence_colors, height=0.62)
    ax_sets.set_yticks(y_positions, labels=evidence_labels)
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Set size")
    ax_sets.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_sets.set_xlim(0, max(int(set_values.max()), 1) * 1.22)
    for y_pos, value in zip(y_positions, set_values):
        ax_sets.text(value + max(int(set_values.max()), 1) * 0.025, y_pos, str(int(value)), va="center", fontsize=8)

    active_matrix = intersections[evidence_columns].to_numpy(dtype=bool).T
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
        active_colors = [evidence_colors[row_idx] for row_idx in active_rows]
        if active_rows.size:
            ax_matrix.scatter(
                np.repeat(col_idx, active_rows.size),
                active_rows,
                s=58,
                color=active_colors,
                edgecolors="#111827",
                linewidths=0.35,
                zorder=3,
            )

    ax_matrix.set_ylim(-0.5, n_sets - 0.5)
    ax_matrix.invert_yaxis()
    ax_matrix.set_yticks([])
    ax_matrix.set_xticks([])
    ax_matrix.set_xlabel(f"Top {n_intersections} evidence intersections")
    ax_matrix.spines["top"].set_visible(False)

    ax_summary.axis("off")
    summary_text = (
        f"Evidence-positive edges: {n_edges:,}\n"
        f"Displayed intersections: {n_intersections:,}\n"
        "Unit: metabolite-gene edge"
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

    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.13)
    fig._skip_default_tight_layout = True
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "EVIDENCE_SPECS",
    "_coerce_evidence_edge_table",
    "_build_evidence_intersection_table",
    "plot_association_evidence_upset",
]
