from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.stats import t

from .base import (
    PALETTE,
    _gene_expression_df,
    _metabolomics_df,
    _ordered_unique_nonempty,
    _save_figure,
)
from .module import _coerce_module_eigengene_df, _module_order_from_summary
from .network import _build_circos_module_color_map

def _optional_plot_color(value) -> str | None:
    if pd.isna(value):
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        return colors.to_hex(colors.to_rgba(candidate), keep_alpha=False)
    except ValueError:
        return None


def _module_annotation_maps(engine) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    gene_to_module: dict[str, str] = {}
    gene_to_color: dict[str, str] = {}
    module_to_color: dict[str, str] = {}

    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty:
        if {"Module", "ModuleColorHex"}.issubset(module_summary_df.columns):
            for _, row in module_summary_df.loc[:, ["Module", "ModuleColorHex"]].iterrows():
                module_name = str(row["Module"]).strip()
                color_value = _optional_plot_color(row["ModuleColorHex"])
                if module_name and color_value:
                    module_to_color[module_name] = color_value

    module_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    if isinstance(module_df, pd.DataFrame) and not module_df.empty and {"Gene", "Module"}.issubset(module_df.columns):
        keep_cols = [col for col in ["Gene", "Module", "ModuleColorHex"] if col in module_df.columns]
        for _, row in module_df.loc[:, keep_cols].iterrows():
            gene = str(row["Gene"]).strip()
            module_name = str(row["Module"]).strip()
            if not gene or not module_name:
                continue
            gene_to_module[gene] = module_name
            color_value = _optional_plot_color(row.get("ModuleColorHex", None))
            if color_value:
                module_to_color[module_name] = color_value
                gene_to_color[gene] = color_value

    all_modules = _ordered_unique_nonempty([*module_to_color.keys(), *gene_to_module.values()])
    fallback_module_colors = _build_circos_module_color_map(all_modules)
    for module_name in all_modules:
        module_to_color.setdefault(module_name, fallback_module_colors.get(module_name, "#9ca3af"))

    for gene, module_name in gene_to_module.items():
        gene_to_color.setdefault(gene, module_to_color.get(module_name, "#9ca3af"))

    return gene_to_module, gene_to_color, module_to_color


def _finite_xy_arrays(x_values, y_values) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    return x[valid_mask], y[valid_mask]


def _draw_regression_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    point_color: str,
    ci_color: str | None = None,
    line_color: str = "#111111",
) -> None:
    ci_color = ci_color or point_color
    ax.scatter(
        x,
        y,
        s=28,
        color=point_color,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    if x.size < 2 or y.size < 2 or float(np.nanstd(x)) <= 0:
        return

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except (ValueError, np.linalg.LinAlgError):
        return

    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return

    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = intercept + slope * x_grid

    dof = int(x.size - 2)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if dof > 0 and sxx > 0:
        fitted = intercept + slope * x
        residual_ss = float(np.sum((y - fitted) ** 2))
        residual_se = np.sqrt(residual_ss / dof)
        t_value = float(t.ppf(0.975, dof))
        se_mean = residual_se * np.sqrt((1.0 / x.size) + ((x_grid - np.mean(x)) ** 2 / sxx))
        ci_delta = t_value * se_mean
        if np.isfinite(ci_delta).all():
            ax.fill_between(
                x_grid,
                y_grid - ci_delta,
                y_grid + ci_delta,
                color=ci_color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )

    ax.plot(x_grid, y_grid, color=line_color, linewidth=1.35, alpha=0.96, zorder=4)


def _regression_panel_figure(n_panels: int) -> tuple[plt.Figure, np.ndarray, int, int]:
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_width = 10.4 if n_cols == 2 else 5.4
    fig_height = max(3.8, 3.35 * n_rows + 0.45)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig._skip_default_tight_layout = True
    return fig, np.atleast_1d(axes).ravel(), n_rows, n_cols


def plot_top_edge_scatter_panels(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    top_n = int(top_n or cfg.top_pairs_plot_n)
    ranked = edge_df.sort_values(["EdgeWeight", "RRARank"], ascending=[False, True], kind="mergesort").head(top_n)
    if ranked.empty:
        return

    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else _metabolomics_df(engine.adata)
    gene_df = engine.gene_expression_df() if hasattr(engine, "gene_expression_df") else _gene_expression_df(engine.adata)
    _gene_to_module, gene_to_color, _module_to_color = _module_annotation_maps(engine)

    n_panels = len(ranked)
    fig, axes, _n_rows, _n_cols = _regression_panel_figure(n_panels)

    for ax, row in zip(axes, ranked.itertuples(index=False)):
        gene = str(row.Gene)
        metab = str(row.Metabolite)
        if gene not in gene_df.columns or metab not in metab_df.columns:
            ax.axis("off")
            continue

        x, y = _finite_xy_arrays(
            gene_df[gene].to_numpy(dtype=float, copy=False),
            metab_df[metab].to_numpy(dtype=float, copy=False),
        )

        if x.size < 2 or y.size < 2:
            ax.axis("off")
            continue

        module_color = gene_to_color.get(gene, PALETTE["gene"])
        _draw_regression_panel(
            ax=ax,
            x=x,
            y=y,
            point_color=module_color,
            ci_color=module_color,
            line_color="#111111",
        )
        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            r_value = float(np.corrcoef(x, y)[0, 1])
            r_text = f"r = {r_value:.2f}" if np.isfinite(r_value) else "r = NA"
        else:
            r_text = "r = NA"

        ax.text(
            0.03,
            0.97,
            r_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.set_title(f"{gene} vs {metab}", fontsize=10.2, pad=7)
        ax.set_xlabel(gene)
        ax.set_ylabel(metab)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Top Gene-Metabolite Association Pairs", y=0.992, fontsize=13)
    try:
        fig.tight_layout(rect=(0.015, 0.012, 0.985, 0.962), h_pad=1.35, w_pad=2.0)
    except Exception:
        fig.subplots_adjust(top=0.94, hspace=0.46, wspace=0.34)
    _save_figure(fig, save_stem, cfg)


def _module_top_metabolite_regression_rows(engine) -> pd.DataFrame:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return pd.DataFrame()
    if not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return pd.DataFrame()

    work = assoc_df.copy()
    work["Module"] = work["Module"].astype(str).str.strip()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work["SpearmanRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce")
    if "FDR" in work.columns:
        work["FDR"] = pd.to_numeric(work["FDR"], errors="coerce")
    if "PValue" in work.columns:
        work["PValue"] = pd.to_numeric(work["PValue"], errors="coerce")
    work = work.loc[work["Module"].ne("") & work["Metabolite"].ne("") & work["SpearmanRho"].notna()].copy()
    work = work.loc[work["Module"].str.lower() != "grey"].copy()
    if work.empty:
        return pd.DataFrame()

    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    summary_rows = pd.DataFrame()
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty:
        required_cols = {"Module", "TopMetabolite", "TopMetaboliteRho"}
        if required_cols.issubset(module_summary_df.columns):
            summary_rows = module_summary_df.loc[:, ["Module", "TopMetabolite", "TopMetaboliteRho"]].copy()
            summary_rows = summary_rows.rename(
                columns={"TopMetabolite": "Metabolite", "TopMetaboliteRho": "SpearmanRho"}
            )
            summary_rows["Module"] = summary_rows["Module"].astype(str).str.strip()
            summary_rows["Metabolite"] = summary_rows["Metabolite"].astype(str).str.strip()
            summary_rows["SpearmanRho"] = pd.to_numeric(summary_rows["SpearmanRho"], errors="coerce")
            summary_rows = summary_rows.loc[
                summary_rows["Module"].ne("")
                & summary_rows["Metabolite"].ne("")
                & summary_rows["SpearmanRho"].notna()
                & (summary_rows["Module"].str.lower() != "grey")
            ].copy()

    if not summary_rows.empty:
        ordered_rows = summary_rows.drop_duplicates(subset=["Module"], keep="first")
        available_pairs = set(zip(work["Module"].astype(str), work["Metabolite"].astype(str)))
        ordered_rows = ordered_rows.loc[
            [
                (str(row.Module), str(row.Metabolite)) in available_pairs
                for row in ordered_rows.itertuples(index=False)
            ]
        ].copy()
        if not ordered_rows.empty:
            return ordered_rows.reset_index(drop=True)

    significance_column = "FDR" if ("FDR" in work.columns and work["FDR"].notna().any()) else "PValue"
    if significance_column not in work.columns:
        work["PValue"] = np.nan
        significance_column = "PValue"

    return (
        work.assign(
            _AbsRho=work["SpearmanRho"].abs(),
            _SigRank=pd.to_numeric(work[significance_column], errors="coerce").fillna(1.0),
        )
        .sort_values(
            ["Module", "_AbsRho", "_SigRank", "Metabolite"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["Module"], keep="first")
        .drop(columns=["_AbsRho", "_SigRank"], errors="ignore")
        .reset_index(drop=True)
    )


def plot_module_top_metabolite_regression_panels(engine, save_stem: str | Path, cfg) -> None:
    pairs_df = _module_top_metabolite_regression_rows(engine)
    if pairs_df.empty:
        return

    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return

    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else _metabolomics_df(engine.adata)
    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return
    metab_df = metab_df.copy(deep=False)
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip(), name=metab_df.index.name)
    metab_df.columns = metab_df.columns.astype(str)

    _gene_to_module, _gene_to_color, module_to_color = _module_annotation_maps(engine)

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    order_lookup = {module_name: idx for idx, module_name in enumerate(module_order)}
    pairs_df = pairs_df.loc[
        pairs_df["Module"].astype(str).isin(eigengenes_df.columns.astype(str))
        & pairs_df["Metabolite"].astype(str).isin(metab_df.columns.astype(str))
    ].copy()
    if pairs_df.empty:
        return
    pairs_df["_Order"] = pairs_df["Module"].astype(str).map(order_lookup).fillna(len(order_lookup)).astype(int)
    pairs_df["_AbsRho"] = pd.to_numeric(pairs_df["SpearmanRho"], errors="coerce").abs()
    pairs_df = pairs_df.sort_values(["_Order", "_AbsRho", "Module"], ascending=[True, False, True], kind="mergesort")
    pairs_df = pairs_df.drop(columns=["_Order", "_AbsRho"], errors="ignore").reset_index(drop=True)

    shared_samples = eigengenes_df.index.intersection(metab_df.index)
    if len(shared_samples) < 2:
        return

    n_panels = len(pairs_df)
    fig, axes, _n_rows, _n_cols = _regression_panel_figure(n_panels)

    for ax, row in zip(axes, pairs_df.itertuples(index=False)):
        module_name = str(row.Module)
        metabolite = str(row.Metabolite)
        x, y = _finite_xy_arrays(
            eigengenes_df.loc[shared_samples, module_name].to_numpy(dtype=float, copy=False),
            metab_df.loc[shared_samples, metabolite].to_numpy(dtype=float, copy=False),
        )
        if x.size < 2 or y.size < 2:
            ax.axis("off")
            continue

        module_color = module_to_color.get(
            module_name,
            _build_circos_module_color_map([module_name]).get(module_name, "#9ca3af"),
        )
        _draw_regression_panel(
            ax=ax,
            x=x,
            y=y,
            point_color=module_color,
            ci_color=module_color,
            line_color="#111111",
        )

        rho_value = float(row.SpearmanRho) if pd.notna(row.SpearmanRho) else np.nan
        rho_text = f"rho = {rho_value:.2f}" if np.isfinite(rho_value) else "rho = NA"
        ax.text(
            0.03,
            0.97,
            rho_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.set_title(f"{module_name} module vs {metabolite}", fontsize=10.2, pad=7)
        ax.set_xlabel(f"{module_name} module eigengene")
        ax.set_ylabel(metabolite)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Module Eigengene and Top Metabolite Associations", y=0.992, fontsize=13)
    try:
        fig.tight_layout(rect=(0.015, 0.012, 0.985, 0.962), h_pad=1.35, w_pad=2.0)
    except Exception:
        fig.subplots_adjust(top=0.94, hspace=0.46, wspace=0.34)
    _save_figure(fig, save_stem, cfg)


__all__ = [
    "plot_top_edge_scatter_panels",
    "plot_module_top_metabolite_regression_panels",
]
