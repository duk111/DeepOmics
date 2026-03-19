from __future__ import annotations

import html
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
from sklearn.decomposition import PCA
from upsetplot import from_contents, plot as upset_plot

from .utils import safe_mkdir


def set_academic_style() -> None:
    """Apply a publication-oriented plotting style."""
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def _save_figure(fig: plt.Figure, save_stem: str | Path, cfg) -> None:
    """Save a figure in vector formats requested by the configuration."""
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)

    if cfg.export_pdf:
        fig.savefig(save_stem.with_suffix(".pdf"), bbox_inches="tight")
    if cfg.export_svg:
        fig.savefig(save_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _module_color_map(modules: list[str]) -> Dict[str, str]:
    """Create a stable module-to-color mapping."""
    unique_modules = [module for module in sorted(set(modules)) if module != "Unassigned"]
    palette = (
        sns.color_palette("tab20", n_colors=max(1, len(unique_modules))).as_hex()
        if unique_modules
        else []
    )
    color_map = {module: palette[idx] for idx, module in enumerate(unique_modules)}
    color_map["Unassigned"] = "#bdbdbd"
    return color_map


def _gene_expression_df(adata) -> pd.DataFrame:
    """Return transcriptome matrix as a sample-by-gene DataFrame."""
    return pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )


def _metabolomics_df(adata) -> pd.DataFrame:
    """Return metabolomics matrix as a sample-by-metabolite DataFrame."""
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        return metab_df.copy()
    return pd.DataFrame(
        np.asarray(metab_df, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=[str(x) for x in adata.uns.get("metabolite_names", [])],
    )


def _pick_display_features(engine, top_genes: int, top_metabolites: int) -> tuple[list[str], list[str]]:
    """Choose compact, publication-friendly feature subsets for multi-omics figures."""
    gene_df = _gene_expression_df(engine.adata)
    metab_df = _metabolomics_df(engine.adata)

    rra_df = engine.ml_results.get("key_genes_rra", pd.DataFrame())
    if isinstance(rra_df, pd.DataFrame) and not rra_df.empty:
        gene_candidates = [g for g in rra_df["Gene"].astype(str).tolist() if g in gene_df.columns]
    else:
        gene_candidates = []
    if len(gene_candidates) < top_genes:
        variance_rank = gene_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        gene_candidates.extend([g for g in variance_rank if g not in gene_candidates])
    selected_genes = gene_candidates[:top_genes]

    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        summary_df = summary_df.sort_values(["RRA_Genes", "Candidate_Genes_PCC"], ascending=[False, False])
        metabolite_candidates = [m for m in summary_df["Metabolite"].astype(str).tolist() if m in metab_df.columns]
    else:
        metabolite_candidates = []
    if len(metabolite_candidates) < top_metabolites:
        variance_rank = metab_df.var(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        metabolite_candidates.extend([m for m in variance_rank if m not in metabolite_candidates])
    selected_metabs = metabolite_candidates[:top_metabolites]

    return selected_genes, selected_metabs


def _text_rotation_for_angle(angle_deg: float) -> tuple[float, str]:
    """Return readable label rotation and alignment for circular layouts."""
    if 90 < angle_deg < 270:
        return angle_deg + 180, "right"
    return angle_deg, "left"


def _plot_pca_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
) -> None:
    """Plot a simple 2D PCA scatter."""
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=cfg.random_state)
    coords = pca.fit_transform(matrix)
    var_exp = pca.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=32, alpha=0.85)
    for x, y, label in zip(coords[:, 0], coords[:, 1], sample_names):
        ax.text(x, y, label, fontsize=7, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    _save_figure(fig, save_stem, cfg)


def plot_sample_dendrogram(adata, save_stem: str | Path, cfg) -> None:
    """Plot a sample clustering dendrogram based on transcriptome profiles."""
    if adata.n_obs < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    linkage_matrix = linkage(np.asarray(adata.X, dtype=np.float32), method="average")
    dendrogram(
        linkage_matrix,
        labels=adata.obs_names.tolist(),
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
    )
    ax.set_title("Sample Clustering Dendrogram")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Euclidean Distance")
    _save_figure(fig, save_stem, cfg)


def plot_transcriptome_pca(adata, save_stem: str | Path, cfg) -> None:
    """Plot PCA for the transcriptome matrix."""
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
    )


def plot_metabolome_pca(adata, save_stem: str | Path, cfg) -> None:
    """Plot PCA for the metabolomics matrix."""
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        matrix = metab_df.to_numpy(dtype=np.float32, copy=False)
    else:
        matrix = np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
    )


def plot_correlation_circle(engine, save_stem: str | Path, cfg) -> None:
    """Plot a PCA-based correlation circle for selected genes and metabolites."""
    gene_names, metabolite_names = _pick_display_features(
        engine,
        top_genes=cfg.correlation_circle_top_genes,
        top_metabolites=cfg.correlation_circle_top_metabolites,
    )
    if len(gene_names) < 2 or len(metabolite_names) < 1:
        return

    gene_df = _gene_expression_df(engine.adata).loc[:, gene_names]
    metab_df = _metabolomics_df(engine.adata).loc[:, metabolite_names]
    combined = pd.concat([gene_df, metab_df], axis=1)
    if combined.shape[0] < 3 or combined.shape[1] < 3:
        return

    X = combined.to_numpy(dtype=float, copy=False)
    Xz = np.nan_to_num(zscore(X, axis=0, ddof=1), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=cfg.random_state)
    scores = pca.fit_transform(Xz)
    score_z = np.nan_to_num(zscore(scores, axis=0, ddof=1), nan=0.0, posinf=0.0, neginf=0.0)
    corr_coords = (Xz.T @ score_z) / max(1, Xz.shape[0] - 1)

    feature_types = ["Gene"] * len(gene_names) + ["Metabolite"] * len(metabolite_names)
    feature_df = pd.DataFrame(
        {
            "Feature": combined.columns.astype(str),
            "PC1": corr_coords[:, 0],
            "PC2": corr_coords[:, 1],
            "Type": feature_types,
        }
    )
    feature_df["Radius"] = np.sqrt(feature_df["PC1"] ** 2 + feature_df["PC2"] ** 2)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", linewidth=1.0, color="#7f7f7f")
    ax.add_patch(circle)
    ax.axhline(0, color="#bdbdbd", linewidth=0.8)
    ax.axvline(0, color="#bdbdbd", linewidth=0.8)

    palette = {"Gene": "#1f77b4", "Metabolite": "#d62728"}
    for feature_type, subset in feature_df.groupby("Type"):
        ax.scatter(subset["PC1"], subset["PC2"], s=28, label=feature_type, color=palette[feature_type], alpha=0.9)
        for _, row in subset.iterrows():
            ax.arrow(0, 0, row["PC1"], row["PC2"], color=palette[feature_type], alpha=0.55,
                     linewidth=0.8, head_width=0.02, length_includes_head=True)
            ax.text(row["PC1"] * 1.07, row["PC2"] * 1.07, row["Feature"], fontsize=7.5,
                    color=palette[feature_type], ha="center", va="center")

    var_exp = pca.explained_variance_ratio_ * 100.0
    ax.set_title("Correlation Circle of Prioritized Multi-Omics Features")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, save_stem, cfg)


def plot_circos_grn(engine, save_stem: str | Path, cfg) -> None:
    """Plot a compact Circos-like GRN for top prioritized gene-metabolite edges."""
    edge_df = engine.ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "In_RRA", "AbsPCC"], ascending=[False, False, False]
    )
    top_edges = ranked.head(cfg.circos_top_edges).copy()
    if top_edges.empty:
        return

    genes = top_edges["Gene"].astype(str).drop_duplicates().tolist()
    metabs = top_edges["Metabolite"].astype(str).drop_duplicates().tolist()
    if len(genes) < 2 or len(metabs) < 1:
        return

    module_df = engine.wgcna_results.get("Gene_Modules", pd.DataFrame())
    gene_to_module = {}
    if isinstance(module_df, pd.DataFrame) and not module_df.empty:
        gene_to_module = dict(zip(module_df["Gene"].astype(str), module_df["Module"].astype(str)))
    module_colors = _module_color_map(list(gene_to_module.values()) or ["Unassigned"])

    gene_angles = np.linspace(np.pi * 0.58, np.pi * 1.42, len(genes))
    metab_angles = np.linspace(-np.pi * 0.42, np.pi * 0.42, len(metabs))
    gene_pos = {gene: (np.cos(theta), np.sin(theta), theta) for gene, theta in zip(genes, gene_angles)}
    metab_pos = {metab: (np.cos(theta), np.sin(theta), theta) for metab, theta in zip(metabs, metab_angles)}

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    outer = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.3, color="#4b5563")
    inner = plt.Circle((0, 0), 0.88, fill=False, linewidth=0.7, linestyle=":", color="#9ca3af")
    ax.add_patch(outer)
    ax.add_patch(inner)

    for _, row in top_edges.iterrows():
        gene = str(row["Gene"])
        metab = str(row["Metabolite"])
        x1, y1, _ = gene_pos[gene]
        x2, y2, _ = metab_pos[metab]
        ctrl_scale = 0.18 + 0.05 * float(row["Support_Count"])
        verts = [(x1 * 0.98, y1 * 0.98), (0.0, ctrl_scale * np.sign(y1 + y2 + 1e-6)), (x2 * 0.98, y2 * 0.98)]
        color = "#d73027" if float(row["PCC_R"]) >= 0 else "#4575b4"
        width = 0.5 + 2.0 * min(1.0, abs(float(row["PCC_R"])))
        alpha = 0.30 + 0.18 * int(row["Support_Count"])
        patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(verts, [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3]),
            facecolor="none", edgecolor=color, linewidth=width, alpha=min(alpha, 0.9),
        )
        ax.add_patch(patch)

    for gene, (x, y, theta) in gene_pos.items():
        module = gene_to_module.get(gene, "Unassigned")
        color = module_colors.get(module, "#bdbdbd")
        ax.scatter([x], [y], s=36, color=color, zorder=3)
        angle_deg = np.degrees(theta)
        rotation, ha = _text_rotation_for_angle(angle_deg)
        ax.text(x * 1.10, y * 1.10, gene, fontsize=7.2, rotation=rotation, ha=ha, va="center")

    for metab, (x, y, theta) in metab_pos.items():
        ax.scatter([x], [y], s=42, color="#111827", zorder=3)
        angle_deg = np.degrees(theta)
        rotation, ha = _text_rotation_for_angle(angle_deg)
        ax.text(x * 1.10, y * 1.10, metab, fontsize=7.2, rotation=rotation, ha=ha, va="center")

    ax.text(-1.22, 1.08, "Genes", fontsize=11, fontweight="bold", color="#374151")
    ax.text(0.92, 1.08, "Metabolites", fontsize=11, fontweight="bold", color="#374151")
    ax.set_title("Circos GRN of Prioritized Gene-Metabolite Associations", pad=18)
    _save_figure(fig, save_stem, cfg)


def plot_complex_gene_metabolite_heatmap(engine, save_stem: str | Path, cfg) -> None:
    """Plot a clustered gene-metabolite heatmap with module and metabolite annotations."""
    edge_df = engine.ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "In_RRA", "AbsPCC"], ascending=[False, False, False]
    )
    top_edges = ranked.head(max(cfg.complex_heatmap_top_genes * cfg.complex_heatmap_top_metabolites, 120)).copy()
    if top_edges.empty:
        return

    top_genes = top_edges["Gene"].astype(str).value_counts().head(cfg.complex_heatmap_top_genes).index.tolist()
    top_metabs = top_edges["Metabolite"].astype(str).value_counts().head(cfg.complex_heatmap_top_metabolites).index.tolist()
    heat_df = (
        top_edges.loc[
            top_edges["Gene"].astype(str).isin(top_genes) & top_edges["Metabolite"].astype(str).isin(top_metabs),
            ["Gene", "Metabolite", "PCC_R"],
        ]
        .drop_duplicates(subset=["Gene", "Metabolite"])
        .pivot(index="Gene", columns="Metabolite", values="PCC_R")
        .reindex(index=top_genes, columns=top_metabs)
        .fillna(0.0)
    )
    if heat_df.shape[0] < 2 or heat_df.shape[1] < 2:
        return

    module_df = engine.wgcna_results.get("Gene_Modules", pd.DataFrame())
    gene_to_module = {}
    if isinstance(module_df, pd.DataFrame) and not module_df.empty:
        gene_to_module = dict(zip(module_df["Gene"].astype(str), module_df["Module"].astype(str)))
    row_modules = [gene_to_module.get(gene, "Unassigned") for gene in heat_df.index.astype(str)]
    module_colors = _module_color_map(row_modules)
    row_colors = pd.DataFrame({"Module": [module_colors.get(module, "#bdbdbd") for module in row_modules]}, index=heat_df.index)

    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    metab_strength = {}
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and "Metabolite" in summary_df.columns:
        metab_strength = dict(zip(summary_df["Metabolite"].astype(str), summary_df["RRA_Genes"].astype(float)))
    col_strength = np.array([metab_strength.get(metab, 0.0) for metab in heat_df.columns.astype(str)], dtype=float)
    if np.ptp(col_strength) == 0:
        col_colors = pd.DataFrame({"RRA_Genes": ["#9ecae1"] * len(col_strength)}, index=heat_df.columns)
    else:
        cmap = matplotlib.cm.get_cmap("Greens")
        normalized = (col_strength - col_strength.min()) / np.ptp(col_strength)
        col_colors = pd.DataFrame({"RRA_Genes": [matplotlib.colors.to_hex(cmap(0.35 + 0.55 * v)) for v in normalized]}, index=heat_df.columns)

    cluster = sns.clustermap(
        heat_df,
        cmap="RdBu_r",
        center=0,
        linewidths=0.2,
        row_colors=row_colors,
        col_colors=col_colors,
        figsize=(max(9, 0.55 * heat_df.shape[1] + 3), max(8, 0.28 * heat_df.shape[0] + 3)),
        cbar_kws={"label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )
    cluster.fig.suptitle("Complex Heatmap of Prioritized Gene-Metabolite Associations", y=1.02)
    cluster.ax_heatmap.set_xlabel("Metabolites")
    cluster.ax_heatmap.set_ylabel("Genes")
    _save_figure(cluster.fig, save_stem, cfg)


def plot_key_genes_upset(ml_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot an UpSet diagram of key genes from multiple ranking strategies."""
    contents = {}
    for strategy in ("intersection", "borda", "rra"):
        df = ml_results.get(f"key_genes_{strategy}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            contents[strategy.upper()] = set(df["Gene"].astype(str).tolist())

    if not contents:
        return

    upset_input = from_contents(contents)
    fig = plt.figure(figsize=(10, 6))
    upset_plot(upset_input, fig=fig, element_size=None)
    fig.suptitle("Overlap of Key Genes Across Integration Strategies", y=1.02)
    _save_figure(fig, save_stem, cfg)


def plot_metabolite_selection_summary(ml_results: dict, save_stem: str | Path, cfg, top_n: int = 20) -> None:
    """Plot candidate and final key-gene counts per metabolite."""
    summary_df = ml_results.get("metabolite_summary")
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return

    plot_df = summary_df.sort_values(["RRA_Genes", "Candidate_Genes_PCC"], ascending=[False, False]).head(top_n).copy()
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(plot_df)), 6))
    ax.bar(x - 1.5 * width, plot_df["Candidate_Genes_PCC"], width=width, label="PCC candidates")
    ax.bar(x - 0.5 * width, plot_df["Intersection_Genes"], width=width, label="Intersection")
    ax.bar(x + 0.5 * width, plot_df["Borda_Genes"], width=width, label="Borda")
    ax.bar(x + 1.5 * width, plot_df["RRA_Genes"], width=width, label="RRA")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Metabolite"], rotation=90)
    ax.set_ylabel("Gene count")
    ax.set_title(f"Metabolite-Level Feature Selection Summary (Top {len(plot_df)})")
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, save_stem, cfg)


def plot_gene_metabolite_heatmap(ml_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot a correlation heatmap for the strongest gene-metabolite pairs."""
    edge_df = ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "AbsPCC"], ascending=[False, False]
    )
    top_edges = ranked.head(80)
    if top_edges.empty:
        return

    top_genes = top_edges["Gene"].astype(str).value_counts().head(25).index.tolist()
    top_metabs = top_edges["Metabolite"].astype(str).value_counts().head(12).index.tolist()

    heat_df = (
        top_edges.loc[
            top_edges["Gene"].astype(str).isin(top_genes) & top_edges["Metabolite"].astype(str).isin(top_metabs),
            ["Gene", "Metabolite", "PCC_R"],
        ]
        .drop_duplicates(subset=["Gene", "Metabolite"])
        .pivot(index="Gene", columns="Metabolite", values="PCC_R")
        .reindex(index=top_genes, columns=top_metabs)
        .fillna(0.0)
    )
    if heat_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * heat_df.shape[1]), max(6, 0.35 * heat_df.shape[0])))
    sns.heatmap(
        heat_df,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
        linewidths=0.2,
    )
    ax.set_title("Strong Gene-Metabolite Associations")
    ax.set_xlabel("Metabolites")
    ax.set_ylabel("Genes")
    _save_figure(fig, save_stem, cfg)


def plot_top_edge_scatter_panels(engine, save_stem: str | Path, cfg, top_n: int = 6) -> None:
    """Plot scatter panels for the strongest prioritized gene-metabolite pairs."""
    edge_df = engine.ml_results.get("grn_edges_df")
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    ranked = edge_df.assign(AbsPCC=edge_df["PCC_R"].abs()).sort_values(
        ["Support_Count", "AbsPCC"], ascending=[False, False]
    )
    top_edges = ranked.head(top_n)
    if top_edges.empty:
        return

    metab_df = engine.adata.obsm.get("metabolomics_scaled", engine.adata.obsm.get("metabolomics"))
    if not isinstance(metab_df, pd.DataFrame):
        metab_df = pd.DataFrame(metab_df, index=engine.adata.obs_names, columns=engine.adata.uns["metabolite_names"])
    gene_df = pd.DataFrame(
        np.asarray(engine.adata.X, dtype=np.float32),
        index=engine.adata.obs_names,
        columns=engine.adata.var_names.astype(str),
    )

    n_panels = len(top_edges)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, top_edges.iterrows()):
        gene = str(row["Gene"])
        metab = str(row["Metabolite"])
        if gene not in gene_df.columns or metab not in metab_df.columns:
            ax.axis("off")
            continue
        x = gene_df[gene].to_numpy(dtype=float, copy=False)
        y = metab_df[metab].to_numpy(dtype=float, copy=False)
        ax.scatter(x, y, s=28, alpha=0.85)
        if np.unique(x).size > 1:
            coeffs = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, linewidth=1.5)
        ax.set_title(f"{gene} vs {metab}\nSupport={int(row['Support_Count'])}, r={row['PCC_R']:.2f}")
        ax.set_xlabel(gene)
        ax.set_ylabel(metab)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Top Gene-Metabolite Pair Scatter Plots", y=1.01)
    _save_figure(fig, save_stem, cfg)


def plot_wgcna_soft_threshold(wgcna_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot WGCNA soft-threshold diagnostics."""
    power_df = wgcna_results.get("Power_Selection")
    if not isinstance(power_df, pd.DataFrame) or power_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(power_df["Power"], power_df["ScaleFreeR2"], marker="o")
    chosen = power_df.loc[power_df["Chosen"]]
    if not chosen.empty:
        axes[0].axvline(float(chosen.iloc[0]["Power"]), linestyle="--")
    axes[0].set_xlabel("Soft-threshold power")
    axes[0].set_ylabel("Scale-free fit (R²)")
    axes[0].set_title("Scale-Free Topology Fit")

    axes[1].plot(power_df["Power"], power_df["MeanConnectivity"], marker="o")
    if not chosen.empty:
        axes[1].axvline(float(chosen.iloc[0]["Power"]), linestyle="--")
    axes[1].set_xlabel("Soft-threshold power")
    axes[1].set_ylabel("Mean connectivity")
    axes[1].set_title("Mean Connectivity")

    fig.suptitle("WGCNA Power Selection Diagnostics")
    _save_figure(fig, save_stem, cfg)


def plot_wgcna_gene_dendrogram_modules(wgcna_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot a gene dendrogram with module-color annotation."""
    linkage_matrix = wgcna_results.get("Linkage_Matrix")
    ordered_df = wgcna_results.get("Ordered_Gene_Modules")
    if linkage_matrix is None or not isinstance(ordered_df, pd.DataFrame) or ordered_df.empty:
        return

    color_map = _module_color_map(ordered_df["Module"].astype(str).tolist())

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 0.55], hspace=0.05)
    ax_tree = fig.add_subplot(gs[0, 0])
    dendrogram(linkage_matrix, no_labels=True, color_threshold=0, ax=ax_tree)
    ax_tree.set_title("WGCNA Gene Dendrogram and Module Colors")
    ax_tree.set_ylabel("Dissimilarity")

    ax_bar = fig.add_subplot(gs[1, 0])
    ordered_modules = ordered_df.sort_values("Dendrogram_Order")["Module"].astype(str).tolist()
    rgb = np.array([mcolors.to_rgb(color_map.get(module, "#bdbdbd")) for module in ordered_modules])[np.newaxis, :, :]
    ax_bar.imshow(rgb, aspect="auto")
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    ax_bar.set_xlabel("Genes ordered by dendrogram")

    _save_figure(fig, save_stem, cfg)


def plot_module_trait_heatmap(wgcna_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot a module-trait correlation heatmap with FDR annotations."""
    corr_df = wgcna_results.get("Trait_Correlation")
    fdr_df = wgcna_results.get("Trait_FDR")
    if not isinstance(corr_df, pd.DataFrame) or corr_df.empty:
        return

    annot = None
    if isinstance(fdr_df, pd.DataFrame) and not fdr_df.empty:
        annot = np.empty(corr_df.shape, dtype=object)
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                annot[i, j] = f"{corr_df.iloc[i, j]:.2f}\nFDR={fdr_df.iloc[i, j]:.2g}"

    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * corr_df.shape[1]), max(6, 0.40 * corr_df.shape[0]))
    )
    sns.heatmap(
        corr_df,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Module-Trait Correlation Heatmap")
    ax.set_xlabel("Modules")
    ax.set_ylabel("Metabolites")
    _save_figure(fig, save_stem, cfg)


def plot_module_eigengene_heatmap(wgcna_results: dict, save_stem: str | Path, cfg) -> None:
    """Plot a module eigengene correlation heatmap."""
    me_df = wgcna_results.get("ME_df")
    if not isinstance(me_df, pd.DataFrame) or me_df.empty or me_df.shape[1] < 2:
        return

    corr_df = me_df.corr()
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * corr_df.shape[1]), max(6, 0.8 * corr_df.shape[0])))
    sns.heatmap(
        corr_df,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Eigengene correlation"},
    )
    ax.set_title("Module Eigengene Correlation")
    _save_figure(fig, save_stem, cfg)


def plot_top_rra_genes(ml_results: dict, save_stem: str | Path, cfg, top_n: int = 20) -> None:
    """Plot the most recurrent RRA-prioritized genes."""
    rra_df = ml_results.get("key_genes_rra")
    if not isinstance(rra_df, pd.DataFrame) or rra_df.empty:
        return

    top_df = rra_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top_df))))
    ax.barh(top_df["Gene"], top_df["Associated_Metabolites_Count"])
    ax.set_title("Top RRA-Prioritized Genes")
    ax.set_xlabel("Associated Metabolite Count")
    ax.set_ylabel("Gene")
    _save_figure(fig, save_stem, cfg)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact DataFrame as GitHub-flavored Markdown without extra deps."""
    if df.empty:
        return "_No data available._"

    preview = df.head(max_rows).copy()
    preview = preview.fillna("")
    columns = preview.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in preview.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def generate_markdown_report(engine, cfg, report_path: str | Path) -> None:
    """Generate a Markdown analysis report."""
    ml_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    rra_df = engine.ml_results.get("key_genes_rra", pd.DataFrame())
    module_summary = engine.wgcna_results.get("Module_Summary", pd.DataFrame())
    hub_df = engine.wgcna_results.get("Hub_Genes", pd.DataFrame())
    power_df = engine.wgcna_results.get("Power_Selection", pd.DataFrame())

    lines = [
        f"# DeepOmics Report: {cfg.project_name}",
        "",
        "## Run Summary",
        f"- Samples: {engine.adata.n_obs}",
        f"- Genes: {engine.adata.n_vars}",
        f"- Metabolites: {len(engine.adata.uns.get('metabolite_names', []))}",
        f"- Output directory: `{cfg.output_dir}`",
        f"- Selected WGCNA power: `{engine.wgcna_results.get('Selected_Power', 'NA')}`",
        f"- TOM enabled for WGCNA: `{engine.wgcna_results.get('Adjacency_Used_TOM', False)}`",
        "",
        "## Main Tables",
        "- `GRN_Edges_Full.csv`: full gene-metabolite edge table with support indicators.",
        "- `GRN_Edges_Cytoscape.csv`: Cytoscape-ready network table.",
        "- `Key_Genes_Intersection.csv`: strict overlap genes across all three models.",
        "- `Key_Genes_Borda.csv`: Borda-aggregated key-gene summary.",
        "- `Key_Genes_Rra.csv`: robust rank aggregation summary.",
        "- `ML_Metabolite_Summary.csv`: metabolite-level screening and selection counts.",
        "- `WGCNA_Power_Selection.csv`: power scan diagnostics and chosen power.",
        "- `WGCNA_Module_Trait_Correlation.csv`: module-trait correlation matrix.",
        "- `WGCNA_Module_Trait_PValue.csv`: module-trait p-value matrix.",
        "- `WGCNA_Module_Trait_FDR.csv`: FDR-adjusted module-trait p-value matrix.",
        "- `WGCNA_Gene_Statistics.csv`: module membership, connectivity and top-trait gene significance.",
        "- `WGCNA_Hub_Genes.csv`: top hub genes per module.",
        "",
        "## Metabolite-Level Summary",
        _df_to_markdown(ml_summary, max_rows=20),
        "",
        "## Top RRA Genes",
        _df_to_markdown(rra_df, max_rows=20),
        "",
        "## WGCNA Module Summary",
        _df_to_markdown(module_summary, max_rows=20),
        "",
        "## WGCNA Power Selection",
        _df_to_markdown(power_df, max_rows=20),
        "",
        "## WGCNA Hub Genes",
        _df_to_markdown(hub_df, max_rows=20),
        "",
        "## Generated Figures",
        "- `plots/sample_clustering_dendrogram.pdf|svg`",
        "- `plots/transcriptome_pca.pdf|svg`",
        "- `plots/metabolome_pca.pdf|svg`",
        "- `plots/key_genes_overlap_upset.pdf|svg`",
        "- `plots/metabolite_selection_summary.pdf|svg`",
        "- `plots/gene_metabolite_correlation_heatmap.pdf|svg`",
        "- `plots/top_gene_metabolite_pairs.pdf|svg`",
        "- `plots/wgcna_soft_threshold_diagnostics.pdf|svg`",
        "- `plots/wgcna_gene_dendrogram_modules.pdf|svg`",
        "- `plots/wgcna_module_trait_heatmap.pdf|svg`",
        "- `plots/wgcna_module_eigengene_heatmap.pdf|svg`",
        "- `plots/top_rra_genes.pdf|svg`",
        "- `plots/correlation_circle.pdf|svg`",
        "- `plots/circos_grn.pdf|svg`",
        "- `plots/complex_gene_metabolite_heatmap.pdf|svg`",
    ]

    report_path = Path(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(engine, cfg, report_path: str | Path) -> None:
    """Generate a lightweight HTML analysis report."""
    ml_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame()).head(50)
    rra_df = engine.ml_results.get("key_genes_rra", pd.DataFrame()).head(50)
    module_summary = engine.wgcna_results.get("Module_Summary", pd.DataFrame()).head(50)
    hub_df = engine.wgcna_results.get("Hub_Genes", pd.DataFrame()).head(50)
    power_df = engine.wgcna_results.get("Power_Selection", pd.DataFrame()).head(50)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Report - {html.escape(cfg.project_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
    h1, h2 {{ color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 6px; }}
  </style>
</head>
<body>
  <h1>DeepOmics Report: {html.escape(cfg.project_name)}</h1>
  <h2>Run Summary</h2>
  <ul>
    <li>Samples: {engine.adata.n_obs}</li>
    <li>Genes: {engine.adata.n_vars}</li>
    <li>Metabolites: {len(engine.adata.uns.get("metabolite_names", []))}</li>
    <li>Output directory: <code>{html.escape(str(cfg.output_dir))}</code></li>
    <li>Selected WGCNA power: <code>{html.escape(str(engine.wgcna_results.get("Selected_Power", "NA")))}</code></li>
    <li>TOM enabled for WGCNA: <code>{html.escape(str(engine.wgcna_results.get("Adjacency_Used_TOM", False)))}</code></li>
  </ul>

  <h2>Metabolite-Level Summary</h2>
  {ml_summary.to_html(index=False, escape=True)}

  <h2>Top RRA Genes</h2>
  {rra_df.to_html(index=False, escape=True)}

  <h2>WGCNA Module Summary</h2>
  {module_summary.to_html(index=False, escape=True)}

  <h2>WGCNA Power Selection</h2>
  {power_df.to_html(index=False, escape=True)}

  <h2>WGCNA Hub Genes</h2>
  {hub_df.to_html(index=False, escape=True)}
</body>
</html>
"""
    Path(report_path).write_text(html_text, encoding="utf-8")


def generate_report_plots(engine, cfg) -> None:
    """Generate vector figures and summary reports."""
    set_academic_style()
    plots_dir = safe_mkdir(Path(cfg.output_dir) / "plots")

    plot_sample_dendrogram(engine.adata, plots_dir / "sample_clustering_dendrogram", cfg)
    plot_transcriptome_pca(engine.adata, plots_dir / "transcriptome_pca", cfg)
    plot_metabolome_pca(engine.adata, plots_dir / "metabolome_pca", cfg)

    plot_key_genes_upset(engine.ml_results, plots_dir / "key_genes_overlap_upset", cfg)
    plot_metabolite_selection_summary(engine.ml_results, plots_dir / "metabolite_selection_summary", cfg)
    plot_gene_metabolite_heatmap(engine.ml_results, plots_dir / "gene_metabolite_correlation_heatmap", cfg)
    plot_complex_gene_metabolite_heatmap(engine, plots_dir / "complex_gene_metabolite_heatmap", cfg)
    plot_correlation_circle(engine, plots_dir / "correlation_circle", cfg)
    plot_circos_grn(engine, plots_dir / "circos_grn", cfg)
    plot_top_edge_scatter_panels(engine, plots_dir / "top_gene_metabolite_pairs", cfg)
    plot_top_rra_genes(engine.ml_results, plots_dir / "top_rra_genes", cfg)

    plot_wgcna_soft_threshold(engine.wgcna_results, plots_dir / "wgcna_soft_threshold_diagnostics", cfg)
    plot_wgcna_gene_dendrogram_modules(engine.wgcna_results, plots_dir / "wgcna_gene_dendrogram_modules", cfg)
    plot_module_trait_heatmap(engine.wgcna_results, plots_dir / "wgcna_module_trait_heatmap", cfg)
    plot_module_eigengene_heatmap(engine.wgcna_results, plots_dir / "wgcna_module_eigengene_heatmap", cfg)

    notes = (
        "Recommended downstream usage:\n"
        "1. Use WGCNA_Power_Selection.csv and the power diagnostic figure to report how the soft threshold was chosen.\n"
        "2. Use WGCNA_Hub_Genes.csv together with Key_Genes_Rra.csv to highlight convergent candidates.\n"
        "3. Use WGCNA_Module_Trait_FDR.csv rather than raw p-values for manuscript-level claims.\n"
        "4. Import GRN_Edges_Cytoscape.csv into Cytoscape for final network rendering.\n"
    )
    (plots_dir / "visualization_notes.txt").write_text(notes, encoding="utf-8")

    if cfg.generate_reports:
        if "md" in cfg.report_formats:
            generate_markdown_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.md")
        if "html" in cfg.report_formats:
            generate_html_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.html")
