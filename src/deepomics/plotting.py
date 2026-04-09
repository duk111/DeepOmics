
from __future__ import annotations

import html
from pathlib import Path

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .utils import get_logger, safe_mkdir

logger = get_logger()


PALETTE = {
    "gene": "#2563eb",
    "metabolite": "#111827",
    "edge_positive": "#dc2626",
    "edge_negative": "#2563eb",
    "grid_aux": "#cbd5e1",
    "pca_scatter": "#4c78a8",
    "bar_total": "#93c5fd",
    "bar_dual_model": "#2563eb",
    "bar_high_conf": "#1d4ed8",
}

FIGURE_FILE_PREFIXES = {
    "sample_clustering_dendrogram": "F01_Sample_Clustering_Dendrogram",
    "transcriptome_pca": "F02_Transcriptome_PCA",
    "metabolome_pca": "F03_Metabolome_PCA",
    "total_association_network": "F04_Total_Association_Network",
    "high_confidence_network": "F05_High_Confidence_Network",
    "top_gene_metabolite_pairs": "F06_Top_Gene_Metabolite_Pairs",
    "metabolite_model_support_summary": "F07_Metabolite_Model_Support_Summary",
    "top_key_genes": "F08_Top_Key_Genes",
}

TABLE_FILE_PREFIXES = {
    "gene_scores": "T01_Metabolite_Gene_Scoring_Table.csv",
    "total_network": "T02_Total_Association_Network.csv",
    "high_confidence_network": "T03_High_Confidence_Network.csv",
    "key_gene_summary": "T04_Key_Gene_Summary.csv",
    "metabolite_summary": "T05_Metabolite_Association_Summary.csv",
    "cytoscape_network": "T06_Association_Network_Cytoscape.csv",
}


def set_academic_style() -> None:
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.15


def _save_figure(fig: plt.Figure, save_stem: str | Path, cfg) -> None:
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    if cfg.export_pdf:
        fig.savefig(save_stem.with_suffix(".pdf"))
    if cfg.export_svg:
        fig.savefig(save_stem.with_suffix(".svg"))
    if getattr(cfg, "export_png", True):
        fig.savefig(save_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _gene_expression_df(adata) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )


def _metabolomics_df(adata) -> pd.DataFrame:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        return metab_df.copy(deep=False)
    return pd.DataFrame(
        np.asarray(metab_df, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=[str(x) for x in adata.uns.get("metabolite_names", [])],
    )


def _plot_pca_from_matrix(matrix: np.ndarray, sample_names: list[str], title: str, save_stem: str | Path, cfg) -> None:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=cfg.random_state)
    coords = pca.fit_transform(matrix)
    var_exp = pca.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
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

    if len(sample_names) <= 20 and adjust_text is not None:
        texts = [ax.text(x, y, label, fontsize=8, alpha=0.90) for x, y, label in zip(coords[:, 0], coords[:, 1], sample_names)]
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grid_aux"], "lw": 0.5})

    ax.axhline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    _save_figure(fig, save_stem, cfg)


def plot_sample_dendrogram(adata, save_stem: str | Path, cfg) -> None:
    if adata.n_obs < 2:
        return

    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, ax = plt.subplots(figsize=(max(12, adata.n_obs * 0.12), 6))
    linkage_matrix = linkage(np.asarray(adata.X, dtype=np.float32), method="average")
    color_threshold = float(np.percentile(linkage_matrix[:, 2], 50))

    dendrogram(
        linkage_matrix,
        labels=adata.obs_names.tolist(),
        leaf_rotation=90,
        leaf_font_size=max(4, min(9, 800 // max(1, adata.n_obs))),
        color_threshold=color_threshold,
        above_threshold_color="#aaaaaa",
        ax=ax,
    )
    ax.set_title("Sample Clustering Dendrogram")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Euclidean Distance")
    _save_figure(fig, save_stem, cfg)


def plot_transcriptome_pca(adata, save_stem: str | Path, cfg) -> None:
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
    )


def plot_metabolome_pca(adata, save_stem: str | Path, cfg) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
    )


def _prepare_network_df(engine, tier: str, max_edges: int | None = None) -> pd.DataFrame:
    if tier == "total":
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    else:
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())

    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return pd.DataFrame()

    ordered = edge_df.sort_values(
        ["EdgeWeight", "RRARank", "ModelSupportCount", "ScreenSupportCount"],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    if max_edges is not None:
        ordered = ordered.head(max(1, int(max_edges)))
    return ordered.copy()


def _plot_association_network(edge_df: pd.DataFrame, title: str, save_stem: str | Path, cfg) -> None:
    if edge_df.empty:
        return

    genes = (
        edge_df.groupby("Gene")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    )
    metabolites = (
        edge_df.groupby("Metabolite")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    )

    if not genes or not metabolites:
        return

    fig_width = max(10, 0.17 * (len(genes) + len(metabolites)) + 6)
    fig_height = max(6.5, 0.24 * max(len(genes), len(metabolites)) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    gene_x = 0.0
    metabolite_x = 1.0
    gene_y = np.linspace(len(genes), 1, len(genes))
    metabolite_y = np.linspace(len(metabolites), 1, len(metabolites))
    gene_pos = {gene: (gene_x, float(y)) for gene, y in zip(genes, gene_y)}
    metabolite_pos = {metab: (metabolite_x, float(y)) for metab, y in zip(metabolites, metabolite_y)}

    for row in edge_df.itertuples(index=False):
        gene = str(row.Gene)
        metab = str(row.Metabolite)
        if gene not in gene_pos or metab not in metabolite_pos:
            continue
        x1, y1 = gene_pos[gene]
        x2, y2 = metabolite_pos[metab]
        color = PALETTE["edge_positive"] if str(row.Sign) == "positive" else PALETTE["edge_negative"]
        width = 0.8 + 4.2 * float(row.EdgeWeight)
        opacity = min(
            0.95,
            0.20
            + 0.35 * (float(row.ModelSupportCount) / 2.0)
            + 0.20 * (float(row.ScreenSupportCount) / 3.0),
        )
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=opacity, solid_capstyle="round", zorder=1)

    ax.scatter(
        [gene_x] * len(genes),
        gene_y,
        s=70,
        c=PALETTE["gene"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="Gene",
    )
    ax.scatter(
        [metabolite_x] * len(metabolites),
        metabolite_y,
        s=85,
        c=PALETTE["metabolite"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="Metabolite",
    )

    for gene, (x, y) in gene_pos.items():
        ax.text(x - 0.02, y, gene, ha="right", va="center", fontsize=8.5)
    for metab, (x, y) in metabolite_pos.items():
        ax.text(x + 0.02, y, metab, ha="left", va="center", fontsize=8.5)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0, max(len(genes), len(metabolites)) + 1.2)
    ax.set_xticks([gene_x, metabolite_x])
    ax.set_xticklabels(["Genes", "Metabolites"])
    ax.set_yticks([])
    ax.set_title(title)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["gene"], markersize=8, label="Gene"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["metabolite"], markersize=8, label="Metabolite"),
        plt.Line2D([0], [0], color=PALETTE["edge_positive"], lw=2.5, label="Positive association"),
        plt.Line2D([0], [0], color=PALETTE["edge_negative"], lw=2.5, label="Negative association"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    _save_figure(fig, save_stem, cfg)


def plot_total_association_network(engine, save_stem: str | Path, cfg) -> None:
    edge_df = _prepare_network_df(engine, tier="total", max_edges=cfg.network_plot_top_edges)
    _plot_association_network(edge_df, "Total Gene-Metabolite Association Network", save_stem, cfg)


def plot_high_confidence_network(engine, save_stem: str | Path, cfg) -> None:
    edge_df = _prepare_network_df(engine, tier="high_confidence", max_edges=cfg.network_plot_top_edges)
    _plot_association_network(edge_df, "High-Confidence Gene-Metabolite Association Network", save_stem, cfg)


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

    n_panels = len(ranked)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.8 * n_rows))
    fig.subplots_adjust(hspace=0.50, wspace=0.35)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, ranked.itertuples(index=False)):
        gene = str(row.Gene)
        metab = str(row.Metabolite)
        if gene not in gene_df.columns or metab not in metab_df.columns:
            ax.axis("off")
            continue

        x = gene_df[gene].to_numpy(dtype=float, copy=False)
        y = metab_df[metab].to_numpy(dtype=float, copy=False)

        sns.regplot(
            x=x,
            y=y,
            ax=ax,
            color=PALETTE["gene"],
            scatter_kws={"s": 34, "alpha": 0.85, "edgecolor": "white", "linewidths": 0.4},
            line_kws={"lw": 1.5},
            ci=95,
        )
        ax.set_title(f"{gene} vs {metab}", fontsize=11)
        ax.set_xlabel(gene)
        ax.set_ylabel(metab)
        ax.text(
            0.03,
            0.97,
            (
                f"EdgeWeight = {float(row.EdgeWeight):.3f}\n"
                f"RRARank = {int(row.RRARank)}\n"
                f"ModelSupport = {int(row.ModelSupportCount)}\n"
                f"ScreenSupport = {int(row.ScreenSupportCount)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.95},
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Top Gene-Metabolite Association Pairs", y=1.005, fontsize=13)
    _save_figure(fig, save_stem, cfg)


def plot_metabolite_model_support_summary(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return

    plot_df = summary_df.sort_values(
        ["HighConfidenceEdges", "TotalAssociationEdges", "CandidateGenes"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(int(top_n or cfg.support_plot_top_metabolites))
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(plot_df)), 6.5))
    bars_total = ax.bar(x - width, plot_df["TotalAssociationEdges"], width=width, label="Total network edges", color=PALETTE["bar_total"])
    bars_dual = ax.bar(x, plot_df["DualModelEdges"], width=width, label="Dual-model supported edges", color=PALETTE["bar_dual_model"])
    bars_high = ax.bar(x + width, plot_df["HighConfidenceEdges"], width=width, label="High-confidence edges", color=PALETTE["bar_high_conf"])

    for container in (bars_total, bars_dual, bars_high):
        ax.bar_label(container, fontsize=8.5, padding=2, fmt="%d")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Metabolite"], rotation=90)
    ax.set_ylabel("Edge count")
    ax.set_title(f"Metabolite-Level Model Support Summary (Top {len(plot_df)})")
    ax.legend(ncol=3, loc="upper right")
    _save_figure(fig, save_stem, cfg)


def plot_top_key_genes(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    summary_df = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return

    top_df = summary_df.head(int(top_n or cfg.top_key_genes_plot_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top_df))))
    bars = ax.barh(top_df["Gene"], top_df["AssociatedMetaboliteCount"], color=PALETTE["gene"])
    labels = [f"{float(value):.0f}" for value in top_df["AssociatedMetaboliteCount"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_title("Top Key Genes Across Metabolites")
    ax.set_xlabel("Associated Metabolite Count")
    ax.set_ylabel("Gene")
    _save_figure(fig, save_stem, cfg)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"

    preview = df.head(max_rows).copy().fillna("")
    columns = preview.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in preview.iterrows()]
    return "\n".join([header, sep, *rows])


def generate_markdown_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())

    lines = [
        f"# DeepOmics Report: {cfg.project_name}",
        "",
        "## Run Summary",
        f"- Samples: {engine.adata.n_obs}",
        f"- Genes: {engine.adata.n_vars}",
        f"- Metabolites: {len(engine.adata.uns.get('metabolite_names', []))}",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Main Tables",
        f"- `{TABLE_FILE_PREFIXES['gene_scores']}`: complete metabolite-level gene scoring table after three-way screening and two-model ranking.",
        f"- `{TABLE_FILE_PREFIXES['total_network']}`: total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.",
        f"- `{TABLE_FILE_PREFIXES['high_confidence_network']}`: high-confidence subnetwork of the total association network after RRA and multi-evidence filtering.",
        f"- `{TABLE_FILE_PREFIXES['key_gene_summary']}`: merged key-gene summary across metabolites.",
        f"- `{TABLE_FILE_PREFIXES['metabolite_summary']}`: metabolite-level candidate and network summary.",
        f"- `{TABLE_FILE_PREFIXES['cytoscape_network']}`: Cytoscape-ready edge table with updated association fields.",
        "",
        "## Metabolite-Level Summary",
        _df_to_markdown(metabolite_summary, max_rows=20),
        "",
        "## Key Gene Summary",
        _df_to_markdown(key_gene_summary, max_rows=20),
        "",
        "## Generated Figures",
        f"- `plots/{FIGURE_FILE_PREFIXES['sample_clustering_dendrogram']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['total_association_network']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['high_confidence_network']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolite_model_support_summary']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['top_key_genes']}.pdf|svg|png`",
        "- `DeepOmics_Interactive_Report.html`",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame()).head(50)
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame()).head(50)

    table_rows = "".join([
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['gene_scores'])}</code></td><td>Complete metabolite-level gene scoring table after three-way screening and two-model ranking.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['total_network'])}</code></td><td>Total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['high_confidence_network'])}</code></td><td>High-confidence subnetwork of the total association network after RRA and multi-evidence filtering.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['key_gene_summary'])}</code></td><td>Merged key-gene summary across metabolites.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['metabolite_summary'])}</code></td><td>Metabolite-level candidate and network summary.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['cytoscape_network'])}</code></td><td>Cytoscape-ready edge table with updated association fields.</td></tr>",
    ])

    figure_rows = "".join([
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['sample_clustering_dendrogram'])}.pdf|svg|png</code></td><td>Sample clustering dendrogram.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca'])}.pdf|svg|png</code></td><td>Transcriptome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca'])}.pdf|svg|png</code></td><td>Metabolome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['total_association_network'])}.pdf|svg|png</code></td><td>Total gene-metabolite association network.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['high_confidence_network'])}.pdf|svg|png</code></td><td>High-confidence gene-metabolite association network.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs'])}.pdf|svg|png</code></td><td>Top association pair scatter panels ranked by EdgeWeight.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolite_model_support_summary'])}.pdf|svg|png</code></td><td>Metabolite-level model support summary.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['top_key_genes'])}.pdf|svg|png</code></td><td>Top key genes across metabolites.</td></tr>",
    ])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Report - {html.escape(cfg.project_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; color: #111827; }}
    h1, h2 {{ color: #1f2937; }}
    .hero {{
      background: #f8fafc;
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .link-card {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 24px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>DeepOmics Report: {html.escape(cfg.project_name)}</h1>
    <p>This page summarizes structured association tables and figure file names.</p>
  </div>

  <h2>Run Summary</h2>
  <ul>
    <li>Samples: {engine.adata.n_obs}</li>
    <li>Genes: {engine.adata.n_vars}</li>
    <li>Metabolites: {len(engine.adata.uns.get("metabolite_names", []))}</li>
    <li>Output directory: <code>{html.escape(str(cfg.output_dir))}</code></li>
  </ul>

  <div class="link-card">
    <strong>Interactive report:</strong>
    Open <a href="DeepOmics_Interactive_Report.html"><code>DeepOmics_Interactive_Report.html</code></a>
    for lightweight browser-native visualization preview and SVG export.
  </div>

  <h2>Main Output Tables</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Generated Figures</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{figure_rows}</tbody>
  </table>

  <h2>Metabolite-Level Summary</h2>
  {metabolite_summary.to_html(index=False, escape=True)}

  <h2>Key Gene Summary</h2>
  {key_gene_summary.to_html(index=False, escape=True)}
</body>
</html>
"""
    Path(report_path).write_text(html_text, encoding="utf-8")


def generate_report_plots(engine, cfg) -> None:
    set_academic_style()
    plots_dir = safe_mkdir(Path(cfg.output_dir) / "plots")

    plot_sample_dendrogram(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["sample_clustering_dendrogram"], cfg)
    plot_transcriptome_pca(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["transcriptome_pca"], cfg)
    plot_metabolome_pca(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["metabolome_pca"], cfg)
    plot_total_association_network(engine, plots_dir / FIGURE_FILE_PREFIXES["total_association_network"], cfg)
    plot_high_confidence_network(engine, plots_dir / FIGURE_FILE_PREFIXES["high_confidence_network"], cfg)
    plot_top_edge_scatter_panels(engine, plots_dir / FIGURE_FILE_PREFIXES["top_gene_metabolite_pairs"], cfg)
    plot_metabolite_model_support_summary(engine, plots_dir / FIGURE_FILE_PREFIXES["metabolite_model_support_summary"], cfg)
    plot_top_key_genes(engine, plots_dir / FIGURE_FILE_PREFIXES["top_key_genes"], cfg)

    notes = (
        "Recommended downstream usage:\n"
        f"1. Use {TABLE_FILE_PREFIXES['gene_scores']} for full metabolite-level candidate scoring.\n"
        f"2. Use {TABLE_FILE_PREFIXES['total_network']} for broad association recovery.\n"
        f"3. Use {TABLE_FILE_PREFIXES['high_confidence_network']} for the stricter high-confidence subset of the total network.\n"
        f"4. Use {TABLE_FILE_PREFIXES['cytoscape_network']} for Cytoscape import.\n"
        "5. Use DeepOmics_Interactive_Report.html for lightweight browser-native visualization preview and export.\n"
    )
    (plots_dir / "visualization_notes.txt").write_text(notes, encoding="utf-8")

    if cfg.generate_reports:
        if "md" in cfg.report_formats:
            generate_markdown_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.md")
        if "html" in cfg.report_formats:
            generate_html_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.html")
            from .interactive import generate_interactive_visual_report

            generate_interactive_visual_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Interactive_Report.html")
