from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from ..outputs import FIGURE_FILE_PREFIXES, TABLE_FILE_PREFIXES
from ..utils import safe_mkdir
from .context import VisualizationContext
from .registry import iter_figure_specs
from .static.base import set_academic_style

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
        f"# OmicsPrism Report: {cfg.project_name}",
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
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_top_metabolite_regressions']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_eigengene_heatmap']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_eigengene_heatmap_group2']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_zscore_line_panels']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_gene_zscore_line_panels']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['compressed_circos_network']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['floating_cnet_circos_network']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['association_evidence_upset']}.pdf|svg|png`",
        "- `OmicsPrism_Interactive_Report.html`",
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
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca_pairs'])}.pdf|svg|png</code></td><td>Transcriptome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca_pairs'])}.pdf|svg|png</code></td><td>Metabolome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs'])}.pdf|svg|png</code></td><td>Top association pair scatter panels ranked by EdgeWeight.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_top_metabolite_regressions'])}.pdf|svg|png</code></td><td>Module eigengene regression panels against each module's top associated metabolite.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_eigengene_heatmap'])}.pdf|svg|png</code></td><td>Module eigengene heatmap with group2 and group1 annotation tracks using PCA group colors.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_eigengene_heatmap_group2'])}.pdf|svg|png</code></td><td>Module eigengene heatmap with group1 and group2 annotation tracks, grouped by group2.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_zscore_line_panels'])}.pdf|svg|png</code></td><td>Module z-score line panels faceted by group1 with group2 color strips.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_gene_zscore_line_panels'])}.pdf|svg|png</code></td><td>Module gene z-score line panels with grey gene trajectories and black module trajectories.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap'])}.pdf|svg|png</code></td><td>Module-metabolite association heatmap colored by Spearman rho with significance stars.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['compressed_circos_network'])}.pdf|svg|png</code></td><td>Compact Circos overview using all unique genes and metabolites from T03 only.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['floating_cnet_circos_network'])}.pdf|svg|png</code></td><td>Floating circular cnetplot-style network using T03 only, with circular non-overlapping nodes and metabolite-colored edges.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['association_evidence_upset'])}.pdf|svg|png</code></td><td>Global UpSet plot showing overlap among PCC, Spearman, MI, ElasticNet, and XGBoost evidence for metabolite-gene candidate edges.</td></tr>",
    ])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OmicsPrism Report - {html.escape(cfg.project_name)}</title>
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
    <h1>OmicsPrism Report: {html.escape(cfg.project_name)}</h1>
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
    Open <a href="OmicsPrism_Interactive_Report.html"><code>OmicsPrism_Interactive_Report.html</code></a>
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
    context = VisualizationContext.from_engine(engine, cfg, plots_dir=plots_dir)

    for figure_spec in iter_figure_specs():
        figure_spec.render(context)

    notes = (
        "Recommended downstream usage:\n"
        f"1. Use {TABLE_FILE_PREFIXES['gene_scores']} for full metabolite-level candidate scoring.\n"
        f"2. Use {TABLE_FILE_PREFIXES['total_network']} for broad association recovery.\n"
        f"3. Use {TABLE_FILE_PREFIXES['high_confidence_network']} for the stricter high-confidence subset of the total network.\n"
        f"4. Use {TABLE_FILE_PREFIXES['cytoscape_network']} for Cytoscape import.\n"
        f"5. Use plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs']}.pdf|svg|png for the transcriptome PCA pairs overview (group1).\n"
        f"6. Use plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs']}.pdf|svg|png for the metabolome PCA pairs overview (group1).\n"
        f"7. Use plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs_subgroups']}.pdf|svg|png for the transcriptome PCA pairs overview (group2).\n"
        f"8. Use plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs_subgroups']}.pdf|svg|png for the metabolome PCA pairs overview (group2).\n"
        f"9. Use plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png for the top regression-panel overview.\n"
        f"10. Use plots/{FIGURE_FILE_PREFIXES['module_top_metabolite_regressions']}.pdf|svg|png for module eigengene vs top metabolite regressions.\n"
        f"11. Use plots/{FIGURE_FILE_PREFIXES['module_eigengene_heatmap']}.pdf|svg|png for the module eigengene heatmap with group2/group1 annotation tracks.\n"
        f"12. Use plots/{FIGURE_FILE_PREFIXES['module_eigengene_heatmap_group2']}.pdf|svg|png for the module eigengene heatmap grouped by group2.\n"
        f"13. Use plots/{FIGURE_FILE_PREFIXES['module_zscore_line_panels']}.pdf|svg|png for module z-score line panels faceted by group1 with group2 color strips.\n"
        f"14. Use plots/{FIGURE_FILE_PREFIXES['module_gene_zscore_line_panels']}.pdf|svg|png for module gene z-score line panels with grey gene trajectories and black module trajectories.\n"
        f"15. Use plots/{FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap']}.pdf|svg|png for the module-metabolite association heatmap.\n"
        f"16. Use plots/{FIGURE_FILE_PREFIXES['compressed_circos_network']}.pdf|svg|png for the compact T03-only Circos overview.\n"
        f"17. Use plots/{FIGURE_FILE_PREFIXES['floating_cnet_circos_network']}.pdf|svg|png for the floating circular T03-only cnetplot-style overview.\n"
        f"18. Use plots/{FIGURE_FILE_PREFIXES['association_evidence_upset']}.pdf|svg|png for global evidence-overlap interpretation across candidate metabolite-gene edges.\n"
        "19. Use OmicsPrism_Interactive_Report.html for lightweight browser-native visualization preview and export.\n"
    )
    (plots_dir / "visualization_notes.txt").write_text(notes, encoding="utf-8")

    if cfg.generate_reports:
        if "md" in cfg.report_formats:
            generate_markdown_report(engine, cfg, Path(cfg.output_dir) / "OmicsPrism_Report.md")
        if "html" in cfg.report_formats:
            generate_html_report(engine, cfg, Path(cfg.output_dir) / "OmicsPrism_Report.html")
            from .interactive import generate_interactive_visual_report

            generate_interactive_visual_report(engine, cfg, Path(cfg.output_dir) / "OmicsPrism_Interactive_Report.html")



__all__ = [
    "generate_markdown_report",
    "generate_html_report",
    "generate_report_plots",
]
