from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from ..outputs import FIGURE_FILE_PREFIXES, OBSOLETE_FIGURE_FILE_PREFIXES, TABLE_FILE_PREFIXES
from ..utils import safe_mkdir
from .context import VisualizationContext
from .registry import iter_figure_specs
from .static.base import set_academic_style


MAIN_TABLE_DESCRIPTIONS = {
    "metabolite_summary": "Metabolite-level association summary with screening counts, candidate counts, high-confidence edge counts, and top linked genes.",
    "high_confidence_network": "High-confidence gene-metabolite network for biological interpretation and Cytoscape import.",
    "key_gene_summary": "Key-gene summary aggregated across associated metabolites.",
    "gene_module_assignment": "Gene module assignment with intramodular metrics and high-confidence association counts.",
    "module_metabolite_association": "Module-metabolite Spearman association table with FDR.",
    "module_summary": "Module-level summary including size, hub gene, and top associated metabolite.",
}

FIGURE_DESCRIPTIONS = {
    "sample_clustering_dendrogram": "Sample clustering dendrogram.",
    "transcriptome_pca": "Transcriptome PCA scatter plot colored by group1.",
    "transcriptome_pca_subgroups": "Transcriptome PCA scatter plot colored by group2.",
    "transcriptome_pca_pairs": "Transcriptome PCA pairs plot colored by group1.",
    "transcriptome_pca_pairs_subgroups": "Transcriptome PCA pairs plot colored by group2.",
    "metabolome_pca": "Metabolome PCA scatter plot colored by group1.",
    "metabolome_pca_subgroups": "Metabolome PCA scatter plot colored by group2.",
    "metabolome_pca_pairs": "Metabolome PCA pairs plot colored by group1.",
    "metabolome_pca_pairs_subgroups": "Metabolome PCA pairs plot colored by group2.",
    "association_evidence_upset": "Global evidence-overlap UpSet plot across candidate metabolite-gene edges.",
    "gene_metabolite_correlation_bubble_heatmap": "High-confidence gene-metabolite bubble heatmap using Spearman rho and EdgeWeight.",
    "top_gene_metabolite_correlation_heatmaps": "Top gene x top metabolite Pearson and Spearman heatmaps.",
    "top_gene_metabolite_pairs": "Top gene-metabolite regression panels ranked by EdgeWeight.",
    "top_metabolite_group1_violin_box": "Top metabolite abundance distributions by group1.",
    "module_eigengene_heatmap": "Module eigengene heatmap with sample group annotations.",
    "module_eigengene_heatmap_group2": "Module eigengene heatmap grouped by group2.",
    "module_zscore_line_panels": "Module z-score line panels faceted by group1.",
    "module_gene_zscore_line_panels": "Module gene z-score line panels with module summaries.",
    "module_eigengene_ridge": "Module eigengene z-score ridge distributions across averaged group states.",
    "module_eigengene_ridge_group1": "Module eigengene z-score ridge distributions overlaid by group1.",
    "module_eigengene_group1_violin_box": "Module eigengene z-score violin and box plots by group1.",
    "module_kme_boxplot": "Intramodular gene kME distribution by module.",
    "module_metabolite_association_heatmap": "Module-metabolite association heatmap with significance stars.",
    "module_metabolite_bubble_plot": "Module-metabolite bubble plot using Spearman rho and FDR-scaled point size.",
    "module_top_metabolite_regressions": "Module eigengene regression panels against each module's top associated metabolite.",
    "module_eigengene_metabolite_trend_panels": "Module eigengene and top metabolite z-score trends across group2 within group1.",
    "association_direction_summary": "Positive and negative high-confidence association counts by module.",
    "edgeweight_distribution_by_module": "High-confidence EdgeWeight distributions by module.",
    "compressed_circos_network": "Compact T02-only Circos network overview.",
    "floating_cnet_circos_network": "Floating circular cnetplot-style T02-only network overview.",
}


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"

    preview = df.head(max_rows).copy().fillna("")
    columns = preview.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in preview.iterrows()]
    return "\n".join([header, sep, *rows])


def _iter_enabled_figure_rows(engine, cfg) -> list[tuple[str, str]]:
    context = VisualizationContext.from_engine(engine, cfg)
    rows: list[tuple[str, str]] = []
    for spec in iter_figure_specs():
        if not spec.enabled(context):
            continue
        prefix = FIGURE_FILE_PREFIXES[spec.prefix_key]
        description = FIGURE_DESCRIPTIONS.get(spec.key, spec.description or spec.key.replace("_", " ").title())
        rows.append((f"plots/{prefix}.pdf|svg|png", description))
    return rows


def _main_table_rows() -> list[tuple[str, str]]:
    return [
        (TABLE_FILE_PREFIXES[key], description)
        for key, description in MAIN_TABLE_DESCRIPTIONS.items()
        if key in TABLE_FILE_PREFIXES
    ]


def generate_markdown_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())

    lines = [
        "# OmicsPrism Report",
        "",
        "## Run Summary",
        f"- Samples: {engine.adata.n_obs}",
        f"- Genes: {engine.adata.n_vars}",
        f"- Metabolites: {len(engine.adata.uns.get('metabolite_names', []))}",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Main Tables",
        *[f"- `{filename}`: {description}" for filename, description in _main_table_rows()],
        "",
        "## Metabolite-Level Summary",
        _df_to_markdown(metabolite_summary, max_rows=20),
        "",
        "## Key Gene Summary",
        _df_to_markdown(key_gene_summary, max_rows=20),
        "",
        "## Generated Figures",
        *[f"- `{filename}`: {description}" for filename, description in _iter_enabled_figure_rows(engine, cfg)],
        "- `OmicsPrism_Interactive_Report.html`",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame()).head(50)
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame()).head(50)

    table_rows = "".join(
        f"<tr><td><code>{html.escape(filename)}</code></td><td>{html.escape(description)}</td></tr>"
        for filename, description in _main_table_rows()
    )

    figure_rows = "".join(
        f"<tr><td><code>{html.escape(filename)}</code></td><td>{html.escape(description)}</td></tr>"
        for filename, description in _iter_enabled_figure_rows(engine, cfg)
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OmicsPrism Report</title>
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
    <h1>OmicsPrism Report</h1>
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
    figure_data_dir = safe_mkdir(Path(cfg.output_dir) / "figure_data")
    for prefix in OBSOLETE_FIGURE_FILE_PREFIXES:
        for suffix in ("pdf", "svg", "png"):
            path = plots_dir / f"{prefix}.{suffix}"
            if path.exists():
                path.unlink()
    context = VisualizationContext.from_engine(engine, cfg, plots_dir=plots_dir)

    from .figure_data import export_figure_data

    for figure_spec in iter_figure_specs():
        figure_spec.render(context)
        # Export figure data JSON for interactive pages
        export_figure_data(context, figure_spec, figure_data_dir)

    notes = (
        "Recommended downstream usage:\n"
        f"1. Use {TABLE_FILE_PREFIXES['metabolite_summary']} for metabolite-level triage.\n"
        f"2. Use {TABLE_FILE_PREFIXES['high_confidence_network']} for the main gene-metabolite network interpretation and Cytoscape import.\n"
        f"3. Use {TABLE_FILE_PREFIXES['key_gene_summary']} for candidate key-gene prioritization.\n"
        f"4. Use {TABLE_FILE_PREFIXES['gene_module_assignment']} and {TABLE_FILE_PREFIXES['module_summary']} for module interpretation.\n"
        f"5. Use {TABLE_FILE_PREFIXES['module_metabolite_association']} for module-metabolite association statistics.\n"
        "6. Use OmicsPrism_Interactive_Report.html for lightweight browser-native visualization preview and export.\n"
        f"7. Set export_audit_tables=True to emit {TABLE_FILE_PREFIXES['gene_scores_audit']} for full scoring audit.\n"
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
