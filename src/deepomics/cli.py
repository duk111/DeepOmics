from __future__ import annotations

from pathlib import Path

import click

from .config import AnalysisConfig
from .core import MultiOmicsEngine
from .io import load_as_anndata, preprocess_adata
from .utils import get_logger, safe_mkdir


def _build_config(
    *,
    output_dir: Path,
    pcc_r: float,
    pcc_p: float,
    project: str,
    threads: int,
    log_level: str,
    report_formats: tuple[str, ...],
    corr_circle_top_genes: int,
    corr_circle_top_metabs: int,
    circos_top_edges: int,
    complex_heatmap_top_genes: int,
    complex_heatmap_top_metabs: int,
    verbose_outputs: bool,
    export_cytoscape: bool,
    no_plots: bool,
    no_save_state: bool,
) -> AnalysisConfig:
    """Build a validated analysis configuration from CLI options."""
    return AnalysisConfig(
        project_name=project,
        output_dir=str(output_dir),
        pcc_r_threshold=pcc_r,
        pcc_p_threshold=pcc_p,
        correlation_circle_top_genes=corr_circle_top_genes,
        correlation_circle_top_metabolites=corr_circle_top_metabs,
        circos_top_edges=circos_top_edges,
        complex_heatmap_top_genes=complex_heatmap_top_genes,
        complex_heatmap_top_metabolites=complex_heatmap_top_metabs,
        n_threads=threads,
        log_level=log_level.upper(),
        report_formats=report_formats if report_formats else ("html",),
        save_h5ad=not no_save_state,
        generate_reports=not no_plots,
        verbose_outputs=verbose_outputs,
        export_cytoscape=export_cytoscape,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """DeepOmics: transcriptome-metabolome integration with ensemble ML."""


@main.command()
@click.option("--genes", "-g", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Transcriptome matrix CSV (features x samples).")
@click.option("--metabs", "-m", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Metabolomics matrix CSV (features x samples).")
@click.option("--output", "-o", default="results", show_default=True, type=click.Path(file_okay=False, path_type=Path), help="Output directory.")
@click.option("--pcc-r", type=float, default=0.30, show_default=True, help="Absolute Pearson correlation threshold.")
@click.option("--pcc-p", type=float, default=0.05, show_default=True, help="Pearson p-value threshold used when FDR is disabled.")
@click.option("--project", default="Analysis_v1", show_default=True, help="Project name.")
@click.option("--threads", type=int, default=-1, show_default=True, help="Number of CPU threads for XGBoost (-1 uses all cores).")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True, help="Logging level.")
@click.option("--report-format", "report_formats", type=click.Choice(["md", "html"], case_sensitive=False), multiple=True, help="Optional report formats. Defaults to HTML only when not set. HTML additionally emits the interactive figure studio.")
@click.option("--corr-circle-top-genes", type=int, default=30, show_default=True, help="Number of genes displayed in the correlation circle plot.")
@click.option("--corr-circle-top-metabs", type=int, default=20, show_default=True, help="Number of metabolites displayed in the correlation circle plot.")
@click.option("--circos-top-edges", type=int, default=80, show_default=True, help="Number of prioritized GRN edges displayed in the Circos plot.")
@click.option("--complex-heatmap-top-genes", type=int, default=30, show_default=True, help="Number of genes displayed in the complex heatmap.")
@click.option("--complex-heatmap-top-metabs", type=int, default=15, show_default=True, help="Number of metabolites displayed in the complex heatmap.")
@click.option("--verbose-outputs", is_flag=True, help="Export verbose diagnostic tables and figures.")
@click.option("--export-cytoscape/--no-export-cytoscape", default=True, show_default=True, help="Whether to export the Cytoscape-specific edge table.")
@click.option("--no-plots", is_flag=True, help="Skip plot and report generation.")
@click.option("--no-save-state", is_flag=True, help="Do not save the final H5AD state file.")
def run(
    genes: Path,
    metabs: Path,
    output: Path,
    pcc_r: float,
    pcc_p: float,
    project: str,
    threads: int,
    log_level: str,
    report_formats: tuple[str, ...],
    corr_circle_top_genes: int,
    corr_circle_top_metabs: int,
    circos_top_edges: int,
    complex_heatmap_top_genes: int,
    complex_heatmap_top_metabs: int,
    verbose_outputs: bool,
    export_cytoscape: bool,
    no_plots: bool,
    no_save_state: bool,
) -> None:
    """Run the end-to-end DeepOmics workflow."""
    output_dir = safe_mkdir(output)
    normalized_log_level = log_level.upper()
    logger = get_logger(log_file=output_dir / "deepomics.log", level=normalized_log_level)

    cfg = _build_config(
        output_dir=output_dir,
        pcc_r=pcc_r,
        pcc_p=pcc_p,
        project=project,
        threads=threads,
        log_level=normalized_log_level,
        report_formats=tuple(fmt.lower() for fmt in report_formats),
        corr_circle_top_genes=corr_circle_top_genes,
        corr_circle_top_metabs=corr_circle_top_metabs,
        circos_top_edges=circos_top_edges,
        complex_heatmap_top_genes=complex_heatmap_top_genes,
        complex_heatmap_top_metabs=complex_heatmap_top_metabs,
        verbose_outputs=verbose_outputs,
        export_cytoscape=export_cytoscape,
        no_plots=no_plots,
        no_save_state=no_save_state,
    )

    logger.info("Launching DeepOmics project: %s", cfg.project_name)
    logger.info("Output directory: %s", Path(cfg.output_dir).resolve())

    try:
        adata = load_as_anndata(genes, metabs)
        adata = preprocess_adata(adata)

        engine = MultiOmicsEngine(adata, cfg)
        engine.run_all(generate_plots=not no_plots)

        logger.info("Analysis completed successfully.")
    except Exception as exc:  # pragma: no cover
        logger.exception("DeepOmics failed: %s", exc)
        raise click.Abort() from exc


if __name__ == "__main__":
    main()
