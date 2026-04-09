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
    project: str,
    threads: int,
    log_level: str,
    group_table: Path | None,
    report_formats: tuple[str, ...],
    use_fdr: bool,
    fdr_alpha: float,
    export_cytoscape: bool,
    no_plots: bool,
    no_save_state: bool,
) -> AnalysisConfig:
    """Build a validated analysis configuration from CLI options."""
    return AnalysisConfig(
        project_name=project,
        output_dir=str(output_dir),
        n_threads=threads,
        log_level=log_level.upper(),
        group_table_path=str(group_table) if group_table is not None else None,
        report_formats=report_formats if report_formats else ("html",),
        use_fdr=use_fdr,
        fdr_alpha=fdr_alpha,
        save_h5ad=not no_save_state,
        generate_reports=not no_plots,
        export_cytoscape=export_cytoscape,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """DeepOmics: transcriptome-metabolome association analysis."""


@main.command()
@click.option("--genes", "-g", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Transcriptome matrix CSV (features x samples).")
@click.option("--metabs", "-m", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Metabolome matrix CSV (features x samples).")
@click.option("--output", "-o", default="results", show_default=True, type=click.Path(file_okay=False, path_type=Path), help="Output directory.")
@click.option("--project", default="Association_Analysis_v1", show_default=True, help="Project name.")
@click.option("--threads", type=int, default=-1, show_default=True, help="Number of CPU threads for XGBoost (-1 uses all cores).")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True, help="Logging level.")
@click.option("--group-table", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Optional sample grouping table for PCA plots. Required columns: sample_id, group.")
@click.option("--report-format", "report_formats", type=click.Choice(["md", "html"], case_sensitive=False), multiple=True, help="Optional report formats. Defaults to HTML only when not set. HTML additionally emits the interactive figure studio.")
@click.option("--use-fdr/--no-use-fdr", default=False, show_default=True, help="Apply soft FDR gating to Pearson and Spearman screening channels. Mutual information screening is always retained.")
@click.option("--fdr-alpha", type=float, default=0.05, show_default=True, help="FDR threshold used when --use-fdr is enabled.")
@click.option("--export-cytoscape/--no-export-cytoscape", default=True, show_default=True, help="Whether to export the Cytoscape-specific edge table.")
@click.option("--no-plots", is_flag=True, help="Skip plot and report generation.")
@click.option("--no-save-state", is_flag=True, help="Do not save the final H5AD state file.")
def run(
    genes: Path,
    metabs: Path,
    output: Path,
    project: str,
    threads: int,
    log_level: str,
    group_table: Path | None,
    report_formats: tuple[str, ...],
    use_fdr: bool,
    fdr_alpha: float,
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
        project=project,
        threads=threads,
        log_level=normalized_log_level,
        group_table=group_table,
        report_formats=tuple(fmt.lower() for fmt in report_formats),
        use_fdr=use_fdr,
        fdr_alpha=fdr_alpha,
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
