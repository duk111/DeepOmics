from __future__ import annotations

from pathlib import Path

import click

from .config import AnalysisConfig
from .core import MultiOmicsEngine
from .io import load_as_anndata, preprocess_adata
from .utils import get_logger, safe_mkdir


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """DeepOmics: transcriptome-metabolome integration with ensemble ML and optimized WGCNA."""
    pass


@main.command()
@click.option("--genes", "-g", required=True, type=click.Path(exists=True, dir_okay=False), help="Transcriptome matrix CSV (features x samples).")
@click.option("--metabs", "-m", required=True, type=click.Path(exists=True, dir_okay=False), help="Metabolomics matrix CSV (features x samples).")
@click.option("--output", "-o", default="results", show_default=True, help="Output directory.")
@click.option("--pcc-r", type=float, default=0.30, show_default=True, help="Absolute Pearson correlation threshold.")
@click.option("--pcc-p", type=float, default=0.05, show_default=True, help="Pearson p-value threshold used when FDR is disabled.")
@click.option("--top-n", type=int, default=3000, show_default=True, help="Number of highly variable genes used by WGCNA.")
@click.option("--project", default="Analysis_v1", show_default=True, help="Project name.")
@click.option("--threads", type=int, default=-1, show_default=True, help="Number of CPU threads for XGBoost (-1 uses all cores).")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True, help="Logging level.")
@click.option("--report-format", "report_formats", type=click.Choice(["md", "html"], case_sensitive=False), multiple=True, help="Optional report formats. Defaults to both when not set.")
@click.option("--wgcna-network-type", type=click.Choice(["unsigned", "signed"], case_sensitive=False), default="unsigned", show_default=True, help="Adjacency type used for WGCNA.")
@click.option("--wgcna-soft-power", type=int, default=None, help="Fixed WGCNA soft-threshold power. By default, the package auto-selects it.")
@click.option("--no-wgcna-tom", is_flag=True, help="Disable TOM-based module detection and use adjacency directly.")
@click.option("--wgcna-tom-max-genes", type=int, default=2000, show_default=True, help="Maximum genes allowed for TOM calculation.")
@click.option("--wgcna-tree-cut-height", type=float, default=None, help="Optional fixed tree cut height for WGCNA module detection.")
@click.option("--wgcna-merge-height", type=float, default=0.25, show_default=True, help="Eigengene dissimilarity threshold for module merging.")
@click.option("--corr-circle-top-genes", type=int, default=30, show_default=True, help="Number of genes displayed in the correlation circle plot.")
@click.option("--corr-circle-top-metabs", type=int, default=20, show_default=True, help="Number of metabolites displayed in the correlation circle plot.")
@click.option("--circos-top-edges", type=int, default=80, show_default=True, help="Number of prioritized GRN edges displayed in the Circos plot.")
@click.option("--complex-heatmap-top-genes", type=int, default=30, show_default=True, help="Number of genes displayed in the complex heatmap.")
@click.option("--complex-heatmap-top-metabs", type=int, default=15, show_default=True, help="Number of metabolites displayed in the complex heatmap.")
@click.option("--no-plots", is_flag=True, help="Skip plot and report generation.")
@click.option("--no-save-state", is_flag=True, help="Do not save the final H5AD state file.")
def run(
    genes: str,
    metabs: str,
    output: str,
    pcc_r: float,
    pcc_p: float,
    top_n: int,
    project: str,
    threads: int,
    log_level: str,
    report_formats: tuple[str, ...],
    wgcna_network_type: str,
    wgcna_soft_power: int | None,
    no_wgcna_tom: bool,
    wgcna_tom_max_genes: int,
    wgcna_tree_cut_height: float | None,
    wgcna_merge_height: float,
    corr_circle_top_genes: int,
    corr_circle_top_metabs: int,
    circos_top_edges: int,
    complex_heatmap_top_genes: int,
    complex_heatmap_top_metabs: int,
    no_plots: bool,
    no_save_state: bool,
) -> None:
    """Run the end-to-end DeepOmics workflow."""
    output_dir = safe_mkdir(output)
    logger = get_logger(log_file=output_dir / "deepomics.log", level=log_level.upper())

    cfg = AnalysisConfig(
        project_name=project,
        output_dir=str(output_dir),
        pcc_r_threshold=pcc_r,
        pcc_p_threshold=pcc_p,
        wgcna_top_n_genes=top_n,
        wgcna_network_type=wgcna_network_type.lower(),
        wgcna_soft_power=wgcna_soft_power,
        wgcna_use_tom=not no_wgcna_tom,
        wgcna_tom_max_genes=wgcna_tom_max_genes,
        wgcna_tree_cut_height=wgcna_tree_cut_height,
        wgcna_merge_cut_height=wgcna_merge_height,
        correlation_circle_top_genes=corr_circle_top_genes,
        correlation_circle_top_metabolites=corr_circle_top_metabs,
        circos_top_edges=circos_top_edges,
        complex_heatmap_top_genes=complex_heatmap_top_genes,
        complex_heatmap_top_metabolites=complex_heatmap_top_metabs,
        n_threads=threads,
        log_level=log_level.upper(),
        report_formats=tuple(fmt.lower() for fmt in report_formats) if report_formats else ("md", "html"),
        save_h5ad=not no_save_state,
        generate_reports=not no_plots,
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
