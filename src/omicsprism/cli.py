from __future__ import annotations

from pathlib import Path

import click

from .config import AnalysisConfig
from .core import MultiOmicsEngine
from .dem.pipeline import run_pipeline as run_dem_pipeline
from .deg.pipeline import run_pipeline as run_deg_pipeline
from .deg.utils import parse_csv_arg
from .io import load_as_anndata, preprocess_adata
from .utils import get_logger, safe_mkdir


def _build_config(
    *,
    output_dir: Path,
    threads: int,
    log_level: str,
    group_table: Path,
    report_formats: tuple[str, ...],
    trans_log2: bool,
) -> AnalysisConfig:
    """Build a validated analysis configuration from CLI options."""
    return AnalysisConfig(
        output_dir=str(output_dir),
        n_threads=threads,
        log_level=log_level.upper(),
        group_table_path=str(group_table),
        report_formats=report_formats if report_formats else ("html",),
        trans_log2=trans_log2,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """OmicsPrism: transcriptome-metabolome association analysis."""


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Local interface host.")
@click.option("--port", default=8501, show_default=True, type=int, help="Local interface port.")
def ui(host: str, port: int) -> None:
    """Launch the local browser UI."""
    try:
        from .local_ui import launch_ui

        launch_ui(host=host, port=port)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--genes", "-g", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Transcriptome matrix CSV (features x samples).")
@click.option("--metabs", "-m", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Metabolome matrix CSV (features x samples).")
@click.option("--output", "-o", default="results", show_default=True, type=click.Path(file_okay=False, path_type=Path), help="Output directory.")
@click.option("--threads", type=int, default=-1, show_default=True, help="Number of CPU threads for XGBoost (-1 uses all cores).")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", show_default=True, help="Logging level.")
@click.option("--group-table", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Required sample grouping table. Required columns: sample_id, group1, group2.")
@click.option("--report-format", "report_formats", type=click.Choice(["md", "html"], case_sensitive=False), multiple=True, help="Optional report formats. Defaults to HTML only when not set. HTML additionally emits the interactive figure studio.")
@click.option("--trans-log2", is_flag=True, help="Apply log2(x+1) to transcriptome input.")
def run(
    genes: Path,
    metabs: Path,
    output: Path,
    threads: int,
    log_level: str,
    group_table: Path,
    report_formats: tuple[str, ...],
    trans_log2: bool,
) -> None:
    """Run the end-to-end OmicsPrism workflow."""
    output_dir = safe_mkdir(output)
    normalized_log_level = log_level.upper()
    logger = get_logger(log_file=output_dir / "omicsprism.log", level=normalized_log_level)

    cfg = _build_config(
        output_dir=output_dir,
        threads=threads,
        log_level=normalized_log_level,
        group_table=group_table,
        report_formats=tuple(fmt.lower() for fmt in report_formats),
        trans_log2=trans_log2,
    )

    logger.info("Launching OmicsPrism")
    logger.info("Output directory: %s", Path(cfg.output_dir).resolve())

    try:
        adata = load_as_anndata(genes, metabs, group_table_path=cfg.group_table_path)
        adata = preprocess_adata(
            adata,
            missing_feature_threshold=cfg.missing_feature_threshold,
            knn_neighbors=cfg.knn_neighbors,
            trans_log2=cfg.trans_log2,
        )

        engine = MultiOmicsEngine(adata, cfg)
        engine.run_all(generate_plots=True)

        logger.info("Analysis completed successfully.")
    except Exception as exc:  # pragma: no cover
        logger.exception("OmicsPrism failed: %s", exc)
        raise click.Abort() from exc


@main.command()
@click.option("--counts", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Raw count matrix CSV. Rows are genes and columns are samples.")
@click.option("--metadata", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Sample metadata CSV. Must contain sample_id.")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path), help="Output directory for DEG results.")
@click.option("--same-fields", default=None, help="Comma-separated metadata fields that must be identical inside each contrast, for example line,timepoint.")
@click.option("--compare-field", required=True, help="Metadata field to compare, for example treatment.")
@click.option("--tested-levels", required=True, help="Comma-separated tested levels, for example salt or salt,drought,heat.")
@click.option("--reference-level", required=True, help="Reference level, for example control.")
@click.option("--padj-cutoff", type=float, default=0.05, show_default=True, help="Adjusted p-value cutoff for significant genes.")
@click.option("--log2fc-cutoff", type=float, default=1.0, show_default=True, help="Absolute log2 fold-change cutoff for significant genes.")
@click.option("--min-total-count", type=int, default=10, show_default=True, help="Keep genes with total raw count greater than or equal to this value.")
@click.option("--min-replicates", type=int, default=2, show_default=True, help="Minimum samples required for each tested/reference group in a contrast.")
@click.option("--n-cpus", type=int, default=8, show_default=True, help="Number of CPUs used by PyDESeq2 inference.")
def deg(
    counts: Path,
    metadata: Path,
    out: Path,
    same_fields: str | None,
    compare_field: str,
    tested_levels: str,
    reference_level: str,
    padj_cutoff: float,
    log2fc_cutoff: float,
    min_total_count: int,
    min_replicates: int,
    n_cpus: int,
) -> None:
    """Run differential expression analysis before the main OmicsPrism workflow."""
    try:
        same_fields_list = parse_csv_arg(same_fields)
        tested_levels_list = parse_csv_arg(tested_levels)

        if not tested_levels_list:
            raise ValueError("--tested-levels must contain at least one level.")
        if padj_cutoff <= 0 or padj_cutoff > 1:
            raise ValueError("--padj-cutoff must be in the interval (0, 1].")
        if log2fc_cutoff < 0:
            raise ValueError("--log2fc-cutoff must be non-negative.")
        if min_total_count < 0:
            raise ValueError("--min-total-count must be non-negative.")
        if min_replicates < 1:
            raise ValueError("--min-replicates must be at least 1.")
        if n_cpus < 1:
            raise ValueError("--n-cpus must be at least 1.")

        result = run_deg_pipeline(
            counts_path=counts,
            metadata_path=metadata,
            out_dir=out,
            same_fields=same_fields_list,
            compare_field=compare_field,
            tested_levels=tested_levels_list,
            reference_level=reference_level,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            min_total_count=min_total_count,
            min_replicates=min_replicates,
            n_cpus=n_cpus,
        )
    except Exception as exc:  # pragma: no cover
        raise click.ClickException(str(exc)) from exc

    click.echo("Differential expression analysis finished.")
    click.echo(f"Output directory: {result['out_dir']}")
    click.echo(f"Valid contrasts: {result['n_contrasts']}")
    click.echo(f"Union significant genes: {result['n_union_significant_genes']}")
    click.echo(f"OmicsPrism-ready VST matrix: {result['union_significant_genes_vst']}")


@main.command()
@click.option("--metabs", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Metabolite abundance matrix CSV. Rows are metabolites and columns are samples.")
@click.option("--metadata", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Sample metadata CSV. Must contain sample_id.")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path), help="Output directory for DEM results.")
@click.option("--same-fields", default=None, help="Comma-separated metadata fields that must be identical inside each contrast, for example line,timepoint.")
@click.option("--compare-field", required=True, help="Metadata field to compare, for example treatment.")
@click.option("--tested-levels", required=True, help="Comma-separated tested levels, for example salt or salt,drought,heat.")
@click.option("--reference-level", required=True, help="Reference level, for example control.")
@click.option("--vip-cutoff", type=float, default=1.0, show_default=True, help="OPLS-DA VIP cutoff for significant metabolites.")
@click.option("--padj-cutoff", type=float, default=0.05, show_default=True, help="Benjamini-Hochberg adjusted p-value cutoff for significant metabolites.")
@click.option("--log2fc-cutoff", type=float, default=1.0, show_default=True, help="Absolute log2 fold-change cutoff for significant metabolites.")
@click.option("--pseudocount", type=float, default=1e-9, show_default=True, help="Small value added before fold-change calculation.")
@click.option("--max-missing-fraction", type=float, default=0.5, show_default=True, help="Drop metabolites above this missing-value fraction within a contrast.")
@click.option("--min-replicates", type=int, default=2, show_default=True, help="Minimum samples required for each tested/reference group in a contrast.")
@click.option("--opls-orthogonal-components", type=int, default=1, show_default=True, help="Number of orthogonal components to remove in the OPLS-DA model.")
def dem(
    metabs: Path,
    metadata: Path,
    out: Path,
    same_fields: str | None,
    compare_field: str,
    tested_levels: str,
    reference_level: str,
    vip_cutoff: float,
    padj_cutoff: float,
    log2fc_cutoff: float,
    pseudocount: float,
    max_missing_fraction: float,
    min_replicates: int,
    opls_orthogonal_components: int,
) -> None:
    """Run differential metabolite analysis before the main OmicsPrism workflow."""
    try:
        same_fields_list = parse_csv_arg(same_fields)
        tested_levels_list = parse_csv_arg(tested_levels)

        if not tested_levels_list:
            raise ValueError("--tested-levels must contain at least one level.")
        if vip_cutoff < 0:
            raise ValueError("--vip-cutoff must be non-negative.")
        if padj_cutoff <= 0 or padj_cutoff > 1:
            raise ValueError("--padj-cutoff must be in the interval (0, 1].")
        if log2fc_cutoff < 0:
            raise ValueError("--log2fc-cutoff must be non-negative.")
        if pseudocount < 0:
            raise ValueError("--pseudocount must be non-negative.")
        if max_missing_fraction < 0 or max_missing_fraction > 1:
            raise ValueError("--max-missing-fraction must be in the interval [0, 1].")
        if min_replicates < 2:
            raise ValueError("--min-replicates must be at least 2 for t-test based DEM analysis.")
        if opls_orthogonal_components < 0:
            raise ValueError("--opls-orthogonal-components must be non-negative.")

        result = run_dem_pipeline(
            metabs_path=metabs,
            metadata_path=metadata,
            out_dir=out,
            same_fields=same_fields_list,
            compare_field=compare_field,
            tested_levels=tested_levels_list,
            reference_level=reference_level,
            vip_cutoff=vip_cutoff,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            pseudocount=pseudocount,
            max_missing_fraction=max_missing_fraction,
            min_replicates=min_replicates,
            n_orthogonal_components=opls_orthogonal_components,
        )
    except Exception as exc:  # pragma: no cover
        raise click.ClickException(str(exc)) from exc

    click.echo("Differential metabolite analysis finished.")
    click.echo(f"Output directory: {result['out_dir']}")
    click.echo(f"Valid contrasts: {result['n_contrasts']}")
    click.echo(f"Union significant metabolites: {result['n_union_significant_metabolites']}")
    click.echo(f"OmicsPrism-ready metabolite matrix: {result['union_significant_metabolites_matrix']}")


if __name__ == "__main__":
    main()
