from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import streamlit as st

from omicsprism.config import AnalysisConfig
from omicsprism.core import MultiOmicsEngine
from omicsprism.io import load_as_anndata, preprocess_adata
from omicsprism.utils import get_logger, safe_mkdir


def _safe_filename(name: str, fallback: str) -> str:
    cleaned = Path(str(name or fallback)).name.strip().replace("\x00", "")
    return cleaned or fallback


def _write_upload(uploaded_file: BinaryIO, target_dir: Path, fallback_name: str) -> Path:
    target_path = target_dir / _safe_filename(getattr(uploaded_file, "name", ""), fallback_name)
    uploaded_file.seek(0)
    with target_path.open("wb") as handle:
        shutil.copyfileobj(uploaded_file, handle)
    return target_path


def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            archive.write(path, path.relative_to(source_dir))
    return zip_path


def _build_config(
    *,
    output_dir: Path,
    group_table_path: Path,
    report_formats: tuple[str, ...],
    generate_reports: bool,
    n_threads: int,
    export_audit_tables: bool,
    trans_log2: bool,
    metab_log2: bool,
) -> AnalysisConfig:
    return AnalysisConfig(
        output_dir=str(output_dir),
        group_table_path=str(group_table_path),
        report_formats=report_formats,
        generate_reports=generate_reports,
        n_threads=n_threads,
        trans_log2=trans_log2,
        metab_log2=metab_log2,
        export_audit_tables=export_audit_tables,
    )


def _run_analysis(
    *,
    gene_path: Path,
    metab_path: Path,
    group_path: Path,
    cfg: AnalysisConfig,
) -> None:
    logger = get_logger(log_file=Path(cfg.output_dir) / "omicsprism.log", level=cfg.log_level)
    logger.info("Launching OmicsPrism local UI")
    logger.info("Output directory: %s", Path(cfg.output_dir).resolve())

    adata = load_as_anndata(gene_path, metab_path, group_table_path=cfg.group_table_path)
    adata = preprocess_adata(
        adata,
        missing_feature_threshold=cfg.missing_feature_threshold,
        knn_neighbors=cfg.knn_neighbors,
        trans_log2=cfg.trans_log2,
        metab_log2=cfg.metab_log2,
    )
    engine = MultiOmicsEngine(adata, cfg)
    engine.run_all(generate_plots=cfg.generate_reports)


def _open_report_button(label: str, path: Path) -> None:
    if path.exists():
        st.link_button(label, path.resolve().as_uri())


def main() -> None:
    st.set_page_config(page_title="OmicsPrism Local UI", layout="wide")
    st.title("OmicsPrism Local UI")

    st.sidebar.header("Input files")
    gene_file = st.sidebar.file_uploader("Transcriptome matrix CSV", type=["csv"])
    metab_file = st.sidebar.file_uploader("Metabolome matrix CSV", type=["csv"])
    group_file = st.sidebar.file_uploader("Group table CSV", type=["csv"])

    st.sidebar.header("Project")
    output_root = Path(st.sidebar.text_input("Output root", value="results")).expanduser()
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = st.sidebar.text_input("Run folder", value=f"omicsprism_ui_{run_label}")

    with st.expander("Analysis options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            n_threads = st.number_input("Threads (-1 = all)", min_value=-1, max_value=256, value=-1, step=1)
            generate_reports = st.checkbox("Generate figures and reports", value=True)
        with col2:
            html_report = st.checkbox("HTML report", value=True)
            md_report = st.checkbox("Markdown report", value=False)
            export_audit_tables = st.checkbox("Export audit table T99", value=False)
            trans_log2 = st.checkbox("Log2 transform transcriptome", value=False)
            metab_log2 = st.checkbox("Log2 transform metabolome", value=True)

    report_formats = tuple(fmt for fmt, enabled in (("html", html_report), ("md", md_report)) if enabled)
    output_dir = output_root / _safe_filename(output_name, "omicsprism_ui_run")

    st.subheader("Run analysis")
    st.write(f"Output directory: `{output_dir}`")

    ready = gene_file is not None and metab_file is not None and group_file is not None
    if not ready:
        st.info("Upload transcriptome, metabolome, and group table CSV files to start.")

    if st.button("Run OmicsPrism", type="primary", disabled=not ready):
        try:
            output_dir = safe_mkdir(output_dir)
            input_dir = safe_mkdir(output_dir / "_inputs")
            gene_path = _write_upload(gene_file, input_dir, "transcriptome.csv")
            metab_path = _write_upload(metab_file, input_dir, "metabolome.csv")
            group_path = _write_upload(group_file, input_dir, "group_table.csv")

            cfg = _build_config(
                output_dir=output_dir,
                group_table_path=group_path,
                report_formats=report_formats,
                generate_reports=generate_reports,
                n_threads=int(n_threads),
                export_audit_tables=bool(export_audit_tables),
                trans_log2=bool(trans_log2),
                metab_log2=bool(metab_log2),
            )

            with st.spinner("Running OmicsPrism analysis. This may take several minutes."):
                _run_analysis(gene_path=gene_path, metab_path=metab_path, group_path=group_path, cfg=cfg)

            st.session_state["last_output_dir"] = str(output_dir)
            st.success("Analysis completed.")
        except Exception as exc:  # pragma: no cover
            st.error(str(exc))
            st.exception(exc)

    last_output = st.session_state.get("last_output_dir")
    if last_output:
        result_dir = Path(last_output)
        st.subheader("Results")
        cols = st.columns(3)
        with cols[0]:
            _open_report_button("Open summary report", result_dir / "OmicsPrism_Report.html")
        with cols[1]:
            _open_report_button("Open interactive report", result_dir / "OmicsPrism_Interactive_Report.html")
        with cols[2]:
            zip_path = _zip_directory(result_dir, result_dir / "OmicsPrism_results.zip")
            st.download_button(
                "Download result ZIP",
                data=zip_path.read_bytes(),
                file_name=zip_path.name,
                mime="application/zip",
            )

        log_path = result_dir / "omicsprism.log"
        if log_path.exists():
            with st.expander("Run log", expanded=False):
                st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:])


if __name__ == "__main__":
    main()
