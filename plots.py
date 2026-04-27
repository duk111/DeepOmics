# visualize_only.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepomics.config import AnalysisConfig
from deepomics import plotting


TABLE_FILES = {
    "gene_scores_df": "T01_Metabolite_Gene_Scoring_Table.csv",
    "total_association_network_df": "T02_Total_Association_Network.csv",
    "high_confidence_network_df": "T03_High_Confidence_Network.csv",
    "key_gene_summary_df": "T04_Key_Gene_Summary.csv",
    "metabolite_summary": "T05_Metabolite_Association_Summary.csv",
    "cytoscape_network_df": "T06_Association_Network_Cytoscape.csv",
    "gene_module_assignment_df": "T07_Gene_Module_Assignment.csv",
    "module_eigengenes_df": "T08_Module_Eigengenes.csv",
    "module_metabolite_assoc_df": "T09_Module_Metabolite_Association.csv",
    "module_summary_df": "T10_Module_Summary.csv",
}


@dataclass
class VisualizationOnlyEngine:
    """Minimal engine object required by deepomics.plotting."""

    adata: ad.AnnData
    ml_results: dict[str, pd.DataFrame]

    def gene_expression_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.adata.X,
            index=self.adata.obs_names.astype(str),
            columns=self.adata.var_names.astype(str),
        )

    def metabolomics_df(self) -> pd.DataFrame:
        metab_df = self.adata.obsm.get(
            "metabolomics_scaled",
            self.adata.obsm.get("metabolomics"),
        )

        if metab_df is None:
            raise KeyError(
                "Missing metabolomics data. Expected adata.obsm['metabolomics'] "
                "or adata.obsm['metabolomics_scaled']."
            )

        if isinstance(metab_df, pd.DataFrame):
            return metab_df.copy(deep=False)

        metabolite_names = [str(x) for x in self.adata.uns.get("metabolite_names", [])]
        return pd.DataFrame(
            metab_df,
            index=self.adata.obs_names.astype(str),
            columns=metabolite_names,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate DeepOmics visualizations from saved h5ad and CSV result tables."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        type=Path,
        help="Existing DeepOmics result directory containing .h5ad and T01~T10 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for regenerated plots/reports. Defaults to --result-dir.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Optional explicit .h5ad path. If omitted, the script auto-detects one in --result-dir.",
    )
    parser.add_argument(
        "--group-table",
        type=Path,
        default=None,
        help="Optional PCA group table with sample_id and group/group1 columns.",
    )
    parser.add_argument(
        "--project",
        default="DeepOmics_Visualization_Test",
        help="Project name used in regenerated reports.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["html"],
        choices=["html", "md"],
        help="Report formats to regenerate.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only regenerate figures under plots/, skip HTML/Markdown reports.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Export only PNG files, useful for fast visual iteration.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def find_h5ad(result_dir: Path, explicit_state: Path | None) -> Path:
    if explicit_state is not None:
        state_path = resolve_path(explicit_state)
        if not state_path.exists():
            raise FileNotFoundError(f"h5ad file not found: {state_path}")
        return state_path

    candidates = sorted(result_dir.glob("*.h5ad"))
    if not candidates:
        raise FileNotFoundError(
            f"No .h5ad file found in {result_dir}. "
            "Run the original analysis without --no-save-state first, or pass --state explicitly."
        )

    if len(candidates) > 1:
        names = "\n".join(f"  - {path.name}" for path in candidates)
        raise RuntimeError(
            f"Multiple .h5ad files found in {result_dir}:\n{names}\n"
            "Please pass --state explicitly."
        )

    return candidates[0]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing table: {path.name}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_ml_results(result_dir: Path) -> dict[str, pd.DataFrame]:
    ml_results: dict[str, pd.DataFrame] = {}

    for result_key, file_name in TABLE_FILES.items():
        ml_results[result_key] = read_csv_if_exists(result_dir / file_name)

    return ml_results


def build_config(args: argparse.Namespace, output_dir: Path) -> AnalysisConfig:
    group_table = resolve_path(args.group_table) if args.group_table is not None else None

    return AnalysisConfig(
        project_name=args.project,
        output_dir=str(output_dir),
        group_table_path=str(group_table) if group_table is not None else None,
        generate_reports=not args.no_report,
        report_formats=tuple(args.formats),
        export_pdf=not args.png_only,
        export_svg=not args.png_only,
        export_png=True,
        save_h5ad=False,
    )


def main() -> None:
    args = parse_args()

    result_dir = resolve_path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = find_h5ad(result_dir, args.state)

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Result dir:   {result_dir}")
    print(f"[INFO] Output dir:   {output_dir}")
    print(f"[INFO] h5ad state:   {state_path}")

    adata = ad.read_h5ad(state_path)
    ml_results = load_ml_results(result_dir)

    engine = VisualizationOnlyEngine(
        adata=adata,
        ml_results=ml_results,
    )
    cfg = build_config(args, output_dir)

    plotting.generate_report_plots(engine, cfg)

    print("[DONE] Visualization regenerated.")
    print(f"[DONE] Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
