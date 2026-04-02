from __future__ import annotations

from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, plotting, selectors
from .utils import get_logger, log_step, safe_mkdir, write_json

logger = get_logger()


TABLE_FILE_PREFIXES = {
    "grn_edges_full": "T01_GRN_Edges_Full.csv",
    "grn_edges_cytoscape": "T02_GRN_Edges_Cytoscape.csv",
    "key_genes_consolidated": "T03_Key_Genes_Consolidated.csv",
    "ml_metabolite_summary": "T04_ML_Metabolite_Summary.csv",
}


class MultiOmicsEngine:
    """Core analysis engine for transcriptome-metabolome integration."""

    def __init__(self, adata: ad.AnnData, cfg: config.AnalysisConfig):
        self.adata = adata
        self.config = cfg
        self._validate_adata()

        self._gene_names = np.asarray(self.adata.var_names, dtype=str)
        self._gene_index = pd.Index(self._gene_names)
        self._gene_matrix = np.asarray(self.adata.X, dtype=np.float32)
        self._metabolomics_df = self._coerce_metabolomics_df()

        self.ml_results: Dict[str, object] = {
            "grn_edges": [],
            "grn_edges_df": pd.DataFrame(),
            "key_genes_intersection": pd.DataFrame(),
            "key_genes_borda": pd.DataFrame(),
            "key_genes_rra": pd.DataFrame(),
            "metabolite_summary": pd.DataFrame(),
        }
        self.run_metadata: Dict[str, object] = {
            "project_name": self.config.project_name,
            "n_samples": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "n_metabolites": int(len(self.adata.uns.get("metabolite_names", []))),
        }

    def _validate_adata(self) -> None:
        if self.adata.n_obs < 3:
            raise ValueError("At least 3 samples are required to run DeepOmics.")
        if self.adata.n_vars < 2:
            raise ValueError("At least 2 genes are required to run DeepOmics.")
        if "metabolomics" not in self.adata.obsm and "metabolomics_scaled" not in self.adata.obsm:
            raise KeyError("AnnData must contain metabolomics data in obsm['metabolomics'].")
        if len(self.adata.uns.get("metabolite_names", [])) == 0:
            raise ValueError("adata.uns['metabolite_names'] is empty.")

    def _coerce_metabolomics_df(self) -> pd.DataFrame:
        metabolites = [str(name) for name in self.adata.uns["metabolite_names"]]
        metab_df = self.adata.obsm.get("metabolomics_scaled", self.adata.obsm["metabolomics"])
        if isinstance(metab_df, pd.DataFrame):
            return metab_df.copy(deep=False)
        return pd.DataFrame(
            np.asarray(metab_df, dtype=np.float32),
            index=self.adata.obs_names.astype(str),
            columns=metabolites,
        )

    def gene_expression_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._gene_matrix, index=self.adata.obs_names.astype(str), columns=self._gene_names)

    def metabolomics_df(self) -> pd.DataFrame:
        return self._metabolomics_df.copy(deep=False)

    def run_all(self, generate_plots: bool = True) -> None:
        logger.info("=" * 80)
        logger.info("Project [%s] started", self.config.project_name)
        logger.info("=" * 80)

        with log_step(logger, "Ensemble machine learning"):
            self._run_ml_ensemble()

        with log_step(logger, "Saving outputs"):
            self.save_results()

        if generate_plots:
            with log_step(logger, "Figure and report generation"):
                plotting.generate_report_plots(self, self.config)

        logger.info("All analyses finished. Results saved to: %s", self.config.output_dir)

    def _run_ml_ensemble(self) -> None:
        metabolites = list(self.adata.uns["metabolite_names"])
        edge_records: List[Dict[str, object]] = []
        all_key_gene_rows: List[Dict[str, object]] = []
        metabolite_rows: List[Dict[str, object]] = []

        logger.info("Processing %d metabolites with ensemble models.", len(metabolites))

        for metab_name in tqdm(metabolites, desc="Ensemble Learning"):
            y = self._metabolomics_df[metab_name].to_numpy(dtype=np.float32, copy=False)

            candidate_genes, pcc_stats = selectors.filter_by_pcc(
                self._gene_matrix,
                y,
                self.config,
                feature_names=self._gene_names,
                return_stats=True,
            )

            summary_row = {
                "Metabolite": metab_name,
                "Candidate_Genes_PCC": int(len(candidate_genes)),
                "Intersection_Genes": 0,
                "Borda_Genes": 0,
                "RRA_Genes": 0,
            }

            if not candidate_genes:
                metabolite_rows.append(summary_row)
                continue

            candidate_positions = self._gene_index.get_indexer(candidate_genes)
            candidate_positions = candidate_positions[candidate_positions >= 0]
            if len(candidate_positions) == 0:
                metabolite_rows.append(summary_row)
                continue

            X_sub = self._gene_matrix[:, candidate_positions]
            feature_names = self._gene_names[candidate_positions]

            result_dict, _score_table = selectors.get_integrated_key_genes(
                X_sub,
                y,
                self.config,
                feature_names=feature_names,
            )

            summary_row.update(
                {
                    "Intersection_Genes": int(len(result_dict["intersection"])),
                    "Borda_Genes": int(len(result_dict["borda"])),
                    "RRA_Genes": int(len(result_dict["rra"])),
                }
            )
            metabolite_rows.append(summary_row)

            support_map: Dict[str, Dict[str, object]] = {}
            for strategy in ("intersection", "borda", "rra"):
                support_key = f"In_{strategy.capitalize() if strategy != 'rra' else 'RRA'}"
                for rank_idx, gene in enumerate(result_dict[strategy], start=1):
                    support = support_map.setdefault(
                        gene,
                        {
                            "Metabolite": metab_name,
                            "Gene": gene,
                            "In_Intersection": 0,
                            "In_Borda": 0,
                            "In_RRA": 0,
                            "Best_Rank": rank_idx,
                        },
                    )
                    support[support_key] = 1
                    support["Best_Rank"] = min(int(support["Best_Rank"]), rank_idx)

                    all_key_gene_rows.append(
                        {
                            "Strategy": strategy,
                            "Metabolite": metab_name,
                            "Gene": gene,
                            "Rank": rank_idx,
                        }
                    )

            for gene, support in support_map.items():
                pcc_r = float(pcc_stats.loc[gene, "R"]) if gene in pcc_stats.index else np.nan
                pcc_p = float(pcc_stats.loc[gene, "P"]) if gene in pcc_stats.index else np.nan
                support_count = int(support["In_Intersection"] + support["In_Borda"] + support["In_RRA"])
                edge_records.append(
                    {
                        "Source": gene,
                        "Target": metab_name,
                        "Interaction": "gene_to_metabolite",
                        "Support_Count": support_count,
                        "Primary_Strategy": self.config.grn_primary_strategy,
                        "PCC_R": pcc_r,
                        "PCC_P": pcc_p,
                        **support,
                    }
                )

        self.ml_results["grn_edges"] = edge_records
        self.ml_results["grn_edges_df"] = pd.DataFrame.from_records(edge_records)
        full_report = pd.DataFrame.from_records(all_key_gene_rows)

        for strategy in ("intersection", "borda", "rra"):
            self.ml_results[f"key_genes_{strategy}"] = self._build_key_gene_table(full_report, strategy)

        self.ml_results["metabolite_summary"] = pd.DataFrame.from_records(metabolite_rows)

    @staticmethod
    def _build_key_gene_table(full_report: pd.DataFrame, strategy: str) -> pd.DataFrame:
        base_columns = ["Gene", "Associated_Metabolites_Count", "Associated_Metabolites", "Median_Rank", "Best_Rank"]
        if full_report.empty:
            return pd.DataFrame(columns=base_columns)

        subset = full_report.loc[full_report["Strategy"] == strategy].copy()
        if subset.empty:
            return pd.DataFrame(columns=base_columns)

        return (
            subset.groupby("Gene", sort=False)
            .agg(
                Associated_Metabolites_Count=("Metabolite", "nunique"),
                Associated_Metabolites=("Metabolite", lambda x: "|".join(sorted(set(x)))),
                Median_Rank=("Rank", "median"),
                Best_Rank=("Rank", "min"),
            )
            .sort_values(["Associated_Metabolites_Count", "Best_Rank"], ascending=[False, True])
            .reset_index()
        )

    def _get_primary_key_gene_df(self) -> pd.DataFrame:
        return self.ml_results.get(f"key_genes_{self.config.grn_primary_strategy}", pd.DataFrame())

    def save_results(self) -> None:
        out_dir = safe_mkdir(self.config.output_dir)

        grn_edges_df = self.ml_results.get("grn_edges_df", pd.DataFrame())
        if isinstance(grn_edges_df, pd.DataFrame) and not grn_edges_df.empty:
            grn_edges_export = (
                grn_edges_df.assign(Abs_PCC_R=grn_edges_df["PCC_R"].abs())
                .sort_values(["Support_Count", "In_RRA", "Abs_PCC_R"], ascending=[False, False, False])
                .drop(columns=["Abs_PCC_R"])
            )
            grn_edges_export.to_csv(out_dir / TABLE_FILE_PREFIXES["grn_edges_full"], index=False)

            if self.config.export_cytoscape:
                (
                    grn_edges_export.loc[:, ["Source", "Target", "Interaction", "Support_Count", "PCC_R", "PCC_P"]]
                    .rename(columns={"Source": "source", "Target": "target", "Interaction": "interaction"})
                    .to_csv(out_dir / TABLE_FILE_PREFIXES["grn_edges_cytoscape"], index=False)
                )

        primary_key_genes = self._get_primary_key_gene_df()
        if isinstance(primary_key_genes, pd.DataFrame) and not primary_key_genes.empty:
            (
                primary_key_genes.assign(Strategy=self.config.grn_primary_strategy.upper())
                .loc[:, ["Strategy", "Gene", "Associated_Metabolites_Count", "Associated_Metabolites", "Median_Rank", "Best_Rank"]]
                .to_csv(out_dir / TABLE_FILE_PREFIXES["key_genes_consolidated"], index=False)
            )

        metabolite_summary = self.ml_results.get("metabolite_summary", pd.DataFrame())
        if isinstance(metabolite_summary, pd.DataFrame) and not metabolite_summary.empty:
            metabolite_summary.to_csv(out_dir / TABLE_FILE_PREFIXES["ml_metabolite_summary"], index=False)

        self.run_metadata["n_grn_edges"] = int(len(grn_edges_df)) if isinstance(grn_edges_df, pd.DataFrame) else 0

        write_json(
            {
                "project": self.config.project_name,
                "config": self.config.to_dict(),
                "summary": self.run_metadata,
            },
            out_dir / "analysis_metadata.json",
        )

        if self.config.save_h5ad:
            self.adata.write_h5ad(out_dir / "DeepOmics_Final_State.h5ad")

        logger.info("Structured result tables have been exported to %s", out_dir)
