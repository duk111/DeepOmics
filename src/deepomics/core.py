
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
    "gene_scores": "T01_Metabolite_Gene_Scoring_Table.csv",
    "total_network": "T02_Total_Association_Network.csv",
    "high_confidence_network": "T03_High_Confidence_Network.csv",
    "key_gene_summary": "T04_Key_Gene_Summary.csv",
    "metabolite_summary": "T05_Metabolite_Association_Summary.csv",
    "cytoscape_network": "T06_Association_Network_Cytoscape.csv",
}


class MultiOmicsEngine:
    """Core analysis engine for transcriptome-metabolome association modeling."""

    def __init__(self, adata: ad.AnnData, cfg: config.AnalysisConfig):
        self.adata = adata
        self.config = cfg
        self._validate_adata()

        self._gene_names = np.asarray(self.adata.var_names, dtype=str)
        self._gene_index = pd.Index(self._gene_names)
        self._gene_matrix = np.asarray(self.adata.X, dtype=np.float32)
        self._metabolomics_df = self._coerce_metabolomics_df()

        self.ml_results: Dict[str, object] = {
            "gene_scores_df": pd.DataFrame(),
            "total_association_network_df": pd.DataFrame(),
            "high_confidence_network_df": pd.DataFrame(),
            "key_gene_summary_df": pd.DataFrame(),
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

        with log_step(logger, "Gene-metabolite association modeling"):
            self._run_association_analysis()

        with log_step(logger, "Saving outputs"):
            self.save_results()

        if generate_plots:
            with log_step(logger, "Figure and report generation"):
                plotting.generate_report_plots(self, self.config)

        logger.info("All analyses finished. Results saved to: %s", self.config.output_dir)

    @staticmethod
    def _infer_association_sign(row: pd.Series) -> str:
        pearson_r = float(row.get("PearsonR", 0.0)) if pd.notna(row.get("PearsonR", np.nan)) else 0.0
        spearman_rho = float(row.get("SpearmanRho", 0.0)) if pd.notna(row.get("SpearmanRho", np.nan)) else 0.0
        basis = pearson_r if abs(pearson_r) >= abs(spearman_rho) else spearman_rho
        return "positive" if basis >= 0 else "negative"

    @staticmethod
    def _compute_group_weights(group_df: pd.DataFrame) -> pd.DataFrame:
        group = group_df.copy()
        n_candidates = int(len(group))
        if n_candidates <= 1:
            group["RRAWeight"] = 1.0
        else:
            group["RRAWeight"] = 1.0 - (group["RRARank"].astype(float) - 1.0) / float(n_candidates - 1)

        group["CorrScore"] = np.maximum(group["PearsonR"].abs(), group["SpearmanRho"].abs()).clip(0.0, 1.0)
        group["ModelScore"] = (group["ModelSupportCount"].astype(float) / 2.0).clip(0.0, 1.0)
        group["ScreenScore"] = (group["ScreenSupportCount"].astype(float) / 3.0).clip(0.0, 1.0)
        group["EdgeWeight"] = (
            0.45 * group["RRAWeight"]
            + 0.25 * group["CorrScore"]
            + 0.20 * group["ModelScore"]
            + 0.10 * group["ScreenScore"]
        ).clip(0.0, 1.0)
        return group

    @staticmethod
    def _network_edge_columns() -> list[str]:
        return [
            "Source",
            "Target",
            "Interaction",
            "Gene",
            "Metabolite",
            "PearsonR",
            "PearsonP",
            "PearsonFDR",
            "SpearmanRho",
            "SpearmanP",
            "SpearmanFDR",
            "MIScore",
            "In_PCC",
            "In_Spearman",
            "In_MI",
            "ScreenSupportCount",
            "ElasticNetScore",
            "ElasticNetRank",
            "ElasticNetSelected",
            "XGBoostScore",
            "XGBoostRank",
            "XGBoostSelected",
            "ModelSupportCount",
            "RRAScore",
            "RRARank",
            "RRAWeight",
            "CorrScore",
            "ModelScore",
            "ScreenScore",
            "EdgeWeight",
            "Sign",
            "EdgeTier",
            "TargetK",
        ]

    def _build_network_tables(self, gene_scores_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if gene_scores_df.empty:
            empty_network = pd.DataFrame(columns=self._network_edge_columns())
            return empty_network.copy(), empty_network.copy()

        base_df = gene_scores_df.copy()
        base_df["Source"] = base_df["Gene"].astype(str)
        base_df["Target"] = base_df["Metabolite"].astype(str)
        base_df["Interaction"] = "association"

        total_mask = (base_df["ElasticNetSelected"] == 1) | (base_df["XGBoostSelected"] == 1)
        total_df = base_df.loc[total_mask].copy()
        total_df["EdgeTier"] = "total"

        high_conf_mask = (
            (base_df["TargetK"] > 0)
            & (base_df["RRARank"] <= base_df["TargetK"])
            & ((base_df["ModelSupportCount"] == 2) | (base_df["ScreenSupportCount"] >= 2))
        )
        high_conf_df = base_df.loc[high_conf_mask].copy()
        high_conf_df["EdgeTier"] = "high_confidence"

        network_columns = self._network_edge_columns()
        total_df = total_df.loc[:, network_columns].sort_values(
            ["Metabolite", "EdgeWeight", "RRARank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        high_conf_df = high_conf_df.loc[:, network_columns].sort_values(
            ["Metabolite", "EdgeWeight", "RRARank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        return total_df.reset_index(drop=True), high_conf_df.reset_index(drop=True)

    @staticmethod
    def _build_key_gene_summary(
        total_network_df: pd.DataFrame,
        high_confidence_network_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if total_network_df.empty and high_confidence_network_df.empty:
            return pd.DataFrame(
                columns=[
                    "Gene",
                    "AssociatedMetaboliteCount",
                    "AssociatedMetabolites",
                    "HighConfidenceMetaboliteCount",
                    "HighConfidenceMetabolites",
                    "MeanRRARank",
                    "BestRRARank",
                    "MeanEdgeWeight",
                    "BestEdgeWeight",
                ]
            )

        total_group = None
        if not total_network_df.empty:
            total_group = total_network_df.groupby("Gene", sort=False)
        high_group = None
        if not high_confidence_network_df.empty:
            high_group = high_confidence_network_df.groupby("Gene", sort=False)

        genes = pd.Index(
            sorted(
                set(total_network_df["Gene"].astype(str).tolist()) | set(high_confidence_network_df["Gene"].astype(str).tolist())
            ),
            name="Gene",
        )

        summary = pd.DataFrame(index=genes)
        if total_group is not None:
            summary["AssociatedMetaboliteCount"] = total_group["Metabolite"].nunique().reindex(genes).fillna(0).astype(int)
            summary["AssociatedMetabolites"] = total_group["Metabolite"].agg(
                lambda values: "|".join(sorted({str(v) for v in values}))
            ).reindex(genes).fillna("")
            summary["MeanRRARank"] = total_group["RRARank"].mean().reindex(genes).fillna(np.nan)
            summary["BestRRARank"] = total_group["RRARank"].min().reindex(genes).fillna(np.nan)
            summary["MeanEdgeWeight"] = total_group["EdgeWeight"].mean().reindex(genes).fillna(np.nan)
            summary["BestEdgeWeight"] = total_group["EdgeWeight"].max().reindex(genes).fillna(np.nan)
        else:
            summary["AssociatedMetaboliteCount"] = 0
            summary["AssociatedMetabolites"] = ""
            summary["MeanRRARank"] = np.nan
            summary["BestRRARank"] = np.nan
            summary["MeanEdgeWeight"] = np.nan
            summary["BestEdgeWeight"] = np.nan

        if high_group is not None:
            summary["HighConfidenceMetaboliteCount"] = high_group["Metabolite"].nunique().reindex(genes).fillna(0).astype(int)
            summary["HighConfidenceMetabolites"] = high_group["Metabolite"].agg(
                lambda values: "|".join(sorted({str(v) for v in values}))
            ).reindex(genes).fillna("")
        else:
            summary["HighConfidenceMetaboliteCount"] = 0
            summary["HighConfidenceMetabolites"] = ""

        summary = summary.reset_index().sort_values(
            ["HighConfidenceMetaboliteCount", "AssociatedMetaboliteCount", "BestEdgeWeight", "BestRRARank"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        return summary

    @staticmethod
    def _build_metabolite_summary(
        gene_scores_df: pd.DataFrame,
        total_network_df: pd.DataFrame,
        high_confidence_network_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if gene_scores_df.empty:
            return pd.DataFrame(
                columns=[
                    "Metabolite",
                    "CandidateGenes",
                    "TargetK",
                    "TotalAssociationEdges",
                    "HighConfidenceEdges",
                    "DualModelEdges",
                    "MultiScreenEdges",
                    "MeanScreenSupportCount",
                    "MeanModelSupportCount",
                    "TopGene",
                    "TopEdgeWeight",
                ]
            )

        base_group = gene_scores_df.groupby("Metabolite", sort=False)
        total_group = total_network_df.groupby("Metabolite", sort=False) if not total_network_df.empty else None
        high_group = (
            high_confidence_network_df.groupby("Metabolite", sort=False)
            if not high_confidence_network_df.empty
            else None
        )

        top_gene_df = gene_scores_df.sort_values(
            ["Metabolite", "EdgeWeight", "RRARank"],
            ascending=[True, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["Metabolite"], keep="first")

        metabolites = pd.Index(gene_scores_df["Metabolite"].astype(str).unique(), name="Metabolite")
        summary = pd.DataFrame(index=metabolites)
        summary["CandidateGenes"] = base_group.size().reindex(metabolites).fillna(0).astype(int)
        summary["TargetK"] = base_group["TargetK"].max().reindex(metabolites).fillna(0).astype(int)
        summary["MeanScreenSupportCount"] = base_group["ScreenSupportCount"].mean().reindex(metabolites).fillna(0.0)
        summary["MeanModelSupportCount"] = base_group["ModelSupportCount"].mean().reindex(metabolites).fillna(0.0)
        summary["DualModelEdges"] = (
            base_group.apply(lambda df: int((df["ModelSupportCount"] == 2).sum()))
            .reindex(metabolites)
            .fillna(0)
            .astype(int)
        )
        summary["MultiScreenEdges"] = (
            base_group.apply(lambda df: int((df["ScreenSupportCount"] >= 2).sum()))
            .reindex(metabolites)
            .fillna(0)
            .astype(int)
        )

        if total_group is not None:
            summary["TotalAssociationEdges"] = total_group.size().reindex(metabolites).fillna(0).astype(int)
        else:
            summary["TotalAssociationEdges"] = 0

        if high_group is not None:
            summary["HighConfidenceEdges"] = high_group.size().reindex(metabolites).fillna(0).astype(int)
        else:
            summary["HighConfidenceEdges"] = 0

        top_gene_map = top_gene_df.set_index("Metabolite")
        summary["TopGene"] = top_gene_map["Gene"].reindex(metabolites).fillna("")
        summary["TopEdgeWeight"] = top_gene_map["EdgeWeight"].reindex(metabolites).fillna(np.nan)

        summary = summary.reset_index().sort_values(
            ["HighConfidenceEdges", "TotalAssociationEdges", "CandidateGenes"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        return summary

    def _run_association_analysis(self) -> None:
        metabolites = list(self.adata.uns["metabolite_names"])
        per_metabolite_tables: List[pd.DataFrame] = []

        logger.info(
            "Processing %d metabolites using three-way screening + ElasticNet/XGBoost/RRA.",
            len(metabolites),
        )

        for metab_name in tqdm(metabolites, desc="Association Modeling"):
            y = self._metabolomics_df[metab_name].to_numpy(dtype=np.float32, copy=False)

            screen_df = selectors.screen_genes_three_way(
                self._gene_matrix,
                y,
                self.config,
                feature_names=self._gene_names,
            )
            if screen_df.empty:
                continue

            candidate_genes = screen_df.index.astype(str).tolist()
            candidate_positions = self._gene_index.get_indexer(candidate_genes)
            candidate_positions = candidate_positions[candidate_positions >= 0]
            if len(candidate_positions) == 0:
                continue

            X_sub = self._gene_matrix[:, candidate_positions]
            feature_names = self._gene_names[candidate_positions]

            model_df, target_k = selectors.run_association_models(
                X_sub,
                y,
                self.config,
                feature_names=feature_names,
            )

            combined_df = (
                screen_df.reindex(feature_names)
                .join(model_df.reindex(feature_names), how="left")
                .reset_index()
                .rename(columns={"index": "Gene"})
            )
            combined_df.insert(0, "Metabolite", str(metab_name))
            combined_df["TargetK"] = int(target_k)
            combined_df["Sign"] = combined_df.apply(self._infer_association_sign, axis=1)
            per_metabolite_tables.append(combined_df)

        if not per_metabolite_tables:
            self.ml_results["gene_scores_df"] = pd.DataFrame()
            self.ml_results["total_association_network_df"] = pd.DataFrame()
            self.ml_results["high_confidence_network_df"] = pd.DataFrame()
            self.ml_results["key_gene_summary_df"] = pd.DataFrame()
            self.ml_results["metabolite_summary"] = pd.DataFrame()
            return

        gene_scores_df = pd.concat(per_metabolite_tables, ignore_index=True)
        gene_scores_df = (
            gene_scores_df.groupby("Metabolite", group_keys=False, sort=False)
            .apply(self._compute_group_weights)
            .reset_index(drop=True)
        )

        gene_scores_df = gene_scores_df.loc[
            :,
            [
                "Gene",
                "Metabolite",
                "PearsonR",
                "PearsonP",
                "PearsonFDR",
                "SpearmanRho",
                "SpearmanP",
                "SpearmanFDR",
                "MIScore",
                "In_PCC",
                "In_Spearman",
                "In_MI",
                "ScreenSupportCount",
                "ElasticNetScore",
                "ElasticNetRank",
                "ElasticNetSelected",
                "XGBoostScore",
                "XGBoostRank",
                "XGBoostSelected",
                "ModelSupportCount",
                "RRAScore",
                "RRARank",
                "RRAWeight",
                "CorrScore",
                "ModelScore",
                "ScreenScore",
                "EdgeWeight",
                "Sign",
                "TargetK",
            ],
        ].sort_values(
            ["Metabolite", "RRARank", "EdgeWeight", "Gene"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        total_network_df, high_confidence_network_df = self._build_network_tables(gene_scores_df)
        key_gene_summary_df = self._build_key_gene_summary(total_network_df, high_confidence_network_df)
        metabolite_summary_df = self._build_metabolite_summary(
            gene_scores_df,
            total_network_df,
            high_confidence_network_df,
        )

        self.ml_results["gene_scores_df"] = gene_scores_df
        self.ml_results["total_association_network_df"] = total_network_df
        self.ml_results["high_confidence_network_df"] = high_confidence_network_df
        self.ml_results["key_gene_summary_df"] = key_gene_summary_df
        self.ml_results["metabolite_summary"] = metabolite_summary_df

    def save_results(self) -> None:
        out_dir = safe_mkdir(self.config.output_dir)

        gene_scores_df = self.ml_results.get("gene_scores_df", pd.DataFrame())
        if isinstance(gene_scores_df, pd.DataFrame) and not gene_scores_df.empty:
            gene_scores_df.to_csv(out_dir / TABLE_FILE_PREFIXES["gene_scores"], index=False)

        total_network_df = self.ml_results.get("total_association_network_df", pd.DataFrame())
        if isinstance(total_network_df, pd.DataFrame) and not total_network_df.empty:
            total_network_df.to_csv(out_dir / TABLE_FILE_PREFIXES["total_network"], index=False)

        high_confidence_network_df = self.ml_results.get("high_confidence_network_df", pd.DataFrame())
        if isinstance(high_confidence_network_df, pd.DataFrame) and not high_confidence_network_df.empty:
            high_confidence_network_df.to_csv(out_dir / TABLE_FILE_PREFIXES["high_confidence_network"], index=False)

        key_gene_summary_df = self.ml_results.get("key_gene_summary_df", pd.DataFrame())
        if isinstance(key_gene_summary_df, pd.DataFrame) and not key_gene_summary_df.empty:
            key_gene_summary_df.to_csv(out_dir / TABLE_FILE_PREFIXES["key_gene_summary"], index=False)

        metabolite_summary = self.ml_results.get("metabolite_summary", pd.DataFrame())
        if isinstance(metabolite_summary, pd.DataFrame) and not metabolite_summary.empty:
            metabolite_summary.to_csv(out_dir / TABLE_FILE_PREFIXES["metabolite_summary"], index=False)

        if self.config.export_cytoscape:
            cytoscape_df = pd.concat(
                [
                    total_network_df if isinstance(total_network_df, pd.DataFrame) else pd.DataFrame(),
                    high_confidence_network_df if isinstance(high_confidence_network_df, pd.DataFrame) else pd.DataFrame(),
                ],
                ignore_index=True,
            )
            if not cytoscape_df.empty:
                cytoscape_df = cytoscape_df.rename(
                    columns={
                        "Source": "source",
                        "Target": "target",
                        "Interaction": "interaction",
                        "EdgeTier": "edge_tier",
                        "EdgeWeight": "edge_weight",
                        "RRARank": "rra_rank",
                        "RRAScore": "rra_score",
                        "ScreenSupportCount": "screen_support_count",
                        "ModelSupportCount": "model_support_count",
                    }
                )
                cytoscape_df.to_csv(out_dir / TABLE_FILE_PREFIXES["cytoscape_network"], index=False)

        self.run_metadata["n_total_association_edges"] = (
            int(len(total_network_df)) if isinstance(total_network_df, pd.DataFrame) else 0
        )
        self.run_metadata["n_high_confidence_edges"] = (
            int(len(high_confidence_network_df)) if isinstance(high_confidence_network_df, pd.DataFrame) else 0
        )

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
