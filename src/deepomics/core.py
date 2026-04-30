from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import re

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, modules, plotting, selectors
from .outputs import TABLE_FILE_PREFIXES
from .utils import get_logger, log_step, safe_mkdir, write_json

logger = get_logger()


class MultiOmicsEngine:
    """Core analysis engine for transcriptome-metabolome association modeling."""

    def __init__(self, adata: ad.AnnData, cfg: config.AnalysisConfig):
        self.plot_adata = adata.copy()
        self.adata = self._average_replicates_for_association(adata, cfg)
        self.config = cfg
        self._validate_adata()

        self._obs_names = self.adata.obs_names.astype(str)
        self._gene_names = np.asarray(self.adata.var_names, dtype=str)
        self._gene_index = pd.Index(self._gene_names)
        self._gene_matrix = np.asarray(self.adata.X, dtype=np.float32)

        self._metabolomics_df = self._coerce_metabolomics_df()
        self._metabolite_names = np.asarray(self.adata.uns["metabolite_names"], dtype=str)
        self._metabolomics_matrix = self._metabolomics_df.to_numpy(dtype=np.float32, copy=False)

        self.ml_results: Dict[str, object] = {
            "gene_scores_df": pd.DataFrame(),
            "screening_summary_df": pd.DataFrame(),
            "total_association_network_df": pd.DataFrame(),
            "high_confidence_network_df": pd.DataFrame(),
            "key_gene_summary_df": pd.DataFrame(),
            "metabolite_summary": pd.DataFrame(),
            "gene_module_assignment_df": pd.DataFrame(),
            "module_eigengenes_df": pd.DataFrame(),
            "module_metabolite_assoc_df": pd.DataFrame(),
            "module_summary_df": pd.DataFrame(),
        }
        self.run_metadata: Dict[str, object] = {
            "project_name": self.config.project_name,
            "n_samples": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "n_metabolites": int(len(self.adata.uns.get("metabolite_names", []))),
            "n_plot_samples": int(self.plot_adata.n_obs),
            "preprocess_summary": self.adata.uns.get("preprocess_summary", {}),
        }

    @staticmethod
    def _normalize_group_columns(group_df: pd.DataFrame) -> pd.DataFrame:
        normalized_columns = {
            str(column).replace("\ufeff", "").strip().lower(): column
            for column in group_df.columns
        }
        required = {"sample_id", "group1", "group2"}
        missing = sorted(required.difference(normalized_columns))
        if missing:
            raise ValueError(
                "Group table used for replicate averaging must contain columns: "
                f"sample_id, group1, group2. Missing: {missing}"
            )
        result = group_df.loc[
            :,
            [
                normalized_columns["sample_id"],
                normalized_columns["group1"],
                normalized_columns["group2"],
            ],
        ].rename(
            columns={
                normalized_columns["sample_id"]: "sample_id",
                normalized_columns["group1"]: "group1",
                normalized_columns["group2"]: "group2",
            }
        )
        for column in ("sample_id", "group1", "group2"):
            result[column] = result[column].astype(str).str.strip()
        result = result.loc[
            result["sample_id"].ne("") & result["group1"].ne("") & result["group2"].ne("")
        ].copy()
        if result.empty:
            raise ValueError("Group table has no valid sample_id + group1 + group2 rows.")
        return result

    @staticmethod
    def _shared_sample_id_field(sample_ids: list[str], fallback: str) -> str:
        clean_ids = [str(sample_id).strip() for sample_id in sample_ids if str(sample_id).strip()]
        if not clean_ids:
            return str(fallback)
        if len(clean_ids) == 1:
            return clean_ids[0]

        split_tokens = [re.split(r"[^A-Za-z0-9]+", sample_id) for sample_id in clean_ids]
        min_len = min(len(tokens) for tokens in split_tokens)
        shared_tokens = []
        for idx in range(min_len):
            token = split_tokens[0][idx]
            if token and all(tokens[idx] == token for tokens in split_tokens[1:]):
                shared_tokens.append(token)
            else:
                break
        if shared_tokens:
            return "_".join(shared_tokens)

        common_prefix = clean_ids[0]
        for sample_id in clean_ids[1:]:
            while common_prefix and not sample_id.startswith(common_prefix):
                common_prefix = common_prefix[:-1]
        common_prefix = re.sub(r"[^A-Za-z0-9]+$", "", common_prefix.strip())
        if common_prefix:
            return common_prefix
        return str(fallback)

    @staticmethod
    def _average_dataframe_rows(df: pd.DataFrame, sample_groups: list[tuple[str, list[str]]]) -> pd.DataFrame:
        averaged_rows = []
        averaged_index = []
        for output_sample_id, member_sample_ids in sample_groups:
            averaged_rows.append(df.loc[member_sample_ids].mean(axis=0).to_numpy(dtype=np.float32))
            averaged_index.append(output_sample_id)
        return pd.DataFrame(averaged_rows, index=pd.Index(averaged_index, name=df.index.name), columns=df.columns)

    @classmethod
    def _average_replicates_for_association(cls, adata: ad.AnnData, cfg: config.AnalysisConfig) -> ad.AnnData:
        group_table_path = getattr(cfg, "group_table_path", None)
        if not group_table_path:
            raise ValueError(
                "group_table_path is required for replicate-aware association analysis. "
                "The group table must contain sample_id, group1, and group2 columns."
            )

        group_path = Path(group_table_path)
        if not group_path.exists():
            raise FileNotFoundError(f"Group table not found: {group_path}")

        group_df = pd.read_csv(group_path, sep=None, engine="python", encoding="utf-8-sig")
        group_df = cls._normalize_group_columns(group_df)

        obs_index = pd.Index(adata.obs_names.astype(str), name="sample_id")
        group_df = group_df.loc[group_df["sample_id"].isin(obs_index)].copy()
        if group_df.empty:
            logger.warning(
                "No group table samples overlap with the omics matrices; association analysis will use unaveraged samples."
            )
            return adata.copy()

        sample_groups: list[tuple[str, list[str]]] = []
        used_samples: set[str] = set()
        for (group1, group2), sub_df in group_df.groupby(["group1", "group2"], sort=False):
            member_ids = [str(sample_id) for sample_id in sub_df["sample_id"].tolist() if str(sample_id) in obs_index]
            if not member_ids:
                continue
            fallback_id = f"{group1}_{group2}"
            output_id = cls._shared_sample_id_field(member_ids, fallback=fallback_id)
            sample_groups.append((output_id, member_ids))
            used_samples.update(member_ids)

        for sample_id in obs_index.tolist():
            if sample_id not in used_samples:
                sample_groups.append((str(sample_id), [str(sample_id)]))

        if len(sample_groups) == adata.n_obs and all(len(members) == 1 and output_id == members[0] for output_id, members in sample_groups):
            logger.info("No repeated group1 + group2 measurements found; association analysis will use unaveraged samples.")
            return adata.copy()

        output_ids = [output_id for output_id, _ in sample_groups]
        if len(output_ids) != len(set(output_ids)):
            deduped: list[str] = []
            counts: dict[str, int] = {}
            for output_id in output_ids:
                counts[output_id] = counts.get(output_id, 0) + 1
                deduped.append(output_id if counts[output_id] == 1 else f"{output_id}_{counts[output_id]}")
            sample_groups = [(deduped[idx], members) for idx, (_, members) in enumerate(sample_groups)]
            output_ids = deduped

        gene_df = pd.DataFrame(
            np.asarray(adata.X, dtype=np.float32),
            index=obs_index,
            columns=adata.var_names.astype(str),
        )
        averaged_gene_df = cls._average_dataframe_rows(gene_df, sample_groups)

        averaged = ad.AnnData(
            X=averaged_gene_df.to_numpy(dtype=np.float32, copy=False),
            obs=pd.DataFrame(index=pd.Index(output_ids, name=adata.obs_names.name or "SampleID")),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
        )

        for key, value in adata.layers.items():
            layer_df = pd.DataFrame(
                np.asarray(value, dtype=np.float32),
                index=obs_index,
                columns=adata.var_names.astype(str),
            )
            averaged.layers[key] = cls._average_dataframe_rows(layer_df, sample_groups).to_numpy(dtype=np.float32, copy=False)

        for key, value in adata.obsm.items():
            if isinstance(value, pd.DataFrame):
                obsm_df = value.copy()
                obsm_df.index = obs_index
            else:
                obsm_df = pd.DataFrame(np.asarray(value), index=obs_index)
            if obsm_df.shape[0] != adata.n_obs:
                continue
            averaged.obsm[key] = cls._average_dataframe_rows(obsm_df, sample_groups)

        averaged.uns["replicate_averaging"] = {
            "enabled": True,
            "group_table_path": str(group_path),
            "grouping_columns": ["group1", "group2"],
            "n_input_samples": int(adata.n_obs),
            "n_output_samples": int(averaged.n_obs),
            "n_averaged_groups": int(sum(1 for _, members in sample_groups if len(members) > 1)),
            "used_for": "gene_metabolite_association_network_screening_and_modeling",
            "not_used_for": "F01_sample_clustering_and_F02_F03_PCA_plots",
        }
        logger.info(
            "Averaged repeated group1 + group2 measurements for association analysis: %d input samples -> %d analysis samples.",
            adata.n_obs,
            averaged.n_obs,
        )
        return averaged

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
            index=self._obs_names,
            columns=metabolites,
        )

    def gene_expression_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._gene_matrix, index=self._obs_names, columns=self._gene_names)

    def metabolomics_df(self) -> pd.DataFrame:
        return self._metabolomics_df.copy(deep=False)

    def run_all(self, generate_plots: bool = True) -> None:
        logger.info("=" * 80)
        logger.info("Project [%s] started", self.config.project_name)
        logger.info("=" * 80)

        with log_step(logger, "Gene-metabolite association modeling"):
            self._run_association_analysis()

        if bool(getattr(self.config, "enable_module_detection", True)):
            with log_step(logger, "Gene module detection"):
                self._run_module_analysis()

        with log_step(logger, "Saving outputs"):
            self.save_results()

        if generate_plots:
            with log_step(logger, "Figure and report generation"):
                plotting.generate_report_plots(self, self.config)

        logger.info("All analyses finished. Results saved to: %s", self.config.output_dir)

    @staticmethod
    def _compute_group_weights_vectorized(gene_scores_df: pd.DataFrame) -> pd.DataFrame:
        if gene_scores_df.empty:
            return gene_scores_df

        result = gene_scores_df.copy()
        group_sizes = result.groupby("Metabolite", sort=False)["Metabolite"].transform("size").astype(float)
        denom = (group_sizes - 1.0).where(group_sizes > 1.0, 1.0)

        rra_rank = result["RRARank"].astype(float)
        result["RRAWeight"] = np.where(
            group_sizes <= 1.0,
            1.0,
            1.0 - (rra_rank - 1.0) / denom,
        )
        result["CorrScore"] = np.maximum(result["PearsonR"].abs(), result["SpearmanRho"].abs()).clip(0.0, 1.0)
        result["ModelScore"] = (result["ModelSupportCount"].astype(float) / 2.0).clip(0.0, 1.0)
        result["ScreenScore"] = (result["ScreenSupportCount"].astype(float) / 3.0).clip(0.0, 1.0)
        result["EdgeWeight"] = (
            0.45 * result["RRAWeight"]
            + 0.25 * result["CorrScore"]
            + 0.20 * result["ModelScore"]
            + 0.10 * result["ScreenScore"]
        ).clip(0.0, 1.0)
        return result

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
            total_mask
            & (base_df["TargetK"] > 0)
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

        dual_model_counts = (
            gene_scores_df.assign(_DualModel=(gene_scores_df["ModelSupportCount"] == 2).astype(np.int8))
            .groupby("Metabolite", sort=False)["_DualModel"]
            .sum()
        )
        multi_screen_counts = (
            gene_scores_df.assign(_MultiScreen=(gene_scores_df["ScreenSupportCount"] >= 2).astype(np.int8))
            .groupby("Metabolite", sort=False)["_MultiScreen"]
            .sum()
        )
        summary["DualModelEdges"] = dual_model_counts.reindex(metabolites).fillna(0).astype(int)
        summary["MultiScreenEdges"] = multi_screen_counts.reindex(metabolites).fillna(0).astype(int)

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

    @staticmethod
    def _vectorized_sign_labels(df: pd.DataFrame) -> np.ndarray:
        pearson = df["PearsonR"].to_numpy(dtype=float, copy=False)
        spearman = df["SpearmanRho"].to_numpy(dtype=float, copy=False)

        pearson_abs = np.abs(np.nan_to_num(pearson, nan=0.0))
        spearman_abs = np.abs(np.nan_to_num(spearman, nan=0.0))
        basis = np.where(pearson_abs >= spearman_abs, np.nan_to_num(pearson, nan=0.0), np.nan_to_num(spearman, nan=0.0))
        return np.where(basis >= 0.0, "positive", "negative")

    def _run_association_analysis(self) -> None:
        per_metabolite_tables: List[pd.DataFrame] = []

        logger.info(
            "Processing %d metabolites using three-way screening + ElasticNet/XGBoost/RRA.",
            len(self._metabolite_names),
        )
        logger.info("Precomputing reusable Pearson/Spearman screening correlations across all metabolites.")
        screening_correlation_cache = selectors.prepare_screening_correlation_cache(
            self._gene_matrix,
            self._metabolomics_matrix,
            feature_names=self._gene_names,
        )

        for metab_idx, metab_name in enumerate(tqdm(self._metabolite_names, desc="Association Modeling")):
            y = self._metabolomics_matrix[:, metab_idx]

            screen_df = selectors.screen_genes_three_way(
                self._gene_matrix,
                y,
                self.config,
                feature_names=self._gene_names,
                correlation_cache=screening_correlation_cache,
                target_index=metab_idx,
            )
            if screen_df.empty:
                continue

            candidate_positions = self._gene_index.get_indexer(screen_df.index)
            candidate_positions = candidate_positions[candidate_positions >= 0]
            if len(candidate_positions) == 0:
                continue

            feature_names = self._gene_names[candidate_positions]
            X_sub = self._gene_matrix[:, candidate_positions]

            model_df, target_k = selectors.run_association_models(
                X_sub,
                y,
                self.config,
                feature_names=feature_names,
            )

            combined_df = (
                screen_df.loc[feature_names]
                .join(model_df, how="left")
                .reset_index()
                .rename(columns={"index": "Gene"})
            )
            combined_df.insert(0, "Metabolite", str(metab_name))
            combined_df["TargetK"] = int(target_k)
            combined_df["Sign"] = self._vectorized_sign_labels(combined_df)
            per_metabolite_tables.append(combined_df)

        if not per_metabolite_tables:
            self.ml_results["gene_scores_df"] = pd.DataFrame()
            self.ml_results["screening_summary_df"] = pd.DataFrame()
            self.ml_results["total_association_network_df"] = pd.DataFrame()
            self.ml_results["high_confidence_network_df"] = pd.DataFrame()
            self.ml_results["key_gene_summary_df"] = pd.DataFrame()
            self.ml_results["metabolite_summary"] = pd.DataFrame()
            return

        gene_scores_df = pd.concat(per_metabolite_tables, ignore_index=True)
        screening_summary_df = (
            gene_scores_df.groupby("Metabolite", sort=False)
            .agg(
                PCCSelectedGenes=("In_PCC", "sum"),
                SpearmanSelectedGenes=("In_Spearman", "sum"),
                MISelectedGenes=("In_MI", "sum"),
                TotalSelectedGenes=("Gene", "nunique"),
            )
            .reset_index()
        )
        screening_summary_df[
            [
                "PCCSelectedGenes",
                "SpearmanSelectedGenes",
                "MISelectedGenes",
                "TotalSelectedGenes",
            ]
        ] = screening_summary_df[
            [
                "PCCSelectedGenes",
                "SpearmanSelectedGenes",
                "MISelectedGenes",
                "TotalSelectedGenes",
            ]
        ].astype(int)

        gene_scores_df = self._compute_group_weights_vectorized(gene_scores_df)

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
        self.ml_results["screening_summary_df"] = screening_summary_df
        self.ml_results["total_association_network_df"] = total_network_df
        self.ml_results["high_confidence_network_df"] = high_confidence_network_df
        self.ml_results["key_gene_summary_df"] = key_gene_summary_df
        self.ml_results["metabolite_summary"] = metabolite_summary_df


    def _run_module_analysis(self) -> None:
        high_confidence_network_df = self.ml_results.get("high_confidence_network_df", pd.DataFrame())
        key_gene_summary_df = self.ml_results.get("key_gene_summary_df", pd.DataFrame())

        if not isinstance(high_confidence_network_df, pd.DataFrame) or high_confidence_network_df.empty:
            logger.info("Module analysis skipped because the high-confidence network is empty.")
            self.ml_results["gene_module_assignment_df"] = pd.DataFrame()
            self.ml_results["module_eigengenes_df"] = pd.DataFrame()
            self.ml_results["module_metabolite_assoc_df"] = pd.DataFrame()
            self.ml_results["module_summary_df"] = pd.DataFrame()
            self.run_metadata["module_method_used"] = "none"
            self.run_metadata["n_non_grey_modules"] = 0
            self.run_metadata["n_module_genes"] = 0
            self.run_metadata["n_grey_genes"] = 0
            return

        artifacts = modules.run_gene_module_analysis(
            expr_df=self.gene_expression_df(),
            metabolomics_df=self.metabolomics_df(),
            high_confidence_network_df=high_confidence_network_df,
            key_gene_summary_df=key_gene_summary_df if isinstance(key_gene_summary_df, pd.DataFrame) else pd.DataFrame(),
            cfg=self.config,
        )

        self.ml_results["gene_module_assignment_df"] = artifacts.gene_module_assignment_df
        self.ml_results["module_eigengenes_df"] = artifacts.module_eigengenes_df
        self.ml_results["module_metabolite_assoc_df"] = artifacts.module_metabolite_assoc_df
        self.ml_results["module_summary_df"] = artifacts.module_summary_df
        self.run_metadata.update(artifacts.metadata)

    def save_results(self) -> None:
        out_dir = safe_mkdir(self.config.output_dir)

        gene_scores_df = self.ml_results.get("gene_scores_df", pd.DataFrame())
        if isinstance(gene_scores_df, pd.DataFrame) and not gene_scores_df.empty:
            gene_scores_df.to_csv(out_dir / TABLE_FILE_PREFIXES["gene_scores"], index=False)

        screening_summary_df = self.ml_results.get("screening_summary_df", pd.DataFrame())
        if isinstance(screening_summary_df, pd.DataFrame) and not screening_summary_df.empty:
            screening_summary_df.to_csv(out_dir / TABLE_FILE_PREFIXES["screening_summary"], index=False)

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

        gene_module_assignment_df = self.ml_results.get("gene_module_assignment_df", pd.DataFrame())
        if isinstance(gene_module_assignment_df, pd.DataFrame) and not gene_module_assignment_df.empty:
            gene_module_assignment_df.to_csv(out_dir / TABLE_FILE_PREFIXES["gene_module_assignment"], index=False)

        module_eigengenes_df = self.ml_results.get("module_eigengenes_df", pd.DataFrame())
        if isinstance(module_eigengenes_df, pd.DataFrame) and not module_eigengenes_df.empty:
            module_eigengenes_df.to_csv(out_dir / TABLE_FILE_PREFIXES["module_eigengenes"], index=True)

        module_metabolite_assoc_df = self.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
        if isinstance(module_metabolite_assoc_df, pd.DataFrame) and not module_metabolite_assoc_df.empty:
            module_metabolite_assoc_df.to_csv(out_dir / TABLE_FILE_PREFIXES["module_metabolite_association"], index=False)

        module_summary_df = self.ml_results.get("module_summary_df", pd.DataFrame())
        if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty:
            module_summary_df.to_csv(out_dir / TABLE_FILE_PREFIXES["module_summary"], index=False)

        if self.config.export_cytoscape:
            export_frames = [
                df
                for df in (
                    total_network_df if isinstance(total_network_df, pd.DataFrame) else pd.DataFrame(),
                    high_confidence_network_df if isinstance(high_confidence_network_df, pd.DataFrame) else pd.DataFrame(),
                )
                if not df.empty
            ]
            if export_frames:
                cytoscape_df = pd.concat(export_frames, ignore_index=True).rename(
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

        gene_module_assignment_df = self.ml_results.get("gene_module_assignment_df", pd.DataFrame())
        module_summary_df = self.ml_results.get("module_summary_df", pd.DataFrame())
        self.run_metadata["n_module_assignment_rows"] = (
            int(len(gene_module_assignment_df)) if isinstance(gene_module_assignment_df, pd.DataFrame) else 0
        )
        self.run_metadata["n_non_grey_modules"] = (
            int(module_summary_df["Module"].nunique()) if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty else 0
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
