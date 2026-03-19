from __future__ import annotations

from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from . import config, plotting, selectors
from .utils import get_logger, log_step, safe_mkdir, write_json

logger = get_logger()


class MultiOmicsEngine:
    """Core analysis engine for transcriptome-metabolome integration."""

    def __init__(self, adata: ad.AnnData, cfg: config.AnalysisConfig):
        """Initialize the engine."""
        self.adata = adata
        self.config = cfg
        self._validate_adata()

        self.ml_results: Dict[str, object] = {
            "grn_edges": [],
            "grn_edges_df": pd.DataFrame(),
            "key_genes_intersection": pd.DataFrame(),
            "key_genes_borda": pd.DataFrame(),
            "key_genes_rra": pd.DataFrame(),
            "metabolite_summary": pd.DataFrame(),
        }
        self.wgcna_results: Dict[str, object] = {}
        self.run_metadata: Dict[str, object] = {
            "project_name": self.config.project_name,
            "n_samples": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "n_metabolites": int(len(self.adata.uns.get("metabolite_names", []))),
        }

    def _validate_adata(self) -> None:
        """Validate required AnnData content before analysis."""
        if self.adata.n_obs < 3:
            raise ValueError("At least 3 samples are required to run DeepOmics.")
        if self.adata.n_vars < 2:
            raise ValueError("At least 2 genes are required to run DeepOmics.")
        if "metabolomics" not in self.adata.obsm and "metabolomics_scaled" not in self.adata.obsm:
            raise KeyError("AnnData must contain metabolomics data in obsm['metabolomics'].")

        metab_names = self.adata.uns.get("metabolite_names", [])
        if len(metab_names) == 0:
            raise ValueError("adata.uns['metabolite_names'] is empty.")

    def run_all(self, generate_plots: bool = True) -> None:
        """Execute the full workflow."""
        logger.info("=" * 80)
        logger.info("Project [%s] started", self.config.project_name)
        logger.info("=" * 80)

        with log_step(logger, "Ensemble machine learning"):
            self._run_ml_ensemble()

        with log_step(logger, "WGCNA analysis"):
            self._run_wgcna_pipeline()

        with log_step(logger, "Saving outputs"):
            self.save_results()

        if generate_plots:
            with log_step(logger, "Figure and report generation"):
                plotting.generate_report_plots(self, self.config)

        logger.info("All analyses finished. Results saved to: %s", self.config.output_dir)

    def _run_ml_ensemble(self) -> None:
        """Run per-metabolite ensemble feature selection and GRN construction."""
        metabolites = list(self.adata.uns["metabolite_names"])
        metab_df = self.adata.obsm.get("metabolomics_scaled", self.adata.obsm["metabolomics"])
        if not isinstance(metab_df, pd.DataFrame):
            metab_df = pd.DataFrame(metab_df, index=self.adata.obs_names, columns=metabolites)

        gene_names = np.asarray(self.adata.var_names, dtype=str)
        X_gene = np.asarray(self.adata.X, dtype=np.float32)
        edge_records: List[Dict[str, object]] = []
        all_key_gene_rows: List[Dict[str, object]] = []
        metabolite_rows: List[Dict[str, object]] = []

        logger.info("Processing %d metabolites with ensemble models.", len(metabolites))

        for metab_name in tqdm(metabolites, desc="Ensemble Learning"):
            y = metab_df[metab_name].to_numpy(dtype=np.float32, copy=False)

            candidate_genes, pcc_stats = selectors.filter_by_pcc(
                X_gene,
                y,
                self.config,
                feature_names=gene_names,
                return_stats=True,
            )

            summary_row = {
                "Metabolite": metab_name,
                "Candidate_Genes_PCC": int(len(candidate_genes)),
                "Intersection_Genes": 0,
                "Borda_Genes": 0,
                "RRA_Genes": 0,
            }

            if len(candidate_genes) == 0:
                metabolite_rows.append(summary_row)
                continue

            candidate_idx = np.isin(gene_names, candidate_genes)
            X_sub = X_gene[:, candidate_idx]
            feature_names = gene_names[candidate_idx]

            result_dict, score_table = selectors.get_integrated_key_genes(
                X_sub,
                y,
                self.config,
                feature_names=feature_names,
            )
            _ = score_table  # reserved for future per-metabolite export

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
                    support[f"In_{strategy.capitalize() if strategy != 'rra' else 'RRA'}"] = 1
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
                support_count = int(
                    support["In_Intersection"] + support["In_Borda"] + support["In_RRA"]
                )
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
        self.ml_results["grn_edges_df"] = pd.DataFrame(edge_records)
        full_report = pd.DataFrame(all_key_gene_rows)

        for strategy in ("intersection", "borda", "rra"):
            self.ml_results[f"key_genes_{strategy}"] = self._build_key_gene_table(full_report, strategy)

        self.ml_results["metabolite_summary"] = pd.DataFrame(metabolite_rows)

    @staticmethod
    def _build_key_gene_table(full_report: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Summarize key genes for a given strategy."""
        base_columns = [
            "Gene",
            "Associated_Metabolites_Count",
            "Associated_Metabolites",
            "Median_Rank",
            "Best_Rank",
        ]
        if full_report.empty:
            return pd.DataFrame(columns=base_columns)

        subset = full_report.loc[full_report["Strategy"] == strategy].copy()
        if subset.empty:
            return pd.DataFrame(columns=base_columns)

        summary = (
            subset.groupby("Gene")
            .agg(
                Associated_Metabolites_Count=("Metabolite", "nunique"),
                Associated_Metabolites=("Metabolite", lambda x: "|".join(sorted(set(x)))),
                Median_Rank=("Rank", "median"),
                Best_Rank=("Rank", "min"),
            )
            .sort_values(["Associated_Metabolites_Count", "Best_Rank"], ascending=[False, True])
            .reset_index()
        )
        return summary

    def _run_wgcna_pipeline(self) -> None:
        """Run an optimized WGCNA workflow for publication-oriented analyses."""
        X = np.asarray(self.adata.X, dtype=np.float32)
        n_samples, n_genes = X.shape
        actual_top_n = min(self.config.wgcna_top_n_genes, n_genes)

        if actual_top_n < 2:
            logger.warning("Skipping WGCNA because fewer than 2 genes are available.")
            return

        variances = np.var(X, axis=0)
        if actual_top_n == n_genes:
            top_indices = np.arange(n_genes)
        else:
            top_indices = np.argpartition(variances, -actual_top_n)[-actual_top_n:]
            top_indices = top_indices[np.argsort(variances[top_indices])[::-1]]

        X_wgcna = X[:, top_indices].astype(np.float32, copy=False)
        wgcna_gene_names = np.asarray(self.adata.var_names[top_indices], dtype=str)

        cor_mat = self._compute_correlation_matrix(X_wgcna)
        power_table, selected_power = self._scan_soft_threshold(cor_mat)
        adjacency = self._build_adjacency(cor_mat, selected_power)

        use_tom = bool(self.config.wgcna_use_tom and actual_top_n <= self.config.wgcna_tom_max_genes)
        if self.config.wgcna_use_tom and not use_tom:
            logger.warning(
                "Skipping TOM because %d genes exceed wgcna_tom_max_genes=%d.",
                actual_top_n,
                self.config.wgcna_tom_max_genes,
            )

        if use_tom:
            logger.info("Using TOM similarity for module detection.")
            similarity = self._compute_tom_similarity(adjacency)
        else:
            logger.info("Using adjacency similarity for module detection.")
            similarity = adjacency.copy()

        dissimilarity = 1.0 - similarity
        dissimilarity = np.clip((dissimilarity + dissimilarity.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(dissimilarity, 0.0)

        linkage_mat = linkage(squareform(dissimilarity, checks=False), method="average")
        tree_cut_height = self.config.resolved_wgcna_tree_cut_height(use_tom=use_tom)
        raw_labels = fcluster(linkage_mat, t=tree_cut_height, criterion="distance")

        module_assignments = pd.Series(raw_labels, index=wgcna_gene_names, dtype=int)
        module_counts = module_assignments.value_counts()
        valid_modules = module_counts[module_counts >= self.config.wgcna_min_module_size].index.tolist()

        if not valid_modules:
            logger.warning("No WGCNA modules passed the minimum module size threshold.")
            self.wgcna_results = {}
            return

        module_assignments = module_assignments[module_assignments.isin(valid_modules)]
        gene_module_df, me_df = self._materialize_modules(
            X_wgcna=X_wgcna,
            gene_names=wgcna_gene_names,
            module_assignments=module_assignments,
        )

        gene_module_df, me_df, merge_map = self._merge_modules_by_eigengene(
            X_wgcna=X_wgcna,
            gene_names=wgcna_gene_names,
            gene_module_df=gene_module_df,
        )

        if gene_module_df.empty or me_df.empty:
            logger.warning("No WGCNA modules remained after module merging.")
            self.wgcna_results = {}
            return

        metab_df = self.adata.obsm.get("metabolomics_scaled", self.adata.obsm["metabolomics"])
        if not isinstance(metab_df, pd.DataFrame):
            metab_df = pd.DataFrame(
                metab_df,
                index=self.adata.obs_names,
                columns=self.adata.uns["metabolite_names"],
            )

        module_trait_corr, module_trait_p = self._correlate_modules_to_traits(me_df, metab_df)
        module_trait_fdr = self._adjust_pvalue_df(module_trait_p)

        ordered_gene_modules = self._build_ordered_gene_module_table(
            gene_names=wgcna_gene_names,
            linkage_matrix=linkage_mat,
            gene_module_df=gene_module_df,
        )

        module_summary = (
            gene_module_df.groupby("Module")
            .size()
            .reset_index(name="Gene_Count")
            .sort_values(["Gene_Count", "Module"], ascending=[False, True])
            .reset_index(drop=True)
        )

        gene_stats_df, hub_genes_df = self._compute_wgcna_gene_statistics(
            X_wgcna=X_wgcna,
            gene_names=wgcna_gene_names,
            adjacency=adjacency,
            gene_module_df=gene_module_df,
            me_df=me_df,
            metab_df=metab_df,
            module_trait_corr=module_trait_corr,
        )

        selected_power_row = power_table.loc[power_table["Chosen"]].head(1)
        self.wgcna_results = {
            "ME_df": me_df,
            "Gene_Modules": gene_module_df,
            "Trait_Correlation": module_trait_corr,
            "Trait_PValue": module_trait_p,
            "Trait_FDR": module_trait_fdr,
            "Module_Summary": module_summary,
            "Power_Selection": power_table,
            "Selected_Power": int(selected_power),
            "Selected_Power_Summary": selected_power_row.reset_index(drop=True),
            "Adjacency_Used_TOM": bool(use_tom),
            "Merge_Map": merge_map,
            "Ordered_Gene_Modules": ordered_gene_modules,
            "Linkage_Matrix": linkage_mat,
            "Gene_Statistics": gene_stats_df,
            "Hub_Genes": hub_genes_df,
        }

        self.run_metadata["wgcna_selected_power"] = int(selected_power)
        self.run_metadata["wgcna_used_tom"] = bool(use_tom)
        self.run_metadata["wgcna_network_type"] = self.config.wgcna_network_type
        self.run_metadata["wgcna_tree_cut_height"] = float(tree_cut_height)

        logger.info(
            "WGCNA completed. Selected power=%d, TOM=%s, detected %d modules.",
            selected_power,
            "yes" if use_tom else "no",
            len(module_summary),
        )

    @staticmethod
    def _compute_correlation_matrix(X_wgcna: np.ndarray) -> np.ndarray:
        """Compute a symmetric gene-gene Pearson correlation matrix."""
        X_centered = X_wgcna - X_wgcna.mean(axis=0, keepdims=True)
        X_std = X_centered.std(axis=0, ddof=1, keepdims=True)
        X_std[X_std == 0] = 1.0
        X_z = X_centered / X_std

        cor_mat = (X_z.T @ X_z) / max(1, X_z.shape[0] - 1)
        cor_mat = np.clip(cor_mat, -1.0, 1.0)
        cor_mat = ((cor_mat + cor_mat.T) / 2.0).astype(np.float32)
        np.fill_diagonal(cor_mat, 1.0)
        return cor_mat

    def _scan_soft_threshold(self, cor_mat: np.ndarray) -> Tuple[pd.DataFrame, int]:
        """Evaluate multiple powers and select a soft-threshold automatically."""
        candidates = sorted(set(int(p) for p in self.config.wgcna_power_candidates if int(p) > 0))
        if self.config.wgcna_soft_power is not None and int(self.config.wgcna_soft_power) not in candidates:
            candidates.append(int(self.config.wgcna_soft_power))
            candidates = sorted(set(candidates))

        rows: List[Dict[str, object]] = []
        for power in candidates:
            adjacency = self._build_adjacency(cor_mat, power=power)
            degree = adjacency.sum(axis=1)
            scale_free_r2, slope = self._scale_free_fit_index(degree)
            rows.append(
                {
                    "Power": int(power),
                    "ScaleFreeR2": float(scale_free_r2),
                    "Slope": float(slope) if np.isfinite(slope) else np.nan,
                    "MeanConnectivity": float(np.mean(degree)),
                    "MedianConnectivity": float(np.median(degree)),
                }
            )

        power_df = pd.DataFrame(rows)
        if self.config.wgcna_soft_power is not None:
            selected_power = int(self.config.wgcna_soft_power)
        else:
            candidates_df = power_df.loc[
                (power_df["ScaleFreeR2"] >= self.config.wgcna_scale_free_r2_threshold)
                & (power_df["Slope"] < 0)
            ]
            if not candidates_df.empty:
                selected_power = int(candidates_df.sort_values("Power").iloc[0]["Power"])
            else:
                selected_power = int(
                    power_df.sort_values(
                        ["ScaleFreeR2", "MeanConnectivity", "Power"],
                        ascending=[False, False, True],
                    ).iloc[0]["Power"]
                )
        power_df["Chosen"] = power_df["Power"] == selected_power
        return power_df, selected_power

    def _build_adjacency(self, cor_mat: np.ndarray, power: int) -> np.ndarray:
        """Transform correlation into adjacency according to the chosen network type."""
        if self.config.wgcna_network_type == "signed":
            adjacency = np.power((1.0 + cor_mat) / 2.0, power, dtype=np.float32)
        else:
            adjacency = np.power(np.abs(cor_mat), power, dtype=np.float32)

        adjacency = np.asarray(adjacency, dtype=np.float32)
        adjacency = ((adjacency + adjacency.T) / 2.0).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    @staticmethod
    def _scale_free_fit_index(degree: np.ndarray) -> Tuple[float, float]:
        """Approximate scale-free topology fit (R²) from node connectivity."""
        k = np.asarray(degree, dtype=float)
        k = k[np.isfinite(k) & (k > 0)]
        if k.size < 10 or np.allclose(k, k[0]):
            return 0.0, np.nan

        n_bins = min(20, max(8, int(np.sqrt(k.size))))
        hist, edges = np.histogram(k, bins=n_bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        mask = (hist > 0) & (mids > 0)
        if mask.sum() < 3:
            return 0.0, np.nan

        x = np.log10(mids[mask])
        y = np.log10(hist[mask] / hist.sum())

        try:
            slope, intercept = np.polyfit(x, y, deg=1)
        except np.linalg.LinAlgError:
            return 0.0, np.nan

        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
        return float(r2), float(slope)

    @staticmethod
    def _compute_tom_similarity(adjacency: np.ndarray) -> np.ndarray:
        """Compute topological overlap similarity."""
        adj = np.asarray(adjacency, dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        shared_neighbors = adj @ adj
        connectivity = adj.sum(axis=1)
        denominator = np.minimum.outer(connectivity, connectivity) + 1.0 - adj
        denominator[denominator == 0] = 1.0
        tom = (shared_neighbors + adj) / denominator
        tom = np.clip((tom + tom.T) / 2.0, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(tom, 1.0)
        return tom

    def _materialize_modules(
        self,
        X_wgcna: np.ndarray,
        gene_names: np.ndarray,
        module_assignments: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create module membership and eigengene tables from assignments."""
        counts = module_assignments.value_counts().sort_values(ascending=False)
        me_dict: Dict[str, np.ndarray] = {}
        module_rows: List[Dict[str, str]] = []

        for module_idx, raw_label in enumerate(counts.index.tolist(), start=1):
            genes_in_module = module_assignments.index[module_assignments == raw_label].tolist()
            module_name = f"Module_{module_idx:02d}"
            idx = np.isin(gene_names, genes_in_module)
            module_expr = X_wgcna[:, idx]
            eigengene = self._compute_module_eigengene(module_expr)
            me_dict[module_name] = eigengene
            for gene in genes_in_module:
                module_rows.append({"Gene": gene, "Module": module_name})

        gene_module_df = pd.DataFrame(module_rows)
        me_df = pd.DataFrame(me_dict, index=self.adata.obs_names)
        return gene_module_df, me_df

    def _merge_modules_by_eigengene(
        self,
        X_wgcna: np.ndarray,
        gene_names: np.ndarray,
        gene_module_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Merge modules with highly correlated eigengenes."""
        if gene_module_df.empty:
            return gene_module_df, pd.DataFrame(index=self.adata.obs_names), pd.DataFrame()

        module_to_genes = {
            module: gene_module_df.loc[gene_module_df["Module"] == module, "Gene"].astype(str).tolist()
            for module in sorted(gene_module_df["Module"].astype(str).unique())
        }
        me_dict = {}
        for module, genes in module_to_genes.items():
            idx = np.isin(gene_names, genes)
            me_dict[module] = self._compute_module_eigengene(X_wgcna[:, idx])

        me_df = pd.DataFrame(me_dict, index=self.adata.obs_names)
        if me_df.shape[1] <= 1:
            merge_map = pd.DataFrame(
                {"Original_Module": list(module_to_genes.keys()), "Merged_Module": list(module_to_genes.keys())}
            )
            return gene_module_df.copy(), me_df, merge_map

        me_corr = me_df.corr().clip(-1.0, 1.0)
        me_dissim = 1.0 - me_corr
        me_dissim = np.clip((me_dissim + me_dissim.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(me_dissim.values, 0.0)

        module_linkage = linkage(squareform(me_dissim.to_numpy(dtype=float), checks=False), method="average")
        merged_labels = fcluster(module_linkage, t=self.config.wgcna_merge_cut_height, criterion="distance")

        merged_series = pd.Series(merged_labels, index=me_df.columns)
        merged_groups = (
            merged_series.groupby(merged_series)
            .apply(lambda x: x.index.tolist())
            .sort_values(key=lambda ser: ser.map(len), ascending=False)
        )

        module_rows: List[Dict[str, str]] = []
        merged_me: Dict[str, np.ndarray] = {}
        merge_map_rows: List[Dict[str, str]] = []

        for merged_idx, old_modules in enumerate(merged_groups.tolist(), start=1):
            merged_name = f"Module_{merged_idx:02d}"
            genes = sorted({gene for module in old_modules for gene in module_to_genes[module]})
            idx = np.isin(gene_names, genes)
            merged_me[merged_name] = self._compute_module_eigengene(X_wgcna[:, idx])
            for gene in genes:
                module_rows.append({"Gene": gene, "Module": merged_name})
            for old_module in old_modules:
                merge_map_rows.append({"Original_Module": old_module, "Merged_Module": merged_name})

        merged_gene_df = pd.DataFrame(module_rows).sort_values(["Module", "Gene"]).reset_index(drop=True)
        merged_me_df = pd.DataFrame(merged_me, index=self.adata.obs_names)
        merge_map = pd.DataFrame(merge_map_rows).sort_values(["Merged_Module", "Original_Module"]).reset_index(drop=True)
        return merged_gene_df, merged_me_df, merge_map

    @staticmethod
    def _compute_module_eigengene(module_expr: np.ndarray) -> np.ndarray:
        """Approximate module eigengene using the first singular vector."""
        centered = module_expr - module_expr.mean(axis=0, keepdims=True)
        if centered.shape[1] == 1:
            eigengene = centered[:, 0]
        else:
            try:
                u, s, _ = np.linalg.svd(centered, full_matrices=False)
                eigengene = u[:, 0] * s[0]
            except np.linalg.LinAlgError:
                eigengene = centered.mean(axis=1)

        module_mean = centered.mean(axis=1)
        if np.std(module_mean, ddof=1) > 0 and np.std(eigengene, ddof=1) > 0:
            corr = np.corrcoef(eigengene, module_mean)[0, 1]
            if np.isfinite(corr) and corr < 0:
                eigengene = -eigengene

        eigengene = eigengene.astype(np.float32, copy=False)
        std = eigengene.std(ddof=1)
        if std == 0:
            return eigengene - eigengene.mean()
        return (eigengene - eigengene.mean()) / std

    @staticmethod
    def _correlate_modules_to_traits(
        me_df: pd.DataFrame,
        metab_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute module-trait correlation and p-value matrices."""
        me_values = me_df.to_numpy(dtype=np.float32, copy=False)
        y_values = metab_df.to_numpy(dtype=np.float32, copy=False)

        me_centered = me_values - me_values.mean(axis=0, keepdims=True)
        y_centered = y_values - y_values.mean(axis=0, keepdims=True)

        me_std = me_centered.std(axis=0, ddof=1, keepdims=True)
        y_std = y_centered.std(axis=0, ddof=1, keepdims=True)
        me_std[me_std == 0] = 1.0
        y_std[y_std == 0] = 1.0

        me_z = me_centered / me_std
        y_z = y_centered / y_std

        corr = (y_z.T @ me_z) / max(1, me_z.shape[0] - 1)
        corr = np.clip(corr, -1.0, 1.0)

        t_stat = corr * np.sqrt((me_z.shape[0] - 2) / np.maximum(1.0 - corr**2, 1e-12))
        p_values = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=me_z.shape[0] - 2))

        corr_df = pd.DataFrame(corr, index=metab_df.columns, columns=me_df.columns)
        pval_df = pd.DataFrame(p_values, index=metab_df.columns, columns=me_df.columns)
        return corr_df, pval_df

    @staticmethod
    def _adjust_pvalue_df(pval_df: pd.DataFrame) -> pd.DataFrame:
        """Apply Benjamini-Hochberg correction to a p-value matrix."""
        if pval_df.empty:
            return pval_df.copy()
        flat = pval_df.to_numpy(dtype=float).ravel()
        _, qvals, _, _ = multipletests(flat, method="fdr_bh")
        return pd.DataFrame(qvals.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns)

    @staticmethod
    def _corr_and_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute Pearson r and corresponding p-value."""
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 3:
            return np.nan, np.nan
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        x_std = np.std(x_centered, ddof=1)
        y_std = np.std(y_centered, ddof=1)
        if x_std == 0 or y_std == 0:
            return np.nan, np.nan
        r = float(np.sum(x_centered * y_centered) / ((x.size - 1) * x_std * y_std))
        r = float(np.clip(r, -1.0, 1.0))
        t_stat = r * np.sqrt((x.size - 2) / max(1e-12, 1.0 - r**2))
        p_value = float(2.0 * (1.0 - t.cdf(abs(t_stat), df=x.size - 2)))
        return r, p_value

    def _compute_wgcna_gene_statistics(
        self,
        X_wgcna: np.ndarray,
        gene_names: np.ndarray,
        adjacency: np.ndarray,
        gene_module_df: pd.DataFrame,
        me_df: pd.DataFrame,
        metab_df: pd.DataFrame,
        module_trait_corr: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute module membership, trait association, and hub-gene summaries."""
        if gene_module_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        module_to_top_trait = (
            module_trait_corr.abs().idxmax(axis=0).to_dict() if not module_trait_corr.empty else {}
        )
        gene_pos = {gene: idx for idx, gene in enumerate(gene_names)}
        rows: List[Dict[str, object]] = []

        for module in sorted(gene_module_df["Module"].unique()):
            genes = gene_module_df.loc[gene_module_df["Module"] == module, "Gene"].astype(str).tolist()
            positions = np.array([gene_pos[g] for g in genes], dtype=int)
            if positions.size == 0:
                continue

            eigengene = me_df[module].to_numpy(dtype=float, copy=False)
            top_trait = module_to_top_trait.get(module)
            top_trait_values = (
                metab_df[top_trait].to_numpy(dtype=float, copy=False) if top_trait in metab_df.columns else None
            )
            intra_connectivity = adjacency[np.ix_(positions, positions)].sum(axis=1)

            for gene, pos, k_within in zip(genes, positions, intra_connectivity):
                expr = X_wgcna[:, pos]
                kme, kme_p = self._corr_and_pvalue(expr, eigengene)

                if top_trait_values is not None:
                    gs_corr, gs_p = self._corr_and_pvalue(expr, top_trait_values)
                else:
                    gs_corr, gs_p = np.nan, np.nan

                rows.append(
                    {
                        "Gene": gene,
                        "Module": module,
                        "ModuleMembership": kme,
                        "ModuleMembership_PValue": kme_p,
                        "AbsModuleMembership": abs(kme) if np.isfinite(kme) else np.nan,
                        "IntramodularConnectivity": float(k_within),
                        "Top_Metabolite": top_trait,
                        "GeneTraitCorrelation": gs_corr,
                        "GeneTraitPValue": gs_p,
                        "GeneSignificanceAbs": abs(gs_corr) if np.isfinite(gs_corr) else np.nan,
                    }
                )

        gene_stats_df = pd.DataFrame(rows)
        if gene_stats_df.empty:
            return gene_stats_df, pd.DataFrame()

        gene_stats_df["Rank_ModuleMembership"] = gene_stats_df.groupby("Module")["AbsModuleMembership"].rank(
            ascending=False, method="min"
        )
        gene_stats_df["Rank_IntramodularConnectivity"] = gene_stats_df.groupby("Module")[
            "IntramodularConnectivity"
        ].rank(ascending=False, method="min")
        gene_stats_df["Rank_GeneSignificance"] = gene_stats_df.groupby("Module")["GeneSignificanceAbs"].rank(
            ascending=False, method="min"
        )
        gene_stats_df["HubScore"] = (
            gene_stats_df["Rank_ModuleMembership"]
            + gene_stats_df["Rank_IntramodularConnectivity"]
            + gene_stats_df["Rank_GeneSignificance"]
        )

        hub_genes_df = (
            gene_stats_df.sort_values(["Module", "HubScore", "AbsModuleMembership"], ascending=[True, True, False])
            .groupby("Module", group_keys=False)
            .head(self.config.wgcna_hub_genes_per_module)
            .reset_index(drop=True)
        )
        gene_stats_df = gene_stats_df.sort_values(
            ["Module", "HubScore", "AbsModuleMembership"], ascending=[True, True, False]
        ).reset_index(drop=True)
        return gene_stats_df, hub_genes_df

    @staticmethod
    def _build_ordered_gene_module_table(
        gene_names: np.ndarray,
        linkage_matrix: np.ndarray,
        gene_module_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create a leaf-ordered gene-module table for dendrogram plotting."""
        leaf_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
        ordered_genes = gene_names[leaf_order]
        module_map = gene_module_df.set_index("Gene")["Module"].to_dict()
        rows = [
            {
                "Gene": gene,
                "Module": module_map.get(gene, "Unassigned"),
                "Dendrogram_Order": order_idx,
            }
            for order_idx, gene in enumerate(ordered_genes, start=1)
        ]
        return pd.DataFrame(rows)

    def save_results(self) -> None:
        """Export tables, Cytoscape edges, metadata, and optional AnnData state."""
        out_dir = safe_mkdir(self.config.output_dir)

        grn_edges_df = self.ml_results.get("grn_edges_df", pd.DataFrame())
        if isinstance(grn_edges_df, pd.DataFrame) and not grn_edges_df.empty:
            grn_edges_df.assign(Abs_PCC_R=grn_edges_df["PCC_R"].abs()).sort_values(
                ["Support_Count", "In_RRA", "Abs_PCC_R"],
                ascending=[False, False, False],
            ).drop(columns=["Abs_PCC_R"]).to_csv(out_dir / "GRN_Edges_Full.csv", index=False)

            grn_edges_df.loc[
                :,
                ["Source", "Target", "Interaction", "Support_Count", "PCC_R", "PCC_P"],
            ].rename(
                columns={"Source": "source", "Target": "target", "Interaction": "interaction"}
            ).to_csv(out_dir / "GRN_Edges_Cytoscape.csv", index=False)

        for strategy in ("intersection", "borda", "rra"):
            df = self.ml_results.get(f"key_genes_{strategy}", pd.DataFrame())
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(out_dir / f"Key_Genes_{strategy.capitalize()}.csv", index=False)

        metabolite_summary = self.ml_results.get("metabolite_summary", pd.DataFrame())
        if isinstance(metabolite_summary, pd.DataFrame) and not metabolite_summary.empty:
            metabolite_summary.to_csv(out_dir / "ML_Metabolite_Summary.csv", index=False)

        if self.wgcna_results:
            self.wgcna_results["Gene_Modules"].to_csv(out_dir / "WGCNA_Gene_Modules.csv", index=False)
            self.wgcna_results["ME_df"].to_csv(out_dir / "WGCNA_Module_Eigengenes.csv")
            self.wgcna_results["Trait_Correlation"].to_csv(out_dir / "WGCNA_Module_Trait_Correlation.csv")
            self.wgcna_results["Trait_PValue"].to_csv(out_dir / "WGCNA_Module_Trait_PValue.csv")
            self.wgcna_results["Trait_FDR"].to_csv(out_dir / "WGCNA_Module_Trait_FDR.csv")
            self.wgcna_results["Module_Summary"].to_csv(out_dir / "WGCNA_Module_Summary.csv", index=False)
            self.wgcna_results["Power_Selection"].to_csv(out_dir / "WGCNA_Power_Selection.csv", index=False)
            self.wgcna_results["Gene_Statistics"].to_csv(out_dir / "WGCNA_Gene_Statistics.csv", index=False)
            self.wgcna_results["Hub_Genes"].to_csv(out_dir / "WGCNA_Hub_Genes.csv", index=False)
            self.wgcna_results["Merge_Map"].to_csv(out_dir / "WGCNA_Module_Merge_Map.csv", index=False)
            self.wgcna_results["Ordered_Gene_Modules"].to_csv(
                out_dir / "WGCNA_Ordered_Gene_Modules.csv",
                index=False,
            )

        self.run_metadata["n_grn_edges"] = int(len(grn_edges_df)) if isinstance(grn_edges_df, pd.DataFrame) else 0
        self.run_metadata["n_wgcna_modules"] = int(
            len(self.wgcna_results.get("Module_Summary", pd.DataFrame()))
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
