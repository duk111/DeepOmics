from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from .utils import safe_mkdir


@dataclass
class AnalysisConfig:
    """Configuration container for the DeepOmics workflow."""

    project_name: str = "DeepOmics_Analysis"
    output_dir: str = "results"
    random_state: int = 42

    pcc_r_threshold: float = 0.30
    pcc_p_threshold: float = 0.05
    use_fdr: bool = True
    fdr_alpha: float = 0.05

    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.10
    xgb_subsample: float = 0.80
    xgb_colsample_bytree: float = 0.80

    elastic_net_alpha_search: bool = True
    elastic_net_fixed_alpha: float = 0.01
    elastic_net_l1_ratio: float = 0.50
    elastic_net_l1_ratio_grid: Tuple[float, ...] = field(
        default_factory=lambda: (0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99)
    )
    elastic_net_max_iter: int = 20000

    # Backward-compatible aliases retained for existing user code/config files.
    lasso_alpha_search: Optional[bool] = None
    lasso_fixed_alpha: Optional[float] = None

    svm_kernel: str = "linear"

    selection_ratio: float = 0.20
    min_features: int = 10
    max_features: int = 50
    max_candidate_genes: Optional[int] = 2000

    enable_intersection: bool = True
    enable_borda: bool = True
    enable_rra: bool = True
    grn_primary_strategy: str = "rra"

    wgcna_top_n_genes: int = 3000
    wgcna_soft_power: Optional[int] = None
    wgcna_power_candidates: Tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 21)))
    wgcna_scale_free_r2_threshold: float = 0.80
    wgcna_network_type: str = "unsigned"
    wgcna_use_tom: bool = True
    wgcna_tom_max_genes: int = 2000
    wgcna_min_module_size: int = 30
    wgcna_tree_cut_height: Optional[float] = None
    wgcna_merge_cut_height: float = 0.25
    wgcna_hub_genes_per_module: int = 10

    correlation_circle_top_genes: int = 30
    correlation_circle_top_metabolites: int = 20
    circos_top_edges: int = 80
    complex_heatmap_top_genes: int = 30
    complex_heatmap_top_metabolites: int = 15

    verbose: bool = True
    n_threads: int = -1
    cv_folds: int = 5
    generate_reports: bool = True
    report_formats: Tuple[str, ...] = field(default_factory=lambda: ("md", "html"))
    export_pdf: bool = True
    export_svg: bool = True
    save_h5ad: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if self.lasso_alpha_search is not None:
            self.elastic_net_alpha_search = bool(self.lasso_alpha_search)
        if self.lasso_fixed_alpha is not None:
            self.elastic_net_fixed_alpha = float(self.lasso_fixed_alpha)

        if not self.project_name.strip():
            raise ValueError("project_name must not be empty.")
        if not (0 < self.pcc_r_threshold <= 1):
            raise ValueError("pcc_r_threshold must be within (0, 1].")
        if not (0 < self.pcc_p_threshold <= 1):
            raise ValueError("pcc_p_threshold must be within (0, 1].")
        if not (0 < self.fdr_alpha <= 1):
            raise ValueError("fdr_alpha must be within (0, 1].")
        if not (0 < self.selection_ratio <= 1):
            raise ValueError("selection_ratio must be within (0, 1].")
        if self.min_features <= 0 or self.max_features <= 0:
            raise ValueError("min_features and max_features must be positive.")
        if self.min_features > self.max_features:
            raise ValueError("min_features cannot be larger than max_features.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.wgcna_top_n_genes <= 1:
            raise ValueError("wgcna_top_n_genes must be greater than 1.")
        if self.wgcna_min_module_size < 2:
            raise ValueError("wgcna_min_module_size must be at least 2.")
        if self.wgcna_tom_max_genes < 2:
            raise ValueError("wgcna_tom_max_genes must be at least 2.")
        if not (0 < self.wgcna_scale_free_r2_threshold <= 1):
            raise ValueError("wgcna_scale_free_r2_threshold must be within (0, 1].")
        if self.wgcna_tree_cut_height is not None and not (0 < self.wgcna_tree_cut_height < 1):
            raise ValueError("wgcna_tree_cut_height must be within (0, 1) when provided.")
        if not (0 < self.wgcna_merge_cut_height < 1):
            raise ValueError("wgcna_merge_cut_height must be within (0, 1).")
        if self.wgcna_network_type not in {"unsigned", "signed"}:
            raise ValueError("wgcna_network_type must be either 'unsigned' or 'signed'.")
        if self.wgcna_hub_genes_per_module < 1:
            raise ValueError("wgcna_hub_genes_per_module must be at least 1.")
        if self.correlation_circle_top_genes < 1 or self.correlation_circle_top_metabolites < 1:
            raise ValueError("correlation circle feature counts must be at least 1.")
        if self.circos_top_edges < 1:
            raise ValueError("circos_top_edges must be at least 1.")
        if self.complex_heatmap_top_genes < 1 or self.complex_heatmap_top_metabolites < 1:
            raise ValueError("complex heatmap feature counts must be at least 1.")
        if self.grn_primary_strategy not in {"intersection", "borda", "rra"}:
            raise ValueError("grn_primary_strategy must be one of: intersection, borda, rra.")
        if not all(fmt in {"md", "html"} for fmt in self.report_formats):
            raise ValueError("report_formats only supports 'md' and 'html'.")
        if self.elastic_net_fixed_alpha <= 0:
            raise ValueError("elastic_net_fixed_alpha must be positive.")
        if not (0 < self.elastic_net_l1_ratio <= 1):
            raise ValueError("elastic_net_l1_ratio must be within (0, 1].")
        if self.elastic_net_max_iter < 1000:
            raise ValueError("elastic_net_max_iter must be at least 1000.")
        if not self.elastic_net_l1_ratio_grid:
            raise ValueError("elastic_net_l1_ratio_grid must not be empty.")

        cleaned_l1_ratio_grid = sorted({float(v) for v in self.elastic_net_l1_ratio_grid if 0 < float(v) <= 1})
        if not cleaned_l1_ratio_grid:
            raise ValueError("elastic_net_l1_ratio_grid values must be within (0, 1].")
        self.elastic_net_l1_ratio_grid = tuple(cleaned_l1_ratio_grid)

        safe_mkdir(self.output_dir)

    def resolved_threads(self) -> int:
        """Resolve thread count from the current environment."""
        if self.n_threads == -1:
            return max(1, os.cpu_count() or 1)
        return max(1, self.n_threads)

    def target_feature_count(self, n_samples: int, n_features: int) -> int:
        """Compute the dynamic target feature count."""
        target_k = max(self.min_features, int(n_samples * self.selection_ratio))
        return min(target_k, self.max_features, n_features)

    def resolved_wgcna_tree_cut_height(self, use_tom: bool) -> float:
        """Return the effective tree-cut height for module detection."""
        if self.wgcna_tree_cut_height is not None:
            return self.wgcna_tree_cut_height
        return 0.25 if use_tom else 0.90

    def to_dict(self) -> Dict[str, object]:
        """Convert configuration to a serializable dictionary."""
        return asdict(self)
