from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from .utils import safe_mkdir


_VALID_REPORT_FORMATS = {"md", "html"}
_VALID_PRIMARY_STRATEGIES = {"intersection", "borda", "rra"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


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

    correlation_circle_top_genes: int = 30
    correlation_circle_top_metabolites: int = 20
    circos_top_edges: int = 80
    complex_heatmap_top_genes: int = 30
    complex_heatmap_top_metabolites: int = 15

    verbose: bool = False
    n_threads: int = -1
    cv_folds: int = 5
    generate_reports: bool = True
    report_formats: Tuple[str, ...] = field(default_factory=lambda: ("html",))
    export_pdf: bool = True
    export_svg: bool = True
    export_png: bool = True
    export_cytoscape: bool = True
    verbose_outputs: bool = False
    save_h5ad: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.lasso_alpha_search is not None:
            self.elastic_net_alpha_search = bool(self.lasso_alpha_search)
        if self.lasso_fixed_alpha is not None:
            self.elastic_net_fixed_alpha = float(self.lasso_fixed_alpha)

        self.project_name = str(self.project_name).strip()
        self.output_dir = str(self.output_dir)
        self.log_level = str(self.log_level).upper().strip()
        self.svm_kernel = str(self.svm_kernel).lower().strip()
        self.grn_primary_strategy = str(self.grn_primary_strategy).lower().strip()
        self.report_formats = self._normalize_report_formats(self.report_formats)

        if not self.project_name:
            raise ValueError("project_name must not be empty.")
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of: {sorted(_VALID_LOG_LEVELS)}.")
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
        if self.max_candidate_genes is not None and self.max_candidate_genes <= 0:
            raise ValueError("max_candidate_genes must be positive when provided.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.correlation_circle_top_genes < 1 or self.correlation_circle_top_metabolites < 1:
            raise ValueError("correlation circle feature counts must be at least 1.")
        if self.circos_top_edges < 1:
            raise ValueError("circos_top_edges must be at least 1.")
        if self.complex_heatmap_top_genes < 1 or self.complex_heatmap_top_metabolites < 1:
            raise ValueError("complex heatmap feature counts must be at least 1.")
        if self.grn_primary_strategy not in _VALID_PRIMARY_STRATEGIES:
            raise ValueError("grn_primary_strategy must be one of: intersection, borda, rra.")
        if self.elastic_net_fixed_alpha <= 0:
            raise ValueError("elastic_net_fixed_alpha must be positive.")
        if not (0 < self.elastic_net_l1_ratio <= 1):
            raise ValueError("elastic_net_l1_ratio must be within (0, 1].")
        if self.elastic_net_max_iter < 1000:
            raise ValueError("elastic_net_max_iter must be at least 1000.")
        if not self.elastic_net_l1_ratio_grid:
            raise ValueError("elastic_net_l1_ratio_grid must not be empty.")
        if self.xgb_n_estimators < 1:
            raise ValueError("xgb_n_estimators must be at least 1.")
        if self.xgb_max_depth < 1:
            raise ValueError("xgb_max_depth must be at least 1.")
        if self.xgb_learning_rate <= 0:
            raise ValueError("xgb_learning_rate must be positive.")
        if not (0 < self.xgb_subsample <= 1):
            raise ValueError("xgb_subsample must be within (0, 1].")
        if not (0 < self.xgb_colsample_bytree <= 1):
            raise ValueError("xgb_colsample_bytree must be within (0, 1].")

        cleaned_l1_ratio_grid = sorted(
            {float(value) for value in self.elastic_net_l1_ratio_grid if 0 < float(value) <= 1}
        )
        if not cleaned_l1_ratio_grid:
            raise ValueError("elastic_net_l1_ratio_grid values must be within (0, 1].")
        self.elastic_net_l1_ratio_grid = tuple(cleaned_l1_ratio_grid)

        safe_mkdir(self.output_dir)

    @staticmethod
    def _normalize_report_formats(report_formats: Tuple[str, ...]) -> Tuple[str, ...]:
        normalized = []
        for fmt in report_formats:
            normalized_fmt = str(fmt).lower().strip()
            if not normalized_fmt:
                continue
            if normalized_fmt not in _VALID_REPORT_FORMATS:
                raise ValueError("report_formats only supports 'md' and 'html'.")
            if normalized_fmt not in normalized:
                normalized.append(normalized_fmt)
        return tuple(normalized) if normalized else ("html",)

    def resolved_threads(self) -> int:
        if self.n_threads == -1:
            return max(1, os.cpu_count() or 1)
        return max(1, int(self.n_threads))

    def target_feature_count(self, n_samples: int, n_features: int) -> int:
        target_k = max(int(self.min_features), int(n_samples * self.selection_ratio))
        return min(target_k, int(self.max_features), int(n_features))

    def diagnostics_enabled(self) -> bool:
        return bool(self.verbose_outputs or self.log_level == "DEBUG")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
