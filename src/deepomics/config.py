from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

from .utils import safe_mkdir


_VALID_REPORT_FORMATS = {"md", "html"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class AnalysisConfig:
    """Configuration container for the DeepOmics workflow."""

    project_name: str = "DeepOmics_Association_Analysis"
    output_dir: str = "results"
    group_table_path: str | None = None
    random_state: int = 42

    missing_feature_threshold: float = 0.5
    knn_neighbors: int = 5

    screen_top_k_per_method: int = 1000
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

    selection_ratio: float = 0.20
    min_features: int = 10
    max_features: int = 50

    network_plot_top_edges: int = 120
    top_pairs_plot_n: int = 6
    support_plot_top_metabolites: int = 20
    top_key_genes_plot_n: int = 20

    enable_module_detection: bool = True
    module_corr_method: str = "spearman"
    module_graph_k: int = 10
    module_min_edge_weight: float = 0.15
    module_method: str = "leiden"
    module_resolution: float = 1.0
    module_min_size: int = 5

    n_threads: int = -1
    cv_folds: int = 5
    generate_reports: bool = True
    report_formats: Tuple[str, ...] = field(default_factory=lambda: ("html",))
    export_pdf: bool = True
    export_svg: bool = True
    export_png: bool = True
    export_cytoscape: bool = True
    save_h5ad: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_name = str(self.project_name).strip()
        self.output_dir = str(self.output_dir)
        self.group_table_path = (
            None if self.group_table_path is None else str(self.group_table_path).strip() or None
        )
        self.log_level = str(self.log_level).upper().strip()
        self.module_corr_method = str(self.module_corr_method).lower().strip()
        self.module_method = str(self.module_method).lower().strip()
        self.report_formats = self._normalize_report_formats(self.report_formats)

        if not self.project_name:
            raise ValueError("project_name must not be empty.")
        if self.group_table_path is None:
            raise ValueError(
                "group_table_path is required and must point to a group table with "
                "sample_id, group1, and group2 columns."
            )
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of: {sorted(_VALID_LOG_LEVELS)}.")
        if self.screen_top_k_per_method < 1:
            raise ValueError("screen_top_k_per_method must be at least 1.")
        if not (0 <= self.missing_feature_threshold < 1):
            raise ValueError("missing_feature_threshold must be within [0, 1).")
        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least 1.")
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
        if self.network_plot_top_edges < 1:
            raise ValueError("network_plot_top_edges must be at least 1.")
        if self.top_pairs_plot_n < 1:
            raise ValueError("top_pairs_plot_n must be at least 1.")
        if self.support_plot_top_metabolites < 1:
            raise ValueError("support_plot_top_metabolites must be at least 1.")
        if self.top_key_genes_plot_n < 1:
            raise ValueError("top_key_genes_plot_n must be at least 1.")

        if str(self.module_corr_method).lower().strip() not in {"spearman"}:
            raise ValueError("module_corr_method currently only supports 'spearman'.")
        if str(self.module_method).lower().strip() not in {"leiden", "hierarchical"}:
            raise ValueError("module_method must be either 'leiden' or 'hierarchical'.")
        if self.module_graph_k < 1:
            raise ValueError("module_graph_k must be at least 1.")
        if not (0 <= self.module_min_edge_weight < 1):
            raise ValueError("module_min_edge_weight must be within [0, 1).")
        if self.module_resolution <= 0:
            raise ValueError("module_resolution must be positive.")
        if self.module_min_size < 1:
            raise ValueError("module_min_size must be at least 1.")
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
        if n_features <= 0:
            return 0
        target_k = max(int(self.min_features), int(n_samples * self.selection_ratio))
        return min(target_k, int(self.max_features), int(n_features))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
