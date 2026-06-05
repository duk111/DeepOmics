from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..outputs import FIGURE_FILE_PREFIXES, TABLE_FILE_PREFIXES
from .context import VisualizationContext
from .static.association import (
    plot_association_direction_summary,
    plot_edgeweight_distribution_by_module,
    plot_gene_metabolite_correlation_bubble_heatmap,
    plot_module_eigengene_metabolite_trend_panels,
    plot_module_metabolite_bubble_plot,
)
from .static.correlation import plot_top_gene_metabolite_correlation_heatmaps
from .static.distribution import (
    plot_module_eigengene_group1_violin_box,
    plot_module_kme_boxplot,
    plot_top_metabolite_group1_violin_box,
)
from .static.module import (
    plot_module_eigengene_heatmap,
    plot_module_eigengene_heatmap_group2,
    plot_module_eigengene_ridge,
    plot_module_eigengene_ridge_group1,
    plot_module_gene_zscore_line_panels,
    plot_module_metabolite_association_heatmap,
    plot_module_zscore_line_panels,
)
from .static.network import plot_compressed_circos_network, plot_floating_cnet_circos_network
from .static.pca import (
    _has_secondary_grouping,
    plot_metabolome_pca,
    plot_metabolome_pca_pairs,
    plot_metabolome_pca_pairs_subgroups,
    plot_metabolome_pca_subgroups,
    plot_sample_dendrogram,
    plot_transcriptome_pca,
    plot_transcriptome_pca_pairs,
    plot_transcriptome_pca_pairs_subgroups,
    plot_transcriptome_pca_subgroups,
)
from .static.regression import plot_module_top_metabolite_regression_panels, plot_top_edge_scatter_panels
from .static.upset import plot_association_evidence_upset


FigurePlotter = Callable[[VisualizationContext, Path], None]
FigurePredicate = Callable[[VisualizationContext], bool]


def _always_enabled(_context: VisualizationContext) -> bool:
    return True


def _has_secondary_pca_grouping(context: VisualizationContext) -> bool:
    return _has_secondary_grouping(context.pca_group_df)


@dataclass(frozen=True)
class FigureSpec:
    key: str
    prefix_key: str
    plotter: FigurePlotter
    enabled: FigurePredicate = _always_enabled
    description: str = ""

    def save_stem(self, context: VisualizationContext) -> Path:
        return context.plots_dir / FIGURE_FILE_PREFIXES[self.prefix_key]

    def render(self, context: VisualizationContext) -> None:
        if self.enabled(context):
            self.plotter(context, self.save_stem(context))


def _render_sample_dendrogram(context: VisualizationContext, save_stem: Path) -> None:
    plot_sample_dendrogram(context.pca_adata, save_stem, context.cfg)


def _render_transcriptome_pca(context: VisualizationContext, save_stem: Path) -> None:
    plot_transcriptome_pca(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.transcriptome_pca_result,
    )


def _render_metabolome_pca(context: VisualizationContext, save_stem: Path) -> None:
    plot_metabolome_pca(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.metabolome_pca_result,
    )


def _render_transcriptome_pca_pairs(context: VisualizationContext, save_stem: Path) -> None:
    plot_transcriptome_pca_pairs(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.transcriptome_pca_result,
    )


def _render_metabolome_pca_pairs(context: VisualizationContext, save_stem: Path) -> None:
    plot_metabolome_pca_pairs(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.metabolome_pca_result,
    )


def _render_transcriptome_pca_subgroups(context: VisualizationContext, save_stem: Path) -> None:
    plot_transcriptome_pca_subgroups(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.transcriptome_pca_result,
    )


def _render_metabolome_pca_subgroups(context: VisualizationContext, save_stem: Path) -> None:
    plot_metabolome_pca_subgroups(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.metabolome_pca_result,
    )


def _render_transcriptome_pca_pairs_subgroups(context: VisualizationContext, save_stem: Path) -> None:
    plot_transcriptome_pca_pairs_subgroups(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.transcriptome_pca_result,
    )


def _render_metabolome_pca_pairs_subgroups(context: VisualizationContext, save_stem: Path) -> None:
    plot_metabolome_pca_pairs_subgroups(
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
        pca_result=context.metabolome_pca_result,
    )


def _render_top_edge_scatter_panels(context: VisualizationContext, save_stem: Path) -> None:
    plot_top_edge_scatter_panels(context.engine, save_stem, context.cfg)


def _render_top_gene_metabolite_correlation_heatmaps(context: VisualizationContext, save_stem: Path) -> None:
    plot_top_gene_metabolite_correlation_heatmaps(context.engine, save_stem, context.cfg)


def _render_gene_metabolite_correlation_bubble_heatmap(context: VisualizationContext, save_stem: Path) -> None:
    plot_gene_metabolite_correlation_bubble_heatmap(context.engine, save_stem, context.cfg)


def _render_top_metabolite_group1_violin_box(context: VisualizationContext, save_stem: Path) -> None:
    plot_top_metabolite_group1_violin_box(
        context.engine,
        context.pca_adata,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_top_metabolite_regression_panels(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_top_metabolite_regression_panels(context.engine, save_stem, context.cfg)


def _render_module_eigengene_heatmap(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_heatmap(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_eigengene_heatmap_group2(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_heatmap_group2(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_zscore_line_panels(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_zscore_line_panels(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_gene_zscore_line_panels(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_gene_zscore_line_panels(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_eigengene_ridge(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_ridge(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_eigengene_ridge_group1(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_ridge_group1(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_eigengene_group1_violin_box(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_group1_violin_box(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_module_kme_boxplot(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_kme_boxplot(context.engine, save_stem, context.cfg)


def _render_module_metabolite_bubble_plot(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_metabolite_bubble_plot(context.engine, save_stem, context.cfg)


def _render_association_direction_summary(context: VisualizationContext, save_stem: Path) -> None:
    plot_association_direction_summary(context.engine, save_stem, context.cfg)


def _render_module_eigengene_metabolite_trend_panels(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_eigengene_metabolite_trend_panels(
        context.engine,
        save_stem,
        context.cfg,
        group_df=context.pca_group_df,
    )


def _render_edgeweight_distribution_by_module(context: VisualizationContext, save_stem: Path) -> None:
    plot_edgeweight_distribution_by_module(context.engine, save_stem, context.cfg)


def _render_module_metabolite_association_heatmap(context: VisualizationContext, save_stem: Path) -> None:
    plot_module_metabolite_association_heatmap(context.engine, save_stem, context.cfg)


def _render_compressed_circos_network(context: VisualizationContext, save_stem: Path) -> None:
    plot_compressed_circos_network(context.engine, save_stem, context.cfg)


def _render_floating_cnet_circos_network(context: VisualizationContext, save_stem: Path) -> None:
    plot_floating_cnet_circos_network(context.engine, save_stem, context.cfg)


def _render_association_evidence_upset(context: VisualizationContext, save_stem: Path) -> None:
    plot_association_evidence_upset(context.engine, save_stem, context.cfg)


STATIC_FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec("sample_clustering_dendrogram", "sample_clustering_dendrogram", _render_sample_dendrogram),
    FigureSpec("transcriptome_pca", "transcriptome_pca", _render_transcriptome_pca),
    FigureSpec(
        "transcriptome_pca_subgroups",
        "transcriptome_pca_subgroups",
        _render_transcriptome_pca_subgroups,
        enabled=_has_secondary_pca_grouping,
    ),
    FigureSpec("transcriptome_pca_pairs", "transcriptome_pca_pairs", _render_transcriptome_pca_pairs),
    FigureSpec(
        "transcriptome_pca_pairs_subgroups",
        "transcriptome_pca_pairs_subgroups",
        _render_transcriptome_pca_pairs_subgroups,
        enabled=_has_secondary_pca_grouping,
    ),
    FigureSpec("metabolome_pca", "metabolome_pca", _render_metabolome_pca),
    FigureSpec(
        "metabolome_pca_subgroups",
        "metabolome_pca_subgroups",
        _render_metabolome_pca_subgroups,
        enabled=_has_secondary_pca_grouping,
    ),
    FigureSpec("metabolome_pca_pairs", "metabolome_pca_pairs", _render_metabolome_pca_pairs),
    FigureSpec(
        "metabolome_pca_pairs_subgroups",
        "metabolome_pca_pairs_subgroups",
        _render_metabolome_pca_pairs_subgroups,
        enabled=_has_secondary_pca_grouping,
    ),
    FigureSpec(
        "association_evidence_upset",
        "association_evidence_upset",
        _render_association_evidence_upset,
    ),
    FigureSpec(
        "gene_metabolite_correlation_bubble_heatmap",
        "gene_metabolite_correlation_bubble_heatmap",
        _render_gene_metabolite_correlation_bubble_heatmap,
    ),
    FigureSpec(
        "top_gene_metabolite_correlation_heatmaps",
        "top_gene_metabolite_correlation_heatmaps",
        _render_top_gene_metabolite_correlation_heatmaps,
    ),
    FigureSpec("top_gene_metabolite_pairs", "top_gene_metabolite_pairs", _render_top_edge_scatter_panels),
    FigureSpec(
        "top_metabolite_group1_violin_box",
        "top_metabolite_group1_violin_box",
        _render_top_metabolite_group1_violin_box,
    ),
    FigureSpec("module_eigengene_heatmap", "module_eigengene_heatmap", _render_module_eigengene_heatmap),
    FigureSpec(
        "module_eigengene_heatmap_group2",
        "module_eigengene_heatmap_group2",
        _render_module_eigengene_heatmap_group2,
    ),
    FigureSpec("module_zscore_line_panels", "module_zscore_line_panels", _render_module_zscore_line_panels),
    FigureSpec(
        "module_gene_zscore_line_panels",
        "module_gene_zscore_line_panels",
        _render_module_gene_zscore_line_panels,
    ),
    FigureSpec("module_eigengene_ridge", "module_eigengene_ridge", _render_module_eigengene_ridge),
    FigureSpec(
        "module_eigengene_ridge_group1",
        "module_eigengene_ridge_group1",
        _render_module_eigengene_ridge_group1,
    ),
    FigureSpec(
        "module_eigengene_group1_violin_box",
        "module_eigengene_group1_violin_box",
        _render_module_eigengene_group1_violin_box,
    ),
    FigureSpec("module_kme_boxplot", "module_kme_boxplot", _render_module_kme_boxplot),
    FigureSpec(
        "module_metabolite_association_heatmap",
        "module_metabolite_association_heatmap",
        _render_module_metabolite_association_heatmap,
    ),
    FigureSpec("module_metabolite_bubble_plot", "module_metabolite_bubble_plot", _render_module_metabolite_bubble_plot),
    FigureSpec(
        "module_top_metabolite_regressions",
        "module_top_metabolite_regressions",
        _render_module_top_metabolite_regression_panels,
    ),
    FigureSpec(
        "module_eigengene_metabolite_trend_panels",
        "module_eigengene_metabolite_trend_panels",
        _render_module_eigengene_metabolite_trend_panels,
    ),
    FigureSpec("association_direction_summary", "association_direction_summary", _render_association_direction_summary),
    FigureSpec(
        "edgeweight_distribution_by_module",
        "edgeweight_distribution_by_module",
        _render_edgeweight_distribution_by_module,
    ),
    FigureSpec("compressed_circos_network", "compressed_circos_network", _render_compressed_circos_network),
    FigureSpec(
        "floating_cnet_circos_network",
        "floating_cnet_circos_network",
        _render_floating_cnet_circos_network,
    ),
)


FIGURE_REGISTRY: dict[str, FigureSpec] = {spec.key: spec for spec in STATIC_FIGURE_SPECS}


def iter_figure_specs() -> tuple[FigureSpec, ...]:
    return STATIC_FIGURE_SPECS

__all__ = [
    "FigureSpec",
    "FIGURE_REGISTRY",
    "FIGURE_FILE_PREFIXES",
    "STATIC_FIGURE_SPECS",
    "TABLE_FILE_PREFIXES",
    "iter_figure_specs",
]
