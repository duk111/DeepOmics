from __future__ import annotations

from .base import PALETTE, set_academic_style
from .correlation import plot_top_gene_metabolite_correlation_heatmaps
from .distribution import (
    plot_module_eigengene_group1_violin_box,
    plot_module_kme_boxplot,
    plot_top_metabolite_group1_violin_box,
)
from .embedding import (
    plot_metabolome_tsne,
    plot_metabolome_tsne_subgroups,
    plot_metabolome_umap,
    plot_metabolome_umap_subgroups,
    plot_transcriptome_tsne,
    plot_transcriptome_tsne_subgroups,
    plot_transcriptome_umap,
    plot_transcriptome_umap_subgroups,
)
from .module import (
    plot_module_eigengene_heatmap,
    plot_module_eigengene_heatmap_group2,
    plot_module_eigengene_ridge,
    plot_module_eigengene_ridge_group1,
    plot_module_gene_zscore_line_panels,
    plot_module_metabolite_association_heatmap,
    plot_module_zscore_line_panels,
)
from .network import plot_compressed_circos_network, plot_floating_cnet_circos_network
from .pca import (
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
from .regression import plot_module_top_metabolite_regression_panels, plot_top_edge_scatter_panels
from .upset import plot_association_evidence_upset

__all__ = [
    "PALETTE",
    "set_academic_style",
    "plot_sample_dendrogram",
    "plot_transcriptome_pca",
    "plot_metabolome_pca",
    "plot_transcriptome_pca_subgroups",
    "plot_metabolome_pca_subgroups",
    "plot_transcriptome_pca_pairs",
    "plot_metabolome_pca_pairs",
    "plot_transcriptome_pca_pairs_subgroups",
    "plot_metabolome_pca_pairs_subgroups",
    "plot_transcriptome_umap",
    "plot_transcriptome_umap_subgroups",
    "plot_metabolome_umap",
    "plot_metabolome_umap_subgroups",
    "plot_transcriptome_tsne",
    "plot_transcriptome_tsne_subgroups",
    "plot_metabolome_tsne",
    "plot_metabolome_tsne_subgroups",
    "plot_top_edge_scatter_panels",
    "plot_top_gene_metabolite_correlation_heatmaps",
    "plot_module_eigengene_group1_violin_box",
    "plot_module_kme_boxplot",
    "plot_top_metabolite_group1_violin_box",
    "plot_module_top_metabolite_regression_panels",
    "plot_module_eigengene_heatmap",
    "plot_module_eigengene_heatmap_group2",
    "plot_module_zscore_line_panels",
    "plot_module_gene_zscore_line_panels",
    "plot_module_eigengene_ridge",
    "plot_module_eigengene_ridge_group1",
    "plot_module_metabolite_association_heatmap",
    "plot_compressed_circos_network",
    "plot_floating_cnet_circos_network",
    "plot_association_evidence_upset",
]
