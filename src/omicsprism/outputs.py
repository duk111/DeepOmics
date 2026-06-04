from __future__ import annotations

FIGURE_FILE_PREFIXES = {
    "sample_clustering_dendrogram": "F01_Sample_Clustering_Dendrogram",
    "transcriptome_pca": "F02_Transcriptome_PCA",
    "metabolome_pca": "F03_Metabolome_PCA",
    "transcriptome_pca_subgroups": "F02B_Transcriptome_PCA_Subgroups",
    "metabolome_pca_subgroups": "F03B_Metabolome_PCA_Subgroups",
    "transcriptome_pca_pairs": "F02C_Transcriptome_PCA_Pairs",
    "metabolome_pca_pairs": "F03C_Metabolome_PCA_Pairs",
    "transcriptome_pca_pairs_subgroups": "F02D_Transcriptome_PCA_Pairs_Subgroups",
    "metabolome_pca_pairs_subgroups": "F03D_Metabolome_PCA_Pairs_Subgroups",
    "top_gene_metabolite_pairs": "F04_Top_Gene_Metabolite_Pairs",
    "module_top_metabolite_regressions": "F04B_Module_Top_Metabolite_Regressions",
    "top_gene_metabolite_correlation_heatmaps": "F04C_Top_Gene_Metabolite_Correlation_Heatmaps",
    "top_metabolite_group1_violin_box": "F04D_Top_Metabolite_Group1_Violin_Box",
    "gene_metabolite_correlation_bubble_heatmap": "F04E_Gene_Metabolite_Correlation_Bubble_Heatmap",
    "module_eigengene_heatmap": "F05A_Module_Eigengene_Heatmap",
    "module_eigengene_heatmap_group2": "F05B_Module_Eigengene_Heatmap_Group2",
    "module_zscore_line_panels": "F05C_Module_Zscore_Line_Panels",
    "module_gene_zscore_line_panels": "F05D_Module_Gene_Zscore_Line_Panels",
    "module_eigengene_ridge": "F05E_Module_Eigengene_Ridge",
    "module_eigengene_ridge_group1": "F05F_Module_Eigengene_Ridge_Group1",
    "module_eigengene_group1_violin_box": "F05G_Module_Eigengene_Group1_Violin_Box",
    "module_kme_boxplot": "F05H_Module_kME_Boxplot",
    "module_metabolite_bubble_plot": "F05I_Module_Metabolite_Bubble_Plot",
    "association_direction_summary": "F05J_Association_Direction_Summary",
    "module_eigengene_metabolite_trend_panels": "F05K_Module_Eigengene_Metabolite_Trend_Panels",
    "edgeweight_distribution_by_module": "F05L_EdgeWeight_Distribution_By_Module",
    "module_metabolite_association_heatmap": "F05_Module_Metabolite_Association_Heatmap",
    "compressed_circos_network": "F06_Compressed_Circos_Network",
    "floating_cnet_circos_network": "F07_Floating_CNet_Circos_Network",
    "association_evidence_upset": "F08_Association_Evidence_UpSet",
}


TABLE_FILE_PREFIXES = {
    "gene_scores": "T01_Metabolite_Gene_Scoring_Table.csv",
    "screening_summary": "T01b_Metabolite_Screening_Summary.csv",
    "total_network": "T02_Total_Association_Network.csv",
    "high_confidence_network": "T03_High_Confidence_Network.csv",
    "key_gene_summary": "T04_Key_Gene_Summary.csv",
    "metabolite_summary": "T05_Metabolite_Association_Summary.csv",
    "gene_module_assignment": "T07_Gene_Module_Assignment.csv",
    "module_eigengenes": "T08_Module_Eigengenes.csv",
    "module_metabolite_association": "T09_Module_Metabolite_Association.csv",
    "module_summary": "T10_Module_Summary.csv",
}


__all__ = [
    "FIGURE_FILE_PREFIXES",
    "TABLE_FILE_PREFIXES",
]
