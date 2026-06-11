"""
Figure data export registry for interactive visualization pages.

Concrete exporters live in :mod:`omicsprism.visualization.figure_exports`.
This module keeps the public dispatch API stable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .figure_exports import (
    export_bubble_heatmap,
    export_circos,
    export_corr_heatmap,
    export_dendrogram,
    export_line_panels,
    export_pca_page,
    export_pca_pairs,
    export_pca_scatter,
    export_ridge,
    export_scatter_panels,
    export_upset,
    export_violin_box,
)

EXPORT_MAP = {
    # PCA scatter: one export serves all 4 variants
    "transcriptome_pca": ("pca", export_pca_scatter),
    "transcriptome_pca_subgroups": ("pca", export_pca_scatter),
    "metabolome_pca": ("pca", export_pca_scatter),
    "metabolome_pca_subgroups": ("pca", export_pca_scatter),
    # PCA pairs
    "transcriptome_pca_pairs": ("pca", export_pca_pairs),
    "transcriptome_pca_pairs_subgroups": ("pca", export_pca_pairs),
    "metabolome_pca_pairs": ("pca", export_pca_pairs),
    "metabolome_pca_pairs_subgroups": ("pca", export_pca_pairs),
    # Individual
    "sample_clustering_dendrogram": ("dendrogram", export_dendrogram),
    "association_evidence_upset": ("upset", export_upset),
    "gene_metabolite_correlation_bubble_heatmap": ("bubble-heatmap", export_bubble_heatmap),
    "top_gene_metabolite_pairs": ("scatter-panels", export_scatter_panels),
    "top_metabolite_group1_violin_box": ("violin-box", export_violin_box),
    "top_gene_metabolite_correlation_heatmaps": ("corr-heatmap", export_corr_heatmap.export_gene_metabolite_heatmap),
    "module_metabolite_association_heatmap": ("corr-heatmap", export_corr_heatmap.export_module_metabolite_heatmap),
    "module_eigengene_ridge_group1": ("ridge", export_ridge),
    "module_eigengene_group1_violin_box": ("violin-box", export_violin_box),
    "module_metabolite_bubble_plot": ("bubble-heatmap", export_bubble_heatmap),
    "module_top_metabolite_regressions": ("scatter-panels", export_scatter_panels),
    "module_eigengene_metabolite_trend_panels": ("line-panels", export_line_panels),
    "compressed_circos_network": ("circos", export_circos),
    "floating_cnet_circos_network": ("circos", export_circos),
}


def export_figure_data(context, figure_spec, save_dir: Path) -> None:
    """Export figure data JSON for a single FigureSpec."""
    key = figure_spec.key
    if key not in EXPORT_MAP:
        return

    page_id, export_fn = EXPORT_MAP[key]
    try:
        data = export_fn(context, save_dir, figure_spec.prefix_key)
    except Exception:
        import traceback
        traceback.print_exc()
        return

    if data is None:
        return

    # Use the page_id as the file name for shared pages
    out_path = save_dir / f"{page_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists (shared page, e.g. bubble-heatmap from F11+F24),
    # merge the new data as an alternative view rather than overwriting
    if out_path.exists() and page_id != "pca":
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            # Merge available_states (union of options)
            for key, vals in data.get("available_states", {}).items():
                if key in existing.get("available_states", {}):
                    merged = list(dict.fromkeys(
                        list(existing["available_states"][key]) + list(vals)
                    ))
                    existing["available_states"][key] = merged
            if page_id == "circos":
                existing_layouts = existing.setdefault("circos_data", {}).setdefault("layouts", {})
                new_layouts = data.get("circos_data", {}).get("layouts", {})
                for layout_name, layout_data in new_layouts.items():
                    if layout_data is not None:
                        existing_layouts[layout_name] = layout_data
                existing.setdefault("available_states", {})["layout"] = list(
                    dict.fromkeys(
                        list(existing.get("available_states", {}).get("layout", []))
                        + list(data.get("available_states", {}).get("layout", []))
                    )
                )
            else:
                # Keep the first (primary) dataset; mark alternate data available
                if "alt_data" not in existing:
                    existing["alt_data"] = {}
                existing["alt_data"][data.get("figure_id", key)] = {
                    "plotly_spec": data.get("plotly_spec", {}),
                    "default_state": data.get("default_state", {}),
                }
            data = existing
        except Exception:
            pass  # If merge fails, overwrite with new data

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "EXPORT_MAP",
    "export_figure_data",
    "export_pca_page",
    "export_pca_scatter",
    "export_pca_pairs",
    "export_dendrogram",
    "export_upset",
    "export_bubble_heatmap",
    "export_scatter_panels",
    "export_violin_box",
    "export_line_panels",
    "export_ridge",
    "export_circos",
]
