from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import (
    _global_secondary_group_color_map,
    _group_color_map,
    _group_marker_map,
    _ordered_unique_nonempty,
    _ordered_unique_with_order,
)
from .common import (
    _base_layout,
    _base_plotly_config,
    _base_style,
    _json_matrix_values,
    _rgba,
    _seaborn_vlag_colorscale,
)

def export_circos(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export circos network data for interactive page 'circos'."""
    style = _base_style()
    engine = context.engine

    from ..static.network import _prepare_circos_node_tables, _attach_circos_module_annotations, _compute_circos_layout

    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return None

    gene_summary, module_color_map = _attach_circos_module_annotations(engine, gene_summary)
    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return None

    is_floating = "floating" in prefix_key.lower() or "cnet" in prefix_key.lower()

    # Prepare nodes
    nodes = []
    for _, row in gene_summary.iterrows():
        node_id = str(row["Node"])
        geo = layout.get(node_id)
        if geo is None:
            continue
        nodes.append({
            "id": node_id,
            "name": node_id,
            "type": "gene",
            "theta_start": float(geo["theta_start"]),
            "theta_end": float(geo["theta_end"]),
            "theta_mid": float(geo["theta_mid"]),
            "module": str(row.get("Module", "")),
            "module_color": str(row.get("ModuleColor", "#9ca3af")),
            "mean_zscore": float(row.get("MeanZScore", 0)),
            "weighted_degree": float(row.get("WeightedDegree", 0)),
            "kme": float(row.get("kME", 0)) if pd.notna(row.get("kME")) else 0.0,
            "direction_bias": float(row.get("DirectionBias", 0)),
        })

    for _, row in metabolite_summary.iterrows():
        node_id = str(row["Node"])
        geo = layout.get(node_id)
        if geo is None:
            continue
        nodes.append({
            "id": node_id,
            "name": node_id,
            "type": "metabolite",
            "theta_start": float(geo["theta_start"]),
            "theta_end": float(geo["theta_end"]),
            "theta_mid": float(geo["theta_mid"]),
            "module": "",
            "module_color": "#c9ad85",
            "mean_zscore": float(row.get("MeanZScore", 0)),
            "weighted_degree": float(row.get("WeightedDegree", 0)),
            "kme": 0.0,
            "direction_bias": float(row.get("DirectionBias", 0)),
        })

    # Prepare edges
    edges = []
    for row in edge_df.itertuples(index=False):
        gene_id = str(row.Gene)
        metabolite_id = str(row.Metabolite)
        if gene_id not in layout or metabolite_id not in layout:
            continue
        edges.append({
            "source": gene_id,
            "target": metabolite_id,
            "weight": float(row.EdgeWeight),
            "sign": str(row.Sign).lower(),
            "model_support": float(row.ModelSupportCount) if hasattr(row, "ModelSupportCount") else 0,
        })

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{'floating' if is_floating else 'compressed'}_circos",
        "title": f"{'Floating CNet' if is_floating else 'Compressed'} Circos Network",
        "chart_type": "circos",
        "interactive_page_id": "circos",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "circos_data": {
            "nodes": nodes,
            "edges": edges,
            "layout_type": "floating" if is_floating else "compressed",
            "module_color_map": module_color_map,
        },
        "default_state": {"layout": "floating" if is_floating else "compressed",
                         "min_edge_weight": 0.0,
                         "sign_filter": "all"},
        "available_states": {"layout": ["compressed", "floating"],
                            "sign_filter": ["all", "positive", "negative"],
                            "min_edge_weight": list(np.arange(0.0, 1.01, 0.05))},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Export dispatcher
# ---------------------------------------------------------------------------

__all__ = [
    "export_circos",
]
