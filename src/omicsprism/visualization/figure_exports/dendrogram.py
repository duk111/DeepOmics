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

def export_dendrogram(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export dendrogram data for interactive page 'dendrogram'."""
    style = _base_style()
    adata = context.pca_adata
    if adata.n_obs < 2:
        return None

    from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

    matrix = np.asarray(adata.X, dtype=np.float32)
    linkage_matrix = linkage(matrix, method="average")
    labels = adata.obs_names.tolist()
    color_threshold = 0.70 * float(linkage_matrix[:, 2].max())
    dendrogram_data = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        color_threshold=color_threshold,
        above_threshold_color="#aaaaaa",
        no_plot=True,
    )
    root = to_tree(linkage_matrix, rd=False)

    def _leaf_names(node) -> list[str]:
        if node.is_leaf():
            return [str(labels[node.id])]
        return _leaf_names(node.left) + _leaf_names(node.right)

    branch_samples: list[list[str]] = []

    def _collect_branch_samples(node) -> None:
        if node.is_leaf():
            return
        _collect_branch_samples(node.left)
        _collect_branch_samples(node.right)
        branch_samples.append(_leaf_names(node))

    _collect_branch_samples(root)

    # Convert linkage to D3-compatible tree structure
    n = len(labels)
    nodes = []
    for i in range(n):
        nodes.append({"id": i, "name": str(labels[i]), "is_leaf": True, "height": 0.0})

    for i, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        dist = float(row[2])
        node = {"id": n + i, "left": left, "right": right, "height": dist, "is_leaf": False,
                "name": f"Node {n+i}", "count": int(row[3]) if len(row) > 3 else 2}
        nodes.append(node)

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": "sample_dendrogram",
        "title": "Sample Clustering Dendrogram",
        "chart_type": "dendrogram",
        "interactive_page_id": "dendrogram",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {},  # rendered with D3, not Plotly
        "tree_data": {"nodes": nodes, "linkage": linkage_matrix.tolist(),
                      "labels": labels, "color_threshold": color_threshold,
                      "n_leaves": n,
                      "dendrogram": {
                          "icoord": dendrogram_data.get("icoord", []),
                          "dcoord": dendrogram_data.get("dcoord", []),
                          "ivl": dendrogram_data.get("ivl", []),
                          "leaves": dendrogram_data.get("leaves", []),
                          "color_list": dendrogram_data.get("color_list", []),
                          "branch_samples": branch_samples,
                      }},
        "default_state": {"orientation": "vertical", "color_threshold": color_threshold},
        "available_states": {},
        "style": style,
    }


# ---------------------------------------------------------------------------
# UpSet (F10)
# ---------------------------------------------------------------------------

__all__ = [
    "export_dendrogram",
]
