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

def export_upset(context, save_dir: Path, prefix_key: str) -> dict[str, Any]:
    """Export UpSet plot data for interactive page 'upset'."""
    style = _base_style()
    engine = context.engine
    gene_scores_df = engine.ml_results.get("gene_scores_df", pd.DataFrame())
    if gene_scores_df.empty:
        return None

    from ..static.upset import EVIDENCE_SPECS, _build_evidence_intersection_table

    intersections, set_sizes, n_edges = _build_evidence_intersection_table(gene_scores_df, max_intersections=30)
    if intersections.empty:
        return None

    evidence_columns = [col for col, _, _ in EVIDENCE_SPECS]
    evidence_labels = [label for _, label, _ in EVIDENCE_SPECS]
    evidence_colors = [color for _, _, color in EVIDENCE_SPECS]

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": "association_evidence_upset",
        "title": "Association Evidence Overlap",
        "chart_type": "upset",
        "interactive_page_id": "upset",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "upset_data": {
            "sets": [{"name": label, "size": int(set_sizes.get(col, 0)), "color": color}
                     for col, label, color in EVIDENCE_SPECS],
            "intersections": [
                {**{col: bool(row[col]) for col in evidence_columns},
                 "count": int(row["Count"]),
                 "support": int(row.get("SupportCount", 0))}
                for _, row in intersections.iterrows()
            ],
            "n_edges": int(n_edges),
        },
        "default_state": {"sort_by": "size", "max_intersections": 30},
        "available_states": {"sort_by": ["size", "degree", "combination"],
                            "max_intersections": [10, 20, 30, 40, 50]},
        "style": style,
    }


# ---------------------------------------------------------------------------
# Bubble Heatmap (F11, F24)
# ---------------------------------------------------------------------------

__all__ = [
    "export_upset",
]
