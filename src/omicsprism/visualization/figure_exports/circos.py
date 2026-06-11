from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import PALETTE, _gene_expression_df, _hue_wheel_color_series, _metabolomics_df
from ..static.pca import _load_pca_group_table
from .common import _base_style


CIRCOS_RADII = {
    "outer_strip_inner": 0.992,
    "outer_strip_outer": 1.035,
    "track_meanbar_inner": 0.86,
    "track_meanbar_outer": 0.975,
    "track_meanheat_inner": 0.795,
    "track_meanheat_outer": 0.85,
    "track_degree_inner": 0.685,
    "track_degree_outer": 0.775,
    "track_core_inner": 0.605,
    "track_core_outer": 0.675,
    "track_bias_inner": 0.53,
    "track_bias_outer": 0.58,
    "link_radius": 0.47,
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _robust_abs_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(arr), 95)), 1e-6)


def _positive_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float(np.nanmax(arr)), 1e-6)


def _metabolite_module_core_map(engine) -> pd.Series:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if isinstance(assoc_df, pd.DataFrame) and not assoc_df.empty and {"Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        work = assoc_df.copy()
        work["Metabolite"] = work["Metabolite"].astype(str)
        work["AbsRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce").abs()
        return work.groupby("Metabolite", sort=False)["AbsRho"].max().astype(float)

    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and {"Metabolite", "EdgeWeight"}.issubset(edge_df.columns):
        return edge_df.groupby("Metabolite", sort=False)["EdgeWeight"].sum().astype(float)
    return pd.Series(dtype=float)


def _node_payload(row, layout: dict[str, dict[str, float | str]]) -> dict[str, Any] | None:
    node_id = str(row.Node)
    geo = layout.get(node_id)
    if geo is None:
        return None
    node_type = str(row.NodeType)
    return {
        "id": node_id,
        "name": node_id,
        "type": node_type,
        "theta_start": _finite_float(geo.get("theta_start")),
        "theta_end": _finite_float(geo.get("theta_end")),
        "theta_mid": _finite_float(geo.get("theta_mid")),
        "module": str(getattr(row, "Module", "")) if node_type == "gene" else "",
        "module_color": str(getattr(row, "ModuleColor", "#c9ad85" if node_type == "metabolite" else "#9ca3af")),
        "mean_zscore": _finite_float(getattr(row, "MeanZScore", 0.0)),
        "weighted_degree": _finite_float(getattr(row, "WeightedDegree", 0.0)),
        "module_core": _finite_float(getattr(row, "ModuleCore", 0.0)),
        "direction_bias": float(np.clip(_finite_float(getattr(row, "DirectionBias", 0.0)), -1.0, 1.0)),
        "positive_edges": int(_finite_float(getattr(row, "PositiveEdgeCount", 0.0))),
        "negative_edges": int(_finite_float(getattr(row, "NegativeEdgeCount", 0.0))),
        "kme": _finite_float(getattr(row, "kME", 0.0)),
    }


def _edge_payloads(edge_df: pd.DataFrame, layout: dict[str, dict[str, float | str]]) -> list[dict[str, Any]]:
    edges = []
    for row in edge_df.itertuples(index=False):
        gene_id = str(row.Gene)
        metabolite_id = str(row.Metabolite)
        if gene_id not in layout or metabolite_id not in layout:
            continue
        edges.append(
            {
                "source": gene_id,
                "target": metabolite_id,
                "weight": _finite_float(getattr(row, "EdgeWeight", 0.0)),
                "sign": str(getattr(row, "Sign", "")).lower(),
                "model_support": _finite_float(getattr(row, "ModelSupportCount", 0.0)),
            }
        )
    return edges


def _module_legend(gene_summary: pd.DataFrame) -> list[dict[str, str]]:
    if gene_summary.empty or "Module" not in gene_summary.columns:
        return []
    seen: set[str] = set()
    items = []
    for row in gene_summary.loc[:, ["Module", "ModuleColor"]].drop_duplicates().itertuples(index=False):
        module = str(row.Module)
        if module in seen:
            continue
        seen.add(module)
        items.append({"label": module, "color": str(row.ModuleColor)})
    return items


def _group1_legend(track_data: dict[str, object] | None) -> list[dict[str, str]]:
    if track_data is None or str(track_data.get("mode", "")) != "group1_mean":
        return []
    color_map = {str(key): str(value) for key, value in dict(track_data.get("group1_color_map", {})).items()}
    items = []
    seen: set[str] = set()
    for group in [str(value) for value in track_data.get("group1_order", []) if str(value).strip()]:
        if group in seen:
            continue
        seen.add(group)
        items.append({"label": group, "color": color_map.get(group, "#6b7280")})
    return items


def _track_values(track_data: dict[str, object] | None, node_id: str) -> list[float]:
    if track_data is None:
        return []
    values = dict(track_data.get("feature_to_values", {})).get(node_id, [])
    return [_finite_float(value, np.nan) for value in values]


def _build_compressed_layout(engine, cfg, edge_df: pd.DataFrame, gene_summary: pd.DataFrame, metabolite_summary: pd.DataFrame) -> dict[str, Any] | None:
    from ..static.network import _compute_circos_layout

    metabolite_core = _metabolite_module_core_map(engine)
    gene_summary = gene_summary.copy()
    metabolite_summary = metabolite_summary.copy()
    metabolite_summary["Module"] = ""
    metabolite_summary["ModuleColor"] = "#c9ad85"
    metabolite_summary["ModuleCore"] = metabolite_summary["Node"].map(metabolite_core).astype(float)
    gene_summary["ModuleCore"] = pd.to_numeric(gene_summary.get("kME", np.nan), errors="coerce").abs()

    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return None

    group_df = _load_pca_group_table(cfg)
    track_adata = getattr(engine, "plot_adata", getattr(engine, "unaggregated_adata", engine.adata))

    from ..static.network import _prepare_group1_mean_track_data

    gene_track_data = _prepare_group1_mean_track_data(_gene_expression_df(track_adata), group_df)
    metabolite_track_data = _prepare_group1_mean_track_data(_metabolomics_df(track_adata), group_df)

    node_df = pd.concat([gene_summary, metabolite_summary], ignore_index=True)
    node_df["Node"] = node_df["Node"].astype(str)
    nodes = []
    for row in node_df.itertuples(index=False):
        payload = _node_payload(row, layout)
        if payload is None:
            continue
        track_data = gene_track_data if payload["type"] == "gene" else metabolite_track_data
        payload["track_values"] = _track_values(track_data, payload["id"])
        payload["track_mode"] = str(track_data.get("mode", "")) if track_data is not None else ""
        nodes.append(payload)

    edges = _edge_payloads(edge_df, layout)
    edge_order = sorted(
        edges,
        key=lambda edge: (edge["weight"], edge["model_support"], edge["source"], edge["target"]),
    )

    return {
        "type": "circos",
        "nodes": nodes,
        "edges": edge_order,
        "gene_nodes": gene_nodes,
        "metabolite_nodes": metabolite_nodes,
        "radii": CIRCOS_RADII,
        "scales": {
            "gene_mean": _robust_abs_scale(gene_summary["MeanZScore"]),
            "metabolite_mean": _robust_abs_scale(metabolite_summary["MeanZScore"]),
            "gene_degree": _positive_scale(gene_summary["WeightedDegree"]),
            "metabolite_degree": _positive_scale(metabolite_summary["WeightedDegree"]),
            "gene_core": _positive_scale(gene_summary["ModuleCore"]),
            "metabolite_core": _positive_scale(metabolite_summary["ModuleCore"]),
            "track_abs": max(
                _finite_float((gene_track_data or {}).get("abs_scale", 1.0), 1.0),
                _finite_float((metabolite_track_data or {}).get("abs_scale", 1.0), 1.0),
            ),
        },
        "group1_order": [str(value) for value in (gene_track_data or {}).get("group1_order", [])],
        "group1_color_map": {str(key): str(value) for key, value in dict((gene_track_data or {}).get("group1_color_map", {})).items()},
        "group_legend": _group1_legend(gene_track_data),
        "module_legend": _module_legend(gene_summary),
        "track_legend": [
            {"label": "track 1", "description": "sector strip"},
            {"label": "track 2", "description": "group-wise mean"},
            {"label": "track 3", "description": "mean z-score heatmap"},
            {"label": "track 4", "description": "weighted degree"},
            {"label": "track 5", "description": "module/core strength"},
            {"label": "track 6", "description": "direction bias"},
        ],
    }


def _build_cnet_layout(edge_df: pd.DataFrame, gene_summary: pd.DataFrame, metabolite_summary: pd.DataFrame) -> dict[str, Any] | None:
    from ..static.network import _compute_circos_layout, _polar_to_xy

    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return None

    ordered_nodes = gene_nodes + metabolite_nodes
    theta_values = np.asarray([float(layout[node]["theta_mid"]) for node in ordered_nodes], dtype=float)
    if theta_values.size == 0:
        return None

    gene_summary = gene_summary.copy()
    metabolite_summary = metabolite_summary.copy()
    gene_summary["EdgeCount"] = (
        pd.to_numeric(gene_summary.get("PositiveEdgeCount", 0), errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(gene_summary.get("NegativeEdgeCount", 0), errors="coerce").fillna(0).astype(int)
    )
    metabolite_summary["EdgeCount"] = (
        pd.to_numeric(metabolite_summary.get("PositiveEdgeCount", 0), errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(metabolite_summary.get("NegativeEdgeCount", 0), errors="coerce").fillna(0).astype(int)
    )
    metabolite_summary["ModuleColor"] = "#c9ad85"

    node_table = pd.concat(
        [
            gene_summary.loc[:, ["Node", "NodeType", "EdgeCount", "Module", "ModuleColor", "MeanZScore", "WeightedDegree", "DirectionBias", "kME"]],
            metabolite_summary.assign(Module="", kME=0.0).loc[:, ["Node", "NodeType", "EdgeCount", "Module", "ModuleColor", "MeanZScore", "WeightedDegree", "DirectionBias", "kME"]],
        ],
        ignore_index=True,
    )
    node_table["Node"] = node_table["Node"].astype(str)
    node_table = node_table.set_index("Node").reindex(ordered_nodes).reset_index()

    wrapped = np.r_[theta_values, theta_values[0] + 2.0 * np.pi]
    theta_diffs = np.diff(wrapped)
    positive_diffs = theta_diffs[theta_diffs > 1e-6]
    min_theta_gap = float(np.min(positive_diffs)) if positive_diffs.size else (2.0 * np.pi)

    base_radius = 1.0
    min_center_distance = 2.0 * base_radius * np.sin(max(min_theta_gap, 1e-6) / 2.0)
    max_node_radius = float(np.clip(min_center_distance * 0.36, 0.012, 0.032))
    min_node_radius = float(np.clip(max_node_radius * 0.42, 0.006, max_node_radius * 0.72))

    edge_counts = pd.to_numeric(node_table["EdgeCount"], errors="coerce").fillna(0).astype(float)
    if float(edge_counts.max()) > float(edge_counts.min()):
        scaled = (edge_counts - edge_counts.min()) / (edge_counts.max() - edge_counts.min())
    else:
        scaled = pd.Series(np.ones(len(edge_counts)), index=node_table.index, dtype=float)

    base_jitter = min(0.060, max(0.016, min_theta_gap * 0.12))
    jitter = base_jitter * np.sin(np.linspace(0.0, 3.2 * np.pi, len(theta_values), endpoint=False) + 0.65)
    node_table["theta"] = theta_values
    node_table["radius"] = (base_radius + jitter).astype(float)
    node_table["node_radius"] = (min_node_radius + scaled * (max_node_radius - min_node_radius)).astype(float)

    xy = [_polar_to_xy(float(theta), float(radius)) for theta, radius in zip(node_table["theta"], node_table["radius"])]
    node_table["x"] = [float(item[0]) for item in xy]
    node_table["y"] = [float(item[1]) for item in xy]

    metabolite_edge_colors = _hue_wheel_color_series(len(metabolite_nodes), hue_start=18.0, lightness=63.0, safety=0.92)
    metabolite_edge_color_map = {metabolite: metabolite_edge_colors[idx] for idx, metabolite in enumerate(metabolite_nodes)}

    nodes = []
    for row in node_table.itertuples(index=False):
        nodes.append(
            {
                "id": str(row.Node),
                "name": str(row.Node),
                "type": str(row.NodeType),
                "module": str(getattr(row, "Module", "")),
                "module_color": str(row.ModuleColor) if pd.notna(row.ModuleColor) else "#9ca3af",
                "x": _finite_float(row.x),
                "y": _finite_float(row.y),
                "theta": _finite_float(row.theta),
                "ring_radius": _finite_float(row.radius),
                "node_radius": _finite_float(row.node_radius),
                "edge_count": int(_finite_float(row.EdgeCount)),
                "mean_zscore": _finite_float(row.MeanZScore),
                "weighted_degree": _finite_float(row.WeightedDegree),
                "direction_bias": _finite_float(row.DirectionBias),
                "kme": _finite_float(row.kME),
            }
        )

    raw_edges = _edge_payloads(edge_df, layout)
    edges = [
        {
            **edge,
            "color": metabolite_edge_color_map.get(str(edge["target"]), "#9ca3af"),
        }
        for edge in sorted(raw_edges, key=lambda edge: (edge["target"], edge["source"], -edge["weight"]))
    ]

    return {
        "type": "cnet",
        "nodes": nodes,
        "edges": edges,
        "metabolite_edge_color_map": metabolite_edge_color_map,
        "legend": [
            {"label": "Gene node", "color": "#9ca3af"},
            {"label": "Metabolite node", "color": "#c9ad85"},
            {"label": "Metabolite-colored edge", "color": "#6b7280"},
        ],
    }


def export_circos(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export F29/F30 network layouts for the shared interactive circos page."""
    engine = context.engine

    from ..static.network import _attach_circos_module_annotations, _prepare_circos_node_tables

    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return None

    gene_summary, module_color_map = _attach_circos_module_annotations(engine, gene_summary)
    compressed = _build_compressed_layout(engine, context.cfg, edge_df, gene_summary, metabolite_summary)
    cnet = _build_cnet_layout(edge_df, gene_summary, metabolite_summary)
    if compressed is None and cnet is None:
        return None

    is_cnet = "floating" in prefix_key.lower() or "cnet" in prefix_key.lower()
    default_layout = "cnet" if is_cnet else "circos"
    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)

    return {
        "figure_id": f"{default_layout}_network",
        "title": "Circos / CNet Network",
        "chart_type": "circos",
        "interactive_page_id": "circos",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "circos_data": {
            "layouts": {
                "circos": compressed,
                "cnet": cnet,
            },
            "module_color_map": module_color_map,
            "edge_palette": {
                "positive": PALETTE["edge_positive"],
                "negative": PALETTE["edge_negative"],
            },
        },
        "default_state": {"layout": default_layout},
        "available_states": {"layout": ["circos", "cnet"]},
        "style": _base_style(),
    }


__all__ = [
    "export_circos",
]
