from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..utils import safe_mkdir
from .context import VisualizationContext
from .static.base import PALETTE as STATIC_PALETTE
from .static.base import (
    _gene_expression_df,
    _global_secondary_group_color_map,
    _group_color_map,
    _group_marker_map,
    _hue_wheel_color_series,
    _metabolomics_df,
    _ordered_unique_with_order,
)
from .static.network import (
    _attach_circos_module_annotations,
    _build_circos_module_color_map,
    _positive_scale,
    _prepare_circos_node_tables,
    _prepare_group1_mean_track_data,
    _prepare_metabolite_module_core_map,
    _robust_abs_scale,
)
from .static.pca import _compute_pca_result, _load_pca_group_table
from .static.regression import _module_annotation_maps
from .static.module import _coerce_module_eigengene_df, _module_order_from_summary


PALETTE = {
    "gene": STATIC_PALETTE["gene"],
    "metabolite": STATIC_PALETTE["metabolite"],
    "edge_positive": STATIC_PALETTE["edge_positive"],
    "edge_negative": STATIC_PALETTE["edge_negative"],
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Index, pd.Series)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def _json_script_payload(data: Any) -> str:
    return _json_dumps(data).replace("</", "<\\/")


def _ordered_unique(values: list[str]) -> list[str]:
    return _ordered_unique_with_order(values, None)


@dataclass(frozen=True)
class ControlSpec:
    id: str
    type: str
    label: str
    default: Any
    options: list[dict[str, Any]] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class InteractiveViewSpec:
    id: str
    title: str
    kind: str
    schema_id: str
    enabled: bool = True
    description: str = ""
    data_key: str = ""


@dataclass(frozen=True)
class InteractiveReportModel:
    meta: dict[str, Any]
    views: tuple[InteractiveViewSpec, ...]
    schemas: dict[str, Any]
    datasets: dict[str, Any]
    initial_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_summary_payload(engine, cfg) -> dict[str, Any]:
    total_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    high_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    return {
        "projectName": str(cfg.project_name),
        "samples": int(engine.adata.n_obs),
        "genes": int(engine.adata.n_vars),
        "metabolites": int(len(engine.adata.uns.get("metabolite_names", []))),
        "totalEdges": int(len(total_df)) if isinstance(total_df, pd.DataFrame) else 0,
        "highConfidenceEdges": int(len(high_df)) if isinstance(high_df, pd.DataFrame) else 0,
    }


def _coerce_association_source_df(engine, adata, getter_name: str, fallback_loader) -> pd.DataFrame:
    if hasattr(engine, getter_name):
        try:
            df = getattr(engine, getter_name)()
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame) or df.empty:
        df = fallback_loader(adata)

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    work = df.copy(deep=False)
    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work.columns = pd.Index(work.columns.astype(str).str.strip(), name=work.columns.name)
    work = work.loc[work.index.astype(str).str.len() > 0, work.columns.astype(str).str.len() > 0].copy()
    work = work.loc[~work.index.duplicated(keep="first"), ~work.columns.duplicated(keep="first")].copy()
    return work.apply(pd.to_numeric, errors="coerce")


def _build_group_annotation_payload(
    sample_names: list[str],
    group_df: pd.DataFrame | None,
) -> dict[str, Any]:
    sample_names = [str(name).strip() for name in sample_names]
    if not sample_names:
        return {
            "sampleAnnotations": [],
            "groupOptions": {"group1": {"order": [], "colors": {}}, "group2": {"order": [], "colors": {}}},
            "hasSecondaryGrouping": False,
        }

    if group_df is None or not isinstance(group_df, pd.DataFrame) or group_df.empty:
        annotations = [
            {
                "id": sample_name,
                "group1": "Missing",
                "group2": "Missing",
                "group1Color": "#6b7280",
                "group2Color": "#6b7280",
            }
            for sample_name in sample_names
        ]
        return {
            "sampleAnnotations": annotations,
            "groupOptions": {"group1": {"order": [], "colors": {}}, "group2": {"order": [], "colors": {}}},
            "hasSecondaryGrouping": False,
        }

    required_columns = {"sample_id", "group1", "group2"}
    if not required_columns.issubset(group_df.columns):
        annotations = [
            {
                "id": sample_name,
                "group1": "Missing",
                "group2": "Missing",
                "group1Color": "#6b7280",
                "group2Color": "#6b7280",
            }
            for sample_name in sample_names
        ]
        return {
            "sampleAnnotations": annotations,
            "groupOptions": {"group1": {"order": [], "colors": {}}, "group2": {"order": [], "colors": {}}},
            "hasSecondaryGrouping": False,
        }

    keep_columns = [col for col in ["sample_id", "group1", "group2", "_group_table_order"] if col in group_df.columns]
    work = group_df.loc[:, keep_columns].copy()
    work["sample_id"] = work["sample_id"].astype(str).str.strip()
    work["group1"] = work["group1"].astype("string").str.strip().replace("", pd.NA)
    work["group2"] = work["group2"].astype("string").str.strip().replace("", pd.NA)
    if "_group_table_order" not in work.columns:
        work["_group_table_order"] = np.arange(len(work), dtype=int)
    work["_group_table_order"] = pd.to_numeric(work["_group_table_order"], errors="coerce")
    valid_mask = work["sample_id"].ne("") & work["group1"].notna() & work["group2"].notna()
    work = work.loc[valid_mask].copy()
    if work.empty:
        annotations = [
            {
                "id": sample_name,
                "group1": "Missing",
                "group2": "Missing",
                "group1Color": "#6b7280",
                "group2Color": "#6b7280",
            }
            for sample_name in sample_names
        ]
        return {
            "sampleAnnotations": annotations,
            "groupOptions": {"group1": {"order": [], "colors": {}}, "group2": {"order": [], "colors": {}}},
            "hasSecondaryGrouping": False,
        }

    work = work.sort_values("_group_table_order", kind="mergesort").drop_duplicates(subset=["sample_id"], keep="first")
    work = work.set_index("sample_id", drop=False)
    group_orders = work["_group_table_order"].astype(int).tolist()
    group1_order = _ordered_unique_with_order(work["group1"].astype(str).tolist(), group_orders)
    group1_color_map = _group_color_map(group1_order)
    group2_order, group2_color_map = _global_secondary_group_color_map(
        work["group2"].astype(str).tolist(),
        group_orders,
    )

    annotations: list[dict[str, Any]] = []
    for sample_name in sample_names:
        if sample_name in work.index:
            row = work.loc[sample_name]
            group1 = str(row["group1"])
            group2 = str(row["group2"])
        else:
            group1 = "Missing"
            group2 = "Missing"
        annotations.append(
            {
                "id": sample_name,
                "group1": group1,
                "group2": group2,
                "group1Color": group1_color_map.get(group1, "#6b7280"),
                "group2Color": group2_color_map.get(group2, "#6b7280"),
            }
        )

    return {
        "sampleAnnotations": annotations,
        "groupOptions": {
            "group1": {"order": group1_order, "colors": group1_color_map},
            "group2": {"order": group2_order, "colors": group2_color_map},
        },
        "hasSecondaryGrouping": bool(work["group2"].notna().any()),
    }


def _safe_stats_from_vectors(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    result: dict[str, float | None] = {
        "pearsonR": None,
        "pearsonP": None,
        "spearmanRho": None,
        "spearmanP": None,
        "sampleCount": int(x.size),
    }
    if x.size < 2 or y.size < 2:
        return result

    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return result

    try:
        pearson_stat = pearsonr(x, y)
        result["pearsonR"] = float(pearson_stat.statistic)
        result["pearsonP"] = float(pearson_stat.pvalue)
    except Exception:
        pass

    try:
        spearman_stat = spearmanr(x, y)
        if hasattr(spearman_stat, "correlation"):
            result["spearmanRho"] = float(spearman_stat.correlation)
            result["spearmanP"] = float(spearman_stat.pvalue)
        else:
            rho, pvalue = spearman_stat
            result["spearmanRho"] = float(rho)
            result["spearmanP"] = float(pvalue)
    except Exception:
        pass

    return result


def _numeric_matrix_payload(df: pd.DataFrame, columns: list[str]) -> dict[str, list[float | None]]:
    payload: dict[str, list[float | None]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float, copy=False)
        payload[str(column)] = [float(value) if np.isfinite(value) else None for value in values.tolist()]
    return payload


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _build_pca_payload(matrix, sample_names, title: str, cfg, group_df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    values = matrix.to_numpy(dtype=float, copy=False) if isinstance(matrix, pd.DataFrame) else np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None

    sample_names = [str(name) for name in sample_names]
    if len(sample_names) != values.shape[0]:
        return None

    X = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    pca_result = _compute_pca_result(
        X,
        sample_names,
        title,
        cfg,
        group_df=group_df,
        max_components=5,
    )
    if pca_result is None:
        return None

    coords = np.asarray(pca_result["coords"], dtype=float)
    var_exp = np.asarray(pca_result["var_exp"], dtype=float)
    if coords.shape[1] < 2 or var_exp.size < 2:
        return None
    component_count = min(5, int(coords.shape[1]), int(var_exp.size))

    plot_group_df = pca_result.get("plot_group_df")
    plot_sample_names = [str(name) for name in pca_result["plot_sample_names"]]
    points: list[dict[str, Any]] = []

    if isinstance(plot_group_df, pd.DataFrame) and not plot_group_df.empty and "sample_id" in plot_group_df.columns:
        group_table = plot_group_df.copy()
        group_table["sample_id"] = group_table["sample_id"].astype(str).str.strip()
        group_table = group_table.set_index("sample_id", drop=False)
        group_orders = group_table["_group_table_order"].astype(int).tolist() if "_group_table_order" in group_table.columns else None
        group1_order = _ordered_unique_with_order(group_table["group1"].astype(str).tolist(), group_orders)
        group1_color_map = _group_color_map(group1_order)
        group1_marker_map = _group_marker_map(group1_order)
        group2_order, group2_color_map = _global_secondary_group_color_map(
            group_table["group2"].astype(str).tolist(),
            group_orders,
        )

        for idx, sample_name in enumerate(plot_sample_names):
            row = group_table.loc[sample_name] if sample_name in group_table.index else None
            group1 = str(row["group1"]) if row is not None else "Missing"
            group2 = str(row["group2"]) if row is not None else "Missing"
            points.append(
                {
                    "id": sample_name,
                    "label": sample_name,
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "components": [float(coords[idx, comp_idx]) for comp_idx in range(component_count)],
                    "group1": group1,
                    "group2": group2,
                    "group1Color": group1_color_map.get(group1, "#6b7280"),
                    "group1Marker": group1_marker_map.get(group1, "circle"),
                    "group2Color": group2_color_map.get(group2, "#6b7280"),
                }
            )
    else:
        group1_order = []
        group2_order = []
        group1_color_map = {}
        group1_marker_map = {}
        group2_color_map = {}
        for idx, sample_name in enumerate(plot_sample_names):
            points.append(
                {
                    "id": sample_name,
                    "label": sample_name,
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "components": [float(coords[idx, comp_idx]) for comp_idx in range(component_count)],
                    "group1": "Missing",
                    "group2": "Missing",
                    "group1Color": "#6b7280",
                    "group1Marker": "circle",
                    "group2Color": "#6b7280",
                }
            )

    return {
        "id": title.lower().replace(" ", "_"),
        "title": title,
        "kind": "pca",
        "sampleCount": len(points),
        "componentCount": component_count,
        "varianceExplained": {
            **{f"pc{idx + 1}": float(var_exp[idx]) for idx in range(component_count)},
            "components": [float(var_exp[idx]) for idx in range(component_count)],
        },
        "points": points,
        "groupOptions": {
            "group1": {
                "order": group1_order if "group1_order" in locals() else [],
                "colors": group1_color_map if "group1_color_map" in locals() else {},
                "markers": group1_marker_map if "group1_marker_map" in locals() else {},
            },
            "group2": {
                "order": group2_order if "group2_order" in locals() else [],
                "colors": group2_color_map if "group2_color_map" in locals() else {},
            },
        },
    }


def _build_association_payload(engine, cfg, tier: str, group_df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    if tier == "high_confidence":
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
        title = "High-Confidence Association Scatter"
    else:
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
        title = "Total Association Scatter"

    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None

    required_columns = {"Gene", "Metabolite", "EdgeWeight", "ModelSupportCount", "ScreenSupportCount"}
    if not required_columns.issubset(edge_df.columns):
        return None

    context = VisualizationContext.from_engine(engine, cfg, group_df=group_df)
    source_adata = getattr(engine, "adata", context.pca_adata)
    gene_df = _coerce_association_source_df(engine, source_adata, "gene_expression_df", _gene_expression_df)
    metab_df = _coerce_association_source_df(engine, source_adata, "metabolomics_df", _metabolomics_df)
    if gene_df.empty or metab_df.empty:
        return None

    sample_names = [sample for sample in gene_df.index.astype(str).tolist() if sample in set(metab_df.index.astype(str))]
    if len(sample_names) < 2:
        return None
    gene_df = gene_df.reindex(sample_names)
    metab_df = metab_df.reindex(sample_names)
    sample_annotations_payload = _build_group_annotation_payload(sample_names, context.pca_group_df)

    ranked = edge_df.copy()
    ranked["Gene"] = ranked["Gene"].astype(str).str.strip()
    ranked["Metabolite"] = ranked["Metabolite"].astype(str).str.strip()
    ranked["EdgeWeight"] = pd.to_numeric(ranked["EdgeWeight"], errors="coerce").fillna(0.0)
    ranked["ModelSupportCount"] = pd.to_numeric(ranked["ModelSupportCount"], errors="coerce").fillna(0).astype(int)
    ranked["ScreenSupportCount"] = pd.to_numeric(ranked["ScreenSupportCount"], errors="coerce").fillna(0).astype(int)
    if "RRARank" in ranked.columns:
        ranked["RRARank"] = pd.to_numeric(ranked["RRARank"], errors="coerce")
    if "PearsonR" in ranked.columns:
        ranked["PearsonR"] = pd.to_numeric(ranked["PearsonR"], errors="coerce")
    if "PearsonP" in ranked.columns:
        ranked["PearsonP"] = pd.to_numeric(ranked["PearsonP"], errors="coerce")
    if "SpearmanRho" in ranked.columns:
        ranked["SpearmanRho"] = pd.to_numeric(ranked["SpearmanRho"], errors="coerce")
    if "SpearmanP" in ranked.columns:
        ranked["SpearmanP"] = pd.to_numeric(ranked["SpearmanP"], errors="coerce")
    if "RRAWeight" in ranked.columns:
        ranked["RRAWeight"] = pd.to_numeric(ranked["RRAWeight"], errors="coerce")

    sort_columns = ["EdgeWeight"]
    ascending = [False]
    if "RRARank" in ranked.columns:
        sort_columns.append("RRARank")
        ascending.append(True)
    sort_columns.extend(["ModelSupportCount", "ScreenSupportCount", "Gene", "Metabolite"])
    ascending.extend([False, False, True, True])
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort")

    max_edges = max(1, int(getattr(cfg, "network_plot_top_edges", 120)))
    ranked = ranked.head(max_edges).copy()
    if ranked.empty:
        return None

    gene_order = (
        ranked.assign(_Weight=ranked["EdgeWeight"].abs())
        .sort_values(["_Weight", "Gene"], ascending=[False, True], kind="mergesort")["Gene"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    metabolite_order = (
        ranked.assign(_Weight=ranked["EdgeWeight"].abs())
        .sort_values(["_Weight", "Metabolite"], ascending=[False, True], kind="mergesort")["Metabolite"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    top_edges: list[dict[str, Any]] = []
    for row in ranked.itertuples(index=False):
        gene = str(row.Gene)
        metabolite = str(row.Metabolite)
        if gene not in gene_df.columns or metabolite not in metab_df.columns:
            continue

        gene_values = pd.to_numeric(gene_df[gene], errors="coerce").to_numpy(dtype=float, copy=False)
        metab_values = pd.to_numeric(metab_df[metabolite], errors="coerce").to_numpy(dtype=float, copy=False)
        stats = _safe_stats_from_vectors(gene_values, metab_values)
        finite_mask = np.isfinite(gene_values) & np.isfinite(metab_values)
        if int(np.sum(finite_mask)) < 2:
            continue

        x_values = [float(value) if np.isfinite(value) else None for value in gene_values.tolist()]
        y_values = [float(value) if np.isfinite(value) else None for value in metab_values.tolist()]

        top_edges.append(
            {
                "id": f"{gene}||{metabolite}",
                "gene": gene,
                "metabolite": metabolite,
                "label": f"{gene} vs {metabolite}",
                "edgeWeight": float(row.EdgeWeight),
                "modelSupportCount": int(row.ModelSupportCount),
                "screenSupportCount": int(row.ScreenSupportCount),
                "pearsonR": float(row.PearsonR) if hasattr(row, "PearsonR") and pd.notna(getattr(row, "PearsonR")) else stats["pearsonR"],
                "pearsonP": float(row.PearsonP) if hasattr(row, "PearsonP") and pd.notna(getattr(row, "PearsonP")) else stats["pearsonP"],
                "spearmanRho": float(row.SpearmanRho) if hasattr(row, "SpearmanRho") and pd.notna(getattr(row, "SpearmanRho")) else stats["spearmanRho"],
                "spearmanP": float(row.SpearmanP) if hasattr(row, "SpearmanP") and pd.notna(getattr(row, "SpearmanP")) else stats["spearmanP"],
                "rraRank": int(getattr(row, "RRARank", 0)) if hasattr(row, "RRARank") and pd.notna(getattr(row, "RRARank")) else None,
                "rraWeight": float(getattr(row, "RRAWeight", np.nan)) if hasattr(row, "RRAWeight") and pd.notna(getattr(row, "RRAWeight")) else None,
                "sign": str(getattr(row, "Sign", "")).strip(),
                "edgeTier": str(getattr(row, "EdgeTier", tier)),
                "sampleCount": int(stats["sampleCount"]),
                "x": x_values,
                "y": y_values,
            }
        )

    if not top_edges:
        return None

    default_edge = top_edges[0]
    return {
        "id": f"associations.{tier}",
        "title": title,
        "kind": "gene_metabolite",
        "tier": tier,
        "sampleIds": sample_names,
        "sampleAnnotations": sample_annotations_payload["sampleAnnotations"],
        "groupOptions": sample_annotations_payload["groupOptions"],
        "hasSecondaryGrouping": sample_annotations_payload["hasSecondaryGrouping"],
        "geneOptions": [{"value": gene, "label": gene} for gene in gene_order],
        "metaboliteOptions": [{"value": metab, "label": metab} for metab in metabolite_order],
        "topEdges": top_edges,
        "defaultSelection": {
            "edgeId": default_edge["id"],
            "gene": default_edge["gene"],
            "metabolite": default_edge["metabolite"],
        },
    }


def _build_gene_metabolite_regression_payload(engine, cfg) -> dict[str, Any] | None:
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return None

    required_columns = {"Gene", "Metabolite", "EdgeWeight"}
    if not required_columns.issubset(edge_df.columns):
        return None

    gene_df = _coerce_association_source_df(engine, engine.adata, "gene_expression_df", _gene_expression_df)
    metab_df = _coerce_association_source_df(engine, engine.adata, "metabolomics_df", _metabolomics_df)
    if gene_df.empty or metab_df.empty:
        return None

    sample_names = [sample for sample in gene_df.index.astype(str).tolist() if sample in set(metab_df.index.astype(str))]
    if len(sample_names) < 2:
        return None
    gene_df = gene_df.reindex(sample_names)
    metab_df = metab_df.reindex(sample_names)

    ranked = edge_df.copy()
    ranked["Gene"] = ranked["Gene"].astype(str).str.strip()
    ranked["Metabolite"] = ranked["Metabolite"].astype(str).str.strip()
    ranked["EdgeWeight"] = pd.to_numeric(ranked["EdgeWeight"], errors="coerce").fillna(0.0)
    if "RRARank" in ranked.columns:
        ranked["RRARank"] = pd.to_numeric(ranked["RRARank"], errors="coerce")
    else:
        ranked["RRARank"] = np.nan
    ranked = ranked.loc[ranked["Gene"].ne("") & ranked["Metabolite"].ne("")].copy()
    if ranked.empty:
        return None

    ranked = ranked.sort_values(["EdgeWeight", "RRARank", "Gene", "Metabolite"], ascending=[False, True, True, True], kind="mergesort")
    gene_to_module, gene_to_color, _module_to_color = _module_annotation_maps(engine)

    gene_order = [gene for gene in _ordered_unique(ranked["Gene"].astype(str).tolist()) if gene in gene_df.columns]
    metabolite_order = [metabolite for metabolite in _ordered_unique(ranked["Metabolite"].astype(str).tolist()) if metabolite in metab_df.columns]
    if not gene_order or not metabolite_order:
        return None

    edge_lookup: dict[str, dict[str, Any]] = {}
    for row in ranked.itertuples(index=False):
        gene = str(row.Gene)
        metabolite = str(row.Metabolite)
        if gene not in gene_df.columns or metabolite not in metab_df.columns:
            continue
        pair_id = f"gene||{gene}||{metabolite}"
        if pair_id in edge_lookup:
            continue
        edge_lookup[pair_id] = {
            "id": pair_id,
            "gene": gene,
            "metabolite": metabolite,
            "label": f"{gene} vs {metabolite}",
            "edgeWeight": float(row.EdgeWeight),
            "rraRank": int(getattr(row, "RRARank", 0)) if hasattr(row, "RRARank") and pd.notna(getattr(row, "RRARank")) else None,
        }

    default_gene = gene_order[0]
    default_metabolite = metabolite_order[0]
    return {
        "id": "regression.gene_metabolite",
        "title": "Gene-Metabolite Regression",
        "kind": "gene_metabolite",
        "sampleIds": sample_names,
        "topEdges": list(edge_lookup.values()),
        "geneOptions": [{"value": gene, "label": gene} for gene in gene_order],
        "metaboliteOptions": [{"value": metabolite, "label": metabolite} for metabolite in metabolite_order],
        "xMatrix": _numeric_matrix_payload(gene_df, gene_order),
        "yMatrix": _numeric_matrix_payload(metab_df, metabolite_order),
        "geneModules": {
            gene: {
                "module": gene_to_module.get(gene, "Unassigned"),
                "color": gene_to_color.get(gene, PALETTE["gene"]),
            }
            for gene in gene_order
        },
        "defaultSelection": {
            "edgeId": f"gene||{default_gene}||{default_metabolite}",
            "gene": default_gene,
            "metabolite": default_metabolite,
        },
    }


def _build_module_metabolite_regression_payload(engine, cfg) -> dict[str, Any] | None:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return None

    metab_df = _coerce_association_source_df(engine, engine.adata, "metabolomics_df", _metabolomics_df)
    if metab_df.empty:
        return None

    _gene_to_module, _gene_to_color, module_to_color = _module_annotation_maps(engine)
    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order = [module for module in module_order if module in eigengenes_df.columns]
    if not module_order:
        return None

    sample_names = [sample for sample in eigengenes_df.index.astype(str).tolist() if sample in set(metab_df.index.astype(str))]
    if len(sample_names) < 2:
        return None
    eigengenes_df = eigengenes_df.reindex(sample_names)
    metab_df = metab_df.reindex(sample_names)

    assoc_df = _get_module_metabolite_association_df(engine)
    metabolite_order: list[str] = []
    pair_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(assoc_df, pd.DataFrame) and not assoc_df.empty and {"Module", "Metabolite"}.issubset(assoc_df.columns):
        assoc_work = assoc_df.copy()
        assoc_work["Module"] = assoc_work["Module"].astype(str).str.strip()
        assoc_work["Metabolite"] = assoc_work["Metabolite"].astype(str).str.strip()
        if "SpearmanRho" in assoc_work.columns:
            assoc_work["SpearmanRho"] = pd.to_numeric(assoc_work["SpearmanRho"], errors="coerce")
            assoc_work["_AbsRho"] = assoc_work["SpearmanRho"].abs()
        else:
            assoc_work["SpearmanRho"] = np.nan
            assoc_work["_AbsRho"] = 0.0
        assoc_work = assoc_work.loc[
            assoc_work["Module"].isin(module_order)
            & assoc_work["Metabolite"].isin(metab_df.columns.astype(str))
        ].copy()
        if not assoc_work.empty:
            assoc_work["_ModuleOrder"] = assoc_work["Module"].map({module: idx for idx, module in enumerate(module_order)}).fillna(len(module_order)).astype(int)
            assoc_work = assoc_work.sort_values(["_ModuleOrder", "_AbsRho", "Metabolite"], ascending=[True, False, True], kind="mergesort")
            metabolite_order = _ordered_unique(assoc_work["Metabolite"].astype(str).tolist())
            for row in assoc_work.itertuples(index=False):
                module_name = str(row.Module)
                metabolite = str(row.Metabolite)
                pair_lookup[f"module||{module_name}||{metabolite}"] = {
                    "id": f"module||{module_name}||{metabolite}",
                    "module": module_name,
                    "gene": module_name,
                    "metabolite": metabolite,
                    "label": f"{module_name} module vs {metabolite}",
                    "spearmanRho": float(row.SpearmanRho) if pd.notna(row.SpearmanRho) else None,
                }

    metabolite_order = _ordered_unique([*metabolite_order, *metab_df.columns.astype(str).tolist()])
    metabolite_order = [metabolite for metabolite in metabolite_order if metabolite in metab_df.columns]
    if not metabolite_order:
        return None

    fallback_colors = _build_circos_module_color_map(module_order)
    default_module = module_order[0]
    default_metabolite = metabolite_order[0]
    return {
        "id": "regression.module_metabolite",
        "title": "Module-Metabolite Regression",
        "kind": "module_metabolite",
        "sampleIds": sample_names,
        "topEdges": list(pair_lookup.values()),
        "geneOptions": [{"value": module, "label": module} for module in module_order],
        "metaboliteOptions": [{"value": metabolite, "label": metabolite} for metabolite in metabolite_order],
        "xMatrix": _numeric_matrix_payload(eigengenes_df, module_order),
        "yMatrix": _numeric_matrix_payload(metab_df, metabolite_order),
        "moduleColors": {
            module: module_to_color.get(module, fallback_colors.get(module, "#9ca3af"))
            for module in module_order
        },
        "defaultSelection": {
            "edgeId": f"module||{default_module}||{default_metabolite}",
            "gene": default_module,
            "metabolite": default_metabolite,
        },
    }


def _significance_star(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    value = float(value)
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    if value <= 0.05:
        return "*"
    return ""


def _get_module_metabolite_association_df(engine) -> pd.DataFrame:
    for key in (
        "module_metabolite_association_df",
        "module_metabolite_assoc_df",
        "module_metabolite_association",
    ):
        df = getattr(engine, key, None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        df = engine.ml_results.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def _build_module_heatmap_payload(engine, cfg) -> dict[str, Any] | None:
    assoc_df = _get_module_metabolite_association_df(engine)
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return None

    required_columns = {"Module", "Metabolite", "SpearmanRho"}
    if not required_columns.issubset(assoc_df.columns):
        return None

    work = assoc_df.copy()
    work["Module"] = work["Module"].astype(str).str.strip()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work["SpearmanRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce")
    if "FDR" in work.columns:
        work["FDR"] = pd.to_numeric(work["FDR"], errors="coerce")
    else:
        work["FDR"] = np.nan
    if "PValue" in work.columns:
        work["PValue"] = pd.to_numeric(work["PValue"], errors="coerce")
    else:
        work["PValue"] = np.nan

    work = work.loc[work["Module"].ne("") & work["Metabolite"].ne("") & work["SpearmanRho"].notna()].copy()
    if work.empty:
        return None

    non_grey_df = work.loc[work["Module"].str.lower() != "grey"].copy()
    if not non_grey_df.empty:
        work = non_grey_df

    significance_column = "FDR" if work["FDR"].notna().any() else "PValue"
    work["_Significance"] = pd.to_numeric(work[significance_column], errors="coerce")
    work["_SigRank"] = work["_Significance"].fillna(1.0)
    work["_AbsRho"] = work["SpearmanRho"].abs()

    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    module_order: list[str] = []
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        available_modules = set(work["Module"].astype(str).tolist())
        module_order = [
            str(module_name)
            for module_name in module_summary_df["Module"].astype(str).tolist()
            if str(module_name) in available_modules
        ]

    if not module_order:
        module_order = (
            work.sort_values(
                ["_SigRank", "_AbsRho", "Module"],
                ascending=[True, False, True],
                kind="mergesort",
            )["Module"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
    else:
        for module_name in work["Module"].astype(str).drop_duplicates().tolist():
            if module_name not in module_order:
                module_order.append(module_name)

    metabolite_order = (
        work.sort_values(
            ["_SigRank", "_AbsRho", "Metabolite"],
            ascending=[True, False, True],
            kind="mergesort",
        )["Metabolite"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    module_rank = {module_name: idx for idx, module_name in enumerate(module_order)}
    metabolite_rank = {metabolite_name: idx for idx, metabolite_name in enumerate(metabolite_order)}

    module_metrics: list[dict[str, Any]] = []
    for module_name in module_order:
        sub_df = work.loc[work["Module"].astype(str).eq(module_name)]
        if sub_df.empty:
            continue
        min_sig = float(sub_df["_SigRank"].min()) if sub_df["_SigRank"].notna().any() else None
        max_abs_rho = float(sub_df["_AbsRho"].max()) if sub_df["_AbsRho"].notna().any() else 0.0
        module_metrics.append(
            {
                "id": module_name,
                "label": module_name,
                "defaultRank": int(module_rank.get(module_name, len(module_rank))),
                "maxAbsRho": max_abs_rho,
                "minSignificance": min_sig,
            }
        )

    metabolite_metrics: list[dict[str, Any]] = []
    for metabolite_name in metabolite_order:
        sub_df = work.loc[work["Metabolite"].astype(str).eq(metabolite_name)]
        if sub_df.empty:
            continue
        min_sig = float(sub_df["_SigRank"].min()) if sub_df["_SigRank"].notna().any() else None
        max_abs_rho = float(sub_df["_AbsRho"].max()) if sub_df["_AbsRho"].notna().any() else 0.0
        metabolite_metrics.append(
            {
                "id": metabolite_name,
                "label": metabolite_name,
                "defaultRank": int(metabolite_rank.get(metabolite_name, len(metabolite_rank))),
                "maxAbsRho": max_abs_rho,
                "minSignificance": min_sig,
            }
        )

    cells: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        fdr_value = float(getattr(row, "FDR")) if pd.notna(getattr(row, "FDR")) else None
        p_value = float(getattr(row, "PValue")) if pd.notna(getattr(row, "PValue")) else None
        sig_value = fdr_value if significance_column == "FDR" else p_value
        cells.append(
            {
                "module": str(row.Module),
                "metabolite": str(row.Metabolite),
                "rho": float(row.SpearmanRho),
                "fdr": fdr_value,
                "pValue": p_value,
                "significance": sig_value,
                "star": _significance_star(sig_value),
            }
        )

    finite_rho = work["SpearmanRho"].to_numpy(dtype=float, copy=False)
    finite_rho = finite_rho[np.isfinite(finite_rho)]
    vmax = float(np.nanmax(np.abs(finite_rho))) if finite_rho.size else 1.0
    vmax = max(vmax, 0.25)

    default_top_modules = min(max(1, len(module_metrics)), 20)
    default_top_metabolites = min(max(1, len(metabolite_metrics)), 30)
    return {
        "id": "module_heatmap",
        "title": "Module-Metabolite Association Heatmap",
        "kind": "module_heatmap",
        "significanceMetric": significance_column,
        "rhoExtent": {"min": -vmax, "max": vmax},
        "modules": module_metrics,
        "metabolites": metabolite_metrics,
        "cells": cells,
        "defaults": {
            "topModules": default_top_modules,
            "topMetabolites": default_top_metabolites,
        },
    }


def _build_network_payload(
    engine,
    cfg,
    tier: str,
    max_edges: int,
    default_top_edges: int | None = None,
) -> dict[str, Any] | None:
    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return None

    module_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    if isinstance(module_df, pd.DataFrame) and not module_df.empty and "IsGrey" not in module_df.columns:
        module_df = module_df.copy()
        if "Module" in module_df.columns:
            module_df["IsGrey"] = module_df["Module"].astype(str).str.lower().eq("grey").astype(int)
        else:
            module_df["IsGrey"] = 1
        engine.ml_results["gene_module_assignment_df"] = module_df

    title = "T03 High-Confidence Network Explorer"
    gene_summary, _module_color_map = _attach_circos_module_annotations(engine, gene_summary)
    metabolite_module_core = _prepare_metabolite_module_core_map(engine)
    metabolite_summary = metabolite_summary.copy()
    metabolite_summary["Module"] = ""
    metabolite_summary["ModuleColor"] = "#c9ad85"
    metabolite_summary["ModuleCore"] = metabolite_summary["Node"].map(metabolite_module_core).astype(float)
    gene_summary["ModuleCore"] = pd.to_numeric(gene_summary.get("kME", np.nan), errors="coerce").abs()

    circos_track_adata = getattr(engine, "plot_adata", getattr(engine, "unaggregated_adata", engine.adata))
    group_df = _load_pca_group_table(cfg)
    gene_track_data = _prepare_group1_mean_track_data(_gene_expression_df(circos_track_adata), group_df)
    metabolite_track_data = _prepare_group1_mean_track_data(_metabolomics_df(circos_track_adata), group_df)

    gene_mean_scale = _robust_abs_scale(gene_summary["MeanZScore"])
    metabolite_mean_scale = _robust_abs_scale(metabolite_summary["MeanZScore"])
    gene_degree_scale = _positive_scale(gene_summary["WeightedDegree"])
    metabolite_degree_scale = _positive_scale(metabolite_summary["WeightedDegree"])
    gene_core_scale = _positive_scale(gene_summary["ModuleCore"])
    metabolite_core_scale = _positive_scale(metabolite_summary["ModuleCore"])

    def _track_values(track_data: dict[str, object] | None, node_id: str) -> list[float]:
        if track_data is None:
            return []
        values = dict(track_data.get("feature_to_values", {})).get(str(node_id), [])
        return [float(value) for value in values if np.isfinite(float(value))]

    genes: list[dict[str, Any]] = []
    for row in gene_summary.itertuples(index=False):
        node = str(row.Node)
        genes.append(
            {
                "id": f"gene:{node}",
                "label": node,
                "type": "gene",
                "degree": int(getattr(row, "PositiveEdgeCount", 0)) + int(getattr(row, "NegativeEdgeCount", 0)),
                "weightedDegree": float(row.WeightedDegree),
                "maxAbsWeight": float(row.WeightedDegree),
                "positiveEdges": int(row.PositiveEdgeCount),
                "negativeEdges": int(row.NegativeEdgeCount),
                "directionBias": float(row.DirectionBias),
                "meanZScore": float(row.MeanZScore),
                "interSampleVariability": float(row.InterSampleVariability),
                "module": str(row.Module),
                "moduleColor": str(row.ModuleColor),
                "moduleSize": int(row.ModuleSize) if pd.notna(row.ModuleSize) else 0,
                "kME": _float_or_none(getattr(row, "kME", None)),
                "intramodularDegree": _float_or_none(getattr(row, "IntramodularDegree", None)),
                "moduleCore": _float_or_none(getattr(row, "ModuleCore", None)),
                "color": str(row.ModuleColor),
                "track2Values": _track_values(gene_track_data, node),
            }
        )

    metabolites: list[dict[str, Any]] = []
    for row in metabolite_summary.itertuples(index=False):
        node = str(row.Node)
        metabolites.append(
            {
                "id": f"metabolite:{node}",
                "label": node,
                "type": "metabolite",
                "degree": int(getattr(row, "PositiveEdgeCount", 0)) + int(getattr(row, "NegativeEdgeCount", 0)),
                "weightedDegree": float(row.WeightedDegree),
                "maxAbsWeight": float(row.WeightedDegree),
                "positiveEdges": int(row.PositiveEdgeCount),
                "negativeEdges": int(row.NegativeEdgeCount),
                "directionBias": float(row.DirectionBias),
                "meanZScore": float(row.MeanZScore),
                "interSampleVariability": float(row.InterSampleVariability),
                "module": "",
                "moduleColor": "#c9ad85",
                "moduleCore": _float_or_none(getattr(row, "ModuleCore", None)),
                "color": "#c9ad85",
                "edgeColor": "#8c6d46",
                "track2Values": _track_values(metabolite_track_data, node),
            }
        )

    metabolite_edge_colors = _hue_wheel_color_series(len(metabolites), hue_start=18.0, lightness=63.0, safety=0.92)
    metabolite_color_map = {str(item["label"]): metabolite_edge_colors[idx] for idx, item in enumerate(metabolites)}
    for metabolite in metabolites:
        metabolite["edgeColor"] = metabolite_color_map.get(str(metabolite["label"]), "#9ca3af")

    edges: list[dict[str, Any]] = []
    edge_ordered = edge_df.sort_values(["EdgeWeight", "ModelSupportCount", "Gene", "Metabolite"], ascending=[True, True, True, True], kind="mergesort")
    for idx, row in enumerate(edge_ordered.itertuples(index=False)):
        gene = str(row.Gene)
        metabolite = str(row.Metabolite)
        edge_weight = float(np.clip(getattr(row, "EdgeWeight", 0.0), 0.0, None))
        sign = str(row.Sign).strip().lower()
        if sign not in {"positive", "negative"}:
            sign = "positive"
        edges.append(
            {
                "id": f"{gene}||{metabolite}||{idx}",
                "source": f"gene:{gene}",
                "target": f"metabolite:{metabolite}",
                "gene": gene,
                "metabolite": metabolite,
                "edgeWeight": edge_weight,
                "absWeight": edge_weight,
                "sign": sign,
                "edgeTier": "high_confidence",
                "modelSupportCount": int(getattr(row, "ModelSupportCount", 0)),
                "screenSupportCount": int(getattr(row, "ScreenSupportCount", 0)) if hasattr(row, "ScreenSupportCount") else 0,
                "rraRank": int(row.RRARank) if hasattr(row, "RRARank") and pd.notna(row.RRARank) else None,
                "rraWeight": _float_or_none(getattr(row, "RRAWeight", None)),
                "pearsonR": _float_or_none(getattr(row, "PearsonR", None)),
                "spearmanRho": _float_or_none(getattr(row, "SpearmanRho", None)),
                "metaboliteColor": metabolite_color_map.get(metabolite, "#9ca3af"),
            }
        )

    if not edges:
        return None

    nodes = genes + metabolites

    return {
        "id": "network.high_confidence",
        "title": title,
        "tier": "high_confidence",
        "layout": "circos",
        "nodes": nodes,
        "edges": edges,
        "nodeGroups": {
            "geneModules": [
                {"module": module_name, "color": module_color}
                for module_name, module_color in {
                    str(value["module"]): str(value["moduleColor"])
                    for value in genes
                    if str(value.get("module", "")).strip()
                }.items()
            ],
            "metaboliteColors": metabolite_color_map,
        },
        "geneOptions": [{"value": item["label"], "label": item["label"]} for item in genes],
        "metaboliteOptions": [{"value": item["label"], "label": item["label"]} for item in metabolites],
        "defaults": {
            "topEdges": len(edges),
            "minEdgeWeight": 0.0,
        },
        "trackScales": {
            "geneMean": float(gene_mean_scale),
            "metaboliteMean": float(metabolite_mean_scale),
            "geneDegree": float(gene_degree_scale),
            "metaboliteDegree": float(metabolite_degree_scale),
            "geneCore": float(gene_core_scale),
            "metaboliteCore": float(metabolite_core_scale),
            "geneTrack2": float(gene_track_data.get("abs_scale", gene_mean_scale)) if gene_track_data is not None else float(gene_mean_scale),
            "metaboliteTrack2": float(metabolite_track_data.get("abs_scale", metabolite_mean_scale)) if metabolite_track_data is not None else float(metabolite_mean_scale),
        },
        "track2": {
            "geneMode": str(gene_track_data.get("mode", "")) if gene_track_data is not None else "",
            "metaboliteMode": str(metabolite_track_data.get("mode", "")) if metabolite_track_data is not None else "",
            "group1Order": [str(value) for value in gene_track_data.get("group1_order", [])] if gene_track_data is not None else [],
            "group1Colors": {str(key): str(value) for key, value in dict(gene_track_data.get("group1_color_map", {})).items()} if gene_track_data is not None else {},
        },
        "summary": {
            "nodes": len(nodes),
            "genes": len(genes),
            "metabolites": len(metabolites),
            "edges": len(edges),
            "maxAbsWeight": float(max(edge["absWeight"] for edge in edges)),
        },
    }


def _build_pca_schema(default_dataset: str) -> dict[str, Any]:
    return {
        "id": "pca.scatter",
        "title": "PCA controls",
        "controls": [
            {
                "id": "dataset",
                "type": "select",
                "label": "Dataset",
                "default": default_dataset,
                "options": [
                    {"value": "transcriptome", "label": "Transcriptome"},
                    {"value": "metabolome", "label": "Metabolome"},
                ],
            },
            {
                "id": "colorBy",
                "type": "select",
                "label": "Color",
                "default": "group1",
                "options": [
                    {"value": "group1", "label": "Group 1"},
                    {"value": "group2", "label": "Group 2"},
                ],
            },
            {
                "id": "xComponent",
                "type": "select",
                "label": "X component",
                "default": 1,
                "options": [
                    {"value": 1, "label": "PC1"},
                    {"value": 2, "label": "PC2"},
                    {"value": 3, "label": "PC3"},
                    {"value": 4, "label": "PC4"},
                    {"value": 5, "label": "PC5"},
                ],
            },
            {
                "id": "yComponent",
                "type": "select",
                "label": "Y component",
                "default": 2,
                "options": [
                    {"value": 1, "label": "PC1"},
                    {"value": 2, "label": "PC2"},
                    {"value": 3, "label": "PC3"},
                    {"value": 4, "label": "PC4"},
                    {"value": 5, "label": "PC5"},
                ],
            },
            {
                "id": "showGroupEnvelope",
                "type": "toggle",
                "label": "Group envelope",
                "default": True,
            },
            {
                "id": "pointSize",
                "type": "range",
                "label": "Point size",
                "default": 5,
                "min": 2,
                "max": 14,
                "step": 0.5,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 900,
                "min": 640,
                "max": 1800,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 620,
                "min": 480,
                "max": 1400,
                "step": 20,
            },
        ],
    }


def _build_association_schema(default_pair_type: str, default_gene: str, default_metabolite: str) -> dict[str, Any]:
    return {
        "id": "association.scatter",
        "title": "Association Scatter Studio",
        "controls": [
            {
                "id": "pairType",
                "type": "select",
                "label": "Type",
                "default": default_pair_type,
                "options": [
                    {"value": "gene_metabolite", "label": "Gene-metabolite"},
                    {"value": "module_metabolite", "label": "Module-metabolite"},
                ],
            },
            {
                "id": "topEdgeId",
                "type": "select",
                "label": "Pair",
                "default": "",
                "optionsSource": "topEdges",
                "allowEmpty": True,
                "emptyLabel": "Custom pair",
            },
            {
                "id": "gene",
                "type": "select",
                "label": "Gene / module",
                "default": default_gene,
                "optionsSource": "geneOptions",
            },
            {
                "id": "metabolite",
                "type": "select",
                "label": "Metabolite",
                "default": default_metabolite,
                "optionsSource": "metaboliteOptions",
            },
            {
                "id": "pointSize",
                "type": "range",
                "label": "Point size",
                "default": 5,
                "min": 2,
                "max": 14,
                "step": 0.5,
            },
            {
                "id": "alpha",
                "type": "range",
                "label": "Opacity",
                "default": 0.85,
                "min": 0.15,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "showRegression",
                "type": "toggle",
                "label": "Regression line",
                "default": True,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 900,
                "min": 640,
                "max": 2000,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 640,
                "min": 480,
                "max": 1800,
                "step": 20,
            },
        ],
    }


def _build_module_heatmap_schema(default_top_modules: int, default_top_metabolites: int) -> dict[str, Any]:
    return {
        "id": "module.heatmap",
        "title": "Module Heatmap Studio",
        "controls": [
            {
                "id": "topModules",
                "type": "number",
                "label": "Top modules",
                "default": int(default_top_modules),
                "min": 1,
                "max": 200,
                "step": 1,
            },
            {
                "id": "topMetabolites",
                "type": "number",
                "label": "Top metabolites",
                "default": int(default_top_metabolites),
                "min": 1,
                "max": 300,
                "step": 1,
            },
            {
                "id": "palette",
                "type": "select",
                "label": "Palette",
                "default": "rdbu",
                "options": [
                    {"value": "rdbu", "label": "Red-Blue"},
                    {"value": "blueorange", "label": "Blue-Orange"},
                    {"value": "purplegreen", "label": "Purple-Green"},
                ],
            },
            {
                "id": "showValues",
                "type": "toggle",
                "label": "Values",
                "default": False,
            },
            {
                "id": "showStars",
                "type": "toggle",
                "label": "Stars",
                "default": True,
            },
            {
                "id": "rowSort",
                "type": "select",
                "label": "Rows",
                "default": "default",
                "options": [
                    {"value": "default", "label": "Module summary order"},
                    {"value": "max_abs_rho", "label": "Max |rho|"},
                    {"value": "significance", "label": "Significance"},
                    {"value": "name", "label": "Name"},
                ],
            },
            {
                "id": "columnSort",
                "type": "select",
                "label": "Columns",
                "default": "significance",
                "options": [
                    {"value": "significance", "label": "Significance"},
                    {"value": "max_abs_rho", "label": "Max |rho|"},
                    {"value": "name", "label": "Name"},
                    {"value": "default", "label": "Default"},
                ],
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 980,
                "min": 720,
                "max": 2400,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 720,
                "min": 520,
                "max": 2000,
                "step": 20,
            },
        ],
    }


def _build_network_schema(default_top_edges: int) -> dict[str, Any]:
    return {
        "id": "network.explorer",
        "title": "Network Explorer",
        "controls": [
            {
                "id": "layout",
                "type": "select",
                "label": "Layout",
                "default": "circos",
                "options": [
                    {"value": "circos", "label": "Circos"},
                    {"value": "cnet", "label": "CNet"},
                ],
            },
            {
                "id": "nodeSize",
                "type": "range",
                "label": "Node size",
                "default": 7,
                "min": 4,
                "max": 18,
                "step": 0.5,
            },
            {
                "id": "showLabels",
                "type": "toggle",
                "label": "Labels",
                "default": False,
            },
            {
                "id": "width",
                "type": "number",
                "label": "Width",
                "default": 1100,
                "min": 760,
                "max": 2400,
                "step": 20,
            },
            {
                "id": "height",
                "type": "number",
                "label": "Height",
                "default": 760,
                "min": 520,
                "max": 2000,
                "step": 20,
            },
        ],
    }


def _build_placeholder_schema(schema_id: str, title: str) -> dict[str, Any]:
    return {
        "id": schema_id,
        "title": title,
        "controls": [],
    }


def _build_interactive_report_model(engine, cfg) -> InteractiveReportModel:
    context = VisualizationContext.from_engine(engine, cfg)
    summary = _build_summary_payload(engine, cfg)
    group_df = context.pca_group_df

    transcriptome_payload = _build_pca_payload(
        context.pca_adata.X,
        context.pca_adata.obs_names.astype(str).tolist(),
        "Transcriptome PCA",
        cfg,
        group_df=group_df,
    )
    metab_source = context.pca_adata.obsm.get("metabolomics_scaled", context.pca_adata.obsm.get("metabolomics"))
    metabolome_payload = _build_pca_payload(
        metab_source,
        context.pca_adata.obs_names.astype(str).tolist(),
        "Metabolome PCA",
        cfg,
        group_df=group_df,
    )
    association_high_payload = _build_association_payload(engine, cfg, "high_confidence", group_df=group_df)
    association_total_payload = _build_association_payload(engine, cfg, "total", group_df=group_df)
    gene_metabolite_payload = _build_gene_metabolite_regression_payload(engine, cfg)
    module_metabolite_payload = _build_module_metabolite_regression_payload(engine, cfg)
    module_heatmap_payload = _build_module_heatmap_payload(engine, cfg)
    network_high_payload = _build_network_payload(
        engine,
        cfg,
        "high_confidence",
        max_edges=0,
        default_top_edges=None,
    )

    datasets = {
        "pca.transcriptome": transcriptome_payload,
        "pca.metabolome": metabolome_payload,
        "association.high_confidence": association_high_payload,
        "association.total": association_total_payload,
        "association.gene_metabolite": gene_metabolite_payload,
        "association.module_metabolite": module_metabolite_payload,
        "module_heatmap": module_heatmap_payload,
        "network.high_confidence": network_high_payload,
    }

    association_default = gene_metabolite_payload or module_metabolite_payload
    association_default_gene = ""
    association_default_metabolite = ""
    if association_default is not None and association_default.get("defaultSelection"):
        association_default_gene = str(association_default["defaultSelection"].get("gene", ""))
        association_default_metabolite = str(association_default["defaultSelection"].get("metabolite", ""))

    views = (
        InteractiveViewSpec(
            id="pca",
            title="PCA Explorer",
            kind="pca",
            schema_id="pca.scatter",
            enabled=transcriptome_payload is not None or metabolome_payload is not None,
            description="Transcriptome and metabolome PCA scatter view.",
            data_key="pca.transcriptome",
        ),
        InteractiveViewSpec(
            id="association",
            title="Association Scatter Studio",
            kind="association",
            schema_id="association.scatter",
            enabled=association_default is not None,
            description="Sample-level regression scatter for gene-metabolite and module-metabolite pairs.",
            data_key="association.gene_metabolite",
        ),
        InteractiveViewSpec(
            id="module_heatmap",
            title="Module Heatmap Studio",
            kind="module_heatmap",
            schema_id="module.heatmap",
            enabled=module_heatmap_payload is not None,
            description="Interactive module-metabolite Spearman association heatmap.",
            data_key="module_heatmap",
        ),
        InteractiveViewSpec(
            id="network_explorer",
            title="Network Explorer",
            kind="network",
            schema_id="network.explorer",
            enabled=network_high_payload is not None,
            description="Bipartite gene-metabolite association network explorer.",
            data_key="network.high_confidence",
        ),
    )

    if transcriptome_payload is not None:
        default_dataset = "transcriptome"
    elif metabolome_payload is not None:
        default_dataset = "metabolome"
    else:
        default_dataset = "transcriptome"
    schemas = {
        "pca.scatter": _build_pca_schema(default_dataset),
        "association.scatter": _build_association_schema(
            "gene_metabolite" if gene_metabolite_payload is not None else "module_metabolite",
            association_default_gene,
            association_default_metabolite,
        ),
        "module.heatmap": _build_module_heatmap_schema(
            int(module_heatmap_payload["defaults"]["topModules"]) if module_heatmap_payload is not None else 10,
            int(module_heatmap_payload["defaults"]["topMetabolites"]) if module_heatmap_payload is not None else 20,
        ),
        "network.explorer": _build_network_schema(
            int((network_high_payload or {"defaults": {"topEdges": 120}})["defaults"]["topEdges"]),
        ),
        "placeholder.network_explorer": _build_placeholder_schema("placeholder.network_explorer", "Network Explorer"),
    }

    pca_defaults = {
        "dataset": default_dataset,
        "colorBy": "group1",
        "xComponent": 1,
        "yComponent": 2,
        "showGroupEnvelope": True,
        "pointSize": 5,
        "showLabels": False,
        "width": 900,
        "height": 620,
    }

    initial_state = {
        "activeViewId": "association" if association_default is not None else "pca",
        "controls": {
            "pca": pca_defaults,
            "association": {
                "pairType": "gene_metabolite" if gene_metabolite_payload is not None else "module_metabolite",
                "topEdgeId": "",
                "gene": association_default_gene,
                "metabolite": association_default_metabolite,
                "pointSize": 5,
                "alpha": 0.85,
                "showLabels": False,
                "showRegression": True,
                "width": 900,
                "height": 640,
            },
            "module_heatmap": {
                "topModules": int(module_heatmap_payload["defaults"]["topModules"]) if module_heatmap_payload is not None else 10,
                "topMetabolites": int(module_heatmap_payload["defaults"]["topMetabolites"]) if module_heatmap_payload is not None else 20,
                "palette": "rdbu",
                "showValues": False,
                "showStars": True,
                "rowSort": "default",
                "columnSort": "significance",
                "width": 980,
                "height": 720,
            },
            "network_explorer": {
                "layout": "circos",
                "nodeSize": 7,
                "showLabels": False,
                "width": 1100,
                "height": 760,
                "selectedNodeId": "",
            },
        },
    }

    implemented_views = ["pca", "association"]
    placeholder_views = []
    if module_heatmap_payload is not None:
        implemented_views.append("module_heatmap")
    else:
        placeholder_views.append("module_heatmap")
    if network_high_payload is not None:
        implemented_views.append("network_explorer")
    else:
        placeholder_views.append("network_explorer")

    meta = {
        **summary,
        "implementedViews": implemented_views,
        "placeholderViews": placeholder_views,
        "groupTableLoaded": bool(group_df is not None and not group_df.empty),
        "secondaryGrouping": bool(group_df is not None and "group2" in group_df.columns and group_df["group2"].notna().any()),
    }

    return InteractiveReportModel(
        meta=meta,
        views=views,
        schemas=schemas,
        datasets=datasets,
        initial_state=initial_state,
    )


def _interactive_html_template() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepOmics Interactive Report - __PROJECT_NAME__</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d7dde5;
      --border-strong: #b8c1cc;
      --text: #111827;
      --muted: #5b6472;
      --accent: #2563eb;
      --accent-soft: #e8eefc;
      --disabled: #cbd5e1;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: #fbfcfe;
      padding: 20px 16px;
    }
    .brand {
      font-size: 20px;
      font-weight: 700;
      line-height: 1.2;
      margin: 0 0 8px 0;
    }
    .subtle {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0 18px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #1e3a8a;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid #c7d2fe;
    }
    .nav {
      display: grid;
      gap: 8px;
      margin-top: 16px;
    }
    .nav button {
      width: 100%;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      box-shadow: var(--shadow);
      font-size: 13px;
      line-height: 1.3;
    }
    .nav button.active {
      border-color: #93c5fd;
      background: #eff6ff;
    }
    .nav button.pending {
      color: var(--disabled);
      background: #f8fafc;
      box-shadow: none;
    }
    .main {
      padding: 20px 20px 24px;
      min-width: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow);
    }
    .panel + .panel {
      margin-top: 16px;
    }
    .panel-head {
      padding: 16px 16px 10px;
      border-bottom: 1px solid var(--border);
    }
    .panel-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .panel-note {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      padding: 14px 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .control {
      min-width: 0;
      display: grid;
      gap: 6px;
    }
    .control label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
    .control input[type="number"],
    .control input[type="text"],
    .control input[type="range"],
    .control select {
      width: 100%;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
      font-size: 13px;
    }
    .control input[type="range"] {
      padding: 8px 0;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
    }
    .toggle-row input {
      width: 16px;
      height: 16px;
    }
    .action-bar {
      padding: 0 16px 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .action-bar button {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
    }
    .action-bar button:hover {
      border-color: #93c5fd;
    }
    .chart-wrap {
      padding: 16px;
      overflow: auto;
    }
    .chart-shell {
      position: relative;
      display: inline-block;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    svg {
      display: block;
      max-width: none;
      background: #fff;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid rgba(15, 23, 42, 0.14);
    }
    .placeholder {
      padding: 24px 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .runtime-error {
      margin: 20px;
      padding: 16px;
      border: 1px solid #fecaca;
      border-radius: 12px;
      background: #fff1f2;
      color: #991b1b;
      font-family: Consolas, monospace;
      white-space: pre-wrap;
    }
    @media (max-width: 1000px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="app" class="app"></div>
  <script id="deepomics-payload" type="application/json">__PAYLOAD__</script>
  <script>
    const report = JSON.parse(document.getElementById("deepomics-payload").textContent);

    const state = {
      activeViewId: report.initial_state.activeViewId,
      controls: JSON.parse(JSON.stringify(report.initial_state.controls || {}))
    };

    const app = document.getElementById("app");

    function renderRuntimeError(error) {
      const message = error && error.stack ? error.stack : String(error);
      const main = el("main", { className: "main" });
      main.appendChild(el("div", { className: "runtime-error", text: message }));
      return main;
    }

    window.addEventListener("error", event => {
      clear(app);
      app.appendChild(renderSidebar());
      app.appendChild(renderRuntimeError(event.error || event.message));
    });

    function el(tag, attrs = {}, children = []) {
      const node = document.createElement(tag);
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null) continue;
        if (key === "className") {
          node.className = value;
        } else if (key === "checked") {
          node.checked = Boolean(value);
        } else if (key === "selected") {
          node.selected = Boolean(value);
        } else if (key === "text") {
          node.textContent = value;
        } else if (key === "html") {
          node.innerHTML = value;
        } else if (key.startsWith("on") && typeof value === "function") {
          node.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
          node.setAttribute(key, String(value));
        }
      }
      for (const child of children) {
        if (child !== null && child !== undefined) node.appendChild(child);
      }
      return node;
    }

    function svgEl(tag, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) {
        if (value !== undefined && value !== null) node.setAttribute(key, String(value));
      }
      return node;
    }

    function clear(node) {
      while (node.firstChild) node.removeChild(node.firstChild);
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function fmtPct(value) {
      return `${Number(value || 0).toFixed(1)}%`;
    }

    function getView(id) {
      return report.views.find(v => v.id === id);
    }

    function getDatasetKeyFromControls() {
      const controls = state.controls.pca || {};
      return controls.dataset === "metabolome" ? "pca.metabolome" : "pca.transcriptome";
    }

    function getActiveDataset() {
      return report.datasets[getDatasetKeyFromControls()] || report.datasets["pca.transcriptome"] || report.datasets["pca.metabolome"] || null;
    }

    function getViewControls(viewId) {
      if (!state.controls[viewId]) state.controls[viewId] = {};
      return state.controls[viewId];
    }

    function getAssociationDataset() {
      const controls = getViewControls("association");
      const pairType = controls.pairType === "module_metabolite" ? "module_metabolite" : "gene_metabolite";
      return report.datasets[`association.${pairType}`] || report.datasets["association.gene_metabolite"] || report.datasets["association.module_metabolite"] || null;
    }

    function getNetworkDataset() {
      return report.datasets["network.high_confidence"] || null;
    }

    function getDatasetForView(viewId) {
      if (viewId === "pca") return getActiveDataset();
      if (viewId === "association") return getAssociationDataset();
      if (viewId === "module_heatmap") return report.datasets.module_heatmap || null;
      if (viewId === "network_explorer") return getNetworkDataset();
      return null;
    }

    function findAssociationEdge(dataset, controls) {
      if (!dataset) return null;
      const topEdgeId = String(controls.topEdgeId || "").trim();
      const geneOptions = Array.isArray(dataset.geneOptions) ? dataset.geneOptions : [];
      const metaboliteOptions = Array.isArray(dataset.metaboliteOptions) ? dataset.metaboliteOptions : [];
      let gene = String(controls.gene || "").trim();
      let metabolite = String(controls.metabolite || "").trim();

      let known = null;
      if (topEdgeId && Array.isArray(dataset.topEdges)) {
        known = dataset.topEdges.find(edge => edge.id === topEdgeId) || null;
        if (known) {
          gene = String(known.gene || known.module || gene).trim();
          metabolite = String(known.metabolite || metabolite).trim();
        }
      }
      if (!gene || !dataset.xMatrix || !Object.prototype.hasOwnProperty.call(dataset.xMatrix, gene)) {
        gene = geneOptions.length ? String(geneOptions[0].value) : "";
      }
      if (!metabolite || !dataset.yMatrix || !Object.prototype.hasOwnProperty.call(dataset.yMatrix, metabolite)) {
        metabolite = metaboliteOptions.length ? String(metaboliteOptions[0].value) : "";
      }
      if (!gene || !metabolite || !dataset.xMatrix?.[gene] || !dataset.yMatrix?.[metabolite]) return null;

      const pairPrefix = dataset.kind === "module_metabolite" ? "module" : "gene";
      const pairId = `${pairPrefix}||${gene}||${metabolite}`;
      if (!known && Array.isArray(dataset.topEdges)) known = dataset.topEdges.find(edge => edge.id === pairId) || null;

      const geneInfo = dataset.geneModules?.[gene] || {};
      const moduleName = dataset.kind === "module_metabolite" ? gene : (geneInfo.module || "Unassigned");
      const moduleColor = dataset.kind === "module_metabolite"
        ? (dataset.moduleColors?.[gene] || "#9ca3af")
        : (geneInfo.color || "#4c78a8");
      return {
        ...(known || {}),
        id: pairId,
        kind: dataset.kind,
        gene,
        module: moduleName,
        metabolite,
        label: dataset.kind === "module_metabolite" ? `${gene} module vs ${metabolite}` : `${gene} vs ${metabolite}`,
        xLabel: dataset.kind === "module_metabolite" ? `${gene} module eigengene` : gene,
        yLabel: metabolite,
        moduleColor,
        pointColor: moduleColor,
        rLabel: dataset.kind === "module_metabolite" ? "rho" : "r",
        rValue: null,
        x: dataset.xMatrix[gene],
        y: dataset.yMatrix[metabolite],
      };
    }

    function syncAssociationControlsFromDataset(controls, dataset) {
      if (!dataset) {
        controls.topEdgeId = "";
        controls.gene = "";
        controls.metabolite = "";
        return;
      }

      const current = findAssociationEdge(dataset, controls);
      if (!current) return;
      const known = Array.isArray(dataset.topEdges) ? dataset.topEdges.find(edge => edge.id === current.id) : null;
      controls.topEdgeId = known ? current.id : "";
      controls.gene = current.gene;
      controls.metabolite = current.metabolite;
    }

    function setControl(viewId, key, value) {
      const controls = getViewControls(viewId);
      controls[key] = value;

      if (viewId === "association") {
        const dataset = getAssociationDataset();
        if (key === "pairType") {
          syncAssociationControlsFromDataset(controls, dataset);
        } else if (key === "topEdgeId") {
          const edge = findAssociationEdge(dataset, controls);
          if (edge) {
            controls.topEdgeId = Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
            controls.gene = edge.gene;
            controls.metabolite = edge.metabolite;
          }
        } else if (key === "gene") {
          controls.topEdgeId = "";
          const edge = findAssociationEdge(dataset, controls);
          controls.topEdgeId = edge && Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
          if (edge) controls.metabolite = edge.metabolite;
        } else if (key === "metabolite") {
          controls.topEdgeId = "";
          const edge = findAssociationEdge(dataset, controls);
          controls.topEdgeId = edge && Array.isArray(dataset.topEdges) && dataset.topEdges.find(item => item.id === edge.id) ? edge.id : "";
          if (edge) controls.gene = edge.gene;
        }
      } else if (viewId === "network_explorer") {
        if (key !== "selectedNodeId") {
          controls.selectedNodeId = "";
        }
      }

      render();
    }

    function resetControls(viewId) {
      state.controls[viewId] = JSON.parse(JSON.stringify(report.initial_state.controls[viewId] || {}));
      if (viewId === "association") {
        const dataset = getAssociationDataset();
        syncAssociationControlsFromDataset(state.controls[viewId], dataset);
      }
      render();
    }

    function downloadSvg(svgNode, filename) {
      const clone = svgNode.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      const text = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([text], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1200);
    }

    function renderSidebar() {
      const sidebar = el("aside", { className: "sidebar" });
      sidebar.appendChild(el("h1", { className: "brand", text: report.meta.projectName || "DeepOmics" }));
      sidebar.appendChild(el("p", { className: "subtle", text: "Offline interactive report" }));

      const chips = el("div", { className: "chips" }, [
        el("span", { className: "chip", text: `Samples: ${report.meta.samples}` }),
        el("span", { className: "chip", text: `Genes: ${report.meta.genes}` }),
        el("span", { className: "chip", text: `Metabolites: ${report.meta.metabolites}` })
      ]);
      sidebar.appendChild(chips);

      const nav = el("div", { className: "nav" });
      for (const view of report.views) {
        const button = el("button", {
          className: [view.id === state.activeViewId ? "active" : "", !view.enabled ? "pending" : ""].filter(Boolean).join(" "),
          text: view.title,
          onclick: () => {
            state.activeViewId = view.id;
            render();
          }
        });
        nav.appendChild(button);
      }
      sidebar.appendChild(nav);
      return sidebar;
    }

    function getControlOptions(view, control) {
      if (!control.optionsSource) return control.options || [];
      const dataset = getDatasetForView(view.id);
      if (!dataset) return [];

      let options = [];
      if (control.optionsSource === "topEdges") {
        options = (dataset.topEdges || []).map(edge => ({
          value: edge.id,
          label: edge.label || `${edge.gene} vs ${edge.metabolite}`,
        }));
      } else if (control.optionsSource === "geneOptions") {
        options = dataset.geneOptions || [];
      } else if (control.optionsSource === "metaboliteOptions") {
        options = dataset.metaboliteOptions || [];
      }

      if (control.allowEmpty) {
        options = [{ value: "", label: control.emptyLabel || "None" }, ...options];
      }
      return options;
    }

    function renderControlField(view, control) {
      const current = getViewControls(view.id)[control.id];
      const wrapper = el("div", { className: "control" });
      wrapper.appendChild(el("label", { text: control.label }));

      if (control.type === "toggle") {
        const input = el("input", {
          type: "checkbox",
          onchange: event => setControl(view.id, control.id, event.target.checked)
        });
        input.checked = Boolean(current);
        wrapper.appendChild(el("div", { className: "toggle-row" }, [input, el("span", { text: control.description || "" })]));
        return wrapper;
      }

      if (control.type === "range") {
        const input = el("input", {
          type: "range",
          min: control.min,
          max: control.max,
          step: control.step,
          value: current ?? control.default,
          oninput: event => setControl(view.id, control.id, Number(event.target.value))
        });
        const value = el("span", { text: String(current ?? control.default) });
        const shell = el("div", { className: "toggle-row" }, [input, value]);
        input.addEventListener("input", () => { value.textContent = String(input.value); });
        wrapper.appendChild(shell);
        return wrapper;
      }

      if (control.type === "number") {
        const input = el("input", {
          type: "number",
          min: control.min,
          max: control.max,
          step: control.step,
          value: current ?? control.default,
          oninput: event => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) setControl(view.id, control.id, next);
          }
        });
        wrapper.appendChild(input);
        return wrapper;
      }

      if (control.type === "text") {
        const input = el("input", {
          type: "text",
          value: current ?? control.default ?? "",
          oninput: event => setControl(view.id, control.id, event.target.value)
        });
        wrapper.appendChild(input);
        return wrapper;
      }

      const select = el("select", {
        onchange: event => setControl(view.id, control.id, event.target.value)
      });
      for (const option of getControlOptions(view, control)) {
        const opt = el("option", {
          value: option.value,
          text: option.label || option.value
        });
        opt.selected = String(current ?? control.default) === String(option.value);
        select.appendChild(opt);
      }
      wrapper.appendChild(select);
      return wrapper;
    }

    function resolveAssociationControlDefaults(dataset, controls) {
      if (!dataset) {
        return;
      }
      syncAssociationControlsFromDataset(controls, dataset);
    }

    function renderPcaLegend(dataset, colorBy) {
      const legend = el("div", { className: "legend" });
      const group1Info = dataset.groupOptions?.group1;
      const group2Info = dataset.groupOptions?.group2;
      if (!group1Info || !Array.isArray(group1Info.order) || group1Info.order.length === 0) {
        legend.appendChild(el("span", { text: "No group legend available." }));
        return legend;
      }
      if (colorBy === "group2" && group2Info && Array.isArray(group2Info.order) && group2Info.order.length > 0) {
        const group2Colors = group2Info.colors || {};
        legend.appendChild(el("span", { className: "legend-item", text: "Color: Group 2" }));
        for (const groupName of group2Info.order) {
          const swatch = el("span", { className: "swatch", style: `background:${group2Colors[groupName] || "#6b7280"}` });
          legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
        }
        legend.appendChild(el("span", { className: "legend-item", text: "Shape: Group 1" }));
        for (const groupName of group1Info.order) {
          const marker = group1Info.markers?.[groupName] || "circle";
          const icon = svgEl("svg", { width: 16, height: 16, viewBox: "0 0 16 16" });
          icon.appendChild(pcaMarkerNode(marker, 8, 8, 5, "#111827", "#ffffff"));
          legend.appendChild(el("span", { className: "legend-item" }, [icon, el("span", { text: groupName })]));
        }
      } else {
        const group1Colors = group1Info.colors || {};
        legend.appendChild(el("span", { className: "legend-item", text: "Color: Group 1" }));
        for (const groupName of group1Info.order) {
          const swatch = el("span", { className: "swatch", style: `background:${group1Colors[groupName] || "#6b7280"}` });
          legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
        }
      }
      return legend;
    }

    function renderAssociationLegend(dataset, colorBy) {
      const legend = el("div", { className: "legend" });
      const groupInfo = colorBy === "group2" ? dataset.groupOptions.group2 : dataset.groupOptions.group1;
      if (!groupInfo || !Array.isArray(groupInfo.order) || groupInfo.order.length === 0 || colorBy === "none") {
        legend.appendChild(el("span", { text: "No group legend available." }));
        return legend;
      }
      const colors = groupInfo.colors || {};
      for (const groupName of groupInfo.order) {
        const swatch = el("span", { className: "swatch", style: `background:${colors[groupName] || "#6b7280"}` });
        legend.appendChild(el("span", { className: "legend-item" }, [swatch, el("span", { text: groupName })]));
      }
      return legend;
    }

    function pcaMarkerNode(marker, cx, cy, size, fill, stroke) {
      const name = String(marker || "circle").toLowerCase();
      if (name === "s" || name === "square") {
        return svgEl("rect", {
          x: cx - size,
          y: cy - size,
          width: size * 2,
          height: size * 2,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "^" || name === "triangle_up") {
        return svgEl("polygon", {
          points: `${cx},${cy - size * 1.25} ${cx - size * 1.1},${cy + size * 0.85} ${cx + size * 1.1},${cy + size * 0.85}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "v" || name === "triangle_down") {
        return svgEl("polygon", {
          points: `${cx},${cy + size * 1.25} ${cx - size * 1.1},${cy - size * 0.85} ${cx + size * 1.1},${cy - size * 0.85}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "<" || name === "triangle_left") {
        return svgEl("polygon", {
          points: `${cx - size * 1.25},${cy} ${cx + size * 0.85},${cy - size * 1.1} ${cx + size * 0.85},${cy + size * 1.1}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === ">" || name === "triangle_right") {
        return svgEl("polygon", {
          points: `${cx + size * 1.25},${cy} ${cx - size * 0.85},${cy - size * 1.1} ${cx - size * 0.85},${cy + size * 1.1}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "d" || name === "diamond") {
        return svgEl("polygon", {
          points: `${cx},${cy - size * 1.25} ${cx - size * 1.25},${cy} ${cx},${cy + size * 1.25} ${cx + size * 1.25},${cy}`,
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "x") {
        const group = svgEl("g", { stroke: fill, "stroke-width": 2.0, "stroke-linecap": "round" });
        group.appendChild(svgEl("line", { x1: cx - size, y1: cy - size, x2: cx + size, y2: cy + size }));
        group.appendChild(svgEl("line", { x1: cx + size, y1: cy - size, x2: cx - size, y2: cy + size }));
        return group;
      }
      if (name === "p" || name === "pentagon") {
        const points = [];
        for (let idx = 0; idx < 5; idx++) {
          const angle = -Math.PI / 2 + idx * 2 * Math.PI / 5;
          points.push(`${cx + Math.cos(angle) * size * 1.18},${cy + Math.sin(angle) * size * 1.18}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "h" || name === "hexagon" || name === "8") {
        const count = name === "8" ? 8 : 6;
        const points = [];
        for (let idx = 0; idx < count; idx++) {
          const angle = -Math.PI / 2 + idx * 2 * Math.PI / count;
          points.push(`${cx + Math.cos(angle) * size * 1.12},${cy + Math.sin(angle) * size * 1.12}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "*" || name === "star") {
        const points = [];
        for (let idx = 0; idx < 10; idx++) {
          const angle = -Math.PI / 2 + idx * Math.PI / 5;
          const radius = idx % 2 === 0 ? size * 1.35 : size * 0.55;
          points.push(`${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`);
        }
        return svgEl("polygon", {
          points: points.join(" "),
          fill,
          stroke,
          "stroke-width": 1.1
        });
      }
      if (name === "plus") {
        const group = svgEl("g", { stroke: fill, "stroke-width": 2.0, "stroke-linecap": "round" });
        group.appendChild(svgEl("line", { x1: cx - size, y1: cy, x2: cx + size, y2: cy }));
        group.appendChild(svgEl("line", { x1: cx, y1: cy - size, x2: cx, y2: cy + size }));
        return group;
      }
      return svgEl("circle", {
        cx,
        cy,
        r: size,
        fill,
        stroke,
        "stroke-width": 1.1
      });
    }

    function pcaComponentValue(point, componentIndex, fallbackField) {
      const components = Array.isArray(point.components) ? point.components : [];
      const value = Number(components[componentIndex]);
      if (Number.isFinite(value)) return value;
      return Number(point[fallbackField] || 0);
    }

    function pcaVariancePct(dataset, componentIndex) {
      const values = dataset.varianceExplained?.components;
      if (Array.isArray(values) && Number.isFinite(Number(values[componentIndex]))) {
        return Number(values[componentIndex]);
      }
      const key = `pc${componentIndex + 1}`;
      return Number(dataset.varianceExplained?.[key]);
    }

    function pcaEnvelopePath(groupPoints) {
      if (!Array.isArray(groupPoints) || groupPoints.length === 0) return "";
      const points = [...groupPoints].sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
      if (points.length === 1) {
        const p = points[0];
        const r = 16;
        return `M ${p.x - r} ${p.y} a ${r} ${r} 0 1 0 ${r * 2} 0 a ${r} ${r} 0 1 0 ${-r * 2} 0`;
      }
      if (points.length === 2) {
        const [a, b] = points;
        return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
      }
      const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
      const lower = [];
      for (const point of points) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
        lower.push(point);
      }
      const upper = [];
      for (let idx = points.length - 1; idx >= 0; idx--) {
        const point = points[idx];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
        upper.push(point);
      }
      const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
      if (hull.length < 3) return "";
      const centroid = hull.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
      centroid.x /= hull.length;
      centroid.y /= hull.length;
      const padded = hull.map(p => {
        const dx = p.x - centroid.x;
        const dy = p.y - centroid.y;
        const length = Math.sqrt(dx * dx + dy * dy) || 1;
        return { x: p.x + (dx / length) * 12, y: p.y + (dy / length) * 12 };
      });
      return `M ${padded.map(p => `${p.x} ${p.y}`).join(" L ")} Z`;
    }

    function renderPcaChart(dataset, controls) {
      const width = clamp(Number(controls.width || 900), 640, 2000);
      const height = clamp(Number(controls.height || 620), 480, 1800);
      const pointSize = clamp(Number(controls.pointSize || 5), 2, 14);
      const showLabels = Boolean(controls.showLabels);
      const showGroupEnvelope = Boolean(controls.showGroupEnvelope);
      const colorBy = controls.colorBy === "group2" ? "group2" : "group1";
      const points = Array.isArray(dataset.points) ? dataset.points : [];
      const componentCount = Math.max(2, Number(dataset.componentCount || 2));
      let xComponent = clamp(Number(controls.xComponent || 1), 1, componentCount);
      let yComponent = clamp(Number(controls.yComponent || 2), 1, componentCount);
      if (xComponent === yComponent) yComponent = xComponent === 1 ? Math.min(2, componentCount) : 1;
      const xComponentIndex = xComponent - 1;
      const yComponentIndex = yComponent - 1;
      const title = dataset.title || "PCA";
      const margin = { top: 48, right: 38, bottom: 60, left: 72 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);

      const xs = points.map(p => pcaComponentValue(p, xComponentIndex, "x"));
      const ys = points.map(p => pcaComponentValue(p, yComponentIndex, "y"));
      const xmin = Math.min(0, ...xs);
      const xmax = Math.max(0, ...xs);
      const ymin = Math.min(0, ...ys);
      const ymax = Math.max(0, ...ys);
      const xpad = Math.max(0.12 * Math.max(1e-6, xmax - xmin), 0.25);
      const ypad = Math.max(0.12 * Math.max(1e-6, ymax - ymin), 0.25);
      const x0 = xmin - xpad;
      const x1 = xmax + xpad;
      const y0 = ymin - ypad;
      const y1 = ymax + ypad;
      const sx = value => margin.left + ((value - x0) / Math.max(1e-6, x1 - x0)) * innerWidth;
      const sy = value => margin.top + ((y1 - value) / Math.max(1e-6, y1 - y0)) * innerHeight;

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(svgEl("line", {
        x1: margin.left,
        y1: sy(0),
        x2: width - margin.right,
        y2: sy(0),
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));
      svg.appendChild(svgEl("line", {
        x1: sx(0),
        y1: margin.top,
        x2: sx(0),
        y2: height - margin.bottom,
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));

      const axisColor = "#334155";
      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      }));
      svg.lastChild.textContent = `${title}`;

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 13,
        fill: axisColor
      }));
      svg.lastChild.textContent = `PC${xComponent} (${fmtPct(pcaVariancePct(dataset, xComponentIndex))})`;

      svg.appendChild(svgEl("text", {
        x: 20,
        y: height / 2,
        transform: `rotate(-90 20 ${height / 2})`,
        "text-anchor": "middle",
        "font-size": 13,
        fill: axisColor
      }));
      svg.lastChild.textContent = `PC${yComponent} (${fmtPct(pcaVariancePct(dataset, yComponentIndex))})`;

      if (showGroupEnvelope) {
        const envelopeGroups = new Map();
        for (const point of points) {
          const groupName = colorBy === "group2" ? (point.group2 || "Missing") : (point.group1 || "Missing");
          const color = colorBy === "group2" ? (point.group2Color || "#6b7280") : (point.group1Color || "#6b7280");
          const cx = sx(pcaComponentValue(point, xComponentIndex, "x"));
          const cy = sy(pcaComponentValue(point, yComponentIndex, "y"));
          if (!envelopeGroups.has(groupName)) envelopeGroups.set(groupName, { color, points: [] });
          envelopeGroups.get(groupName).points.push({ x: cx, y: cy });
        }
        for (const group of envelopeGroups.values()) {
          const pathData = pcaEnvelopePath(group.points);
          if (!pathData) continue;
          const attrs = {
            d: pathData,
            fill: group.points.length >= 3 ? group.color : "none",
            stroke: group.color,
            "stroke-width": group.points.length >= 3 ? 1.4 : 8,
            opacity: group.points.length >= 3 ? 0.18 : 0.16,
            "stroke-linejoin": "round",
            "stroke-linecap": "round"
          };
          svg.appendChild(svgEl("path", attrs));
          if (group.points.length === 2) {
            svg.appendChild(svgEl("path", {
              d: pathData,
              fill: "none",
              stroke: group.color,
              "stroke-width": 1.3,
              opacity: 0.90,
              "stroke-linecap": "round"
            }));
          }
        }
      }

      for (const point of points) {
        const color = colorBy === "group2" ? (point.group2Color || "#6b7280") : (point.group1Color || "#6b7280");
        const marker = colorBy === "group2" ? (point.group1Marker || "circle") : "circle";
        const cx = sx(pcaComponentValue(point, xComponentIndex, "x"));
        const cy = sy(pcaComponentValue(point, yComponentIndex, "y"));
        const markerNode = pcaMarkerNode(marker, cx, cy, pointSize, color, "#ffffff");
        const markerTitle = svgEl("title");
        markerTitle.textContent = point.id || point.label || "";
        markerNode.appendChild(markerTitle);
        svg.appendChild(markerNode);

        if (showLabels) {
          const dx = cx >= width / 2 ? -8 : 8;
          const anchor = cx >= width / 2 ? "end" : "start";
          const label = svgEl("text", {
            x: cx + dx,
            y: cy - 6,
            "text-anchor": anchor,
            "font-size": 10,
            fill: "#334155"
          });
          label.textContent = point.label || point.id;
          svg.appendChild(label);
        }
      }
      return svg;
    }

    function computeLinearFit(points) {
      const clean = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      if (clean.length < 2) return null;
      const xs = clean.map(p => p.x);
      const ys = clean.map(p => p.y);
      const n = clean.length;
      const xMean = xs.reduce((a, b) => a + b, 0) / n;
      const yMean = ys.reduce((a, b) => a + b, 0) / n;
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      for (let i = 0; i < n; i++) {
        const dx = xs[i] - xMean;
        const dy = ys[i] - yMean;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
      }
      if (sxx <= 0 || syy <= 0) return null;
      const slope = sxy / sxx;
      const intercept = yMean - slope * xMean;
      const pearson = sxy / Math.sqrt(sxx * syy);
      const fitted = xs.map(x => intercept + slope * x);
      const residualSs = ys.reduce((acc, y, idx) => acc + Math.pow(y - fitted[idx], 2), 0);
      const dof = n - 2;
      const residualSe = dof > 0 ? Math.sqrt(residualSs / dof) : null;

      const rankedX = xs.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const rx = new Array(n);
      for (let i = 0; i < rankedX.length; i++) rx[rankedX[i].i] = i + 1;
      const rankedY = ys.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const ry = new Array(n);
      for (let i = 0; i < rankedY.length; i++) ry[rankedY[i].i] = i + 1;
      const rxMean = rx.reduce((a, b) => a + b, 0) / n;
      const ryMean = ry.reduce((a, b) => a + b, 0) / n;
      let rsxx = 0;
      let rsyy = 0;
      let rsxy = 0;
      for (let i = 0; i < n; i++) {
        const dx = rx[i] - rxMean;
        const dy = ry[i] - ryMean;
        rsxx += dx * dx;
        rsyy += dy * dy;
        rsxy += dx * dy;
      }
      const spearman = rsxx > 0 && rsyy > 0 ? rsxy / Math.sqrt(rsxx * rsyy) : null;
      return { slope, intercept, pearson, spearman, xMean, sxx, residualSe, dof };
    }

    function approximateTCritical(dof) {
      const df = Math.max(1, Math.floor(Number(dof || 1)));
      const table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
      };
      if (table[df]) return table[df];
      if (df <= 40) return 2.021;
      if (df <= 60) return 2.000;
      if (df <= 120) return 1.980;
      return 1.960;
    }

    function renderAssociationChart(dataset, controls) {
      const width = clamp(Number(controls.width || 900), 640, 2000);
      const height = clamp(Number(controls.height || 640), 480, 1800);
      const pointSize = clamp(Number(controls.pointSize || 5), 2, 14);
      const alpha = clamp(Number(controls.alpha || 0.85), 0.15, 1.0);
      const showLabels = Boolean(controls.showLabels);
      const showRegression = Boolean(controls.showRegression);
      const selected = findAssociationEdge(dataset, controls) || (Array.isArray(dataset.topEdges) ? dataset.topEdges[0] : null);
      const title = dataset.title || "Association Scatter";
      const margin = { top: 48, right: 38, bottom: 60, left: 72 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);
      const samples = dataset.sampleIds || [];

      if (!selected) {
        return el("div", { className: "placeholder", text: "No regression payload available for the selected type." });
      }

      const xs = selected.x || [];
      const ys = selected.y || [];
      const finiteX = xs.filter(v => Number.isFinite(v));
      const finiteY = ys.filter(v => Number.isFinite(v));
      const xmin = Math.min(0, ...finiteX);
      const xmax = Math.max(0, ...finiteX);
      const ymin = Math.min(0, ...finiteY);
      const ymax = Math.max(0, ...finiteY);
      const xpad = Math.max(0.12 * Math.max(1e-6, xmax - xmin), 0.25);
      const ypad = Math.max(0.12 * Math.max(1e-6, ymax - ymin), 0.25);
      const x0 = xmin - xpad;
      const x1 = xmax + xpad;
      const y0 = ymin - ypad;
      const y1 = ymax + ypad;
      const sx = value => margin.left + ((value - x0) / Math.max(1e-6, x1 - x0)) * innerWidth;
      const sy = value => margin.top + ((y1 - value) / Math.max(1e-6, y1 - y0)) * innerHeight;
      const moduleColor = selected.pointColor || selected.moduleColor || "#4c78a8";

      const points = samples.map((sampleId, idx) => {
        const x = xs[idx] === null || xs[idx] === undefined ? Number.NaN : Number(xs[idx]);
        const y = ys[idx] === null || ys[idx] === undefined ? Number.NaN : Number(ys[idx]);
        return {
          id: sampleId,
          x,
          y
        };
      }).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));

      const stats = computeLinearFit(points);
      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });

      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      svg.appendChild(svgEl("line", {
        x1: margin.left,
        y1: sy(0),
        x2: width - margin.right,
        y2: sy(0),
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));
      svg.appendChild(svgEl("line", {
        x1: sx(0),
        y1: margin.top,
        x2: sx(0),
        y2: height - margin.bottom,
        stroke: "#cbd5e1",
        "stroke-dasharray": "6 4"
      }));

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      }));
      svg.lastChild.textContent = `${title}: ${selected.label || `${selected.gene} vs ${selected.metabolite}`}`;

      svg.appendChild(svgEl("text", {
        x: width / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 13,
        fill: "#334155"
      }));
      svg.lastChild.textContent = selected.xLabel || selected.gene;

      svg.appendChild(svgEl("text", {
        x: 20,
        y: height / 2,
        transform: `rotate(-90 20 ${height / 2})`,
        "text-anchor": "middle",
        "font-size": 13,
        fill: "#334155"
      }));
      svg.lastChild.textContent = selected.yLabel || selected.metabolite;

      if (showRegression && stats) {
        const xStart = x0;
        const xEnd = x1;
        const yStart = stats.intercept + stats.slope * xStart;
        const yEnd = stats.intercept + stats.slope * xEnd;
        if (stats.residualSe !== null && stats.dof > 0 && stats.sxx > 0) {
          const tValue = approximateTCritical(stats.dof);
          const upper = [];
          const lower = [];
          for (let idx = 0; idx < 80; idx++) {
            const xValue = xStart + (xEnd - xStart) * idx / 79;
            const yFit = stats.intercept + stats.slope * xValue;
            const seMean = stats.residualSe * Math.sqrt((1 / points.length) + Math.pow(xValue - stats.xMean, 2) / stats.sxx);
            const delta = tValue * seMean;
            if (!Number.isFinite(delta)) continue;
            upper.push({ x: sx(xValue), y: sy(yFit + delta) });
            lower.push({ x: sx(xValue), y: sy(yFit - delta) });
          }
          if (upper.length > 1 && lower.length > 1) {
            const bandPoints = upper.concat(lower.reverse()).map(p => `${p.x},${p.y}`).join(" ");
            svg.appendChild(svgEl("polygon", {
              points: bandPoints,
              fill: moduleColor,
              opacity: 0.16,
              stroke: "none"
            }));
          }
        }
        svg.appendChild(svgEl("line", {
          x1: sx(xStart),
          y1: sy(yStart),
          x2: sx(xEnd),
          y2: sy(yEnd),
          stroke: "#111827",
          "stroke-width": 1.6
        }));
      }

      const rLabel = selected.rLabel || "r";
      const realtimeR = rLabel === "rho" ? (stats ? stats.spearman : null) : (stats ? stats.pearson : null);
      const rValue = realtimeR !== null && realtimeR !== undefined ? Number(realtimeR) : (
        selected.rValue !== null && selected.rValue !== undefined ? Number(selected.rValue) : null
      );
      const rText = Number.isFinite(rValue) ? `${rLabel} = ${rValue.toFixed(2)}` : `${rLabel} = NA`;
      const rGroup = svgEl("g");
      rGroup.appendChild(svgEl("rect", {
        x: margin.left + 10,
        y: margin.top + 10,
        width: Math.max(58, rText.length * 8 + 14),
        height: 22,
        fill: "#ffffff",
        opacity: 0.75,
        stroke: "none",
        rx: 3
      }));
      const rTextNode = svgEl("text", {
        x: margin.left + 17,
        y: margin.top + 26,
        "font-size": 12,
        "font-weight": 700,
        fill: "#111827"
      });
      rTextNode.textContent = rText;
      rGroup.appendChild(rTextNode);
      svg.appendChild(rGroup);

      for (const point of points) {
        const cx = sx(point.x);
        const cy = sy(point.y);
        const circle = svgEl("circle", {
          cx,
          cy,
          r: pointSize,
          fill: moduleColor,
          opacity: alpha,
          stroke: "#ffffff",
          "stroke-width": 1.0
        });
        circle.dataset.sampleId = point.id;
        svg.appendChild(circle);

        if (showLabels) {
          const dx = cx >= width / 2 ? -8 : 8;
          const anchor = cx >= width / 2 ? "end" : "start";
          const label = svgEl("text", {
            x: cx + dx,
            y: cy - 6,
            "text-anchor": anchor,
            "font-size": 10,
            fill: "#334155"
          });
          label.textContent = point.id;
          svg.appendChild(label);
        }
      }

      const summary = el("div", { className: "legend" });
      const chips = [`Module: ${selected.module || "Unassigned"}`, `Samples: ${points.length}`];
      if (selected.edgeWeight !== undefined && selected.edgeWeight !== null) chips.splice(1, 0, `EdgeWeight: ${Number(selected.edgeWeight).toFixed(3)}`);
      for (const text of chips) summary.appendChild(el("span", { className: "legend-item", text }));

      return { svg, summary, selected };
    }

    function formatSignificanceMetric(metric) {
      return metric === "FDR" ? "FDR" : "PValue";
    }

    function sortHeatmapItems(items, mode) {
      const sorted = [...items];
      if (mode === "max_abs_rho") {
        sorted.sort((a, b) => {
          const delta = Number(b.maxAbsRho || 0) - Number(a.maxAbsRho || 0);
          return delta || String(a.label || a.id).localeCompare(String(b.label || b.id));
        });
      } else if (mode === "significance") {
        sorted.sort((a, b) => {
          const av = a.minSignificance !== null && a.minSignificance !== undefined && Number.isFinite(Number(a.minSignificance)) ? Number(a.minSignificance) : Number.POSITIVE_INFINITY;
          const bv = b.minSignificance !== null && b.minSignificance !== undefined && Number.isFinite(Number(b.minSignificance)) ? Number(b.minSignificance) : Number.POSITIVE_INFINITY;
          return (av - bv) || String(a.label || a.id).localeCompare(String(b.label || b.id));
        });
      } else if (mode === "name") {
        sorted.sort((a, b) => String(a.label || a.id).localeCompare(String(b.label || b.id)));
      } else {
        sorted.sort((a, b) => Number(a.defaultRank || 0) - Number(b.defaultRank || 0));
      }
      return sorted;
    }

    function hexToRgb(hex) {
      const clean = String(hex || "").replace("#", "");
      const value = clean.length === 3
        ? clean.split("").map(ch => ch + ch).join("")
        : clean.padEnd(6, "0").slice(0, 6);
      return {
        r: parseInt(value.slice(0, 2), 16),
        g: parseInt(value.slice(2, 4), 16),
        b: parseInt(value.slice(4, 6), 16)
      };
    }

    function rgbToHex(rgb) {
      const toHex = value => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0");
      return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
    }

    function mixHex(a, b, t) {
      const ac = hexToRgb(a);
      const bc = hexToRgb(b);
      return rgbToHex({
        r: ac.r + (bc.r - ac.r) * t,
        g: ac.g + (bc.g - ac.g) * t,
        b: ac.b + (bc.b - ac.b) * t
      });
    }

    function heatmapColor(value, extent, paletteName) {
      if (!Number.isFinite(value)) return "#f8fafc";
      const palettes = {
        rdbu: ["#2166ac", "#f7f7f7", "#b2182b"],
        blueorange: ["#2563eb", "#f8fafc", "#ea580c"],
        purplegreen: ["#7e22ce", "#f7f7f7", "#15803d"]
      };
      const palette = palettes[paletteName] || palettes.rdbu;
      const maxAbs = Math.max(
        Math.abs(Number(extent?.min ?? -1)),
        Math.abs(Number(extent?.max ?? 1)),
        0.25
      );
      const normalized = clamp(value / maxAbs, -1, 1);
      if (normalized < 0) return mixHex(palette[0], palette[1], normalized + 1);
      return mixHex(palette[1], palette[2], normalized);
    }

    function contrastTextColor(fill) {
      const rgb = hexToRgb(fill);
      const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
      return luminance < 0.54 ? "#ffffff" : "#111827";
    }

    function prepareHeatmapGrid(dataset, controls) {
      const modules = sortHeatmapItems(dataset.modules || [], controls.rowSort || "default")
        .slice(0, clamp(Math.round(Number(controls.topModules || 10)), 1, Math.max(1, (dataset.modules || []).length)));
      const metabolites = sortHeatmapItems(dataset.metabolites || [], controls.columnSort || "significance")
        .slice(0, clamp(Math.round(Number(controls.topMetabolites || 20)), 1, Math.max(1, (dataset.metabolites || []).length)));
      const moduleIds = new Set(modules.map(item => item.id));
      const metaboliteIds = new Set(metabolites.map(item => item.id));
      const cellMap = new Map();
      for (const cell of dataset.cells || []) {
        if (moduleIds.has(cell.module) && metaboliteIds.has(cell.metabolite)) {
          cellMap.set(`${cell.module}||${cell.metabolite}`, cell);
        }
      }
      return { modules, metabolites, cellMap };
    }

    function renderModuleHeatmapChart(dataset, controls) {
      const width = clamp(Number(controls.width || 980), 720, 2400);
      const height = clamp(Number(controls.height || 720), 520, 2000);
      const showValues = Boolean(controls.showValues);
      const showStars = Boolean(controls.showStars);
      const palette = controls.palette || "rdbu";
      const grid = prepareHeatmapGrid(dataset, controls);
      const modules = grid.modules;
      const metabolites = grid.metabolites;
      if (!modules.length || !metabolites.length) return null;

      const rowLabelWidth = Math.min(220, Math.max(92, 8 * Math.max(...modules.map(item => String(item.label || item.id).length)) + 18));
      const colLabelHeight = Math.min(190, Math.max(92, 7 * Math.max(...metabolites.map(item => String(item.label || item.id).length)) + 20));
      const margin = { top: 58, right: 130, bottom: colLabelHeight + 48, left: rowLabelWidth + 22 };
      const innerWidth = Math.max(1, width - margin.left - margin.right);
      const innerHeight = Math.max(1, height - margin.top - margin.bottom);
      const cellSize = Math.max(8, Math.min(innerWidth / metabolites.length, innerHeight / modules.length));
      const gridWidth = cellSize * metabolites.length;
      const gridHeight = cellSize * modules.length;
      const x0 = margin.left;
      const y0 = margin.top;
      const title = dataset.title || "Module-Metabolite Association Heatmap";
      const metricLabel = formatSignificanceMetric(dataset.significanceMetric);

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": title
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));

      const titleText = svgEl("text", {
        x: width / 2,
        y: 24,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      });
      titleText.textContent = title;
      svg.appendChild(titleText);

      for (let rowIndex = 0; rowIndex < modules.length; rowIndex++) {
        const module = modules[rowIndex];
        const y = y0 + rowIndex * cellSize;
        const label = svgEl("text", {
          x: x0 - 8,
          y: y + cellSize / 2 + 4,
          "text-anchor": "end",
          "font-size": 11,
          fill: "#334155"
        });
        label.textContent = module.label || module.id;
        svg.appendChild(label);
      }

      for (let colIndex = 0; colIndex < metabolites.length; colIndex++) {
        const metabolite = metabolites[colIndex];
        const x = x0 + colIndex * cellSize;
        const label = svgEl("text", {
          x: x + cellSize / 2,
          y: y0 + gridHeight + 10,
          transform: `rotate(55 ${x + cellSize / 2} ${y0 + gridHeight + 10})`,
          "text-anchor": "start",
          "font-size": 10,
          fill: "#334155"
        });
        label.textContent = metabolite.label || metabolite.id;
        svg.appendChild(label);
      }

      for (let rowIndex = 0; rowIndex < modules.length; rowIndex++) {
        for (let colIndex = 0; colIndex < metabolites.length; colIndex++) {
          const module = modules[rowIndex];
          const metabolite = metabolites[colIndex];
          const cell = grid.cellMap.get(`${module.id}||${metabolite.id}`);
          const rho = cell ? Number(cell.rho) : NaN;
          const fill = heatmapColor(rho, dataset.rhoExtent, palette);
          const x = x0 + colIndex * cellSize;
          const y = y0 + rowIndex * cellSize;
          const rect = svgEl("rect", {
            x,
            y,
            width: cellSize,
            height: cellSize,
            fill,
            stroke: "#f1f5f9",
            "stroke-width": 0.8
          });
          if (cell) {
            rect.dataset.module = module.id;
            rect.dataset.metabolite = metabolite.id;
            rect.dataset.rho = String(cell.rho);
          }
          svg.appendChild(rect);

          if (cell && (showValues || showStars) && cellSize >= 16) {
            const textParts = [];
            if (showValues) textParts.push(Number(rho).toFixed(2));
            if (showStars && cell.star) textParts.push(cell.star);
            if (textParts.length) {
              const valueText = svgEl("text", {
                x: x + cellSize / 2,
                y: y + cellSize / 2 + 4,
                "text-anchor": "middle",
                "font-size": cellSize < 24 ? 9 : 10,
                "font-weight": cell.star ? 700 : 500,
                fill: contrastTextColor(fill)
              });
              valueText.textContent = textParts.join(" ");
              svg.appendChild(valueText);
            }
          }
        }
      }

      svg.appendChild(svgEl("rect", {
        x: x0,
        y: y0,
        width: gridWidth,
        height: gridHeight,
        fill: "none",
        stroke: "#94a3b8",
        "stroke-width": 1
      }));

      const legendX = x0 + gridWidth + 34;
      const legendY = y0 + 10;
      const legendHeight = Math.min(220, Math.max(140, gridHeight - 20));
      const legendWidth = 14;
      const stops = 80;
      for (let i = 0; i < stops; i++) {
        const t0 = i / stops;
        const rho = Number(dataset.rhoExtent?.max || 1) - t0 * (Number(dataset.rhoExtent?.max || 1) - Number(dataset.rhoExtent?.min || -1));
        svg.appendChild(svgEl("rect", {
          x: legendX,
          y: legendY + t0 * legendHeight,
          width: legendWidth,
          height: legendHeight / stops + 0.8,
          fill: heatmapColor(rho, dataset.rhoExtent, palette)
        }));
      }
      svg.appendChild(svgEl("rect", {
        x: legendX,
        y: legendY,
        width: legendWidth,
        height: legendHeight,
        fill: "none",
        stroke: "#94a3b8",
        "stroke-width": 1
      }));

      const maxLabel = svgEl("text", { x: legendX + 22, y: legendY + 4, "font-size": 11, fill: "#334155" });
      maxLabel.textContent = Number(dataset.rhoExtent?.max || 1).toFixed(2);
      svg.appendChild(maxLabel);
      const zeroLabel = svgEl("text", { x: legendX + 22, y: legendY + legendHeight / 2 + 4, "font-size": 11, fill: "#334155" });
      zeroLabel.textContent = "0";
      svg.appendChild(zeroLabel);
      const minLabel = svgEl("text", { x: legendX + 22, y: legendY + legendHeight + 4, "font-size": 11, fill: "#334155" });
      minLabel.textContent = Number(dataset.rhoExtent?.min || -1).toFixed(2);
      svg.appendChild(minLabel);
      const legendLabel = svgEl("text", {
        x: legendX - 8,
        y: legendY + legendHeight / 2,
        transform: `rotate(-90 ${legendX - 8} ${legendY + legendHeight / 2})`,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      legendLabel.textContent = "Spearman rho";
      svg.appendChild(legendLabel);

      const axisLabel = svgEl("text", {
        x: x0 + gridWidth / 2,
        y: height - 16,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      axisLabel.textContent = "Metabolite";
      svg.appendChild(axisLabel);

      const rowAxisLabel = svgEl("text", {
        x: 18,
        y: y0 + gridHeight / 2,
        transform: `rotate(-90 18 ${y0 + gridHeight / 2})`,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155"
      });
      rowAxisLabel.textContent = "Module";
      svg.appendChild(rowAxisLabel);

      const subtitle = svgEl("text", {
        x: width / 2,
        y: 42,
        "text-anchor": "middle",
        "font-size": 11,
        fill: "#64748b"
      });
      subtitle.textContent = `Stars: ${metricLabel} <= 0.05/0.01/0.001`;
      svg.appendChild(subtitle);

      return { svg, modules, metabolites };
    }

    function renderModuleHeatmapSummary(dataset, rendered) {
      const legend = el("div", { className: "legend" });
      legend.appendChild(el("span", { className: "legend-item", text: `Modules shown: ${rendered.modules.length}/${(dataset.modules || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Metabolites shown: ${rendered.metabolites.length}/${(dataset.metabolites || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Cells: ${(dataset.cells || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Significance: ${formatSignificanceMetric(dataset.significanceMetric)}` }));
      return legend;
    }

    function normalizeSearch(value) {
      return String(value || "").trim().toLowerCase();
    }

    function filterNetworkEdges(dataset, controls) {
      return dataset.edges || [];
    }

    function prepareNetworkGraph(dataset, controls) {
      const edges = filterNetworkEdges(dataset, controls);
      const nodeLookup = new Map((dataset.nodes || []).map(node => [node.id, node]));
      const activeNodeIds = new Set();
      for (const edge of edges) {
        activeNodeIds.add(edge.source);
        activeNodeIds.add(edge.target);
      }
      const genes = (dataset.nodes || [])
        .filter(node => node.type === "gene" && activeNodeIds.has(node.id))
        .sort((a, b) => {
          const aGrey = String(a.module || "grey").toLowerCase() === "grey" ? 1 : 0;
          const bGrey = String(b.module || "grey").toLowerCase() === "grey" ? 1 : 0;
          return aGrey - bGrey
            || Number(b.moduleSize || 0) - Number(a.moduleSize || 0)
            || String(a.module || "").localeCompare(String(b.module || ""))
            || Number(b.degree || 0) - Number(a.degree || 0)
            || Number(b.maxAbsWeight || 0) - Number(a.maxAbsWeight || 0)
            || String(a.label).localeCompare(String(b.label));
        });
      const metabolites = (dataset.nodes || [])
        .filter(node => node.type === "metabolite" && activeNodeIds.has(node.id))
        .sort((a, b) => Number(b.degree || 0) - Number(a.degree || 0) || String(a.label).localeCompare(String(b.label)));
      const adjacency = new Map();
      for (const edge of edges) {
        if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set());
        if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set());
        adjacency.get(edge.source).add(edge.target);
        adjacency.get(edge.target).add(edge.source);
      }
      return { edges, nodes: [...genes, ...metabolites], genes, metabolites, nodeLookup, adjacency };
    }

    function nodeDetailText(node) {
      if (!node) return "";
      const typeLabel = node.type === "gene" ? "Gene" : "Metabolite";
      const lines = [
        `${typeLabel}: ${node.label}`,
        `Degree: ${node.degree}`,
        `Max |EdgeWeight|: ${Number(node.maxAbsWeight || 0).toFixed(3)}`,
        `Positive edges: ${node.positiveEdges || 0}`,
        `Negative edges: ${node.negativeEdges || 0}`
      ];
      if (node.type === "gene") {
        lines.push(`Module: ${node.module || "grey"}`);
        if (node.kME !== null && node.kME !== undefined) lines.push(`kME: ${Number(node.kME).toFixed(3)}`);
      }
      return lines.join("\\n");
    }

    function edgeColor(edge) {
      return edge.sign === "negative" ? "#2563eb" : "#dc2626";
    }

    function nodeColor(node) {
      if (!node) return "#9ca3af";
      if (node.type === "gene") return node.color || node.moduleColor || "#9ca3af";
      return node.color || node.moduleColor || "#c9ad85";
    }

    function edgeDetailText(edge) {
      return [
        `${edge.gene} - ${edge.metabolite}`,
        `EdgeWeight: ${Number(edge.edgeWeight || 0).toFixed(3)}`,
        `Sign: ${edge.sign}`,
        `ModelSupportCount: ${edge.modelSupportCount}`,
        `ScreenSupportCount: ${edge.screenSupportCount}`,
        edge.rraRank !== null && edge.rraRank !== undefined ? `RRARank: ${edge.rraRank}` : "",
        edge.spearmanRho !== null && edge.spearmanRho !== undefined ? `Spearman rho: ${Number(edge.spearmanRho).toFixed(3)}` : ""
      ].filter(Boolean).join("\\n");
    }

    function polarPoint(cx, cy, radius, theta) {
      return { x: cx + radius * Math.cos(theta), y: cy + radius * Math.sin(theta) };
    }

    function computeCircosLayout(genes, metabolites) {
      const nGene = genes.length;
      const nMetabolite = metabolites.length;
      const nTotal = nGene + nMetabolite;
      if (!nTotal) return new Map();

      const fullCircle = Math.PI * 2;
      const meanItemSpan = fullCircle / nTotal;
      let itemGap = Math.min(Math.PI / 400, meanItemSpan * 0.10);
      let groupGap = Math.max(7 * Math.PI / 180, itemGap * 8);
      let totalGap = Math.max(0, nTotal - 2) * itemGap + 2 * groupGap;
      if (totalGap >= fullCircle * 0.92) {
        itemGap = meanItemSpan * 0.04;
        groupGap = Math.max(4 * Math.PI / 180, itemGap * 6);
        totalGap = Math.max(0, nTotal - 2) * itemGap + 2 * groupGap;
      }

      let itemWidth = (fullCircle - totalGap) / nTotal;
      if (itemWidth <= 0) {
        itemGap = 0;
        groupGap = 0;
        itemWidth = fullCircle / nTotal;
      }

      const layout = new Map();
      let currentAngle = Math.PI * 0.76 + groupGap / 2;
      const assign = (nodes, nodeType, afterGroupGap) => {
        nodes.forEach((node, index) => {
          const thetaStart = currentAngle;
          const thetaEnd = thetaStart + itemWidth;
          layout.set(node.id, {
            thetaStart,
            thetaEnd,
            thetaMid: (thetaStart + thetaEnd) / 2,
            nodeType
          });
          currentAngle = thetaEnd + (index < nodes.length - 1 ? itemGap : afterGroupGap);
        });
      };
      assign(genes, "gene", groupGap);
      assign(metabolites, "metabolite", groupGap);
      return layout;
    }

    function annularPath(cx, cy, innerRadius, outerRadius, thetaStart, thetaEnd) {
      const largeArc = Math.abs(thetaEnd - thetaStart) > Math.PI ? 1 : 0;
      const p1 = polarPoint(cx, cy, outerRadius, thetaStart);
      const p2 = polarPoint(cx, cy, outerRadius, thetaEnd);
      const p3 = polarPoint(cx, cy, innerRadius, thetaEnd);
      const p4 = polarPoint(cx, cy, innerRadius, thetaStart);
      return [
        `M ${p1.x.toFixed(3)} ${p1.y.toFixed(3)}`,
        `A ${outerRadius.toFixed(3)} ${outerRadius.toFixed(3)} 0 ${largeArc} 1 ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}`,
        `L ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}`,
        `A ${innerRadius.toFixed(3)} ${innerRadius.toFixed(3)} 0 ${largeArc} 0 ${p4.x.toFixed(3)} ${p4.y.toFixed(3)}`,
        "Z"
      ].join(" ");
    }

    function chordPath(cx, cy, thetaStart, thetaEnd, radius, tension = 0.18) {
      const p1 = polarPoint(cx, cy, radius, thetaStart);
      const p4 = polarPoint(cx, cy, radius, thetaEnd);
      const p2 = polarPoint(cx, cy, radius * tension, thetaStart);
      const p3 = polarPoint(cx, cy, radius * tension, thetaEnd);
      return `M ${p1.x.toFixed(3)} ${p1.y.toFixed(3)} C ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}, ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}, ${p4.x.toFixed(3)} ${p4.y.toFixed(3)}`;
    }

    function pointChordPath(source, target, cx, cy, tension = 0.20) {
      const p2 = { x: cx + (source.x - cx) * tension, y: cy + (source.y - cy) * tension };
      const p3 = { x: cx + (target.x - cx) * tension, y: cy + (target.y - cy) * tension };
      return `M ${source.x.toFixed(3)} ${source.y.toFixed(3)} C ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}, ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}, ${target.x.toFixed(3)} ${target.y.toFixed(3)}`;
    }

    function networkSelectionState(graph, positions, controls) {
      const selectedNodeId = String(controls.selectedNodeId || "");
      const neighborIds = selectedNodeId && graph.adjacency.has(selectedNodeId) ? graph.adjacency.get(selectedNodeId) : new Set();
      const hasSelection = Boolean(selectedNodeId && graph.nodeLookup.has(selectedNodeId));
      const selectedVisible = selectedNodeId && positions.has(selectedNodeId);
      return { selectedNodeId, neighborIds, hasSelection, selectedVisible };
    }

    function biasColor(node) {
      const bias = Number.isFinite(Number(node.directionBias)) ? Number(node.directionBias) : (() => {
        const total = Number(node.positiveEdges || 0) + Number(node.negativeEdges || 0);
        return total ? (Number(node.positiveEdges || 0) - Number(node.negativeEdges || 0)) / total : 0;
      })();
      return bias >= 0 ? mixHex("#f8fafc", "#dc2626", Math.min(1, Math.abs(bias))) : mixHex("#f8fafc", "#2563eb", Math.min(1, Math.abs(bias)));
    }

    function signedHeatColor(value, scale) {
      const limit = Math.max(1e-6, Number(scale || 1));
      const normalized = clamp((Number(value || 0) + limit) / (2 * limit), 0, 1);
      return heatmapColor(normalized * 2 - 1, { min: -1, max: 1 }, "rdbu");
    }

    function addNetworkTitle(svg, dataset, graph, layoutName, width) {
      const title = svgEl("text", {
        x: width / 2,
        y: 28,
        "text-anchor": "middle",
        "font-size": 18,
        "font-weight": 700,
        fill: "#111827"
      });
      title.textContent = dataset.title || "Network Explorer";
      svg.appendChild(title);

      const subtitle = svgEl("text", {
        x: width / 2,
        y: 48,
        "text-anchor": "middle",
        "font-size": 11,
        fill: "#64748b"
      });
      subtitle.textContent = `${layoutName}; ${graph.edges.length} edges, ${graph.genes.length} genes, ${graph.metabolites.length} metabolites`;
      svg.appendChild(subtitle);
    }

    function addNetworkLegend(svg, x, y, mode) {
      const legend = svgEl("g", {});
      legend.appendChild(svgEl("rect", { x: x - 16, y: y - 22, width: 172, height: mode === "cnet" ? 96 : 118, fill: "#ffffff", stroke: "#d7dde5", rx: 8 }));
      const legendTitle = svgEl("text", { x, y, "font-size": 12, "font-weight": 700, fill: "#334155" });
      legendTitle.textContent = "Legend";
      legend.appendChild(legendTitle);
      legend.appendChild(svgEl("circle", { cx: x + 8, cy: y + 24, r: 7, fill: "#9ca3af", stroke: "#ffffff", "stroke-width": 1 }));
      const geneLabel = svgEl("text", { x: x + 24, y: y + 28, "font-size": 11, fill: "#334155" });
      geneLabel.textContent = "Gene module";
      legend.appendChild(geneLabel);
      legend.appendChild(svgEl("circle", { cx: x + 8, cy: y + 48, r: 7, fill: "#c9ad85", stroke: "#ffffff", "stroke-width": 1 }));
      const metabLabel = svgEl("text", { x: x + 24, y: y + 52, "font-size": 11, fill: "#334155" });
      metabLabel.textContent = "Metabolite";
      legend.appendChild(metabLabel);
      if (mode === "cnet") {
        legend.appendChild(svgEl("line", { x1: x, y1: y + 72, x2: x + 34, y2: y + 72, stroke: "#8b5cf6", "stroke-width": 3, opacity: 0.75 }));
        const edgeLabel = svgEl("text", { x: x + 44, y: y + 76, "font-size": 11, fill: "#334155" });
        edgeLabel.textContent = "Metabolite edge";
        legend.appendChild(edgeLabel);
      } else {
        legend.appendChild(svgEl("line", { x1: x, y1: y + 72, x2: x + 34, y2: y + 72, stroke: "#dc2626", "stroke-width": 3, opacity: 0.75 }));
        const positiveLabel = svgEl("text", { x: x + 44, y: y + 76, "font-size": 11, fill: "#334155" });
        positiveLabel.textContent = "Positive";
        legend.appendChild(positiveLabel);
        legend.appendChild(svgEl("line", { x1: x, y1: y + 96, x2: x + 34, y2: y + 96, stroke: "#2563eb", "stroke-width": 3, opacity: 0.75 }));
        const negativeLabel = svgEl("text", { x: x + 44, y: y + 100, "font-size": 11, fill: "#334155" });
        negativeLabel.textContent = "Negative";
        legend.appendChild(negativeLabel);
      }
      svg.appendChild(legend);
    }

    function addTrackAnnotationLegend(svg, x, y) {
      const rows = [
        ["track 1", "sector strip"],
        ["track 2", "group-wise mean"],
        ["track 3", "mean z-score heatmap"],
        ["track 4", "weighted degree"],
        ["track 5", "module/core strength"],
        ["track 6", "direction bias"]
      ];
      const legend = svgEl("g", {});
      legend.appendChild(svgEl("rect", { x: x - 16, y: y - 22, width: 210, height: 178, fill: "#ffffff", stroke: "#d7dde5", rx: 8 }));
      const title = svgEl("text", { x, y, "font-size": 12, "font-weight": 700, fill: "#334155" });
      title.textContent = "Track annotations";
      legend.appendChild(title);
      for (let idx = 0; idx < rows.length; idx++) {
        const [label, desc] = rows[idx];
        const yy = y + 24 + idx * 20;
        const labelNode = svgEl("text", { x, y: yy, "font-size": 10.5, "font-weight": 700, fill: "#374151" });
        labelNode.textContent = label;
        legend.appendChild(labelNode);
        const descNode = svgEl("text", { x: x + 58, y: yy, "font-size": 10.5, fill: "#64748b" });
        descNode.textContent = desc;
        legend.appendChild(descNode);
      }
      svg.appendChild(legend);
    }

    function shouldShowNetworkLabel(node, graph, showLabels) {
      if (!showLabels) return false;
      if (graph.nodes.length <= 70) return true;
      return Number(node.degree || 0) >= 3 || node.type === "metabolite";
    }

    function addCircularLabel(svg, node, pos, radius, graph, showLabels, dimmed, onClick) {
      if (!shouldShowNetworkLabel(node, graph, showLabels)) return;
      const rightSide = Math.cos(pos.theta) >= 0;
      const label = svgEl("text", {
        x: pos.x + (rightSide ? radius + 8 : -radius - 8),
        y: pos.y + 4,
        "text-anchor": rightSide ? "start" : "end",
        "font-size": node.type === "metabolite" ? 10 : 9,
        fill: dimmed ? "#94a3b8" : "#334155",
        cursor: "pointer"
      });
      label.textContent = node.label;
      label.addEventListener("click", onClick);
      const labelTitle = svgEl("title");
      labelTitle.textContent = nodeDetailText(node);
      label.appendChild(labelTitle);
      svg.appendChild(label);
    }

    function renderNetworkChart(dataset, controls) {
      const width = clamp(Number(controls.width || 1100), 760, 2400);
      const height = clamp(Number(controls.height || 760), 520, 2000);
      const baseNodeSize = clamp(Number(controls.nodeSize || 7), 4, 18);
      const showLabels = Boolean(controls.showLabels);
      const graph = prepareNetworkGraph(dataset, controls);
      if (!graph.edges.length) return null;

      const layout = controls.layout === "cnet" ? "cnet" : "circos";
      if (layout === "cnet") {
        return renderNetworkCnetChart(dataset, controls, graph, width, height, baseNodeSize, showLabels);
      }
      return renderNetworkCircosChart(dataset, controls, graph, width, height, baseNodeSize, showLabels);
    }

    function renderNetworkCircosChart(dataset, controls, graph, width, height, baseNodeSize, showLabels) {
      const cx = width / 2;
      const cy = height / 2 + 18;
      const radius = Math.max(190, Math.min(width - 260, height - 120) / 2);
      const outerR = radius;
      const scaleR = radius / 1.035;
      const radii = {
        outerStripInner: scaleR * 0.992,
        outerStripOuter: scaleR * 1.035,
        trackMeanbarInner: scaleR * 0.86,
        trackMeanbarOuter: scaleR * 0.975,
        trackMeanheatInner: scaleR * 0.795,
        trackMeanheatOuter: scaleR * 0.85,
        trackDegreeInner: scaleR * 0.685,
        trackDegreeOuter: scaleR * 0.775,
        trackCoreInner: scaleR * 0.605,
        trackCoreOuter: scaleR * 0.675,
        trackBiasInner: scaleR * 0.53,
        trackBiasOuter: scaleR * 0.58,
        linkRadius: scaleR * 0.47
      };
      const layoutMap = computeCircosLayout(graph.genes, graph.metabolites);
      const positions = new Map();
      for (const [nodeId, geometry] of layoutMap.entries()) {
        const xy = polarPoint(cx, cy, outerR + 4, geometry.thetaMid);
        positions.set(nodeId, { ...geometry, x: xy.x, y: xy.y, theta: geometry.thetaMid });
      }
      const selection = networkSelectionState(graph, positions, controls);
      const maxDegree = Math.max(1, ...graph.nodes.map(node => Number(node.weightedDegree || node.degree || 0)));
      const maxCore = Math.max(1e-6, ...graph.nodes.map(node => Number(node.moduleCore || 0)).filter(Number.isFinite));
      const maxAbs = Number(dataset.summary?.maxAbsWeight || 1);
      const trackScales = dataset.trackScales || {};

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": dataset.title || "Network Explorer"
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      addNetworkTitle(svg, dataset, graph, "Circos layout", width);

      for (const edge of [...graph.edges].sort((a, b) => Number(a.absWeight || 0) - Number(b.absWeight || 0))) {
        const source = layoutMap.get(edge.source);
        const target = layoutMap.get(edge.target);
        if (!source || !target) continue;
        const connectedToSelection = selection.hasSelection && selection.selectedVisible && (edge.source === selection.selectedNodeId || edge.target === selection.selectedNodeId);
        const dimmed = selection.hasSelection && selection.selectedVisible && !connectedToSelection;
        const absWeight = Number(edge.absWeight || Math.abs(edge.edgeWeight || 0));
        const strokeWidth = 0.35 + 3.2 * Math.sqrt(Math.min(1, absWeight / Math.max(1e-6, maxAbs)));
        const path = svgEl("path", {
          d: chordPath(cx, cy, source.thetaMid, target.thetaMid, radii.linkRadius),
          fill: "none",
          stroke: edgeColor(edge),
          "stroke-width": connectedToSelection ? strokeWidth + 1.2 : strokeWidth,
          opacity: dimmed ? 0.05 : connectedToSelection ? 0.82 : 0.40,
          "stroke-linecap": "round"
        });
        const lineTitle = svgEl("title");
        lineTitle.textContent = edgeDetailText(edge);
        path.appendChild(lineTitle);
        svg.appendChild(path);
      }

      for (const node of graph.nodes) {
        const geometry = layoutMap.get(node.id);
        if (!geometry) continue;
        const isSelected = node.id === selection.selectedNodeId;
        const isNeighbor = selection.hasSelection && selection.selectedVisible && selection.neighborIds.has(node.id);
        const dimmed = selection.hasSelection && selection.selectedVisible && !isSelected && !isNeighbor;
        const outerSegment = svgEl("path", {
          d: annularPath(cx, cy, radii.outerStripInner, radii.outerStripOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: nodeColor(node),
          opacity: dimmed ? 0.22 : 1,
          stroke: isSelected ? "#f59e0b" : isNeighbor ? "#fbbf24" : "#ffffff",
          "stroke-width": isSelected ? 2.4 : isNeighbor ? 1.9 : 0.6,
          cursor: "pointer"
        });
        outerSegment.dataset.nodeId = node.id;
        outerSegment.addEventListener("click", event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
        const titleNode = svgEl("title");
        titleNode.textContent = nodeDetailText(node);
        outerSegment.appendChild(titleNode);
        svg.appendChild(outerSegment);

        const track2Values = Array.isArray(node.track2Values) ? node.track2Values.map(Number).filter(Number.isFinite) : [];
        const track2Scale = node.type === "gene" ? Number(trackScales.geneTrack2 || trackScales.geneMean || 1) : Number(trackScales.metaboliteTrack2 || trackScales.metaboliteMean || 1);
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackMeanbarInner, radii.trackMeanbarOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: "#fbfbfb",
          opacity: dimmed ? 0.18 : 1,
          stroke: "#eef2f7",
          "stroke-width": 0.25
        }));
        if (track2Values.length > 1) {
          const groupColors = dataset.track2?.group1Colors || {};
          const groupOrder = dataset.track2?.group1Order || [];
          for (let idx = 0; idx < track2Values.length; idx++) {
            const value = track2Values[idx];
            const theta = (geometry.thetaStart + geometry.thetaEnd) / 2 + (idx - (track2Values.length - 1) / 2) * Math.max(0.002, (geometry.thetaEnd - geometry.thetaStart) * 0.10);
            const rMid = 0.5 * (radii.trackMeanbarInner + radii.trackMeanbarOuter);
            const radialHalf = 0.42 * (radii.trackMeanbarOuter - radii.trackMeanbarInner);
            const r = rMid + clamp(value / Math.max(1e-6, track2Scale), -1, 1) * radialHalf;
            const xy = polarPoint(cx, cy, r, theta);
            svg.appendChild(svgEl("circle", {
              cx: xy.x,
              cy: xy.y,
              r: 2.2,
              fill: groupColors[groupOrder[idx]] || "#6b7280",
              opacity: dimmed ? 0.18 : 0.92,
              stroke: "none"
            }));
          }
        } else {
          const value = track2Values.length ? track2Values[0] : Number(node.meanZScore || 0);
          const rMid = 0.5 * (radii.trackMeanbarInner + radii.trackMeanbarOuter);
          const rOuter = value >= 0
            ? rMid + clamp(value / Math.max(1e-6, track2Scale), 0, 1) * (radii.trackMeanbarOuter - rMid)
            : rMid + clamp(value / Math.max(1e-6, track2Scale), -1, 0) * (rMid - radii.trackMeanbarInner);
          svg.appendChild(svgEl("path", {
            d: annularPath(cx, cy, Math.min(rMid, rOuter), Math.max(rMid, rOuter), geometry.thetaStart, geometry.thetaEnd),
            fill: "#6b7280",
            opacity: dimmed ? 0.12 : 0.88,
            stroke: "none"
          }));
        }

        const meanScale = node.type === "gene" ? Number(trackScales.geneMean || 1) : Number(trackScales.metaboliteMean || 1);
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackMeanheatInner, radii.trackMeanheatOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: signedHeatColor(Number(node.meanZScore || 0), meanScale),
          opacity: dimmed ? 0.15 : 1,
          stroke: "#ffffff",
          "stroke-width": 0.25
        }));

        const degreeScale = node.type === "gene" ? Number(trackScales.geneDegree || maxDegree) : Number(trackScales.metaboliteDegree || maxDegree);
        const degreeOuterR = radii.trackDegreeInner + (radii.trackDegreeOuter - radii.trackDegreeInner) * Math.min(1, Number(node.weightedDegree || 0) / Math.max(1e-6, degreeScale));
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackDegreeInner, degreeOuterR, geometry.thetaStart, geometry.thetaEnd),
          fill: "#4b5563",
          opacity: dimmed ? 0.15 : 0.92,
          stroke: "none"
        }));

        const coreScale = node.type === "gene" ? Number(trackScales.geneCore || maxCore) : Number(trackScales.metaboliteCore || maxCore);
        const coreOuterR = radii.trackCoreInner + (radii.trackCoreOuter - radii.trackCoreInner) * Math.min(1, Number(node.moduleCore || 0) / Math.max(1e-6, coreScale));
        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackCoreInner, coreOuterR, geometry.thetaStart, geometry.thetaEnd),
          fill: node.type === "gene" ? (node.moduleColor || "#9ca3af") : "#8c6d46",
          opacity: dimmed ? 0.14 : 0.92,
          stroke: "none"
        }));

        svg.appendChild(svgEl("path", {
          d: annularPath(cx, cy, radii.trackBiasInner, radii.trackBiasOuter, geometry.thetaStart, geometry.thetaEnd),
          fill: biasColor(node),
          opacity: dimmed ? 0.15 : 0.95,
          stroke: "#ffffff",
          "stroke-width": 0.25
        }));

        const pos = positions.get(node.id);
        if (pos) {
          addCircularLabel(svg, node, pos, baseNodeSize, graph, showLabels, dimmed, event => {
            event.stopPropagation();
            setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
          });
        }
      }

      const geneLabelPos = polarPoint(cx, cy, outerR + 36, Math.PI * 1.18);
      const geneLabel = svgEl("text", { x: geneLabelPos.x, y: geneLabelPos.y, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#334155" });
      geneLabel.textContent = "Genes";
      svg.appendChild(geneLabel);
      const metabLabelPos = polarPoint(cx, cy, outerR + 36, Math.PI * 0.06);
      const metabLabel = svgEl("text", { x: metabLabelPos.x, y: metabLabelPos.y, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#334155" });
      metabLabel.textContent = "Metabolites";
      svg.appendChild(metabLabel);
      addNetworkLegend(svg, 24, 88, "circos");
      addTrackAnnotationLegend(svg, 24, 206);

      svg.addEventListener("click", () => {
        if (getViewControls("network_explorer").selectedNodeId) {
          setControl("network_explorer", "selectedNodeId", "");
        }
      });

      return { svg, graph, layout: "circos" };
    }

    function renderNetworkCnetChart(dataset, controls, graph, width, height, baseNodeSize, showLabels) {
      const cx = width / 2;
      const cy = height / 2 + 18;
      const ringR = Math.max(190, Math.min(width - 250, height - 130) / 2);
      const layoutMap = computeCircosLayout(graph.genes, graph.metabolites);
      const positions = new Map();
      const maxDegree = Math.max(1, ...graph.nodes.map(node => Number(node.degree || 0)));
      for (const node of graph.nodes) {
        const geometry = layoutMap.get(node.id);
        if (!geometry) continue;
        const jitter = 18 * Math.sin((positions.size + 1) * 1.71);
        const xy = polarPoint(cx, cy, ringR + jitter, geometry.thetaMid);
        positions.set(node.id, { ...geometry, x: xy.x, y: xy.y, theta: geometry.thetaMid });
      }
      const selection = networkSelectionState(graph, positions, controls);
      const maxAbs = Number(dataset.summary?.maxAbsWeight || 1);

      const svg = svgEl("svg", {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": dataset.title || "Network Explorer"
      });
      svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      addNetworkTitle(svg, dataset, graph, "CNet circular layout", width);

      for (const edge of [...graph.edges].sort((a, b) => Number(a.absWeight || 0) - Number(b.absWeight || 0))) {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) continue;
        const connectedToSelection = selection.hasSelection && selection.selectedVisible && (edge.source === selection.selectedNodeId || edge.target === selection.selectedNodeId);
        const dimmed = selection.hasSelection && selection.selectedVisible && !connectedToSelection;
        const absWeight = Number(edge.absWeight || Math.abs(edge.edgeWeight || 0));
        const strokeWidth = 0.30 + 2.5 * Math.sqrt(Math.min(1, absWeight / Math.max(1e-6, maxAbs)));
        const path = svgEl("path", {
          d: pointChordPath(source, target, cx, cy, 0.18),
          fill: "none",
          stroke: edge.metaboliteColor || edgeColor(edge),
          "stroke-width": connectedToSelection ? strokeWidth + 1.1 : strokeWidth,
          opacity: dimmed ? 0.06 : connectedToSelection ? 0.86 : 0.56,
          "stroke-linecap": "round"
        });
        const title = svgEl("title");
        title.textContent = edgeDetailText(edge);
        path.appendChild(title);
        svg.appendChild(path);
      }

      for (const node of graph.nodes) {
        const pos = positions.get(node.id);
        if (!pos) continue;
        const isSelected = node.id === selection.selectedNodeId;
        const isNeighbor = selection.hasSelection && selection.selectedVisible && selection.neighborIds.has(node.id);
        const dimmed = selection.hasSelection && selection.selectedVisible && !isSelected && !isNeighbor;
        const radius = baseNodeSize + Math.min(13, Math.sqrt(Number(node.degree || 1) / maxDegree) * 13);
        const circle = svgEl("circle", {
          cx: pos.x,
          cy: pos.y,
          r: isSelected ? radius + 3 : radius,
          fill: nodeColor(node),
          opacity: dimmed ? 0.24 : 0.97,
          stroke: isSelected ? "#f59e0b" : isNeighbor ? "#fbbf24" : "#ffffff",
          "stroke-width": isSelected ? 3 : isNeighbor ? 2.2 : 1.1,
          cursor: "pointer"
        });
        circle.dataset.nodeId = node.id;
        circle.addEventListener("click", event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
        const nodeTitle = svgEl("title");
        nodeTitle.textContent = nodeDetailText(node);
        circle.appendChild(nodeTitle);
        svg.appendChild(circle);

        addCircularLabel(svg, node, pos, radius, graph, showLabels, dimmed, event => {
          event.stopPropagation();
          setControl("network_explorer", "selectedNodeId", node.id === selection.selectedNodeId ? "" : node.id);
        });
      }

      addNetworkLegend(svg, 24, 88, "cnet");
      svg.addEventListener("click", () => {
        if (getViewControls("network_explorer").selectedNodeId) {
          setControl("network_explorer", "selectedNodeId", "");
        }
      });

      return { svg, graph, layout: "cnet" };
    }

    function renderNetworkSummary(dataset, rendered) {
      const legend = el("div", { className: "legend" });
      legend.appendChild(el("span", { className: "legend-item", text: "Source: T03 high-confidence network" }));
      legend.appendChild(el("span", { className: "legend-item", text: `Edges: ${rendered.graph.edges.length}/${(dataset.edges || []).length}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Genes shown: ${rendered.graph.genes.length}/${dataset.summary?.genes || 0}` }));
      legend.appendChild(el("span", { className: "legend-item", text: `Metabolites shown: ${rendered.graph.metabolites.length}/${dataset.summary?.metabolites || 0}` }));
      if (getViewControls("network_explorer").selectedNodeId) {
        const node = rendered.graph.nodeLookup.get(getViewControls("network_explorer").selectedNodeId);
        if (node) legend.appendChild(el("span", { className: "legend-item", text: `Selected: ${node.label}` }));
      }
      return legend;
    }

    function renderPcaView(view) {
      const dataset = getActiveDataset();
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      const title = dataset ? dataset.title : view.title;
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: title }),
        el("p", {
          className: "panel-note",
          text: "Switch dataset, inspect samples by hover, and export the current SVG snapshot."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => downloadSvg(chartShell.querySelector("svg"), `${dataset ? dataset.id : "pca"}.svg`) }),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        chartShell.appendChild(renderPcaChart(dataset, controls));
        chartWrap.appendChild(chartShell);
        panel.appendChild(chartWrap);
        panel.appendChild(renderPcaLegend(dataset, controls.colorBy));
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No PCA payload available for the selected dataset." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderAssociationView(view) {
      const dataset = getAssociationDataset();
      const controls = getViewControls(view.id);
      if (dataset) {
        resolveAssociationControlDefaults(dataset, controls);
      }

      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? `${dataset.title}` : view.title }),
        el("p", {
          className: "panel-note",
          text: "Switch between gene-metabolite and module-metabolite pairs. Scatter points and confidence bands use the associated module color."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => downloadSvg(chartShell.querySelector("svg"), `${dataset ? dataset.id : "association"}.svg`) }),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderAssociationChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(rendered.summary);
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No valid association payload available." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No regression payload available for the selected type." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderModuleHeatmapView(view) {
      const dataset = report.datasets.module_heatmap || null;
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? dataset.title : view.title }),
        el("p", {
          className: "panel-note",
          text: "Filter modules and metabolites, sort rows and columns, and export the current Spearman association heatmap."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => {
          const svg = chartShell.querySelector("svg");
          if (svg) downloadSvg(svg, "module_heatmap.svg");
        }}),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderModuleHeatmapChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(renderModuleHeatmapSummary(dataset, rendered));
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No module-metabolite cells are available for the selected filters." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No module-metabolite association payload available." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderNetworkView(view) {
      const dataset = getNetworkDataset();
      const controls = getViewControls(view.id);
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: dataset ? dataset.title : view.title }),
        el("p", {
          className: "panel-note",
          text: "T03-only Circos and CNet views. Inspect nodes by hover and click a node to highlight first-order neighbors."
        })
      ]));

      const schema = report.schemas[view.schema_id];
      const controlsRow = el("div", { className: "controls" });
      for (const control of schema.controls || []) {
        controlsRow.appendChild(renderControlField(view, control));
      }
      panel.appendChild(controlsRow);

      const actionBar = el("div", { className: "action-bar" }, [
        el("button", { text: "Export SVG", onclick: () => {
          const svg = chartShell.querySelector("svg");
          if (svg) downloadSvg(svg, `${dataset ? dataset.id : "network"}.svg`);
        }}),
        el("button", { text: "Reset", onclick: () => resetControls(view.id) })
      ]);
      panel.appendChild(actionBar);

      const chartWrap = el("div", { className: "chart-wrap" });
      const chartShell = el("div", { className: "chart-shell" });
      if (dataset) {
        const rendered = renderNetworkChart(dataset, controls);
        if (rendered && rendered.svg) {
          chartShell.appendChild(rendered.svg);
          chartWrap.appendChild(chartShell);
          panel.appendChild(chartWrap);
          panel.appendChild(renderNetworkSummary(dataset, rendered));
        } else {
          chartWrap.appendChild(el("div", { className: "placeholder", text: "No network edges match the selected filters." }));
          panel.appendChild(chartWrap);
        }
      } else {
        chartWrap.appendChild(el("div", { className: "placeholder", text: "No T03 high-confidence network payload available." }));
        panel.appendChild(chartWrap);
      }
      return panel;
    }

    function renderPlaceholderView(view) {
      const panel = el("section", { className: "panel" });
      panel.appendChild(el("div", { className: "panel-head" }, [
        el("h2", { className: "panel-title", text: view.title }),
        el("p", { className: "panel-note", text: "This view is reserved for a later stage." })
      ]));
      panel.appendChild(el("div", { className: "placeholder", text: view.description || "Not implemented yet." }));
      return panel;
    }

    function renderMain() {
      const main = el("main", { className: "main" });
      const view = getView(state.activeViewId) || report.views[0];
      if (view.kind === "pca") {
        main.appendChild(renderPcaView(view));
      } else if (view.kind === "association") {
        main.appendChild(renderAssociationView(view));
      } else if (view.kind === "module_heatmap") {
        main.appendChild(renderModuleHeatmapView(view));
      } else if (view.kind === "network") {
        main.appendChild(renderNetworkView(view));
      } else {
        main.appendChild(renderPlaceholderView(view));
      }
      return main;
    }

    function render() {
      clear(app);
      app.appendChild(renderSidebar());
      try {
        app.appendChild(renderMain());
      } catch (error) {
        app.appendChild(renderRuntimeError(error));
      }
    }

    render();
  </script>
</body>
</html>
"""


def render_interactive_report_html(engine, cfg) -> str:
    model = _build_interactive_report_model(engine, cfg)
    html_text = _interactive_html_template()
    html_text = html_text.replace("__PROJECT_NAME__", html.escape(str(cfg.project_name)))
    html_text = html_text.replace("__PAYLOAD__", _json_script_payload(model.to_dict()))
    return html_text


def generate_interactive_visual_report(engine, cfg, report_path: str | Path) -> None:
    output_path = Path(report_path)
    safe_mkdir(output_path.parent)
    output_path.write_text(render_interactive_report_html(engine, cfg), encoding="utf-8")


__all__ = [
    "PALETTE",
    "ControlSpec",
    "InteractiveReportModel",
    "InteractiveViewSpec",
    "_build_module_heatmap_payload",
    "_build_network_payload",
    "_build_pca_payload",
    "_build_summary_payload",
    "_interactive_html_template",
    "_json_default",
    "_json_dumps",
    "generate_interactive_visual_report",
    "render_interactive_report_html",
]
