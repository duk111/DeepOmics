from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .context import VisualizationContext
from .interactive_assets import _interactive_html_template
from .interactive_model import ControlSpec, InteractiveReportModel, InteractiveViewSpec
from .interactive_render import _json_default, _json_dumps, _json_script_payload
from .interactive_render import generate_interactive_visual_report as _generate_interactive_visual_report
from .interactive_render import render_interactive_report_html as _render_interactive_report_html
from ..outputs import FIGURE_FILE_PREFIXES
from .interactive_schemas import (
    _build_association_schema,
    _build_module_heatmap_schema,
    _build_network_schema,
    _build_pca_schema,
    _build_placeholder_schema,
)
from .registry import iter_figure_specs
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


_FIGURE_GALLERY_META: dict[str, dict[str, Any]] = {
    "sample_clustering_dendrogram": {
        "title": "Sample Clustering Dendrogram",
        "category": "Sample QC",
        "description": "Hierarchical clustering of samples after preprocessing.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
    "transcriptome_pca": {
        "title": "Transcriptome PCA",
        "category": "PCA",
        "description": "Transcriptome PCA scatter view.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "transcriptome", "colorBy": "group1", "xComponent": 1, "yComponent": 2},
    },
    "metabolome_pca": {
        "title": "Metabolome PCA",
        "category": "PCA",
        "description": "Metabolome PCA scatter view.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "metabolome", "colorBy": "group1", "xComponent": 1, "yComponent": 2},
    },
    "transcriptome_pca_subgroups": {
        "title": "Transcriptome PCA Subgroups",
        "category": "PCA",
        "description": "Transcriptome PCA colored by the secondary grouping.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "transcriptome", "colorBy": "group2", "xComponent": 1, "yComponent": 2},
    },
    "metabolome_pca_subgroups": {
        "title": "Metabolome PCA Subgroups",
        "category": "PCA",
        "description": "Metabolome PCA colored by the secondary grouping.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "metabolome", "colorBy": "group2", "xComponent": 1, "yComponent": 2},
    },
    "transcriptome_pca_pairs": {
        "title": "Transcriptome PCA Pairs",
        "category": "PCA",
        "description": "Transcriptome PCA pairs overview.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "transcriptome", "colorBy": "group1", "xComponent": 1, "yComponent": 2},
    },
    "metabolome_pca_pairs": {
        "title": "Metabolome PCA Pairs",
        "category": "PCA",
        "description": "Metabolome PCA pairs overview.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "metabolome", "colorBy": "group1", "xComponent": 1, "yComponent": 2},
    },
    "transcriptome_pca_pairs_subgroups": {
        "title": "Transcriptome PCA Pairs Subgroups",
        "category": "PCA",
        "description": "Transcriptome PCA pairs overview colored by the secondary grouping.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "transcriptome", "colorBy": "group2", "xComponent": 1, "yComponent": 2},
    },
    "metabolome_pca_pairs_subgroups": {
        "title": "Metabolome PCA Pairs Subgroups",
        "category": "PCA",
        "description": "Metabolome PCA pairs overview colored by the secondary grouping.",
        "interactiveViewId": "pca",
        "interactiveControls": {"dataset": "metabolome", "colorBy": "group2", "xComponent": 1, "yComponent": 2},
    },
    "top_gene_metabolite_pairs": {
        "title": "Top Gene-Metabolite Pairs",
        "category": "Association",
        "description": "Top gene-metabolite regression panels ranked by edge weight.",
        "interactiveViewId": "association",
        "interactiveControls": {"pairType": "gene_metabolite"},
    },
    "module_top_metabolite_regressions": {
        "title": "Module Top Metabolite Regressions",
        "category": "Association",
        "description": "Module eigengene regression panels against each module's top metabolite.",
        "interactiveViewId": "association",
        "interactiveControls": {"pairType": "module_metabolite"},
    },
    "module_eigengene_heatmap": {
        "title": "Module Eigengene Heatmap",
        "category": "Module",
        "description": "Module eigengene heatmap with group annotation tracks.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
    "module_eigengene_heatmap_group2": {
        "title": "Module Eigengene Heatmap Group2",
        "category": "Module",
        "description": "Module eigengene heatmap grouped by the secondary grouping.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
    "module_zscore_line_panels": {
        "title": "Module Z-score Line Panels",
        "category": "Module",
        "description": "Module z-score line panels faceted by the primary grouping.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
    "module_gene_zscore_line_panels": {
        "title": "Module Gene Z-score Line Panels",
        "category": "Module",
        "description": "Gene-level z-score line panels with module summaries.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
    "module_metabolite_association_heatmap": {
        "title": "Module-Metabolite Association Heatmap",
        "category": "Module",
        "description": "Spearman association heatmap for modules and metabolites.",
        "interactiveViewId": "module_heatmap",
        "interactiveControls": {"rowSort": "significance", "columnSort": "significance"},
    },
    "compressed_circos_network": {
        "title": "Compressed Circos Network",
        "category": "Network",
        "description": "Compact Circos overview for the high-confidence network.",
        "interactiveViewId": "network_explorer",
        "interactiveControls": {"layout": "circos"},
    },
    "floating_cnet_circos_network": {
        "title": "Floating CNet Circos Network",
        "category": "Network",
        "description": "Floating circular cnet-style network view.",
        "interactiveViewId": "network_explorer",
        "interactiveControls": {"layout": "cnet"},
    },
    "association_evidence_upset": {
        "title": "Association Evidence UpSet",
        "category": "Evidence",
        "description": "Evidence-overlap UpSet plot for candidate metabolite-gene edges.",
        "interactiveViewId": None,
        "interactiveControls": {},
    },
}


def _ordered_unique(values: list[str]) -> list[str]:
    return _ordered_unique_with_order(values, None)


def _build_summary_payload(engine, cfg) -> dict[str, Any]:
    total_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    high_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    return {
        "projectName": "OmicsPrism",
        "samples": int(engine.adata.n_obs),
        "genes": int(engine.adata.n_vars),
        "metabolites": int(len(engine.adata.uns.get("metabolite_names", []))),
        "totalEdges": int(len(total_df)) if isinstance(total_df, pd.DataFrame) else 0,
        "highConfidenceEdges": int(len(high_df)) if isinstance(high_df, pd.DataFrame) else 0,
    }


def _build_figure_gallery_payload(context: VisualizationContext) -> tuple[dict[str, Any], ...]:
    figures: list[dict[str, Any]] = []
    for index, spec in enumerate(iter_figure_specs()):
        if not spec.enabled(context):
            continue
        meta = _FIGURE_GALLERY_META.get(spec.key, {})
        prefix = FIGURE_FILE_PREFIXES[spec.prefix_key]
        interactive_view_id = meta.get("interactiveViewId")
        interactive_controls = dict(meta.get("interactiveControls", {}))
        figures.append(
            {
                "id": spec.key,
                "title": str(meta.get("title") or spec.key.replace("_", " ").title()),
                "category": str(meta.get("category") or "Figure"),
                "description": str(meta.get("description") or spec.description or ""),
                "index": int(index),
                "enabled": True,
                "interactiveViewId": interactive_view_id,
                "interactiveControls": interactive_controls,
                "badge": "Interactive" if interactive_view_id else "Static",
                "previewPath": f"plots/{prefix}.png",
                "staticPaths": {
                    "png": f"plots/{prefix}.png",
                    "svg": f"plots/{prefix}.svg",
                    "pdf": f"plots/{prefix}.pdf",
                },
            }
        )
    return tuple(figures)


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


def _build_interactive_report_model(engine, cfg) -> InteractiveReportModel:
    context = VisualizationContext.from_engine(engine, cfg)
    summary = _build_summary_payload(engine, cfg)
    figures = _build_figure_gallery_payload(context)
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
            id="gallery",
            title="Figure Gallery",
            kind="gallery",
            schema_id="gallery.home",
            enabled=True,
            description="Static figure gallery for the association analysis report.",
            data_key="",
        ),
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
        "gallery.home": _build_placeholder_schema("gallery.home", "Figure Gallery"),
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
        "activeViewId": "gallery",
        "activeFigureId": "",
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

    implemented_views = ["gallery", "pca", "association"]
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
        "figureCount": len(figures),
        "interactiveFigureCount": sum(1 for figure in figures if figure.get("interactiveViewId")),
        "implementedViews": implemented_views,
        "placeholderViews": placeholder_views,
        "groupTableLoaded": bool(group_df is not None and not group_df.empty),
        "secondaryGrouping": bool(group_df is not None and "group2" in group_df.columns and group_df["group2"].notna().any()),
    }

    return InteractiveReportModel(
        meta=meta,
        figures=figures,
        views=views,
        schemas=schemas,
        datasets=datasets,
        initial_state=initial_state,
    )


def render_interactive_report_html(engine, cfg) -> str:
    return _render_interactive_report_html(engine, cfg, _build_interactive_report_model)


def generate_interactive_visual_report(engine, cfg, report_path: str | Path) -> None:
    _generate_interactive_visual_report(engine, cfg, report_path, _build_interactive_report_model)


__all__ = [
    "PALETTE",
    "ControlSpec",
    "InteractiveReportModel",
    "InteractiveViewSpec",
    "_build_module_heatmap_payload",
    "_build_network_payload",
    "_build_pca_payload",
    "_build_association_schema",
    "_build_module_heatmap_schema",
    "_build_network_schema",
    "_build_pca_schema",
    "_build_placeholder_schema",
    "_build_summary_payload",
    "_interactive_html_template",
    "_json_default",
    "_json_dumps",
    "generate_interactive_visual_report",
    "render_interactive_report_html",
]
