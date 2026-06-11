from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import _gene_expression_df, _metabolomics_df
from ..static.module import _coerce_module_eigengene_df, _module_order_from_summary
from ..static.regression import _module_annotation_maps, _module_top_metabolite_regression_rows
from .common import _base_plotly_config, _base_style


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy(deep=False)
    work.index = pd.Index(work.index.astype(str).str.strip(), name=work.index.name or "SampleID")
    work.columns = pd.Index(work.columns.astype(str).str.strip(), name=work.columns.name)
    work = work.loc[work.index.astype(str).str.len() > 0, work.columns.astype(str).str.len() > 0]
    work = work.loc[~work.index.duplicated(keep="first"), ~work.columns.duplicated(keep="first")]
    return work.apply(pd.to_numeric, errors="coerce")


def _metabolomics_df_from_engine(engine) -> pd.DataFrame:
    if hasattr(engine, "metabolomics_df"):
        try:
            df = engine.metabolomics_df()
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    if (not isinstance(df, pd.DataFrame) or df.empty) and hasattr(engine, "adata"):
        df = _metabolomics_df(engine.adata)
    return _numeric_matrix(df) if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()


def _gene_expression_df_from_engine(engine) -> pd.DataFrame:
    if hasattr(engine, "gene_expression_df"):
        try:
            df = engine.gene_expression_df()
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    if (not isinstance(df, pd.DataFrame) or df.empty) and hasattr(engine, "adata"):
        df = _gene_expression_df(engine.adata)
    return _numeric_matrix(df) if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()


def _paired_arrays(
    x_series: pd.Series,
    y_series: pd.Series,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    shared = x.index.astype(str).intersection(y.index.astype(str), sort=False)
    x = x.reindex(shared)
    y = y.reindex(shared)
    valid = x.notna() & y.notna() & np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    sample_ids = shared[valid.to_numpy()].astype(str).tolist()
    return sample_ids, x.loc[valid].to_numpy(dtype=float), y.loc[valid].to_numpy(dtype=float)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or float(np.nanstd(x)) <= 0 or float(np.nanstd(y)) <= 0:
        return None
    try:
        value = float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _regression_payload(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    if x.size < 2 or y.size < 2 or float(np.nanstd(x)) <= 0:
        return None
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return None

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return None

    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = intercept + slope * x_grid
    payload: dict[str, Any] = {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
    }

    dof = int(x.size - 2)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if dof > 0 and sxx > 0:
        fitted = intercept + slope * x
        residual_ss = float(np.sum((y - fitted) ** 2))
        residual_se = np.sqrt(residual_ss / dof)
        t_value = float(t.ppf(0.975, dof))
        se_mean = residual_se * np.sqrt((1.0 / x.size) + ((x_grid - np.mean(x)) ** 2 / sxx))
        ci_delta = t_value * se_mean
        if np.isfinite(ci_delta).all():
            payload["ci"] = {
                "x": x_grid.tolist(),
                "lower": (y_grid - ci_delta).tolist(),
                "upper": (y_grid + ci_delta).tolist(),
            }
    return payload


def _panel_payload(
    *,
    panel_type: str,
    rank: int,
    entity_id: str,
    metabolite_id: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    sample_ids: list[str],
    color: str,
    metric_value: float | None,
    metric_label: str = "rho",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title_entity = f"{entity_id} module" if panel_type == "module-metabolite" else entity_id
    panel = {
        "id": f"{panel_type}:{entity_id}:{metabolite_id}",
        "rank": int(rank),
        "type": panel_type,
        "entity_id": entity_id,
        "entity_label": title_entity,
        "metabolite_id": metabolite_id,
        "title": f"{title_entity} vs {metabolite_id}",
        "x": x_values.tolist(),
        "y": y_values.tolist(),
        "sample_ids": sample_ids,
        "x_label": x_label,
        "y_label": y_label,
        "color": color,
        "metric_label": metric_label,
        "metric_value": metric_value,
        "regression": _regression_payload(x_values, y_values),
    }
    if extra:
        panel.update(extra)
    return panel


def _f13_panels(engine, cfg) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    gene_df = _gene_expression_df_from_engine(engine)
    metab_df = _metabolomics_df_from_engine(engine)
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if gene_df.empty or metab_df.empty or not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return [], {"entity_options": [], "metabolite_options": [], "pair_options": []}

    required = {"Gene", "Metabolite"}
    if not required.issubset(edge_df.columns):
        return [], {"entity_options": [], "metabolite_options": [], "pair_options": []}

    _gene_to_module, gene_to_color, _module_to_color = _module_annotation_maps(engine)
    rows: list[dict[str, Any]] = []
    for _, row in edge_df.iterrows():
        gene = str(row.get("Gene", "")).strip()
        metabolite = str(row.get("Metabolite", "")).strip()
        if not gene or not metabolite or gene not in gene_df.columns or metabolite not in metab_df.columns:
            continue
        sample_ids, x, y = _paired_arrays(gene_df[gene], metab_df[metabolite])
        if x.size < 2 or y.size < 2:
            continue
        rho = _spearman_rho(x, y)
        edge_weight = pd.to_numeric(pd.Series([row.get("EdgeWeight", np.nan)]), errors="coerce").iloc[0]
        rra_rank = pd.to_numeric(pd.Series([row.get("RRARank", np.nan)]), errors="coerce").iloc[0]
        rows.append({
            "_gene": gene,
            "_metabolite": metabolite,
            "_x": x,
            "_y": y,
            "_sample_ids": sample_ids,
            "_rho": rho,
            "_abs_rho": abs(rho) if rho is not None else -np.inf,
            "_edge_weight": float(edge_weight) if pd.notna(edge_weight) else None,
            "_edge_weight_sort": float(edge_weight) if pd.notna(edge_weight) else -np.inf,
            "_rra_rank": float(rra_rank) if pd.notna(rra_rank) else None,
            "_rra_rank_sort": float(rra_rank) if pd.notna(rra_rank) else np.inf,
            "_module": str(row.get("Module", "")).strip() if "Module" in row.index else "",
        })

    rows = sorted(
        rows,
        key=lambda item: (
            -float(item["_abs_rho"]),
            -float(item["_edge_weight_sort"]),
            float(item["_rra_rank_sort"]),
            str(item["_gene"]),
            str(item["_metabolite"]),
        ),
    )
    panels: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        panels.append(
            _panel_payload(
                panel_type="gene-metabolite",
                rank=rank,
                entity_id=str(row["_gene"]),
                metabolite_id=str(row["_metabolite"]),
                x_label=str(row["_gene"]),
                y_label=str(row["_metabolite"]),
                x_values=row["_x"],
                y_values=row["_y"],
                sample_ids=row["_sample_ids"],
                color=gene_to_color.get(str(row["_gene"]), "#1f77b4"),
                metric_value=row["_rho"],
                extra={
                    "edge_weight": row["_edge_weight"],
                    "rra_rank": row["_rra_rank"],
                    "module": row["_module"],
                },
            )
        )

    return panels, {
        "entity_options": sorted({panel["entity_id"] for panel in panels}),
        "metabolite_options": sorted({panel["metabolite_id"] for panel in panels}),
        "pair_options": [panel["id"] for panel in panels],
    }


def _f25_panels(engine) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    metab_df = _metabolomics_df_from_engine(engine)
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if metab_df.empty or eigengenes_df.empty or not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return [], {"entity_options": [], "metabolite_options": [], "pair_options": []}
    if not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return [], {"entity_options": [], "metabolite_options": [], "pair_options": []}

    _gene_to_module, _gene_to_color, module_to_color = _module_annotation_maps(engine)
    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    order_lookup = {module_name: idx for idx, module_name in enumerate(module_order)}

    top_rows = _module_top_metabolite_regression_rows(engine)
    top_pair_ids = set()
    if isinstance(top_rows, pd.DataFrame) and not top_rows.empty:
        top_pair_ids = {
            (str(row.Module).strip(), str(row.Metabolite).strip())
            for row in top_rows.itertuples(index=False)
        }

    work = assoc_df.copy()
    work["Module"] = work["Module"].astype(str).str.strip()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work["SpearmanRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce")
    for column in ("FDR", "PValue"):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        work["Module"].ne("")
        & work["Metabolite"].ne("")
        & work["SpearmanRho"].notna()
        & (work["Module"].str.lower() != "grey")
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        module = str(row["Module"])
        metabolite = str(row["Metabolite"])
        if module not in eigengenes_df.columns or metabolite not in metab_df.columns:
            continue
        sample_ids, x, y = _paired_arrays(eigengenes_df[module], metab_df[metabolite])
        if x.size < 2 or y.size < 2:
            continue
        rho = float(row["SpearmanRho"]) if pd.notna(row["SpearmanRho"]) else _spearman_rho(x, y)
        fdr = float(row["FDR"]) if "FDR" in row.index and pd.notna(row["FDR"]) else None
        p_value = float(row["PValue"]) if "PValue" in row.index and pd.notna(row["PValue"]) else None
        rows.append({
            "_module": module,
            "_metabolite": metabolite,
            "_x": x,
            "_y": y,
            "_sample_ids": sample_ids,
            "_rho": rho,
            "_abs_rho": abs(rho) if rho is not None and np.isfinite(rho) else -np.inf,
            "_fdr": fdr,
            "_p_value": p_value,
            "_sig_sort": fdr if fdr is not None else p_value if p_value is not None else np.inf,
            "_module_order": order_lookup.get(module, len(order_lookup)),
            "_is_static_top": (module, metabolite) in top_pair_ids,
        })

    rows = sorted(
        rows,
        key=lambda item: (
            -float(item["_abs_rho"]),
            float(item["_sig_sort"]),
            int(item["_module_order"]),
            str(item["_module"]),
            str(item["_metabolite"]),
        ),
    )
    panels: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        module = str(row["_module"])
        panels.append(
            _panel_payload(
                panel_type="module-metabolite",
                rank=rank,
                entity_id=module,
                metabolite_id=str(row["_metabolite"]),
                x_label=f"{module} module eigengene",
                y_label=str(row["_metabolite"]),
                x_values=row["_x"],
                y_values=row["_y"],
                sample_ids=row["_sample_ids"],
                color=module_to_color.get(module, "#9ca3af"),
                metric_value=row["_rho"],
                extra={
                    "fdr": row["_fdr"],
                    "p_value": row["_p_value"],
                    "is_static_top": row["_is_static_top"],
                },
            )
        )

    return panels, {
        "entity_options": sorted({panel["entity_id"] for panel in panels}, key=lambda m: order_lookup.get(m, len(order_lookup))),
        "metabolite_options": sorted({panel["metabolite_id"] for panel in panels}),
        "pair_options": [panel["id"] for panel in panels],
    }


def export_scatter_panels(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export regression scatter panel data for interactive page 'scatter-panels'."""
    style = _base_style()
    is_module = "module" in prefix_key.lower()

    if is_module:
        panels, options = _f25_panels(context.engine)
        panel_type = "module-metabolite"
        title = "Module-Metabolite Regression Panels"
    else:
        panels, options = _f13_panels(context.engine, context.cfg)
        panel_type = "gene-metabolite"
        title = "Gene-Metabolite Regression Panels"

    if not panels:
        return None

    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    default_pair_ids = [panel["id"] for panel in panels[:4]]
    while len(default_pair_ids) < 4:
        default_pair_ids.append("")

    return {
        "figure_id": f"{panel_type}_panels",
        "title": title,
        "chart_type": "scatter_panels",
        "interactive_page_id": "scatter-panels",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "panels": panels,
            "default_pair_ids": default_pair_ids,
            "entity_options": options["entity_options"],
            "metabolite_options": options["metabolite_options"],
            "pair_options": options["pair_options"],
            "config": _base_plotly_config(),
        },
        "default_state": {
            "panel_type": panel_type,
            "panel_1_pair_id": default_pair_ids[0],
            "panel_2_pair_id": default_pair_ids[1],
            "panel_3_pair_id": default_pair_ids[2],
            "panel_4_pair_id": default_pair_ids[3],
            "show_sample_id": False,
            "show_regression_line": True,
        },
        "available_states": {
            "panel_type": [panel_type],
            "entity_options": options["entity_options"],
            "metabolite_options": options["metabolite_options"],
            "pair_options": options["pair_options"],
        },
        "style": style,
    }


__all__ = ["export_scatter_panels"]
