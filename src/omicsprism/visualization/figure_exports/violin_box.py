from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from ..static.base import _group_color_map, _metabolomics_df, _ordered_unique_with_order
from ..static.distribution import _align_exact_group1_to_samples, _top_metabolite_order
from ..static.module import (
    _align_group_annotations_to_samples,
    _coerce_module_eigengene_df,
    _module_group_orders_and_colors,
    _module_order_from_summary,
    _row_zscore,
)
from .common import _base_plotly_config, _base_style


def _feature_group_payload(
    feature_id: str,
    rank: int,
    feature_type: str,
    values: pd.Series,
    annotation: pd.DataFrame,
    group_order: list[str],
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    numeric = pd.to_numeric(values, errors="coerce")
    for group_name in group_order:
        samples = annotation.index[annotation["group1"].astype(str).eq(str(group_name))].astype(str).tolist()
        group_values: list[float] = []
        sample_ids: list[str] = []
        for sample_id in samples:
            if sample_id not in numeric.index:
                continue
            value = numeric.loc[sample_id]
            if not np.isfinite(value):
                continue
            group_values.append(float(value))
            sample_ids.append(str(sample_id))
        groups.append({"group": str(group_name), "values": group_values, "sample_ids": sample_ids})

    return {
        "id": str(feature_id),
        "rank": int(rank),
        "type": feature_type,
        "label": str(feature_id),
        "feature": str(feature_id),
        "groups": groups,
    }


def _default_feature_ids(features: list[dict[str, Any]]) -> list[str]:
    ids = [str(feature["id"]) for feature in features[:4]]
    while len(ids) < 4:
        ids.append("")
    return ids


def _f14_metabolite_payload(engine, group_df: pd.DataFrame | None, adata=None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if adata is not None:
        metab_df = _metabolomics_df(adata)
    else:
        metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else pd.DataFrame()
    if not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return [], [], []

    metab_df = metab_df.copy(deep=False)
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip(), name=metab_df.index.name)
    metab_df.columns = pd.Index(metab_df.columns.astype(str).str.strip(), name=metab_df.columns.name)

    annotation = _align_exact_group1_to_samples(metab_df.index.astype(str).tolist(), group_df)
    if annotation.empty:
        return [], [], []
    shared = annotation.index.intersection(metab_df.index, sort=False)
    if len(shared) < 2:
        return [], [], []
    metab_df = metab_df.reindex(shared).apply(pd.to_numeric, errors="coerce")
    annotation = annotation.reindex(shared)

    group_orders = annotation["_group_table_order"].astype(int).tolist()
    group_order = _ordered_unique_with_order(annotation["group1"].astype(str).tolist(), group_orders)
    group_colors = [_group_color_map(group_order).get(group_name, "#9ca3af") for group_name in group_order]
    metabolites = _top_metabolite_order(engine, metab_df, 12)

    features = [
        _feature_group_payload(
            feature_id=metabolite,
            rank=rank,
            feature_type="metabolite",
            values=metab_df[metabolite],
            annotation=annotation,
            group_order=group_order,
        )
        for rank, metabolite in enumerate(metabolites, start=1)
        if metabolite in metab_df.columns
    ]
    return features, group_order, group_colors


def _f21_module_payload(engine, group_df: pd.DataFrame | None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    if eigengenes_df.empty:
        return [], [], []

    module_order = _module_order_from_summary(
        engine.ml_results.get("module_summary_df", pd.DataFrame()),
        eigengenes_df.columns.astype(str).tolist(),
    )
    module_order = [module for module in module_order if module in eigengenes_df.columns]
    if not module_order:
        return [], [], []

    zscore_df = _row_zscore(eigengenes_df.loc[:, module_order].T).T
    samples = zscore_df.index.astype(str).tolist()
    annotation = _align_group_annotations_to_samples(samples, group_df)
    if annotation.empty:
        return [], [], []
    annotation = annotation.reindex(samples)

    group_orders = annotation["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation["group1"].astype(str).tolist(),
        annotation["group2"].astype(str).tolist(),
        group_orders,
    )
    group_order = group_orders_by_col.get("group1", [])
    group1_color_map = color_maps_by_col.get("group1", {})
    group_colors = [group1_color_map.get(group_name, "#9ca3af") for group_name in group_order]

    features = [
        _feature_group_payload(
            feature_id=module,
            rank=rank,
            feature_type="module",
            values=zscore_df[module],
            annotation=annotation,
            group_order=group_order,
        )
        for rank, module in enumerate(module_order, start=1)
        if module in zscore_df.columns
    ]
    return features, group_order, group_colors


def _build_response(
    *,
    prefix_key: str,
    feature_type: str,
    view: str,
    title: str,
    y_label: str,
    features: list[dict[str, Any]],
    group_order: list[str],
    group_colors: list[str],
) -> dict[str, Any] | None:
    if not features or not group_order:
        return None

    default_ids = _default_feature_ids(features)
    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)
    return {
        "figure_id": f"{feature_type}_violin_box",
        "title": title,
        "chart_type": "violin_box",
        "interactive_page_id": "violin-box",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "view": view,
            "features": features,
            "feature_options": [str(feature["id"]) for feature in features],
            "default_feature_ids": default_ids,
            "group_order": group_order,
            "group_colors": group_colors,
            "y_label": y_label,
            "config": _base_plotly_config(),
        },
        "default_state": {
            "view": view,
            "feature_type": feature_type,
            "plot_type": "violin+box",
            "chart_style": "violin+box",
            "panel_1_feature_id": default_ids[0],
            "panel_2_feature_id": default_ids[1],
            "panel_3_feature_id": default_ids[2],
            "panel_4_feature_id": default_ids[3],
        },
        "available_states": {
            "view": [view],
            "feature_type": [feature_type],
            "plot_type": ["violin", "box", "violin+box"],
            "chart_style": ["violin", "box", "violin+box"],
            "feature_options": [str(feature["id"]) for feature in features],
        },
        "style": _base_style(),
    }


def export_violin_box(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export violin/box plot data for the shared interactive page 'violin-box'."""
    lower_key = prefix_key.lower()
    engine = context.engine
    group_df = context.pca_group_df

    if "kme" in lower_key:
        return None

    is_metabolite = "metabolite" in lower_key and "group1_violin" in lower_key
    if is_metabolite:
        features, group_order, group_colors = _f14_metabolite_payload(engine, group_df, context.pca_adata)
        return _build_response(
            prefix_key=prefix_key,
            feature_type="metabolite",
            view="metabolite",
            title="Metabolite Abundance Distribution",
            y_label="Metabolite abundance z-score",
            features=features,
            group_order=group_order,
            group_colors=group_colors,
        )

    features, group_order, group_colors = _f21_module_payload(engine, group_df)
    return _build_response(
        prefix_key=prefix_key,
        feature_type="module-eigengene",
        view="module",
        title="Module Eigengene Distribution",
        y_label="Module eigengene z-score",
        features=features,
        group_order=group_order,
        group_colors=group_colors,
    )


__all__ = ["export_violin_box"]
