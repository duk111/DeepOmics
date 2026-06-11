from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...outputs import FIGURE_FILE_PREFIXES
from .common import _base_plotly_config, _base_style


def _pair_key(module: str, metabolite: str) -> str:
    return f"{module}||{metabolite}"


def _association_lookup(engine) -> dict[tuple[str, str], float]:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return {}
    if not {"Module", "Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        return {}
    work = assoc_df.loc[:, ["Module", "Metabolite", "SpearmanRho"]].copy()
    work["Module"] = work["Module"].astype(str).str.strip()
    work["Metabolite"] = work["Metabolite"].astype(str).str.strip()
    work["SpearmanRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce")
    work = work.loc[work["Module"].ne("") & work["Metabolite"].ne("") & work["SpearmanRho"].notna()]
    return {(str(row.Module), str(row.Metabolite)): float(row.SpearmanRho) for row in work.itertuples(index=False)}


def _static_top_pair_keys(engine) -> list[tuple[str, str]]:
    from ..static.association import _module_top_metabolite_pairs

    pairs_df = _module_top_metabolite_pairs(engine)
    if pairs_df.empty:
        return []
    keys = []
    for row in pairs_df.itertuples(index=False):
        module = str(row.Module).strip()
        metabolite = str(row.Metabolite).strip()
        if module and metabolite:
            keys.append((module, metabolite))
    return list(dict.fromkeys(keys))


def export_line_panels(context, save_dir: Path, prefix_key: str) -> dict[str, Any] | None:
    """Export F26 module-metabolite trend panels for the interactive page."""
    if "trend" not in prefix_key.lower():
        return None

    engine = context.engine
    group_df = context.pca_group_df

    from ..static.association import CIRCOS_METABOLITE_COLOR, _module_maps
    from ..static.module import (
        _align_group_annotations_to_samples,
        _coerce_module_eigengene_df,
        _module_group_orders_and_colors,
        _row_zscore,
    )

    eigengenes_df = _coerce_module_eigengene_df(engine.ml_results.get("module_eigengenes_df", pd.DataFrame()))
    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else pd.DataFrame()
    if eigengenes_df.empty or not isinstance(metab_df, pd.DataFrame) or metab_df.empty:
        return None

    metab_df = metab_df.copy(deep=False)
    metab_df.index = pd.Index(metab_df.index.astype(str).str.strip())
    metab_df.columns = metab_df.columns.astype(str)

    shared_samples = eigengenes_df.index.intersection(metab_df.index, sort=False)
    if len(shared_samples) < 2:
        return None
    shared_samples = pd.Index(shared_samples.astype(str))
    eigengenes_df = eigengenes_df.reindex(shared_samples)
    metab_df = metab_df.reindex(shared_samples)

    annotation_df = _align_group_annotations_to_samples(shared_samples.tolist(), group_df)
    if annotation_df.empty:
        return None
    annotation_df = annotation_df.reindex(shared_samples.tolist())
    group_orders = annotation_df["_group_table_order"].astype(int).tolist()
    group_orders_by_col, color_maps_by_col = _module_group_orders_and_colors(
        group_df,
        annotation_df["group1"].astype(str).tolist(),
        annotation_df["group2"].astype(str).tolist(),
        group_orders,
    )
    group1_order = [str(value) for value in group_orders_by_col.get("group1", [])]
    group2_order = [str(value) for value in group_orders_by_col.get("group2", [])]
    group1_color_map = {str(key): str(value) for key, value in color_maps_by_col.get("group1", {}).items()}
    if not group1_order or not group2_order:
        return None

    _gene_to_module, _gene_to_color, module_to_color, _all_module_order = _module_maps(engine)
    rho_lookup = _association_lookup(engine)

    module_options = [str(module) for module in eigengenes_df.columns.astype(str).tolist() if str(module).strip()]
    metabolite_options = [str(metabolite) for metabolite in metab_df.columns.astype(str).tolist() if str(metabolite).strip()]
    module_options = [module for module in module_options if module.lower() != "grey"]
    if not module_options or not metabolite_options:
        return None

    static_top_keys = [
        (module, metabolite)
        for module, metabolite in _static_top_pair_keys(engine)
        if module in module_options and metabolite in metabolite_options
    ]
    static_rank_lookup = {key: idx for idx, key in enumerate(static_top_keys, start=1)}

    all_pair_keys = [(module, metabolite) for module in module_options for metabolite in metabolite_options]

    trend_modules = module_options
    trend_metabolites = metabolite_options
    module_z = _row_zscore(eigengenes_df.loc[:, trend_modules].T).T
    metab_z = _row_zscore(metab_df.loc[:, trend_metabolites].T).T

    pairs_payload: list[dict[str, Any]] = []
    for combo_rank, (module, metabolite) in enumerate(all_pair_keys, start=1):
        groups = []
        for group1 in group1_order:
            module_values = []
            metabolite_values = []
            counts = []
            for group2 in group2_order:
                samples = annotation_df.index[
                    annotation_df["group1"].astype(str).eq(str(group1))
                    & annotation_df["group2"].astype(str).eq(str(group2))
                ].astype(str).tolist()
                valid_module_samples = [sample for sample in samples if sample in module_z.index]
                valid_metabolite_samples = [sample for sample in samples if sample in metab_z.index]
                module_value = module_z.loc[valid_module_samples, module].mean() if valid_module_samples else np.nan
                metabolite_value = metab_z.loc[valid_metabolite_samples, metabolite].mean() if valid_metabolite_samples else np.nan
                module_values.append(float(module_value) if pd.notna(module_value) else None)
                metabolite_values.append(float(metabolite_value) if pd.notna(metabolite_value) else None)
                counts.append(len(samples))

            groups.append(
                {
                    "group1": group1,
                    "color": group1_color_map.get(group1, "#9ca3af"),
                    "module_values": module_values,
                    "metabolite_values": metabolite_values,
                    "counts": counts,
                }
            )

        rho = rho_lookup.get((module, metabolite))
        static_rank = static_rank_lookup.get((module, metabolite))
        pairs_payload.append(
            {
                "id": _pair_key(module, metabolite),
                "static_rank": static_rank,
                "combo_rank": combo_rank,
                "module": module,
                "metabolite": metabolite,
                "spearman_rho": rho,
                "abs_rho": abs(rho) if rho is not None else None,
                "module_color": module_to_color.get(module, "#111827"),
                "metabolite_color": CIRCOS_METABOLITE_COLOR,
                "groups": groups,
            }
        )

    pair_options = sorted(
        pairs_payload,
        key=lambda pair: (
            -float(pair["abs_rho"]) if pair["abs_rho"] is not None else float("inf"),
            int(pair["combo_rank"]),
            str(pair["module"]),
            str(pair["metabolite"]),
        ),
    )
    pair_options_payload = [
        {
            "id": pair["id"],
            "label": f"{pair['module']} - {pair['metabolite']}",
            "module": pair["module"],
            "metabolite": pair["metabolite"],
            "spearman_rho": pair["spearman_rho"],
            "abs_rho": pair["abs_rho"],
            "static_rank": pair["static_rank"],
            "combo_rank": pair["combo_rank"],
        }
        for pair in pair_options
    ]

    default_pair_key = static_top_keys[0] if static_top_keys else all_pair_keys[0]
    default_pair = next((pair for pair in pairs_payload if pair["module"] == default_pair_key[0] and pair["metabolite"] == default_pair_key[1]), pairs_payload[0])
    visible_groups = group1_order
    static_prefix = FIGURE_FILE_PREFIXES.get(prefix_key, prefix_key)

    return {
        "figure_id": "module_metabolite_trend_line_panels",
        "title": "Module-Metabolite Trends by group1",
        "chart_type": "line_panels",
        "interactive_page_id": "line-panels",
        "static_files": {"png": f"plots/{static_prefix}.png", "svg": f"plots/{static_prefix}.svg"},
        "plotly_spec": {
            "pairs": pairs_payload,
            "pair_options": pair_options_payload,
            "module_options": module_options,
            "metabolite_options": metabolite_options,
            "group1_order": group1_order,
            "group1_colors": group1_color_map,
            "group2_order": group2_order,
            "config": _base_plotly_config(),
        },
        "default_state": {
            "view_type": "trend",
            "pair_id": default_pair["id"],
            "module_id": default_pair["module"],
            "metabolite_id": default_pair["metabolite"],
            "visible_groups": visible_groups,
        },
        "available_states": {"view_type": ["trend"]},
        "style": _base_style(),
    }


__all__ = [
    "export_line_panels",
]
