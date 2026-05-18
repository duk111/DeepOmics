from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorspacious import cspace_convert
from matplotlib import colors

from ...utils import get_logger

logger = get_logger()

PALETTE = {
    "gene": "#2563eb",
    "metabolite": "#111827",
    "edge_positive": "#dc2626",
    "edge_negative": "#2563eb",
    "grid_aux": "#cbd5e1",
    "pca_scatter": "#4c78a8",
    "bar_total": "#93c5fd",
    "bar_dual_model": "#2563eb",
    "bar_high_conf": "#1d4ed8",
}


PCA_GROUP_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#003f5c",
    "#7a5195",
]


PCA_GROUP_MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "p",
    "H",
    "8",
    "d",
    "*",
]


def set_academic_style() -> None:
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.15


def _save_figure(fig: plt.Figure, save_stem: str | Path, cfg) -> None:
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)

    if not getattr(fig, "_skip_default_tight_layout", False):
        try:
            fig.tight_layout()
        except Exception:
            pass

    savefig_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.15,
    }

    if cfg.export_pdf:
        fig.savefig(save_stem.with_suffix(".pdf"), **savefig_kwargs)
    if cfg.export_svg:
        fig.savefig(save_stem.with_suffix(".svg"), **savefig_kwargs)
    if getattr(cfg, "export_png", True):
        fig.savefig(save_stem.with_suffix(".png"), dpi=int(getattr(fig, "_png_dpi", 300)), **savefig_kwargs)
    plt.close(fig)


def _gene_expression_df(adata) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )


def _metabolomics_df(adata) -> pd.DataFrame:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    if isinstance(metab_df, pd.DataFrame):
        return metab_df.copy(deep=False)
    return pd.DataFrame(
        np.asarray(metab_df, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=[str(x) for x in adata.uns.get("metabolite_names", [])],
    )


def _ordered_unique_nonempty(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def _categorical_colors(n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []

    if n_colors <= len(PCA_GROUP_PALETTE):
        return PCA_GROUP_PALETTE[:n_colors]

    extended_palette = list(PCA_GROUP_PALETTE)

    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        cmap_size = getattr(cmap, "N", 20)
        for idx in range(cmap_size):
            extended_palette.append(colors.to_hex(cmap(idx / max(cmap_size - 1, 1))))
            if len(extended_palette) >= n_colors:
                return extended_palette[:n_colors]

    remaining = n_colors - len(extended_palette)
    if remaining > 0:
        hsv = plt.get_cmap("hsv")
        for idx in range(remaining):
            extended_palette.append(colors.to_hex(hsv(idx / max(remaining, 1))))

    return extended_palette[:n_colors]


def _group_color_map(groups: list[str]) -> dict[str, str]:
    unique_groups = _ordered_unique_nonempty(groups)
    palette = _categorical_colors(len(unique_groups))
    return {group: palette[idx] for idx, group in enumerate(unique_groups)}


def _group_marker_map(groups: list[str]) -> dict[str, str]:
    unique_groups = _ordered_unique_nonempty(groups)
    return {
        group: PCA_GROUP_MARKERS[idx % len(PCA_GROUP_MARKERS)]
        for idx, group in enumerate(unique_groups)
    }


def _ordered_unique_with_order(
    values,
    orders: list[int] | None = None,
) -> list[str]:
    if orders is None:
        return _ordered_unique_nonempty(values)

    ordered_rows: list[tuple[int, int, str]] = []
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        label = str(value).strip()
        if not label:
            continue
        order_value = orders[idx] if idx < len(orders) else idx
        ordered_rows.append((int(order_value), idx, label))

    if not ordered_rows:
        return []

    ordered_rows.sort(key=lambda item: (item[0], item[1]))

    result: list[str] = []
    seen: set[str] = set()
    for _, _, label in ordered_rows:
        if label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


def _hue_wheel_color_series(
    n_colors: int,
    *,
    hue_start: float = 15.0,
    lightness: float = 65.0,
    safety: float = 0.95,
    chroma_resolution: float = 0.5,
) -> list[str]:
    if n_colors <= 0:
        return []

    hue_start = float(hue_start) % 360.0
    lightness = float(lightness)
    safety = float(np.clip(safety, 0.0, 1.0))
    chroma_resolution = max(float(chroma_resolution), 1e-3)

    hues = (hue_start + np.arange(n_colors, dtype=float) * (360.0 / float(n_colors))) % 360.0

    def _lch_to_clipped_rgb(l_value: float, c_value: float, h_value: float) -> np.ndarray:
        lch = np.array([l_value, c_value, h_value], dtype=float)
        rgb = np.asarray(cspace_convert(lch, "CIELCh", "sRGB1"), dtype=float)
        return np.clip(rgb, 0.0, 1.0)

    def _is_in_gamut(l_value: float, c_value: float, h_value: float, tol: float = 1e-9) -> bool:
        lch = np.array([l_value, c_value, h_value], dtype=float)
        rgb = np.asarray(cspace_convert(lch, "CIELCh", "sRGB1"), dtype=float)
        return bool(np.all(rgb >= -tol) and np.all(rgb <= 1.0 + tol))

    def _max_chroma_for_hue(l_value: float, h_value: float) -> float:
        low = 0.0
        high = chroma_resolution

        while _is_in_gamut(l_value, high, h_value):
            low = high
            high *= 2.0
            if high >= 300.0:
                break

        high = min(high, 300.0)

        for _ in range(24):
            mid = 0.5 * (low + high)
            if _is_in_gamut(l_value, mid, h_value):
                low = mid
            else:
                high = mid
        return low

    c_max_list = [_max_chroma_for_hue(lightness, float(hue)) for hue in hues]
    c_use = min(c_max_list) * safety if c_max_list else 0.0

    return [
        colors.to_hex(_lch_to_clipped_rgb(lightness, c_use, float(hue)))
        for hue in hues
    ]


def _global_secondary_group_color_map(
    secondary_groups: list[str],
    group_orders: list[int] | None = None,
) -> tuple[list[str], dict[str, str]]:
    ordered_secondary = _ordered_unique_with_order(secondary_groups, group_orders)
    ordered_colors = _hue_wheel_color_series(len(ordered_secondary))
    return ordered_secondary, {
        group_name: ordered_colors[idx]
        for idx, group_name in enumerate(ordered_secondary)
    }


def _related_color_series(base_color: str, n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []

    if n_colors == 1:
        return [base_color]

    base_rgb = np.array(colors.to_rgb(base_color), dtype=float).reshape(1, 1, 3)
    base_hsv = colors.rgb_to_hsv(base_rgb)[0, 0]
    hue = float(base_hsv[0])
    saturation = float(base_hsv[1])
    value = float(base_hsv[2])

    sat_start = max(0.38, saturation * 0.55)
    sat_end = min(0.95, max(saturation, 0.65))
    val_start = min(0.98, max(value, 0.92))
    val_end = max(0.45, min(value * 0.72, 0.78))

    sat_values = np.linspace(sat_start, sat_end, n_colors)
    val_values = np.linspace(val_start, val_end, n_colors)

    color_list: list[str] = []
    for sat, val in zip(sat_values, val_values):
        hsv = np.array([[[hue, float(sat), float(val)]]], dtype=float)
        rgb = colors.hsv_to_rgb(hsv)[0, 0]
        color_list.append(colors.to_hex(rgb))
    return color_list


__all__ = [
    "PALETTE",
    "PCA_GROUP_PALETTE",
    "PCA_GROUP_MARKERS",
    "logger",
    "set_academic_style",
    "_save_figure",
    "_gene_expression_df",
    "_metabolomics_df",
    "_ordered_unique_nonempty",
    "_categorical_colors",
    "_group_color_map",
    "_group_marker_map",
    "_ordered_unique_with_order",
    "_hue_wheel_color_series",
    "_global_secondary_group_color_map",
    "_related_color_series",
]
