from __future__ import annotations

import html
from pathlib import Path

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorspacious import cspace_convert
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Wedge, PathPatch
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2
from sklearn.decomposition import PCA

from .utils import get_logger, safe_mkdir

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

FIGURE_FILE_PREFIXES = {
    "sample_clustering_dendrogram": "F01_Sample_Clustering_Dendrogram",
    "transcriptome_pca": "F02_Transcriptome_PCA",
    "metabolome_pca": "F03_Metabolome_PCA",
    "transcriptome_pca_subgroups": "F02B_Transcriptome_PCA_Subgroups",
    "metabolome_pca_subgroups": "F03B_Metabolome_PCA_Subgroups",
    "transcriptome_pca_pairs": "F02C_Transcriptome_PCA_Pairs",
    "metabolome_pca_pairs": "F03C_Metabolome_PCA_Pairs",
    "transcriptome_pca_pairs_subgroups": "F02D_Transcriptome_PCA_Pairs_Subgroups",
    "metabolome_pca_pairs_subgroups": "F03D_Metabolome_PCA_Pairs_Subgroups",
    "top_gene_metabolite_pairs": "F04_Top_Gene_Metabolite_Pairs",
    "module_metabolite_association_heatmap": "F05_Module_Metabolite_Association_Heatmap",
    "compressed_circos_network": "F06_Compressed_Circos_Network",
}

TABLE_FILE_PREFIXES = {
    "gene_scores": "T01_Metabolite_Gene_Scoring_Table.csv",
    "total_network": "T02_Total_Association_Network.csv",
    "high_confidence_network": "T03_High_Confidence_Network.csv",
    "key_gene_summary": "T04_Key_Gene_Summary.csv",
    "metabolite_summary": "T05_Metabolite_Association_Summary.csv",
    "cytoscape_network": "T06_Association_Network_Cytoscape.csv",
    "gene_module_assignment": "T07_Gene_Module_Assignment.csv",
    "module_eigengenes": "T08_Module_Eigengenes.csv",
    "module_metabolite_association": "T09_Module_Metabolite_Association.csv",
    "module_summary": "T10_Module_Summary.csv",
}


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
        fig.savefig(save_stem.with_suffix(".png"), dpi=300, **savefig_kwargs)
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



def _load_pca_group_table(cfg) -> pd.DataFrame | None:
    group_table_path = getattr(cfg, "group_table_path", None)
    if not group_table_path:
        return None

    group_table_path = Path(group_table_path)
    group_df = pd.read_csv(group_table_path, sep=None, engine="python", encoding="utf-8-sig")
    normalized_columns = {
        str(column).replace("\ufeff", "").strip().lower(): column
        for column in group_df.columns
    }

    if "sample_id" not in normalized_columns:
        raise ValueError("PCA group table must contain a sample_id column.")

    primary_source_column = None
    if "group1" in normalized_columns:
        primary_source_column = normalized_columns["group1"]
    elif "group" in normalized_columns:
        primary_source_column = normalized_columns["group"]
    elif "group2" in normalized_columns:
        primary_source_column = normalized_columns["group2"]

    if primary_source_column is None:
        logger.info(
            "Loaded PCA group table from %s with %d samples but no grouping columns were found; PCA plots will be generated without grouping.",
            group_table_path,
            len(group_df),
        )
        return None

    rename_map = {
        normalized_columns["sample_id"]: "sample_id",
        primary_source_column: "group1",
    }
    source_columns = [normalized_columns["sample_id"], primary_source_column]
    if "group2" in normalized_columns and normalized_columns["group2"] != primary_source_column:
        rename_map[normalized_columns["group2"]] = "group2"
        source_columns.append(normalized_columns["group2"])

    group_df = group_df.loc[:, source_columns].rename(columns=rename_map).copy()

    group_df["sample_id"] = group_df["sample_id"].astype(str).str.strip()
    group_df["group1"] = group_df["group1"].astype("string").str.strip().replace("", pd.NA)

    if "group2" in group_df.columns:
        group_df["group2"] = group_df["group2"].astype("string").str.strip().replace("", pd.NA)

    valid_mask = group_df["sample_id"].ne("") & group_df["group1"].notna()
    dropped_rows = int((~valid_mask).sum())
    if dropped_rows > 0:
        logger.warning(
            "Dropped %d invalid rows from PCA group table because sample_id or group1 was empty.",
            dropped_rows,
        )
        group_df = group_df.loc[valid_mask].copy()

    if group_df.empty:
        logger.warning(
            "PCA group table is empty after removing invalid rows; PCA plots will be generated without grouping."
        )
        return None

    duplicated_mask = group_df["sample_id"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_ids = group_df.loc[duplicated_mask, "sample_id"].astype(str).unique().tolist()
        raise ValueError(
            "PCA group table contains duplicated sample_id values: "
            f"{duplicated_ids[:5]}"
        )

    column_order = ["sample_id", "group1"]
    if "group2" in group_df.columns and group_df["group2"].notna().any():
        column_order.append("group2")
    group_df = group_df.loc[:, column_order].copy()
    group_df["_group_table_order"] = np.arange(len(group_df), dtype=int)

    group_df.attrs["source_path"] = str(group_table_path)
    logger.info(
        "Loaded PCA group table from %s with %d samples across %d primary groups%s.",
        group_table_path,
        len(group_df),
        group_df["group1"].nunique(),
        f" and {group_df['group2'].nunique()} subgroup labels" if "group2" in group_df.columns else "",
    )
    return group_df


def _prepare_grouped_pca_inputs(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    group_df: pd.DataFrame | None,
) -> tuple[np.ndarray, list[str], pd.DataFrame | None]:
    values = np.asarray(matrix, dtype=np.float32)
    samples = [str(name).strip() for name in sample_names]

    if group_df is None or "sample_id" not in group_df.columns:
        return values, samples, None

    sample_index = pd.Index(samples, dtype=str, name="sample_id")
    group_table = group_df.copy()
    group_table["sample_id"] = group_table["sample_id"].astype(str).str.strip()
    group_table = group_table.set_index("sample_id", drop=True)

    matched_mask = sample_index.isin(group_table.index)
    missing_samples = sample_index[~matched_mask].tolist()
    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        suffix = " ..." if len(missing_samples) > 10 else ""
        logger.warning(
            "[%s] Skipped %d samples without group annotation in %s: %s%s",
            title,
            len(missing_samples),
            Path(group_df.attrs.get("source_path", "group_table")).name,
            preview,
            suffix,
        )

    unused_group_rows = group_table.index.difference(sample_index, sort=False).tolist()
    if unused_group_rows:
        logger.info(
            "[%s] Ignored %d group table entries not present in the current matrix.",
            title,
            len(unused_group_rows),
        )

    filtered_samples = sample_index[matched_mask].tolist()
    filtered_matrix = values[matched_mask]
    if len(filtered_samples) == 0:
        logger.warning("[%s] No overlapping samples were found between the matrix and PCA group table.", title)
        return filtered_matrix, filtered_samples, None

    grouped_plot_df = group_table.reindex(filtered_samples).reset_index()
    if len(grouped_plot_df.columns) <= 1:
        return filtered_matrix, filtered_samples, None
    return filtered_matrix, filtered_samples, grouped_plot_df


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



def _nested_group_color_map(
    primary_groups: list[str],
    secondary_groups: list[str],
    group_orders: list[int] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    rows: list[dict[str, object]] = []
    if group_orders is None:
        group_orders = list(range(len(primary_groups)))

    for primary, secondary, order in zip(primary_groups, secondary_groups, group_orders):
        if pd.isna(primary):
            continue
        primary_label = str(primary).strip()
        if not primary_label:
            continue

        if pd.isna(secondary):
            secondary_label = primary_label
        else:
            secondary_label = str(secondary).strip() or primary_label

        rows.append(
            {
                "primary": primary_label,
                "secondary": secondary_label,
                "order": int(order),
            }
        )

    if not rows:
        return {}, {}

    work_df = pd.DataFrame(rows).sort_values("order", kind="mergesort").reset_index(drop=True)

    primary_names = _ordered_unique_nonempty(work_df["primary"].tolist())
    primary_color_map = _group_color_map(primary_names)
    nested_color_map: dict[str, str] = {}

    for primary_name in primary_names:
        group_df = work_df.loc[work_df["primary"] == primary_name].sort_values("order", kind="mergesort")
        subgroup_names = _ordered_unique_nonempty(group_df["secondary"].tolist())
        subgroup_colors = _related_color_series(primary_color_map[primary_name], len(subgroup_names))
        for subgroup_name, subgroup_color in zip(subgroup_names, subgroup_colors):
            nested_color_map[f"{primary_name} | {subgroup_name}"] = subgroup_color

    return primary_color_map, nested_color_map

def _has_secondary_grouping(group_df: pd.DataFrame | None) -> bool:
    if group_df is None or "group2" not in group_df.columns:
        return False
    return group_df["group2"].notna().any()


def _try_add_confidence_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    *,
    confidence: float = 0.95,
) -> bool:
    if points.shape[0] < 3:
        return False

    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return False

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues <= 0):
        return False

    scale = float(np.sqrt(chi2.ppf(confidence, df=2)))
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.18,
        zorder=2,
    )
    ax.add_patch(ellipse)
    return True


def _try_add_convex_hull(ax: plt.Axes, points: np.ndarray, color: str) -> bool:
    if points.shape[0] < 3:
        return False

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return False

    hull_points = points[hull.vertices]
    polygon = Polygon(
        hull_points,
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.18,
        zorder=2,
        joinstyle="round",
    )
    ax.add_patch(polygon)
    return True


def _add_group_envelope(ax: plt.Axes, points: np.ndarray, color: str, fallback_radius: float) -> None:
    if points.shape[0] >= 3 and _try_add_confidence_ellipse(ax, points, color):
        return
    if points.shape[0] >= 3 and _try_add_convex_hull(ax, points, color):
        return

    if points.shape[0] == 2:
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=10.0,
            alpha=0.14,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.3,
            alpha=0.95,
            zorder=2,
            solid_capstyle="round",
        )
        return

    if points.shape[0] == 1:
        circle = plt.Circle(
            (float(points[0, 0]), float(points[0, 1])),
            radius=float(fallback_radius),
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.18,
            zorder=2,
        )
        ax.add_patch(circle)



def _plot_pca_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
    *,
    group_df: pd.DataFrame | None = None,
    primary_group_col: str = "group1",
    secondary_group_col: str | None = None,
    add_group_envelope: bool = True,
) -> None:
    plot_matrix, plot_sample_names, plot_group_df = _prepare_grouped_pca_inputs(
        matrix=np.asarray(matrix, dtype=np.float32),
        sample_names=sample_names,
        title=title,
        group_df=group_df,
    )

    if plot_matrix.shape[0] < 2 or plot_matrix.shape[1] < 2:
        logger.warning("[%s] PCA plot was skipped because fewer than 2 samples or features remained for plotting.", title)
        return

    pca = PCA(n_components=2, random_state=cfg.random_state)
    coords = pca.fit_transform(plot_matrix)
    var_exp = pca.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    legend_anchor = (1.02, 1.0)
    legend_marker_size = 7
    legend_fontsize = 10
    legend_handlelength = 0.8
    legend_handletextpad = 0.5
    legend_columnspacing = 1.0
    legend_labelspacing = 0.5

    if plot_group_df is None or primary_group_col not in plot_group_df.columns:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=42,
            alpha=0.90,
            color=PALETTE["pca_scatter"],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    else:
        plot_group_df = plot_group_df.copy()
        plot_group_df[primary_group_col] = plot_group_df[primary_group_col].astype(str).str.strip()
        plot_group_df["PC1"] = coords[:, 0]
        plot_group_df["PC2"] = coords[:, 1]

        span = max(
            float(np.ptp(coords[:, 0])) if coords.shape[0] > 0 else 0.0,
            float(np.ptp(coords[:, 1])) if coords.shape[0] > 0 else 0.0,
            1.0,
        )
        fallback_radius = 0.035 * span
        group_orders = plot_group_df["_group_table_order"].tolist() if "_group_table_order" in plot_group_df.columns else None
        primary_groups = _ordered_unique_with_order(
            plot_group_df[primary_group_col].tolist(),
            group_orders,
        )
        marker_map = _group_marker_map(primary_groups)

        has_secondary = (
            secondary_group_col is not None
            and secondary_group_col in plot_group_df.columns
            and plot_group_df[secondary_group_col].notna().any()
        )

        if has_secondary:
            subgroup_values = (
                plot_group_df[secondary_group_col].astype("string").fillna("").astype(str).str.strip()
            )
            subgroup_values = subgroup_values.where(subgroup_values.ne(""), "Missing")
            plot_group_df["_subgroup_label"] = subgroup_values

            secondary_groups, color_map = _global_secondary_group_color_map(
                plot_group_df["_subgroup_label"].astype(str).tolist(),
                group_orders,
            )

            for subgroup_name in secondary_groups:
                subgroup_df = plot_group_df.loc[plot_group_df["_subgroup_label"].astype(str) == subgroup_name]
                for primary_group_name in primary_groups:
                    group_points_df = subgroup_df.loc[
                        subgroup_df[primary_group_col].astype(str) == primary_group_name
                    ]
                    if group_points_df.empty:
                        continue

                    group_points = group_points_df.loc[:, ["PC1", "PC2"]].to_numpy(
                        dtype=float,
                        copy=False,
                    )
                    ax.scatter(
                        group_points[:, 0],
                        group_points[:, 1],
                        s=42,
                        alpha=0.90,
                        color=color_map[subgroup_name],
                        marker=marker_map[primary_group_name],
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=3,
                    )

            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in secondary_groups
            ]
            shape_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]

            legend_handles = color_handles + shape_handles
            total_legend_items = len(legend_handles)
            legend_ncol = 1 if total_legend_items <= 10 else 2 if total_legend_items <= 20 else 3

            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=legend_anchor,
                borderaxespad=0.0,
                ncol=legend_ncol,
                handlelength=legend_handlelength,
                handletextpad=legend_handletextpad,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                fontsize=legend_fontsize,
            )
        else:
            color_map = _group_color_map(plot_group_df[primary_group_col].tolist())
            primary_groups = _ordered_unique_nonempty(plot_group_df[primary_group_col].tolist())

            for group_name in primary_groups:
                group_points_df = plot_group_df.loc[plot_group_df[primary_group_col].astype(str) == group_name]
                group_points = group_points_df.loc[:, ["PC1", "PC2"]].to_numpy(dtype=float, copy=False)
                group_color = color_map[group_name]

                if add_group_envelope:
                    _add_group_envelope(ax, group_points, group_color, fallback_radius)

                ax.scatter(
                    group_points[:, 0],
                    group_points[:, 1],
                    s=42,
                    alpha=0.90,
                    color=group_color,
                    marker=marker_map[group_name],
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                    label=group_name,
                )

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=legend_marker_size,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=group_name,
                )
                for group_name in primary_groups
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=legend_anchor,
                borderaxespad=0.0,
                handlelength=legend_handlelength,
                handletextpad=legend_handletextpad,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                fontsize=legend_fontsize,
            )

    if len(plot_sample_names) <= 20 and adjust_text is not None:
        texts = [
            ax.text(x, y, label, fontsize=8, alpha=0.90)
            for x, y, label in zip(coords[:, 0], coords[:, 1], plot_sample_names)
        ]
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grid_aux"], "lw": 0.5})

    ax.axhline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=PALETTE["grid_aux"], linewidth=0.8, zorder=1)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.4,
        colors="black",
    )
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    _save_figure(fig, save_stem, cfg)

def _plot_pca_pairs_from_matrix(
    matrix: np.ndarray,
    sample_names: list[str],
    title: str,
    save_stem: str | Path,
    cfg,
    *,
    group_df: pd.DataFrame | None = None,
    primary_group_col: str = "group1",
    secondary_group_col: str | None = None,
    max_components: int = 10,
) -> None:
    plot_matrix, _, plot_group_df = _prepare_grouped_pca_inputs(
        matrix=np.asarray(matrix, dtype=np.float32),
        sample_names=sample_names,
        title=title,
        group_df=group_df,
    )

    if plot_matrix.shape[0] < 2 or plot_matrix.shape[1] < 2:
        logger.warning("[%s] PCA pairs plot was skipped because fewer than 2 samples or features remained for plotting.", title)
        return

    n_components = min(int(max_components), int(plot_matrix.shape[0]), int(plot_matrix.shape[1]))
    if n_components < 2:
        logger.warning("[%s] PCA pairs plot was skipped because fewer than 2 principal components were available.", title)
        return

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    coords = pca.fit_transform(plot_matrix)
    var_exp = pca.explained_variance_ratio_ * 100.0

    panel_size = 1.12
    fig_size = max(7.8, panel_size * n_components + 1.8)
    fig, axes = plt.subplots(n_components, n_components, figsize=(fig_size, fig_size))
    axes = np.asarray(axes, dtype=object)

    legend_handles: list[Line2D] = []
    ordered_groups: list[str] = []
    color_map: dict[str, str] = {}
    marker_map: dict[str, str] = {}
    secondary_groups: list[str] = []
    has_secondary = False

    if plot_group_df is not None and primary_group_col in plot_group_df.columns:
        plot_group_df = plot_group_df.copy()
        plot_group_df[primary_group_col] = plot_group_df[primary_group_col].astype(str).str.strip()
        group_orders = plot_group_df["_group_table_order"].tolist() if "_group_table_order" in plot_group_df.columns else None
        ordered_groups = _ordered_unique_with_order(
            plot_group_df[primary_group_col].tolist(),
            group_orders,
        )
        marker_map = _group_marker_map(ordered_groups)

        has_secondary = (
            secondary_group_col is not None
            and secondary_group_col in plot_group_df.columns
            and plot_group_df[secondary_group_col].notna().any()
        )

        if has_secondary:
            subgroup_values = (
                plot_group_df[secondary_group_col].astype("string").fillna("").astype(str).str.strip()
            )
            subgroup_values = subgroup_values.where(subgroup_values.ne(""), "Missing")
            plot_group_df["_subgroup_label"] = subgroup_values

            secondary_groups, color_map = _global_secondary_group_color_map(
                plot_group_df["_subgroup_label"].astype(str).tolist(),
                group_orders,
            )

            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7.5,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in secondary_groups
            ]
            shape_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[group_name],
                    linestyle="",
                    markersize=7.5,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in ordered_groups
            ]
            legend_handles = color_handles + shape_handles
        else:
            color_map = _group_color_map(ordered_groups)
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7.5,
                    markerfacecolor=color_map[group_name],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=group_name,
                )
                for group_name in ordered_groups
            ]

    for row_idx in range(n_components):
        for col_idx in range(n_components):
            ax = axes[row_idx, col_idx]

            if col_idx < row_idx:
                ax.axis("off")
                continue

            if col_idx == row_idx:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.56,
                    f"PC{row_idx + 1},\n{var_exp[row_idx]:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
                continue

            x = coords[:, col_idx]
            y = coords[:, row_idx]
            ax.axhline(0, color="#7a7a7a", linewidth=0.55, linestyle=(0, (6, 4)), zorder=1)
            ax.axvline(0, color="#7a7a7a", linewidth=0.55, linestyle=(0, (6, 4)), zorder=1)

            if ordered_groups:
                if has_secondary:
                    for subgroup_name in secondary_groups:
                        subgroup_df = plot_group_df.loc[plot_group_df["_subgroup_label"].astype(str) == subgroup_name]
                        for primary_group_name in ordered_groups:
                            group_points_df = subgroup_df.loc[
                                subgroup_df[primary_group_col].astype(str) == primary_group_name
                            ]
                            if group_points_df.empty:
                                continue
                            mask = group_points_df.index.to_numpy(dtype=int, copy=False)
                            ax.scatter(
                                x[mask],
                                y[mask],
                                s=10,
                                alpha=0.85,
                                color=color_map[subgroup_name],
                                marker=marker_map[primary_group_name],
                                edgecolors="white",
                                linewidths=0.45,
                                zorder=2,
                            )
                else:
                    for group_name in ordered_groups:
                        mask = plot_group_df[primary_group_col].astype(str).eq(group_name).to_numpy(dtype=bool, copy=False)
                        if not np.any(mask):
                            continue
                        ax.scatter(
                            x[mask],
                            y[mask],
                            s=9,
                            alpha=0.85,
                            color=color_map[group_name],
                            marker=marker_map.get(group_name, "o"),
                            edgecolors="white",
                            linewidths=0.45,
                            zorder=2,
                        )
            else:
                ax.scatter(
                    x,
                    y,
                    s=9,
                    alpha=0.85,
                    color=PALETTE["pca_scatter"],
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=2,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#6b7280")
            ax.set_facecolor("white")

    fig.suptitle(title, fontsize=23, fontweight="bold", y=0.985)
    if legend_handles:
        legend_ncol = 1 if len(legend_handles) <= 12 else 2 if len(legend_handles) <= 24 else 3
        fig.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.055, 0.06),
            ncol=legend_ncol,
            frameon=False,
            fontsize=11,
            handletextpad=0.45,
            columnspacing=1.0,
            labelspacing=0.5,
            borderaxespad=0.0,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.06,
        top=0.93,
        wspace=0.14,
        hspace=0.14,
    )
    _save_figure(fig, save_stem, cfg)


def plot_sample_dendrogram(adata, save_stem: str | Path, cfg) -> None:
    if adata.n_obs < 2:
        return

    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, ax = plt.subplots(figsize=(max(12, adata.n_obs * 0.12), 6))
    linkage_matrix = linkage(np.asarray(adata.X, dtype=np.float32), method="average")
    color_threshold = float(np.percentile(linkage_matrix[:, 2], 50))

    dendrogram(
        linkage_matrix,
        labels=adata.obs_names.tolist(),
        leaf_rotation=90,
        leaf_font_size=max(4, min(9, 800 // max(1, adata.n_obs))),
        color_threshold=color_threshold,
        above_threshold_color="#aaaaaa",
        ax=ax,
    )
    ax.set_title("Sample Clustering Dendrogram")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Euclidean Distance")
    _save_figure(fig, save_stem, cfg)



def plot_transcriptome_pca(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col=None,
        add_group_envelope=True,
    )


def plot_metabolome_pca(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col=None,
        add_group_envelope=True,
    )


def plot_transcriptome_pca_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_pca_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        add_group_envelope=False,
    )


def plot_metabolome_pca_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
        add_group_envelope=False,
    )

def plot_transcriptome_pca_pairs(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_pca_pairs_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
    )



def plot_metabolome_pca_pairs(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_pairs_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
    )


def plot_transcriptome_pca_pairs_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    _plot_pca_pairs_from_matrix(
        matrix=np.asarray(adata.X, dtype=np.float32),
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Transcriptome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


def plot_metabolome_pca_pairs_subgroups(adata, save_stem: str | Path, cfg, group_df: pd.DataFrame | None = None) -> None:
    metab_df = adata.obsm.get("metabolomics_scaled", adata.obsm.get("metabolomics"))
    matrix = metab_df.to_numpy(dtype=np.float32, copy=False) if isinstance(metab_df, pd.DataFrame) else np.asarray(metab_df, dtype=np.float32)
    _plot_pca_pairs_from_matrix(
        matrix=matrix,
        sample_names=adata.obs_names.astype(str).tolist(),
        title="Metabolome PCA Pairs Plot",
        save_stem=save_stem,
        cfg=cfg,
        group_df=group_df,
        primary_group_col="group1",
        secondary_group_col="group2",
    )


def _build_signed_count_summary(edge_df: pd.DataFrame, node_column: str) -> pd.DataFrame:
    """Aggregate node-level weighted degree and direction bias from T03 edges."""
    if edge_df.empty:
        return pd.DataFrame(
            columns=[
                node_column,
                "WeightedDegree",
                "PositiveEdgeCount",
                "NegativeEdgeCount",
                "DirectionBias",
            ]
        )

    work = edge_df.loc[:, [node_column, "EdgeWeight", "Sign"]].copy()
    work[node_column] = work[node_column].astype(str)
    work["EdgeWeight"] = pd.to_numeric(work["EdgeWeight"], errors="coerce").fillna(0.0)

    summary = work.groupby(node_column, sort=False)["EdgeWeight"].sum().rename("WeightedDegree").to_frame()
    summary["PositiveEdgeCount"] = (
        work["Sign"].astype(str).str.lower().eq("positive").groupby(work[node_column], sort=False).sum().astype(int)
    )
    summary["NegativeEdgeCount"] = (
        work["Sign"].astype(str).str.lower().eq("negative").groupby(work[node_column], sort=False).sum().astype(int)
    )

    total_counts = summary["PositiveEdgeCount"] + summary["NegativeEdgeCount"]
    summary["DirectionBias"] = np.where(
        total_counts > 0,
        (summary["PositiveEdgeCount"] - summary["NegativeEdgeCount"]) / total_counts,
        0.0,
    )
    return summary.reset_index()


def _compute_standardized_feature_variability(feature_df: pd.DataFrame) -> pd.Series:
    """Compute per-feature SD from the current standardized matrix without re-z-scoring."""
    if feature_df.empty:
        return pd.Series(dtype=float)

    values = feature_df.to_numpy(dtype=float, copy=False)
    variability = np.nanstd(values, axis=0, ddof=0)
    variability = np.where(np.isfinite(variability), variability, 0.0)
    variability = np.where(variability > 0, variability, 0.0)
    return pd.Series(variability, index=feature_df.columns.astype(str), dtype=float)


def _prepare_circos_node_tables(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ordered gene/metabolite node tables strictly from T03."""
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    required_columns = {"Gene", "Metabolite", "EdgeWeight", "Sign", "ModelSupportCount"}
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty or not required_columns.issubset(edge_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    edge_df = edge_df.copy()
    edge_df["Gene"] = edge_df["Gene"].astype(str)
    edge_df["Metabolite"] = edge_df["Metabolite"].astype(str)
    edge_df["EdgeWeight"] = pd.to_numeric(edge_df["EdgeWeight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    edge_df["ModelSupportCount"] = pd.to_numeric(edge_df["ModelSupportCount"], errors="coerce").fillna(0.0)
    edge_df["Sign"] = edge_df["Sign"].astype(str).str.lower()

    gene_df = _gene_expression_df(engine.adata)
    metab_df = _metabolomics_df(engine.adata)
    gene_mean_z = gene_df.mean(axis=0)
    metab_mean_z = metab_df.mean(axis=0)
    gene_variability = _compute_standardized_feature_variability(gene_df)
    metab_variability = _compute_standardized_feature_variability(metab_df)

    gene_summary = _build_signed_count_summary(edge_df, "Gene").rename(columns={"Gene": "Node"})
    gene_summary["NodeType"] = "gene"
    gene_summary["MeanZScore"] = gene_summary["Node"].map(gene_mean_z).fillna(0.0).astype(float)
    gene_summary["InterSampleVariability"] = gene_summary["Node"].map(gene_variability).fillna(0.0).astype(float)
    gene_summary["AbsDirectionBias"] = gene_summary["DirectionBias"].abs()
    gene_summary = gene_summary.sort_values(
        ["WeightedDegree", "AbsDirectionBias", "MeanZScore", "Node"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    metab_summary = _build_signed_count_summary(edge_df, "Metabolite").rename(columns={"Metabolite": "Node"})
    metab_summary["NodeType"] = "metabolite"
    metab_summary["MeanZScore"] = metab_summary["Node"].map(metab_mean_z).fillna(0.0).astype(float)
    metab_summary["InterSampleVariability"] = metab_summary["Node"].map(metab_variability).fillna(0.0).astype(float)
    metab_summary["AbsDirectionBias"] = metab_summary["DirectionBias"].abs()
    metab_summary = metab_summary.sort_values(
        ["WeightedDegree", "AbsDirectionBias", "MeanZScore", "Node"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return edge_df, gene_summary, metab_summary



def _build_circos_module_color_map(module_names: list[str]) -> dict[str, str]:
    ordered_modules = [str(name) for name in module_names if str(name).strip()]
    unique_modules = _ordered_unique_nonempty(ordered_modules)

    canonical_map = {
        "turquoise": "#40E0D0",
        "blue": "#1F77B4",
        "brown": "#8B4513",
        "yellow": "#FFD700",
        "green": "#2CA02C",
        "red": "#D62728",
        "black": "#000000",
        "pink": "#FFC0CB",
        "magenta": "#FF00FF",
        "purple": "#800080",
        "greenyellow": "#ADFF2F",
        "tan": "#D2B48C",
        "salmon": "#FA8072",
        "cyan": "#00FFFF",
        "midnightblue": "#191970",
        "lightcyan": "#E0FFFF",
        "royalblue": "#4169E1",
        "darkred": "#8B0000",
        "darkgreen": "#006400",
        "darkturquoise": "#00CED1",
        "darkgrey": "#A9A9A9",
        "orange": "#FFA500",
        "white": "#FFFFFF",
        "skyblue": "#87CEEB",
        "saddlebrown": "#8B4513",
        "steelblue": "#4682B4",
        "paleturquoise": "#AFEEEE",
        "violet": "#EE82EE",
        "darkorange": "#FF8C00",
        "darkmagenta": "#8B008B",
        "grey": "#E5E7EB",
    }

    fallback_palette = _categorical_colors(len(unique_modules))
    color_map: dict[str, str] = {}
    for idx, module_name in enumerate(unique_modules):
        key = str(module_name).strip().lower()
        color_map[module_name] = canonical_map.get(key, fallback_palette[idx % len(fallback_palette)] if fallback_palette else "#9ca3af")
    return color_map

def _attach_circos_module_annotations(engine, gene_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    work = gene_summary.copy()
    work["Module"] = "grey"
    work["ModuleColor"] = "#E5E7EB"
    work["ModuleSize"] = 0
    work["kME"] = np.nan
    work["IntramodularDegree"] = np.nan
    work["IsGrey"] = 1

    module_df = engine.ml_results.get("gene_module_assignment_df", pd.DataFrame())
    required_columns = {"Gene", "Module"}
    if isinstance(module_df, pd.DataFrame) and not module_df.empty and required_columns.issubset(module_df.columns):
        keep_cols = [
            "Gene",
            "Module",
            "ModuleColorHex",
            "ModuleSize",
            "kME",
            "IntramodularDegree",
            "IsGrey",
        ]
        module_keep = module_df.loc[:, [col for col in keep_cols if col in module_df.columns]].copy()
        module_keep = module_keep.rename(columns={"Gene": "Node"})
        module_keep["Node"] = module_keep["Node"].astype(str)
        module_keep["Module"] = module_keep["Module"].astype(str).replace("", "grey")
        work = work.merge(module_keep, on="Node", how="left", suffixes=("", "_Module"))

        work["Module"] = work.get("Module_Module", work["Module"]).fillna("grey").astype(str)
        work["ModuleSize"] = pd.to_numeric(work.get("ModuleSize_Module", 0), errors="coerce").fillna(0).astype(int)
        work["kME"] = pd.to_numeric(work.get("kME_Module", np.nan), errors="coerce").astype(float)
        work["IntramodularDegree"] = pd.to_numeric(work.get("IntramodularDegree_Module", np.nan), errors="coerce").astype(float)
        work["IsGrey"] = pd.to_numeric(work.get("IsGrey_Module", 1), errors="coerce").fillna(1).astype(int)

        if "ModuleColorHex_Module" in work.columns:
            work["ModuleColor"] = work["ModuleColorHex_Module"].fillna("#E5E7EB").astype(str)

        drop_cols = [col for col in work.columns if col.endswith("_Module")]
        if drop_cols:
            work = work.drop(columns=drop_cols)

    work["Module"] = work["Module"].fillna("grey").astype(str)
    work["IsGrey"] = (work["Module"].str.lower() == "grey").astype(int)

    non_grey = work.loc[work["IsGrey"] == 0, ["Module", "ModuleSize"]].drop_duplicates()
    module_order = non_grey.sort_values(
        ["ModuleSize", "Module"],
        ascending=[False, True],
        kind="mergesort",
    )["Module"].astype(str).tolist()
    if "grey" in work["Module"].astype(str).tolist():
        module_order.append("grey")

    module_color_map = _build_circos_module_color_map(module_order)
    missing_color_mask = work["ModuleColor"].isna() | work["ModuleColor"].astype(str).eq("")
    work.loc[missing_color_mask, "ModuleColor"] = work.loc[missing_color_mask, "Module"].map(module_color_map).fillna("#E5E7EB")
    work["ModuleColor"] = work["Module"].map(module_color_map).fillna(work["ModuleColor"]).fillna("#E5E7EB")

    work = work.sort_values(
        ["IsGrey", "ModuleSize", "Module", "kME", "IntramodularDegree", "WeightedDegree", "Node"],
        ascending=[True, False, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return work, module_color_map

def _compute_gene_module_spans(gene_summary: pd.DataFrame, layout: dict[str, dict[str, float | str]]) -> list[dict[str, object]]:
    if gene_summary.empty or not layout:
        return []

    spans: list[dict[str, object]] = []
    grouped = gene_summary.groupby("Module", sort=False)
    for module_name, group in grouped:
        nodes = [str(node) for node in group["Node"].astype(str).tolist() if str(node) in layout]
        if not nodes:
            continue
        theta_start = float(layout[nodes[0]]["theta_start"])
        theta_end = float(layout[nodes[-1]]["theta_end"])
        spans.append(
            {
                "Module": str(module_name),
                "ThetaStart": theta_start,
                "ThetaEnd": theta_end,
                "Color": str(group["ModuleColor"].iloc[0]),
                "GeneCount": int(len(nodes)),
            }
        )
    return spans

def _compute_circos_layout(gene_nodes: list[str], metabolite_nodes: list[str]) -> dict[str, dict[str, float | str]]:
    """Assign compact angular positions with genes and metabolites in two consecutive sectors."""
    n_gene = len(gene_nodes)
    n_metabolite = len(metabolite_nodes)
    n_total = n_gene + n_metabolite
    if n_total == 0:
        return {}

    full_circle = 2.0 * np.pi
    mean_item_span = full_circle / float(n_total)
    item_gap = min(np.deg2rad(0.45), mean_item_span * 0.10)
    group_gap = max(np.deg2rad(7.0), item_gap * 8.0)

    total_gap = max(0, n_total - 2) * item_gap + 2.0 * group_gap
    if total_gap >= full_circle * 0.92:
        item_gap = mean_item_span * 0.04
        group_gap = max(np.deg2rad(4.0), item_gap * 6.0)
        total_gap = max(0, n_total - 2) * item_gap + 2.0 * group_gap

    item_width = (full_circle - total_gap) / float(n_total)
    if item_width <= 0:
        item_gap = 0.0
        group_gap = 0.0
        item_width = full_circle / float(n_total)

    layout: dict[str, dict[str, float | str]] = {}
    current_angle = np.pi * 0.76 + group_gap / 2.0

    def _assign(node_ids: list[str], node_type: str, after_group_gap: float) -> float:
        nonlocal current_angle
        for idx, node_id in enumerate(node_ids):
            theta_start = current_angle
            theta_end = theta_start + item_width
            layout[str(node_id)] = {
                "theta_start": theta_start,
                "theta_end": theta_end,
                "theta_mid": 0.5 * (theta_start + theta_end),
                "node_type": node_type,
            }
            current_angle = theta_end
            if idx < len(node_ids) - 1:
                current_angle += item_gap
            else:
                current_angle += after_group_gap
        return current_angle

    if n_gene > 0:
        _assign(gene_nodes, "gene", group_gap)
    if n_metabolite > 0:
        _assign(metabolite_nodes, "metabolite", group_gap)

    return layout


def _polar_to_xy(theta: float, radius: float) -> tuple[float, float]:
    return float(radius * np.cos(theta)), float(radius * np.sin(theta))


def _add_annular_segment(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    facecolor,
    edgecolor: str = "#ffffff",
    linewidth: float = 0.35,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    if r_outer <= r_inner:
        return

    patch = Wedge(
        center=(0.0, 0.0),
        r=float(r_outer),
        theta1=float(np.degrees(theta_start)),
        theta2=float(np.degrees(theta_end)),
        width=float(r_outer - r_inner),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    patch.set_zorder(zorder)
    ax.add_patch(patch)


def _add_circos_link(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    radius: float,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int = 0,
) -> None:
    start_xy = np.asarray(_polar_to_xy(theta_start, radius), dtype=float)
    end_xy = np.asarray(_polar_to_xy(theta_end, radius), dtype=float)

    path = MplPath(
        [
            tuple(start_xy),
            tuple(start_xy * 0.18),
            tuple(end_xy * 0.18),
            tuple(end_xy),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    patch.set_zorder(zorder)
    ax.add_patch(patch)


def _prepare_network_df(engine, tier: str, max_edges: int | None = None) -> pd.DataFrame:
    if tier == "total":
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    else:
        edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())

    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return pd.DataFrame()

    ordered = edge_df.sort_values(
        ["EdgeWeight", "RRARank", "ModelSupportCount", "ScreenSupportCount"],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    if max_edges is not None:
        ordered = ordered.head(max(1, int(max_edges)))
    return ordered.copy()


def _plot_association_network(edge_df: pd.DataFrame, title: str, save_stem: str | Path, cfg) -> None:
    if edge_df.empty:
        return

    genes = (
        edge_df.groupby("Gene")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    )
    metabolites = (
        edge_df.groupby("Metabolite")["EdgeWeight"].max().sort_values(ascending=False).index.astype(str).tolist()
    )

    if not genes or not metabolites:
        return

    fig_width = max(10, 0.17 * (len(genes) + len(metabolites)) + 6)
    fig_height = max(6.5, 0.24 * max(len(genes), len(metabolites)) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    gene_x = 0.0
    metabolite_x = 1.0
    gene_y = np.linspace(len(genes), 1, len(genes))
    metabolite_y = np.linspace(len(metabolites), 1, len(metabolites))
    gene_pos = {gene: (gene_x, float(y)) for gene, y in zip(genes, gene_y)}
    metabolite_pos = {metab: (metabolite_x, float(y)) for metab, y in zip(metabolites, metabolite_y)}

    for row in edge_df.itertuples(index=False):
        gene = str(row.Gene)
        metab = str(row.Metabolite)
        if gene not in gene_pos or metab not in metabolite_pos:
            continue
        x1, y1 = gene_pos[gene]
        x2, y2 = metabolite_pos[metab]
        color = PALETTE["edge_positive"] if str(row.Sign) == "positive" else PALETTE["edge_negative"]
        width = 0.8 + 4.2 * float(row.EdgeWeight)
        opacity = min(
            0.95,
            0.20
            + 0.35 * (float(row.ModelSupportCount) / 2.0)
            + 0.20 * (float(row.ScreenSupportCount) / 3.0),
        )
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=opacity, solid_capstyle="round", zorder=1)

    ax.scatter(
        [gene_x] * len(genes),
        gene_y,
        s=70,
        c=PALETTE["gene"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="Gene",
    )
    ax.scatter(
        [metabolite_x] * len(metabolites),
        metabolite_y,
        s=85,
        c=PALETTE["metabolite"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="Metabolite",
    )

    for gene, (x, y) in gene_pos.items():
        ax.text(x - 0.02, y, gene, ha="right", va="center", fontsize=8.5)
    for metab, (x, y) in metabolite_pos.items():
        ax.text(x + 0.02, y, metab, ha="left", va="center", fontsize=8.5)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0, max(len(genes), len(metabolites)) + 1.2)
    ax.set_xticks([gene_x, metabolite_x])
    ax.set_xticklabels(["Genes", "Metabolites"])
    ax.set_yticks([])
    ax.set_title(title)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["gene"], markersize=8, label="Gene"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["metabolite"], markersize=8, label="Metabolite"),
        plt.Line2D([0], [0], color=PALETTE["edge_positive"], lw=2.5, label="Positive association"),
        plt.Line2D([0], [0], color=PALETTE["edge_negative"], lw=2.5, label="Negative association"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    _save_figure(fig, save_stem, cfg)


def plot_total_association_network(engine, save_stem: str | Path, cfg) -> None:
    edge_df = _prepare_network_df(engine, tier="total", max_edges=cfg.network_plot_top_edges)
    _plot_association_network(edge_df, "Total Gene-Metabolite Association Network", save_stem, cfg)


def plot_high_confidence_network(engine, save_stem: str | Path, cfg) -> None:
    edge_df = _prepare_network_df(engine, tier="high_confidence", max_edges=cfg.network_plot_top_edges)
    _plot_association_network(edge_df, "High-Confidence Gene-Metabolite Association Network", save_stem, cfg)




def _robust_abs_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    scale = float(np.nanpercentile(np.abs(arr), 95))
    return max(scale, 1e-6)


def _positive_scale(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    scale = float(np.nanmax(arr))
    return max(scale, 1e-6)


def _prepare_group1_mean_track_data(feature_df: pd.DataFrame, group_df: pd.DataFrame | None) -> dict[str, object] | None:
    if feature_df.empty:
        return None

    feature_work = feature_df.copy()
    feature_work.index = feature_work.index.astype(str)

    if group_df is None or "sample_id" not in group_df.columns or "group1" not in group_df.columns:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
        }

    group_work = group_df.copy()
    group_work["sample_id"] = group_work["sample_id"].astype(str).str.strip()
    group_work["group1"] = group_work["group1"].astype(str).str.strip()
    group_work = group_work.loc[group_work["sample_id"].isin(feature_work.index)].copy()
    if group_work.empty:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
        }

    group_work = group_work.drop_duplicates(subset=["sample_id"], keep="first").set_index("sample_id", drop=True)
    aligned_samples = [sample for sample in feature_work.index.tolist() if sample in group_work.index]
    if not aligned_samples:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
        }

    feature_work = feature_work.loc[aligned_samples].copy()
    aligned_group = group_work.reindex(aligned_samples).copy()
    group1_order = _ordered_unique_nonempty(aligned_group["group1"].tolist())
    if not group1_order:
        mean_series = feature_work.mean(axis=0).astype(float)
        return {
            "mode": "overall_mean",
            "feature_to_values": {str(feature): [float(mean_series.get(feature, np.nan))] for feature in feature_work.columns.astype(str)},
            "abs_scale": _robust_abs_scale(mean_series.tolist()),
            "group1_order": [],
        }

    agg_input = feature_work.copy()
    agg_input["group1"] = aligned_group["group1"].astype(str).to_numpy()
    agg_df = agg_input.groupby("group1", sort=False)[feature_work.columns.astype(str).tolist()].mean()
    feature_to_values = {
        str(feature): [float(agg_df.loc[group_name, feature]) for group_name in group1_order if group_name in agg_df.index]
        for feature in feature_work.columns.astype(str).tolist()
    }
    flattened = [float(v) for values in feature_to_values.values() for v in values if np.isfinite(v)]
    return {
        "mode": "group1_mean",
        "feature_to_values": feature_to_values,
        "abs_scale": _robust_abs_scale(flattened),
        "group1_order": group1_order,
    }


def _draw_track_baseline(ax: plt.Axes, theta_start: float, theta_end: float, radius: float, *, color: str = "#d1d5db", linewidth: float = 0.18, alpha: float = 1.0, zorder: float = 2.7) -> None:
    n_points = 32
    thetas = np.linspace(theta_start, theta_end, n_points)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_group1_scatter_track(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    values: list[float],
    value_scale: float,
    random_state: int,
    zorder: float = 3.1,
) -> None:
    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        r_inner,
        r_outer,
        facecolor="#fbfbfb",
        edgecolor="#eef2f7",
        linewidth=0.14,
        alpha=1.0,
        zorder=int(zorder),
    )
    r_mid = 0.5 * (r_inner + r_outer)
    _draw_track_baseline(ax, theta_start, theta_end, r_mid, color="#d1d5db", linewidth=0.18, alpha=0.9, zorder=zorder)

    clean_values = [float(v) for v in values if np.isfinite(v)]
    if not clean_values:
        return

    rng_seed = int(random_state + round(float(theta_start) * 1e6)) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)

    theta_width = float(theta_end - theta_start)
    span_scale = max(theta_width * 0.06, np.deg2rad(0.08))
    radial_half_span = 0.42 * (r_outer - r_inner)
    scale = max(float(value_scale), 1e-6)

    for value in clean_values:
        theta = 0.5 * (theta_start + theta_end) + float(rng.uniform(-span_scale, span_scale))
        clipped = float(np.clip(value, -scale, scale))
        radius = r_mid + (clipped / scale) * radial_half_span
        x, y = _polar_to_xy(theta, radius)
        ax.scatter([x], [y], s=5.0, c=["#6b7280"], edgecolors="none", alpha=0.92, zorder=zorder + 0.15)


def _draw_mean_hist_track(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    r_inner: float,
    r_outer: float,
    *,
    value: float,
    value_scale: float,
    color: str = "#6b7280",
    zorder: float = 3.0,
) -> None:
    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        r_inner,
        r_outer,
        facecolor="#fbfbfb",
        edgecolor="#eef2f7",
        linewidth=0.14,
        alpha=1.0,
        zorder=int(zorder),
    )
    r_mid = 0.5 * (r_inner + r_outer)
    _draw_track_baseline(ax, theta_start, theta_end, r_mid, color="#d1d5db", linewidth=0.18, alpha=0.9, zorder=zorder)

    scale = max(float(value_scale), 1e-6)
    clipped = float(np.clip(value, -scale, scale))
    if clipped >= 0:
        bar_inner = r_mid
        bar_outer = r_mid + (clipped / scale) * (r_outer - r_mid)
    else:
        bar_inner = r_mid + (clipped / scale) * (r_mid - r_inner)
        bar_outer = r_mid

    _add_annular_segment(
        ax,
        theta_start,
        theta_end,
        bar_inner,
        bar_outer,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.88,
        zorder=int(zorder + 0.1),
    )


def _prepare_metabolite_module_core_map(engine) -> pd.Series:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if isinstance(assoc_df, pd.DataFrame) and not assoc_df.empty and {"Metabolite", "SpearmanRho"}.issubset(assoc_df.columns):
        work = assoc_df.copy()
        work["Metabolite"] = work["Metabolite"].astype(str)
        work["AbsRho"] = pd.to_numeric(work["SpearmanRho"], errors="coerce").abs()
        best = work.groupby("Metabolite", sort=False)["AbsRho"].max()
        return best.astype(float)
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if isinstance(edge_df, pd.DataFrame) and not edge_df.empty and {"Metabolite", "EdgeWeight"}.issubset(edge_df.columns):
        fallback = edge_df.groupby("Metabolite", sort=False)["EdgeWeight"].sum()
        return fallback.astype(float)
    return pd.Series(dtype=float)


def _prepare_module_legend_items(gene_summary: pd.DataFrame) -> list[tuple[str, str]]:
    if gene_summary.empty or "Module" not in gene_summary.columns:
        return []
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for row in gene_summary.loc[:, ["Module", "ModuleColor"]].drop_duplicates().itertuples(index=False):
        module_name = str(row.Module)
        if module_name in seen:
            continue
        seen.add(module_name)
        items.append((module_name, str(row.ModuleColor)))
    return items


def _add_corner_module_legend(
    ax: plt.Axes,
    legend_items: list[tuple[str, str]],
    *,
    x_left: float,
    y_top: float,
    row_height: float = 0.072,
    swatch_width: float = 0.12,
    swatch_height: float = 0.028,
) -> None:
    if not legend_items:
        return

    for idx, (module_name, module_color) in enumerate(legend_items):
        y = y_top - idx * row_height
        rect = plt.Rectangle(
            (x_left, y - 0.5 * swatch_height),
            swatch_width,
            swatch_height,
            facecolor=module_color,
            edgecolor="#9ca3af",
            linewidth=0.3,
            zorder=7,
        )
        ax.add_patch(rect)
        ax.text(
            x_left + swatch_width + 0.03,
            y,
            module_name,
            ha="left",
            va="center",
            fontsize=8.0,
            color="#374151",
            zorder=7,
        )


def plot_compressed_circos_network(engine, save_stem: str | Path, cfg) -> None:
    """Plot a compact static Circos figure using only T03 nodes and edges."""
    edge_df, gene_summary, metabolite_summary = _prepare_circos_node_tables(engine)
    if edge_df.empty or gene_summary.empty or metabolite_summary.empty:
        return

    gene_summary, _module_color_map = _attach_circos_module_annotations(engine, gene_summary)

    metabolite_module_core = _prepare_metabolite_module_core_map(engine)
    metabolite_summary = metabolite_summary.copy()
    metabolite_summary["Module"] = ""
    metabolite_summary["ModuleColor"] = "#c9ad85"
    metabolite_summary["ModuleCore"] = metabolite_summary["Node"].map(metabolite_module_core).astype(float)

    gene_summary["ModuleCore"] = pd.to_numeric(gene_summary.get("kME", np.nan), errors="coerce").abs()
    gene_nodes = gene_summary["Node"].astype(str).tolist()
    metabolite_nodes = metabolite_summary["Node"].astype(str).tolist()
    layout = _compute_circos_layout(gene_nodes, metabolite_nodes)
    if not layout:
        return

    node_df = pd.concat([gene_summary, metabolite_summary], ignore_index=True)
    node_df["Node"] = node_df["Node"].astype(str)

    gene_mean_scale = _robust_abs_scale(gene_summary["MeanZScore"])
    metabolite_mean_scale = _robust_abs_scale(metabolite_summary["MeanZScore"])
    gene_degree_scale = _positive_scale(gene_summary["WeightedDegree"])
    metabolite_degree_scale = _positive_scale(metabolite_summary["WeightedDegree"])
    gene_core_scale = _positive_scale(gene_summary["ModuleCore"])
    metabolite_core_scale = _positive_scale(metabolite_summary["ModuleCore"])

    gene_mean_norm = colors.TwoSlopeNorm(vmin=-gene_mean_scale, vcenter=0.0, vmax=gene_mean_scale)
    metabolite_mean_norm = colors.TwoSlopeNorm(vmin=-metabolite_mean_scale, vcenter=0.0, vmax=metabolite_mean_scale)
    bias_norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    mean_cmap = plt.get_cmap("RdBu_r")
    bias_cmap = plt.get_cmap("RdBu_r")

    group_df = _load_pca_group_table(cfg)
    gene_track_data = _prepare_group1_mean_track_data(_gene_expression_df(engine.adata), group_df)
    metabolite_track_data = _prepare_group1_mean_track_data(_metabolomics_df(engine.adata), group_df)

    radii = {
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

    fig, ax = plt.subplots(figsize=(11.7, 10.8))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    edge_ordered = edge_df.sort_values(
        ["EdgeWeight", "ModelSupportCount", "Gene", "Metabolite"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    support_min = float(edge_ordered["ModelSupportCount"].min()) if not edge_ordered.empty else 0.0
    support_max = float(edge_ordered["ModelSupportCount"].max()) if not edge_ordered.empty else 1.0

    for row in edge_ordered.itertuples(index=False):
        gene_id = str(row.Gene)
        metabolite_id = str(row.Metabolite)
        if gene_id not in layout or metabolite_id not in layout:
            continue

        edge_weight = float(np.clip(getattr(row, "EdgeWeight", 0.0), 0.0, None))
        line_width = 0.18 + 1.72 * np.sqrt(min(1.0, edge_weight))

        model_support = float(getattr(row, "ModelSupportCount", 0.0))
        if support_max > support_min:
            line_alpha = 0.05 + 0.30 * (model_support - support_min) / (support_max - support_min)
        else:
            line_alpha = 0.22 if support_max > 0 else 0.08

        line_color = PALETTE["edge_positive"] if str(row.Sign).lower() == "positive" else PALETTE["edge_negative"]
        _add_circos_link(
            ax,
            float(layout[gene_id]["theta_mid"]),
            float(layout[metabolite_id]["theta_mid"]),
            radii["link_radius"],
            color=line_color,
            linewidth=line_width,
            alpha=float(np.clip(line_alpha, 0.04, 0.92)),
            zorder=0,
        )

    for row in node_df.itertuples(index=False):
        node_id = str(row.Node)
        geometry = layout.get(node_id)
        if geometry is None:
            continue

        theta_start = float(geometry["theta_start"])
        theta_end = float(geometry["theta_end"])
        node_type = str(row.NodeType)
        mean_value = float(row.MeanZScore)
        degree_value = float(max(0.0, row.WeightedDegree))
        direction_bias = float(np.clip(row.DirectionBias, -1.0, 1.0))
        core_value = float(max(0.0, getattr(row, "ModuleCore", np.nan))) if pd.notna(getattr(row, "ModuleCore", np.nan)) else 0.0

        if node_type == "gene":
            outer_color = getattr(row, "ModuleColor", "#7db8ab")
            mean_norm = gene_mean_norm
            degree_scale = gene_degree_scale
            core_scale = gene_core_scale
            track_data = gene_track_data
            core_color = outer_color
        else:
            outer_color = "#c9ad85"
            mean_norm = metabolite_mean_norm
            degree_scale = metabolite_degree_scale
            core_scale = metabolite_core_scale
            track_data = metabolite_track_data
            core_color = "#8c6d46"

        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["outer_strip_inner"],
            radii["outer_strip_outer"],
            facecolor=outer_color,
            edgecolor="#ffffff",
            linewidth=0.45,
            alpha=1.0,
            zorder=4,
        )

        mean_color = mean_cmap(mean_norm(mean_value))
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_meanheat_inner"],
            radii["track_meanheat_outer"],
            facecolor=mean_color,
            edgecolor="#ffffff",
            linewidth=0.22,
            alpha=1.0,
            zorder=3.2,
        )

        track_values = track_data["feature_to_values"].get(node_id, []) if track_data is not None else []
        if track_data is not None and str(track_data.get("mode", "")) == "group1_mean":
            _draw_group1_scatter_track(
                ax,
                theta_start,
                theta_end,
                radii["track_meanbar_inner"],
                radii["track_meanbar_outer"],
                values=list(track_values),
                value_scale=float(track_data.get("abs_scale", 1.0)),
                random_state=int(getattr(cfg, "random_state", 42)),
                zorder=3.45,
            )
        else:
            mean_value_for_bar = float(track_values[0]) if track_values else float(mean_value)
            _draw_mean_hist_track(
                ax,
                theta_start,
                theta_end,
                radii["track_meanbar_inner"],
                radii["track_meanbar_outer"],
                value=mean_value_for_bar,
                value_scale=float(track_data.get("abs_scale", 1.0) if track_data is not None else max(gene_mean_scale, metabolite_mean_scale)),
                color="#6b7280",
                zorder=3.45,
            )

        degree_outer = radii["track_degree_inner"] + (radii["track_degree_outer"] - radii["track_degree_inner"]) * min(
            1.0, degree_value / max(degree_scale, 1e-6)
        )
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_degree_inner"],
            degree_outer,
            facecolor="#4b5563",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=2.3,
        )

        core_outer = radii["track_core_inner"] + (radii["track_core_outer"] - radii["track_core_inner"]) * min(
            1.0, core_value / max(core_scale, 1e-6)
        )
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_core_inner"],
            core_outer,
            facecolor=core_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=1.8,
        )

        bias_color = bias_cmap(bias_norm(direction_bias))
        _add_annular_segment(
            ax,
            theta_start,
            theta_end,
            radii["track_bias_inner"],
            radii["track_bias_outer"],
            facecolor=bias_color,
            edgecolor="#ffffff",
            linewidth=0.22,
            alpha=1.0,
            zorder=1.0,
        )

    legend_items = _prepare_module_legend_items(gene_summary)
    _add_corner_module_legend(
        ax,
        legend_items,
        x_left=-1.48,
        y_top=-0.46,
        row_height=0.072,
        swatch_width=0.11,
        swatch_height=0.026,
    )

    outer_limit_x = 1.58
    outer_limit_y = 1.12
    ax.set_xlim(-outer_limit_x, 1.12)
    ax.set_ylim(-outer_limit_y, outer_limit_y)
    _save_figure(fig, save_stem, cfg)

def generate_markdown_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())

    lines = [
        f"# DeepOmics Report: {cfg.project_name}",
        "",
        "## Run Summary",
        f"- Samples: {engine.adata.n_obs}",
        f"- Genes: {engine.adata.n_vars}",
        f"- Metabolites: {len(engine.adata.uns.get('metabolite_names', []))}",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Main Tables",
        f"- `{TABLE_FILE_PREFIXES['gene_scores']}`: complete metabolite-level gene scoring table after three-way screening and two-model ranking.",
        f"- `{TABLE_FILE_PREFIXES['total_network']}`: total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.",
        f"- `{TABLE_FILE_PREFIXES['high_confidence_network']}`: high-confidence subnetwork of the total association network after RRA and multi-evidence filtering.",
        f"- `{TABLE_FILE_PREFIXES['key_gene_summary']}`: merged key-gene summary across metabolites.",
        f"- `{TABLE_FILE_PREFIXES['metabolite_summary']}`: metabolite-level candidate and network summary.",
        f"- `{TABLE_FILE_PREFIXES['cytoscape_network']}`: Cytoscape-ready edge table with updated association fields.",
        "",
        "## Metabolite-Level Summary",
        _df_to_markdown(metabolite_summary, max_rows=20),
        "",
        "## Key Gene Summary",
        _df_to_markdown(key_gene_summary, max_rows=20),
        "",
        "## Generated Figures",
        f"- `plots/{FIGURE_FILE_PREFIXES['sample_clustering_dendrogram']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['compressed_circos_network']}.pdf|svg|png`",
        "- `DeepOmics_Interactive_Report.html`",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame()).head(50)
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame()).head(50)

    table_rows = "".join([
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['gene_scores'])}</code></td><td>Complete metabolite-level gene scoring table after three-way screening and two-model ranking.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['total_network'])}</code></td><td>Total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['high_confidence_network'])}</code></td><td>High-confidence subnetwork of the total association network after RRA and multi-evidence filtering.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['key_gene_summary'])}</code></td><td>Merged key-gene summary across metabolites.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['metabolite_summary'])}</code></td><td>Metabolite-level candidate and network summary.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['cytoscape_network'])}</code></td><td>Cytoscape-ready edge table with updated association fields.</td></tr>",
    ])

    figure_rows = "".join([
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['sample_clustering_dendrogram'])}.pdf|svg|png</code></td><td>Sample clustering dendrogram.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca'])}.pdf|svg|png</code></td><td>Transcriptome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca'])}.pdf|svg|png</code></td><td>Metabolome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca_pairs'])}.pdf|svg|png</code></td><td>Transcriptome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca_pairs'])}.pdf|svg|png</code></td><td>Metabolome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs'])}.pdf|svg|png</code></td><td>Top association pair scatter panels ranked by EdgeWeight.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap'])}.pdf|svg|png</code></td><td>Module-metabolite association heatmap colored by Spearman rho with significance stars.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['compressed_circos_network'])}.pdf|svg|png</code></td><td>Compact Circos overview using all unique genes and metabolites from T03 only.</td></tr>",
    ])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Report - {html.escape(cfg.project_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; color: #111827; }}
    h1, h2 {{ color: #1f2937; }}
    .hero {{
      background: #f8fafc;
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .link-card {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 24px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>DeepOmics Report: {html.escape(cfg.project_name)}</h1>
    <p>This page summarizes structured association tables and figure file names.</p>
  </div>

  <h2>Run Summary</h2>
  <ul>
    <li>Samples: {engine.adata.n_obs}</li>
    <li>Genes: {engine.adata.n_vars}</li>
    <li>Metabolites: {len(engine.adata.uns.get("metabolite_names", []))}</li>
    <li>Output directory: <code>{html.escape(str(cfg.output_dir))}</code></li>
  </ul>

  <div class="link-card">
    <strong>Interactive report:</strong>
    Open <a href="DeepOmics_Interactive_Report.html"><code>DeepOmics_Interactive_Report.html</code></a>
    for lightweight browser-native visualization preview and SVG export.
  </div>

  <h2>Main Output Tables</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Generated Figures</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{figure_rows}</tbody>
  </table>

  <h2>Metabolite-Level Summary</h2>
  {metabolite_summary.to_html(index=False, escape=True)}

  <h2>Key Gene Summary</h2>
  {key_gene_summary.to_html(index=False, escape=True)}
</body>
</html>
"""
    Path(report_path).write_text(html_text, encoding="utf-8")



def plot_top_edge_scatter_panels(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    edge_df = engine.ml_results.get("high_confidence_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        edge_df = engine.ml_results.get("total_association_network_df", pd.DataFrame())
    if not isinstance(edge_df, pd.DataFrame) or edge_df.empty:
        return

    top_n = int(top_n or cfg.top_pairs_plot_n)
    ranked = edge_df.sort_values(["EdgeWeight", "RRARank"], ascending=[False, True], kind="mergesort").head(top_n)
    if ranked.empty:
        return

    metab_df = engine.metabolomics_df() if hasattr(engine, "metabolomics_df") else _metabolomics_df(engine.adata)
    gene_df = engine.gene_expression_df() if hasattr(engine, "gene_expression_df") else _gene_expression_df(engine.adata)

    n_panels = len(ranked)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.0, 4.15 * n_rows))
    fig._skip_default_tight_layout = True
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, ranked.itertuples(index=False)):
        gene = str(row.Gene)
        metab = str(row.Metabolite)
        if gene not in gene_df.columns or metab not in metab_df.columns:
            ax.axis("off")
            continue

        x = gene_df[gene].to_numpy(dtype=float, copy=False)
        y = metab_df[metab].to_numpy(dtype=float, copy=False)
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if x.size < 2 or y.size < 2:
            ax.axis("off")
            continue

        sns.regplot(
            x=x,
            y=y,
            ax=ax,
            color=PALETTE["gene"],
            scatter_kws={"s": 26, "alpha": 0.85, "edgecolor": "white", "linewidths": 0.35},
            line_kws={"lw": 1.3},
            ci=95,
        )

        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            r_value = float(np.corrcoef(x, y)[0, 1])
            r_text = f"r = {r_value:.2f}" if np.isfinite(r_value) else "r = NA"
        else:
            r_text = "r = NA"

        ax.text(
            0.03,
            0.97,
            r_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.set_title(f"{gene} vs {metab}", fontsize=10.5, pad=9)
        ax.set_xlabel(gene)
        ax.set_ylabel(metab)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Top Gene-Metabolite Association Pairs", y=0.985, fontsize=13)
    fig.subplots_adjust(top=0.84, hspace=0.62, wspace=0.38)
    try:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    except Exception:
        pass
    _save_figure(fig, save_stem, cfg)


def _significance_star(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    if value <= 0.05:
        return "*"
    return ""


def plot_module_metabolite_association_heatmap(engine, save_stem: str | Path, cfg) -> None:
    assoc_df = engine.ml_results.get("module_metabolite_assoc_df", pd.DataFrame())
    if not isinstance(assoc_df, pd.DataFrame) or assoc_df.empty:
        return

    required_columns = {"Module", "Metabolite", "SpearmanRho"}
    if not required_columns.issubset(assoc_df.columns):
        return

    plot_df = assoc_df.copy()
    plot_df["Module"] = plot_df["Module"].astype(str)
    plot_df["Metabolite"] = plot_df["Metabolite"].astype(str)
    plot_df["SpearmanRho"] = pd.to_numeric(plot_df["SpearmanRho"], errors="coerce")
    if "FDR" in plot_df.columns:
        plot_df["FDR"] = pd.to_numeric(plot_df["FDR"], errors="coerce")
    if "PValue" in plot_df.columns:
        plot_df["PValue"] = pd.to_numeric(plot_df["PValue"], errors="coerce")

    non_grey_df = plot_df.loc[plot_df["Module"].str.lower() != "grey"].copy()
    if not non_grey_df.empty:
        plot_df = non_grey_df

    significance_column = "FDR" if ("FDR" in plot_df.columns and plot_df["FDR"].notna().any()) else "PValue"
    if significance_column not in plot_df.columns:
        plot_df["PValue"] = np.nan
        significance_column = "PValue"

    module_summary_df = engine.ml_results.get("module_summary_df", pd.DataFrame())
    if isinstance(module_summary_df, pd.DataFrame) and not module_summary_df.empty and "Module" in module_summary_df.columns:
        module_order = [
            str(module_name)
            for module_name in module_summary_df["Module"].astype(str).tolist()
            if str(module_name) in set(plot_df["Module"].tolist())
        ]
    else:
        module_order = []

    if not module_order:
        module_order = (
            plot_df.assign(
                _AbsRho=plot_df["SpearmanRho"].abs(),
                _SigRank=pd.to_numeric(plot_df[significance_column], errors="coerce").fillna(1.0),
            )
            .sort_values(
                ["_SigRank", "_AbsRho", "Module"],
                ascending=[True, False, True],
                kind="mergesort",
            )["Module"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

    metabolite_order = (
        plot_df.assign(
            _AbsRho=plot_df["SpearmanRho"].abs(),
            _SigRank=pd.to_numeric(plot_df[significance_column], errors="coerce").fillna(1.0),
        )
        .sort_values(
            ["_SigRank", "_AbsRho", "Metabolite"],
            ascending=[True, False, True],
            kind="mergesort",
        )["Metabolite"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    rho_matrix = (
        plot_df.pivot(index="Module", columns="Metabolite", values="SpearmanRho")
        .reindex(index=module_order, columns=metabolite_order)
    )
    sig_matrix = (
        plot_df.pivot(index="Module", columns="Metabolite", values=significance_column)
        .reindex(index=module_order, columns=metabolite_order)
    )

    if rho_matrix.empty:
        return

    annotation_matrix = (
        sig_matrix.map(_significance_star)
        if hasattr(sig_matrix, "map")
        else sig_matrix.applymap(_significance_star)
    )

    finite_rho = rho_matrix.to_numpy(dtype=float)
    finite_rho = finite_rho[np.isfinite(finite_rho)]
    vmax = float(np.nanmax(np.abs(finite_rho))) if finite_rho.size > 0 else 1.0
    vmax = max(vmax, 0.25)

    fig_width = max(9.0, min(28.0, 0.42 * max(1, rho_matrix.shape[1]) + 4.5))
    fig_height = max(4.5, min(18.0, 0.50 * max(1, rho_matrix.shape[0]) + 2.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        rho_matrix,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#f3f4f6",
        annot=annotation_matrix,
        fmt="",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Spearman rho"},
        mask=rho_matrix.isna(),
    )

    metric_label = "FDR" if significance_column == "FDR" else "P value"
    ax.set_title(f"Module-Metabolite Association Heatmap\nColor: Spearman rho; stars: {metric_label}", pad=8)
    ax.set_xlabel("Metabolite")
    ax.set_ylabel("Module")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save_figure(fig, save_stem, cfg)


def plot_metabolite_model_support_summary(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    summary_df = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return

    plot_df = summary_df.sort_values(
        ["HighConfidenceEdges", "TotalAssociationEdges", "CandidateGenes"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(int(top_n or cfg.support_plot_top_metabolites))
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(plot_df)), 6.5))
    bars_total = ax.bar(x - width, plot_df["TotalAssociationEdges"], width=width, label="Total network edges", color=PALETTE["bar_total"])
    bars_dual = ax.bar(x, plot_df["DualModelEdges"], width=width, label="Dual-model supported edges", color=PALETTE["bar_dual_model"])
    bars_high = ax.bar(x + width, plot_df["HighConfidenceEdges"], width=width, label="High-confidence edges", color=PALETTE["bar_high_conf"])

    for container in (bars_total, bars_dual, bars_high):
        ax.bar_label(container, fontsize=8.5, padding=2, fmt="%d")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Metabolite"], rotation=90)
    ax.set_ylabel("Edge count")
    ax.set_title(f"Metabolite-Level Model Support Summary (Top {len(plot_df)})")
    ax.legend(ncol=3, loc="upper right")
    _save_figure(fig, save_stem, cfg)


def plot_top_key_genes(engine, save_stem: str | Path, cfg, top_n: int | None = None) -> None:
    summary_df = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return

    top_df = summary_df.head(int(top_n or cfg.top_key_genes_plot_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top_df))))
    bars = ax.barh(top_df["Gene"], top_df["AssociatedMetaboliteCount"], color=PALETTE["gene"])
    labels = [f"{float(value):.0f}" for value in top_df["AssociatedMetaboliteCount"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_title("Top Key Genes Across Metabolites")
    ax.set_xlabel("Associated Metabolite Count")
    ax.set_ylabel("Gene")
    _save_figure(fig, save_stem, cfg)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"

    preview = df.head(max_rows).copy().fillna("")
    columns = preview.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in preview.iterrows()]
    return "\n".join([header, sep, *rows])


def generate_markdown_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame())
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame())

    lines = [
        f"# DeepOmics Report: {cfg.project_name}",
        "",
        "## Run Summary",
        f"- Samples: {engine.adata.n_obs}",
        f"- Genes: {engine.adata.n_vars}",
        f"- Metabolites: {len(engine.adata.uns.get('metabolite_names', []))}",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Main Tables",
        f"- `{TABLE_FILE_PREFIXES['gene_scores']}`: complete metabolite-level gene scoring table after three-way screening and two-model ranking.",
        f"- `{TABLE_FILE_PREFIXES['total_network']}`: total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.",
        f"- `{TABLE_FILE_PREFIXES['high_confidence_network']}`: high-confidence subnetwork of the total association network after RRA and multi-evidence filtering.",
        f"- `{TABLE_FILE_PREFIXES['key_gene_summary']}`: merged key-gene summary across metabolites.",
        f"- `{TABLE_FILE_PREFIXES['metabolite_summary']}`: metabolite-level candidate and network summary.",
        f"- `{TABLE_FILE_PREFIXES['cytoscape_network']}`: Cytoscape-ready edge table with updated association fields.",
        "",
        "## Metabolite-Level Summary",
        _df_to_markdown(metabolite_summary, max_rows=20),
        "",
        "## Key Gene Summary",
        _df_to_markdown(key_gene_summary, max_rows=20),
        "",
        "## Generated Figures",
        f"- `plots/{FIGURE_FILE_PREFIXES['sample_clustering_dendrogram']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap']}.pdf|svg|png`",
        f"- `plots/{FIGURE_FILE_PREFIXES['compressed_circos_network']}.pdf|svg|png`",
        "- `DeepOmics_Interactive_Report.html`",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(engine, cfg, report_path: str | Path) -> None:
    metabolite_summary = engine.ml_results.get("metabolite_summary", pd.DataFrame()).head(50)
    key_gene_summary = engine.ml_results.get("key_gene_summary_df", pd.DataFrame()).head(50)

    table_rows = "".join([
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['gene_scores'])}</code></td><td>Complete metabolite-level gene scoring table after three-way screening and two-model ranking.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['total_network'])}</code></td><td>Total gene-metabolite association network from ElasticNet top-k union XGBoost top-k.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['high_confidence_network'])}</code></td><td>High-confidence subnetwork of the total association network after RRA and multi-evidence filtering.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['key_gene_summary'])}</code></td><td>Merged key-gene summary across metabolites.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['metabolite_summary'])}</code></td><td>Metabolite-level candidate and network summary.</td></tr>",
        f"<tr><td><code>{html.escape(TABLE_FILE_PREFIXES['cytoscape_network'])}</code></td><td>Cytoscape-ready edge table with updated association fields.</td></tr>",
    ])

    figure_rows = "".join([
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['sample_clustering_dendrogram'])}.pdf|svg|png</code></td><td>Sample clustering dendrogram.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca'])}.pdf|svg|png</code></td><td>Transcriptome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca'])}.pdf|svg|png</code></td><td>Metabolome PCA scatter plot.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['transcriptome_pca_pairs'])}.pdf|svg|png</code></td><td>Transcriptome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['metabolome_pca_pairs'])}.pdf|svg|png</code></td><td>Metabolome PCA pairs plot using the first principal components.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs'])}.pdf|svg|png</code></td><td>Top association pair scatter panels ranked by EdgeWeight.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap'])}.pdf|svg|png</code></td><td>Module-metabolite association heatmap colored by Spearman rho with significance stars.</td></tr>",
        f"<tr><td><code>plots/{html.escape(FIGURE_FILE_PREFIXES['compressed_circos_network'])}.pdf|svg|png</code></td><td>Compact Circos overview using all unique genes and metabolites from T03 only.</td></tr>",
    ])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepOmics Report - {html.escape(cfg.project_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; color: #111827; }}
    h1, h2 {{ color: #1f2937; }}
    .hero {{
      background: #f8fafc;
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .link-card {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 24px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>DeepOmics Report: {html.escape(cfg.project_name)}</h1>
    <p>This page summarizes structured association tables and figure file names.</p>
  </div>

  <h2>Run Summary</h2>
  <ul>
    <li>Samples: {engine.adata.n_obs}</li>
    <li>Genes: {engine.adata.n_vars}</li>
    <li>Metabolites: {len(engine.adata.uns.get("metabolite_names", []))}</li>
    <li>Output directory: <code>{html.escape(str(cfg.output_dir))}</code></li>
  </ul>

  <div class="link-card">
    <strong>Interactive report:</strong>
    Open <a href="DeepOmics_Interactive_Report.html"><code>DeepOmics_Interactive_Report.html</code></a>
    for lightweight browser-native visualization preview and SVG export.
  </div>

  <h2>Main Output Tables</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Generated Figures</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>{figure_rows}</tbody>
  </table>

  <h2>Metabolite-Level Summary</h2>
  {metabolite_summary.to_html(index=False, escape=True)}

  <h2>Key Gene Summary</h2>
  {key_gene_summary.to_html(index=False, escape=True)}
</body>
</html>
"""
    Path(report_path).write_text(html_text, encoding="utf-8")




def generate_report_plots(engine, cfg) -> None:
    set_academic_style()
    plots_dir = safe_mkdir(Path(cfg.output_dir) / "plots")
    pca_group_df = _load_pca_group_table(cfg)

    plot_sample_dendrogram(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["sample_clustering_dendrogram"], cfg)
    plot_transcriptome_pca(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["transcriptome_pca"], cfg, group_df=pca_group_df)
    plot_metabolome_pca(engine.adata, plots_dir / FIGURE_FILE_PREFIXES["metabolome_pca"], cfg, group_df=pca_group_df)
    plot_transcriptome_pca_pairs(
        engine.adata,
        plots_dir / FIGURE_FILE_PREFIXES["transcriptome_pca_pairs"],
        cfg,
        group_df=pca_group_df,
    )
    plot_metabolome_pca_pairs(
        engine.adata,
        plots_dir / FIGURE_FILE_PREFIXES["metabolome_pca_pairs"],
        cfg,
        group_df=pca_group_df,
    )

    if _has_secondary_grouping(pca_group_df):
        plot_transcriptome_pca_subgroups(
            engine.adata,
            plots_dir / FIGURE_FILE_PREFIXES["transcriptome_pca_subgroups"],
            cfg,
            group_df=pca_group_df,
        )
        plot_metabolome_pca_subgroups(
            engine.adata,
            plots_dir / FIGURE_FILE_PREFIXES["metabolome_pca_subgroups"],
            cfg,
            group_df=pca_group_df,
        )
        plot_transcriptome_pca_pairs_subgroups(
            engine.adata,
            plots_dir / FIGURE_FILE_PREFIXES["transcriptome_pca_pairs_subgroups"],
            cfg,
            group_df=pca_group_df,
        )
        plot_metabolome_pca_pairs_subgroups(
            engine.adata,
            plots_dir / FIGURE_FILE_PREFIXES["metabolome_pca_pairs_subgroups"],
            cfg,
            group_df=pca_group_df,
        )

    plot_top_edge_scatter_panels(engine, plots_dir / FIGURE_FILE_PREFIXES["top_gene_metabolite_pairs"], cfg)
    plot_module_metabolite_association_heatmap(
        engine,
        plots_dir / FIGURE_FILE_PREFIXES["module_metabolite_association_heatmap"],
        cfg,
    )
    plot_compressed_circos_network(engine, plots_dir / FIGURE_FILE_PREFIXES["compressed_circos_network"], cfg)

    notes = (
        "Recommended downstream usage:\n"
        f"1. Use {TABLE_FILE_PREFIXES['gene_scores']} for full metabolite-level candidate scoring.\n"
        f"2. Use {TABLE_FILE_PREFIXES['total_network']} for broad association recovery.\n"
        f"3. Use {TABLE_FILE_PREFIXES['high_confidence_network']} for the stricter high-confidence subset of the total network.\n"
        f"4. Use {TABLE_FILE_PREFIXES['cytoscape_network']} for Cytoscape import.\n"
        f"5. Use plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs']}.pdf|svg|png for the transcriptome PCA pairs overview (group1).\n"
        f"6. Use plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs']}.pdf|svg|png for the metabolome PCA pairs overview (group1).\n"
        f"7. Use plots/{FIGURE_FILE_PREFIXES['transcriptome_pca_pairs_subgroups']}.pdf|svg|png for the transcriptome PCA pairs overview (group2).\n"
        f"8. Use plots/{FIGURE_FILE_PREFIXES['metabolome_pca_pairs_subgroups']}.pdf|svg|png for the metabolome PCA pairs overview (group2).\n"
        f"9. Use plots/{FIGURE_FILE_PREFIXES['top_gene_metabolite_pairs']}.pdf|svg|png for the top regression-panel overview.\n"
        f"10. Use plots/{FIGURE_FILE_PREFIXES['module_metabolite_association_heatmap']}.pdf|svg|png for the module-metabolite association heatmap.\n"
        f"11. Use plots/{FIGURE_FILE_PREFIXES['compressed_circos_network']}.pdf|svg|png for the compact T03-only Circos overview.\n"
        "12. Use DeepOmics_Interactive_Report.html for lightweight browser-native visualization preview and export.\n"
    )
    (plots_dir / "visualization_notes.txt").write_text(notes, encoding="utf-8")

    if cfg.generate_reports:
        if "md" in cfg.report_formats:
            generate_markdown_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.md")
        if "html" in cfg.report_formats:
            generate_html_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Report.html")
            from .interactive import generate_interactive_visual_report

            generate_interactive_visual_report(engine, cfg, Path(cfg.output_dir) / "DeepOmics_Interactive_Report.html")

