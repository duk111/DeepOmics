from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .utils import get_logger

logger = get_logger()


WGCNA_COLOR_SEQUENCE = [
    "turquoise",
    "blue",
    "brown",
    "yellow",
    "green",
    "red",
    "black",
    "pink",
    "magenta",
    "purple",
    "greenyellow",
    "tan",
    "salmon",
    "cyan",
    "midnightblue",
    "lightcyan",
    "royalblue",
    "darkred",
    "darkgreen",
    "darkturquoise",
    "darkgrey",
    "orange",
    "white",
    "skyblue",
    "saddlebrown",
    "steelblue",
    "paleturquoise",
    "violet",
    "darkorange",
    "darkmagenta",
]

WGCNA_COLOR_HEX = {
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


try:  # pragma: no cover - optional dependency
    import igraph as ig
    import leidenalg

    _HAS_LEIDEN = True
except Exception:  # pragma: no cover - optional dependency
    ig = None
    leidenalg = None
    _HAS_LEIDEN = False


@dataclass
class ModuleAnalysisArtifacts:
    gene_module_assignment_df: pd.DataFrame
    module_eigengenes_df: pd.DataFrame
    module_metabolite_assoc_df: pd.DataFrame
    module_summary_df: pd.DataFrame
    metadata: Dict[str, object]


def _empty_artifacts() -> ModuleAnalysisArtifacts:
    return ModuleAnalysisArtifacts(
        gene_module_assignment_df=pd.DataFrame(
            columns=[
                "Gene",
                "Module",
                "ModuleColorHex",
                "ModuleSize",
                "kME",
                "IntramodularDegree",
                "IsGrey",
                "BestEdgeWeight",
                "MeanEdgeWeight",
                "AssociatedMetaboliteCount",
                "HighConfidenceMetaboliteCount",
            ]
        ),
        module_eigengenes_df=pd.DataFrame(),
        module_metabolite_assoc_df=pd.DataFrame(
            columns=["Module", "Metabolite", "SpearmanRho", "PValue", "FDR"]
        ),
        module_summary_df=pd.DataFrame(
            columns=[
                "Module",
                "ModuleColorHex",
                "ModuleSize",
                "MeanKME",
                "MeanIntramodularDegree",
                "TopHubGene",
                "TopHubKME",
                "MetaboliteAssociationCount",
                "TopMetabolite",
                "TopMetaboliteRho",
            ]
        ),
        metadata={
            "module_method_used": "none",
            "n_module_genes": 0,
            "n_non_grey_modules": 0,
            "n_grey_genes": 0,
        },
    )


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1D array.")
    if p.size == 0:
        return p.copy()

    result = np.full_like(p, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return result

    valid_p = np.clip(p[valid_mask], 0.0, 1.0)
    order = np.argsort(valid_p, kind="mergesort")
    ranked = valid_p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    back = np.empty_like(adjusted)
    back[order] = adjusted
    result[valid_mask] = back
    return result


def extract_module_gene_set(high_confidence_network_df: pd.DataFrame) -> list[str]:
    if high_confidence_network_df.empty or "Gene" not in high_confidence_network_df.columns:
        return []
    return sorted(pd.Index(high_confidence_network_df["Gene"].astype(str).unique()).tolist())


def compute_spearman_corr(expr_df: pd.DataFrame) -> pd.DataFrame:
    if expr_df.empty:
        return pd.DataFrame()
    corr_df = expr_df.corr(method="spearman").astype(float)
    corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)
    return corr_df


def build_signed_positive_adjacency(corr_df: pd.DataFrame) -> pd.DataFrame:
    if corr_df.empty:
        return corr_df.copy()
    adjacency = corr_df.clip(lower=0.0).copy()
    np.fill_diagonal(adjacency.values, 0.0)
    return adjacency


def sparsify_topk(adj_df: pd.DataFrame, k: int, min_weight: float) -> pd.DataFrame:
    if adj_df.empty:
        return adj_df.copy()

    k = max(1, int(k))
    min_weight = float(min_weight)

    values = adj_df.to_numpy(dtype=float, copy=True)
    n = values.shape[0]
    sparse = np.zeros_like(values, dtype=float)

    for i in range(n):
        row = values[i].copy()
        row[i] = 0.0
        keep = np.flatnonzero(row > min_weight)
        if keep.size == 0:
            continue
        keep = keep[np.argsort(row[keep])[::-1][:k]]
        sparse[i, keep] = row[keep]

    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return pd.DataFrame(sparse, index=adj_df.index.copy(), columns=adj_df.columns.copy())


def _detect_modules_leiden(
    adj_df: pd.DataFrame,
    resolution: float,
    random_state: int,
) -> pd.Series:
    if adj_df.empty:
        return pd.Series(dtype="string", name="Module")

    if not _HAS_LEIDEN:
        raise ImportError(
            "Leiden dependencies are not available. Install with the network extra: "
            "python-igraph and leidenalg."
        )

    names = adj_df.index.astype(str).tolist()
    values = adj_df.to_numpy(dtype=float, copy=False)
    upper_i, upper_j = np.triu_indices_from(values, k=1)
    upper_w = values[upper_i, upper_j]
    mask = upper_w > 0

    edges = list(zip(upper_i[mask].tolist(), upper_j[mask].tolist()))
    weights = upper_w[mask].astype(float).tolist()

    graph = ig.Graph(n=len(names), edges=edges, directed=False)
    graph.vs["name"] = names
    if weights:
        graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights if weights else None,
        resolution_parameter=float(resolution),
        seed=int(random_state),
    )
    labels = pd.Series(
        np.asarray(partition.membership, dtype=int),
        index=pd.Index(names, name="Gene"),
        name="Module",
    )
    return labels


def _detect_modules_hierarchical(
    corr_df: pd.DataFrame,
    resolution: float,
) -> pd.Series:
    if corr_df.empty:
        return pd.Series(dtype="string", name="Module")
    if corr_df.shape[0] == 1:
        return pd.Series([0], index=corr_df.index.copy(), name="Module")

    distance = 1.0 - corr_df.clip(lower=0.0, upper=1.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    threshold = float(np.clip(1.0 - 0.25 * float(resolution), 0.35, 0.80))
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance") - 1
    return pd.Series(labels.astype(int), index=corr_df.index.copy(), name="Module")


def detect_modules(
    corr_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    *,
    method: str,
    resolution: float,
    random_state: int,
) -> tuple[pd.Series, str]:
    method = str(method).lower().strip()
    if method == "leiden":
        try:
            return _detect_modules_leiden(adj_df, resolution=resolution, random_state=random_state), "leiden"
        except Exception as exc:  # pragma: no cover - depends on optional dependency availability
            logger.warning(
                "Leiden community detection failed or is unavailable; falling back to hierarchical clustering. Reason: %s",
                exc,
            )

    return _detect_modules_hierarchical(corr_df, resolution=resolution), "hierarchical"


def collapse_small_modules(
    labels: pd.Series,
    min_size: int,
    grey_label: str = "grey",
) -> pd.Series:
    if labels.empty:
        return labels.astype("string")

    labels = labels.copy()
    min_size = max(1, int(min_size))
    sizes = labels.value_counts(dropna=False)
    small = set(sizes[sizes < min_size].index.tolist())
    collapsed = labels.astype(object).where(~labels.isin(small), other=grey_label)

    non_grey = [value for value in collapsed.unique().tolist() if value != grey_label]
    ranked = sorted(
        non_grey,
        key=lambda value: (-int((collapsed == value).sum()), str(value)),
    )

    rename_map: dict[object, str] = {}
    for idx, value in enumerate(ranked):
        if idx < len(WGCNA_COLOR_SEQUENCE):
            rename_map[value] = WGCNA_COLOR_SEQUENCE[idx]
        else:
            rename_map[value] = f"module{idx + 1:02d}"
    rename_map[grey_label] = grey_label

    renamed = collapsed.map(lambda x: rename_map.get(x, grey_label)).astype("string")
    renamed.name = "Module"
    return renamed


def compute_module_eigengenes(
    expr_df: pd.DataFrame,
    module_labels: pd.Series,
) -> pd.DataFrame:
    if expr_df.empty or module_labels.empty:
        return pd.DataFrame(index=expr_df.index.copy())

    module_labels = module_labels.reindex(expr_df.columns)
    eigengenes: dict[str, pd.Series] = {}

    for module_name in module_labels.dropna().astype(str).unique().tolist():
        if module_name == "grey":
            continue
        genes = module_labels.index[module_labels == module_name].astype(str).tolist()
        if len(genes) == 0:
            continue

        module_expr = expr_df.loc[:, genes].to_numpy(dtype=float, copy=False)

        if module_expr.shape[1] == 1:
            eig = module_expr[:, 0].astype(float)
            eig = eig - float(np.mean(eig))
            scale = float(np.std(eig, ddof=1))
            if np.isfinite(scale) and scale > 0:
                eig = eig / scale
        else:
            pca = PCA(n_components=1, random_state=0)
            eig = pca.fit_transform(module_expr).reshape(-1).astype(float)

        eig_series = pd.Series(eig, index=expr_df.index.copy(), name=module_name)

        gene_corrs = expr_df.loc[:, genes].corrwith(eig_series, method="spearman").fillna(0.0)
        if float(gene_corrs.mean()) < 0:
            eig_series = -eig_series

        eigengenes[module_name] = eig_series

    if not eigengenes:
        return pd.DataFrame(index=expr_df.index.copy())

    eigengenes_df = pd.DataFrame(eigengenes, index=expr_df.index.copy())
    eigengenes_df.index.name = expr_df.index.name
    return eigengenes_df


def compute_kme(
    expr_df: pd.DataFrame,
    eigengenes_df: pd.DataFrame,
    module_labels: pd.Series,
) -> pd.Series:
    result = pd.Series(np.nan, index=expr_df.columns.copy(), dtype=float, name="kME")
    if expr_df.empty or eigengenes_df.empty or module_labels.empty:
        return result

    module_labels = module_labels.reindex(expr_df.columns)
    for module_name in eigengenes_df.columns.astype(str).tolist():
        genes = module_labels.index[module_labels == module_name].astype(str).tolist()
        if len(genes) == 0:
            continue
        eig = eigengenes_df[module_name]
        corrs = expr_df.loc[:, genes].corrwith(eig, method="spearman").astype(float)
        result.loc[genes] = corrs.reindex(genes).to_numpy(dtype=float)
    return result


def compute_intramodular_degree(
    adj_df: pd.DataFrame,
    module_labels: pd.Series,
) -> pd.Series:
    result = pd.Series(0.0, index=adj_df.index.copy(), dtype=float, name="IntramodularDegree")
    if adj_df.empty or module_labels.empty:
        return result

    module_labels = module_labels.reindex(adj_df.index)
    for module_name in module_labels.dropna().astype(str).unique().tolist():
        if module_name == "grey":
            continue
        genes = module_labels.index[module_labels == module_name].astype(str).tolist()
        if len(genes) == 0:
            continue
        sub = adj_df.loc[genes, genes]
        result.loc[genes] = sub.sum(axis=1).astype(float).to_numpy(dtype=float)
    return result


def compute_module_metabolite_associations(
    eigengenes_df: pd.DataFrame,
    metabolomics_df: pd.DataFrame,
) -> pd.DataFrame:
    if eigengenes_df.empty or metabolomics_df.empty:
        return pd.DataFrame(columns=["Module", "Metabolite", "SpearmanRho", "PValue", "FDR"])

    rows: list[dict[str, object]] = []
    for module_name in eigengenes_df.columns.astype(str).tolist():
        eig = eigengenes_df[module_name]
        for metabolite_name in metabolomics_df.columns.astype(str).tolist():
            rho, p_value = spearmanr(
                eig.to_numpy(dtype=float, copy=False),
                metabolomics_df[metabolite_name].to_numpy(dtype=float, copy=False),
            )
            if not np.isfinite(rho):
                rho = 0.0
            if not np.isfinite(p_value):
                p_value = 1.0
            rows.append(
                {
                    "Module": module_name,
                    "Metabolite": metabolite_name,
                    "SpearmanRho": float(rho),
                    "PValue": float(p_value),
                }
            )

    result = pd.DataFrame(rows)
    result["FDR"] = _bh_fdr(result["PValue"].to_numpy(dtype=float))
    result = result.sort_values(
        ["Module", "FDR", "PValue", "SpearmanRho", "Metabolite"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


def build_gene_module_assignment(
    module_labels: pd.Series,
    kme: pd.Series,
    intramodular_degree: pd.Series,
    key_gene_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if module_labels.empty:
        return _empty_artifacts().gene_module_assignment_df

    assignment = pd.DataFrame(
        {
            "Gene": module_labels.index.astype(str),
            "Module": module_labels.astype(str).to_numpy(),
            "kME": kme.reindex(module_labels.index).to_numpy(dtype=float),
            "IntramodularDegree": intramodular_degree.reindex(module_labels.index).to_numpy(dtype=float),
        }
    )
    assignment["ModuleColorHex"] = assignment["Module"].map(
        lambda value: WGCNA_COLOR_HEX.get(str(value).lower(), "#9ca3af")
    )
    module_sizes = assignment.groupby("Module", sort=False)["Gene"].transform("size").astype(int)
    assignment["ModuleSize"] = module_sizes
    assignment["IsGrey"] = (assignment["Module"] == "grey").astype(int)

    if isinstance(key_gene_summary_df, pd.DataFrame) and not key_gene_summary_df.empty and "Gene" in key_gene_summary_df.columns:
        keep_cols = [
            "Gene",
            "AssociatedMetaboliteCount",
            "HighConfidenceMetaboliteCount",
            "MeanEdgeWeight",
            "BestEdgeWeight",
        ]
        key_gene_summary = key_gene_summary_df.loc[:, [col for col in keep_cols if col in key_gene_summary_df.columns]].copy()
        assignment = assignment.merge(key_gene_summary, on="Gene", how="left")

    for column in [
        "AssociatedMetaboliteCount",
        "HighConfidenceMetaboliteCount",
    ]:
        if column not in assignment.columns:
            assignment[column] = 0
        assignment[column] = assignment[column].fillna(0).astype(int)

    for column in ["MeanEdgeWeight", "BestEdgeWeight"]:
        if column not in assignment.columns:
            assignment[column] = np.nan
        assignment[column] = assignment[column].astype(float)

    assignment = assignment.loc[
        :,
        [
            "Gene",
            "Module",
            "ModuleColorHex",
            "ModuleSize",
            "kME",
            "IntramodularDegree",
            "IsGrey",
            "BestEdgeWeight",
            "MeanEdgeWeight",
            "AssociatedMetaboliteCount",
            "HighConfidenceMetaboliteCount",
        ],
    ].sort_values(
        ["Module", "IsGrey", "kME", "IntramodularDegree", "BestEdgeWeight", "Gene"],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return assignment


def build_module_summary(
    gene_module_assignment_df: pd.DataFrame,
    module_metabolite_assoc_df: pd.DataFrame,
) -> pd.DataFrame:
    if gene_module_assignment_df.empty:
        return _empty_artifacts().module_summary_df

    non_grey = gene_module_assignment_df.loc[gene_module_assignment_df["Module"].astype(str) != "grey"].copy()
    if non_grey.empty:
        return _empty_artifacts().module_summary_df

    grouped = non_grey.groupby("Module", sort=False)
    summary = grouped.agg(
        ModuleSize=("Gene", "size"),
        MeanKME=("kME", "mean"),
        MeanIntramodularDegree=("IntramodularDegree", "mean"),
    ).reset_index()
    summary["ModuleColorHex"] = summary["Module"].map(
        lambda value: WGCNA_COLOR_HEX.get(str(value).lower(), "#9ca3af")
    )

    top_hub = non_grey.sort_values(
        ["Module", "kME", "IntramodularDegree", "BestEdgeWeight", "Gene"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).drop_duplicates(subset=["Module"], keep="first")
    summary = summary.merge(
        top_hub.loc[:, ["Module", "Gene", "kME"]].rename(columns={"Gene": "TopHubGene", "kME": "TopHubKME"}),
        on="Module",
        how="left",
    )

    if not module_metabolite_assoc_df.empty:
        top_metab = module_metabolite_assoc_df.copy()
        top_metab["AbsRho"] = top_metab["SpearmanRho"].abs()
        assoc_counts = (
            top_metab.assign(_Significant=(top_metab["FDR"].fillna(1.0) <= 0.05).astype(int))
            .groupby("Module", sort=False)["_Significant"]
            .sum()
            .rename("MetaboliteAssociationCount")
            .reset_index()
        )
        top_metab = top_metab.sort_values(
            ["Module", "AbsRho", "FDR", "PValue", "Metabolite"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
        ).drop_duplicates(subset=["Module"], keep="first")
        summary = summary.merge(assoc_counts, on="Module", how="left")
        summary = summary.merge(
            top_metab.loc[:, ["Module", "Metabolite", "SpearmanRho"]].rename(
                columns={"Metabolite": "TopMetabolite", "SpearmanRho": "TopMetaboliteRho"}
            ),
            on="Module",
            how="left",
        )
    else:
        summary["MetaboliteAssociationCount"] = 0
        summary["TopMetabolite"] = ""
        summary["TopMetaboliteRho"] = np.nan

    summary["MetaboliteAssociationCount"] = summary["MetaboliteAssociationCount"].fillna(0).astype(int)
    summary = summary.loc[
        :,
        [
            "Module",
            "ModuleColorHex",
            "ModuleSize",
            "MeanKME",
            "MeanIntramodularDegree",
            "TopHubGene",
            "TopHubKME",
            "MetaboliteAssociationCount",
            "TopMetabolite",
            "TopMetaboliteRho",
        ],
    ].sort_values(
        ["ModuleSize", "MeanKME", "MeanIntramodularDegree", "Module"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return summary


def run_gene_module_analysis(
    expr_df: pd.DataFrame,
    metabolomics_df: pd.DataFrame,
    high_confidence_network_df: pd.DataFrame,
    key_gene_summary_df: pd.DataFrame,
    cfg,
) -> ModuleAnalysisArtifacts:
    gene_list = extract_module_gene_set(high_confidence_network_df)
    if len(gene_list) < 2:
        logger.info("Module analysis skipped because fewer than 2 genes were present in the high-confidence network.")
        return _empty_artifacts()

    gene_list = [gene for gene in gene_list if gene in expr_df.columns]
    if len(gene_list) < 2:
        logger.info("Module analysis skipped because high-confidence network genes were not found in the expression matrix.")
        return _empty_artifacts()

    selected_expr = expr_df.loc[:, gene_list].copy()
    corr_df = compute_spearman_corr(selected_expr)
    adj_df = build_signed_positive_adjacency(corr_df)
    sparse_adj_df = sparsify_topk(
        adj_df,
        k=int(getattr(cfg, "module_graph_k", 10)),
        min_weight=float(getattr(cfg, "module_min_edge_weight", 0.15)),
    )

    labels, method_used = detect_modules(
        corr_df=corr_df,
        adj_df=sparse_adj_df,
        method=str(getattr(cfg, "module_method", "leiden")),
        resolution=float(getattr(cfg, "module_resolution", 1.0)),
        random_state=int(getattr(cfg, "random_state", 42)),
    )
    labels = collapse_small_modules(labels, min_size=int(getattr(cfg, "module_min_size", 5)))
    labels = labels.reindex(selected_expr.columns)

    eigengenes_df = compute_module_eigengenes(selected_expr, labels)
    kme = compute_kme(selected_expr, eigengenes_df, labels)
    intramodular_degree = compute_intramodular_degree(sparse_adj_df, labels)
    gene_module_assignment_df = build_gene_module_assignment(
        labels,
        kme,
        intramodular_degree,
        key_gene_summary_df=key_gene_summary_df,
    )
    module_metabolite_assoc_df = compute_module_metabolite_associations(eigengenes_df, metabolomics_df)
    module_summary_df = build_module_summary(gene_module_assignment_df, module_metabolite_assoc_df)

    metadata = {
        "module_method_used": method_used,
        "n_module_genes": int(len(gene_list)),
        "n_non_grey_modules": int(module_summary_df["Module"].nunique()) if not module_summary_df.empty else 0,
        "n_grey_genes": int((gene_module_assignment_df["Module"].astype(str) == "grey").sum()) if not gene_module_assignment_df.empty else 0,
        "module_graph_k": int(getattr(cfg, "module_graph_k", 10)),
        "module_min_edge_weight": float(getattr(cfg, "module_min_edge_weight", 0.15)),
        "module_resolution": float(getattr(cfg, "module_resolution", 1.0)),
    }

    return ModuleAnalysisArtifacts(
        gene_module_assignment_df=gene_module_assignment_df,
        module_eigengenes_df=eigengenes_df,
        module_metabolite_assoc_df=module_metabolite_assoc_df,
        module_summary_df=module_summary_df,
        metadata=metadata,
    )
