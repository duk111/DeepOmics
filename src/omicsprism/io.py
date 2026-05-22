from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .utils import get_logger

logger = get_logger()


def _read_feature_table(path: str | Path, label: str) -> pd.DataFrame:
    """Read and validate an omics matrix."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} file not found: {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    if df.empty:
        raise ValueError(f"{label} table is empty: {file_path}")

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{label} contains duplicated feature IDs: {duplicates}")
    if df.columns.has_duplicates:
        duplicates = df.columns[df.columns.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{label} contains duplicated sample IDs: {duplicates}")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().all(axis=None):
        raise ValueError(f"{label} table does not contain numeric values: {file_path}")

    missing_count = int(numeric_df.isna().sum().sum())
    if missing_count > 0:
        logger.warning(
            "%s table contains %d missing values; they will be imputed during preprocessing.",
            label,
            missing_count,
        )

    return numeric_df


def _apply_log2p1(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Apply log2(x+1) to a matrix whose finite values are greater than -1."""
    values = df.to_numpy(dtype=np.float32, copy=False)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"{label} table does not contain finite numeric values.")

    min_value = float(np.nanmin(finite_values))
    if min_value <= -1:
        raise ValueError(
            f"{label} contains values less than or equal to -1; log2(x+1) cannot be applied safely."
        )

    transformed = np.log2(values + 1.0).astype(np.float32)
    logger.info("Applied log2(x+1) transformation to %s.", label)
    return pd.DataFrame(transformed, index=df.index.copy(), columns=df.columns.copy())


def _filter_high_missing_features(
    df: pd.DataFrame,
    *,
    label: str,
    missing_feature_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Drop features missing in more than the configured fraction of samples."""
    threshold = float(missing_feature_threshold)
    if not (0.0 <= threshold < 1.0):
        raise ValueError("missing_feature_threshold must be within [0, 1).")

    missing_fraction = df.isna().mean(axis=0)
    keep_mask = missing_fraction <= threshold
    dropped = int((~keep_mask).sum())
    if dropped > 0:
        logger.warning(
            "Removed %d %s features with missing values in more than %.0f%% of samples.",
            dropped,
            label,
            threshold * 100.0,
        )

    filtered = df.loc[:, keep_mask].copy()
    if filtered.shape[1] == 0:
        raise ValueError(
            f"No {label} features remain after filtering features with missing values "
            f"in more than {threshold:.0%} of samples."
        )

    return filtered, {
        "threshold": threshold,
        "n_before": int(df.shape[1]),
        "n_after": int(filtered.shape[1]),
        "n_removed": dropped,
    }


def _impute_missing_with_knn(
    df: pd.DataFrame,
    *,
    label: str,
    n_neighbors: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fill missing values with KNN imputation across samples."""
    neighbors = max(1, int(n_neighbors))
    missing_before = int(df.isna().sum().sum())
    if missing_before == 0:
        return df, {
            "method": "none",
            "n_neighbors": neighbors,
            "missing_before": 0,
            "missing_after": 0,
        }

    resolved_neighbors = min(neighbors, max(1, df.shape[0] - 1))
    imputer = KNNImputer(n_neighbors=resolved_neighbors)
    imputed = imputer.fit_transform(df.to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
    result = pd.DataFrame(imputed, index=df.index.copy(), columns=df.columns.copy())

    missing_after = int(result.isna().sum().sum())
    if missing_after > 0:
        raise ValueError(f"KNN imputation left {missing_after} missing values in {label}.")

    logger.info(
        "Imputed %d missing values in %s using KNNImputer(n_neighbors=%d).",
        missing_before,
        label,
        resolved_neighbors,
    )
    return result, {
        "method": "knn",
        "n_neighbors": resolved_neighbors,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def _read_group_sample_order(group_table_path: str | Path) -> list[str]:
    """Read sample_id values from a group table in file order."""
    group_path = Path(group_table_path)
    if not group_path.exists():
        raise FileNotFoundError(f"Group table not found: {group_path}")

    group_df = pd.read_csv(group_path, sep=None, engine="python", encoding="utf-8-sig")
    normalized_columns = {
        str(column).replace("\ufeff", "").strip().lower(): column
        for column in group_df.columns
    }
    required_columns = {"sample_id", "group1", "group2"}
    missing_columns = sorted(required_columns.difference(normalized_columns))
    if missing_columns:
        raise ValueError(
            "Group table must contain columns: sample_id, group1, group2. "
            f"Missing: {missing_columns}"
        )

    sample_ids = group_df[normalized_columns["sample_id"]].astype(str).str.strip()
    sample_ids = sample_ids.loc[sample_ids.ne("")]
    duplicated_mask = sample_ids.duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_ids = sample_ids.loc[duplicated_mask].unique().tolist()
        raise ValueError(
            "Group table contains duplicated sample_id values: "
            f"{duplicated_ids[:5]}"
        )
    return sample_ids.tolist()


def _order_common_samples(
    common_samples: pd.Index,
    group_table_path: str | Path,
) -> tuple[pd.Index, dict[str, object]]:
    """Order shared samples by group table sample_id order when available."""
    common_sample_ids = [str(sample_id) for sample_id in common_samples.astype(str).tolist()]

    group_order = _read_group_sample_order(group_table_path)
    common_set = set(common_sample_ids)
    ordered = []
    seen = set()
    for sample_id in group_order:
        if sample_id in common_set and sample_id not in seen:
            ordered.append(sample_id)
            seen.add(sample_id)

    appended = [sample_id for sample_id in common_sample_ids if sample_id not in seen]
    if appended:
        logger.warning(
            "Appended %d shared samples not present in group table after group-table ordered samples.",
            len(appended),
        )
    if not ordered:
        logger.warning(
            "No shared samples were found in the group table; falling back to shared sample input order."
        )
        ordered = common_sample_ids
        appended = []
    else:
        ordered.extend(appended)

    ignored_group_rows = len([sample_id for sample_id in group_order if sample_id not in common_set])
    if ignored_group_rows > 0:
        logger.info(
            "Ignored %d group table sample_id values not present in both omics matrices.",
            ignored_group_rows,
        )

    return pd.Index(ordered, name="SampleID"), {
        "method": "group_table_sample_id_order",
        "group_table_path": str(group_table_path),
        "n_group_ordered_samples": int(len(seen)),
        "n_appended_samples": int(len(appended)),
        "n_ignored_group_table_samples": int(ignored_group_rows),
    }


def load_as_anndata(
    gene_path: str | Path,
    metab_path: str | Path,
    group_table_path: str | Path,
) -> ad.AnnData:
    """Load transcriptome and metabolome matrices into a single AnnData object."""
    logger.info("Loading transcriptome data from %s", gene_path)
    df_gene = _read_feature_table(gene_path, label="Transcriptome")

    logger.info("Loading metabolomics data from %s", metab_path)
    df_metab = _read_feature_table(metab_path, label="Metabolomics")

    df_gene_t = df_gene.T
    df_metab_t = df_metab.T

    common_samples = df_gene_t.index.intersection(df_metab_t.index, sort=False)
    common_samples = pd.Index(common_samples.astype(str), name="SampleID")
    if len(common_samples) == 0:
        raise ValueError("No shared sample IDs were found between transcriptome and metabolomics tables.")
    common_samples, sample_order_info = _order_common_samples(common_samples, group_table_path)

    logger.info("Sample alignment completed. Shared samples: %d", len(common_samples))

    gene_aligned = df_gene_t.loc[common_samples].astype(np.float32, copy=False)
    metab_aligned = df_metab_t.loc[common_samples].astype(np.float32, copy=False)

    adata = ad.AnnData(
        X=gene_aligned.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame(index=pd.Index(common_samples.astype(str), name="SampleID")),
        var=pd.DataFrame(index=pd.Index(gene_aligned.columns.astype(str), name="Gene")),
    )
    adata.obsm["metabolomics"] = pd.DataFrame(
        metab_aligned.to_numpy(dtype=np.float32, copy=False),
        index=adata.obs_names.copy(),
        columns=metab_aligned.columns.astype(str),
    )
    adata.uns["metabolite_names"] = metab_aligned.columns.astype(str).tolist()
    adata.uns["input_summary"] = {
        "n_samples": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_metabolites": int(len(adata.uns["metabolite_names"])),
        "sample_order": sample_order_info,
    }
    return adata


def preprocess_adata(
    adata: ad.AnnData,
    scale: bool = True,
    missing_feature_threshold: float = 0.5,
    knn_neighbors: int = 5,
    trans_log2: bool = False,
) -> ad.AnnData:
    """Apply filtering, optional transcriptome log transformation, KNN imputation, variance filtering, and scaling."""
    if "metabolomics" not in adata.obsm:
        raise KeyError("adata.obsm['metabolomics'] is required.")

    gene_df = pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )
    gene_df, gene_missing_filter_info = _filter_high_missing_features(
        gene_df,
        label="Transcriptome",
        missing_feature_threshold=missing_feature_threshold,
    )
    if trans_log2:
        gene_df = _apply_log2p1(gene_df, label="Transcriptome")
        gene_log_info = {
            "label": "Transcriptome",
            "applied": True,
            "method": "log2(x+1)",
            "stage": "after_high_missing_filter_before_knn_imputation",
        }
    else:
        logger.info("Skipped log2(x+1) transformation for Transcriptome.")
        gene_log_info = {
            "label": "Transcriptome",
            "applied": False,
            "method": "none",
            "stage": "after_high_missing_filter_before_knn_imputation",
        }
    gene_df, gene_impute_info = _impute_missing_with_knn(
        gene_df,
        label="Transcriptome",
        n_neighbors=knn_neighbors,
    )

    gene_var = np.nanvar(gene_df.to_numpy(dtype=np.float32, copy=False), axis=0)
    keep_genes = gene_var > 0
    if not np.all(keep_genes):
        dropped = int((~keep_genes).sum())
        logger.warning("Removed %d constant genes before modeling.", dropped)
        gene_df = gene_df.loc[:, keep_genes]

    if gene_df.shape[1] == 0:
        raise ValueError("No genes remain after preprocessing.")

    adata = adata[:, pd.Index(gene_df.columns.astype(str))].copy()
    adata.X = gene_df.to_numpy(dtype=np.float32, copy=False)
    adata.var_names = pd.Index(gene_df.columns.astype(str), name="Gene")

    metab_df = adata.obsm["metabolomics"]
    if not isinstance(metab_df, pd.DataFrame):
        metabolite_names = adata.uns.get("metabolite_names", [])
        metab_df = pd.DataFrame(metab_df, index=adata.obs_names, columns=metabolite_names)
    metab_df = metab_df.copy()
    metab_df.index = adata.obs_names.astype(str)
    metab_df.columns = metab_df.columns.astype(str)
    metab_df = metab_df.astype(np.float32, copy=False)
    metab_df, metab_missing_filter_info = _filter_high_missing_features(
        metab_df,
        label="Metabolomics",
        missing_feature_threshold=missing_feature_threshold,
    )
    metab_df = _apply_log2p1(metab_df, label="Metabolomics")
    metab_df, metab_impute_info = _impute_missing_with_knn(
        metab_df,
        label="Metabolomics",
        n_neighbors=knn_neighbors,
    )
    metab_log_info = {
        "label": "Metabolomics",
        "applied": True,
        "method": "log2(x+1)",
        "stage": "after_high_missing_filter_before_knn_imputation",
    }

    metab_var = np.nanvar(metab_df.to_numpy(dtype=np.float32, copy=False), axis=0)
    keep_metabs = metab_var > 0
    if not np.all(keep_metabs):
        dropped = int((~keep_metabs).sum())
        logger.warning("Removed %d constant metabolites before modeling.", dropped)
        metab_df = metab_df.loc[:, keep_metabs]

    if metab_df.shape[1] == 0:
        raise ValueError("No metabolites remain after preprocessing.")

    adata.obsm["metabolomics"] = metab_df.astype(np.float32, copy=False)
    adata.uns["metabolite_names"] = metab_df.columns.astype(str).tolist()

    if scale:
        logger.info("Applying z-score scaling to transcriptome and metabolomics matrices.")
        adata.layers["raw"] = np.asarray(adata.X, dtype=np.float32).copy()

        scaler_x = StandardScaler(copy=True)
        adata.X = scaler_x.fit_transform(np.asarray(adata.X, dtype=np.float32)).astype(np.float32)

        scaler_y = StandardScaler(copy=True)
        metab_scaled = scaler_y.fit_transform(
            adata.obsm["metabolomics"].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32)
        adata.obsm["metabolomics_scaled"] = pd.DataFrame(
            metab_scaled,
            index=adata.obs_names.copy(),
            columns=adata.uns["metabolite_names"],
        )
    else:
        adata.obsm.pop("metabolomics_scaled", None)

    adata.uns["preprocess_summary"] = {
        "n_samples": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_metabolites": int(len(adata.uns["metabolite_names"])),
        "scaled": bool(scale),
        "log_transform": {
            "transcriptome": bool(trans_log2),
            "metabolomics": True,
        },
        "missing_feature_threshold": float(missing_feature_threshold),
        "knn_neighbors": int(knn_neighbors),
        "transcriptome_missing_filter": gene_missing_filter_info,
        "transcriptome_imputation": gene_impute_info,
        "transcriptome_log2p1": gene_log_info,
        "metabolomics_missing_filter": metab_missing_filter_info,
        "metabolomics_imputation": metab_impute_info,
        "metabolomics_log2p1": metab_log_info,
    }
    return adata


def read_h5ad(filename: str | Path) -> ad.AnnData:
    """Load a previously saved AnnData object."""
    return ad.read_h5ad(str(filename))
