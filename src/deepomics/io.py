from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import get_logger

logger = get_logger()


def _read_feature_table(path: str | Path, label: str) -> pd.DataFrame:
    """Read and validate an omics matrix.

    The input format is expected to be features x samples.

    Args:
        path: CSV file path.
        label: Human-readable label used in log messages.

    Returns:
        Cleaned numeric table.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, duplicated, or non-numeric.
    """
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


def _impute_missing_with_column_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using column means.

    Args:
        df: Numeric data frame.

    Returns:
        Imputed data frame.
    """
    if not df.isna().values.any():
        return df
    return df.fillna(df.mean(axis=0))


def load_as_anndata(gene_path: str, metab_path: str) -> ad.AnnData:
    """Load transcriptome and metabolome matrices into a single AnnData object.

    Args:
        gene_path: Path to a gene expression CSV file in features x samples format.
        metab_path: Path to a metabolite abundance CSV file in features x samples format.

    Returns:
        AnnData object where ``X`` stores transcriptome data and ``obsm["metabolomics"]``
        stores metabolite data aligned by shared samples.
    """
    logger.info("Loading transcriptome data from %s", gene_path)
    df_gene = _read_feature_table(gene_path, label="Transcriptome")

    logger.info("Loading metabolomics data from %s", metab_path)
    df_metab = _read_feature_table(metab_path, label="Metabolomics")

    df_gene_t = _impute_missing_with_column_mean(df_gene.T)
    df_metab_t = _impute_missing_with_column_mean(df_metab.T)

    common_samples = [sample for sample in df_gene_t.index if sample in set(df_metab_t.index)]
    if not common_samples:
        raise ValueError("No shared sample IDs were found between transcriptome and metabolomics tables.")

    logger.info("Sample alignment completed. Shared samples: %d", len(common_samples))

    gene_aligned = df_gene_t.loc[common_samples].astype(np.float32)
    metab_aligned = df_metab_t.loc[common_samples].astype(np.float32)

    adata = ad.AnnData(
        X=gene_aligned.to_numpy(dtype=np.float32, copy=False),
        obs=pd.DataFrame(index=pd.Index(common_samples, name="SampleID")),
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
    }
    return adata


def preprocess_adata(adata: ad.AnnData, scale: bool = True) -> ad.AnnData:
    """Apply basic preprocessing to the AnnData object.

    Steps include missing-value imputation, constant-feature removal, and z-score scaling.

    Args:
        adata: Input AnnData object.
        scale: Whether to z-score transcriptome and metabolomics matrices.

    Returns:
        Processed AnnData object.
    """
    if "metabolomics" not in adata.obsm:
        raise KeyError("adata.obsm['metabolomics'] is required.")

    X = np.asarray(adata.X, dtype=np.float32)
    gene_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    gene_df = _impute_missing_with_column_mean(gene_df)
    X = gene_df.to_numpy(dtype=np.float32, copy=False)

    gene_var = np.nanvar(X, axis=0)
    keep_genes = gene_var > 0
    if not np.all(keep_genes):
        dropped = int((~keep_genes).sum())
        logger.warning("Removed %d constant genes before modeling.", dropped)
        adata = adata[:, keep_genes].copy()
        X = np.asarray(adata.X, dtype=np.float32)

    metab_df = adata.obsm["metabolomics"]
    if not isinstance(metab_df, pd.DataFrame):
        metab_df = pd.DataFrame(metab_df, index=adata.obs_names)
    metab_df = _impute_missing_with_column_mean(metab_df.astype(np.float32))

    metab_var = np.nanvar(metab_df.to_numpy(dtype=np.float32, copy=False), axis=0)
    keep_metabs = metab_var > 0
    if not np.all(keep_metabs):
        dropped = int((~keep_metabs).sum())
        logger.warning("Removed %d constant metabolites before modeling.", dropped)
        metab_df = metab_df.loc[:, keep_metabs]
        adata.uns["metabolite_names"] = metab_df.columns.astype(str).tolist()

    adata.obsm["metabolomics"] = metab_df.astype(np.float32)

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

    adata.uns["preprocess_summary"] = {
        "n_samples": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_metabolites": int(len(adata.uns["metabolite_names"])),
        "scaled": bool(scale),
    }
    return adata


def save_h5ad(adata: ad.AnnData, filename: str | Path) -> None:
    """Persist an AnnData object to disk.

    Args:
        adata: AnnData object to save.
        filename: Output path.
    """
    adata.write_h5ad(str(filename))
    logger.info("Analysis state saved to %s", filename)


def read_h5ad(filename: str | Path) -> ad.AnnData:
    """Load a previously saved AnnData object.

    Args:
        filename: Input ``.h5ad`` path.

    Returns:
        Loaded AnnData object.
    """
    return ad.read_h5ad(str(filename))
