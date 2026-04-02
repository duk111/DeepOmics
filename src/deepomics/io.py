from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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


def _impute_missing_with_column_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using column means."""
    if not df.isna().values.any():
        return df

    all_missing = df.columns[df.isna().all(axis=0)].tolist()
    if all_missing:
        raise ValueError(
            "The following features contain only missing values and cannot be imputed: "
            f"{all_missing[:5]}"
        )
    return df.fillna(df.mean(axis=0))


def load_as_anndata(gene_path: str | Path, metab_path: str | Path) -> ad.AnnData:
    """Load transcriptome and metabolome matrices into a single AnnData object."""
    logger.info("Loading transcriptome data from %s", gene_path)
    df_gene = _read_feature_table(gene_path, label="Transcriptome")

    logger.info("Loading metabolomics data from %s", metab_path)
    df_metab = _read_feature_table(metab_path, label="Metabolomics")

    df_gene_t = _impute_missing_with_column_mean(df_gene.T)
    df_metab_t = _impute_missing_with_column_mean(df_metab.T)

    common_samples = df_gene_t.index.intersection(df_metab_t.index, sort=False)
    if len(common_samples) == 0:
        raise ValueError("No shared sample IDs were found between transcriptome and metabolomics tables.")

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
    }
    return adata


def preprocess_adata(adata: ad.AnnData, scale: bool = True) -> ad.AnnData:
    """Apply basic preprocessing to the AnnData object."""
    if "metabolomics" not in adata.obsm:
        raise KeyError("adata.obsm['metabolomics'] is required.")

    gene_df = pd.DataFrame(
        np.asarray(adata.X, dtype=np.float32),
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )
    gene_df = _impute_missing_with_column_mean(gene_df)

    gene_var = np.nanvar(gene_df.to_numpy(dtype=np.float32, copy=False), axis=0)
    keep_genes = gene_var > 0
    if not np.all(keep_genes):
        dropped = int((~keep_genes).sum())
        logger.warning("Removed %d constant genes before modeling.", dropped)
        gene_df = gene_df.loc[:, keep_genes]
        adata = adata[:, keep_genes].copy()

    if gene_df.shape[1] == 0:
        raise ValueError("No genes remain after preprocessing.")

    adata.X = gene_df.to_numpy(dtype=np.float32, copy=False)
    adata.var_names = pd.Index(gene_df.columns.astype(str), name="Gene")

    metab_df = adata.obsm["metabolomics"]
    if not isinstance(metab_df, pd.DataFrame):
        metabolite_names = adata.uns.get("metabolite_names", [])
        metab_df = pd.DataFrame(metab_df, index=adata.obs_names, columns=metabolite_names)
    metab_df = metab_df.copy()
    metab_df.index = adata.obs_names.astype(str)
    metab_df.columns = metab_df.columns.astype(str)
    metab_df = _impute_missing_with_column_mean(metab_df.astype(np.float32, copy=False))

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
    }
    return adata


def save_h5ad(adata: ad.AnnData, filename: str | Path) -> None:
    """Persist an AnnData object to disk."""
    adata.write_h5ad(str(filename))
    logger.info("Analysis state saved to %s", filename)


def read_h5ad(filename: str | Path) -> ad.AnnData:
    """Load a previously saved AnnData object."""
    return ad.read_h5ad(str(filename))
