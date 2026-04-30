from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .static.pca import _compute_pca_result, _load_pca_group_table


@dataclass(frozen=True)
class VisualizationContext:
    engine: Any
    cfg: Any
    output_dir: Path
    plots_dir: Path
    pca_group_df: pd.DataFrame | None = None
    pca_adata: Any | None = None
    sample_names: tuple[str, ...] = ()
    transcriptome_pca_result: dict[str, object] | None = None
    metabolome_pca_result: dict[str, object] | None = None

    @property
    def group_df(self) -> pd.DataFrame | None:
        return self.pca_group_df

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        cfg: Any,
        group_df: pd.DataFrame | None = None,
        plots_dir: str | Path | None = None,
    ) -> "VisualizationContext":
        output_dir = Path(cfg.output_dir)
        resolved_plots_dir = Path(plots_dir) if plots_dir is not None else output_dir / "plots"
        pca_group_df = group_df if group_df is not None else _load_pca_group_table(cfg)

        pca_adata = getattr(engine, "plot_adata", getattr(engine, "unaggregated_adata", engine.adata))
        sample_names = tuple(pca_adata.obs_names.astype(str).tolist())
        transcriptome_matrix = np.asarray(pca_adata.X, dtype=np.float32)
        metabolomics_source = pca_adata.obsm.get("metabolomics_scaled", pca_adata.obsm.get("metabolomics"))
        metabolome_matrix = (
            metabolomics_source.to_numpy(dtype=np.float32, copy=False)
            if isinstance(metabolomics_source, pd.DataFrame)
            else np.asarray(metabolomics_source, dtype=np.float32)
        )

        transcriptome_pca_result = _compute_pca_result(
            transcriptome_matrix,
            list(sample_names),
            "Transcriptome PCA",
            cfg,
            group_df=pca_group_df,
            max_components=10,
        )
        metabolome_pca_result = _compute_pca_result(
            metabolome_matrix,
            list(sample_names),
            "Metabolome PCA",
            cfg,
            group_df=pca_group_df,
            max_components=10,
        )

        return cls(
            engine=engine,
            cfg=cfg,
            output_dir=output_dir,
            plots_dir=resolved_plots_dir,
            pca_group_df=pca_group_df,
            pca_adata=pca_adata,
            sample_names=sample_names,
            transcriptome_pca_result=transcriptome_pca_result,
            metabolome_pca_result=metabolome_pca_result,
        )
