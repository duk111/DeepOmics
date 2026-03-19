"""DeepOmics public package interface."""

from .config import AnalysisConfig
from .core import MultiOmicsEngine
from .io import load_as_anndata, preprocess_adata, read_h5ad, save_h5ad

__version__ = "0.3.0"

__all__ = [
    "AnalysisConfig",
    "MultiOmicsEngine",
    "load_as_anndata",
    "preprocess_adata",
    "read_h5ad",
    "save_h5ad",
    "__version__",
]
