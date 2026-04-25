from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata, t
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from xgboost import XGBRegressor

from .utils import get_logger

logger = get_logger()


def _as_array_and_names(
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert input features to a dense array and validated feature names."""
    if isinstance(X, pd.DataFrame):
        array = X.to_numpy(dtype=np.float32, copy=False)
        names = X.columns.astype(str).to_numpy()
    else:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        if feature_names is None:
            names = np.asarray([f"Feature_{idx}" for idx in range(array.shape[1])], dtype=str)
        else:
            names = np.asarray(feature_names, dtype=str)

    if len(names) != array.shape[1]:
        raise ValueError("feature_names length does not match the number of columns in X.")
    return array, names


def _zero_score_series(names: Sequence[str]) -> pd.Series:
    return pd.Series(0.0, index=pd.Index(names, dtype=str), dtype=float)


def _safe_nonconstant_mask(X_arr: np.ndarray) -> np.ndarray:
    if X_arr.size == 0:
        return np.zeros(X_arr.shape[1], dtype=bool)
    variances = np.nanvar(X_arr, axis=0)
    return np.isfinite(variances) & (variances > 0)


def _resolved_cv_folds(n_samples: int, requested_folds: int) -> int:
    if n_samples < 4:
        return 2
    max_safe_folds = max(2, n_samples // 2)
    return min(max_safe_folds, max(2, requested_folds))


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


def _vectorized_pearson(X_arr: np.ndarray, y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X64 = np.asarray(X_arr, dtype=np.float64)
    y64 = np.asarray(y_arr, dtype=np.float64).reshape(-1)

    if X64.shape[0] != y64.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X64.shape[0] < 3:
        raise ValueError("At least 3 samples are required for correlation screening.")

    X_centered = X64 - X64.mean(axis=0, keepdims=True)
    y_centered = y64 - y64.mean()
    x_std = X_centered.std(axis=0, ddof=1)
    y_std = float(y_centered.std(ddof=1))

    correlations = np.zeros(X64.shape[1], dtype=np.float64)
    valid = (x_std > 0) & np.isfinite(x_std) & np.isfinite(y_std) & (y_std > 0)
    if np.any(valid):
        denom = (X64.shape[0] - 1) * x_std[valid] * y_std
        correlations[valid] = (X_centered[:, valid].T @ y_centered) / denom

    correlations = np.clip(correlations, -1.0, 1.0)
    denom = np.maximum(1.0 - correlations**2, 1e-12)
    t_stat = correlations * np.sqrt((X64.shape[0] - 2) / denom)
    p_values = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=X64.shape[0] - 2))
    return correlations.astype(float), np.clip(p_values.astype(float), 0.0, 1.0)


def _vectorized_spearman(X_arr: np.ndarray, y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_rank = np.apply_along_axis(rankdata, 0, np.asarray(X_arr, dtype=np.float64))
    y_rank = rankdata(np.asarray(y_arr, dtype=np.float64).reshape(-1))
    return _vectorized_pearson(X_rank, y_rank)


class ScreeningCorrelationCache:
    """Reusable Pearson/Spearman screening state for one feature matrix and many targets."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        pearson_r: np.ndarray,
        spearman_rho: np.ndarray,
        feature_nonconstant_mask: np.ndarray,
        n_samples: int,
    ) -> None:
        self.feature_names = np.asarray(feature_names, dtype=str)
        self.pearson_r = np.asarray(pearson_r, dtype=float)
        self.spearman_rho = np.asarray(spearman_rho, dtype=float)
        self.feature_nonconstant_mask = np.asarray(feature_nonconstant_mask, dtype=bool)
        self.n_samples = int(n_samples)

        if self.pearson_r.shape != self.spearman_rho.shape:
            raise ValueError("pearson_r and spearman_rho must have the same shape.")
        if self.pearson_r.shape[0] != len(self.feature_names):
            raise ValueError("Correlation matrices must contain one row per feature name.")
        if len(self.feature_nonconstant_mask) != len(self.feature_names):
            raise ValueError("feature_nonconstant_mask length must match feature_names.")

    def correlations_for(self, target_index: int) -> tuple[np.ndarray, np.ndarray]:
        target_index = int(target_index)
        if target_index < 0 or target_index >= self.pearson_r.shape[1]:
            raise IndexError("target_index is out of range for the screening correlation cache.")
        return (
            self.pearson_r[:, target_index].astype(float, copy=False),
            self.spearman_rho[:, target_index].astype(float, copy=False),
        )


def _standardize_columns_for_correlation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be a 2D array.")

    centered = arr - arr.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1)
    valid = np.isfinite(std) & (std > 0)

    standardized = np.zeros_like(centered, dtype=np.float64)
    if np.any(valid):
        standardized[:, valid] = centered[:, valid] / std[valid]
    return standardized, valid


def _correlation_p_values(correlations: np.ndarray, n_samples: int) -> np.ndarray:
    corr = np.clip(np.nan_to_num(np.asarray(correlations, dtype=float), nan=0.0), -1.0, 1.0)
    if int(n_samples) < 3:
        raise ValueError("At least 3 samples are required for correlation screening.")

    denom = np.maximum(1.0 - corr**2, 1e-12)
    t_stat = corr * np.sqrt((int(n_samples) - 2) / denom)
    p_values = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=int(n_samples) - 2))
    return np.clip(p_values.astype(float), 0.0, 1.0)


def prepare_screening_correlation_cache(
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> ScreeningCorrelationCache:
    """Precompute reusable Pearson/Spearman correlations for all screening targets.

    The feature matrix is standardized and rank-transformed once, then multiplied
    against all target columns at once. This avoids repeating the same gene-matrix
    centering, scaling, and rank transformation for every metabolite.
    """
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    Y_arr = np.asarray(Y, dtype=np.float32)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    if Y_arr.ndim != 2:
        raise ValueError("Y must be a 1D or 2D target array/DataFrame.")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if X_arr.shape[0] < 3:
        raise ValueError("At least 3 samples are required for correlation screening.")

    X_z, feature_nonconstant_mask = _standardize_columns_for_correlation(X_arr)
    Y_z, _ = _standardize_columns_for_correlation(Y_arr)
    pearson_r = np.clip((X_z.T @ Y_z) / float(X_arr.shape[0] - 1), -1.0, 1.0)

    X_rank = np.apply_along_axis(rankdata, 0, np.asarray(X_arr, dtype=np.float64))
    Y_rank = np.apply_along_axis(rankdata, 0, np.asarray(Y_arr, dtype=np.float64))
    X_rank_z, _ = _standardize_columns_for_correlation(X_rank)
    Y_rank_z, _ = _standardize_columns_for_correlation(Y_rank)
    spearman_rho = np.clip((X_rank_z.T @ Y_rank_z) / float(X_arr.shape[0] - 1), -1.0, 1.0)

    return ScreeningCorrelationCache(
        feature_names=names,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        feature_nonconstant_mask=feature_nonconstant_mask,
        n_samples=int(X_arr.shape[0]),
    )


def _top_names_from_series(
    series: pd.Series,
    top_k: int,
    *,
    absolute: bool = False,
    secondary: Optional[pd.Series] = None,
) -> list[str]:
    if series.empty:
        return []

    work = pd.DataFrame({"Primary": series.astype(float)})
    work["_GeneName"] = work.index.astype(str)
    work["PrimarySort"] = work["Primary"].abs() if absolute else work["Primary"]

    if secondary is not None:
        work["Secondary"] = secondary.reindex(work.index).astype(float).fillna(np.inf)
        sort_cols = ["PrimarySort", "Secondary", "_GeneName"]
        ascending = [False, True, True]
    else:
        sort_cols = ["PrimarySort", "_GeneName"]
        ascending = [False, True]

    ranked = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return ranked.head(max(1, int(top_k))).index.astype(str).tolist()


def screen_genes_three_way(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
    *,
    correlation_cache: ScreeningCorrelationCache | None = None,
    target_index: int | None = None,
) -> pd.DataFrame:
    """Three-way gene screening using Pearson, Spearman, and mutual information."""
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X_arr.shape[0] < 3:
        raise ValueError("At least 3 samples are required for screening.")
    if X_arr.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "PearsonR",
                "PearsonP",
                "PearsonFDR",
                "SpearmanRho",
                "SpearmanP",
                "SpearmanFDR",
                "MIScore",
                "In_PCC",
                "In_Spearman",
                "In_MI",
                "ScreenSupportCount",
            ],
            index=pd.Index([], name="Gene"),
        )

    if correlation_cache is not None:
        if target_index is None:
            raise ValueError("target_index must be provided when correlation_cache is used.")
        if correlation_cache.n_samples != X_arr.shape[0]:
            raise ValueError("correlation_cache sample count does not match X.")
        if not np.array_equal(correlation_cache.feature_names.astype(str), names.astype(str)):
            raise ValueError("correlation_cache feature names do not match X/feature_names.")
        pearson_r, spearman_rho = correlation_cache.correlations_for(int(target_index))
        pearson_p = _correlation_p_values(pearson_r, X_arr.shape[0])
        spearman_p = _correlation_p_values(spearman_rho, X_arr.shape[0])
        valid_mask = correlation_cache.feature_nonconstant_mask
    else:
        pearson_r, pearson_p = _vectorized_pearson(X_arr, y_arr)
        spearman_rho, spearman_p = _vectorized_spearman(X_arr, y_arr)
        valid_mask = _safe_nonconstant_mask(X_arr)

    pearson_fdr = _bh_fdr(pearson_p)
    spearman_fdr = _bh_fdr(spearman_p)

    mi_scores = np.zeros(X_arr.shape[1], dtype=float)
    if np.any(valid_mask) and np.nanstd(y_arr, ddof=1) > 0:
        try:
            mi_scores_valid = mutual_info_regression(
                np.asarray(X_arr[:, valid_mask], dtype=np.float64),
                np.asarray(y_arr, dtype=np.float64),
                random_state=config.random_state,
            )
            mi_scores[valid_mask] = np.asarray(mi_scores_valid, dtype=float)
        except Exception as exc:  # pragma: no cover
            logger.warning("Mutual information screening failed; MI scores will be zero. Reason: %s", exc)

    screen_df = pd.DataFrame(
        {
            "PearsonR": pearson_r,
            "PearsonP": pearson_p,
            "PearsonFDR": pearson_fdr,
            "SpearmanRho": spearman_rho,
            "SpearmanP": spearman_p,
            "SpearmanFDR": spearman_fdr,
            "MIScore": mi_scores,
        },
        index=pd.Index(names, name="Gene"),
    ).replace([np.inf, -np.inf], np.nan)

    top_k = max(1, int(getattr(config, "screen_top_k_per_method", 1000)))
    use_fdr = bool(getattr(config, "use_fdr", False))
    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))

    if use_fdr:
        pcc_mask = screen_df["PearsonFDR"].fillna(1.0) <= fdr_alpha
        pcc_primary = screen_df.loc[pcc_mask, "PearsonR"]
        pcc_secondary = screen_df.loc[pcc_mask, "PearsonFDR"]
    else:
        pcc_primary = screen_df["PearsonR"]
        pcc_secondary = screen_df["PearsonFDR"]

    pcc_genes = set(
        _top_names_from_series(
            pcc_primary,
            top_k,
            absolute=True,
            secondary=pcc_secondary,
        )
    )

    if use_fdr:
        spearman_mask = screen_df["SpearmanFDR"].fillna(1.0) <= fdr_alpha
        spearman_primary = screen_df.loc[spearman_mask, "SpearmanRho"]
        spearman_secondary = screen_df.loc[spearman_mask, "SpearmanFDR"]
    else:
        spearman_primary = screen_df["SpearmanRho"]
        spearman_secondary = screen_df["SpearmanFDR"]

    spearman_genes = set(
        _top_names_from_series(
            spearman_primary,
            top_k,
            absolute=True,
            secondary=spearman_secondary,
        )
    )
    mi_genes = set(_top_names_from_series(screen_df["MIScore"], top_k, absolute=False))

    candidate_genes = pcc_genes | spearman_genes | mi_genes
    if not candidate_genes:
        return screen_df.iloc[0:0].copy()

    candidate_df = screen_df.loc[sorted(candidate_genes)].copy()
    candidate_index = candidate_df.index.to_series()
    candidate_df["In_PCC"] = candidate_index.isin(pcc_genes).astype(int)
    candidate_df["In_Spearman"] = candidate_index.isin(spearman_genes).astype(int)
    candidate_df["In_MI"] = candidate_index.isin(mi_genes).astype(int)
    candidate_df["ScreenSupportCount"] = (
        candidate_df["In_PCC"] + candidate_df["In_Spearman"] + candidate_df["In_MI"]
    ).astype(int)

    candidate_df["MaxAbsCorr"] = np.maximum(
        candidate_df["PearsonR"].abs(),
        candidate_df["SpearmanRho"].abs(),
    )
    candidate_df = candidate_df.sort_values(
        ["ScreenSupportCount", "MaxAbsCorr", "MIScore", "PearsonFDR", "SpearmanFDR"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).drop(columns=["MaxAbsCorr"])
    return candidate_df


def _fit_elastic_net_prepared(
    X_work: np.ndarray,
    y_arr: np.ndarray,
    config,
) -> np.ndarray:
    try:
        if config.elastic_net_alpha_search and X_work.shape[0] >= 4:
            cv = _resolved_cv_folds(X_work.shape[0], config.cv_folds)
            model = ElasticNetCV(
                l1_ratio=config.elastic_net_l1_ratio_grid,
                cv=cv,
                random_state=config.random_state,
                max_iter=config.elastic_net_max_iter,
                n_jobs=1,
            )
        else:
            model = ElasticNet(
                alpha=config.elastic_net_fixed_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
                random_state=config.random_state,
                max_iter=config.elastic_net_max_iter,
            )

        model.fit(X_work, y_arr)
        return np.abs(np.asarray(model.coef_, dtype=float))
    except Exception as exc:  # pragma: no cover
        logger.warning("Elastic Net fitting failed; falling back to fixed-alpha mode. Reason: %s", exc)
        try:
            fallback_model = ElasticNet(
                alpha=config.elastic_net_fixed_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
                random_state=config.random_state,
                max_iter=config.elastic_net_max_iter,
            )
            fallback_model.fit(X_work, y_arr)
            return np.abs(np.asarray(fallback_model.coef_, dtype=float))
        except Exception as fallback_exc:  # pragma: no cover
            logger.warning("Elastic Net fallback also failed. Reason: %s", fallback_exc)
            return np.zeros(X_work.shape[1], dtype=float)


def run_elastic_net(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        return _zero_score_series(names)

    X_work = X_arr[:, valid_mask]
    valid_names = names[valid_mask]

    if np.nanstd(y_arr, ddof=1) == 0 or X_work.shape[0] < 3:
        return _zero_score_series(names)

    coef = _fit_elastic_net_prepared(X_work, y_arr, config)

    result = _zero_score_series(names)
    result.loc[valid_names] = coef
    return result


def _fit_xgboost_prepared(
    X_work: np.ndarray,
    y_arr: np.ndarray,
    config,
) -> np.ndarray:
    try:
        model = XGBRegressor(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=config.resolved_threads(),
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X_work, y_arr)
        return np.asarray(model.feature_importances_, dtype=float)
    except Exception as exc:  # pragma: no cover
        logger.warning("XGBoost fitting failed; using zero scores for this metabolite. Reason: %s", exc)
        return np.zeros(X_work.shape[1], dtype=float)


def run_xgboost(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        return _zero_score_series(names)
    if np.nanstd(y_arr, ddof=1) == 0:
        return _zero_score_series(names)

    X_work = X_arr[:, valid_mask]
    valid_names = names[valid_mask]

    importance = _fit_xgboost_prepared(X_work, y_arr, config)

    result = _zero_score_series(names)
    result.loc[valid_names] = importance
    return result


def _ordinal_rank_desc(score_series: pd.Series) -> pd.Series:
    if score_series.empty:
        return pd.Series(dtype=int)

    work = pd.DataFrame(
        {
            "Score": score_series.astype(float).to_numpy(),
            "_GeneName": score_series.index.astype(str).to_numpy(),
        },
        index=score_series.index,
    )
    ordered = work.sort_values(
        ["Score", "_GeneName"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered, dtype=int)
    return ranks.reindex(score_series.index).astype(int)


def _ordinal_rank_asc(score_series: pd.Series) -> pd.Series:
    if score_series.empty:
        return pd.Series(dtype=int)

    work = pd.DataFrame(
        {
            "Score": score_series.astype(float).to_numpy(),
            "_GeneName": score_series.index.astype(str).to_numpy(),
        },
        index=score_series.index,
    )
    ordered = work.sort_values(
        ["Score", "_GeneName"],
        ascending=[True, True],
        kind="mergesort",
    ).index
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered, dtype=int)
    return ranks.reindex(score_series.index).astype(int)


def rra_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """Robust rank aggregation from rank columns where smaller ranks are better."""
    if rank_df.empty:
        return pd.Series(dtype=float)

    ranks = rank_df.astype(float).clip(lower=1.0)
    n_genes = max(1, ranks.shape[0])
    n_methods = max(1, ranks.shape[1])
    normalized = np.sort((ranks.to_numpy(dtype=float) / n_genes), axis=1)

    best_p = np.ones(normalized.shape[0], dtype=float)
    for k in range(n_methods):
        best_p = np.minimum(best_p, beta.cdf(normalized[:, k], k + 1, n_methods - k))

    adjusted = np.minimum(best_p * n_methods, 1.0)
    return pd.Series(adjusted, index=rank_df.index, dtype=float)


def _selected_from_scores(
    score_series: pd.Series,
    rank_series: pd.Series,
    target_k: int,
    *,
    require_positive: bool = True,
) -> pd.Series:
    selected = rank_series <= max(1, int(target_k))
    if require_positive:
        selected = selected & (score_series > 0)
    return selected.astype(int)


def _empty_model_table(index: Sequence[str]) -> pd.DataFrame:
    gene_index = pd.Index(index, name="Gene", dtype=str)
    return pd.DataFrame(
        {
            "ElasticNetScore": pd.Series(0.0, index=gene_index, dtype=float),
            "ElasticNetRank": pd.Series(np.arange(1, len(gene_index) + 1), index=gene_index, dtype=int),
            "ElasticNetSelected": pd.Series(0, index=gene_index, dtype=int),
            "XGBoostScore": pd.Series(0.0, index=gene_index, dtype=float),
            "XGBoostRank": pd.Series(np.arange(1, len(gene_index) + 1), index=gene_index, dtype=int),
            "XGBoostSelected": pd.Series(0, index=gene_index, dtype=int),
            "ModelSupportCount": pd.Series(0, index=gene_index, dtype=int),
            "RRAScore": pd.Series(1.0, index=gene_index, dtype=float),
            "RRARank": pd.Series(np.arange(1, len(gene_index) + 1), index=gene_index, dtype=int),
        },
        index=gene_index,
    )


def run_association_models(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """Run ElasticNet + XGBoost modeling and aggregate ranks with RRA."""
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return _empty_model_table([]), 0

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        logger.info("All candidate genes are constant after screening; skipping model fitting.")
        return _empty_model_table(names), 0

    if X_arr.shape[0] < 3 or np.nanstd(y_arr, ddof=1) == 0:
        logger.info("Skipping model fitting because the target metabolite has insufficient variation.")
        return _empty_model_table(names), 0

    target_k = config.target_feature_count(
        n_samples=X_arr.shape[0],
        n_features=int(valid_mask.sum()),
    )

    X_work = X_arr[:, valid_mask]
    enet_coef = _fit_elastic_net_prepared(X_work, y_arr, config)
    xgb_importance = _fit_xgboost_prepared(X_work, y_arr, config)

    enet_values = np.zeros(len(names), dtype=float)
    xgb_values = np.zeros(len(names), dtype=float)
    enet_values[valid_mask] = enet_coef
    xgb_values[valid_mask] = xgb_importance

    s_enet = pd.Series(enet_values, index=pd.Index(names, name="Gene"), dtype=float)
    s_xgb = pd.Series(xgb_values, index=pd.Index(names, name="Gene"), dtype=float)

    enet_rank = _ordinal_rank_desc(s_enet)
    xgb_rank = _ordinal_rank_desc(s_xgb)

    rank_input = pd.DataFrame(
        {
            "ElasticNetRank": enet_rank,
            "XGBoostRank": xgb_rank,
        },
        index=pd.Index(names, name="Gene"),
    )
    rra_score = rra_aggregation(rank_input).reindex(names).fillna(1.0)
    rra_rank = _ordinal_rank_asc(rra_score)

    score_df = pd.DataFrame(
        {
            "ElasticNetScore": s_enet,
            "ElasticNetRank": enet_rank,
            "ElasticNetSelected": _selected_from_scores(s_enet, enet_rank, target_k, require_positive=True),
            "XGBoostScore": s_xgb,
            "XGBoostRank": xgb_rank,
            "XGBoostSelected": _selected_from_scores(s_xgb, xgb_rank, target_k, require_positive=True),
            "RRAScore": rra_score,
            "RRARank": rra_rank,
        },
        index=pd.Index(names, name="Gene"),
    )
    score_df["ModelSupportCount"] = (
        score_df["ElasticNetSelected"] + score_df["XGBoostSelected"]
    ).astype(int)

    ordered_score_df = score_df.sort_values(
        ["RRARank", "ModelSupportCount", "ElasticNetRank", "XGBoostRank"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return ordered_score_df, int(target_k)
