from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta, t
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from .utils import get_logger

logger = get_logger()


def _as_array_and_names(
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert input features to a dense array and validated feature names.

    Args:
        X: Feature matrix.
        feature_names: Optional feature names when ``X`` is a NumPy array.

    Returns:
        Tuple of dense matrix and feature-name array.
    """
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


def _empty_score_table(index: Sequence[str]) -> pd.DataFrame:
    """Create an empty score table with the expected schema."""
    return pd.DataFrame(
        {
            "ElasticNetScore": pd.Series(0.0, index=index, dtype=float),
            "SVMRFEscore": pd.Series(0.0, index=index, dtype=float),
            "XGBoostScore": pd.Series(0.0, index=index, dtype=float),
            "BordaScore": pd.Series(0.0, index=index, dtype=float),
            "RRAScore": pd.Series(1.0, index=index, dtype=float),
            "BordaRank": pd.Series(np.arange(1, len(index) + 1), index=index, dtype=int),
            "RRARank": pd.Series(np.arange(1, len(index) + 1), index=index, dtype=int),
        },
        index=pd.Index(index, name="Gene"),
    )


def _safe_nonconstant_mask(X_arr: np.ndarray) -> np.ndarray:
    """Identify columns with finite, non-zero variance."""
    if X_arr.size == 0:
        return np.zeros(X_arr.shape[1], dtype=bool)
    variances = np.nanvar(X_arr, axis=0)
    return np.isfinite(variances) & (variances > 0)


def _resolved_cv_folds(n_samples: int, requested_folds: int) -> int:
    """Resolve a safe CV fold number for small-sample regression."""
    if n_samples < 4:
        return 2
    max_safe_folds = max(2, n_samples // 2)
    return min(max_safe_folds, max(2, requested_folds))


def filter_by_pcc(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
    return_stats: bool = False,
):
    """Perform vectorized Pearson correlation screening.

    Args:
        X: Gene matrix with shape ``(n_samples, n_features)``.
        y: Target vector with shape ``(n_samples,)``.
        config: ``AnalysisConfig`` instance.
        feature_names: Optional feature names when ``X`` is a NumPy array.
        return_stats: Whether to additionally return the full statistics table.

    Returns:
        List of selected feature names, or a tuple ``(selected_names, stats_df)``.
    """
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X_arr.shape[0] < 3:
        raise ValueError("At least 3 samples are required for Pearson correlation screening.")

    X_centered = X_arr - X_arr.mean(axis=0, keepdims=True)
    y_centered = y_arr - y_arr.mean()
    x_std = X_centered.std(axis=0, ddof=1)
    y_std = float(y_centered.std(ddof=1))

    valid = (x_std > 0) & np.isfinite(x_std) & np.isfinite(y_std) & (y_std > 0)
    correlations = np.zeros(X_arr.shape[1], dtype=np.float64)
    if np.any(valid):
        denom = (X_arr.shape[0] - 1) * x_std[valid] * y_std
        correlations[valid] = (X_centered[:, valid].T @ y_centered) / denom

    correlations = np.clip(correlations, -1.0, 1.0)
    t_stat = correlations * np.sqrt((X_arr.shape[0] - 2) / np.maximum(1.0 - correlations**2, 1e-12))
    p_values = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=X_arr.shape[0] - 2))

    stats_df = pd.DataFrame(
        {
            "R": correlations,
            "AbsR": np.abs(correlations),
            "P": p_values,
        },
        index=names,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if config.use_fdr:
        ranked = stats_df["P"].sort_values()
        n_tests = max(1, len(ranked))
        bh_threshold = config.fdr_alpha * (np.arange(1, n_tests + 1) / n_tests)
        reject = ranked.values <= bh_threshold
        max_idx = np.where(reject)[0].max() if reject.any() else -1
        p_cutoff = ranked.iloc[max_idx] if max_idx >= 0 else -np.inf
        mask = (stats_df["P"] <= p_cutoff) & (stats_df["AbsR"] >= config.pcc_r_threshold)
    else:
        mask = (stats_df["P"] <= config.pcc_p_threshold) & (stats_df["AbsR"] >= config.pcc_r_threshold)

    selected_df = stats_df.loc[mask].sort_values(["AbsR", "P"], ascending=[False, True])

    if config.max_candidate_genes is not None and len(selected_df) > config.max_candidate_genes:
        logger.info(
            "PCC prefilter selected %d genes; retaining top %d by |r| for memory stability.",
            len(selected_df),
            config.max_candidate_genes,
        )
        selected_df = selected_df.head(config.max_candidate_genes)

    selected_names = selected_df.index.tolist()
    if return_stats:
        return selected_names, selected_df
    return selected_names


def run_elastic_net(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Fit an Elastic Net model and return absolute coefficients.

    The implementation is robust to small-sample high-dimensional settings and
    falls back to a fixed-alpha model when CV is not reliable.
    """
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        return pd.Series(0.0, index=names, dtype=float)

    X_work = X_arr[:, valid_mask]
    valid_names = names[valid_mask]

    if np.nanstd(y_arr, ddof=1) == 0 or X_work.shape[0] < 3:
        return pd.Series(0.0, index=names, dtype=float)

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
        coef = np.abs(np.asarray(model.coef_, dtype=float))
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
            coef = np.abs(np.asarray(fallback_model.coef_, dtype=float))
        except Exception as fallback_exc:  # pragma: no cover
            logger.warning("Elastic Net fallback also failed. Reason: %s", fallback_exc)
            return pd.Series(0.0, index=names, dtype=float)

    result = pd.Series(0.0, index=names, dtype=float)
    result.loc[valid_names] = coef
    return result


def run_svm_rfe(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    target_k: int,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Run LinearSVR-based recursive feature elimination."""
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        return pd.Series(0.0, index=names, dtype=float)

    if np.nanstd(y_arr, ddof=1) == 0:
        return pd.Series(0.0, index=names, dtype=float)

    X_work = X_arr[:, valid_mask]
    valid_names = names[valid_mask]
    safe_target_k = min(max(1, target_k), X_work.shape[1])

    if X_work.shape[1] <= safe_target_k:
        result = pd.Series(0.0, index=names, dtype=float)
        result.loc[valid_names] = 1.0
        return result

    try:
        estimator = LinearSVR(random_state=config.random_state, max_iter=5000)
        step = max(1, int(X_work.shape[1] * 0.10))
        selector = RFE(estimator, n_features_to_select=safe_target_k, step=step)
        selector.fit(X_work, y_arr)
        scores = 1.0 / selector.ranking_.astype(float)
    except Exception as exc:  # pragma: no cover
        logger.warning("SVM-RFE failed; using zero scores for this metabolite. Reason: %s", exc)
        return pd.Series(0.0, index=names, dtype=float)

    result = pd.Series(0.0, index=names, dtype=float)
    result.loc[valid_names] = scores
    return result


def run_xgboost(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    target_k: int,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Fit an XGBoost regressor and return feature importance scores."""
    del target_k
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        return pd.Series(0.0, index=names, dtype=float)

    if np.nanstd(y_arr, ddof=1) == 0:
        return pd.Series(0.0, index=names, dtype=float)

    X_work = X_arr[:, valid_mask]
    valid_names = names[valid_mask]

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
        importance = np.asarray(model.feature_importances_, dtype=float)
    except Exception as exc:  # pragma: no cover
        logger.warning("XGBoost fitting failed; using zero scores for this metabolite. Reason: %s", exc)
        return pd.Series(0.0, index=names, dtype=float)

    result = pd.Series(0.0, index=names, dtype=float)
    result.loc[valid_names] = importance
    return result


def borda_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """Aggregate rankings using Borda count."""
    if rank_df.empty:
        return pd.Series(dtype=float)
    ranks = rank_df.rank(ascending=False, method="average")
    n_genes = len(rank_df)
    borda_scores = (n_genes - ranks + 1).sum(axis=1)
    return borda_scores.sort_values(ascending=False)


def rra_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """Approximate robust rank aggregation using beta order statistics."""
    if rank_df.empty:
        return pd.Series(dtype=float)
    ranks = rank_df.rank(ascending=False, method="average")
    n_genes = max(1, rank_df.shape[0])
    n_methods = max(1, rank_df.shape[1])
    normalized = np.sort((ranks.to_numpy(dtype=float) / n_genes), axis=1)

    best_p = np.ones(normalized.shape[0], dtype=float)
    for k in range(n_methods):
        best_p = np.minimum(best_p, beta.cdf(normalized[:, k], k + 1, n_methods - k))

    adjusted = np.minimum(best_p * n_methods, 1.0)
    return pd.Series(adjusted, index=rank_df.index, dtype=float).sort_values(ascending=True)


def _top_k_names(score_series: pd.Series, target_k: int, min_positive: bool = False) -> list[str]:
    """Return the top-k feature names from a score series."""
    if score_series.empty:
        return []
    series = score_series.copy()
    if min_positive:
        series = series[series > 0]
    if series.empty:
        return []
    return series.sort_values(ascending=False).head(target_k).index.tolist()


def get_integrated_key_genes(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
):
    """Run the ensemble feature-selection workflow."""
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return {"intersection": [], "borda": [], "rra": []}, _empty_score_table([])

    valid_mask = _safe_nonconstant_mask(X_arr)
    if not np.any(valid_mask):
        logger.info("All candidate genes are constant after PCC screening; skipping model fitting.")
        return {"intersection": [], "borda": [], "rra": []}, _empty_score_table(names)

    if X_arr.shape[0] < 3 or np.nanstd(y_arr, ddof=1) == 0:
        logger.info("Skipping model fitting because the target metabolite has insufficient variation.")
        return {"intersection": [], "borda": [], "rra": []}, _empty_score_table(names)

    target_k = config.target_feature_count(n_samples=X_arr.shape[0], n_features=int(valid_mask.sum()))

    s_enet = run_elastic_net(X_arr, y_arr, config, feature_names=names)
    s_svm = run_svm_rfe(X_arr, y_arr, target_k, config, feature_names=names)
    s_xgb = run_xgboost(X_arr, y_arr, target_k, config, feature_names=names)

    score_df = pd.DataFrame(
        {
            "ElasticNetScore": s_enet.reindex(names).fillna(0.0),
            "SVMRFEscore": s_svm.reindex(names).fillna(0.0),
            "XGBoostScore": s_xgb.reindex(names).fillna(0.0),
        },
        index=names,
    )

    borda_scores = borda_aggregation(score_df)
    rra_scores = rra_aggregation(score_df)

    score_df["BordaScore"] = borda_scores.reindex(names).fillna(0.0)
    score_df["RRAScore"] = rra_scores.reindex(names).fillna(1.0)
    score_df["BordaRank"] = score_df["BordaScore"].rank(ascending=False, method="min").astype(int)
    score_df["RRARank"] = score_df["RRAScore"].rank(ascending=True, method="min").astype(int)

    selected_enet = set(_top_k_names(score_df["ElasticNetScore"], target_k, min_positive=True))
    selected_svm = set(_top_k_names(score_df["SVMRFEscore"], target_k))
    selected_xgb = set(_top_k_names(score_df["XGBoostScore"], target_k))

    intersection = selected_enet & selected_svm & selected_xgb
    ordered_intersection = [gene for gene in rra_scores.index.tolist() if gene in intersection]

    results: Dict[str, list[str]] = {
        "intersection": ordered_intersection if config.enable_intersection else [],
        "borda": borda_scores.head(target_k).index.tolist() if config.enable_borda else [],
        "rra": rra_scores.head(target_k).index.tolist() if config.enable_rra else [],
    }

    ordered_score_df = score_df.sort_values(["RRARank", "BordaRank"], ascending=[True, True])
    ordered_score_df.index.name = "Gene"
    return results, ordered_score_df
