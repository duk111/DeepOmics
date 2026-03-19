from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta, t
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LassoCV
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


def run_lasso(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Fit a Lasso model and return absolute coefficients."""
    X_arr, names = _as_array_and_names(X, feature_names=feature_names)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

    if X_arr.shape[1] == 0:
        return pd.Series(dtype=float)

    if config.lasso_alpha_search:
        cv = min(config.cv_folds, max(2, X_arr.shape[0] // 2))
        model = LassoCV(
            cv=cv,
            random_state=config.random_state,
            max_iter=10000,
            n_jobs=1,
        )
    else:
        model = Lasso(
            alpha=config.lasso_fixed_alpha,
            random_state=config.random_state,
            max_iter=10000,
        )

    model.fit(X_arr, y_arr)
    return pd.Series(np.abs(model.coef_), index=names, dtype=float)


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

    if X_arr.shape[1] <= target_k:
        return pd.Series(np.ones(X_arr.shape[1]), index=names, dtype=float)

    estimator = LinearSVR(random_state=config.random_state, max_iter=5000)
    step = max(1, int(X_arr.shape[1] * 0.10))
    selector = RFE(estimator, n_features_to_select=target_k, step=step)
    selector.fit(X_arr, y_arr)

    scores = 1.0 / selector.ranking_.astype(float)
    return pd.Series(scores, index=names, dtype=float)


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
    model.fit(X_arr, y_arr)
    importance = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(importance, index=names, dtype=float)


def borda_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """Aggregate rankings using Borda count."""
    ranks = rank_df.rank(ascending=False, method="average")
    n_genes = len(rank_df)
    borda_scores = (n_genes - ranks + 1).sum(axis=1)
    return borda_scores.sort_values(ascending=False)


def rra_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """Approximate robust rank aggregation using beta order statistics."""
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
        empty_df = pd.DataFrame(
            columns=[
                "LassoScore",
                "SVMRFEscore",
                "XGBoostScore",
                "BordaScore",
                "RRAScore",
                "BordaRank",
                "RRARank",
            ]
        )
        return {"intersection": [], "borda": [], "rra": []}, empty_df

    target_k = config.target_feature_count(n_samples=X_arr.shape[0], n_features=X_arr.shape[1])

    s_lasso = run_lasso(X_arr, y_arr, config, feature_names=names)
    s_svm = run_svm_rfe(X_arr, y_arr, target_k, config, feature_names=names)
    s_xgb = run_xgboost(X_arr, y_arr, target_k, config, feature_names=names)

    score_df = pd.DataFrame(
        {
            "LassoScore": s_lasso.reindex(names).fillna(0.0),
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

    selected_lasso = set(_top_k_names(score_df["LassoScore"], target_k, min_positive=True))
    selected_svm = set(_top_k_names(score_df["SVMRFEscore"], target_k))
    selected_xgb = set(_top_k_names(score_df["XGBoostScore"], target_k))

    intersection = selected_lasso & selected_svm & selected_xgb
    ordered_intersection = [gene for gene in rra_scores.index.tolist() if gene in intersection]

    results: Dict[str, list[str]] = {
        "intersection": ordered_intersection if config.enable_intersection else [],
        "borda": borda_scores.head(target_k).index.tolist() if config.enable_borda else [],
        "rra": rra_scores.head(target_k).index.tolist() if config.enable_rra else [],
    }

    ordered_score_df = score_df.sort_values(["RRARank", "BordaRank"], ascending=[True, True])
    ordered_score_df.index.name = "Gene"
    return results, ordered_score_df
