from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .utils import (
    align_metabolites_and_metadata,
    build_contrasts,
    build_dem_group,
    load_metadata,
    load_metabolites,
    validate_inputs,
)


def _ensure_output_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_plot_dir(out_dir: Path, name: str) -> Path:
    plot_dir = out_dir / "plots" / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.notna() & np.isfinite(pvalues)
    if not valid.any():
        return adjusted

    p = pvalues.loc[valid].astype(float)
    order = np.argsort(p.to_numpy())
    ordered = p.to_numpy()[order]
    n = len(ordered)
    ranks = np.arange(1, n + 1, dtype=float)
    q = ordered * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    valid_index = p.index.to_numpy()
    adjusted.loc[valid_index[order]] = q
    return adjusted


def _median_impute(values: pd.DataFrame) -> pd.DataFrame:
    medians = values.median(axis=0, skipna=True)
    if medians.isna().any():
        missing = medians.loc[medians.isna()].index.astype(str).tolist()
        raise ValueError(f"Metabolites contain only missing values after filtering, for example: {missing[:10]}")
    return values.fillna(medians)


def _fit_predictive_component(x_scaled: np.ndarray, y_centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = x_scaled.T @ y_centered
    weight_norm = float(np.linalg.norm(weights))
    if weight_norm <= np.finfo(float).eps:
        raise ValueError("OPLS-DA cannot be fitted because X has no covariance with the class labels.")

    weights = weights / weight_norm
    scores = x_scaled @ weights
    score_ss = float(scores.T @ scores)
    if score_ss <= np.finfo(float).eps:
        raise ValueError("OPLS-DA cannot be fitted because the predictive score has near-zero variance.")

    loadings = x_scaled.T @ scores / score_ss
    y_loading = float(y_centered.T @ scores / score_ss)
    return weights, scores, loadings, y_loading


def _fit_opls_da_vip(
    x: pd.DataFrame,
    y_binary: np.ndarray,
    n_orthogonal_components: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any], pd.Series | None]:
    """Fit a two-class OPLS-DA style model and return predictive VIP values."""
    if len(np.unique(y_binary)) != 2:
        raise ValueError("OPLS-DA requires exactly two classes in each contrast.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.to_numpy(dtype=float))
    y_centered = y_binary.astype(float) - float(np.mean(y_binary))

    initial_weights, _, initial_loadings, _ = _fit_predictive_component(x_scaled, y_centered)
    corrected_x = x_scaled.copy()
    residual_x = x_scaled.copy()
    orthogonal_scores: list[np.ndarray] = []

    for _ in range(max(0, n_orthogonal_components)):
        projection = float(initial_weights.T @ initial_loadings) / float(initial_weights.T @ initial_weights)
        orthogonal_weights = initial_loadings - initial_weights * projection
        weight_norm = float(np.linalg.norm(orthogonal_weights))
        if weight_norm <= np.finfo(float).eps:
            break

        orthogonal_weights = orthogonal_weights / weight_norm
        orthogonal_score = residual_x @ orthogonal_weights
        score_ss = float(orthogonal_score.T @ orthogonal_score)
        if score_ss <= np.finfo(float).eps:
            break

        orthogonal_loading = residual_x.T @ orthogonal_score / score_ss
        orthogonal_part = np.outer(orthogonal_score, orthogonal_loading)
        corrected_x = corrected_x - orthogonal_part
        residual_x = residual_x - orthogonal_part
        orthogonal_scores.append(orthogonal_score)

    weights, predictive_score, _, y_loading = _fit_predictive_component(corrected_x, y_centered)
    weight_ss = float(weights.T @ weights)
    vip = np.sqrt(len(weights) * np.square(weights) / weight_ss)

    y_hat = predictive_score * y_loading
    sst = float(np.sum(np.square(y_centered)))
    sse = float(np.sum(np.square(y_centered - y_hat)))
    r2y = float(1.0 - sse / sst) if sst > 0 else np.nan

    first_orthogonal_score = None
    if orthogonal_scores:
        first_orthogonal_score = pd.Series(orthogonal_scores[0], index=x.index, name="to1")
    predictive_score_series = pd.Series(predictive_score, index=x.index, name="tp1")

    summary = {
        "n_metabolites_in_model": int(x.shape[1]),
        "n_samples_in_model": int(x.shape[0]),
        "n_orthogonal_components": int(len(orthogonal_scores)),
        "r2y": r2y,
    }
    return pd.Series(vip, index=x.columns, name="vip"), predictive_score_series, summary, first_orthogonal_score


def _classify_points(
    results_df: pd.DataFrame,
    vip_cutoff: float,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> pd.Series:
    significant = (
        results_df["vip"].notna()
        & (results_df["vip"] >= vip_cutoff)
        & results_df["padj_bh"].notna()
        & (results_df["padj_bh"] <= padj_cutoff)
        & results_df["log2FoldChange"].notna()
        & (results_df["log2FoldChange"].abs() >= log2fc_cutoff)
    )
    up = significant & (results_df["log2FoldChange"] > 0)
    down = significant & (results_df["log2FoldChange"] < 0)

    status = pd.Series("Non-significant", index=results_df.index, dtype="object")
    status.loc[up] = "Up"
    status.loc[down] = "Down"
    return status


def _plot_volcano(
    results_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
    log2fc_cutoff: float,
    padj_cutoff: float,
) -> None:
    plot_df = results_df.loc[:, ["metabolite_id", "log2FoldChange", "padj_bh", "dem_status"]].copy()
    finite_padj = plot_df.loc[plot_df["padj_bh"].notna() & (plot_df["padj_bh"] > 0), "padj_bh"]
    min_positive_padj = float(finite_padj.min()) if not finite_padj.empty else 1e-300
    plot_df["padj_for_plot"] = plot_df["padj_bh"].fillna(1.0).clip(lower=min_positive_padj)
    plot_df["neg_log10_padj"] = -np.log10(plot_df["padj_for_plot"])

    colors = {
        "Down": "#2B6CB0",
        "Non-significant": "#B8B8B8",
        "Up": "#C53030",
    }
    order = ["Non-significant", "Down", "Up"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for status in order:
        sub = plot_df.loc[plot_df["dem_status"] == status]
        ax.scatter(
            sub["log2FoldChange"],
            sub["neg_log10_padj"],
            s=18,
            c=colors[status],
            alpha=0.78 if status != "Non-significant" else 0.48,
            linewidths=0,
            label=f"{status} (n={len(sub)})",
        )

    ax.axvline(log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.axvline(-log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.axhline(-np.log10(padj_cutoff), color="#606060", linestyle="--", linewidth=0.9)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 adjusted P value")
    ax.set_title(f"DEM Volcano Plot: {comparison}")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_vip(
    results_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
    vip_cutoff: float,
    top_n: int = 30,
) -> None:
    plot_df = (
        results_df.loc[results_df["vip"].notna(), ["metabolite_id", "vip", "dem_status"]]
        .sort_values("vip", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    if plot_df.empty:
        return

    colors = plot_df["dem_status"].map({"Up": "#C53030", "Down": "#2B6CB0"}).fillna("#8A8A8A")
    height = max(4.8, min(12.0, 0.28 * len(plot_df) + 2.0))
    fig, ax = plt.subplots(figsize=(8.0, height))
    ax.barh(plot_df["metabolite_id"], plot_df["vip"], color=colors)
    ax.axvline(vip_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.set_xlabel("OPLS-DA VIP")
    ax.set_ylabel("")
    ax.set_title(f"Top VIP Metabolites: {comparison}")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_scores(
    scores_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
) -> None:
    if scores_df.empty:
        return

    groups = list(dict.fromkeys(scores_df["class_label"].astype(str)))
    colors = ["#C53030", "#2B6CB0", "#2F855A", "#805AD5"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    y_column = "to1" if "to1" in scores_df.columns else "__zero"
    plot_df = scores_df.copy()
    if y_column == "__zero":
        plot_df[y_column] = 0.0

    for idx, group in enumerate(groups):
        sub = plot_df.loc[plot_df["class_label"].astype(str) == group]
        ax.scatter(
            sub["tp1"],
            sub[y_column],
            s=44,
            c=colors[idx % len(colors)],
            alpha=0.85,
            linewidths=0,
            label=f"{group} (n={len(sub)})",
        )

    ax.axvline(0, color="#D0D0D0", linewidth=0.8)
    ax.axhline(0, color="#D0D0D0", linewidth=0.8)
    ax.set_xlabel("Predictive score (tp1)")
    ax.set_ylabel("Orthogonal score (to1)" if y_column == "to1" else "")
    ax.set_title(f"OPLS-DA Scores: {comparison}")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_differential_metabolite_counts(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "comparison",
        "up_count",
        "down_count",
        "significant_count",
        "non_significant_count",
        "total_metabolites",
    ]
    if not all_results:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for df in all_results:
        comparison = str(df["comparison"].iloc[0]) if not df.empty else ""
        up_count = int((df["dem_status"] == "Up").sum())
        down_count = int((df["dem_status"] == "Down").sum())
        non_significant_count = int((df["dem_status"] == "Non-significant").sum())
        rows.append(
            {
                "comparison": comparison,
                "up_count": up_count,
                "down_count": down_count,
                "significant_count": up_count + down_count,
                "non_significant_count": non_significant_count,
                "total_metabolites": int(len(df)),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _plot_differential_metabolite_counts(counts_df: pd.DataFrame, output_path: Path) -> None:
    if counts_df.empty:
        return

    plot_df = counts_df.copy()
    x = np.arange(len(plot_df))
    down_values = -plot_df["down_count"].to_numpy(dtype=float)
    up_values = plot_df["up_count"].to_numpy(dtype=float)

    width = max(8.0, min(18.0, 0.38 * len(plot_df) + 5.0))
    fig, ax = plt.subplots(figsize=(width, 5.4))
    ax.bar(x, up_values, color="#C53030", label="Up")
    ax.bar(x, down_values, color="#2B6CB0", label="Down")
    ax.axhline(0, color="#303030", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["comparison"], rotation=60, ha="right")
    ax.set_ylabel("Differential metabolite count")
    ax.set_title("Differential Metabolite Counts")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.7)

    max_abs = max(float(np.abs(down_values).max()), float(np.abs(up_values).max()), 1.0)
    ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _analyze_contrast(
    metabolites_sample_feature: pd.DataFrame,
    metadata: pd.DataFrame,
    contrast: dict[str, Any],
    compare_field: str,
    vip_cutoff: float,
    padj_cutoff: float,
    log2fc_cutoff: float,
    pseudocount: float,
    max_missing_fraction: float,
    n_orthogonal_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comparison = str(contrast["name"])
    tested_group = str(contrast["tested_group"])
    reference_group = str(contrast["reference_group"])

    tested_samples = metadata.index[metadata["__dem_group"] == tested_group].tolist()
    reference_samples = metadata.index[metadata["__dem_group"] == reference_group].tolist()
    selected_samples = reference_samples + tested_samples
    selected = metabolites_sample_feature.loc[selected_samples].copy()

    missing_fraction = selected.isna().mean(axis=0)
    valid = missing_fraction <= max_missing_fraction
    if not valid.any():
        raise ValueError(f"No metabolites remain for contrast {comparison} after missing-value filtering.")

    selected = selected.loc[:, valid]
    imputed = _median_impute(selected)
    variances = imputed.var(axis=0, ddof=1)
    imputed = imputed.loc[:, variances > 0]
    selected = selected.loc[:, imputed.columns]
    if imputed.shape[1] == 0:
        raise ValueError(f"No variable metabolites remain for contrast {comparison}.")

    y_binary = np.array([0] * len(reference_samples) + [1] * len(tested_samples), dtype=float)
    vip, predictive_score, model_summary, orthogonal_score = _fit_opls_da_vip(
        x=imputed,
        y_binary=y_binary,
        n_orthogonal_components=n_orthogonal_components,
    )

    rows: list[dict[str, Any]] = []
    tested_block = selected.loc[tested_samples]
    reference_block = selected.loc[reference_samples]
    for metabolite_id in selected.columns:
        tested_values = tested_block[metabolite_id].dropna().astype(float)
        reference_values = reference_block[metabolite_id].dropna().astype(float)

        tested_mean = float(tested_values.mean()) if not tested_values.empty else np.nan
        reference_mean = float(reference_values.mean()) if not reference_values.empty else np.nan
        fc_numerator = tested_mean + pseudocount
        fc_denominator = reference_mean + pseudocount
        if np.isfinite(fc_numerator) and np.isfinite(fc_denominator) and fc_numerator > 0 and fc_denominator > 0:
            fold_change = float(fc_numerator / fc_denominator)
            log2_fold_change = float(np.log2(fold_change))
        else:
            fold_change = np.nan
            log2_fold_change = np.nan

        if len(tested_values) >= 2 and len(reference_values) >= 2:
            t_res = stats.ttest_ind(tested_values, reference_values, equal_var=False, nan_policy="omit")
            t_stat = float(t_res.statistic) if np.isfinite(t_res.statistic) else np.nan
            pvalue = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan
        else:
            t_stat = np.nan
            pvalue = np.nan

        rows.append(
            {
                "metabolite_id": str(metabolite_id),
                "tested_mean": tested_mean,
                "reference_mean": reference_mean,
                "fold_change": fold_change,
                "log2FoldChange": log2_fold_change,
                "t_stat": t_stat,
                "pvalue": pvalue,
                "vip": float(vip.loc[metabolite_id]),
                "comparison": comparison,
                "tested_level": str(contrast["tested_level"]),
                "reference_level": str(contrast["reference_level"]),
                "n_tested": int(len(tested_values)),
                "n_reference": int(len(reference_values)),
            }
        )

    results = pd.DataFrame(rows)
    results["padj_bh"] = _benjamini_hochberg(results["pvalue"])
    results["dem_status"] = _classify_points(
        results,
        vip_cutoff=vip_cutoff,
        padj_cutoff=padj_cutoff,
        log2fc_cutoff=log2fc_cutoff,
    )
    results = results.sort_values(
        ["dem_status", "pvalue", "vip", "metabolite_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    score_rows = []
    score_df = pd.DataFrame(index=selected_samples)
    score_df["sample_id"] = selected_samples
    score_df["class_label"] = metadata.loc[selected_samples, compare_field].astype(str).to_numpy()
    score_df["tp1"] = predictive_score.loc[selected_samples].to_numpy(dtype=float)
    if orthogonal_score is not None:
        score_df["to1"] = orthogonal_score.loc[selected_samples].to_numpy(dtype=float)
    score_rows.append(score_df.reset_index(drop=True))
    scores = pd.concat(score_rows, ignore_index=True)

    model_summary.update(
        {
            "comparison": comparison,
            "tested_level": str(contrast["tested_level"]),
            "reference_level": str(contrast["reference_level"]),
            "n_tested_samples": int(len(tested_samples)),
            "n_reference_samples": int(len(reference_samples)),
        }
    )
    return results, scores, model_summary


def _build_union_significant_metabolites(sig_results: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "metabolite_id",
        "n_significant_contrasts",
        "best_padj",
        "best_pvalue",
        "max_vip",
        "max_abs_log2FoldChange",
    ]
    non_empty = [df for df in sig_results if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(non_empty, ignore_index=True)
    combined["_abs_log2FoldChange"] = combined["log2FoldChange"].abs()
    union = (
        combined.groupby("metabolite_id", as_index=False)
        .agg(
            n_significant_contrasts=("comparison", "nunique"),
            best_padj=("padj_bh", "min"),
            best_pvalue=("pvalue", "min"),
            max_vip=("vip", "max"),
            max_abs_log2FoldChange=("_abs_log2FoldChange", "max"),
        )
        .loc[:, columns]
        .sort_values(["best_padj", "metabolite_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return union


def _extract_union_metabolite_matrix(
    metabolites: pd.DataFrame,
    union_metabolites: list[str],
    sample_order: list[str],
) -> pd.DataFrame:
    output_columns = ["metabolite_id"] + sample_order
    if not union_metabolites:
        return pd.DataFrame(columns=output_columns)

    missing = [metabolite for metabolite in union_metabolites if metabolite not in metabolites.index]
    if missing:
        raise ValueError(f"Union metabolites are missing from the input matrix, for example: {missing[:10]}")

    out = metabolites.loc[union_metabolites, sample_order].copy()
    out.index.name = "metabolite_id"
    return out.reset_index().loc[:, output_columns]


def run_pipeline(
    metabs_path: Path,
    metadata_path: Path,
    out_dir: Path,
    same_fields: list[str],
    compare_field: str,
    tested_levels: list[str],
    reference_level: str,
    vip_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    pseudocount: float = 1e-9,
    max_missing_fraction: float = 0.5,
    min_replicates: int = 2,
    n_orthogonal_components: int = 1,
) -> dict[str, Any]:
    """Run differential metabolite analysis and export OmicsPrism-ready metabolites."""
    out_dir = _ensure_output_dir(out_dir)
    volcano_plot_dir = _ensure_plot_dir(out_dir, "volcano")
    vip_plot_dir = _ensure_plot_dir(out_dir, "vip")
    score_plot_dir = _ensure_plot_dir(out_dir, "oplsda_scores")
    count_plot_dir = _ensure_plot_dir(out_dir, "dem_counts")

    metabolites_feature_sample = load_metabolites(metabs_path)
    metadata = load_metadata(metadata_path)
    validate_inputs(
        metabolites=metabolites_feature_sample,
        metadata=metadata,
        same_fields=same_fields,
        compare_field=compare_field,
        tested_levels=tested_levels,
        reference_level=reference_level,
    )

    metabolites_sample_feature, metadata_aligned = align_metabolites_and_metadata(
        metabolites=metabolites_feature_sample,
        metadata=metadata,
    )
    metadata_aligned = build_dem_group(
        metadata=metadata_aligned,
        same_fields=same_fields,
        compare_field=compare_field,
    )
    contrasts = build_contrasts(
        metadata=metadata_aligned,
        same_fields=same_fields,
        compare_field=compare_field,
        tested_levels=tested_levels,
        reference_level=reference_level,
        min_replicates=min_replicates,
    )

    sample_order = list(metabolites_sample_feature.index)
    aligned_feature_sample = metabolites_sample_feature.T
    aligned_feature_sample.index.name = "metabolite_id"

    sig_results: list[pd.DataFrame] = []
    all_contrast_results: list[pd.DataFrame] = []
    for contrast in contrasts:
        contrast_name = str(contrast["name"])
        all_results, scores, _model_summary = _analyze_contrast(
            metabolites_sample_feature=metabolites_sample_feature,
            metadata=metadata_aligned,
            contrast=contrast,
            compare_field=compare_field,
            vip_cutoff=vip_cutoff,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            pseudocount=pseudocount,
            max_missing_fraction=max_missing_fraction,
            n_orthogonal_components=n_orthogonal_components,
        )
        all_results.to_csv(out_dir / f"{contrast_name}.all.csv", index=False)
        scores.to_csv(out_dir / f"{contrast_name}.oplsda_scores.csv", index=False)
        all_contrast_results.append(all_results)

        sig = all_results.loc[all_results["dem_status"].isin(["Up", "Down"])].copy()
        sig = sig.sort_values(["pvalue", "vip", "metabolite_id"], ascending=[True, False, True])
        sig.to_csv(out_dir / f"{contrast_name}.sig.csv", index=False)
        sig_results.append(sig)

        _plot_volcano(
            results_df=all_results,
            comparison=contrast_name,
            output_path=volcano_plot_dir / f"{contrast_name}.volcano.png",
            log2fc_cutoff=log2fc_cutoff,
            padj_cutoff=padj_cutoff,
        )
        _plot_vip(
            results_df=all_results,
            comparison=contrast_name,
            output_path=vip_plot_dir / f"{contrast_name}.vip.png",
            vip_cutoff=vip_cutoff,
        )
        _plot_scores(
            scores_df=scores,
            comparison=contrast_name,
            output_path=score_plot_dir / f"{contrast_name}.oplsda_scores.png",
        )

    dem_counts = _build_differential_metabolite_counts(all_contrast_results)
    dem_counts_path = out_dir / "differential_metabolite_counts.csv"
    dem_counts.to_csv(dem_counts_path, index=False)
    _plot_differential_metabolite_counts(
        counts_df=dem_counts,
        output_path=count_plot_dir / "differential_metabolite_counts.bar.png",
    )

    union = _build_union_significant_metabolites(sig_results)
    union_path = out_dir / "union_significant_metabolites.csv"
    union.to_csv(union_path, index=False)

    union_metabolites = union["metabolite_id"].astype(str).tolist() if not union.empty else []
    union_matrix = _extract_union_metabolite_matrix(
        metabolites=aligned_feature_sample,
        union_metabolites=union_metabolites,
        sample_order=sample_order,
    )
    union_matrix_path = out_dir / "union_significant_metabolites.matrix.csv"
    union_matrix.to_csv(union_matrix_path, index=False)

    return {
        "out_dir": str(out_dir),
        "n_contrasts": len(contrasts),
        "n_union_significant_metabolites": int(union.shape[0]),
        "union_significant_metabolites": str(union_path),
        "union_significant_metabolites_matrix": str(union_matrix_path),
    }
