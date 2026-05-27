from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omicsprism.differential_plots import plot_differential_sankey, plot_differential_upset

from .utils import (
    align_counts_and_metadata,
    build_contrasts,
    build_de_group,
    filter_low_count_genes,
    load_counts,
    load_metadata,
    validate_inputs,
)


def _ensure_output_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_volcano_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots" / "volcano"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _ensure_ma_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots" / "ma"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _ensure_deg_count_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots" / "deg_counts"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _ensure_upset_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots" / "upset"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _ensure_sankey_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots" / "sankey"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _standardize_results_df(results_df: pd.DataFrame, comparison: str) -> pd.DataFrame:
    required_columns = [
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
    ]
    missing = [col for col in required_columns if col not in results_df.columns]
    if missing:
        raise ValueError(f"PyDESeq2 results_df is missing expected columns: {missing}")

    out = results_df.loc[:, required_columns].copy()
    out.index.name = "gene_id"
    out = out.reset_index()
    out["comparison"] = comparison
    return out[
        [
            "gene_id",
            "baseMean",
            "log2FoldChange",
            "lfcSE",
            "stat",
            "pvalue",
            "padj",
            "comparison",
        ]
    ]


def _classify_volcano_points(
    results_df: pd.DataFrame,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> pd.Series:
    significant = results_df["padj"].notna() & (results_df["padj"] < padj_cutoff)
    up = significant & results_df["log2FoldChange"].notna() & (results_df["log2FoldChange"] >= log2fc_cutoff)
    down = significant & results_df["log2FoldChange"].notna() & (results_df["log2FoldChange"] <= -log2fc_cutoff)

    status = pd.Series("Non-significant", index=results_df.index, dtype="object")
    status.loc[up] = "Up"
    status.loc[down] = "Down"
    return status


def _plot_volcano(
    results_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> None:
    plot_df = results_df.loc[:, ["gene_id", "log2FoldChange", "padj"]].copy()
    plot_df["status"] = _classify_volcano_points(
        plot_df,
        padj_cutoff=padj_cutoff,
        log2fc_cutoff=log2fc_cutoff,
    )

    finite_padj = plot_df.loc[plot_df["padj"].notna() & (plot_df["padj"] > 0), "padj"]
    min_positive_padj = float(finite_padj.min()) if not finite_padj.empty else 1e-300
    plot_df["padj_for_plot"] = plot_df["padj"].fillna(1.0).clip(lower=min_positive_padj)
    plot_df["neg_log10_padj"] = -np.log10(plot_df["padj_for_plot"])

    colors = {
        "Down": "#2B6CB0",
        "Non-significant": "#B8B8B8",
        "Up": "#C53030",
    }
    order = ["Non-significant", "Down", "Up"]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for status in order:
        sub = plot_df.loc[plot_df["status"] == status]
        ax.scatter(
            sub["log2FoldChange"],
            sub["neg_log10_padj"],
            s=10,
            c=colors[status],
            alpha=0.75 if status != "Non-significant" else 0.45,
            linewidths=0,
            label=f"{status} (n={len(sub)})",
        )

    ax.axvline(log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.axvline(-log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.axhline(-np.log10(padj_cutoff), color="#606060", linestyle="--", linewidth=0.9)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 adjusted P value")
    ax.set_title(f"Volcano Plot: {comparison}")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_ma(
    results_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> None:
    plot_df = results_df.loc[:, ["gene_id", "baseMean", "log2FoldChange", "padj"]].copy()
    plot_df["status"] = _classify_volcano_points(
        plot_df,
        padj_cutoff=padj_cutoff,
        log2fc_cutoff=log2fc_cutoff,
    )

    plot_df["baseMean_for_plot"] = np.log2(plot_df["baseMean"].fillna(0.0).clip(lower=0.0) + 1.0)

    colors = {
        "Down": "#2B6CB0",
        "Non-significant": "#B8B8B8",
        "Up": "#C53030",
    }
    order = ["Non-significant", "Down", "Up"]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for status in order:
        sub = plot_df.loc[plot_df["status"] == status]
        ax.scatter(
            sub["baseMean_for_plot"],
            sub["log2FoldChange"],
            s=10,
            c=colors[status],
            alpha=0.75 if status != "Non-significant" else 0.45,
            linewidths=0,
            label=f"{status} (n={len(sub)})",
        )

    ax.axhline(0, color="#404040", linestyle="-", linewidth=0.9)
    ax.axhline(log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.axhline(-log2fc_cutoff, color="#606060", linestyle="--", linewidth=0.9)
    ax.set_xlabel("log2 mean normalized count (baseMean + 1)")
    ax.set_ylabel("log2 Fold Change")
    ax.set_title(f"MA Plot: {comparison}")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_differential_gene_counts(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "comparison",
        "up_count",
        "down_count",
        "significant_count",
        "non_significant_count",
        "total_genes",
    ]
    if not all_results:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for df in all_results:
        comparison = str(df["comparison"].iloc[0]) if not df.empty else ""
        up_count = int((df["volcano_status"] == "Up").sum())
        down_count = int((df["volcano_status"] == "Down").sum())
        non_significant_count = int((df["volcano_status"] == "Non-significant").sum())
        rows.append(
            {
                "comparison": comparison,
                "up_count": up_count,
                "down_count": down_count,
                "significant_count": up_count + down_count,
                "non_significant_count": non_significant_count,
                "total_genes": int(len(df)),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _plot_differential_gene_counts(counts_df: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_ylabel("Differential gene count")
    ax.set_title("Differential Gene Counts")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.7)

    max_abs = max(float(np.abs(down_values).max()), float(np.abs(up_values).max()), 1.0)
    ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _filter_significant_genes(
    results_df: pd.DataFrame,
    comparison: str,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> pd.DataFrame:
    standardized = _standardize_results_df(results_df, comparison=comparison)
    mask = (
        standardized["padj"].notna()
        & (standardized["padj"] < padj_cutoff)
        & standardized["log2FoldChange"].notna()
        & (standardized["log2FoldChange"].abs() >= log2fc_cutoff)
    )
    sig = standardized.loc[mask].copy()
    return sig.sort_values(["padj", "gene_id"], ascending=[True, True])


def _build_union_significant_genes(sig_results: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "gene_id",
        "n_significant_contrasts",
        "best_padj",
        "max_abs_log2FoldChange",
    ]
    non_empty = [df for df in sig_results if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(non_empty, ignore_index=True)
    combined["_abs_log2FoldChange"] = combined["log2FoldChange"].abs()
    union = (
        combined.groupby("gene_id", as_index=False)
        .agg(
            n_significant_contrasts=("comparison", "nunique"),
            best_padj=("padj", "min"),
            max_abs_log2FoldChange=("_abs_log2FoldChange", "max"),
        )
        .loc[:, columns]
        .sort_values(["best_padj", "gene_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return union


def _extract_vst_matrix(
    dds: Any,
    union_genes: list[str],
    sample_order: list[str],
) -> pd.DataFrame:
    """Extract VST values for union significant genes as genes x samples."""
    output_columns = ["gene_id"] + sample_order
    if not union_genes:
        return pd.DataFrame(columns=output_columns)

    if "vst_counts" not in dds.layers:
        raise ValueError("dds.layers['vst_counts'] was not found. Did dds.vst() finish successfully?")

    vst_sample_gene = pd.DataFrame(
        np.asarray(dds.layers["vst_counts"]),
        index=list(dds.obs_names),
        columns=list(dds.var_names),
    )

    missing_genes = [gene for gene in union_genes if gene not in vst_sample_gene.columns]
    if missing_genes:
        raise ValueError(
            "Some union significant genes are missing from the VST matrix, for example: "
            f"{missing_genes[:10]}"
        )

    missing_samples = [sample for sample in sample_order if sample not in vst_sample_gene.index]
    if missing_samples:
        raise ValueError(
            "Some expected samples are missing from the VST matrix, for example: "
            f"{missing_samples[:10]}"
        )

    vst_gene_sample = vst_sample_gene.loc[sample_order, union_genes].T
    vst_gene_sample.index.name = "gene_id"
    out = vst_gene_sample.reset_index()
    return out.loc[:, output_columns]


def run_pipeline(
    counts_path: Path,
    metadata_path: Path,
    out_dir: Path,
    same_fields: list[str],
    compare_field: str,
    tested_levels: list[str],
    reference_level: str,
    padj_cutoff: float = 0.05,
    log2fc_cutoff: float = 1.0,
    min_total_count: int = 10,
    min_replicates: int = 2,
    n_cpus: int = 8,
) -> dict[str, Any]:
    """Run differential expression analysis and export OmicsPrism-ready VST genes."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "The DEG module requires PyDESeq2. Install it with: "
            "python -m pip install -e .[deg]"
        ) from exc

    out_dir = _ensure_output_dir(out_dir)
    volcano_plot_dir = _ensure_volcano_plot_dir(out_dir)
    ma_plot_dir = _ensure_ma_plot_dir(out_dir)
    deg_count_plot_dir = _ensure_deg_count_plot_dir(out_dir)
    upset_plot_dir = _ensure_upset_plot_dir(out_dir)
    sankey_plot_dir = _ensure_sankey_plot_dir(out_dir)

    counts_gene_sample = load_counts(counts_path)
    metadata = load_metadata(metadata_path)
    validate_inputs(
        counts=counts_gene_sample,
        metadata=metadata,
        same_fields=same_fields,
        compare_field=compare_field,
        tested_levels=tested_levels,
        reference_level=reference_level,
    )

    counts_sample_gene, metadata_aligned = align_counts_and_metadata(
        counts=counts_gene_sample,
        metadata=metadata,
    )
    counts_sample_gene = filter_low_count_genes(
        counts_sample_gene=counts_sample_gene,
        min_total_count=min_total_count,
    )
    metadata_aligned = build_de_group(
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

    metadata_for_dds = metadata_aligned.copy()
    metadata_for_dds["__de_group"] = metadata_for_dds["__de_group"].astype(str)
    sample_order = list(counts_sample_gene.index)

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts_sample_gene,
        metadata=metadata_for_dds,
        design="~ __de_group",
        refit_cooks=True,
        inference=inference,
        quiet=False,
    )

    dds.deseq2()
    dds.vst()

    sig_results: list[pd.DataFrame] = []
    all_contrast_results: list[pd.DataFrame] = []
    for contrast in contrasts:
        contrast_name = str(contrast["name"])
        stat_res = DeseqStats(
            dds,
            contrast=[
                "__de_group",
                str(contrast["tested_group"]),
                str(contrast["reference_group"]),
            ],
            alpha=padj_cutoff,
            inference=inference,
            n_cpus=n_cpus,
            quiet=True,
        )
        stat_res.summary()

        all_results = _standardize_results_df(
            results_df=stat_res.results_df,
            comparison=contrast_name,
        )
        all_results["volcano_status"] = _classify_volcano_points(
            all_results,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
        )
        all_results.to_csv(out_dir / f"{contrast_name}.all.csv", index=False)
        all_contrast_results.append(all_results)
        _plot_volcano(
            results_df=all_results,
            comparison=contrast_name,
            output_path=volcano_plot_dir / f"{contrast_name}.volcano.png",
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
        )
        _plot_ma(
            results_df=all_results,
            comparison=contrast_name,
            output_path=ma_plot_dir / f"{contrast_name}.ma.png",
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
        )

        sig = _filter_significant_genes(
            results_df=stat_res.results_df,
            comparison=contrast_name,
            padj_cutoff=padj_cutoff,
            log2fc_cutoff=log2fc_cutoff,
        )
        sig.to_csv(out_dir / f"{contrast_name}.sig.csv", index=False)
        sig_results.append(sig)

    deg_counts = _build_differential_gene_counts(all_contrast_results)
    deg_counts_path = out_dir / "differential_gene_counts.csv"
    deg_counts.to_csv(deg_counts_path, index=False)
    _plot_differential_gene_counts(
        counts_df=deg_counts,
        output_path=deg_count_plot_dir / "differential_gene_counts.bar.png",
    )
    plot_differential_sankey(
        deg_counts,
        contrasts,
        same_fields=same_fields,
        same_field_orders={
            field: metadata_aligned[field].astype(str).drop_duplicates().tolist()
            for field in same_fields
        },
        tested_level_order=tested_levels,
        tested_level_count=len(tested_levels),
        title="Differential Gene Count Flow",
        output_html=sankey_plot_dir / "differential_sankey.html",
        output_png=sankey_plot_dir / "differential_sankey.png",
    )

    union = _build_union_significant_genes(sig_results)
    union_path = out_dir / "union_significant_genes.csv"
    union.to_csv(union_path, index=False)
    plot_differential_upset(
        sig_results,
        feature_col="gene_id",
        title="Differential Gene Overlap",
        unit_label="Genes",
        output_path=upset_plot_dir / "differential_gene_upset.png",
    )

    union_genes = union["gene_id"].astype(str).tolist() if not union.empty else []
    union_vst = _extract_vst_matrix(
        dds=dds,
        union_genes=union_genes,
        sample_order=sample_order,
    )
    union_vst_path = out_dir / "union_significant_genes.vst.csv"
    union_vst.to_csv(union_vst_path, index=False)

    return {
        "out_dir": str(out_dir),
        "n_contrasts": len(contrasts),
        "n_union_significant_genes": int(union.shape[0]),
        "union_significant_genes": str(union_path),
        "union_significant_genes_vst": str(union_vst_path),
    }
