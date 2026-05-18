from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_csv_arg(value: str | None) -> list[str]:
    """Parse comma-separated CLI values while ignoring empty items."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def sanitize_value(value: Any) -> str:
    """Sanitize metadata values for model labels and filenames."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "NA"


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def load_counts(counts_path: Path) -> pd.DataFrame:
    """Load a raw count matrix as genes x samples."""
    counts_path = Path(counts_path)

    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file does not exist: {counts_path}")
    if not counts_path.is_file():
        raise ValueError(f"Counts path is not a file: {counts_path}")

    try:
        raw = pd.read_csv(counts_path, dtype=str)
    except Exception as exc:
        raise ValueError(f"Failed to read counts CSV: {counts_path}") from exc

    if raw.empty:
        raise ValueError("Counts file is empty.")
    if raw.shape[1] < 2:
        raise ValueError("Counts file must contain one gene ID column and at least one sample column.")

    gene_col = raw.columns[0]
    sample_cols = list(raw.columns[1:])

    if any(str(col).strip() == "" for col in sample_cols):
        raise ValueError("Counts file contains an empty sample column name.")

    gene_ids = raw[gene_col].astype(str)
    if gene_ids.isna().any() or (gene_ids.str.strip() == "").any():
        raise ValueError("Counts file contains empty gene IDs.")
    if gene_ids.duplicated().any():
        duplicates = gene_ids[gene_ids.duplicated()].unique()[:10]
        raise ValueError(f"Counts file contains duplicated gene IDs, for example: {list(duplicates)}")

    counts_values = raw.loc[:, sample_cols].copy()
    if counts_values.isna().any().any():
        raise ValueError("Counts file contains missing count values.")

    try:
        counts_numeric = counts_values.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError("Counts file contains non-numeric count values.") from exc

    values = counts_numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Counts file contains non-finite values.")
    if (values < 0).any():
        raise ValueError("Counts file contains negative values. Raw counts must be non-negative integers.")
    if not np.all(np.equal(values, np.floor(values))):
        raise ValueError("Counts file contains non-integer values. Raw counts must be non-negative integers.")

    counts_numeric.index = gene_ids
    counts_numeric.index.name = "gene_id"
    counts_numeric.columns = [str(col) for col in sample_cols]
    return counts_numeric.astype(np.int64)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata with a required sample_id column."""
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    if not metadata_path.is_file():
        raise ValueError(f"Metadata path is not a file: {metadata_path}")

    try:
        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Failed to read metadata CSV: {metadata_path}") from exc

    if metadata.empty:
        raise ValueError("Metadata file is empty.")
    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata file must contain a column named sample_id.")

    metadata["sample_id"] = metadata["sample_id"].astype(str)
    if (metadata["sample_id"].str.strip() == "").any():
        raise ValueError("Metadata file contains empty sample_id values.")
    if metadata["sample_id"].duplicated().any():
        duplicates = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].unique()[:10]
        raise ValueError(f"Metadata file contains duplicated sample_id values, for example: {list(duplicates)}")

    return metadata


def validate_inputs(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    same_fields: list[str],
    compare_field: str,
    tested_levels: list[str],
    reference_level: str,
) -> None:
    """Validate matrix shape, metadata fields, levels, and sample overlap."""
    if counts.empty:
        raise ValueError("Counts matrix is empty.")
    if metadata.empty:
        raise ValueError("Metadata table is empty.")
    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata must contain sample_id.")

    counts_samples = set(map(str, counts.columns))
    metadata_samples = set(map(str, metadata["sample_id"]))
    matched_samples = counts_samples.intersection(metadata_samples)

    if not matched_samples:
        raise ValueError("No matching samples between counts columns and metadata.sample_id.")

    if matched_samples != counts_samples or matched_samples != metadata_samples:
        missing_in_metadata = sorted(counts_samples - metadata_samples)
        missing_in_counts = sorted(metadata_samples - counts_samples)
        message_parts = ["Counts and metadata only partially match. The sample intersection will be used."]
        if missing_in_metadata:
            message_parts.append(f"Samples present in counts but missing from metadata: {missing_in_metadata[:10]}")
        if missing_in_counts:
            message_parts.append(f"Samples present in metadata but missing from counts: {missing_in_counts[:10]}")
        _warn(" ".join(message_parts))

    if compare_field not in metadata.columns:
        raise ValueError(f"--compare-field does not exist in metadata: {compare_field}")

    missing_same_fields = [field for field in same_fields if field not in metadata.columns]
    if missing_same_fields:
        raise ValueError(f"--same-fields contains fields missing from metadata: {missing_same_fields}")

    if not tested_levels:
        raise ValueError("--tested-levels must contain at least one level.")

    observed_levels = set(metadata[compare_field].astype(str))
    for level in [level for level in tested_levels if level not in observed_levels]:
        _warn(
            f"Tested level '{level}' does not appear in metadata column '{compare_field}'. "
            "No contrast will be generated for this level unless it appears after sample matching."
        )

    if reference_level not in observed_levels:
        raise ValueError(
            f"--reference-level '{reference_level}' does not appear in metadata column '{compare_field}'."
        )


def align_counts_and_metadata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample x gene counts and sample-aligned metadata."""
    counts_sample_order = [str(sample) for sample in counts.columns]
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata_by_sample = metadata.set_index("sample_id", drop=False)

    matched_samples = [sample for sample in counts_sample_order if sample in metadata_by_sample.index]
    if not matched_samples:
        raise ValueError("No matching samples between counts columns and metadata.sample_id.")

    counts_aligned = counts.loc[:, matched_samples].T
    counts_aligned.index.name = "sample_id"

    metadata_aligned = metadata_by_sample.loc[matched_samples].copy()
    metadata_aligned.index.name = "sample_id"
    return counts_aligned, metadata_aligned


def filter_low_count_genes(
    counts_sample_gene: pd.DataFrame,
    min_total_count: int,
) -> pd.DataFrame:
    """Filter genes by total count across retained samples."""
    keep_genes = counts_sample_gene.sum(axis=0) >= min_total_count
    filtered = counts_sample_gene.loc[:, keep_genes].copy()
    if filtered.shape[1] == 0:
        raise ValueError(f"No genes remain after filtering by --min-total-count >= {min_total_count}.")
    return filtered


def _group_parts_for_row(
    row: pd.Series,
    same_fields: list[str],
    compare_field: str,
) -> list[str]:
    parts = [sanitize_value(row[field]) for field in same_fields]
    parts.append(sanitize_value(row[compare_field]))
    return parts


def build_de_group(
    metadata: pd.DataFrame,
    same_fields: list[str],
    compare_field: str,
) -> pd.DataFrame:
    """Create the sanitized __de_group column used by PyDESeq2."""
    metadata = metadata.copy()
    group_values: list[str] = []
    sanitized_to_raw: dict[str, tuple[str, ...]] = {}

    for _, row in metadata.iterrows():
        raw_parts = tuple(str(row[field]) for field in same_fields + [compare_field])
        sanitized = "__".join(_group_parts_for_row(row, same_fields, compare_field))

        if sanitized in sanitized_to_raw and sanitized_to_raw[sanitized] != raw_parts:
            raise ValueError(
                "Sanitized metadata values collide. "
                f"Both {sanitized_to_raw[sanitized]} and {raw_parts} map to '{sanitized}'. "
                "Please rename metadata values to avoid ambiguity."
            )

        sanitized_to_raw[sanitized] = raw_parts
        group_values.append(sanitized)

    metadata["__de_group"] = group_values
    if (metadata["__de_group"].astype(str).str.strip() == "").any():
        raise ValueError("Internal __de_group contains empty values after sanitization.")
    return metadata


def _format_group_value(
    same_values: tuple[Any, ...],
    tested_or_reference_level: str,
) -> str:
    parts = [sanitize_value(value) for value in same_values]
    parts.append(sanitize_value(tested_or_reference_level))
    return "__".join(parts)


def _format_contrast_name(
    same_values: tuple[Any, ...],
    tested_level: str,
    reference_level: str,
) -> str:
    parts = [sanitize_value(value) for value in same_values]
    parts.append(sanitize_value(tested_level))
    parts.append("vs")
    parts.append(sanitize_value(reference_level))
    return "_".join(parts)


def build_contrasts(
    metadata: pd.DataFrame,
    same_fields: list[str],
    compare_field: str,
    tested_levels: list[str],
    reference_level: str,
    min_replicates: int,
) -> list[dict[str, Any]]:
    """Build all valid contrasts from metadata."""
    if "__de_group" not in metadata.columns:
        raise ValueError("metadata must contain __de_group. Run build_de_group first.")

    contrasts: list[dict[str, Any]] = []
    de_groups = set(metadata["__de_group"])
    iterable = metadata.groupby(same_fields, dropna=False, sort=True) if same_fields else [((), metadata)]

    for same_key, sub_metadata in iterable:
        if same_fields:
            same_values = same_key if isinstance(same_key, tuple) else (same_key,)
        else:
            same_values = ()

        for tested_level in tested_levels:
            n_tested = int((sub_metadata[compare_field] == tested_level).sum())
            n_reference = int((sub_metadata[compare_field] == reference_level).sum())

            if n_tested == 0 or n_reference == 0:
                continue

            if n_tested < min_replicates or n_reference < min_replicates:
                _warn(
                    "Skipping contrast because one group has fewer samples than "
                    f"--min-replicates={min_replicates}: same_values={same_values}, "
                    f"tested={tested_level} n={n_tested}, reference={reference_level} n={n_reference}"
                )
                continue

            tested_group = _format_group_value(same_values, tested_level)
            reference_group = _format_group_value(same_values, reference_level)
            contrast_name = _format_contrast_name(same_values, tested_level, reference_level)

            if tested_group not in de_groups:
                raise ValueError(f"Internal error: tested group not found in __de_group: {tested_group}")
            if reference_group not in de_groups:
                raise ValueError(f"Internal error: reference group not found in __de_group: {reference_group}")

            contrasts.append(
                {
                    "name": contrast_name,
                    "tested_group": tested_group,
                    "reference_group": reference_group,
                    "tested_level": tested_level,
                    "reference_level": reference_level,
                    "same_values": same_values,
                    "n_tested": n_tested,
                    "n_reference": n_reference,
                }
            )

    if not contrasts:
        raise ValueError("No valid contrasts were generated.")

    names = [contrast["name"] for contrast in contrasts]
    duplicated_names = sorted({name for name in names if names.count(name) > 1})
    if duplicated_names:
        raise ValueError(
            "Sanitized contrast names are not unique. "
            f"Duplicated names: {duplicated_names}. "
            "Please rename metadata values to avoid filename collisions."
        )

    return contrasts
