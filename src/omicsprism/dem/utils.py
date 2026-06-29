from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omicsprism.deg.utils import sanitize_value


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def load_metabolites(metabs_path: Path) -> pd.DataFrame:
    """Load a metabolite abundance matrix as metabolites x samples."""
    metabs_path = Path(metabs_path)

    if not metabs_path.exists():
        raise FileNotFoundError(f"Metabolite file does not exist: {metabs_path}")
    if not metabs_path.is_file():
        raise ValueError(f"Metabolite path is not a file: {metabs_path}")

    try:
        raw = pd.read_csv(metabs_path, dtype=str)
    except Exception as exc:
        raise ValueError(f"Failed to read metabolite CSV: {metabs_path}") from exc

    if raw.empty:
        raise ValueError("Metabolite file is empty.")
    if raw.shape[1] < 2:
        raise ValueError("Metabolite file must contain one metabolite ID column and at least one sample column.")

    metabolite_col = raw.columns[0]
    sample_cols = [str(col) for col in raw.columns[1:]]
    if any(col.strip() == "" for col in sample_cols):
        raise ValueError("Metabolite file contains an empty sample column name.")
    if len(sample_cols) != len(set(sample_cols)):
        duplicated = sorted({col for col in sample_cols if sample_cols.count(col) > 1})
        raise ValueError(f"Metabolite file contains duplicated sample columns, for example: {duplicated[:10]}")

    metabolite_ids = raw[metabolite_col].astype(str)
    if metabolite_ids.isna().any() or (metabolite_ids.str.strip() == "").any():
        raise ValueError("Metabolite file contains empty metabolite IDs.")
    if metabolite_ids.duplicated().any():
        duplicates = metabolite_ids[metabolite_ids.duplicated()].unique()[:10]
        raise ValueError(f"Metabolite file contains duplicated metabolite IDs, for example: {list(duplicates)}")

    values = raw.loc[:, raw.columns[1:]].apply(pd.to_numeric, errors="coerce")
    values.columns = sample_cols
    matrix = values.to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("Metabolite file contains infinite values.")

    values.index = metabolite_ids
    values.index.name = "metabolite_id"
    return values


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
    metabolites: pd.DataFrame,
    metadata: pd.DataFrame,
    same_fields: list[str],
    compare_field: str,
    tested_levels: list[str],
    reference_level: str,
) -> None:
    """Validate matrix shape, metadata fields, levels, and sample overlap."""
    if metabolites.empty:
        raise ValueError("Metabolite matrix is empty.")
    if metadata.empty:
        raise ValueError("Metadata table is empty.")
    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata must contain sample_id.")

    metabolite_samples = set(map(str, metabolites.columns))
    metadata_samples = set(map(str, metadata["sample_id"]))
    matched_samples = metabolite_samples.intersection(metadata_samples)
    if not matched_samples:
        raise ValueError("No matching samples between metabolite columns and metadata.sample_id.")

    if matched_samples != metabolite_samples or matched_samples != metadata_samples:
        missing_in_metadata = sorted(metabolite_samples - metadata_samples)
        missing_in_metabolites = sorted(metadata_samples - metabolite_samples)
        message_parts = ["Metabolites and metadata only partially match. The sample intersection will be used."]
        if missing_in_metadata:
            message_parts.append(f"Samples present in metabolites but missing from metadata: {missing_in_metadata[:10]}")
        if missing_in_metabolites:
            message_parts.append(f"Samples present in metadata but missing from metabolites: {missing_in_metabolites[:10]}")
        _warn(" ".join(message_parts))

    if compare_field not in metadata.columns:
        raise ValueError(f"--compare-field does not exist in metadata: {compare_field}")

    if compare_field in same_fields:
        raise ValueError(
            "--same-fields must not include --compare-field. "
            "Use --same-fields only for blocking variables that should be matched within each contrast."
        )

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


def align_metabolites_and_metadata(
    metabolites: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample x metabolite abundances and sample-aligned metadata."""
    sample_order = [str(sample) for sample in metabolites.columns]
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata_by_sample = metadata.set_index("sample_id", drop=False)

    matched_samples = [sample for sample in sample_order if sample in metadata_by_sample.index]
    if not matched_samples:
        raise ValueError("No matching samples between metabolite columns and metadata.sample_id.")

    metabolites_aligned = metabolites.loc[:, matched_samples].T
    metabolites_aligned.index.name = "sample_id"

    metadata_aligned = metadata_by_sample.loc[matched_samples].copy()
    metadata_aligned.index.name = "sample_id"
    return metabolites_aligned, metadata_aligned


def build_dem_group(
    metadata: pd.DataFrame,
    same_fields: list[str],
    compare_field: str,
) -> pd.DataFrame:
    """Create the sanitized __dem_group column used for pairwise contrasts."""
    metadata = metadata.copy()
    group_values: list[str] = []
    sanitized_to_raw: dict[str, tuple[str, ...]] = {}

    for _, row in metadata.iterrows():
        raw_parts = tuple(str(row[field]) for field in same_fields + [compare_field])
        sanitized_parts = [sanitize_value(row[field]) for field in same_fields]
        sanitized_parts.append(sanitize_value(row[compare_field]))
        sanitized = "__".join(sanitized_parts)

        if sanitized in sanitized_to_raw and sanitized_to_raw[sanitized] != raw_parts:
            raise ValueError(
                "Sanitized metadata values collide. "
                f"Both {sanitized_to_raw[sanitized]} and {raw_parts} map to '{sanitized}'. "
                "Please rename metadata values to avoid ambiguity."
            )

        sanitized_to_raw[sanitized] = raw_parts
        group_values.append(sanitized)

    metadata["__dem_group"] = group_values
    if (metadata["__dem_group"].astype(str).str.strip() == "").any():
        raise ValueError("Internal __dem_group contains empty values after sanitization.")
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
    """Build all valid tested-vs-reference contrasts from metadata."""
    if "__dem_group" not in metadata.columns:
        raise ValueError("metadata must contain __dem_group. Run build_dem_group first.")

    contrasts: list[dict[str, Any]] = []
    dem_groups = set(metadata["__dem_group"])
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

            if tested_group not in dem_groups:
                raise ValueError(f"Internal error: tested group not found in __dem_group: {tested_group}")
            if reference_group not in dem_groups:
                raise ValueError(f"Internal error: reference group not found in __dem_group: {reference_group}")

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
