"""Validation and loading for experiment-informed weekly response data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "week_start",
    "cell_id",
    "category",
    "channel",
    "spend",
    "context_index",
    "measured_incremental_contribution",
    "evidence_type",
    "evidence_reference",
)

ALLOWED_EVIDENCE_TYPES = {
    "randomized_experiment",
    "geo_experiment",
    "causal_estimate",
    "synthetic_fixture",
}
DECISION_ELIGIBLE_EVIDENCE_TYPES = ALLOWED_EVIDENCE_TYPES - {"synthetic_fixture"}
NUMERIC_COLUMNS = (
    "spend",
    "context_index",
    "measured_incremental_contribution",
)
TEXT_COLUMNS = (
    "cell_id",
    "category",
    "channel",
    "evidence_type",
    "evidence_reference",
)


def validate_weekly_response(
    frame: pd.DataFrame,
    *,
    validation_weeks: int,
    test_weeks: int,
) -> pd.DataFrame:
    """Validate the input contract required by curve fitting and temporal evaluation."""

    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Weekly response data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Weekly response data is empty")
    if validation_weeks < 1 or test_weeks < 1:
        raise ValueError("validation_weeks and test_weeks must be positive")

    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    clean["week_start"] = pd.to_datetime(clean["week_start"], errors="coerce")
    if clean["week_start"].isna().any():
        raise ValueError("week_start must contain valid dates")
    if not clean["week_start"].dt.normalize().eq(clean["week_start"]).all():
        raise ValueError("week_start must contain dates without time-of-day values")

    for column in TEXT_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip()
        if clean[column].isna().any() or clean[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty values")

    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any() or not np.isfinite(clean[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")

    if clean["spend"].le(0).any():
        raise ValueError("spend must be strictly positive")
    if clean["context_index"].le(0).any():
        raise ValueError("context_index must be strictly positive")
    if clean["measured_incremental_contribution"].lt(0).any():
        raise ValueError("measured_incremental_contribution must be non-negative")

    unsupported = sorted(set(clean["evidence_type"]) - ALLOWED_EVIDENCE_TYPES)
    if unsupported:
        raise ValueError(f"Unsupported evidence_type values: {unsupported}")

    if clean.duplicated(["week_start", "cell_id"]).any():
        raise ValueError("Weekly response data must contain one row per week_start and cell_id")

    cell_mapping = clean.groupby("cell_id")[["category", "channel"]].nunique()
    if cell_mapping.gt(1).any(axis=None):
        raise ValueError("Each cell_id must map to one stable category and channel")
    pair_mapping = clean[["cell_id", "category", "channel"]].drop_duplicates()
    if pair_mapping.duplicated(["category", "channel"]).any():
        raise ValueError("Each category-channel pair must map to one cell_id")

    weeks = pd.Index(sorted(clean["week_start"].unique()))
    if len(weeks) < 2:
        raise ValueError("Weekly response data must contain multiple weeks")
    gaps = pd.Series(weeks[1:] - weeks[:-1])
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("week_start values must form a contiguous seven-day calendar")

    cell_count = clean["cell_id"].nunique()
    if not clean.groupby("week_start")["cell_id"].nunique().eq(cell_count).all():
        raise ValueError("Every week must contain every decision cell")
    if not clean.groupby("cell_id")["week_start"].nunique().eq(len(weeks)).all():
        raise ValueError("Every decision cell must contain every week")

    if len(weeks) <= test_weeks + 24:
        raise ValueError(
            "Weekly response data needs more than test_weeks + 24 complete weeks"
        )
    history_weeks = len(weeks) - test_weeks
    if history_weeks <= validation_weeks + 8:
        raise ValueError(
            "Pre-test history needs more than validation_weeks + 8 complete weeks"
        )

    test_start = pd.Timestamp(weeks[-test_weeks])
    validation_start = pd.Timestamp(weeks[-(test_weeks + validation_weeks)])
    calibration = clean.loc[clean["week_start"] < validation_start]
    if calibration.groupby("cell_id")["spend"].nunique().lt(3).any():
        raise ValueError("Every decision cell needs at least three pre-validation spend levels")
    for split_name, split in {
        "validation": clean.loc[
            clean["week_start"].between(
                validation_start, test_start, inclusive="left"
            )
        ],
        "test": clean.loc[clean["week_start"] >= test_start],
    }.items():
        if split.groupby("cell_id")["measured_incremental_contribution"].sum().le(0).any():
            raise ValueError(
                f"Every decision cell needs positive {split_name} contribution total"
            )

    return clean.sort_values(["week_start", "category", "channel"]).reset_index(
        drop=True
    )


def load_weekly_response(
    path: str | Path,
    *,
    validation_weeks: int,
    test_weeks: int,
) -> pd.DataFrame:
    """Read and validate an aggregate weekly response CSV."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Weekly response input does not exist: {input_path}")
    return validate_weekly_response(
        pd.read_csv(input_path),
        validation_weeks=validation_weeks,
        test_weeks=test_weeks,
    )


def evidence_status(frame: pd.DataFrame) -> str:
    """Classify whether the supplied evidence is eligible for planning interpretation."""

    evidence_types = set(frame["evidence_type"])
    return (
        "decision_eligible"
        if evidence_types and evidence_types.issubset(DECISION_ELIGIBLE_EVIDENCE_TYPES)
        else "demonstration_only"
    )
