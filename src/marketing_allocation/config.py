"""Configuration for the reproducible allocation case study."""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ("Beauty", "Electronics", "Grocery", "Home")
CHANNELS = ("Affiliate", "Lifecycle CRM", "Paid Search")
DEFAULT_SEED = 42
DEFAULT_WEEKS = 104
DEFAULT_TEST_WEEKS = 20
DEFAULT_VALIDATION_WEEKS = 16
DEFAULT_BUDGET = 1_200_000.0


@dataclass(frozen=True)
class AllocationRules:
    """Business constraints applied to every allocation strategy."""

    total_budget: float = DEFAULT_BUDGET
    minimum_category_share: float = 0.15
    maximum_category_share: float = 0.35
    minimum_channel_share: float = 0.12
    maximum_channel_share: float = 0.48
    minimum_cell_multiple: float = 0.40
    maximum_cell_multiple: float = 2.20
    piecewise_segments: int = 24


@dataclass(frozen=True)
class ProjectConfig:
    """End-to-end run settings."""

    seed: int = DEFAULT_SEED
    n_weeks: int = DEFAULT_WEEKS
    test_weeks: int = DEFAULT_TEST_WEEKS
    validation_weeks: int = DEFAULT_VALIDATION_WEEKS
    rules: AllocationRules = AllocationRules()

