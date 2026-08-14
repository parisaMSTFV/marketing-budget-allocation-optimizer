from __future__ import annotations

import unittest

import pandas as pd

from marketing_allocation.input_data import validate_weekly_response
from marketing_allocation.simulation import generate_portfolio


def experiment_input(weeks: int = 36) -> pd.DataFrame:
    frame = generate_portfolio(seed=9, n_weeks=52).observations
    keep = sorted(frame["week_start"].unique())[:weeks]
    frame = frame.loc[frame["week_start"].isin(keep)].copy()
    frame["evidence_type"] = "geo_experiment"
    frame["evidence_reference"] = "test-design-v1"
    return frame


class InputDataTests(unittest.TestCase):
    def test_valid_experiment_input_is_typed_and_sorted(self) -> None:
        validated = validate_weekly_response(
            experiment_input(), validation_weeks=5, test_weeks=6
        )

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(validated["week_start"]))
        self.assertEqual(validated["cell_id"].nunique(), 12)
        self.assertEqual(validated["week_start"].nunique(), 36)
        self.assertEqual(set(validated["evidence_type"]), {"geo_experiment"})

    def test_missing_provenance_is_rejected(self) -> None:
        broken = experiment_input().drop(columns="evidence_reference")

        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_weekly_response(broken, validation_weeks=5, test_weeks=6)

    def test_incomplete_weekly_panel_is_rejected(self) -> None:
        broken = experiment_input().iloc[1:].copy()

        with self.assertRaisesRegex(ValueError, "Every week"):
            validate_weekly_response(broken, validation_weeks=5, test_weeks=6)

    def test_observational_attribution_is_not_accepted_as_incremental_evidence(self) -> None:
        broken = experiment_input()
        broken["evidence_type"] = "platform_attribution"

        with self.assertRaisesRegex(ValueError, "Unsupported evidence_type"):
            validate_weekly_response(broken, validation_weeks=5, test_weeks=6)

    def test_zero_contribution_test_cell_is_rejected_before_modeling(self) -> None:
        broken = experiment_input()
        test_weeks = sorted(broken["week_start"].unique())[-6:]
        mask = broken["week_start"].isin(test_weeks) & broken["cell_id"].eq(
            "Beauty | Affiliate"
        )
        broken.loc[mask, "measured_incremental_contribution"] = 0.0

        with self.assertRaisesRegex(ValueError, "positive test contribution"):
            validate_weekly_response(broken, validation_weeks=5, test_weeks=6)


if __name__ == "__main__":
    unittest.main()
