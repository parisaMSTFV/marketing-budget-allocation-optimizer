from __future__ import annotations

import unittest

import pandas as pd

from marketing_allocation.simulation import generate_portfolio


class SimulationTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = generate_portfolio(seed=11, n_weeks=52)
        second = generate_portfolio(seed=11, n_weeks=52)
        pd.testing.assert_frame_equal(first.observations, second.observations)
        pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)

    def test_schema_contains_only_aggregate_decision_cells(self) -> None:
        portfolio = generate_portfolio(seed=7, n_weeks=52)
        self.assertEqual(portfolio.observations["cell_id"].nunique(), 12)
        self.assertNotIn("customer_id", portfolio.observations.columns)
        self.assertTrue((portfolio.observations["spend"] > 0).all())
        self.assertTrue(
            (portfolio.observations["measured_incremental_contribution"] >= 0).all()
        )


if __name__ == "__main__":
    unittest.main()

