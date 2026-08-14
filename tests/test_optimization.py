from __future__ import annotations

import unittest

import numpy as np

from marketing_allocation.config import AllocationRules
from marketing_allocation.curves import fit_response_curves
from marketing_allocation.optimization import (
    InfeasibleAllocationError,
    ScenarioPlan,
    default_scenarios,
    optimize_budget,
    planning_cells,
    project_target_allocation,
)
from marketing_allocation.simulation import generate_portfolio


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = AllocationRules()
        cls.history = generate_portfolio(seed=21, n_weeks=64).observations
        cls.models, _ = fit_response_curves(cls.history, validation_weeks=10)
        cls.cells = planning_cells(cls.history, cls.rules)

    def test_optimizer_respects_all_constraints(self) -> None:
        scenario = default_scenarios(self.rules)[0]
        result = optimize_budget(self.models, self.cells, scenario, self.rules)
        self.assertAlmostEqual(
            result.allocation["allocated_spend"].sum(),
            scenario.budget,
            places=4,
        )
        self.assertEqual(result.diagnostics["constraint_violations"], 0)
        self.assertAlmostEqual(result.diagnostics["budget_utilization"], 1.0, places=7)

    def test_optimizer_beats_feasible_historical_mix_on_its_objective(self) -> None:
        scenario = default_scenarios(self.rules)[0]
        optimized = optimize_budget(self.models, self.cells, scenario, self.rules)
        historical = project_target_allocation(
            self.cells,
            self.cells["historical_spend"].to_numpy(),
            scenario,
            self.rules,
        )
        predicted_historical = 0.0
        for _, row in historical.allocation.iterrows():
            predicted_historical += float(
                self.models[row["cell_id"]].predict(float(row["allocated_spend"]))
            )
        self.assertGreaterEqual(optimized.objective_value, predicted_historical - 1e-5)

    def test_infeasible_share_constraints_fail_without_fallback(self) -> None:
        impossible_rules = AllocationRules(
            minimum_category_share=0.30,
            maximum_category_share=0.35,
        )
        cells = planning_cells(self.history, impossible_rules)
        scenario = ScenarioPlan(
            name="Impossible",
            budget=impossible_rules.total_budget,
            category_context={category: 1.0 for category in cells["category"].unique()},
        )

        with self.assertRaisesRegex(
            InfeasibleAllocationError, "No allocation satisfies"
        ):
            optimize_budget(self.models, cells, scenario, impossible_rules)

    def test_zero_discretionary_budget_returns_exact_minimum_allocation(self) -> None:
        anchor_cells = planning_cells(self.history, AllocationRules())
        fixed_budget = float(anchor_cells["historical_spend"].sum())
        fixed_rules = AllocationRules(
            total_budget=fixed_budget,
            minimum_category_share=0.0,
            maximum_category_share=1.0,
            minimum_channel_share=0.0,
            maximum_channel_share=1.0,
            minimum_cell_multiple=1.0,
            maximum_cell_multiple=1.0,
        )
        fixed_cells = planning_cells(self.history, fixed_rules)
        scenario = ScenarioPlan(
            name="Fixed",
            budget=fixed_budget,
            category_context={
                category: 1.0 for category in fixed_cells["category"].unique()
            },
        )

        result = optimize_budget(self.models, fixed_cells, scenario, fixed_rules)

        np.testing.assert_allclose(
            result.allocation["allocated_spend"], fixed_cells["minimum_spend"]
        )
        self.assertIn("no discretionary budget", result.solver_status)
        self.assertEqual(result.diagnostics["constraint_violations"], 0)


if __name__ == "__main__":
    unittest.main()
