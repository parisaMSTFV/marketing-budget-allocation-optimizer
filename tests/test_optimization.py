from __future__ import annotations

import unittest

from marketing_allocation.config import AllocationRules
from marketing_allocation.curves import fit_response_curves
from marketing_allocation.optimization import (
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


if __name__ == "__main__":
    unittest.main()
