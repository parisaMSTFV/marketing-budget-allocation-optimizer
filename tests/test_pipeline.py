from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from marketing_allocation.config import AllocationRules, ProjectConfig
from marketing_allocation.pipeline import run_pipeline
from marketing_allocation.simulation import generate_portfolio


class PipelineTests(unittest.TestCase):
    def test_end_to_end_run_writes_decision_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_pipeline(
                root,
                ProjectConfig(
                    seed=4,
                    n_weeks=60,
                    validation_weeks=10,
                    test_weeks=10,
                ),
            )
            self.assertEqual(summary["decision_cells"], 12)
            self.assertTrue((root / "reports" / "recommended_allocation.csv").exists())
            self.assertTrue((root / "reports" / "run_summary.md").exists())
            self.assertTrue((root / "reports" / "figures" / "allocation_heatmap.png").exists())

    def test_experiment_input_runs_without_simulation_truth_or_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "weekly_response.csv"
            observations = generate_portfolio(seed=15, n_weeks=60).observations
            observations["evidence_type"] = "randomized_experiment"
            observations["evidence_reference"] = "test-experiment-registry-001"
            observations.to_csv(input_path, index=False)
            stale_truth = root / "data/generated/simulation_truth.csv"
            stale_truth.parent.mkdir(parents=True, exist_ok=True)
            stale_truth.write_text("stale synthetic truth", encoding="utf-8")

            summary = run_pipeline(
                root,
                ProjectConfig(
                    seed=4,
                    n_weeks=60,
                    validation_weeks=10,
                    test_weeks=10,
                    rules=AllocationRules(total_budget=1_200_000.0),
                ),
                input_weekly_response=input_path,
            )

            self.assertEqual(summary["data_mode"], "experiment_informed_input")
            self.assertEqual(summary["evidence_status"], "decision_eligible")
            self.assertNotIn("Oracle", summary["base_policy_results"])
            self.assertIn(
                "modeled_incremental_profit",
                summary["base_policy_results"]["Optimized"],
            )
            self.assertFalse(stale_truth.exists())
            report = (root / "reports/run_summary.md").read_text(encoding="utf-8")
            self.assertIn("fitted-model planning estimates", report)
            self.assertNotIn("Regret vs oracle", report)
            recommendation = pd.read_csv(root / "reports/recommended_allocation.csv")
            self.assertEqual(len(recommendation), 12)


if __name__ == "__main__":
    unittest.main()
