from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from marketing_allocation.config import ProjectConfig
from marketing_allocation.pipeline import run_pipeline


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


if __name__ == "__main__":
    unittest.main()

