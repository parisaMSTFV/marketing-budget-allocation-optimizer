from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from marketing_allocation.curves import fit_response_curves
from marketing_allocation.pipeline import split_history_holdout
from marketing_allocation.simulation import generate_portfolio


class CurveTests(unittest.TestCase):
    def test_future_outcomes_do_not_change_fitted_curves(self) -> None:
        observations = generate_portfolio(seed=13, n_weeks=64).observations
        history, holdout = split_history_holdout(observations, test_weeks=12)
        first_models, _ = fit_response_curves(history, validation_weeks=10)

        mutated = pd.concat([history, holdout.assign(measured_incremental_contribution=9e9)])
        mutated_history, _ = split_history_holdout(mutated, test_weeks=12)
        second_models, _ = fit_response_curves(mutated_history, validation_weeks=10)

        for cell_id in first_models:
            self.assertEqual(first_models[cell_id].model_name, second_models[cell_id].model_name)
            np.testing.assert_allclose(
                first_models[cell_id].parameters,
                second_models[cell_id].parameters,
            )

    def test_training_end_precedes_holdout(self) -> None:
        observations = generate_portfolio(seed=9, n_weeks=60).observations
        history, holdout = split_history_holdout(observations, test_weeks=10)
        models, _ = fit_response_curves(history, validation_weeks=10)
        holdout_start = holdout["week_start"].min()
        self.assertTrue(all(model.training_end < holdout_start for model in models.values()))


if __name__ == "__main__":
    unittest.main()

