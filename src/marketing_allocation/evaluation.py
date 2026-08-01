"""Out-of-sample model and allocation policy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketing_allocation.curves import ResponseCurveModel
from marketing_allocation.optimization import ScenarioPlan


def evaluate_allocation(
    allocation: pd.DataFrame,
    truth_models: dict[str, ResponseCurveModel],
    scenario: ScenarioPlan,
    historical_allocation: pd.DataFrame,
) -> dict[str, float]:
    """Score one policy against hidden synthetic truth."""

    true_values = []
    for _, row in allocation.iterrows():
        model = truth_models[str(row["cell_id"])]
        context = scenario.category_context[str(row["category"])]
        true_values.append(float(model.predict(float(row["allocated_spend"])) * context))

    spend = float(allocation["allocated_spend"].sum())
    contribution = float(np.sum(true_values))
    historical = historical_allocation.set_index("cell_id")["allocated_spend"]
    aligned = allocation.set_index("cell_id")["allocated_spend"].reindex(historical.index)
    turnover = 0.5 * float(np.abs(aligned - historical).sum()) / spend
    return {
        "budget": spend,
        "true_incremental_contribution": contribution,
        "true_incremental_profit": contribution - spend,
        "incremental_contribution_roi": contribution / spend,
        "incremental_profit_roi": (contribution - spend) / spend,
        "allocation_turnover": turnover,
    }


def add_oracle_regret(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add scenario-level regret relative to a perfect-information allocation."""

    result = comparison.copy()
    oracle = (
        result.loc[result["policy"] == "Oracle"]
        .set_index("scenario")["true_incremental_contribution"]
        .to_dict()
    )
    result["regret_vs_oracle"] = result.apply(
        lambda row: (
            oracle[str(row["scenario"])] - row["true_incremental_contribution"]
        )
        / oracle[str(row["scenario"])],
        axis=1,
    )
    return result

