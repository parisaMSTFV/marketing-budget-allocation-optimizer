"""Constrained piecewise-linear marketing budget optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from marketing_allocation.config import CATEGORIES, CHANNELS, AllocationRules
from marketing_allocation.curves import ResponseCurveModel


@dataclass(frozen=True)
class ScenarioPlan:
    """Planning assumptions for one budget scenario."""

    name: str
    budget: float
    category_context: dict[str, float]
    risk_aversion: float = 0.0


@dataclass(frozen=True)
class AllocationResult:
    """Optimizer output and diagnostics."""

    allocation: pd.DataFrame
    objective_value: float
    solver_status: str
    diagnostics: dict[str, float]


def default_scenarios(rules: AllocationRules) -> tuple[ScenarioPlan, ...]:
    """Return transparent synthetic planning scenarios."""

    return (
        ScenarioPlan(
            name="Base",
            budget=rules.total_budget,
            category_context={category: 1.0 for category in CATEGORIES},
        ),
        ScenarioPlan(
            name="Growth",
            budget=rules.total_budget * 1.15,
            category_context={
                "Beauty": 1.08,
                "Electronics": 1.15,
                "Grocery": 1.06,
                "Home": 1.12,
            },
            risk_aversion=0.15,
        ),
        ScenarioPlan(
            name="Conservative",
            budget=rules.total_budget * 0.85,
            category_context={
                "Beauty": 0.91,
                "Electronics": 0.82,
                "Grocery": 0.96,
                "Home": 0.88,
            },
            risk_aversion=1.0,
        ),
    )


def planning_cells(history: pd.DataFrame, rules: AllocationRules) -> pd.DataFrame:
    """Build cell-level historical anchors and permitted spend ranges."""

    cells = (
        history.groupby(["cell_id", "category", "channel"], as_index=False)["spend"]
        .mean()
        .rename(columns={"spend": "historical_spend"})
        .sort_values(["category", "channel"])
        .reset_index(drop=True)
    )
    cells["minimum_spend"] = cells["historical_spend"] * rules.minimum_cell_multiple
    cells["maximum_spend"] = cells["historical_spend"] * rules.maximum_cell_multiple
    return cells


def _allocation_constraints(
    cells: pd.DataFrame,
    budget: float,
    rules: AllocationRules,
) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for category in CATEGORIES:
        mask = (cells["category"] == category).to_numpy(dtype=float)
        rows.extend([mask, -mask])
        limits.extend(
            [
                rules.maximum_category_share * budget,
                -rules.minimum_category_share * budget,
            ]
        )
    for channel in CHANNELS:
        mask = (cells["channel"] == channel).to_numpy(dtype=float)
        rows.extend([mask, -mask])
        limits.extend(
            [
                rules.maximum_channel_share * budget,
                -rules.minimum_channel_share * budget,
            ]
        )
    return rows, limits


def allocation_diagnostics(
    allocation: pd.DataFrame,
    budget: float,
    rules: AllocationRules,
) -> dict[str, float]:
    """Measure budget and business-rule compliance."""

    tolerance = max(budget * 1e-7, 0.01)
    spend = float(allocation["allocated_spend"].sum())
    violations: list[float] = [abs(spend - budget)]

    category_spend = allocation.groupby("category")["allocated_spend"].sum()
    for category in CATEGORIES:
        value = float(category_spend.get(category, 0.0))
        violations.append(max(rules.minimum_category_share * budget - value, 0.0))
        violations.append(max(value - rules.maximum_category_share * budget, 0.0))

    channel_spend = allocation.groupby("channel")["allocated_spend"].sum()
    for channel in CHANNELS:
        value = float(channel_spend.get(channel, 0.0))
        violations.append(max(rules.minimum_channel_share * budget - value, 0.0))
        violations.append(max(value - rules.maximum_channel_share * budget, 0.0))

    violations.extend(
        np.maximum(
            allocation["minimum_spend"].to_numpy()
            - allocation["allocated_spend"].to_numpy(),
            0.0,
        ).tolist()
    )
    violations.extend(
        np.maximum(
            allocation["allocated_spend"].to_numpy()
            - allocation["maximum_spend"].to_numpy(),
            0.0,
        ).tolist()
    )
    material = [value for value in violations if value > tolerance]
    return {
        "budget_gap": spend - budget,
        "budget_utilization": spend / budget,
        "constraint_violations": float(len(material)),
        "maximum_violation": max(violations),
    }


def project_target_allocation(
    cells: pd.DataFrame,
    target_spend: np.ndarray,
    scenario: ScenarioPlan,
    rules: AllocationRules,
) -> AllocationResult:
    """Find the closest feasible allocation to a historical or equal target."""

    n_cells = len(cells)
    target = np.asarray(target_spend, dtype=float)
    target = target / target.sum() * scenario.budget

    # Variables are allocation, positive deviation, and negative deviation.
    objective = np.r_[np.zeros(n_cells), np.ones(2 * n_cells)]
    equality_rows = []
    equality_limits = []
    for index in range(n_cells):
        row = np.zeros(3 * n_cells)
        row[index] = 1.0
        row[n_cells + index] = -1.0
        row[2 * n_cells + index] = 1.0
        equality_rows.append(row)
        equality_limits.append(target[index])
    budget_row = np.zeros(3 * n_cells)
    budget_row[:n_cells] = 1.0
    equality_rows.append(budget_row)
    equality_limits.append(scenario.budget)

    base_rows, base_limits = _allocation_constraints(cells, scenario.budget, rules)
    inequality_rows = [np.r_[row, np.zeros(2 * n_cells)] for row in base_rows]
    bounds = list(
        zip(cells["minimum_spend"], cells["maximum_spend"], strict=True)
    ) + [(0.0, None)] * (2 * n_cells)

    result = linprog(
        objective,
        A_ub=np.asarray(inequality_rows),
        b_ub=np.asarray(base_limits),
        A_eq=np.asarray(equality_rows),
        b_eq=np.asarray(equality_limits),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Target projection failed: {result.message}")

    allocation = cells.copy()
    allocation["allocated_spend"] = result.x[:n_cells]
    allocation["allocation_share"] = allocation["allocated_spend"] / scenario.budget
    diagnostics = allocation_diagnostics(allocation, scenario.budget, rules)
    return AllocationResult(
        allocation=allocation,
        objective_value=-float(result.fun),
        solver_status=str(result.message),
        diagnostics=diagnostics,
    )


def _risk_adjusted_value(
    model: ResponseCurveModel,
    spend: np.ndarray,
    context: float,
    risk_aversion: float,
) -> np.ndarray:
    penalty = max(1.0 - risk_aversion * model.relative_uncertainty, 0.20)
    return model.predict(spend) * context * penalty


def optimize_budget(
    models: dict[str, ResponseCurveModel],
    cells: pd.DataFrame,
    scenario: ScenarioPlan,
    rules: AllocationRules,
) -> AllocationResult:
    """Maximize risk-adjusted contribution subject to allocation constraints."""

    segment_count = rules.piecewise_segments
    lower = cells["minimum_spend"].to_numpy(dtype=float)
    upper = cells["maximum_spend"].to_numpy(dtype=float)
    remaining_budget = scenario.budget - float(lower.sum())
    if remaining_budget < -0.01 or scenario.budget > float(upper.sum()) + 0.01:
        raise ValueError("Scenario budget is infeasible under cell bounds")

    segment_widths: list[float] = []
    slopes: list[float] = []
    segment_cells: list[int] = []
    for cell_index, row in cells.iterrows():
        model = models[str(row["cell_id"])]
        grid = np.linspace(lower[cell_index], upper[cell_index], segment_count + 1)
        context = scenario.category_context[str(row["category"])]
        values = _risk_adjusted_value(
            model,
            grid,
            context=context,
            risk_aversion=scenario.risk_aversion,
        )
        widths = np.diff(grid)
        cell_slopes = np.diff(values) / widths
        if np.any(cell_slopes < -1e-8):
            raise ValueError(f"Non-monotonic response for {row['cell_id']}")
        for width, slope in zip(widths, cell_slopes, strict=True):
            segment_widths.append(float(width))
            slopes.append(float(slope))
            segment_cells.append(int(cell_index))

    n_variables = len(segment_widths)
    equality = np.ones((1, n_variables))
    equality_limit = np.asarray([remaining_budget])
    inequality_rows: list[np.ndarray] = []
    inequality_limits: list[float] = []

    def segment_mask(cell_mask: np.ndarray) -> np.ndarray:
        return np.asarray([cell_mask[cell_index] for cell_index in segment_cells], dtype=float)

    for category in CATEGORIES:
        cell_mask = (cells["category"] == category).to_numpy(dtype=float)
        row = segment_mask(cell_mask)
        lower_total = float(lower[cell_mask.astype(bool)].sum())
        inequality_rows.extend([row, -row])
        inequality_limits.extend(
            [
                rules.maximum_category_share * scenario.budget - lower_total,
                lower_total - rules.minimum_category_share * scenario.budget,
            ]
        )
    for channel in CHANNELS:
        cell_mask = (cells["channel"] == channel).to_numpy(dtype=float)
        row = segment_mask(cell_mask)
        lower_total = float(lower[cell_mask.astype(bool)].sum())
        inequality_rows.extend([row, -row])
        inequality_limits.extend(
            [
                rules.maximum_channel_share * scenario.budget - lower_total,
                lower_total - rules.minimum_channel_share * scenario.budget,
            ]
        )

    result = linprog(
        -np.asarray(slopes),
        A_ub=np.asarray(inequality_rows),
        b_ub=np.asarray(inequality_limits),
        A_eq=equality,
        b_eq=equality_limit,
        bounds=[(0.0, width) for width in segment_widths],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Budget optimization failed: {result.message}")

    allocation_values = lower.copy()
    for value, cell_index in zip(result.x, segment_cells, strict=True):
        allocation_values[cell_index] += value

    allocation = cells.copy()
    allocation["allocated_spend"] = allocation_values
    allocation["allocation_share"] = allocation["allocated_spend"] / scenario.budget
    predicted_values = []
    for _, row in allocation.iterrows():
        model = models[str(row["cell_id"])]
        predicted_values.append(
            float(
                _risk_adjusted_value(
                    model,
                    np.asarray([row["allocated_spend"]]),
                    context=scenario.category_context[str(row["category"])],
                    risk_aversion=scenario.risk_aversion,
                )[0]
            )
        )
    allocation["predicted_incremental_contribution"] = predicted_values
    allocation["predicted_incremental_profit"] = (
        allocation["predicted_incremental_contribution"] - allocation["allocated_spend"]
    )
    diagnostics = allocation_diagnostics(allocation, scenario.budget, rules)
    return AllocationResult(
        allocation=allocation,
        objective_value=float(allocation["predicted_incremental_contribution"].sum()),
        solver_status=str(result.message),
        diagnostics=diagnostics,
    )


def without_risk_penalty(scenario: ScenarioPlan) -> ScenarioPlan:
    """Create a scenario with the same context and budget but perfect information."""

    return replace(scenario, risk_aversion=0.0)
