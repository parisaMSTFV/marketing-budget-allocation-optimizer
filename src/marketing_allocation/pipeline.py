"""End-to-end simulation, curve fitting, optimization, and reporting pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from marketing_allocation.config import ProjectConfig
from marketing_allocation.curves import (
    ResponseCurveModel,
    evaluate_on_holdout,
    fit_response_curves,
)
from marketing_allocation.evaluation import (
    add_oracle_regret,
    evaluate_allocation,
    evaluate_modeled_allocation,
)
from marketing_allocation.input_data import evidence_status, load_weekly_response
from marketing_allocation.optimization import (
    ScenarioPlan,
    default_scenarios,
    optimize_budget,
    planning_cells,
    project_target_allocation,
    without_risk_penalty,
)
from marketing_allocation.reporting import (
    plot_allocation_heatmaps,
    plot_holdout_quality,
    plot_response_curves,
    plot_scenario_comparison,
    write_decision_note,
    write_json,
    write_run_summary,
)
from marketing_allocation.simulation import generate_portfolio


def split_history_holdout(
    observations: pd.DataFrame,
    test_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest complete weeks as an untouched temporal holdout."""

    weeks = np.sort(observations["week_start"].unique())
    if len(weeks) <= test_weeks + 24:
        raise ValueError("Not enough history for temporal training and testing")
    test_start = pd.Timestamp(weeks[-test_weeks])
    history = observations.loc[observations["week_start"] < test_start].copy()
    holdout = observations.loc[observations["week_start"] >= test_start].copy()
    return history.reset_index(drop=True), holdout.reset_index(drop=True)


def _truth_models(
    ground_truth: pd.DataFrame,
    training_end: pd.Timestamp,
) -> dict[str, ResponseCurveModel]:
    models = {}
    for _, row in ground_truth.iterrows():
        cell_id = str(row["cell_id"])
        models[cell_id] = ResponseCurveModel(
            cell_id=cell_id,
            category=str(row["category"]),
            channel=str(row["channel"]),
            model_name="saturation",
            parameters=(
                float(row["maximum_contribution"]),
                float(row["half_saturation_spend"]),
            ),
            validation_wape=0.0,
            validation_rmse=0.0,
            relative_uncertainty=0.0,
            training_end=training_end,
        )
    return models


def run_pipeline(
    project_root: str | Path,
    config: ProjectConfig | None = None,
    input_weekly_response: str | Path | None = None,
) -> dict[str, object]:
    """Run the synthetic case study or an experiment-informed aggregate input."""

    config = config or ProjectConfig()
    project_root = Path(project_root)
    generated_dir = project_root / "data" / "generated"
    sample_dir = project_root / "data" / "sample"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    for directory in (generated_dir, sample_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    portfolio = None
    if input_weekly_response is None:
        portfolio = generate_portfolio(seed=config.seed, n_weeks=config.n_weeks)
        observations = portfolio.observations
        data_mode = "synthetic_simulation"
    else:
        observations = load_weekly_response(
            input_weekly_response,
            validation_weeks=config.validation_weeks,
            test_weeks=config.test_weeks,
        )
        data_mode = "experiment_informed_input"
    observations.to_csv(generated_dir / "weekly_observations.csv", index=False)
    truth_path = generated_dir / "simulation_truth.csv"
    if portfolio is not None:
        portfolio.ground_truth.to_csv(truth_path, index=False)
        observations.head(36).to_csv(
            sample_dir / "synthetic_weekly_sample.csv", index=False
        )
    else:
        truth_path.unlink(missing_ok=True)

    history, holdout = split_history_holdout(observations, config.test_weeks)
    models, model_comparison = fit_response_curves(
        history,
        validation_weeks=config.validation_weeks,
    )
    _, holdout_metrics = evaluate_on_holdout(models, holdout)
    truth_models = (
        _truth_models(
            portfolio.ground_truth,
            training_end=pd.Timestamp(history["week_start"].max()),
        )
        if portfolio is not None
        else None
    )

    recent_weeks = sorted(history["week_start"].unique())[-12:]
    cells = planning_cells(
        history.loc[history["week_start"].isin(recent_weeks)],
        config.rules,
    )

    allocation_rows: list[pd.DataFrame] = []
    evaluation_rows: list[dict[str, object]] = []
    scenarios = (
        default_scenarios(config.rules)
        if portfolio is not None
        else (
            ScenarioPlan(
                name="Base",
                budget=config.rules.total_budget,
                category_context={
                    str(category): 1.0
                    for category in sorted(cells["category"].unique())
                },
            ),
        )
    )
    for scenario in scenarios:
        historical = project_target_allocation(
            cells,
            cells["historical_spend"].to_numpy(),
            scenario,
            config.rules,
        )
        equal = project_target_allocation(
            cells,
            np.ones(len(cells)),
            scenario,
            config.rules,
        )
        optimized = optimize_budget(models, cells, scenario, config.rules)
        policies = {
            "Historical": historical.allocation,
            "Equal": equal.allocation,
            "Optimized": optimized.allocation,
        }
        if truth_models is not None:
            oracle = optimize_budget(
                truth_models,
                cells,
                without_risk_penalty(scenario),
                config.rules,
            )
            policies["Oracle"] = oracle.allocation
        for policy_name, allocation in policies.items():
            output = allocation.copy()
            output.insert(0, "policy", policy_name)
            output.insert(0, "scenario", scenario.name)
            allocation_rows.append(output)
            metrics = (
                evaluate_allocation(
                    allocation,
                    truth_models,
                    scenario,
                    historical.allocation,
                )
                if truth_models is not None
                else evaluate_modeled_allocation(
                    allocation,
                    models,
                    scenario,
                    historical.allocation,
                )
            )
            evaluation_rows.append(
                {
                    "scenario": scenario.name,
                    "policy": policy_name,
                    **metrics,
                }
            )

    allocations = pd.concat(allocation_rows, ignore_index=True)
    scenario_comparison = pd.DataFrame(evaluation_rows)
    if truth_models is not None:
        scenario_comparison = add_oracle_regret(scenario_comparison)
    recommended = allocations.loc[
        (allocations["scenario"] == "Base") & (allocations["policy"] == "Optimized")
    ].copy()
    base_historical = (
        allocations.loc[
            (allocations["scenario"] == "Base")
            & (allocations["policy"] == "Historical"),
            ["cell_id", "allocated_spend"],
        ]
        .rename(columns={"allocated_spend": "reference_spend"})
        .copy()
    )
    recommended = recommended.merge(base_historical, on="cell_id", how="left")

    model_comparison.to_csv(reports_dir / "curve_model_comparison.csv", index=False)
    holdout_metrics.to_csv(reports_dir / "holdout_metrics.csv", index=False)
    allocations.to_csv(reports_dir / "all_scenario_allocations.csv", index=False)
    scenario_comparison.to_csv(reports_dir / "scenario_comparison.csv", index=False)
    recommended.to_csv(reports_dir / "recommended_allocation.csv", index=False)

    plot_response_curves(
        history,
        models,
        figures_dir / "response_curves.png",
        synthetic_mode=portfolio is not None,
    )
    plot_holdout_quality(holdout_metrics, figures_dir / "holdout_quality.png")
    plot_allocation_heatmaps(allocations, figures_dir / "allocation_heatmap.png")
    plot_scenario_comparison(
        scenario_comparison,
        figures_dir / "scenario_comparison.png",
    )

    base_policy_results = (
        scenario_comparison.loc[scenario_comparison["scenario"] == "Base"]
        .set_index("policy")
        .to_dict(orient="index")
    )
    summary: dict[str, object] = {
        "data_mode": data_mode,
        "seed": config.seed if portfolio is not None else None,
        "observations": len(observations),
        "decision_cells": observations["cell_id"].nunique(),
        "total_weeks": observations["week_start"].nunique(),
        "history_weeks": history["week_start"].nunique() - config.validation_weeks,
        "validation_weeks": config.validation_weeks,
        "test_weeks": holdout["week_start"].nunique(),
        "mean_holdout_wape": float(holdout_metrics["holdout_wape"].mean()),
        "median_holdout_wape": float(holdout_metrics["holdout_wape"].median()),
        "selected_models": {
            cell_id: model.model_name for cell_id, model in sorted(models.items())
        },
        "base_policy_results": base_policy_results,
        "config": asdict(config),
    }
    if portfolio is None:
        summary.update(
            {
                "input_file": Path(input_weekly_response).name,
                "evidence_types": sorted(observations["evidence_type"].unique()),
                "evidence_references": sorted(
                    observations["evidence_reference"].unique()
                ),
                "evidence_status": evidence_status(observations),
            }
        )
    write_json(reports_dir / "run_metrics.json", summary)
    write_run_summary(reports_dir / "run_summary.md", summary)
    write_decision_note(
        reports_dir / "decision_note.md",
        recommended,
        scenario_comparison,
    )
    return summary
