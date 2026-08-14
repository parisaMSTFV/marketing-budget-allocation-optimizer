"""Create reproducible tables, figures, and decision-facing summaries."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketing_allocation.curves import ResponseCurveModel

INK = "#17202A"
BLUE = "#2563EB"
TEAL = "#0F766E"
AMBER = "#B45309"
GRAY = "#64748B"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def plot_response_curves(
    history: pd.DataFrame,
    models: dict[str, ResponseCurveModel],
    output_path: Path,
    *,
    synthetic_mode: bool = True,
) -> None:
    """Plot all selected curves against pre-test normalized observations."""

    cell_ids = sorted(models)
    column_count = min(3, len(cell_ids))
    row_count = math.ceil(len(cell_ids) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 4 * row_count),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    flat_axes = list(axes.flat)
    for axis, cell_id in zip(flat_axes, cell_ids, strict=False):
        frame = history.loc[history["cell_id"] == cell_id]
        normalized = frame["measured_incremental_contribution"] / frame["context_index"]
        axis.scatter(frame["spend"], normalized, s=12, alpha=0.32, color=GRAY)
        grid = np.linspace(0.0, float(frame["spend"].max()) * 1.08, 160)
        model = models[cell_id]
        axis.plot(grid, model.predict(grid), color=TEAL, linewidth=2.2)
        axis.set_title(f"{cell_id}\n{model.model_name}", fontsize=10, color=INK)
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(alpha=0.18)
    for axis in flat_axes[len(cell_ids) :]:
        axis.set_visible(False)
    figure.supxlabel(
        "Weekly spend (synthetic USD)"
        if synthetic_mode
        else "Weekly spend (input currency)",
        fontsize=11,
    )
    figure.supylabel("Context-normalized incremental contribution", fontsize=11)
    figure.suptitle("Selected diminishing-return curves", fontsize=17, color=INK, y=0.998)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_holdout_quality(holdout_metrics: pd.DataFrame, output_path: Path) -> None:
    """Show untouched-period WAPE by category-channel cell."""

    ordered = holdout_metrics.sort_values("holdout_wape", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 6.5))
    colors = [
        TEAL if value <= ordered["holdout_wape"].median() else BLUE
        for value in ordered["holdout_wape"]
    ]
    axis.barh(ordered["cell_id"], ordered["holdout_wape"] * 100, color=colors)
    axis.axvline(
        ordered["holdout_wape"].median() * 100,
        color=AMBER,
        linestyle="--",
        linewidth=1.6,
        label="Median",
    )
    axis.set_xlabel("Holdout WAPE (%)")
    axis.set_title("Response-curve accuracy on the untouched period", color=INK, fontsize=15)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_allocation_heatmaps(allocations: pd.DataFrame, output_path: Path) -> None:
    """Compare historical and optimized Base allocations."""

    subset = allocations.loc[
        (allocations["scenario"] == "Base")
        & (allocations["policy"].isin(["Historical", "Optimized"]))
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    maximum = float(subset["allocation_share"].max())
    for axis, policy in zip(axes, ["Historical", "Optimized"], strict=True):
        matrix = (
            subset.loc[subset["policy"] == policy]
            .pivot(index="category", columns="channel", values="allocation_share")
            .reindex(index=sorted(subset["category"].unique()))
        )
        image = axis.imshow(matrix.to_numpy() * 100, cmap="Blues", vmin=0, vmax=maximum * 100)
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title(f"{policy} allocation", color=INK, fontsize=14)
        for row_index in range(len(matrix.index)):
            for column_index in range(len(matrix.columns)):
                value = matrix.iloc[row_index, column_index] * 100
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > maximum * 50 else INK,
                    fontsize=9,
                )
    figure.colorbar(image, ax=axes, shrink=0.78, label="Share of scenario budget (%)")
    figure.suptitle("Where the optimizer reallocates the budget", fontsize=16, color=INK)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scenario_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Compare synthetic truth or fitted-model policy value by scenario."""

    synthetic_mode = "true_incremental_profit" in comparison.columns
    value_column = (
        "true_incremental_profit" if synthetic_mode else "modeled_incremental_profit"
    )
    policy_order = ["Historical", "Equal", "Optimized", "Oracle"]
    policies = [
        policy for policy in policy_order if policy in set(comparison["policy"])
    ]
    scenario_order = ["Base", "Growth", "Conservative"]
    scenarios = [
        scenario
        for scenario in scenario_order
        if scenario in set(comparison["scenario"])
    ]
    pivot = comparison.pivot(
        index="scenario",
        columns="policy",
        values=value_column,
    ).reindex(index=scenarios, columns=policies)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(scenarios))
    width = 0.72 / len(policies)
    colors = [GRAY, AMBER, TEAL, BLUE]
    for index, policy in enumerate(policies):
        axis.bar(
            x + (index - (len(policies) - 1) / 2) * width,
            pivot[policy] / 1_000_000,
            width,
            label=policy,
            color=colors[index],
        )
    axis.set_xticks(x, scenarios)
    axis.set_ylabel(
        "True incremental profit (synthetic USD, millions)"
        if synthetic_mode
        else "Modeled incremental profit (millions)"
    )
    axis.set_title(
        "Policy value across planning scenarios"
        if synthetic_mode
        else "Policy value under documented evidence",
        color=INK,
        fontsize=15,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=4, loc="upper center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_run_summary(path: Path, summary: dict[str, object]) -> None:
    """Write a concise, reproducible run record."""

    base = summary["base_policy_results"]
    selected = Counter(summary["selected_models"].values())
    lines = [
        "# Reproducible Run Summary",
        "",
        f"- Data mode: `{summary['data_mode']}`",
        f"- Seed: `{summary['seed']}`" if summary["seed"] is not None else "- Seed: not used",
        f"- Weekly observations: {summary['observations']:,}",
        f"- Decision cells: {summary['decision_cells']}",
        f"- History / validation / test weeks: {summary['history_weeks']} / "
        f"{summary['validation_weeks']} / {summary['test_weeks']}",
        f"- Mean holdout WAPE: {summary['mean_holdout_wape']:.1%}",
        f"- Median holdout WAPE: {summary['median_holdout_wape']:.1%}",
        f"- Selected curves: {dict(selected)}",
        "",
        "## Base scenario",
        "",
    ]
    if summary["data_mode"] == "synthetic_simulation":
        lines.extend(
            [
                "| Policy | Budget | Incremental profit | Contribution ROI | Regret vs oracle |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for policy in ["Historical", "Equal", "Optimized", "Oracle"]:
            row = base[policy]
            lines.append(
                f"| {policy} | {_currency(row['budget'])} | "
                f"{_currency(row['true_incremental_profit'])} | "
                f"{row['incremental_contribution_roi']:.2f}x | "
                f"{row['regret_vs_oracle']:.1%} |"
            )
        lines.extend(
            [
                "",
                "All values are generated by the committed synthetic simulator. "
                "The oracle is available only because this is a simulation.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Evidence status: `{summary['evidence_status']}`",
                f"- Evidence types: {', '.join(summary['evidence_types'])}",
                f"- Evidence references: {', '.join(summary['evidence_references'])}",
                "",
                "| Policy | Budget | Modeled incremental profit | "
                "Modeled contribution ROI | Turnover |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for policy in ["Historical", "Equal", "Optimized"]:
            row = base[policy]
            lines.append(
                f"| {policy} | {_currency(row['budget'])} | "
                f"{_currency(row['modeled_incremental_profit'])} | "
                f"{row['modeled_incremental_contribution_roi']:.2f}x | "
                f"{row['allocation_turnover']:.1%} |"
            )
        lines.extend(
            [
                "",
                "Values in this table are fitted-model planning estimates. They are not "
                "realized policy impact, and no simulation oracle is available.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_note(
    path: Path,
    recommended: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Turn model output into a short planning decision note."""

    base = comparison.loc[comparison["scenario"] == "Base"].set_index("policy")
    synthetic_mode = "true_incremental_profit" in base.columns
    profit_column = (
        "true_incremental_profit" if synthetic_mode else "modeled_incremental_profit"
    )
    historical_profit = float(base.loc["Historical", profit_column])
    optimized_profit = float(base.loc["Optimized", profit_column])
    uplift = (
        optimized_profit / historical_profit - 1.0
        if abs(historical_profit) > 1e-9
        else None
    )
    if synthetic_mode:
        impact_sentence = (
            f"In the synthetic test environment it produces {uplift:.1%} more "
            "incremental profit than the feasible historical mix while using the same budget."
            if uplift is not None
            else "The synthetic historical profit is zero, so a percentage improvement "
            "is not reported."
        )
    else:
        impact_sentence = (
            f"The fitted curves estimate {uplift:.1%} more incremental profit than "
            "the feasible historical mix at the same budget; this is not realized impact."
            if uplift is not None
            else "The fitted historical profit is zero, so a percentage improvement is "
            "not reported; modeled levels and constraints still require review."
        )
    largest_moves = recommended.assign(
        change=recommended["allocated_spend"] - recommended["reference_spend"]
    ).sort_values("change", ascending=False)
    increase = largest_moves.iloc[0]
    decrease = largest_moves.iloc[-1]
    lines = [
        "# Allocation Decision Note",
        "",
        "## Recommendation",
        "",
        "Use the constrained Base allocation as the next planning proposal. "
        + impact_sentence,
        "",
        "## Largest modeled moves",
        "",
        f"- Increase: **{increase['cell_id']}** by {_currency(increase['change'])}.",
        f"- Decrease: **{decrease['cell_id']}** by {_currency(abs(decrease['change']))}.",
        "",
        "## Guardrails before production use",
        "",
        "- Treat the allocation as a planning recommendation, not an automatic spend change.",
        "- Confirm inventory, eligibility, brand, and operational constraints outside this model.",
        "- Preserve randomized holdouts so future response curves remain incremental.",
        "- Require a documented override reason when planners change the recommendation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
