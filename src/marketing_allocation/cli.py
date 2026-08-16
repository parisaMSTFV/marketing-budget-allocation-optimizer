"""Command-line interface for the allocation case study."""

from __future__ import annotations

import argparse
from pathlib import Path

from marketing_allocation.config import AllocationRules, ProjectConfig
from marketing_allocation.optimization import InfeasibleAllocationError
from marketing_allocation.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run marketing budget allocation from synthetic or supplied evidence.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("local-runs/latest"),
        help="Output root; defaults to an ignored local-run directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weeks", type=int, default=104)
    parser.add_argument("--test-weeks", type=int, default=20)
    parser.add_argument("--validation-weeks", type=int, default=16)
    parser.add_argument(
        "--input-weekly-response",
        type=Path,
        help="Aggregate experiment-informed weekly response CSV.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="Required planning budget when --input-weekly-response is supplied.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.input_weekly_response is not None and args.budget is None:
        parser.error("--budget is required with --input-weekly-response")
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")

    rules = AllocationRules(
        total_budget=args.budget
        if args.budget is not None
        else AllocationRules().total_budget
    )
    config = ProjectConfig(
        seed=args.seed,
        n_weeks=args.weeks,
        test_weeks=args.test_weeks,
        validation_weeks=args.validation_weeks,
        rules=rules,
    )
    try:
        summary = run_pipeline(
            args.project_root,
            config,
            input_weekly_response=args.input_weekly_response,
        )
    except InfeasibleAllocationError as error:
        parser.exit(2, f"Allocation stopped: {error}\n")
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Input rejected: {error}\n")
    base = summary["base_policy_results"]
    optimized = base["Optimized"]
    historical = base["Historical"]
    profit_key = (
        "true_incremental_profit"
        if summary["data_mode"] == "synthetic_simulation"
        else "modeled_incremental_profit"
    )
    historical_profit = float(historical[profit_key])
    improvement = (
        f"{float(optimized[profit_key]) / historical_profit - 1.0:.1%}"
        if abs(historical_profit) > 1e-9
        else "not defined because historical profit is zero"
    )
    print(
        "Completed allocation run: "
        f"data_mode={summary['data_mode']}, "
        f"mean holdout WAPE={summary['mean_holdout_wape']:.1%}, "
        f"Base profit improvement={improvement}"
    )


if __name__ == "__main__":
    main()
