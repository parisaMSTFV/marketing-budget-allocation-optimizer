"""Command-line interface for the allocation case study."""

from __future__ import annotations

import argparse
from pathlib import Path

from marketing_allocation.config import ProjectConfig
from marketing_allocation.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic marketing budget allocation pipeline.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for generated reports.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weeks", type=int, default=104)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(
        args.project_root,
        ProjectConfig(seed=args.seed, n_weeks=args.weeks),
    )
    base = summary["base_policy_results"]
    optimized = base["Optimized"]
    historical = base["Historical"]
    improvement = (
        optimized["true_incremental_profit"] / historical["true_incremental_profit"]
        - 1.0
    )
    print(
        "Completed allocation run: "
        f"mean holdout WAPE={summary['mean_holdout_wape']:.1%}, "
        f"Base profit improvement={improvement:.1%}"
    )


if __name__ == "__main__":
    main()

