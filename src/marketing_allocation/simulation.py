"""Generate experiment-informed synthetic allocation evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketing_allocation.config import CATEGORIES, CHANNELS, DEFAULT_SEED, DEFAULT_WEEKS


@dataclass(frozen=True)
class SyntheticPortfolio:
    """Observed weekly evidence plus hidden truth for simulation-only evaluation."""

    observations: pd.DataFrame
    ground_truth: pd.DataFrame


def saturation_response(
    spend: np.ndarray | float,
    maximum: float,
    half_spend: float,
) -> np.ndarray:
    """Monotonic response with diminishing marginal returns."""

    spend_array = np.asarray(spend, dtype=float)
    return maximum * spend_array / (half_spend + spend_array)


def generate_portfolio(
    seed: int = DEFAULT_SEED,
    n_weeks: int = DEFAULT_WEEKS,
) -> SyntheticPortfolio:
    """Create weekly holdout-calibrated contribution observations for 12 cells.

    The measured outcome represents incremental contribution before marketing
    cost. It is intentionally synthetic and behaves like an experiment- or
    geo-test-calibrated outcome, not platform-attributed revenue.
    """

    if n_weeks < 52:
        raise ValueError("n_weeks must be at least 52")

    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    observations: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    # The synthetic historical mix contains stable legacy planning bias. This
    # makes the policy comparison realistic: operating spend is not assumed to
    # begin at the economic optimum.
    legacy_spend_bias = (
        1.30,
        0.85,
        1.15,
        0.75,
        1.25,
        0.90,
        1.20,
        0.80,
        1.05,
        0.70,
        1.30,
        0.90,
    )
    response_efficiency = (
        0.82,
        1.32,
        0.96,
        1.28,
        0.78,
        1.18,
        0.88,
        1.30,
        0.90,
        1.22,
        0.76,
        1.25,
    )

    for category_index, category in enumerate(CATEGORIES):
        for channel_index, channel in enumerate(CHANNELS):
            cell_index = category_index * len(CHANNELS) + channel_index
            economic_scale = 58_000 + 7_000 * category_index + 4_500 * channel_index
            base_spend = economic_scale * legacy_spend_bias[cell_index]
            half_spend = economic_scale * (0.72 + 0.10 * ((cell_index + 1) % 4))
            maximum = (
                economic_scale
                * (3.10 + 0.12 * category_index - 0.06 * channel_index)
                * response_efficiency[cell_index]
            )
            noise_rate = 0.045 + 0.012 * ((cell_index * 2 + 1) % 5)
            cell_id = f"{category} | {channel}"

            truth_rows.append(
                {
                    "cell_id": cell_id,
                    "category": category,
                    "channel": channel,
                    "base_spend": float(base_spend),
                    "maximum_contribution": float(maximum),
                    "half_saturation_spend": float(half_spend),
                    "noise_rate": float(noise_rate),
                }
            )

            cell_phase = 0.45 * cell_index
            for week_index, week_start in enumerate(weeks):
                seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * week_index / 26 + cell_phase)
                planned_pulse = 1.13 if week_index % 13 in {8, 9} else 1.0
                context_index = seasonal * planned_pulse

                # Randomized exploration makes the synthetic spend-response
                # relationship identifiable without pretending that last-click
                # attribution is incremental evidence.
                exploration = rng.uniform(0.48, 1.92)
                spend = base_spend * exploration * (0.94 + 0.12 * np.sin(week_index / 8))
                structural_value = context_index * saturation_response(
                    spend,
                    maximum=maximum,
                    half_spend=half_spend,
                )
                noise = rng.normal(0.0, noise_rate * structural_value + 900.0)
                measured = max(float(structural_value + noise), 0.0)

                observations.append(
                    {
                        "week_start": week_start,
                        "cell_id": cell_id,
                        "category": category,
                        "channel": channel,
                        "spend": float(spend),
                        "context_index": float(context_index),
                        "measured_incremental_contribution": measured,
                    }
                )

    observation_frame = pd.DataFrame(observations).sort_values(
        ["week_start", "category", "channel"]
    )
    truth_frame = pd.DataFrame(truth_rows).sort_values(["category", "channel"])
    return SyntheticPortfolio(
        observations=observation_frame.reset_index(drop=True),
        ground_truth=truth_frame.reset_index(drop=True),
    )
