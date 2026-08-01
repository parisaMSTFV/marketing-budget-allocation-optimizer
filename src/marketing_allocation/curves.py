"""Fit and validate diminishing-return response curves."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


def _linear(spend: np.ndarray, coefficient: float) -> np.ndarray:
    return coefficient * spend


def _log_response(spend: np.ndarray, scale: float, bend: float) -> np.ndarray:
    return scale * np.log1p(spend / bend)


def _saturation(spend: np.ndarray, maximum: float, half_spend: float) -> np.ndarray:
    return maximum * spend / (half_spend + spend)


MODEL_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "linear": _linear,
    "log": _log_response,
    "saturation": _saturation,
}


@dataclass(frozen=True)
class ResponseCurveModel:
    """Selected curve and its validation uncertainty for one decision cell."""

    cell_id: str
    category: str
    channel: str
    model_name: str
    parameters: tuple[float, ...]
    validation_wape: float
    validation_rmse: float
    relative_uncertainty: float
    training_end: pd.Timestamp

    def predict(self, spend: np.ndarray | float) -> np.ndarray:
        spend_array = np.asarray(spend, dtype=float)
        return MODEL_FUNCTIONS[self.model_name](spend_array, *self.parameters)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    if denominator <= 0:
        raise ValueError("WAPE requires a positive actual total")
    return float(np.abs(actual - predicted).sum() / denominator)


def _initial_values(model_name: str, spend: np.ndarray, outcome: np.ndarray):
    positive_spend = spend[spend > 0]
    median_spend = float(np.median(positive_spend))
    maximum_outcome = max(float(np.max(outcome)), 1.0)
    if model_name == "linear":
        return [maximum_outcome / max(float(np.max(spend)), 1.0)], ([0.0], [20.0])
    if model_name == "log":
        return [maximum_outcome, median_spend], ([1.0, 100.0], [1e8, 1e8])
    return [maximum_outcome * 1.4, median_spend], ([1.0, 100.0], [1e8, 1e8])


def fit_candidate(model_name: str, frame: pd.DataFrame) -> tuple[float, ...]:
    """Fit one candidate to context-normalized incremental contribution."""

    spend = frame["spend"].to_numpy(dtype=float)
    outcome = (
        frame["measured_incremental_contribution"] / frame["context_index"]
    ).to_numpy(dtype=float)
    initial, bounds = _initial_values(model_name, spend, outcome)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        parameters, _ = curve_fit(
            MODEL_FUNCTIONS[model_name],
            spend,
            outcome,
            p0=initial,
            bounds=bounds,
            maxfev=30_000,
        )
    return tuple(float(value) for value in parameters)


def predict_observations(
    model_name: str,
    parameters: tuple[float, ...],
    frame: pd.DataFrame,
) -> np.ndarray:
    base = MODEL_FUNCTIONS[model_name](frame["spend"].to_numpy(dtype=float), *parameters)
    return base * frame["context_index"].to_numpy(dtype=float)


def fit_response_curves(
    train_frame: pd.DataFrame,
    validation_weeks: int,
) -> tuple[dict[str, ResponseCurveModel], pd.DataFrame]:
    """Select a curve per cell on the latest training-only validation window."""

    unique_weeks = np.sort(train_frame["week_start"].unique())
    if len(unique_weeks) <= validation_weeks + 8:
        raise ValueError("Training history is too short for curve selection")
    validation_start = pd.Timestamp(unique_weeks[-validation_weeks])

    selected_models: dict[str, ResponseCurveModel] = {}
    comparison_rows: list[dict[str, object]] = []

    for cell_id, cell_frame in train_frame.groupby("cell_id", sort=True):
        calibration = cell_frame[cell_frame["week_start"] < validation_start]
        validation = cell_frame[cell_frame["week_start"] >= validation_start]
        candidate_results: list[tuple[str, float, float]] = []

        for model_name in MODEL_FUNCTIONS:
            parameters = fit_candidate(model_name, calibration)
            predicted = predict_observations(model_name, parameters, validation)
            actual = validation["measured_incremental_contribution"].to_numpy(dtype=float)
            candidate_wape = wape(actual, predicted)
            candidate_rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
            candidate_results.append((model_name, candidate_wape, candidate_rmse))
            comparison_rows.append(
                {
                    "cell_id": cell_id,
                    "category": cell_frame["category"].iloc[0],
                    "channel": cell_frame["channel"].iloc[0],
                    "model_name": model_name,
                    "validation_wape": candidate_wape,
                    "validation_rmse": candidate_rmse,
                }
            )

        selected_name, selected_wape, selected_rmse = min(
            candidate_results,
            key=lambda row: (row[1], row[0]),
        )
        refit_parameters = fit_candidate(selected_name, cell_frame)
        normalized_mean = float(
            (cell_frame["measured_incremental_contribution"] / cell_frame["context_index"]).mean()
        )
        selected_models[cell_id] = ResponseCurveModel(
            cell_id=cell_id,
            category=str(cell_frame["category"].iloc[0]),
            channel=str(cell_frame["channel"].iloc[0]),
            model_name=selected_name,
            parameters=refit_parameters,
            validation_wape=selected_wape,
            validation_rmse=selected_rmse,
            relative_uncertainty=min(selected_rmse / max(normalized_mean, 1.0), 0.50),
            training_end=pd.Timestamp(cell_frame["week_start"].max()),
        )

    return selected_models, pd.DataFrame(comparison_rows)


def evaluate_on_holdout(
    models: dict[str, ResponseCurveModel],
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the later untouched period without re-estimating any curve."""

    scored_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for cell_id, cell_frame in holdout.groupby("cell_id", sort=True):
        model = models[cell_id]
        scored = cell_frame.copy()
        scored["predicted_incremental_contribution"] = predict_observations(
            model.model_name,
            model.parameters,
            scored,
        )
        actual = scored["measured_incremental_contribution"].to_numpy(dtype=float)
        predicted = scored["predicted_incremental_contribution"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "cell_id": cell_id,
                "category": model.category,
                "channel": model.channel,
                "selected_model": model.model_name,
                "holdout_wape": wape(actual, predicted),
                "holdout_bias": float((predicted - actual).sum() / actual.sum()),
                "holdout_rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
            }
        )
        scored_parts.append(scored)
    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(metric_rows)
