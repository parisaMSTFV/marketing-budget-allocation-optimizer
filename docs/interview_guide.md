# Interview Guide

## Thirty-second explanation

The project estimates diminishing-return curves for each category-channel pair,
selects the curve family using a time-based validation window, and allocates a
fixed budget with a linear program. The recommendation respects operating floors,
concentration caps, and cell capacity. I then compare it with historical and
equal baselines on a later synthetic period and report regret versus the known
simulation oracle.

## Why not use a weighted score?

A score can rank opportunities, but it does not answer how much budget each
opportunity should receive. It also does not naturally express saturation. The
optimizer uses marginal returns, so the next unit of budget moves to the best
remaining feasible segment rather than to the cell with the highest static rank.

## Why is incrementality explicit?

Platform-attributed revenue can reward demand capture that would have happened
anyway. The synthetic outcome is framed as holdout-calibrated incremental
contribution, and the documentation states that real use requires experiments
or another credible incremental design.

## How is leakage prevented?

The latest weeks are reserved before curve fitting. Model-family selection uses
only a validation slice inside the earlier history. A test changes every future
outcome to an extreme value and confirms that selected models and parameters do
not move.

## Why piecewise linear optimization?

It preserves diminishing returns while keeping category, channel, budget, and
cell constraints transparent and auditable. The solver is fast, deterministic,
and easier to explain than a black-box search.

## What would change in production?

I would replace the synthetic outcome with experiment-calibrated contribution,
add real eligibility and capacity constraints, model parameter uncertainty more
formally, use approval workflows for overrides, and monitor realized incremental
profit after every allocation cycle.

