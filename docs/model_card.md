# Model Card

## Intended use

This case study demonstrates how a planning team can translate incremental
marketing evidence into a constrained category-channel budget proposal. It is
designed for portfolio review, learning, and interview discussion.

## Model families

Each category-channel cell compares:

- a linear response;
- a logarithmic diminishing-return response;
- a saturation response.

Validation WAPE selects one family per cell. The winner is refit on all pre-test
history and evaluated on a later untouched period.

## Optimization layer

The chosen response is approximated with piecewise-linear segments. The solver
maximizes risk-adjusted contribution under exact budget use, category and
channel concentration limits, and per-cell minimum and maximum spend.

## Appropriate interpretation

The output is a planning recommendation. It shows where spend would move if
the fitted response curves, scenario assumptions, and constraints were accepted.
Human review is required before execution.

## Inappropriate use

- Automatic campaign changes without operational review.
- Causal claims from platform attribution or observational correlation alone.
- Extrapolation beyond the configured spend range.
- Use with real customer or employer data without privacy, legal, and security review.
- Treating the synthetic scenario multipliers as forecasts.

## Monitoring required in production

Track experiment coverage, curve error and bias, marginal-return ordering,
budget utilization, constraint overrides, allocation turnover, and realized
incremental profit. Re-estimate when the response relationship or channel
operating range changes materially.

