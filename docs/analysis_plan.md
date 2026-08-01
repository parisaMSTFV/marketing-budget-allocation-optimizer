# Analysis Plan

## Decision

Allocate one planning-period marketing budget across category-channel cells to
maximize incremental contribution, subject to category floors and caps,
channel floors and caps, and cell-level operating ranges.

## Evidence design

The synthetic weekly outcome is an experiment-informed incremental contribution
estimate before marketing cost. Spend includes randomized exploration so the
curve fit is not based on a deterministic historical policy. The project does
not treat platform-attributed revenue as causal evidence.

## Temporal evaluation

1. Reserve the latest 20 weeks as an untouched test period.
2. Within the earlier history, use the latest 16 weeks for curve selection.
3. Compare linear, log, and saturation candidates using validation WAPE.
4. Refit the selected family on all pre-test weeks.
5. Evaluate the fitted curve once on the untouched test period.

Changing any outcome in the test period must leave every fitted parameter
unchanged. A unit test enforces this rule.

## Optimization

Selected nonlinear curves are converted to 24 diminishing-return segments per
cell. A linear program allocates the remaining spend above each operating floor.
It uses the same total budget and business constraints for every policy in a
scenario.

The primary comparison is true synthetic incremental profit for:

- the closest feasible historical mix;
- the closest feasible equal mix;
- the estimated-curve optimized mix;
- an oracle mix using hidden simulator parameters.

The oracle is an evaluation device, not a production feature.

## Pre-specified metrics

- Curve quality: WAPE, RMSE, and signed bias on the untouched test period.
- Policy value: incremental contribution, incremental profit, contribution ROI,
  profit ROI, regret versus oracle, and allocation turnover.
- Feasibility: full budget use, zero material constraint violations, and cell
  spend inside permitted ranges.

## Scenario assumptions

- Base: original budget and neutral demand context.
- Growth: 15% larger budget and explicit category demand multipliers.
- Conservative: 15% smaller budget, lower category demand multipliers, and a
  validation-error penalty.

All multipliers are synthetic planning inputs and are not forecasts.

