# Metric Dictionary

| Metric | Definition | Decision use | Limitation |
|---|---|---|---|
| Measured incremental contribution | Noisy synthetic estimate of contribution created by spend, before marketing cost | Fits response curves | Real use requires experiments or another credible incremental design |
| Context index | Seasonal and planned-event multiplier | Separates spend response from known demand context | Simplifies real demand and competitive effects |
| WAPE | Sum of absolute errors divided by sum of absolute actual values | Selects and evaluates response curves | Can hide error differences between small and large cells |
| Signed bias | Sum of prediction error divided by sum of actual values | Detects systematic over- or under-prediction | Positive and negative weekly errors may cancel |
| Incremental contribution ROI | Incremental contribution divided by spend | Compares policy efficiency | Excludes fixed cost and nonfinancial objectives |
| Incremental profit | Incremental contribution minus spend | Primary economic outcome | Depends on the contribution definition being consistent across cells |
| Regret versus oracle | Oracle contribution minus policy contribution, divided by oracle contribution | Measures distance from simulated perfect information | Oracle parameters do not exist in production |
| Allocation turnover | Half the absolute spend movement from the feasible historical allocation, divided by total budget | Indicates operational disruption | Does not price the actual cost of changing campaigns |
| Marginal return | Additional predicted contribution from the next spend segment divided by segment spend | Drives the optimizer | Reliable only inside the observed operating range |

