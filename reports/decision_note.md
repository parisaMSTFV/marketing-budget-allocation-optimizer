# Allocation Decision Note

## Recommendation

Use the constrained Base allocation as the next planning proposal. In the synthetic test environment it produces 10.2% more incremental profit than the feasible historical mix while using the same budget.

## Largest modeled moves

- Increase: **Grocery | Lifecycle CRM** by $66,513.
- Decrease: **Electronics | Lifecycle CRM** by $64,118.

## Guardrails before production use

- Treat the allocation as a planning recommendation, not an automatic spend change.
- Confirm inventory, eligibility, brand, and operational constraints outside this model.
- Preserve randomized holdouts so future response curves remain incremental.
- Require a documented override reason when planners change the recommendation.
