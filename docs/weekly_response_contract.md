# Weekly Response Input Contract

`--input-weekly-response` accepts an aggregate CSV. It does not accept
customer-level records or platform-attributed revenue presented as causal lift.

## Required grain

One row per `week_start × cell_id`. The file must be a balanced weekly panel:
every decision cell must appear in every week, dates must be seven days apart,
and a `cell_id` must always map to one category and one channel. Each cell needs
at least three distinct pre-validation spend levels and positive contribution
totals in both the validation and test periods.

With the default temporal settings, the file needs more than 44 complete weeks.
The exact rule is:

```text
total weeks > test_weeks + 24
pre-test weeks > validation_weeks + 8
```

## Required columns

| Column | Type and rule | Meaning |
|---|---|---|
| `week_start` | Date, no time component | Start of the evaluation week |
| `cell_id` | Non-empty string | Stable decision-cell key |
| `category` | Non-empty string | Category constraint group |
| `channel` | Non-empty string | Channel constraint group |
| `spend` | Finite number, `> 0` | Weekly marketing cost in one consistent currency |
| `context_index` | Finite number, `> 0` | Pre-specified demand/context multiplier; use `1.0` if none is approved |
| `measured_incremental_contribution` | Finite number, `>= 0` | Experiment-informed contribution before marketing cost, in the same currency as spend |
| `evidence_type` | Controlled string | Identification design behind the incremental outcome |
| `evidence_reference` | Non-empty string | Experiment registry ID, analysis version, or other traceable source reference |

Accepted `evidence_type` values:

- `randomized_experiment`
- `geo_experiment`
- `causal_estimate`
- `synthetic_fixture` — accepted only for interface testing and reported as
  `demonstration_only`

Last-click, platform-attributed, or observational revenue is not accepted under
another label. A real run is decision-eligible only when every row uses one of
the first three evidence types and the identification assumptions are documented
outside this repository.

## Validation and output boundary

The loader rejects missing columns, duplicate cell-weeks, unstable cell mappings,
gaps in the weekly calendar, incomplete panels, invalid numeric values, unsupported
evidence types, and insufficient temporal history.

For supplied data, the pipeline reports fitted-model planning estimates and omits
the simulation oracle and regret metrics. The recommendation still requires human
review and a future randomized holdout. Input rows are copied only to the ignored
`data/generated/` directory; they must not be committed.

The committed
[`weekly_response_fixture.csv`](../data/sample/weekly_response_fixture.csv) is a
synthetic schema fixture for CI. It is not evidence of real marketing performance.
