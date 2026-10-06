# LB-14 scaling analysis

- Implemented scaling-analysis utilities in `layerbirth.scaling`:
  - susceptibility
  - Binder-like cumulant
  - grouped observation summaries
  - pairwise crossing estimator
  - collapse objective
  - grid-search fit
  - bootstrap CI
  - report formatting
- Mocked-data config:
  - `configs/scaling/mock_fss.json`
  - true parameters: `lambda_c=0.50`, `beta_over_nu=0.125`, `inv_nu=1.0`
- Smoke bundle:
  - `results/scaling_smoke/mock_fss/`
- Recovered fit summary (smoke):
  - see `analysis/fit_summary.json` and `metrics/metrics.csv`
- CI summary:
  - see bootstrap intervals in `analysis/fit_summary.json`

Did the mocked-data fit recover the known parameters within tolerance? **yes**.
