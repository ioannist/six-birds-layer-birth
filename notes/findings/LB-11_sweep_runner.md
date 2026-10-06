# LB-11 Sweep runner

- Added deterministic sweep runner in `layerbirth.sweep` with:
  - grid expansion over family/size/seed/lambda/lens/lift/tau
  - operational control hook `mix_with_packaging_projector`
  - stable run IDs from canonical spec hash
  - run-level and sweep-level manifest-style bundles
  - cache behavior (`executed` vs `cached`)
- Tiny grid config (`configs/sweeps/tiny_grid.json`) spans:
  - families: reversible block + metastable block
  - size: 8
  - seeds: null, 7, 11
  - lambda: 0.25, 0.75
  - lens: manual family block lens
  - lift: uniform
  - tau protocol: fixed tau=1
- Stable aggregate metrics schema fields:
  - `sweep_id, run_id, family_name, size, seed, closure_strength_lambda, control_application_name, lens_name, lift_name, tau_protocol_name, resolved_tau, analysis_k, closure_error, objecthood_order, staging_gap, affinity, holonomy, cache_status, manifest_path`
- Rerun cache observation:
  - first pass executes all runs
  - second pass returns clean cache hits for all runs

Real caveat:
- The current run-note/env payloads are intentionally minimal and operational, not full provenance snapshots.
