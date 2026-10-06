# LB-03 upstream smoke

- Chosen vendored baseline path: `vendor/six-birds-pica-ma_snapshot_v71/`.
- Wrapper path: `scripts/run_upstream_smoke.py`.
- Exercised upstream functionality: `closurelab.numeric` path (`build_Q`, `build_U_uniform`, `empirical_endomap`, `macro_kernel`, `idempotence_defect_tv`) on a deterministic toy transition kernel.
- Caveats: wrapper expects `numpy` to be importable in the execution environment; no vendored code was modified.
