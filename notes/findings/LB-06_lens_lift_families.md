# LB-06 lens/lift families

- Lens families implemented: `manual_partition_lens`, `spectral_sign_pattern_lens`, `diffusion_quantile_lens`, `random_surjective_lens`.
- Lift families implemented: `uniform_lift_family`, `prototype_lift_family`, `stationary_within_fiber_lift`.
- Deterministic defaults: canonical relabeling for all lenses, deterministic eigenvector sign convention for spectral families, deterministic quantile binning, deterministic `first_in_fiber` prototype strategy when explicit prototypes are absent.
- Smoke highlights: reversible toy yields `[0,0,1,1]` for both spectral families; prototype lift with `[1,3]` gives rows `[[0,1,0,0],[0,0,0,1]]`; stationary-within-fiber rows on `P_stat` are `[[0.8,0.2,0,0],[0,0,1/9,8/9]]`; representative `UQ_identity_error` values are at machine precision.
- Caveats: spectral family behavior can be unstable near eigenvalue/eigenspace degeneracy; stationary-within-fiber lift falls back to uniform within any zero stationary-mass fiber and records those fibers in details.
