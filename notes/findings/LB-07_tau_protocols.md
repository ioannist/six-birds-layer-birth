# LB-07 tau protocols

- Protocols implemented: fixed, adaptive (`spectral_gap_scaled`), matched group resolver (reference/follower).
- Default adaptive formula: `tau_raw = ceil(alpha / max(spectral_gap, gap_floor))` with clipping to `[tau_min, tau_max]`.
- Continuous control interface: `closure_strength_lambda` validated as finite numeric in `[0,1]`.
- Sample resolutions on toy kernels:
  - fixed demo: `tau=3`, `closure_strength_lambda=0.35`.
  - adaptive demo (`P_slow`): `mu2_abs=0.8`, `spectral_gap=0.2`, `tau_raw=5`, `resolved_tau=5`.
  - matched demo: reference resolves `tau=5`; follower on faster kernel inherits `tau=5` despite standalone adaptive value `2`.
- Why matched is group-based: it enforces shared-`tau` provenance explicitly at comparison time and prevents hidden adaptive-`tau` confounds between compared conditions.
- Provisional choices: spectral-gap scaling is a pragmatic default; future tickets may want alternates (mixing-time heuristics, objective-coupled rules, or metric-driven tau selection).
