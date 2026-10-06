# LB-13 robustness: driven_low_bias

- config path: `configs/robustness/driven_low_bias_robustness.json`
- artifact root: `results/robustness/driven_low_bias`
- baseline window: `[0.5, 0.75]`
- per-variant windows:
- `baseline_fixed_manual_uniform`: candidate_window=[0.5, 0.75] stable=True
- `adaptive_manual_uniform`: candidate_window=[0.75, 1.0] stable=True
- `matched_manual_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_spectral_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_diffusion_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_manual_prototype`: candidate_window=[0.25, 0.5] stable=True
- family-level verdict: `robust enough to continue`
- affinity note: `Aff is sensitive; some variants approach non-driven levels`
