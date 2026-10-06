# LB-13 robustness: equilibrium_like

- config path: `configs/robustness/equilibrium_like_robustness.json`
- artifact root: `results/robustness/equilibrium_like`
- baseline window: `[0.75, 1.0]`
- per-variant windows:
- `baseline_fixed_manual_uniform`: candidate_window=[0.75, 1.0] stable=True
- `adaptive_manual_uniform`: candidate_window=[0.75, 1.0] stable=True
- `matched_manual_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_spectral_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_diffusion_uniform`: candidate_window=[0.75, 1.0] stable=True
- `fixed_manual_prototype`: candidate_window=[0.75, 1.0] stable=True
- `fixed_manual_stationary`: candidate_window=[0.75, 1.0] stable=True
- family-level verdict: `robust enough to continue`
- affinity note: `Aff remains near equilibrium scale for this family`
