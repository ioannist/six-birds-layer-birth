# LB-16 class-II scaling campaign

- campaign config path: `configs/campaigns/class_ii_driven_scaling.json`
- artifact root: `results/campaigns/class_ii_driven`
- primary panel: driven_cycle_family low-bias sizes [8,16,32,64]
- shadow panel: driven_cycle_family high-bias sizes [8,16,32,64]
- fitted parameters: `{'lambda_c': 0.8, 'beta_over_nu': 0.0, 'inv_nu': 0.5, 'objective': 0.027327045960924553}`
- class-I comparison: `{'class_i_lambda_c_fit': 0.85, 'class_i_beta_over_nu_fit': 0.0, 'class_i_inv_nu_fit': 0.5, 'class_i_collapse_objective': 0.002349921744986759, 'class_ii_lambda_c_fit': 0.8, 'class_ii_beta_over_nu_fit': 0.0, 'class_ii_inv_nu_fit': 0.5, 'class_ii_collapse_objective': 0.027327045960924553, 'class_i_affinity_window_mean': 1.2379511121410735e-18, 'class_ii_affinity_window_mean': 0.010117349957641894, 'class_ii_shadow_affinity_window_mean': 0.14134819402621182, 'delta_lambda_c_fit': -0.04999999999999993, 'delta_beta_over_nu_fit': 0.0, 'delta_inv_nu_fit': 0.0, 'affinity_contrast_ratio': 1011734.9957641894, 'affinity_contrast_difference': 0.010117349957641892, 'class_separation_plausible': True, 'diagnosis': 'driven/equilibrium separation appears plausible under affinity and fit diagnostics'}`
- candidate class-II collapse obtained: `yes`
- class separation looks real enough to keep pursuing: `yes`
