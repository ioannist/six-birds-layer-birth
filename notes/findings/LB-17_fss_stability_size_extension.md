# LB-17 FSS stability and size-extension audit

- config path: `configs/campaigns/fss_stability_size_extension.json`
- artifact root: `results/campaigns/fss_stability_size_extension`
- class-I grid sensitivity: `{'grid_spec_sensitive': True, 'shifts': {'orig_to_ident': {'lambda_c': 0.07500000000000007, 'beta_over_nu': 0.0, 'inv_nu': 0.25}, 'orig_to_union': {'lambda_c': 0.15000000000000002, 'beta_over_nu': 0.0, 'inv_nu': 0.5}}, 'original_steps': {'lambda_c': 0.04999999999999993, 'beta_over_nu': 0.024999999999999994, 'inv_nu': 0.25}, 'boundary_flip': False}`
- class-II grid sensitivity: `{'grid_spec_sensitive': True, 'shifts': {'orig_to_ident': {'lambda_c': 0.07499999999999996, 'beta_over_nu': 0.0, 'inv_nu': 0.25}, 'orig_to_union': {'lambda_c': 0.07499999999999996, 'beta_over_nu': 0.0, 'inv_nu': 0.5}}, 'original_steps': {'lambda_c': 0.04999999999999993, 'beta_over_nu': 0.024999999999999994, 'inv_nu': 0.25}, 'boundary_flip': False}`
- class-I size128 extension: `{'size128_helpful': False, 'plateau_reduction_frac': -3.0, 'inv_span_reduction_frac': 0.0, 'beta_span_reduction_frac': 0.0, 'boundary_release': False, 'contrast_improve_frac': 0.05199099809891011}`
- class-II size128 extension: `{'size128_helpful': False, 'plateau_reduction_frac': -0.25, 'inv_span_reduction_frac': -0.5, 'beta_span_reduction_frac': 0.0, 'boundary_release': False, 'contrast_improve_frac': 0.0}`
- are the current exponent fits grid-specification sensitive? `yes`
- does adding size 128 materially stabilize the FSS story? `no`
- should the paper currently make exponent claims? `no`
- affinity-separation primary distinction remains supported: `True`
