# LB-15 class-I scaling campaign

- campaign config path: `configs/campaigns/class_i_equilibrium_scaling.json`
- artifact root: `results/campaigns/class_i_equilibrium`
- primary panel: reversible_block_family sizes [8,16,32,64]
- shadow panel: metastable_block_family sizes [8,16,32,64], seeds [7,11,13,17,19,23]
- fitted parameters: `{'lambda_c': 0.85, 'beta_over_nu': 0.0, 'inv_nu': 0.5, 'objective': 0.002349921744986759}`
- collapse diagnostics objective: `0.002349921744986759`
- shadow crossing summary: `{'pairwise_crossings': [{'size_pair': [8, 16], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}, {'size_pair': [8, 32], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}, {'size_pair': [8, 64], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}, {'size_pair': [16, 32], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}, {'size_pair': [16, 64], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}, {'size_pair': [32, 64], 'lambda_interval': [0.95, 1.0], 'crossing_lambda': 1.0}], 'crossing_lambda_c_mean': 1.0, 'crossing_count': 6, 'size_pairs': [[8, 16], [8, 32], [8, 64], [16, 32], [16, 64], [32, 64]]}`
- candidate collapse obtained: `yes`
- recommend proceed to class-II campaign: `go`
