# LB-17 holonomy crossover campaign

- config path: `configs/campaigns/holonomy_crossover.json`
- artifact root: `results/campaigns/holonomy_crossover`
- cases: balanced_direct, balanced_routed, unbalanced_direct, unbalanced_routed
- summary metrics: `{'balanced_direct': {'case_name': 'balanced', 'control_route': 'direct', 'mean_holonomy': 0.0, 'mean_affinity': 0.0, 'max_closure_error': 0.029296875, 'birth_window': [None, None], 'birth_location_estimate': None, 'window_present': False, 'max_window_score': 0.018671874999999925}, 'balanced_routed': {'case_name': 'balanced', 'control_route': 'routed', 'mean_holonomy': 0.0, 'mean_affinity': 0.0, 'max_closure_error': 0.029296875, 'birth_window': [None, None], 'birth_location_estimate': None, 'window_present': False, 'max_window_score': 0.018671874999999925}, 'unbalanced_direct': {'case_name': 'unbalanced', 'control_route': 'direct', 'mean_holonomy': 0.25, 'mean_affinity': 0.0, 'max_closure_error': 0.029296875, 'birth_window': [None, None], 'birth_location_estimate': None, 'window_present': False, 'max_window_score': 0.018671874999999925}, 'unbalanced_routed': {'case_name': 'unbalanced', 'control_route': 'routed', 'mean_holonomy': 0.25, 'mean_affinity': 0.0016777810355067505, 'max_closure_error': 0.029296875000000003, 'birth_window': [None, None], 'birth_location_estimate': None, 'window_present': False, 'max_window_score': 0.018671874999999925}}`
- did holonomy behave like a crossover field rather than a fake arrow source? `no`
- family adequate for crossover campaign use: `no`
- diagnosis: `fails no-fake-arrow requirement: affinity contamination prevents clean crossover interpretation`
