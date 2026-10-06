# LB-04 numeric backend

- Notation choice: canonical row-vector convention (`Q_f: (n,k)`, `U_f: (k,n)`, `E=P^tau Q_f U_f`, `P_hat=U_f P^tau Q_f`).
- Public API: `validate_lens`, `fiber_indices`, `pushforward_matrix`, `uniform_lift_matrix`, `prototype_lift_matrix`, `validate_lift_matrix`, `validate_row_stochastic`, `row_normalize`, `kernel_power`, `packaging_projector`, `macro_kernel`, `empirical_endomap`, `tv_distance`, `idempotence_defect_tv`, `retention_error`, `fiber_level_mismatch`.
- Toy A outputs: `idempotence_defect_tv=0.0`, `retention_error=(0.0,[0.0,0.0])`, `fiber_level_mismatch=([0.0,0.0],0.0)`, `macro_kernel=I_2`, `empirical_endomap=Q_f U_f`.
- Toy B outputs: `macro_kernel=[[0.5,0.5],[0.0,1.0]]`, `empirical_endomap=[[0.5,0.5,0,0],[0,0,0.5,0.5],[0,0,0.5,0.5],[0,0,0.5,0.5]]`, `tv_distance([1,0],[0.5,0.5])=0.5`, `idempotence_defect_tv=0.5`, `retention_error=(0.5,[0.5,0.0])`, `fiber_level_mismatch=([1.0,0.0],1.0)`.
- Independence: `src/layerbirth/numeric.py` does not import from vendored runtime code.
