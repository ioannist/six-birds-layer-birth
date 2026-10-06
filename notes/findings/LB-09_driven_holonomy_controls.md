# LB-09 Driven and holonomy-control families

Implemented families:
- `driven_cycle_family`
- `holonomy_control_family`

Config examples:
- `configs/substrates/driven_cycle_low_bias.json`
- `configs/substrates/driven_cycle_high_bias.json`
- `configs/substrates/holonomy_control_balanced.json`
- `configs/substrates/holonomy_control_unbalanced.json`

Smoke metrics summary:
- Driven low/high cases both produce `Aff > 0`, with high-bias larger than low-bias.
- Holonomy balanced/unbalanced cases share the same base reversible kernel and coarse lens.
- Balanced gives `Hol = 0`, unbalanced gives `Hol = 0.25`.
- `Aff` remains near zero for both holonomy-control cases.
- Coarse-lens metrics (`CE`, `M_obj`, `SG`) match across balanced/unbalanced to numerical tolerance.

Holonomy control verdict:
- **keep**
- The control changes route mismatch structurally (`Hol`) while leaving irreversibility (`Aff`) near zero and coarse-kernel metrics unchanged, so it cleanly isolates the intended effect.
