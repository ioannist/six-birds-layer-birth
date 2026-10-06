# LB-08 Equilibrium-like substrate families

- Implemented families in `layerbirth.substrates`:
  - `reversible_block_family` (deterministic, symmetric block-ring construction)
  - `metastable_block_family` (seeded heterogeneous symmetric/reversible construction)
  - `null_flat_mixing_family` (uniform complete mixing null)
- Config examples added under `configs/substrates/`:
  - strong/weak reversible separation pair
  - seeded metastable reversible block
  - null flat mixing
- Smoke summary written to:
  - `results/substrate_smoke/equilibrium_like/summary.csv`
  - `results/substrate_smoke/equilibrium_like/summary.json`

Why each family matters:
- Deterministic reversible block is the clean class-I baseline because separation is controlled directly by `inter_block_weight` while preserving reversibility.
- Seeded metastable block is the heterogeneous reversible control because it introduces reproducible irregularity without introducing irreversible driving.
- Null flat mixing is the correct no-SG control because all non-leading eigenvalues are zero, giving default `SG = 0` for `k >= 2`.

Preferred class-I baseline:
- `reversible_block_family` with the strong-separation config.

Real caveats:
- Spectral metrics can be sensitive when separation is weak or near-degenerate.
- `metastable_block_family` is reversible by symmetric construction, but finite precision can produce tiny nonzero Aff values near machine tolerance.
