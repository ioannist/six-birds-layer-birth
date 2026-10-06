# Six Birds: Layer-Birth Taxonomy

This repository contains the manuscript, result bundles, and supporting code for the paper:

> **To Classify a Stone with Six Birds: Primitive Activation and Observable Separation of Layer-Birth Classes**
>
> Version 2 (October 3, 2026): https://doi.org/10.5281/zenodo.23118249
>
> Version 1 (March 16, 2026): https://doi.org/10.5281/zenodo.19047665

This paper develops a taxonomy of experimentally accessible layer-birth events within the Six-Birds framework. The core contribution is not a final exponent-level universality classification. It is a primitive taxonomy and observable separation of four canonical low-complexity layer-birth classes, organized by activation of packaging ($P5$), directed circulation ($P6_{\mathrm{drive}}$), and staging anomaly ($P4$).

## What this repository provides

This repository includes:

- **Manuscript sources**: the complete paper under `paper/`, including appendices and bibliography
- **Reproducible result bundles**: structured outputs under `results/` for the canonical rubric, observables, class confirmation, phenomenology, robustness, capacity, and upstream bridge assets
- **Analysis code**: the `src/layerbirth/` package that generates and summarizes the experiment bundles used by the manuscript
- **Observable-map synthesis**: the four-class observable map and supporting dashboards used to make the taxonomy visible in measured observable space
- **Evidence-ledger layer**: a reproducibility-freeze bundle under `results/freeze/reproducibility_freeze/` that records claim coverage and manuscript-supporting assets in machine-readable form
- **Build contract for the paper**: the manuscript build writes all outputs to `paper/build/`, including the compiled PDF and a flattened TeX source suitable for archival workflows

## Scope and limitations

The paper is explicit about what it does and does not establish:

- The main result is a primitive taxonomy and observable separation, not a final universality-exponent classification
- Class-I and Class-II are the strongest empirical result, separated robustly by the affinity / drive channel and validated by no-fake-arrow controls
- Class-III and Class-IV are confirmed in the canonical manual channel (exact rational certificates at sizes 16–128; the undriven Class-III family is proved size-independent), but lose their staging label under the two automatic lenses tested
- Capacity/depth proxies are not classifiers: under the packaging control the capacity index reduces to a rescaling of the structural boundary
- The upstream/PICA bridge places three kernels built from compiled upstream features (not native upstream dynamics) on the map; all land in Class-I
- Finite-size-scaling fits were explored, but current exponent estimates remain too grid-sensitive and boundary-attached to support strong asymptotic claims

## Install

```bash
make install
```

## Test

```bash
pytest
python -m layerbirth.smoke
```

## Run selected workflows

```bash
python scripts/run_class_iii_class_iv_shadow_panel.py
python scripts/run_upstream_smoke.py
```

The upstream smoke run and the PICA bridge test expect a local checkout of
[six-birds-pica](https://github.com/ioannist/six-birds-pica) at `vendor/six-birds-pica/`;
vendored upstream code is not tracked in this repository.

## Build paper

```bash
.venv/bin/python scripts/audit_canonical_mathematics.py   # exact rational certificates (optional)
.venv/bin/python scripts/make_paper_figures.py           # regenerate paper/figures/
make paper
```

The current manuscript is version 2 (October 3, 2026); its version history is in Appendix E.

Build outputs are written to:

```text
paper/build/main.pdf
paper/build/main_flat.tex
```
