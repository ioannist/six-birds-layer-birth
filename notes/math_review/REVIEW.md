# Mathematics review — completed within the declared operational scope

Scope: thoroughly review and repair the repository mathematics and mechanizations;
preserve the claims where possible; discuss a material main-claim downgrade before
adopting one. The paper was not edited during the review.

The clean pre-review repository was checkpointed in commit `12a41bd`. Top-level
`lean/` contains only `.gitkeep`, but the vendored upstream snapshot has a Lean
library at `vendor/six-birds-pica-ma_snapshot_v71/lean`. The initial review missed
that library; the follow-up reviewed and built it explicitly with
`lake build ClosureLean` (the unqualified build has no default target).
The three retraction/idempotency declarations compile under Lean 4.27.0 and
`notes/math_review/lean_axioms.lean` verifies that none depends on axioms.
These associativity proofs establish idempotency of a retraction composite.
They do not prove idempotency of arbitrary empirical `P^tau Q U`, the taxonomy,
or the rational audit below. Do not call those Python computations Lean proofs.

Methods applied: `mathematics-proof-review` and `six-birds-proof-construction`.
This has been a single-agent review with a distinct adversarial self-review pass;
it is not an independent human/agent referee report. An independent arithmetic
implementation is identified below, which is a different kind of independence.

## Repair findings and their witnesses

1. **Entropy production omitted one-way flux.** `affinity_metric` restricted both
   directions to fluxes greater than an arbitrary cutoff. A directed permutation
   cycle consequently had affinity zero, despite reversal divergence infinity.
   The repaired definition retains all positive represented flux. A reverse zero
   gives infinity; the cutoff remains only in optional diagnostics. Unordered
   edge-pair summation also removes negative cancellation artifacts near balance.

2. **Unconverged stationary iterates could fake an arrow.** The reversible chain
   `[[0,1,0],[1/4,0,3/4],[0,1,0]]` produced affinity `0.231049...` with 20 ordinary
   power steps instead of its stationary value zero. Its correct law is
   `(1/8,1/2,3/8)`. A common communicating-class/absorption solver now supplies the
   stationary law, handles periodic/reducible support, assigns transient states
   zero mass, and checks the balance residual. A strictly positive kernel can
   still use the converged power shortcut. Invalid/nonstationary results raise
   rather than passing unverified iterates into observables. This remains floating
   computation with a stated residual tolerance, not an exact generic solver or a
   uniform relative-error certificate for ill-conditioned chains.

3. **Affinity at the wrong boundary and fallback substitution.** The taxonomy
   preferred the CE crossing even if the objecthood crossing was later; absent
   crossings were replaced by unrelated mean/supplied affinities. It now measures
   at `max(CE crossing, objecthood crossing)` and preserves unavailable evidence
   as unknown instead of certifying drive absence. Missing channels do not become
   evidence of absence when candidate affinities are aggregated across tau.

4. **Separate structural hits were mistaken for joint birth.** On a nonmonotone
   curve, CE could hit early and then fail when objecthood hit later. The sampled
   piecewise-linear reference now checks both thresholds together. A supplied
   summary without this extra field remains an explicit input assertion; fresh
   curve-derived summaries populate and verify it.

5. **Approximate equality corrupted boundary interpolation.** `np.isclose` could
   replace a small asymmetric crossing by the interval midpoint. The common
   threshold helper uses the actual linear fraction. Duplicate coordinates and
   NaN input are rejected. A hit at the first sampled point is a left-censored
   proxy, not a resolved continuum crossing.

6. **Lift ablations measured a different lift.** Sweeps, pilots, robustness runs,
   and scaling campaigns selected a lift for the control but used uniform lifts
   for CE, objecthood, and macro relaxation measurements. The metric APIs now
   accept an explicit validated lift and these workflows pass the selected
   matrix. Canonical uniform-channel results retain their original meaning.

7. **Moment normalization was inconsistent.** An absolute small-denominator
   cutoff zeroed the Binder ratio of any sufficiently small nonzero observable.
   The ratio is now computed after scaling the sample, retaining scale invariance.
   The exactly zero sample retains a documented undefined-ratio fallback. The
   shadow susceptibility used replicate count rather than system size; it now
   follows the common `N Var(m)` convention. Duplicating an unchanged empirical
   distribution consequently does not change susceptibility.

8. **Invalid inputs could masquerade as legal operations.** A nonstochastic
   pushforward could pass `UQ=I`; fractional lenses were silently integer-cast;
   tiny positive weight rows were discarded; and a schedule containing all
   required phases plus a duplicate passed its “exactly once” test. The repaired
   validators and schedule check have counterexample regressions. Rectangular
   kernels and noninteger tau are rejected at the power operation.

9. **No-fake-arrow aggregate gate was vacuous on missing controls.** A hidden
   positive alone could pass because missing reversible/phase control lists had
   zero false positives. The gate now requires the named negative control cases
   and actual candidate readouts.

The dependency declaration now includes the plotting library used by the
experiment workflows. The manifest uses the executing Python interpreter rather
than assuming an unrelated `python` binary exists on PATH. These repairs permit
the required numerical verification in a fresh local environment.

## Constructive return to the canonical claims

The represented target is the declared finite-grid operational taxonomy with
equal two-block manual lenses and uniform lifts. It is not an asymptotic
universality claim, a proof of arbitrary lens robustness, or a theorem identifying
latent physical sectors from tau sensitivity.

The carrier is a finite row-stochastic kernel `P`, deterministic surjective lens
matrix `Q`, and stochastic fiber-supported lift `U` with `UQ=I`. The legal control
in these certificates is `P_lambda=(1-lambda)P+lambda QU`. The observable interface
is `E=P_lambda^tau Q U`, CE as maximum row TV between `E` and `E^2`, normalized
prototype retention for objecthood, and a microstate stationary reversal audit
for drive. All four representatives below use explicit native kernels, not
records populated with desired class conclusions.

The Six Birds packaging definitions were checked against the local corpus source
`Tsiokos_2026_To_Lay_a_Stone_with_Six_Birds_Finite_State_Semantics_for_Packaging_Directionality_and_Coarse_Graining.tex`,
especially `def:empirical-packaging` and `prop:sigma-expansion`. The latter is not
being imported as a generic certificate for arbitrary initial distributions:
the following entropy argument uses a matched stationary law explicitly.

### Packaging reduction (derived)

Set `A=P_lambda^tau Q` and `B=UA`. Then `E=AU` and `E^2=ABU`. Because the rows of
`U` have disjoint fiber support and mass one, every macro signed row `v` satisfies
`||vU||_1=||v||_1`. Therefore

`CE = (1/2) max_i ||A_i-(AB)_i||_1`.

For two equal blocks, writing `a_i=A_i0`, `b0=mean(a_i on block 0)`, and
`b1=mean(a_i on block 1)`, the exact formulas reduce to

`CE = max_i |a_i-a_i b0-(1-a_i)b1|`,

`M_obj = clip(b0-b1, 0, 1)`.

The retention equality follows from
`TV(B_x,delta_x)=1-B_xx`; hence the soft stable count is `trace(B)` and its
two-object normalization is `clip(trace(B)-1,0,1)`. This supplies the exact return
from the reduced arithmetic to the original implemented definitions. No raw
nonlinear observable was inserted into a linear adequacy/Gram residual calculus.

### Stationary reversal lower bound (derived)

Let `J_ij=pi_i P_ij`. Both `J` and `J^T` are probability distributions. The
stationary reversal divergence is

`Aff = KL(J || J^T) = sum_(i<j) (J_ij-J_ji) log(J_ij/J_ji)`.

Positive one-way flux gives infinity. For bidirected flux, each pair is
nonnegative. Writing `x=a/b >= 1`, the derivative of
`log(x)-2(x-1)/(x+1)` is `(x-1)^2/[x(x+1)^2] >= 0`, and its value at one is zero.
Swapping `a,b` covers the other ordering, so

`(a-b) log(a/b) >= 2(a-b)^2/(a+b)`.

The independent rational constructors check exact row and column sums equal one,
so `pi_i=1/n` is an established stationary law, including reducible cases. This
provides an exact rational lower bound for affinity without numerical logarithms.
Symmetric kernels have exact zero affinity. The affine interpolation of the
sampled lower bounds bounds the declared interpolated affinity readout; it is not
a claim about an unmeasured continuous metric curve at an interpolated lambda.

### Current exact evidence

Reproduce with:

```sh
.venv/bin/python scripts/audit_canonical_mathematics.py
```

The independent implementation is `src/layerbirth/exact_audit.py`. The script reads
the actual decimal configurations as rationals, checks the rubric thresholds,
constructs the native kernels independently, computes every CE/objecthood grid
point in both tau channels, checks joint structural activation, and compares the
results with fresh production metrics. It does not load campaign result caches.
Exact fractions, configuration hashes, and comparison errors are persisted in
`notes/math_review/canonical_exact_audit.json`.

| Representative | Sizes verified | Exact justified conclusion |
| --- | --- | --- |
| Class-I reference | 32 | Packaging active, exact reversibility, dual shift < 0.15 |
| Class-II reference | 32 | Packaging active, affinity lower bound > 0.034, dual shift = 0 |
| replicated_portal_sp4 | 16, 32, 64, 128 | Packaging active, exact reversibility, dual shift > 0.2016 |
| replicated_portal_sp4_drive_f | 16, 32, 64, 128 | Packaging active, affinity lower bound > 0.001 at every size, dual shift > 0.1825 at every size |

At size 64 the Class-IV lower bound is approximately `0.00296401848`; its dual
shift is approximately `0.21568034325`. The largest observed production CE or
objecthood error versus the exact grid is below `3e-15`. These are ten explicit
finite certificates. They do not establish all-size Class-IV persistence. The
Class-III equality across the checked sizes now has an algebraic return in
`replicated_portal_reduction.md`: strong lumpability to four state types, fixed
3:1 type proportions in each uniform lift, and symmetry transport its exact
operational signature to every allowed `n=8m>=16`. This concerns the exact
undriven mathematical family, not an all-size floating error certificate or an
extension of the driven Class-IV bound.

### Non-descent and audit-carrier discipline

Micro affinity does not descend through every lens. A biased four-cycle with
parity lens has positive micro affinity and a reversible two-state macro kernel.
The regression witness exercises this actual distinction. The metric bundle
explicitly records its affinity carrier as the microstate kernel. A micro audit
can legitimately classify the declared experiment if that audit is part of the
observable interface; it must not be called a reconstructed macro-only arrow.

The protocol control checks the ordered product against its constituent
reversible phase kernels. It does not build a full autonomous phase-register
process. Reversibility of each phase does not prove reversibility of a scheduler
or its autonomous lift. Keep this finite control interpretation visible in any
later paper edits. No general phase-aware zero-arrow theorem was verified here.

Shadow ensembles are parameter-perturbation distributions, not a thermodynamic
equilibrium ensemble. `N Var(m)` and the moment ratio are operational proxies;
their repair does not establish fluctuation-dissipation or universal exponents.

## Follow-up repairs

- **Cache identity and prerequisite provenance.** All row computation identities
  now include full generating panel/variant specifications and a package,
  repository-configuration, orchestration-script, vendored Python-source, declared dependency-version, Python, and NumPy fingerprint. Sweep cache reuse also
  verifies its manifest and input snapshot, even if a caller reuses an external
  run ID. Prerequisite consumers check enclosing manifests rather than directory
  existence. The fingerprint is process-scoped; restart after changing sources.
  This protects against stale implementation/configuration replay, not arbitrary
  corruption of cached numerical values, the contents of untracked input data, or
  a complete attestation of operating-system/BLAS/compiler behavior. Known
  prerequisite producers now check the expected configuration snapshot too; a
  reduced test panel cannot be replayed as its requested full default panel.
- **Fresh common-size map.** The observable map recomputes its coordinates from
  current canonical configurations at the recorded size. It derives actual class
  labels, records the measurement channel and micro affinity carrier, marks
  left-censored boundaries, and pairs tau1 affinity with the displayed tau1
  boundary. Candidate max-over-tau affinity remains a separately named quantity.
  Onsets use the declared interpolation rule throughout. Missing/nonfinite or
  negative values, duplicate/missing classes, and contradictory class labels do
  not pass the publication gate. Size-64 I/II coordinates differ from the old
  size-32 values; the four freshly measured signatures still separate.
- **Persistence and lens comparison.** One size cannot confirm scale persistence,
  duplicate/empty panels are invalid, declared panel coverage is checked, and CV
  is scale invariant. Persistent hits with unstable shifts are named separately
  from failures of persistence. A persistent near-miss requires its objecthood
  shift at every size, rather than at one favorable size. Lens evaluations share
  the same unknown-aware candidate classifier; comparisons use the actual manual
  signature at each size and verify the manual label before calling reproduction
  robust. Contradictory synthetic tests were corrected, not used as a rubric.
- **Capacity/depth.** Birth references are recomputed per size with the same
  representative kernel; single-channel fallback is not joint birth. Missing
  times and failed earlier samples terminate sampled-prefix depth survival.
  Objecthood area is a normalized trapezoidal integral over the actual tau span.
  P5/P6 retention is explicitly named and makes no P4 retention claim. Axis
  contrasts use sizes shared by all four classes. Unavailable plot points are
  gaps rather than zero. The added `capacity_depth_summary.json` records the
  algebraic saturation dependence: under projector mixing at lambda one, every
  positive tau gives CE=0, objecthood=1, affinity=0, and the saturation index is
  `(1-birth_reference)*max(sampled_tau)`. These are dependent operational proxies,
  not independent capacity observations or a capacity theorem.
- **Upstream feature bridge.** The requested source root and compiler entrypoint
  are respected and verified; feature dimensions, weights, sector labels, and
  drive mixing are validated. The kernel is explicitly identified as constructed
  from compiled upstream Gram/sector features and an artificial preference
  gradient, with diagram/compiler hashes. It is not native upstream dynamics.
  Classification derives from primitive evidence and rejects contradictory
  labels. Overlay readiness requires finite measured coordinates in the displayed
  channel. Repeated copies of the same diagram do not increase the support count;
  confidence remains an explicitly described heuristic, not statistical evidence.
- **Scaling and uncertainty scope.** The declared objective is explicitly a
  heuristic sum of pairwise curve error, raw-lambda variance, and a weighted slope
  penalty; it is not a pure collapse objective or an exponent certificate.
  Nonfinite/no-comparison fits are rejected. When raw observations identify seed
  trajectories, bootstrap resamples whole lambda trajectories within each size.
  Cross-size independence remains an assumption of that mode. Singleton grouped
  primary observations have no sampling uncertainty, and their degenerate grid
  interval is labelled conditionally rather than treated as exponent precision.
- **Freeze and synthesis gates.** Missing mapped assets and empty claim sets fail
  readiness. Freeze reruns set `LAYERBIRTH_FORCE_RECOMPUTE=1`, bypass row caches,
  and require the current fingerprint plus a recorded disabled-cache mode in the
  recomputed manifest. Stable files alone cannot verify recomputation. Synthesis
  labels derive from primitive states; distinct unknown signatures do not become
  four classes. Its I/II references are recomputed at its recorded size, and
  absent alignment evidence is not replaced by a positive default. Campaign
  drivers now use the running interpreter when refreshing prerequisites.

## Final aggregation and representation repairs

- Alignment uses the declared CE=0.025 / Mobj=0.9 targets, with no target
  substitution. Fewer than two observed comparable crossings are unavailable,
  not zero variation. Secondary affinity is measured at a joint structural
  reference; missing channels do not establish monotonicity or lens robustness.
- Visibility fits require at least two distinct finite coordinates. The R-squared
  calculation is scale invariant, including tiny nonlinear signals. Bias
  invariance checks intercepts as well as slopes. These are finite scan diagnoses,
  not a theorem asserting orthogonality for all kernels or lenses.
- P4 criterion selection requires its observed positive and negative profiles.
  Missing boundary fields are distinct from an explicit measured absent crossing.
  Unknown staging evidence remains an unknown activation state. The consolidated
  I/II publication gate now consumes the complete no-fake-arrow gate, rather than
  accepting zero false positives from an incomplete control panel.
- Shadow coverage counts distinct declared lambda points, verifies replicate
  counts and the configured size panel, and rejects duplicate points. Empty size
  panels cannot certify class preservation. Close distinct perturbations have
  distinct identifiers. The reported usable panel still concerns an artificial
  parameter-perturbation distribution, with the scope stated above.
- Capacity dashboard support requires measured finite coordinates for each of the
  four distinct classes and an available matched-size contrast. Merely having a
  directory, four row names and a permitted verdict string is insufficient.
- Synthesis spot checks use current candidate parameters. A diffusion tau1 P5/P6
  check no longer claims full P4/lens-label stability. Affinity consolidation
  recomputes I/II measurements at its recorded size and derives its verdict from
  evidence rather than writing an unconditional positive interpretation.
- Scientific artifact JSON writers use `serialization.py`: unavailable numeric
  observations become `null`, positive/negative infinity becomes the explicit
  string tag `"Infinity"`/`"-Infinity"`, and finite values remain numbers. Readers
  recover unknown as NaN and infinity as an extended observation, never zero.
  Config/computation hashing still rejects NaN. Numeric CSV readers preserve
  missing observations, and plotted unavailable crossing values are gaps.
  Empty Class-I affinity data cannot certify a driven/equilibrium comparison.
- Freeze conclusion checks are separate from output reproducibility. The default
  ledger declares the expected scientific conclusions in each relevant producing
  summary; a reproducible false conclusion blocks readiness. This verifies the
  declared finite analyses, not arbitrary new claims or a formal theorem.

## Verification and completion audit

The initial installed baseline was 198 passed / 1 failed (manifest interpreter
assumption). The first repair checkpoint reached a full 217-test pass and then
added the micro/macro carrier regression. The follow-up includes adversarial
regressions for stale computation replay, fresh freeze reruns, missing size and
claim coverage, contradictory labels, negative/unknown coordinates, per-size
births, sampled-prefix survival, irregular tau area, seed-trajectory resampling,
and the exact four-type reduction at sizes 16, 24 and 40.

A complete follow-up run of the 234 tests preceding the final reduction regression
passed. The final full run then passed all 235 tests in 115.38 seconds. A fresh
upstream execution found a missing PyYAML dependency that plumbing-only tests had
allowed to remain hidden. PyYAML is now declared and installed, and the bridge
execution test requires a usable positive result. The subsequent bridge/cache
regression run passed all eight tests. All three full upstream configurations
execute successfully (22, 50 and 54 CP states), provide finite overlay coordinates,
and measure Class-I in their constructed feature kernels. The fresh common-size
map remains separable and the corrected capacity result remains a secondary,
mixed operational field. Reproducible check results are recorded in
`followup_fresh_checks.json`. The fresh canonical audit also reproduces all ten
finite rational certificates. The Lean audit commands and their scope appear
above. The `.venv` is ignored. Tests rewrite some tracked `notes/findings` even
with temporary result roots; restore only those incidental generated-note edits
before committing.

No manuscript source has been changed. Historical generated bundles and old
freeze receipts are not a substitute for the fresh scope-specific arithmetic
certificates or the new provenance checks.

Final required-work audit:

| Obligation | Evidence and scope |
| --- | --- |
| Preserve the original state | Clean pre-review checkpoint `12a41bd`; original commit `4173012` preserved |
| Review the mathematical targets and exact returns | Metric definitions, TV-isometry/trace reduction, stationary reversal bound, and operational threshold returns above |
| Repair substantive implementation counterexamples | Core repairs in `36288e8`, provenance/support repairs in `f958c71`, final representation and aggregation repairs in the completion checkpoint |
| Maintain the canonical main claims | Ten independent rational finite certificates; undriven Class-III four-type proof extends to every allowed exact size |
| Review mechanizations | Explicit `lake build ClosureLean`, all three declarations with no axioms; no unsupported claim that the Python taxonomy is Lean-formalized |
| Audit supporting claims | Fresh alignment, visibility, identifiability/size-extension, controls, robustness, map, capacity and upstream executions; shadow panels checked on all declared sizes |
| Check unavailable evidence and claim gates | Scientific JSON/CSV round-trip, unknown primitive states, coverage/control counterexamples, and independent freeze conclusion checks |
| Verify repaired code | Final full integration run: 245 passed; exact certificates and explicit Lean audit reproduced |
| Preserve paper constraint | No manuscript or paper-side notes/figures edited by the review |
| Record limitations without smuggling | Exact vs floating, finite vs asymptotic, manual vs other lenses, micro vs macro audit, perturbation vs equilibrium ensemble, feature adapter vs native upstream dynamics, dependent capacity proxies and conditional bootstrap scope are explicitly separated |
| Review current evidence reproducibility | Full default production regeneration and 14-asset forced-computation freeze; final result recorded below |

The fresh size-extension audit still rejects exponent claims as grid-sensitive;
this agrees with the paper's existing exclusion of stabilized exponents. The
fresh alignment audit preserves canonical structural-boundary invariance and
secondary affinity activation but does not establish broader lens robustness.
Both Class-III and Class-IV shadow panels have finite observations, sufficient
replicates, and required lambda coverage at sizes 16, 32, 64, 128. The capacity
comparison remains mixed and secondary. The upstream result concerns the
explicit feature adapter, not recovered native upstream dynamics.

No required all-size Class-IV theorem, universal exponent certificate, generic
stationary-law relative-error theorem, or full non-manual P4 robustness theorem
is being deferred as if proved: those are outside the actual supported claim
scope. Historical scouting/refinement summaries are descriptive finite selections,
not selection-adjusted statistics or an exhaustive proof over parameter space.
The fresh evidence record, rather than historical generated receipts, is the
completion evidence for the declared finite claim set.

There is currently no evidence requiring a material downgrade of the four main
canonical operational class claims. The exact undriven Class-III family has a
stronger size-independent construction. If subsequent review finds a material
main-claim problem, develop restoration options and discuss them with the user
before adopting a downgrade. Preserve the unchanged-paper constraint throughout.

## Completion record

The final full-tree test run passed **245 tests in 64.33 seconds**. This follows
counterexample regressions for missing equilibrium affinity and incomplete
no-fake-arrow controls. The ten rational canonical certificates reproduced; the
explicit Lean library build completed five jobs and all three audited declarations
reported no axioms.

Fresh default controls, robustness, identifiability, size-extension and affinity
consolidation ran successfully, followed by every distinct production entrypoint
in the default freeze. All **14 assets** passed fresh output-hash reproducibility;
all declared scientific conclusion checks passed; all seven mapped claim groups
have their required evidence. Readiness is true with zero blocking issues and the
three retained scope caveats. This is a finite evidence readiness result, not a
formal proof of arbitrary future claims.

`completion_evidence.json` records the current implementation fingerprint,
production commands, exact certificate count, Lean scope, conclusion summaries,
full per-asset rerun hash ledger, and final test result. The integration suite can
rewrite shared generated result roots. The genuine frozen production outputs were
preserved before that run, restored afterward, and independently rechecked against
all 14 frozen output hashes, conclusion gates and current fingerprints. This
restoration removes test contamination; it does not substitute cached data for the
fresh reruns recorded in the freeze. Incidental generated `notes/findings` edits
were restored; the mathematical review and evidence records are the intentional
new documentation. The manuscript tree is unchanged.

The requested mathematical/implementation review is complete. No known required
repair is left pending for the declared canonical finite operational claims.
Future manuscript reconciliation, stronger asymptotic claims, a native-dynamics
upstream bridge, independent refereeing and a full Lean formalization of the
Python taxonomy are distinct work, not accomplishments being claimed here.
