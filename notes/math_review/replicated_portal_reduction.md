# Exact reduction for the undriven replicated-portal family

This is an algebraic extension of the finite arithmetic witness, not a Lean
formalization or an all-size claim for the driven family. It concerns the exact
kernel specified by the constructor, with the manual two-block lens and uniform
lift. Floating implementations remain covered by the finite comparisons in the
canonical audit.

Let `n=8m`, `m>=2`. Each block contains `m` modules, each having three fast states
and one portal. Use nonnegative weights `f,g,c,s` for fast-fast, fast-portal,
portal-cross, and portal-self edges. Suppose

`D=max(2f+g, 3g+c+s)>0`.

The symmetric degree completion used by `replicated_portal_reversible_family`
adds `D-(2f+g)` to each fast diagonal and `D-(3g+c+s)` to each portal diagonal,
then divides by `D`. Pairing portals across blocks changes no row totals as `m`
varies. Group microstates into types `(fast0, portal0, fast1, portal1)` and let `H`
be the deterministic type pushforward. Every microstate of a given type has the
same probabilities to these four types. Consequently `P H=H T`, where

```
T = [1-g/D,  g/D,      0,       0;
     3g/D,   1-(3g+c)/D, 0,     c/D;
     0,      0,        1-g/D,   g/D;
     0,      c/D,      3g/D,    1-(3g+c)/D].
```

This row-stochastic matrix is independent of `m`. The self weight `s` still
enters through `D`; completion incorporates its contribution into the portal
stay probability.

Let `L` map the four types to their blocks, and let `V` lift each block to its
fast and portal types with probabilities `(3/4,1/4)`. The uniform micro lift `U`
satisfies `U H=V`, while the coarse lens is `Q=H L`. Therefore

`(Q U)H=H L V`,

`P_lambda H=H R_lambda`, with `R_lambda=(1-lambda)T+lambda L V`.

Induction gives `P_lambda^tau Q=H R_lambda^tau L` for every nonnegative integer
`tau`. Thus the rows of `A=P_lambda^tau Q` depend only on these four types and
are independent of `m`. Also

`U A=V R_lambda^tau L`.

The disjoint-support TV isometry established in `REVIEW.md` gives
`CE=(1/2)max_i ||A_i-(A U A)_i||_1`. All four types occur for every `m>=2`, so
this maximum is the same four-row maximum for every size. The normalized
objecthood equals `clip(trace(U A)-1,0,1)` and is independent of size too.
These equalities hold for the entire lambda interval, every stated tau, and
every sampled grid, including absence/presence of crossings.

Finally `P_lambda` is symmetric: both `P` and the uniform equal-block `Q U` are
symmetric. The uniform stationary law therefore gives reversal divergence zero
exactly, even when the undriven module kernel is reducible at lambda zero.
The decomposition into disconnected modules is not assumed away.

For the declared `replicated_portal_sp4` weights and lambda grid, the exact
size-16 certificate already establishes joint P5 activation, zero P6, and dual
P4 shift greater than `0.2016`. The equalities above transport that operational
signature to every `n=8m>=16` in this exact undriven mathematical family. They do
not extend the finite-size Class-IV drive bounds: adding the global directed
cycle changes the type transition probabilities at module/block boundaries.

Adversarial self-review: the return uses strong lumpability of `P_lambda` to
four types, rather than asserting descent of arbitrary micro observables to the
two-block lens. It explicitly transports the nonlinear TV/objecthood formulas
through their original arguments. The lift weights are fixed by the 3:1 type
counts in every block. No continuum boundary, universal exponent, arbitrary
lens, or macro-only directionality claim follows from this reduction.
