# Does the tree search earn its place? (handover item 7)

Date: 2026-09-14. Branch `independent-core`. Reproduce with
`test/mcts/bench_search_strategy.py`; families are the shipped descriptions in
`data/model_search/`.

Handover open item 7 asked for MCTS to be compared with ordinary fitting at
equal budgets. It was never run. It has now been run, and it answers a
different and more serious question than the one it was asked.

## What was measured

Three strategies over the same problem transitions, so warm starting, the
native minimiser and rollback are identical and only the traversal differs:
`single` (evaluate the initial topology and stop — what **Fit** does today),
`enumerate` (reach every topology once by the shortest path from the root),
and `mcts` at budgets 8/16/32/200 over seven seeds.

## Searching is worth it. That was never in doubt.

| family | single | best found | gain |
|---|---|---|---|
| tcspc_lifetime | −711.8 (1 component) | −10.5 (2 components) | **+701** |
| fcs_two_species | −1882.3 (1 species) | −29.7 (2 species) | **+1853** |

The committed FCS fixtures are generated from the simplest topology, so the
root is already optimal and every strategy ties. Those rows say nothing about
search; `fcs_two_species` was added because of it.

## MCTS is not the problem, and neither is enumeration

| | evals | reward | agrees with enumerate |
|---|---|---|---|
| enumerate | 13 | −29.7 | — |
| mcts @ 8 / 16 / 32 | 6–8 | −112.0 | 0/7 |
| mcts @ 200 (default) | 11 | −98.5 | 0/7, and two different winners |

On `tcspc_lifetime` MCTS matches enumeration exactly at every budget. On
`fcs_two_species` it never does, and at the default budget it does not even
agree with itself across seeds. Note also that 200 simulations cost 11
evaluations, not 200: the tree caches, so an earlier worry that the default
budget was wildly oversized for an eight-topology space was simply wrong.

**But the comparison is not meaningful, because the quantity being compared
is not well defined.**

## The finding: a topology's score depends on how the search reached it

`fcs.3d.1diff.1relax`, same data, same free parameters, two orderings of two
moves that commute:

```
  -2640.919   use-3d-diffusion -> add-relaxation
   -111.974   add-relaxation -> use-3d-diffusion
```

**2528.9 apart.** `MultiStructureModelSearchProblem::apply_transition` warm
starts every parameter free in both parent and child from the parent's fitted
values, so a topology inherits whichever basin its path happened to land in.
A candidate therefore has no score of its own, only a score-given-a-route.

This is not an MCTS defect. Enumeration inherits it too — its "optimum" is
the best of one arbitrary path per structure, not the best of each structure.
It explains the disagreement above: the two strategies are not choosing
differently between the same candidates, they are scoring different local
optima and calling them by the same name.

`../chisurf/.omx/plans/mcts-all-fitting-models.md` predicted exactly this:
*"define candidate identity … and define initialization explicitly so caching
cannot make warm-start results traversal-dependent."*

## A second, narrower defect

On a curve built from two 2-D species, the generating topology
`fcs.2d.2diff.0relax` scores **worst of all eight** (−2909.7) while
`fcs.2d.2diff.1relax` scores best (−29.7). The two-component seeding —
`td2 = 4·td1`, `a1 = 0.5` — does not reach the right basin, and a spurious
relaxation term buys the freedom to compensate. Model selection over this
family currently prefers a wrong model for a seeding reason.

## What was done about it (2026-09-14, same day)

**Warm starting is off by default.** `MultiStructureModelSearchProblem::set_warm_start`
exists and defaults to `false`, so every structure is fitted from the seeds it
declares. The spread above is now **exactly 0.000**. Turn it on only to refine
a single known topology, never to choose between topologies.

**Every declared start is tried and the best kept.**
`add_structure_start` takes further starting points, and a description
declares them under `"starts"`. Because the starts are declared rather than
inherited from a route, the best of them is still a property of the model and
the data. The analytical FCS two-component topologies now declare brackets
either side of the one-component estimate instead of assuming `td2 = 4·td1`.

Effect on the two-species curve, where the generating model is
`fcs.2d.2diff.0relax`:

| | reward | rank of the generating model |
|---|---|---|
| before | −915.3 | 8 of 8, the worst |
| canonical seeds only | −915.3 | 5 of 8 |
| canonical + bracketing starts | **−25.6** | **3 of 8** |

All four two-component topologies now sit within 5 reward of each other, which
is the right shape — they are near-equivalent fits separated by their BIC
penalties.

**What it cost.** On the committed fixtures no selection outcome changed. TCSPC
rewards are identical to the last digit; only the component *labels* permuted
between slots, which is the sorting contract `mcts-generalization-cleanup.md`
asks for and nothing here provides yet. FCS rewards are 0.004 to 1.0 worse,
because warm starting genuinely helped on fixtures whose root is already
optimal. That is the price of comparability and it is small.

**What is still not solved.** A declared start set does not guarantee the
global optimum. Seeded at the truth the generating topology reaches −12.6 and
wins outright; the bracketing starts reach −25.6 and come third. Adding more
brackets (/2 /5 /20, and fraction variants) changes nothing — the basin is
narrow. This is a hard nonlinear-fitting problem, not a plumbing one, and
tuning seeds against one synthetic curve would be overfitting rather than
progress.

## The strategy question, re-run after the fix

With candidates scoring properly, the comparison finally means something, and
the answer is unambiguous:

| case | enumerate | mcts @ 8 | mcts @ 200 | agreement |
|---|---|---|---|---|
| tcspc_lifetime (3) | 4 evals, −10.515 | 3 evals, −10.515 | 3 evals, −10.515 | **7/7** |
| fcs_two_species (8) | 13 evals, −21.727 | 6.4 evals, −21.727 | 13 evals, −21.727 | **7/7** |

**MCTS finds the same answer as an exhaustive walk, every seed, every budget,
and reaches it in fewer evaluations.** The 0/7 disagreement measured before the
fix was an artifact of path-dependent scoring, not a defect of the search: the
two strategies were scoring different local optima and calling them by the same
name. The tree search is earning its place.

Two earlier worries are retracted. The default budget is not oversized — 200
simulations cost 13 evaluations, because the tree caches. And there is no case
here for replacing the search with enumeration; enumeration's value is as an
oracle for tests like these, not as the strategy.

## Seeding is the binding constraint, and it gets worse with size

The VV/VH/VM anisotropy family (`data/model_search/tcspc_anisotropy.json`,
24 canonical parameters, six topologies) is the sharpest case measured so far.
Against data it generated itself, Poisson-sampled:

| starts | picks the generating topology |
|---|---|
| one canonical start | 0 of 7 noise draws |
| two geometric spreads (x3, x6) | **3 of 7** |
| spread around the vm characteristic decay | 0 of 7 |

Seeded at the truth the generating topology wins outright (-450.7 against
-458.4 for its nearest rival), and the fitted parameters come back -- every
channel's intensity within 1%, r0 within 2%. So the *ranking* is right and the
*model* is right; what fails is reaching the optimum from a generic start. A
topology that does not converge scores in the tens of thousands and loses to
one that happens to.

Data-driven starts were tried and are worse than fixed geometric ones here.
Tuning the spread constant until the seven noise draws pass would be fitting
the test rather than the problem, so it has not been done.

This is the same finding as the FCS two-component case, at four times the
parameter count: **a declared start set does not scale to a coupled joint
fit.** Multistart raised FCS from -915 to -25.6 and anisotropy from 0/7 to
3/7; neither is a solution, both are evidence that the search needs a way to
*place* parameters before it compares topologies, not a better guess at where
they start.

## A family teaching itself where to start (2026-09-14)

Because the bottleneck is placement rather than choice, the self-play engine
(`ModelSearchSelfPlay`) learns starting points, not an action policy. A
description can already produce the curve it predicts, so any family -- one
added tomorrow as a file included -- generates its own episodes with no
hand-written simulator: sample parameters, simulate the measurement they
imply, keep the pair. Training uses the vendored MlpCore backward pass and
writes the JSON `NeuralNet` already reads, so there is one MLP
implementation and tttrlib owns it.

Measured on the two-species FCS curve, learning the five parameters
`fcs.2d.2diff.0relax` frees:

| target | MSE (unit scale) | proposal vs truth |
|---|---|---|
| `fcs.N` | 0.0006 | 1.976 against 2.0 |
| `fcs.baseline` | 0.0005 | 1.015 against 1.0 |
| `fcs.diffusion_time.1` | 0.015 | 0.130 against 0.08 |
| `fcs.diffusion_time.2` | 0.027 | 1.149 against 0.8 |
| `fcs.diffusion_fraction.1` | 0.054 | 0.55 against 0.4 |

Predicting the mean of these targets scores 0.083, which is the bar.

Those absolute numbers flatter it, though, and the comparison that matters is
against the start the proposal *replaces* -- because the declared seeds are
themselves computed from the curve:

| parameter | declared seed | proposed | truth |
|---|---|---|---|
| `fcs.N` | 2.063 (3% out) | 1.955 (2% out) | 2.0 |
| `fcs.baseline` | 1.012 (1% out) | 0.988 (1% out) | 1.0 |
| `fcs.diffusion_time.1` | 0.312 (**290% out**) | 0.130 (62% out) | 0.08 |
| `fcs.diffusion_time.2` | 1.247 (56% out) | 1.097 (37% out) | 0.8 |
| `fcs.diffusion_fraction.1` | 0.500 (25% out) | 0.547 (37% out) | 0.4 |

So the amplitude and the plateau were already right and the network adds
nothing there; its contribution is concentrated on the diffusion times, where
one characteristic lag had been standing in for both species. That is the
right place for it -- those are the parameters whose basin the search keeps
missing -- and a 4.7-fold improvement on the faster one is still not enough
to move the selection.

**It does not yet change the selection.** Adding the proposal as a declared
start leaves the generating topology at -25.6 and third of eight: the
diffusion times are bracketed to within a factor of 1.6, and this landscape
needs better than that -- seeded at the truth the same topology reaches -12.6
and wins. So the engine works and the proposal is real, but it is not yet
accurate enough on the separations that matter.

Two defects were found by measuring rather than by reading, and both had the
same shape -- a quantity described relative to its *bound* instead of to the
data. Features were peak-normalised, which threw away the amplitude and the
plateau and then asked the network to predict them; and parameters were
sampled across their declared bounds, so an FCS baseline bounded at +/-1e6
made every episode a flat line at 10^6 with identical features. Both showed
up as a mean squared error sitting exactly at the variance of the target,
which is the signature of a network that has been given nothing.

## What a record of a search can honestly promise (2026-09-14)

Three rounds of cross-platform CI settled this, each round finding that the
record claimed more than the numbers support:

| claimed | actually reproducible | why |
|---|---|---|
| every topology's fitted values to 1e-6 | the **winner's free** parameters to 1e-4 | an over-parameterised topology has no single answer, and a fixed slot holds whichever start won |
| every topology's score to 1e-3 | the **winner's** score, and the **ranking** exactly | topologies sit within hundredths of each other and multistart then picks between near-equal optima -- 2.5e-3 apart on `fcs.3d.2diff.0relax` |
| a joint fit recovers its parameters on noisy data | it recovers them on the **curves themselves** | the 24-parameter fit converges in 2 of 7 noise draws; the assertion had been passing on the luck of one |

So the contract is: ids, topology keys, transitions, priors and masks exactly;
the ranking exactly, because that is what selection *means* and it is ordinal;
and the winner's score and free parameters to what a converged fit gives. A
topology nobody picked contributes its position, not its number.

Checked against deliberate regressions rather than assumed: the winner's score
degrading by one percent, its free values moving by one percent, the ranking
reordering, a mask bit flipping, a prior changing and a topology vanishing are
all still caught. The defects found earlier today moved these quantities by
factors of forty.

The last row is the one worth remembering. A test that passes on one machine
because a fit happened to converge there is not measuring what its name says,
and the fix was not a tolerance -- it was asking one question at a time.
Whether the model recovers its parameters and whether the optimiser reaches
them from generic seeds are different questions, and only the first belongs in
a unit test while the second stays measured here at 3 of 7.

## The joint anisotropy fit: two defects, both fixed (2026-09-14)

"Converges in 3 of 7 noise draws" was never a hard optimisation problem. Four
cross-platform CI rounds and two wrong diagnoses later, it was two defects,
and with both fixed the family selects the generating topology in **7 of 7**.

**Defect one: a parameter seeded on its own bound.** Every per-channel
background was seeded at zero, which is also its lower bound, where the
optimiser can only push inward. If the first step raises the intensity
instead, the background stays pinned and `n0` absorbs it -- measured at 63719
against a truth of 30000. A background cannot exceed the smallest thing
measured, so a `min` statistic now seeds it off the bound. Recovered
intensities went from 112% out in two of four backgrounds to within 1% in all
four.

**Defect two: a budget sized for the wrong problem.** MINPACK's
`200 * (n + 1)` is a convention for fitting n parameters once. Three channels
sharing one photophysics converge more slowly than that allows, so the fit
stopped on the iteration count -- status 5 -- and the candidate rolled back to
its parent. It was never diverging: given about twenty times the budget it
reaches the same optimum the easy cases reach, in under a second. A family may
now declare `maxfev`, and this one declares 60000.

**Two wrong diagnoses, both worth keeping.** The first blamed near-singular
Jacobian columns, on a measured sensitivity of 2.05e-11 against ~1e-5. That
was the probe, not the model: it stepped by `|v| * 1e-4` against parameters
sitting at zero, so it measured its own 1e-8 floor. *A sensitivity measured
with a step proportional to the value says nothing about a parameter whose
value is zero.* The second called it divergence; the fitted parameters said
otherwise, and reading them rather than probing them is what finally settled
it.

**What this cost in tests.** Four assertions were written and removed over
four CI rounds, each hostage to the broken fit. The strongest of them --
recovery of the shared photophysics and the per-channel intensities from
Poisson-sampled curves -- is now restored and passing, which is the right
order: fix the fit, then let the test say what it wanted to say.

## What follows

1. **Sorting.** TCSPC component labels permute between slots depending on the
   seeds, because nothing orders a fitted component family.
   `mcts-generalization-cleanup.md` asks for descending characteristic value;
   nothing provides it. Two identical fits should not produce two different
   records.
2. **Better starts, principled ones.** A declared start set does not guarantee
   the global optimum, and tuning brackets against one synthetic curve is
   overfitting. A real answer probably looks like a coarse grid over the
   component separation, or a cheap pre-fit that places the components before
   the full model is fitted.
3. **The joint factory may use the tree search.** The earlier note against it
   was written on the pre-fix measurement and no longer holds. What still
   holds: a joint problem's candidates need scores of their own just as much
   as a single-dataset one's do, so warm starting must stay off across members.
4. Add fixtures whose generating topology is *not* the root. The committed
   ones cannot detect this class of bug, which is how both defects survived.

## Status

Handover item 7 is answered. The comparison it asked for could not be made
honestly until warm starting stopped making a candidate's score depend on the
route to it; that is fixed, and with it made the answer is that the tree search
matches exhaustive selection at lower cost.
