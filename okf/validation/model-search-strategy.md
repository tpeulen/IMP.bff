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

**And the fix does not generalise, which was checked rather than assumed.**
The FCS two-species case -- where the generating topology reaches -25.6 from
its declared starts and -12.6 seeded at the truth -- is *not* budget limited:
it returns -25.57 at every budget from the default to 500000. That one is a
genuine local minimum, so the two failures that looked alike are different
problems. The anisotropy family needed iterations; FCS needs to start
somewhere else, which is what the self-play proposer is for.

## A residual action policy, and what it may and may not do (2026-09-23)

*Superseded the same day by the family-agnostic policy in the next section; the
keyed API described here (`set_residual_action_policy`) no longer exists. Kept
for the defects it records.*

The section above argued that learning *which action* repairs a fit learns the
part that already works. That still holds for selection. A policy can still pay
for itself on cost, by steering PUCT's early expansions towards the move the
residual shape asks for, so fewer candidates are fitted before the winner is
reached. `ModelSearchSelfPlay::generate_policy`/`train_policy` and
`set_residual_action_policy` on both fitting problems provide this. The reward
is untouched: BIC/AIC or the declared score still decide.

The first cut of that plumbing had four defects, fixed before anything was
trained on it:

- **Expansion refitted the graph.** `get_actions` restored the state and
  re-ran `objective->update()` every time, and threw when the state was not
  cached. Now the weighted residual is stored on the snapshot when the state
  is scored. Expansion reads it and has no side effects, and a state without
  one keeps its declared priors.
- **Two scales in one distribution.** Actions the network named got softmax
  probabilities, taken over keys the state might not even offer, while the rest
  kept raw declared weights. Now the declared priors are normalised over the
  available actions, and the softmax, restricted to the available named
  actions, reallocates only the mass those actions jointly hold.
- **Self-play read a port by convention** (`"residuals"`) and had its own copy
  of the profile reduction. Now it goes through `get_active_residual()` and
  the one public `get_residual_profile`.
- **Biased labels.** Episodes were drawn uniformly over transitions, so an
  action reachable from many parents dominated, and a terminal action was
  never a label, so the policy could not learn to stop. Now each episode draws
  an action first, and terminal actions get episodes that fit the right
  topology to its own data. `train_policy` takes a seed and a held-out
  validation fraction and reports validation loss and top-1 accuracy.

Self-play episodes were also silently empty of axes. `FitDataset::set_values`
clears coordinates, so the simulated dataset lost `curve.axis` and every parent
fit refused. The new `FitDataset::replace_values` swaps observations and keeps
the shape, coordinates, mask and variance.

**No trained policy exists yet, and none ships.** Whether one helps is
unmeasured. The gate is evaluations-to-generating-topology with and without a
policy, on held-out simulated data per family.

## One action policy for every game (2026-09-23)

The keyed residual policy above scored a fixed list of action names, so a
network trained on one family meant nothing to another. What landed instead
scores *moves by their features*, so one network serves every family and
every kind of data:

- **State features** (`get_policy_state_features`): the fitted weighted
  residual as 32 bucket z-scores (`asinh(sum/sqrt(count))`, so a 64-point FCS
  curve and a 4096-channel decay read alike), log10 of the reduced chi-square,
  lag-1 autocorrelation, the compressed Wald-Wolfowitz runs z-score, the share
  of positive residuals and log(1 + free parameters). A joint objective's
  residual is its members' blocks end to end, so FCS + TCSPC is the same case.
- **Move features** (`get_policy_action_features`): terminal, returns to the
  same structure, the change in free parameters, adds/removes, log(1 + target
  free parameters).
- **Scoring**: one MLP output per (state, move) row; a move's prior is its
  declared prior times `exp(score)`, normalised over the available moves
  (`FittingModelSearchProblem::set_action_policy`). A declared zero stays
  zero; an untrained (zero-output) network is exactly the declared priors.
- **Episodes** (`ModelSearchSelfPlay::generate_policy`): draw a generating
  structure, then a starting structure with a declared path to it; simulate
  the generator's measurement (`simulate`), fit the start, label the first
  move of a shortest path (or the terminal move when they coincide). Count
  data are recorded as photons by TTTRLib's `SimEngine`
  (`PhotonExperiment::record_pattern`), everything else gets the noise its
  dataset declares. Episodes with a single move, a failed simulation or an
  unconverged fit are skipped. `ModelSearchPolicyData` pools families and
  round-trips JSON.
- **Training** (`train_action_policy`): listwise softmax of `log(prior) +
  score`, Adam, weight decay, standardised inputs. Held-out episodes are split
  in two: one half picks the epoch (early stopping), the other half is the
  only thing reported, overall and per family against the priors alone.
- **The games** (`test/mcts/_games.py`): FCS, TCSPC lifetime, polarised,
  anisotropy, pddem, discrete/Gaussian FRET, equations and the new
  `kinetic_fcs_tcspc` family -- a chain of 1-3 states whose populations
  weight the decay (TCSPC) and whose relaxations modulate the correlation
  (FCS), fitted under one `FitJointChiSquared` (`KineticSchemeNode`).
  Single-structure families (worm-like chain, SAW, Ising FRET) offer no
  decision and produce no episodes.

**Measured** (`test/mcts/train_action_policy.py`, 200 episodes per game,
photons on, 1671 episodes; hidden [16], lr 0.002, weight decay 1e-3):
held-out top-1 move accuracy 0.582 against 0.406 for the declared priors,
cross-entropy 0.847 against 0.949, and at least the priors' accuracy in every
family. A first, unregularised network (hidden [32, 32], lr 0.01) reached 0.96
on training episodes and 0.52 held out -- memorisation, fixed by early
stopping and weight decay.

**The gate is the search, not the ranking** (`test/mcts/bench_action_policy.py`):
fresh simulated measurements, searched at budgets 2/4/8 with and without the
policy, scored by how often the search ends on the structure an exhaustive
walk selects. Candidate 1 (above) matched the priors at budgets 4 and 8 with
up to 13% fewer evaluations, but was two runs worse of 72 at budget 2 (one
FCS, one Gaussian FRET); the pre-registered gate says do not ship, and it was
withdrawn from `data/model_search/`.

**Candidates 3 and 4.** A prior-share move feature and a minimiser fix
(starts on a bound, see `okf/log.md`) gave candidate 3: search accuracy equal
to the priors at budget 2, +5/144 at budget 4, -2/144 at budget 8, 9-18%
fewer evaluations, every difference in FCS. Training labelled the move
towards the *generating* structure while the gate scores the structure
*selection* picks, and on noisy FCS the two differ. Candidate 4 labels by
selection: each simulated measurement is walked exhaustively once and every
reached structure becomes an episode towards the best-scoring one. Held-out
move accuracy rose to 0.788 against 0.217 for the priors (every family
better), and search evaluations fell 45% at budget 8, but accuracy stayed
2-3/144 short at budgets 2 and 8 -- the policy stops too early (kinetic scheme,
pddem). `set_action_policy` takes a temperature for that trade-off.

Two defects surfaced on the way, both fixed where they live: the worm-like
chain multiplied a vanishing exponential by a Bessel I0 that overflows first
(`boost::math::cyl_bessel_i` raised mid-fit for stiff chains; the density is
now formed in log space with an asymptotic `log I0`, guarded by
`test_a_stiff_chain_does_not_overflow_and_stretches_out`); and the arm64 env's
tttrlib C++ libraries linked a `libhdf5.200` the env no longer has (rebuilt
against the env's hdf5 1.14 and reinstalled; bff's `dependency/tttrlib`
now links `tttrlib` optionally, `IMP_BFF_HAS_TTTRLIB`).

## What follows

0. **Ship an action policy that passes the search gate.** Train more
   episodes and re-benchmark with enough measurements that one or two runs
   are not noise (the budget-2 deficit of candidate 1 is 2/72). If budget 2
   stays behind, the likely cause is the first expansion: with two
   evaluations the policy's first choice is final, so weight the loss towards
   the root's move or cap how far the policy may pull a prior. Then write
   `data/model_search/action_policy.json`; chisurf loads it by default
   (`NativeSearchSettings.action_policy = "shipped"`), and ships declared
   priors while the file is absent. Still open from PRD-152: block-aware
   features (one profile per member of a joint residual, with a modality
   token), a `request_information` outcome, and full photon-stream scenarios
   (one `SimEngine` stream reduced to decay *and* correlation) for the
   kinetic family instead of per-curve pattern recording.
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
