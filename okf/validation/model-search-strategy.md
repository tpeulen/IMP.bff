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
