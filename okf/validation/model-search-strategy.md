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

## What follows

1. **Make a candidate's score well defined before comparing strategies.**
   Either a declared canonical initialisation per structure, independent of
   path, or multistart per candidate with the best retained. Until then no
   selection score over this family means what it appears to mean.
2. **Then re-run this.** The strategy question is cheap to answer once the
   scores are real; this file is the harness.
3. **Do not build the joint factory on an eagerly materialised cross-product
   justified by tree search.** Nothing measured here shows MCTS earning the
   combinatorics. On spaces of 1, 3 and 8 topologies enumeration costs at
   most a few more evaluations and is deterministic.
4. Fix the two-component FCS seeding, and add a fixture whose generating
   topology is recovered — the current fixtures cannot detect this class of
   bug because their answer is the root.

## Status

Handover item 7 is answered, and answered against its own premise: the
comparison it asked for cannot be completed honestly until warm starting
stops making a candidate's score depend on the route to it.
