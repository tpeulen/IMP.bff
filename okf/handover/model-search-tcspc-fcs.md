# Handover: BFF-native model search for TCSPC and FCS

Date: 2026-09-14 (supersedes the 2026-09-12 revision)
Branch: `independent-core`
Status: model families are declared in data; both shipped families are ported
and the C++ factories are deleted. ChiSurf migration is still deferred.

## Owner decisions

- ChiSurf is the application/car; BFF is the engine.
- Parameters, links, model topology, objectives, fitting decisions,
  diagnostics and model search live in BFF.
- Required model-search scope is TCSPC and FCS, including future global
  composition.
- No numerical fallback. If BFF cannot represent a model/objective
  completely, return unsupported and do nothing.
- No duplicated Python evaluator, optimizer, reward or topology declaration.
- Candidate graphs do not exchange parameter values. Every topology links to
  one BFF-owned canonical parameter registry.
- **A model family is data.** What can be C++ must be C++, and a family is
  not a numerical kernel: it is a registry, some graphs, and the moves
  between them. Kernels stay in C++; families are read.

## The shape of it

**Declaration happens once, at build. Evaluation happens N times, in C++.**
That is what makes a description safe against the no-fallback rule: a
description is read once and compiled into a complete native graph, so the
search itself crosses no language boundary.

`include/ModelSearch.h`, `src/ModelSearch.cpp`
- `ModelSearch`: native PUCT traversal, lazy expansion, deterministic seed,
  Dirichlet exploration, cycle/collapse handling, cancellation.
- `ModelSearchProblem`: model-independent C++ problem boundary, no director.
- `TabularModelSearchProblem`: callback-free finite reference problem.
- `FittingModelSearchProblem`: one graph, groups fixed and freed.
- `MultiStructureModelSearchProblem`: one canonical registry shared across
  complete structure-specific objective graphs. **Every family description
  builds one of these.**

`include/GraphNodeRegistry.h`, `src/GraphNodeRegistry.cpp`
- Type name to constructor, twelve types, installed on first use. The only
  such map in the library, deliberately.
- `GraphNode::get_node_type()` and `GraphNode::configure(json_text)`: each
  node reads its own settings beside its own setters, so no loader knows what
  a kernel can be told. `GraphNode::bind_dataset(role, dataset)` is the same
  idea for measurements.
- Unknown type names list what is registered; unconsumed settings are refused
  by name. A misspelling must fail where it is read.

`include/ModelSearchSpec.h`, `src/ModelSearchSpec.cpp`
- `from_name` / `from_file` / `from_json`, then `set_dataset`, `set_scalar`,
  `set_parameter`, then `build()`.
- Seeds and bounds that depend on the data are arithmetic over named
  statistics of the bound measurement, evaluated by `GraphExpression` -- the
  evaluator the models themselves use. `config` is literal, `config_rules` is
  arithmetic, and giving a setting both ways is refused.
- Complexity is counted from the structure's free list, never declared.

`data/model_search/tcspc_lifetime.json`, `data/model_search/fcs_analytical.json`
- The two shipped families. `ModelSearchSpec::get_available_names()` lists
  them, which is most of a capability query already.

## Verification evidence

Environment: `/Users/tpeulen/mambaforge/envs/arm64`
Build tree: `/Users/tpeulen/dev/imp/cmake-build-arm64`

```sh
ninja -C /Users/tpeulen/dev/imp/cmake-build-arm64 IMP.bff-python -j4
/Users/tpeulen/mambaforge/envs/arm64/bin/python -m pytest -q \
  test/mcts/ test/graph/ test/session/ test/portnode/ test/minimizer/ \
  test/dataset/ test/test_public_api_names.py test/test_base_header.py \
  test/test_api_taxonomy.py
```

647 passed, 2 xfailed. Do not launch concurrent Ninja builds against the same
IMP build tree.

**`test/mcts/golden/` is the contract.** It records what each family does
observably -- ids, topologies, transitions, fitted values, masks, rewards --
and was taken from the C++ factories before they were deleted. The ported
descriptions reproduce it exactly. Regenerate with
`test/mcts/generate_golden.py` only for a reviewed behaviour change, and read
the diff.

## Read this before planning the next step

`okf/validation/model-search-strategy.md`. Handover item 7 has been answered
and it changes the priorities below. In short: searching is worth a great
deal, but **a candidate currently has no score of its own** -- warm starting
makes a topology inherit whichever basin its route landed in, measured at
2528.9 in reward between two orderings of two commuting moves. Model
selection over these families does not yet mean what it appears to.

## Next BFF work, in order

1. **Give a candidate a score of its own.** Either a declared canonical
   initialisation per structure, independent of path, or multistart per
   candidate. Everything below is built on comparing candidates, so this
   comes first.
2. Fix the analytical FCS two-component seeding (`td2 = 4*td1`, `a1 = 0.5`),
   which currently makes the generating topology score worst of its family,
   and add a fixture whose answer is not the root.
3. Re-run `test/mcts/bench_search_strategy.py` once 1 and 2 land, and decide
   the search strategy on evidence. Do not assume the tree search.
4. FCS MDF and kinetics/saturation topologies. `FCSMdfCurve` and
   `FCSSaturationCurve` are registered node types that already accept their
   settings from a description, so these are new files, not new C++.
5. A joint TCSPC+FCS family. `FitJointChiSquared` is a registered type and
   `MultiStructureModelSearchProblem` already takes a joint objective, so the
   mechanism exists. **Do not eagerly materialise a cross-product justified
   by tree search** -- nothing measured supports that.
6. Structured capability/refusal records. `get_available_names()` plus a
   description's declared datasets and scalars is most of the answer.
7. Versioned search-result serialization for `.cs.pto` and history.

`GraphSession` still writes chinet's JSONL, which carries no node type and so
reloads every node as a behaviourless `GraphNode`. Rewriting it onto the
registry is the remaining piece of the one-format goal; its only external
caller is `../chisurf/chisurf/core/project/project.py:225`, and the `.csp`
format it serves is already ruled to need no compatibility.

## Deferred ChiSurf migration

Tracked at `../chisurf/.omx/plans/chisurf-changes-after-bff-tcspc-fcs-mcts.md`.
The ChiSurf MCTS tree is now committed on the `mcts-native-baseline` branch,
so the deletions that plan schedules are revertible. Its bridge still targets
the older generic `FittingModelSearchProblem` rather than these descriptions.

Three ChiSurf tests fail against current BFF and did so before any of this
work: they assert a child state scores better than the root, which stopped
being true when initial structures began being optimized before root scoring.

## Resume point

Start at item 1. Read this file, `okf/validation/model-search-strategy.md` and
the shared agent board, claim a ticket, inspect the dirty tree before editing,
and run the gate above before and after.
