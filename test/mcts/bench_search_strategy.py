"""Is the tree search earning its place? Measure, do not assume.

Run it directly; it is a benchmark, not a test, so nothing here asserts.

    $E/bin/python test/mcts/bench_search_strategy.py

The model-search engine was built around MCTS, but the spaces it searches are
small -- the analytical FCS family has eight topologies and the TCSPC
lifetime family three, against a default of two hundred simulations. On a
space that size an exhaustive walk is not merely competitive, it is
*exhaustive*: it returns the optimum, deterministically, and it cannot be
tuned into the wrong answer by an exploration constant.

Three strategies, all driven through the same problem transitions, so the
evaluation semantics -- warm starting, the native minimiser, rollback -- are
identical and only the traversal differs:

``single``
    Evaluate the initial topology and stop. This is what pressing **Fit**
    gives today, and it is the baseline any search has to beat to exist.
``enumerate``
    Reach every topology once by the shortest path from the root. Costs one
    evaluation per structure and returns the best of them.
``mcts``
    ``IMP.bff.ModelSearch``, at its default budget and at smaller ones, over
    several seeds. Reported with how often it agrees with ``enumerate``,
    because a stochastic search that usually finds the optimum is a different
    proposition from one that always does.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import IMP.bff as bff  # noqa: E402

import _characterize  # noqa: E402
import _fixtures  # noqa: E402


def single(build):
    problem = build()
    start = time.perf_counter()
    root = problem.get_initial_state()
    return {
        "structure": root.get_structure_key(),
        "reward": root.get_reward(),
        "evaluations": 1,
        "seconds": time.perf_counter() - start,
    }


def enumerate_all(build):
    problem = build()
    start = time.perf_counter()
    root = problem.get_initial_state()
    graph = _characterize.action_graph(problem)
    best_key, best_reward, evaluations = root.get_structure_key(), root.get_reward(), 1
    for structure, path in sorted(
        _characterize._shortest_paths(graph, root.get_structure_key()).items()
    ):
        if not path:
            continue
        state = _characterize._walk(problem, root, path)
        evaluations += len(path)
        if state.get_reward() > best_reward:
            best_key, best_reward = state.get_structure_key(), state.get_reward()
    return {
        "structure": best_key,
        "reward": best_reward,
        "evaluations": evaluations,
        "seconds": time.perf_counter() - start,
    }


def mcts(build, simulations, seed):
    problem = build()
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(simulations)
    config.set_seed(seed)
    search = bff.ModelSearch(problem)
    search.set_config(config)
    start = time.perf_counter()
    result = search.run()
    return {
        "structure": result.get_best_state().get_structure_key(),
        "reward": result.get_best_state().get_reward(),
        "evaluations": result.get_number_of_states_evaluated(),
        "seconds": time.perf_counter() - start,
    }


def _fcs_two_component():
    """A curve the simplest topology cannot fit, so the search has work.

    The committed fixtures are generated from one 2-D species, which makes
    the root already optimal and the family search worth exactly nothing on
    them -- true, but uninformative about whether searching helps. This is
    two species with a tenfold separation in diffusion time.
    """
    import numpy as np

    axis = np.geomspace(1.0e-3, 20.0, 120)
    fast, slow, fraction, n, baseline = 0.08, 0.8, 0.4, 2.0, 1.0
    curve = baseline + (1.0 / n) * (
        fraction / (1.0 + axis / fast) + (1.0 - fraction) / (1.0 + axis / slow)
    )
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset(
        "curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002)
    )
    return spec.build()


CASES = dict(_fixtures.FIXTURES, fcs_two_species=_fcs_two_component)

SEEDS = (1, 7, 11, 23, 42, 101, 1009)
BUDGETS = (8, 16, 32, 200)


def path_dependence():
    """Does a topology score the same however the search reached it?

    It has to, for model selection to mean anything: comparing candidates
    assumes each one has *a* score. Warm starting breaks that assumption --
    parameters free in both parent and child carry the parent's fitted values
    over, so a topology inherits whichever basin its path happened to land
    in. Two orderings of the same commuting moves is the smallest possible
    demonstration.
    """
    build = CASES["fcs_two_species"]
    target = "fcs.3d.1diff.1relax"
    print(f"\npath dependence of one topology ({target})")
    print(f"  {'reward':>13}   route")
    rewards = []
    for route in (["use-3d-diffusion", "add-relaxation"],
                  ["add-relaxation", "use-3d-diffusion"]):
        problem = build()
        root = problem.get_initial_state()
        state = _characterize._walk(problem, root, route)
        rewards.append(state.get_reward())
        print(f"  {state.get_reward():>13.3f}   {' -> '.join(route)}")
    spread = max(rewards) - min(rewards)
    print(f"  same topology, same data, same free parameters: {spread:.1f} apart")


def main() -> int:
    for name, build in sorted(CASES.items()):
        problem = build()
        topologies = len(problem.get_structure_keys())
        print(f"\n{name}  ({topologies} topologies)")
        print(f"  {'strategy':<22}{'evals':>7}{'ms':>9}{'reward':>13}  {'agrees':>7}  winner")

        base = single(build)
        print(f"  {'single (Fit today)':<22}{base['evaluations']:>7}"
              f"{base['seconds']*1e3:>9.1f}{base['reward']:>13.3f}  {'-':>7}  "
              f"{base['structure']}")

        exhaustive = enumerate_all(build)
        print(f"  {'enumerate':<22}{exhaustive['evaluations']:>7}"
              f"{exhaustive['seconds']*1e3:>9.1f}{exhaustive['reward']:>13.3f}"
              f"  {'optimum':>7}  {exhaustive['structure']}")

        for budget in BUDGETS:
            runs = [mcts(build, budget, seed) for seed in SEEDS]
            agree = sum(r["structure"] == exhaustive["structure"] for r in runs)
            evals = sum(r["evaluations"] for r in runs) / len(runs)
            ms = sum(r["seconds"] for r in runs) / len(runs) * 1e3
            winners = {r["structure"] for r in runs}
            shown = (next(iter(winners)) if len(winners) == 1
                     else f"{len(winners)} different: " + ", ".join(sorted(winners)))
            reward = sum(r["reward"] for r in runs) / len(runs)
            print(f"  {'mcts @ ' + str(budget):<22}{evals:>7.1f}{ms:>9.1f}"
                  f"{reward:>13.3f}  {agree}/{len(SEEDS):<5}  {shown}")

        improvement = exhaustive["reward"] - base["reward"]
        print(f"  search is worth {improvement:+.3f} reward over a single fit")
    path_dependence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
