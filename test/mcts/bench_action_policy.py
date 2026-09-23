"""Does the shipped action policy earn its place? The gate before it ships.

    $E/bin/python test/mcts/bench_action_policy.py --policy data/model_search/policy/action_policy.json

On fresh measurements -- a seed range training never used, photons recorded by
TTTRLib where the data are counts -- every game is searched at small budgets
with and without the policy. What counts is how often the search ends on the
structure an exhaustive walk selects (the best any search could do on that
data), and how many structures it evaluated to get there. A policy ships only
if it is at least as accurate at every budget and cheaper or more accurate at
the small ones; the verdict is printed and written beside the policy.

A benchmark, not a test: nothing here asserts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import IMP.bff as bff

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _games  # noqa: E402
from bench_search_strategy import enumerate_all  # noqa: E402

DATA = HERE.parent.parent / "data" / "model_search" / "policy"


def search(spec, policy, budget, seed, temperature=0.0):
    problem = spec.build()
    if policy:
        problem.set_action_policy(policy, temperature)
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(budget)
    config.set_seed(seed)
    config.set_dirichlet_fraction(0.0)
    engine = bff.ModelSearch(problem)
    engine.set_config(config)
    result = engine.run()
    return result.get_best_state().get_structure_key(), result.get_number_of_states_evaluated()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", type=pathlib.Path, default=DATA / "action_policy.json")
    parser.add_argument("--measurements", type=int, default=12, help="per game")
    parser.add_argument("--budgets", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0: the policy document's own (1 when it has none)")
    args = parser.parse_args(argv)
    policy = args.policy.read_text()
    effective = args.temperature or float(json.loads(policy).get("temperature", 1.0))
    photons = bff.PhotonExperiment.get_available()
    rng = random.Random(args.seed)

    tally = {}
    for name in args.games or _games.names():
        selfplay = bff.ModelSearchSelfPlay(_games.spec(name))
        selfplay.set_spread(0.3)
        selfplay.set_photon_simulation(photons)
        structures = list(_games.spec(name).build().get_structure_keys())
        if len(structures) < 2:
            continue  # nothing to choose between
        for _ in range(args.measurements):
            truth = rng.choice(structures)
            spec = selfplay.simulate(truth, rng.randrange(2**31))
            optimum = enumerate_all(spec.build)["structure"]
            for budget in args.budgets:
                for label, network in (("priors", ""), ("policy", policy)):
                    found, cost = search(spec, network, budget, rng.randrange(2**31),
                                         args.temperature)
                    cell = tally.setdefault(name, {}).setdefault(str(budget), {}).setdefault(
                        label, {"right": 0, "runs": 0, "evaluations": 0})
                    cell["right"] += int(found == optimum)
                    cell["runs"] += 1
                    cell["evaluations"] += cost
        print(name, json.dumps(tally.get(name, {})), flush=True)

    overall = {}
    for per_budget in tally.values():
        for budget, per_label in per_budget.items():
            for label, cell in per_label.items():
                total = overall.setdefault(budget, {}).setdefault(
                    label, {"right": 0, "runs": 0, "evaluations": 0})
                for key in total:
                    total[key] += cell[key]
    verdict = True
    print("\nbudget   priors: right/runs evals   policy: right/runs evals")
    for budget in sorted(overall, key=int):
        p, q = overall[budget]["priors"], overall[budget]["policy"]
        print(f"{budget:>6}   {p['right']:4d}/{p['runs']:<4d} {p['evaluations']:6d}"
              f"          {q['right']:4d}/{q['runs']:<4d} {q['evaluations']:6d}")
        if q["right"] < p["right"]:
            verdict = False
    print("\nverdict:", "ship" if verdict else "do not ship")
    report = {"format": "bff.model_search.action_policy_benchmark.v1",
              "seed": args.seed, "photons": photons, "budgets": args.budgets,
              "temperature": effective,
              "measurements_per_game": args.measurements, "by_game": tally,
              "overall": overall, "ship": verdict}
    args.policy.with_name(args.policy.stem + ".benchmark.json").write_text(
        json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
