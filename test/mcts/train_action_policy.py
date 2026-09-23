"""Train the one action policy bff ships, on every game at once.

Run it; it is a generator, like ``generate_golden.py``, not a test:

    $E/bin/python test/mcts/train_action_policy.py

Every family in ``_games.py`` plays search episodes against itself: a
generating structure's measurement is simulated -- through TTTRLib's photon
engine wherever the data are counts -- every reachable structure is fitted to
it, and from each one the move towards the structure selection picks is the
label. The episodes pool
into one data set, because the features describe fitted residuals and moves
rather than any family's names, and one network is trained on all of it.

Writes ``data/model_search/policy/action_policy.msgpack`` (the network the search
loads, a ``bff.neural_net`` document as msgpack), ``action_policy.report.json`` (what it was trained on and how it
ranks held-out moves against the declared priors alone) and, with
``--save-episodes``, the episodes themselves so a retrain need not replay.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import msgpack
import IMP.bff as bff

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _games  # noqa: E402

DATA = HERE.parent.parent / "data" / "model_search" / "policy"


def play(name, episodes, seed, photons):
    selfplay = bff.ModelSearchSelfPlay(_games.spec(name))
    selfplay.set_spread(0.3)
    selfplay.set_photon_simulation(photons)
    start = time.perf_counter()
    data = selfplay.generate_policy(episodes, seed)
    return data, time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--measurements", type=int, default=200,
                        help="simulated measurements per game; each yields an episode per structure")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--hidden", type=int, nargs="*", default=[16])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--validation", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--no-photons", action="store_true",
                        help="sample counting noise instead of recording photons")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--episodes-file", type=pathlib.Path, default=None,
                        help="reuse saved episodes instead of playing")
    parser.add_argument("--save-episodes", type=pathlib.Path, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=DATA / "action_policy.msgpack")
    args = parser.parse_args(argv)

    photons = not args.no_photons and bff.PhotonExperiment.get_available()
    games = args.games or _games.names()
    timing = {}
    if args.episodes_file:
        pooled = bff.ModelSearchPolicyData.from_json(args.episodes_file.read_text())
    else:
        pooled = bff.ModelSearchPolicyData()
        for index, name in enumerate(games):
            data, seconds = play(name, args.measurements, args.seed + index, photons)
            timing[name] = round(seconds, 2)
            print(f"{name:30s} {data.get_number_of_episodes():5d} episodes "
                  f"{data.get_number_of_stop_labels():4d} stop  {seconds:7.1f} s", flush=True)
            pooled.extend(data)
    if args.save_episodes:
        args.save_episodes.write_text(pooled.to_json())

    trained = bff.train_action_policy(pooled, list(args.hidden), args.epochs,
                                      args.learning_rate, args.seed, args.validation,
                                      args.weight_decay)
    families = list(trained.get_families())
    accuracy = dict(zip(families, trained.get_family_validation_accuracy()))
    baseline = dict(zip(families, trained.get_family_baseline_accuracy()))
    counts = dict(zip(pooled.get_families(), pooled.get_family_counts()))
    print(f"\nkept epoch {trained.get_best_epoch()} of {args.epochs}; held-out "
          f"{trained.get_number_of_validation_episodes()} episodes: accuracy "
          f"{trained.get_validation_accuracy():.3f} (priors {trained.get_baseline_accuracy():.3f}), "
          f"loss {trained.get_validation_loss():.3f} (priors {trained.get_baseline_loss():.3f})")
    for name in families:
        print(f"  {name:30s} {accuracy[name]:.3f}  priors {baseline[name]:.3f}")

    args.out.write_bytes(msgpack.packb(json.loads(trained.get_network()), use_bin_type=True))
    report = {
        "format": "bff.model_search.action_policy_report.v1",
        "seed": args.seed,
        "photons": photons,
        "measurements_per_game": args.measurements,
        "episodes": pooled.get_number_of_episodes(),
        "episodes_by_family": counts,
        "stop_labels": pooled.get_number_of_stop_labels(),
        "hidden": list(args.hidden),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_loss": trained.get_training_loss(),
        "training_accuracy": trained.get_training_accuracy(),
        "validation_episodes": trained.get_number_of_validation_episodes(),
        "validation_loss": trained.get_validation_loss(),
        "validation_accuracy": trained.get_validation_accuracy(),
        "baseline_accuracy": trained.get_baseline_accuracy(),
        "validation_accuracy_by_family": accuracy,
        "baseline_accuracy_by_family": baseline,
        "baseline_loss": trained.get_baseline_loss(),
        "best_epoch": trained.get_best_epoch(),
        "weight_decay": args.weight_decay,
        "seconds_by_game": timing,
    }
    args.out.with_name(args.out.stem + ".report.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
