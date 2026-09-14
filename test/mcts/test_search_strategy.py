"""What a model-search score has to mean for selection to be selection.

The measurements behind these are in `okf/validation/model-search-strategy.md`
and reproducible with `test/mcts/bench_search_strategy.py`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _characterize  # noqa: E402
import _fixtures  # noqa: E402


def _two_species_problem():
    """A correlation curve from two species a tenfold apart in diffusion."""
    axis = np.geomspace(1.0e-3, 20.0, 120)
    fast, slow, fraction, n, baseline = 0.08, 0.8, 0.4, 2.0, 1.0
    curve = baseline + (1.0 / n) * (
        fraction / (1.0 + axis / fast) + (1.0 - fraction) / (1.0 + axis / slow)
    )
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002))
    return spec.build()


def test_searching_beats_fitting_the_initial_topology():
    """The premise of the whole engine: the root is not always the answer."""
    problem = _fixtures.tcspc_lifetime()
    record = _characterize.characterize(problem)
    best = max(
        [record["root"]] + list(record["structures"].values()),
        key=lambda state: state["reward"],
    )
    assert best["structure"] != record["root"]["structure"]
    assert best["reward"] > record["root"]["reward"]


def test_enumeration_recovers_the_generating_component_count():
    problem = _fixtures.tcspc_lifetime()
    record = _characterize.characterize(problem)
    rewards = {record["root"]["structure"]: record["root"]["reward"]}
    rewards.update({k: v["reward"] for k, v in record["structures"].items()})
    assert max(rewards, key=rewards.get) == "lifetime.components.2"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Warm starting carries the parent's fitted values into every parameter "
        "free in both, so a topology inherits whichever basin its route landed "
        "in. Measured spread for fcs.3d.1diff.1relax is ~2529 in reward between "
        "two orderings of the same commuting moves. Until a candidate has a "
        "score of its own, comparing candidates -- by any search strategy -- "
        "does not mean what it appears to. See "
        "okf/validation/model-search-strategy.md."
    ),
)
def test_a_topology_scores_the_same_however_it_is_reached():
    problem = _two_species_problem()
    root = problem.get_initial_state()
    rewards = []
    for route in (
        ["use-3d-diffusion", "add-relaxation"],
        ["add-relaxation", "use-3d-diffusion"],
    ):
        problem = _two_species_problem()
        root = problem.get_initial_state()
        state = _characterize._walk(problem, root, route)
        assert state.get_structure_key() == "fcs.3d.1diff.1relax"
        rewards.append(state.get_reward())
    assert abs(rewards[0] - rewards[1]) < 1.0e-6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The two-component seeding (td2 = 4*td1, a1 = 0.5) does not reach the "
        "right basin, so on a genuine two-species curve the generating "
        "topology scores worst of the eight and a spurious relaxation term "
        "scores best. See okf/validation/model-search-strategy.md."
    ),
)
def test_the_generating_fcs_topology_is_not_the_worst_of_the_family():
    record = _characterize.characterize(_two_species_problem())
    rewards = {record["root"]["structure"]: record["root"]["reward"]}
    rewards.update({k: v["reward"] for k, v in record["structures"].items()})
    ranked = sorted(rewards, key=rewards.get, reverse=True)
    assert ranked[-1] != "fcs.2d.2diff.0relax"
