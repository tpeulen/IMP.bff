"""A second action space: how a candidate is fitted, searched like its structure.

A structure that does not converge from its declared starts within its
budget loses a comparison it might win. With the fit-settings space on, a
state is a structure fitted one particular way, structural moves keep that
way, and settings moves refit the same structure another way -- all scored by
the same reward, so a settings move only wins by fitting better.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _run(problem, simulations=24, seed=5):
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(simulations)
    config.set_dirichlet_fraction(0.0)
    config.set_seed(seed)
    search = bff.ModelSearch(problem)
    search.set_config(config)
    return search.run()


def test_searching_fit_settings_finds_the_structure_a_structure_search_finds():
    alone = _run(_fixtures.tcspc_lifetime())
    problem = _fixtures.tcspc_lifetime()
    problem.add_default_fit_settings_candidates()
    problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)

    both = _run(problem)

    best = both.get_best_state()
    assert best.get_structure_key() == alone.get_best_state().get_structure_key()
    assert best.get_reward() == pytest.approx(alone.get_best_state().get_reward(), abs=1e-3)
    assert best.get_fit_settings_key() in [c.get_key() for c in problem.get_fit_settings_candidates()]
    assert best.get_search_key() == f"{best.get_structure_key()}|{best.get_fit_settings_key()}"
    assert both.get_number_of_states_evaluated() >= alone.get_number_of_states_evaluated()


def test_a_patient_fit_rescues_a_structure_the_budget_lost():
    short = _fixtures.tcspc_lifetime()
    short.set_minimizer_maxfev(_fixtures.SHORT_BUDGET)
    stuck = _run(short)
    assert stuck.get_best_state().get_structure_key() == "lifetime.components.1"

    rescued = _run(_fixtures.tcspc_lifetime_fit_settings())

    best = rescued.get_best_state()
    assert best.get_structure_key() == "lifetime.components.2"
    assert best.get_fit_settings_key() == "patient"
    assert best.get_reward() > stuck.get_best_state().get_reward() + 100.0


def test_the_moves_out_of_a_state_keep_one_axis_and_change_the_other():
    problem = _fixtures.tcspc_lifetime()
    problem.add_default_fit_settings_candidates()
    problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)
    root = problem.get_initial_state()
    assert root.get_fit_settings_key() == "declared"

    actions = {a.get_key(): a for a in problem.get_actions(root)}

    assert actions["add-component"].get_predicted_state_key() == "lifetime.components.2|declared"
    settings = [k for k in actions if k.startswith("fit-settings:")]
    assert sorted(settings) == sorted(
        f"fit-settings:{c.get_key()}" for c in problem.get_fit_settings_candidates()[1:])
    for key in settings:
        assert actions[key].get_predicted_state_key().startswith("lifetime.components.1|")
    structural = sum(a.get_prior() for k, a in actions.items() if k not in settings)
    shared = sum(actions[k].get_prior() for k in settings)
    assert shared == pytest.approx(problem.get_fit_settings_prior() * structural)


def test_a_policy_reads_a_settings_move_as_a_self_loop():
    problem = _fixtures.tcspc_lifetime()
    problem.add_default_fit_settings_candidates()
    problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)
    root = problem.get_initial_state()
    actions = problem.get_actions(root)
    width = bff.get_policy_state_width() + bff.get_policy_action_width()
    rows = problem.get_policy_rows(root.get_structure_key(),
                                   list(problem.get_cached_residual(root.get_key())), actions)
    self_loop = bff.get_policy_state_width() + 1
    for index, action in enumerate(actions):
        if action.get_key().startswith("fit-settings:"):
            assert rows[index * width + self_loop] == 1.0


def test_fit_settings_are_validated():
    problem = _fixtures.tcspc_lifetime()
    with pytest.raises(ValueError, match="candidates"):
        problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)
    with pytest.raises(ValueError, match="start"):
        problem.add_fit_settings_candidate(bff.ModelSearchFitSettings(
            "odd", "leastsq", 0.0, 0.0, 0.0, 0, 0.0, "anywhere", True))
    with pytest.raises(ValueError):
        problem.add_fit_settings_candidate(bff.ModelSearchFitSettings(
            "odd", "simplex", 0.0, 0.0, 0.0, 0, 0.0, "declared", True))
    with pytest.raises(ValueError, match="step factor"):
        problem.add_fit_settings_candidate(bff.ModelSearchFitSettings(
            "odd", "leastsq", 0.0, 0.0, 0.0, 0, 0.01, "declared", True))
    problem.add_default_fit_settings_candidates()
    with pytest.raises(ValueError, match="duplicate"):
        problem.add_fit_settings_candidate(bff.ModelSearchFitSettings())
    assert all(c.get_use_bounds() for c in problem.get_fit_settings_candidates())
    problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)
    with pytest.raises(ValueError, match="leave"):
        problem.clear_fit_settings_candidates()
    problem.set_action_space(bff.ACTION_SPACE_STRUCTURE)
    root = problem.get_initial_state()
    assert root.get_fit_settings_key() == "" and root.get_search_key() == root.get_structure_key()
