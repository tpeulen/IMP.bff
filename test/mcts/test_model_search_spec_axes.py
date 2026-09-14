"""A family declares the range of a component count; the search picks within it.

Writing every topology out by hand stops working as soon as a family has more
than one thing to vary -- lifetimes by rotations by whether a term is present
is a cross-product, and a cross-product written out is neither reviewable nor
maintainable. `axes` and `template` are the `for` loop the C++ factories had,
moved into the description: one construct, not a language.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402

FAMILY = json.loads(
    (pathlib.Path(bff.get_data_path("model_search/tcspc_lifetime.json"))).read_text()
)


def _built(document):
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_json(json.dumps(document))
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec.build()


def _with_ceiling(components):
    document = copy.deepcopy(FAMILY)
    document["axes"]["n"]["to"] = components
    return document


@pytest.mark.parametrize("ceiling", [1, 2, 3, 5, 8])
def test_the_component_ceiling_is_one_number(ceiling):
    """Raising it adds topologies and the parameters they need, nothing else."""
    problem = _built(_with_ceiling(ceiling))
    assert len(problem.get_structure_keys()) == ceiling
    # Two canonical parameters per component, plus the eight instrument ones
    # (scatter, background, n0, timeshift, the response's lamp background, and
    # a modelled response's width, shape and position).
    assert len(problem.get_parameter_ids()) == 2 * ceiling + 8
    assert f"lifetime.components.{ceiling}" in problem.get_structure_keys()


def test_a_wider_family_still_selects_the_generating_component_count():
    """Offering more topologies must not talk the search out of the right one."""
    for ceiling in (3, 5, 8):
        problem = _built(_with_ceiling(ceiling))
        config = bff.ModelSearchConfig()
        config.set_number_of_simulations(40)
        config.set_dirichlet_fraction(0.0)
        config.set_seed(5)
        search = bff.ModelSearch(problem)
        search.set_config(config)
        assert (
            search.run().get_best_state().get_structure_key()
            == "lifetime.components.2"
        )


def test_moves_that_would_leave_the_family_are_not_offered():
    problem = _built(_with_ceiling(3))
    for key, expected in (
        ("lifetime.components.1", {"stop", "add-component"}),
        ("lifetime.components.2", {"stop", "add-component", "remove-component"}),
        ("lifetime.components.3", {"stop", "remove-component"}),
    ):
        state = bff.ModelSearchState(key, key, 0.0, False)
        assert {a.get_key() for a in problem.get_actions(state)} == expected


def test_a_repeat_reaches_the_index_it_declares():
    """Per-component ports and ids are generated, not listed."""
    problem = _built(_with_ceiling(4))
    objective = problem.get_structure_objective("lifetime.components.4")
    decay = objective.get_input_port("model").link.get_node()
    for index in range(4):
        owner = problem.get_parameter(f"lifetime.tau.{index}")
        assert decay.get_input_port(f"t{index}").link.uid == owner.uid


def test_a_family_written_both_ways_is_refused():
    document = copy.deepcopy(FAMILY)
    document["structures"] = {"anything": {}}
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.ModelSearchSpec.from_json(json.dumps(document))
    assert "structures" in str(caught.value)


def test_a_move_along_an_axis_the_family_lacks_is_refused():
    document = copy.deepcopy(FAMILY)
    document["moves"].append({"action": "add-rotation", "axis": "rot", "delta": 1})
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.ModelSearchSpec.from_json(json.dumps(document))
    assert "rot" in str(caught.value)
