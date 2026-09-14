"""A family built from a catalogue of equations over any measurement.

Nothing here names an experiment. The frame knows no equation, the catalogue
knows no measurement, and which variables are axes is read off the bound data:
a curve has one coordinate, a carpet three, and it is the same code.
"""

import json

import numpy as np
import pytest

import IMP.bff as bff


def _curve(names_values, values, errors=None):
    dataset = bff.FitDataset()
    dataset.set_values_array(np.ascontiguousarray(np.asarray(values, dtype=float)))
    for k, (name, coordinate) in enumerate(names_values):
        dataset.set_coordinate_array(k, name, np.ascontiguousarray(np.asarray(coordinate, dtype=float)))
    if errors is not None:
        dataset.set_noise_family(bff.FIT_NOISE_FAMILY_STORED)
        dataset.set_stored_variance_array(np.ascontiguousarray(np.asarray(errors, dtype=float) ** 2))
    return dataset


def _spec(catalogue, dataset):
    spec = bff.ModelSearchSpec.from_name("equations")
    spec.set_equations(json.dumps(catalogue))
    spec.set_dataset("curve", dataset)
    return spec


CATALOGUE = {
    "one decay": {"equation": "b + a1*exp(-x/t1)", "initial": {"a1": 2.0, "t1": 1.5, "b": 0.1}},
    "two decays": {
        "equation": "b + a1*exp(-x/t1) + a2*exp(-x/t2)",
        "initial": {"a2": 1.0, "t2": 5.0},
        "bounds": {"t2": [0.01, 100.0]},
    },
}


def _two_decay_data(noise=0.0):
    x = np.linspace(0.0, 20.0, 200)
    y = 0.2 + 3.0 * np.exp(-x / 0.8) + 1.5 * np.exp(-x / 6.0)
    return x, y


def test_each_equation_is_a_structure_sharing_parameters_by_name():
    x, y = _two_decay_data()
    spec = _spec(CATALOGUE, _curve([("x", x)], y, np.full(y.size, 0.01)))
    model = spec.get_model()
    assert list(model.get_structure_keys()) == ["one decay", "two decays"]
    assert list(model.get_parameter_ids()) == ["b", "a1", "t1", "a2", "t2"]
    assert set(model.get_structure_parameter_ids("one decay")) == {"b", "a1", "t1"}
    assert model.get_parameter("t2").get_upper_bound() == 100.0
    assert np.isinf(model.get_parameter("a1").get_upper_bound())
    assert model.get_parameter("t1").value == 1.5


def test_an_equation_over_three_coordinates_is_the_same_case():
    rng = np.random.default_rng(3)
    u, v, w = (rng.uniform(-1, 1, 64) for _ in range(3))
    catalogue = {"surface": {"equation": "a*exp(-(u**2+v**2)/s)+w*b",
                             "initial": {"a": 2.0, "s": 0.5, "b": 0.3}}}
    expected = 2.0 * np.exp(-(u ** 2 + v ** 2) / 0.5) + w * 0.3
    spec = _spec(catalogue, _curve([("u", u), ("v", v), ("w", w)], expected, np.ones(64)))
    model = spec.get_model()
    node = model.get_structure_curve_node("surface", "curve")
    assert np.asarray(model.get_structure_output("surface", node)) == pytest.approx(expected)
    assert list(model.get_parameter_ids()) == ["a", "s", "b"]


def test_the_search_picks_the_equation_that_made_the_data_by_bic():
    x, y = _two_decay_data()
    spec = _spec(CATALOGUE, _curve([("x", x)], y, np.full(y.size, 0.01)))
    model = spec.get_model()
    root = model.get_initial_state()
    assert {a.get_key() for a in model.get_actions(root)} == {"stop", "use:two decays"}
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(8)
    config.set_dirichlet_fraction(0.0)
    search = bff.ModelSearch(model)
    search.set_config(config)
    best = search.run().get_best_state()
    assert best.get_structure_key() == "two decays"


def test_a_held_variable_is_not_fitted():
    x, y = _two_decay_data()
    catalogue = {"one decay": dict(CATALOGUE["one decay"], fixed=["b"])}
    model = _spec(catalogue, _curve([("x", x)], y, np.full(y.size, 0.01))).get_model()
    model.fit_active_structure()
    assert model.get_parameter("b").value == 0.1


def test_a_new_catalogue_keeps_the_ports_it_shares():
    x, y = _two_decay_data()
    spec = _spec(CATALOGUE, _curve([("x", x)], y, np.full(y.size, 0.01)))
    before = spec.get_model()
    uid = before.get_parameter("t1").uid
    spec.set_equations(json.dumps({"custom": {"equation": "b + a1*exp(-(x/t1)**beta)"}}))
    after = spec.get_model()
    assert after.get_parameter("t1").uid == uid
    assert "beta" in list(after.get_parameter_ids())
    assert list(after.get_structure_keys()) == ["custom"]


@pytest.mark.parametrize("catalogue, fragment", [
    ({"broken": {"equation": "a*("}}, "broken"),
    ({"shadow": {"equation": "a*x", "initial": {"x": 1.0}}}, "coordinate"),
    ({"constant": {"equation": "a+b"}}, "none of the coordinates"),
])
def test_what_does_not_add_up_is_refused_by_name(catalogue, fragment):
    x, y = _two_decay_data()
    spec = _spec(catalogue, _curve([("x", x)], y, np.ones(y.size)))
    with pytest.raises((ValueError, RuntimeError)) as caught:
        spec.get_model()
    assert fragment in str(caught.value)


def test_a_family_with_declared_structures_takes_no_catalogue():
    with pytest.raises((ValueError, RuntimeError)):
        bff.ModelSearchSpec.from_name("tcspc_lifetime").set_equations(json.dumps(CATALOGUE))
