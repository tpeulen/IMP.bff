"""A kinetic scheme fitted to a decay and a correlation curve at once.

TCSPC sees the scheme's populations (one lifetime per state, weighted by how
much time a molecule spends there); FCS sees its relaxations (brightness
follows lifetime, so interconversion flickers the signal). Neither alone
identifies the rates and the lifetimes together; one joint objective does.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _node(n, taus, kf=(), kb=(), lag=None):
    node = bff.KineticSchemeNode("kin")
    node.set_number_of_states(n)
    for i, t in enumerate(taus):
        node.get_input_port("t%d" % i).value = t
    for i, (f, b) in enumerate(zip(kf, kb), start=1):
        node.get_input_port("kf%d" % i).value = f
        node.get_input_port("kb%d" % i).value = b
    if lag is not None:
        node.get_input_port("x").value = list(lag)
    node.add_output_port("lifetime_spectrum", bff.GraphPort([0.0], False, True))
    node.add_output_port("correlation", bff.GraphPort([0.0], False, True))
    node.update()
    return node


def test_one_state_is_one_lifetime_and_no_flicker():
    node = _node(1, [2.5], lag=[0.0, 1.0, 100.0])
    assert list(node.get_output_port("lifetime_spectrum").value) == [1.0, 2.5]
    np.testing.assert_allclose(node.get_output_port("correlation").value, 1.0)


def test_two_states_relax_at_the_sum_of_their_rates():
    kf, kb, t0, t1 = 3.0, 1.0, 1.0, 4.0
    lag = np.geomspace(1e-3, 10.0, 40)
    node = _node(2, [t0, t1], [kf], [kb], lag)
    p1 = kf / (kf + kb)
    p0 = 1.0 - p1
    np.testing.assert_allclose(node.get_populations(), [p0, p1])
    np.testing.assert_allclose(node.get_relaxation_rates(), [kf + kb])
    mean = p0 * t0 + p1 * t1
    amplitude = p0 * p1 * (t0 - t1) ** 2 / mean ** 2
    expected = 1.0 + amplitude * np.exp(-(kf + kb) * lag)
    np.testing.assert_allclose(node.get_output_port("correlation").value, expected, rtol=1e-12)
    np.testing.assert_allclose(node.get_output_port("lifetime_spectrum").value, [p0, t0, p1, t1])


def test_equal_lifetimes_do_not_flicker_whatever_the_rates():
    node = _node(3, [2.0, 2.0, 2.0], [5.0, 0.3], [1.0, 2.0], [0.0, 0.1, 1.0])
    np.testing.assert_allclose(node.get_output_port("correlation").value, 1.0)


def _truth(model, structure):
    """Replace every bound curve by what one topology of the family predicts."""
    spec = _fixtures.kinetic_fcs_tcspc_spec()
    model.activate_structure(structure)
    for name in model.get_structure_curve_datasets(structure):
        dataset = spec.get_dataset(name)
        dataset.replace_values(
            model.get_structure_output(structure, model.get_structure_curve_node(structure, name)))
        spec.set_dataset(name, dataset)
    return spec


def _set(model, canonical, value):
    model.get_parameter(canonical).value = value


def test_the_family_joins_both_measurements_under_one_objective():
    model = _fixtures.kinetic_fcs_tcspc_spec().build()
    keys = list(model.get_structure_keys())
    assert keys == ["kinetic.states.1", "kinetic.states.2", "kinetic.states.3"]
    for key in keys:
        assert set(model.get_structure_curve_datasets(key)) == {"decay", "curve"}
    model.activate_structure("kinetic.states.2")
    joint = model.get_active_objective()
    joint.update()
    assert joint.get_node_type() == "FitJointChiSquared"
    spec = _fixtures.kinetic_fcs_tcspc_spec()
    sizes = [len(spec.get_dataset_values(name)) for name in ("decay", "curve")]
    assert joint.get_number_of_residuals() == sum(sizes)


def test_the_search_recovers_a_two_state_scheme_from_its_own_data():
    generator = _fixtures.kinetic_fcs_tcspc_spec().build()
    generator.activate_structure("kinetic.states.2")
    _set(generator, "state.tau.0", 0.8)
    _set(generator, "state.tau.1", 3.6)
    _set(generator, "kinetics.forward.1", 20.0)
    _set(generator, "kinetics.backward.1", 12.0)
    problem = _truth(generator, "kinetic.states.2").build()

    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(40)
    config.set_seed(3)
    config.set_dirichlet_fraction(0.0)
    search = bff.ModelSearch(problem)
    search.set_config(config)
    best = search.run().get_best_state()

    assert best.get_structure_key() == "kinetic.states.2"
