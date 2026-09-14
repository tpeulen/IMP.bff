"""A mixture of lifetime models: each model's lifetimes, weighted, in one decay.

A sample that is several species is described by one model per species. What
the mixture reads from each is its published lifetime spectrum -- a live port
that follows whichever topology that model stands at -- so fitting or
switching a mixed model changes the mixture without anything being copied.
"""

import sys
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _response():
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    return data, response, dt, period


def _lifetime_model(taus, amplitudes=None):
    data, response, dt, period = _response()
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    model = spec.get_model()
    model.select_structure(f"lifetime.components.{len(taus)}")
    amplitudes = amplitudes or [1.0 / len(taus)] * len(taus)
    for i, (a, t) in enumerate(zip(amplitudes, taus)):
        _set(model, f"lifetime.amplitude.{i}", a)
        _set(model, f"lifetime.tau.{i}", t)
    return spec, model


def _set(model, canonical, value):
    port = model.get_parameter(canonical)
    held = port.fixed
    port.fixed = False
    port.value = value
    port.fixed = held


def _mixture(sources, fractions):
    data, response, dt, period = _response()
    spec = bff.ModelSearchSpec.from_name("tcspc_mixture")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    spec.set_scalar("number_of_models", len(sources))
    for i, source in enumerate(sources):
        spec.set_port(f"lifetime_spectrum.{i}", source.get_output_port("lifetime_spectrum"))
    model = spec.get_model()
    for i, x in enumerate(fractions):
        _set(model, f"mixture.fraction.{i}", x)
    return spec, model


def _curve(model):
    active = model.get_active_structure()
    return np.asarray(model.get_structure_output(active, model.get_structure_curve_node(active, "decay")))


def test_the_node_weights_each_species_and_joins_them():
    node = bff.GraphNodeRegistry.create("LifetimeSpectrumMixture", "mix")
    node.configure('{"number_of_species": 2}')
    node.add_input_port("s0", bff.GraphPort([2.0, 1.0, 2.0, 4.0]))
    node.add_input_port("s1", bff.GraphPort([1.0, 0.5]))
    node.add_input_port("x0", bff.GraphPort(3.0))
    node.add_input_port("x1", bff.GraphPort(1.0))
    node.add_output_port("mix", bff.GraphPort([0.0], False, True))
    node.update()
    np.testing.assert_allclose(node.get_output_port("mix").value,
                               [0.375, 1.0, 0.375, 4.0, 0.25, 0.5])


def test_a_published_output_follows_the_topology_and_survives_a_rebuild():
    spec, model = _lifetime_model([0.7, 3.8], [0.6, 0.4])
    port = model.get_output_port("lifetime_spectrum")
    node = bff.GraphNodeRegistry.create("LifetimeSpectrumMixture", "follower")
    node.configure('{"number_of_species": 1}')
    follower = bff.GraphPort([0.0])
    node.add_input_port("s0", follower)
    node.add_input_port("x0", bff.GraphPort(1.0))
    node.add_output_port("follower", bff.GraphPort([0.0], False, True))
    follower.link = port
    node.update()
    np.testing.assert_allclose(np.asarray(node.get_output_port("follower").value)[1::2], [0.7, 3.8])

    model.select_structure("lifetime.components.1")
    node.update()
    assert np.asarray(node.get_output_port("follower").value).size == 2

    spec.set_scalar("period", 12.0)
    rebuilt = spec.get_model()
    assert rebuilt.get_output_port("lifetime_spectrum").uid == port.uid
    _set(rebuilt, "lifetime.tau.0", 2.5)
    node.update()
    assert np.asarray(node.get_output_port("follower").value)[1] == pytest.approx(2.5)


def test_a_mixture_is_the_lifetime_model_of_all_mixed_lifetimes():
    _, fast = _lifetime_model([0.5, 1.5], [0.5, 0.5])
    _, slow = _lifetime_model([4.0])
    _, mixture = _mixture([fast, slow], [1.0, 3.0])
    # One quarter fast (split evenly), three quarters slow.
    _, reference = _lifetime_model([0.5, 1.5, 4.0], [0.125, 0.125, 0.75])
    for model in (mixture, reference):
        _set(model, "instrument.n0", 5000.0)
        _set(model, "instrument.background", 1.0)
    np.testing.assert_allclose(_curve(mixture), _curve(reference), rtol=1e-10, atol=1e-10)

    # A change in a mixed model is a change in the mixture.
    _set(slow, "lifetime.tau.0", 3.0)
    _, reference = _lifetime_model([0.5, 1.5, 3.0], [0.125, 0.125, 0.75])
    _set(reference, "instrument.n0", 5000.0)
    _set(reference, "instrument.background", 1.0)
    np.testing.assert_allclose(_curve(mixture), _curve(reference), rtol=1e-10, atol=1e-10)


def test_the_first_fraction_is_held_and_the_others_fit():
    _, fast = _lifetime_model([0.7])
    _, slow = _lifetime_model([3.8])
    _, mixture = _mixture([fast, slow], [1.0, 1.0])
    assert mixture.get_parameter("mixture.fraction.0").fixed
    assert not mixture.get_parameter("mixture.fraction.1").fixed


def test_an_unbound_model_is_refused_by_name():
    data, response, dt, period = _response()
    spec = bff.ModelSearchSpec.from_name("tcspc_mixture")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    spec.set_scalar("number_of_models", 2)
    assert "lifetime_spectrum.1" in spec.get_port_names()
    with pytest.raises((ValueError, RuntimeError)) as caught:
        spec.get_model()
    assert "lifetime_spectrum.0" in str(caught.value)


def test_a_model_cannot_mix_itself():
    _, fast = _lifetime_model([0.7])
    spec, mixture = _mixture([fast], [1.0])
    with pytest.raises(Exception):
        spec.set_port("lifetime_spectrum.0", mixture.get_output_port("lifetime_spectrum"))
        spec.get_model()
