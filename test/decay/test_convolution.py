"""A generic convolution node, and the TCSPC instrument fed from a port.

The convolution knows nothing about what it convolves; the instrument stage
is TCSPCDecay's own, run on a curve that arrives instead of a spectrum. And a
catalogue of decay equations meets both through the convolved-equations frame.
"""

import json

import numpy as np
import pytest

import IMP.bff as bff

N = 128
DT = 0.05


def _response(center=1.0, width=0.1):
    x = np.arange(N) * DT
    return x, np.exp(-0.5 * ((x - center) / width) ** 2)


def _node(kind, name, response, **config):
    node = bff.GraphNodeRegistry.create(kind, name)
    if config:
        node.configure(json.dumps(config))
    dataset = bff.FitDataset()
    dataset.set_values_array(np.ascontiguousarray(response))
    node.bind_dataset("response", dataset)
    node.add_output_port(name, bff.GraphPort([0.0], False, True))
    return node


def _vector_input(node, key, values):
    port = node.get_input_port(key)
    if port is None:
        port = bff.GraphPort([0.0])
        node.add_input_port(key, port)
    port.set_values_array(np.ascontiguousarray(np.asarray(values, dtype=float)))
    return port


def test_the_node_is_numpys_causal_convolution_with_a_unit_response():
    x, response = _response()
    curve = 3.0 * np.exp(-x / 1.2) + 0.5 * np.exp(-x / 0.3)
    node = _node("Convolution", "conv", response)
    _vector_input(node, "curve", curve)
    node.update()
    got = np.asarray(node.get_output_port("conv").value)
    expected = np.convolve(curve, response / response.sum(), "full")[:N]
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-15)


def test_the_node_shifts_the_response_as_the_instrument_does():
    """A delta convolves to the prepared response, which TCSPCDecay's scatter term also is."""
    _, response = _response()
    delta = np.zeros(N)
    delta[0] = 1.0
    conv = _node("Convolution", "conv", response)
    _vector_input(conv, "curve", delta)
    conv.add_input_port("timeshift", bff.GraphPort(2.37))
    conv.update()
    shifted = np.asarray(conv.get_output_port("conv").value)

    decay = _node("TCSPCDecay", "decay", response, curve_from_port=True, timing=[DT, 12.5])
    _vector_input(decay, "curve", np.zeros(N))
    decay.get_input_port("scatter").value = 1.0
    decay.get_input_port("n0").value = 1.0
    decay.get_input_port("timeshift").value = 2.37
    decay.update()
    np.testing.assert_allclose(np.asarray(decay.get_output_port("decay").value), shifted, rtol=1e-13, atol=1e-15)
    assert not np.allclose(shifted, response / response.sum())


def test_the_instrument_runs_on_a_curve_from_a_port():
    _, response = _response()
    curve = np.linspace(-1.0, 4.0, N)
    decay = _node("TCSPCDecay", "decay", response, curve_from_port=True, timing=[DT, 12.5])
    _vector_input(decay, "curve", curve)
    decay.get_input_port("scatter").value = 0.2
    decay.get_input_port("n0").value = 3.0
    decay.get_input_port("background").value = 0.5
    decay.update()
    expected = np.maximum(3.0 * (curve + 0.2 * response / response.sum()) + 0.5, 0.0)
    np.testing.assert_allclose(np.asarray(decay.get_output_port("decay").value), expected, rtol=1e-13)


def test_a_curve_of_the_wrong_length_is_refused():
    _, response = _response()
    decay = _node("TCSPCDecay", "decay", response, curve_from_port=True, timing=[DT, 12.5])
    _vector_input(decay, "curve", np.ones(N - 1))
    with pytest.raises((ValueError, RuntimeError)):
        decay.update()


def test_a_basis_from_a_curve_is_refused():
    decay = bff.TCSPCDecay("decay")
    decay.set_curve_from_port(True)
    with pytest.raises((ValueError, RuntimeError)):
        decay.set_emit_basis(True)


def _convolved_spec(y, response, x):
    measured = bff.FitDataset()
    measured.set_values_array(np.ascontiguousarray(y))
    measured.set_coordinate_array(0, "x", np.ascontiguousarray(x))
    measured.set_noise_family(bff.FIT_NOISE_FAMILY_STORED)
    measured.set_stored_variance_array(np.ascontiguousarray(np.maximum(y, 1.0)))
    irf = bff.FitDataset()
    irf.set_values_array(np.ascontiguousarray(response))
    spec = bff.ModelSearchSpec.from_name("equations_convolved")
    spec.set_equations(json.dumps({
        "one": {"equation": "a1*exp(-x/tau1)", "initial": {"a1": 1000.0, "tau1": 1.0}},
        "two": {"equation": "a1*exp(-x/tau1)+a2*exp(-x/tau2)", "initial": {"a2": 500.0, "tau2": 0.3}},
    }))
    spec.set_dataset("decay", measured)
    spec.set_dataset("response", irf)
    spec.set_scalar("period", 12.5)
    return spec


def test_decay_equations_meet_the_response_and_the_instrument_through_the_frame():
    x, response = _response()
    model = _convolved_spec(np.full(N, 50.0), response, x).get_model()
    assert list(model.get_structure_keys()) == ["one", "two"]
    ids = list(model.get_parameter_ids())
    assert ids[:4] == ["instrument.scatter", "instrument.background", "instrument.n0", "instrument.timeshift"]
    assert {"a1", "tau1", "a2", "tau2"} <= set(ids)
    assert "instrument.timeshift" in model.get_structure_parameter_ids("one")

    model.select_structure("two")
    truth = {"a1": 2000.0, "tau1": 2.0, "a2": 3000.0, "tau2": 0.4, "instrument.background": 3.0}
    for name, value in truth.items():
        port = model.get_parameter(name)
        port.fixed = False
        port.value = value
    node = model.get_structure_curve_node("two", "decay")
    clean = np.array(model.get_structure_output("two", node))
    expected = np.convolve(2000.0 * np.exp(-x / 2.0) + 3000.0 * np.exp(-x / 0.4),
                           response / response.sum(), "full")[:N] + 3.0
    np.testing.assert_allclose(clean, expected, rtol=1e-12)

    fitted = _convolved_spec(clean, response, x).get_model()
    fitted.select_structure("two")
    for name, value in {"a1": 1500.0, "tau1": 1.5, "a2": 2500.0, "tau2": 0.5}.items():
        fitted.get_parameter(name).value = value
    fitted.fit_active_structure()
    taus = sorted(fitted.get_parameter(t).value for t in ("tau1", "tau2"))
    assert taus == pytest.approx([0.4, 2.0], rel=1e-4)



def test_an_autoscaled_scale_is_published_to_the_models_parameter():
    """A fitted scale nobody can read is a fit nobody can report."""
    x, response = _response()
    y = 400.0 * np.convolve(np.exp(-x / 1.5), response / response.sum(), "full")[:N] + 5.0
    spec = _convolved_spec(y, response, x)
    spec.set_scalar("autoscale", 1.0)
    model = spec.get_model()
    model.select_structure("one")
    for name, value in {"a1": 1.0, "tau1": 1.5, "instrument.background": 5.0}.items():
        model.get_parameter(name).value = value
    node = model.get_structure_curve_node("one", "decay")
    curve = np.array(model.get_structure_output("one", node))
    assert model.get_parameter("instrument.n0").value == pytest.approx(400.0, rel=1e-9)
    assert model.get_parameter("instrument.n0").fixed
    np.testing.assert_allclose(curve, y, rtol=1e-9)
