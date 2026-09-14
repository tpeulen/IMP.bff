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


def _mode_node(response, curve, **config):
    node = _node("Convolution", "conv", response, **config)
    _vector_input(node, "curve", curve)
    node.update()
    return np.asarray(node.get_output_port("conv").value)


def _fold(full, n, period):
    """What periodic excitation adds: the spill of every earlier pulse, interpolated."""
    out = np.array(full[:n], dtype=float)
    for i in range(n):
        k = 1
        while i + k * period <= full.size - 1:
            position = i + k * period
            j = int(np.floor(position))
            f = position - j
            out[i] += (1 - f) * full[j] + (f * full[j + 1] if j + 1 < full.size else 0.0)
            k += 1
    return out


def test_the_centered_mode_is_numpys_same():
    x, _ = _response()
    kernel = np.exp(-0.5 * ((x - x.mean()) / 0.2) ** 2)
    curve = np.sin(x) ** 2
    expected = np.convolve(curve, kernel / kernel.sum(), "same")
    np.testing.assert_allclose(_mode_node(kernel, curve, mode="centered"), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("period", [64.0, 51.37])
def test_the_periodic_mode_folds_the_spill_of_earlier_pulses(period):
    x, response = _response(center=4.5, width=0.4)   # a late, broad response spills far
    curve = np.exp(-x / 2.0)
    unit = response / response.sum()
    expected = _fold(np.convolve(curve, unit, "full"), N, period)
    got = _mode_node(response, curve, mode="periodic", period=period)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
    causal = _mode_node(response, curve)
    assert np.all(got >= causal - 1e-15)
    assert got.sum() > causal.sum()


def test_a_mode_that_is_not_one_is_refused_and_a_period_is_required():
    _, response = _response()
    with pytest.raises((ValueError, RuntimeError)):
        _node("Convolution", "conv", response, mode="reflect")
    node = _node("Convolution", "conv", response, mode="periodic")
    _vector_input(node, "curve", np.ones(N))
    with pytest.raises((ValueError, RuntimeError)):
        node.update()


def test_a_frame_switches_the_convolution_to_periodic_by_a_flag():
    x, response = _response(center=4.5, width=0.4)
    spec = _convolved_spec(np.full(N, 50.0), response, x)
    spec.set_scalar("period", 5.0)            # 100 channels at dt = 0.05
    causal = spec.get_model()
    causal.select_structure("one")
    causal.get_parameter("instrument.background").value = 0.0
    node = causal.get_structure_curve_node("one", "decay")
    before = np.array(causal.get_structure_output("one", node))

    spec.set_scalar("periodic", 1.0)
    periodic = spec.get_model()
    periodic.select_structure("one")
    after = np.array(periodic.get_structure_output("one", node))
    a1, tau1 = periodic.get_parameter("a1").value, periodic.get_parameter("tau1").value
    full = np.convolve(a1 * np.exp(-x / tau1), response / response.sum(), "full")
    np.testing.assert_allclose(after, _fold(full, N, 5.0 / DT), rtol=1e-12)
    assert after.sum() > before.sum()

    spec.set_scalar("periodic", 2.0)
    with pytest.raises((ValueError, RuntimeError)):
        spec.get_model()


def _prepared(response, background, start, stop, timeshift):
    """ChiSurf's IRF preparation: background, clip, window, shift, unit sum."""
    from numpy import clip
    r = clip(response - background, 0.0, None)
    r[:start] = 0.0
    r[stop:] = 0.0
    ts = -timeshift
    i0 = int(np.floor(ts)); f = ts - i0
    shifted = np.array([
        (1 - f) * (r[j + i0] if 0 <= j + i0 < r.size else 0.0)
        + f * (r[j + i0 + 1] if 0 <= j + i0 + 1 < r.size else 0.0)
        for j in range(r.size)])
    return shifted / shifted.sum()


@pytest.mark.parametrize("kind", ["Convolution", "TCSPCDecay"])
def test_both_nodes_prepare_the_response_the_same_way(kind):
    x, response = _response(center=1.5, width=0.12)
    response = response * 1000.0 + 7.0          # a lamp background under the peak
    delta = np.zeros(N)
    delta[0] = 1.0
    if kind == "Convolution":
        node = _node(kind, "conv", response, response_range=[10, 90])
        _vector_input(node, "curve", delta)
        node.add_input_port("timeshift", bff.GraphPort(-1.4))
        key = "conv"
    else:
        node = _node(kind, "decay", response, curve_from_port=True, timing=[DT, 12.5],
                     response_range=[10, 90])
        _vector_input(node, "curve", np.zeros(N))
        node.get_input_port("scatter").value = 1.0
        node.get_input_port("timeshift").value = -1.4
        key = "decay"
    node.add_input_port("response_background", bff.GraphPort(7.5))
    node.update()
    np.testing.assert_allclose(np.asarray(node.get_output_port(key).value),
                               _prepared(response, 7.5, 10, 90, -1.4), rtol=1e-12, atol=1e-15)


def _classic_background_pattern(curve, pattern, data, timeshift, t_background, t_decay, shift=True):
    """ChiSurf's LifetimeModel: shift the pattern, split the counts, add."""
    if shift and timeshift != 0.0:
        ts = -timeshift
        i0 = int(np.floor(ts)); f = ts - i0
        pattern_used = np.array([
            (1 - f) * (pattern[j + i0] if 0 <= j + i0 < pattern.size else 0.0)
            + f * (pattern[j + i0 + 1] if 0 <= j + i0 + 1 < pattern.size else 0.0)
            for j in range(pattern.size)])
    else:
        pattern_used = pattern
    n_bg = pattern.sum() / t_background * t_decay
    n_fl = max(data.sum() - n_bg, 1.0)
    return curve * n_fl / curve.sum() + pattern_used * n_bg / pattern_used.sum()


@pytest.mark.parametrize("shift", [True, False])
def test_a_background_pattern_takes_its_share_of_the_counts(shift):
    x, response = _response()
    curve = np.convolve(np.exp(-x / 1.5), response / response.sum(), "full")[:N] * 500.0
    pattern = 40.0 + 30.0 * np.exp(-0.5 * ((x - 2.0) / 0.5) ** 2)
    data = np.full(N, 25.0)
    decay = _node("TCSPCDecay", "decay", response, curve_from_port=True, timing=[DT, 12.5],
                  background_times=[4.0, 2.5], shift_background_with_response=shift)
    measured = bff.FitDataset()
    measured.set_values_array(np.ascontiguousarray(data))
    decay.bind_dataset("data", measured)
    background = bff.FitDataset()
    background.set_values_array(np.ascontiguousarray(pattern))
    decay.bind_dataset("background_pattern", background)
    _vector_input(decay, "curve", curve)
    decay.get_input_port("timeshift").value = 1.3
    decay.get_input_port("n0").value = 1.0
    decay.update()
    expected = _classic_background_pattern(curve, pattern, data, 1.3, 4.0, 2.5, shift)
    np.testing.assert_allclose(np.asarray(decay.get_output_port("decay").value), expected, rtol=1e-12)


def _spectrum_decay(response, **config):
    node = _node("TCSPCDecay", "decay", response, timing=[DT, 6.0], **config)
    node.set_number_of_lifetimes(2) if not config.get("number_of_lifetimes") else None
    for key, value in {"a0": 0.3, "t0": 0.7, "a1": 0.7, "t1": 3.4, "n0": 1.0}.items():
        node.get_input_port(key).value = value
    node.update()
    return np.asarray(node.get_output_port("decay").value)


def test_a_single_excitation_is_tttrlibs_non_periodic_recursion():
    tttrlib = pytest.importorskip("tttrlib")
    x, response = _response()
    got = _spectrum_decay(response, number_of_lifetimes=2, convolution_mode="single")
    expected = np.zeros(N)
    tttrlib.fconv(fit=expected, irf=response / response.sum(), x=np.array([0.3, 0.7, 0.7, 3.4]),
                  start=0, stop=N, dt=DT)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
    periodic = _spectrum_decay(response, number_of_lifetimes=2)
    assert periodic.sum() > got.sum()      # earlier pulses add their tails


@pytest.mark.parametrize("mode", ["periodic", "single"])
def test_without_convolution_the_model_is_the_ideal_decay(mode):
    x, response = _response()
    got = _spectrum_decay(response, number_of_lifetimes=2, convolution_mode=mode, convolve=False)
    tail = (lambda tau: 1.0 / (1.0 - np.exp(-6.0 / tau))) if mode == "periodic" else (lambda tau: 1.0)
    expected = 0.3 * tail(0.7) * np.exp(-x / 0.7) + 0.7 * tail(3.4) * np.exp(-x / 3.4)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_the_basis_is_only_the_periodic_convolutions():
    decay = bff.TCSPCDecay("decay")
    decay.set_number_of_lifetimes(1)
    decay.set_convolution_mode("single")
    with pytest.raises((ValueError, RuntimeError)):
        decay.set_emit_basis(True)
    with pytest.raises((ValueError, RuntimeError)):
        decay.set_convolution_mode("reflected")
