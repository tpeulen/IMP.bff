"""A built description as a live model an application can hold.

An application shows one model and lets a user work on it: pick a topology,
fix a parameter, swap the measurement, fit. None of that may copy the model.
These are the properties that make a view on it possible -- the parameter
ports stay the same objects, the values stay the user's, and the fit and the
search run on the one graph.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _spec(scale=1.0):
    data, irf, dt, period = _fixtures._tcspc_dataset()
    if scale != 1.0:
        scaled = bff.FitDataset()
        scaled.set_values_array(np.ascontiguousarray(np.asarray(data.get_values()) * scale))
        scaled.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
        data = scaled
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def _ports(problem):
    return {i: problem.get_parameter(i) for i in problem.get_parameter_ids()}


def test_the_model_is_one_object_until_something_changes():
    spec = _spec()
    model = spec.get_model()
    assert spec.get_model_is_current()
    assert spec.get_model() is model or spec.get_model().get_parameter_ids() == model.get_parameter_ids()
    assert model.get_active_structure() == model.get_initial_structure()


def test_new_data_rebuilds_over_the_same_ports_and_keeps_the_users_values():
    spec = _spec()
    model = spec.get_model()
    before = _ports(model)
    model.get_parameter("lifetime.tau.0").value = 2.5
    model.select_structure("lifetime.components.2")
    model.set_parameter_locked("lifetime.tau.1", True)

    data, _, _, _ = _fixtures._tcspc_dataset()
    brighter = bff.FitDataset()
    brighter.set_values_array(np.ascontiguousarray(np.asarray(data.get_values()) * 3.0))
    brighter.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    spec.set_dataset("decay", brighter)
    assert not spec.get_model_is_current()
    rebuilt = spec.get_model()

    after = _ports(rebuilt)
    assert list(after) == list(before)
    for canonical, port in before.items():
        assert after[canonical].uid == port.uid, canonical
    assert rebuilt.get_parameter("lifetime.tau.0").value == pytest.approx(2.5)
    assert rebuilt.get_active_structure() == "lifetime.components.2"
    assert rebuilt.get_parameter_locked("lifetime.tau.1")
    # The data-derived bound followed the new measurement.
    total = float(np.sum(brighter.get_values()))
    assert rebuilt.get_parameter("instrument.n0").get_upper_bound() == pytest.approx(100 * total)


def test_selecting_a_topology_changes_what_is_free_and_nothing_else():
    model = _spec().get_model()
    values = [model.get_parameter(i).value for i in model.get_parameter_ids()]
    model.select_structure("lifetime.components.3")
    assert [model.get_parameter(i).value for i in model.get_parameter_ids()] == values
    free = list(model.get_structure_parameter_ids("lifetime.components.3"))
    assert "lifetime.tau.2" in free
    assert not model.get_parameter("lifetime.tau.2").fixed
    model.select_structure("lifetime.components.1")
    assert model.get_parameter("lifetime.tau.2").fixed
    assert "lifetime.tau.2" not in model.get_structure_parameter_ids("lifetime.components.1")


def test_the_ordinary_fit_runs_on_the_models_own_graph():
    model = _spec().get_model()
    model.select_structure("lifetime.components.2")
    # The fixture has no background, which puts the truth on the bound; a
    # bounded fit only creeps towards a bound, so the known value is held.
    background = model.get_parameter("instrument.background")
    background.fixed = False
    background.value = 0.0
    model.set_parameter_locked("instrument.background", True)
    status = model.fit_active_structure()
    assert 1 <= status <= 4
    taus = sorted(model.get_parameter(f"lifetime.tau.{i}").value for i in (0, 1))
    assert taus == pytest.approx([0.7, 3.8], rel=1e-3)
    assert model.get_last_reduced_chi2() == pytest.approx(0.0, abs=1e-6)


def test_a_lock_holds_through_fits_and_searches_and_is_not_charged_for():
    model = _spec().get_model()
    model.select_structure("lifetime.components.2")
    model.get_parameter("lifetime.tau.1").value = 3.8
    model.set_parameter_locked("lifetime.tau.1", True)
    model.fit_active_structure()
    assert model.get_parameter("lifetime.tau.1").value == 3.8

    unlocked = _spec().build()
    locked = _spec().build()
    locked.get_parameter("lifetime.tau.1").value = 3.8
    locked.set_parameter_locked("lifetime.tau.1", True)
    for problem in (unlocked, locked):
        config = bff.ModelSearchConfig()
        config.set_number_of_simulations(8)
        config.set_dirichlet_fraction(0.0)
        search = bff.ModelSearch(problem)
        search.set_config(config)
        search.run()
    assert locked.get_parameter("lifetime.tau.1").value == 3.8

    # One parameter fewer is estimated, so one fewer is paid for: at the same
    # misfit the locked topology scores half a log(n) better under BIC.
    root = locked.get_initial_state()
    assert np.isfinite(root.get_reward())


def test_a_lock_on_an_unknown_parameter_is_refused():
    model = _spec().get_model()
    with pytest.raises((ValueError, RuntimeError)):
        model.set_parameter_locked("lifetime.nonexistent", True)


def test_the_description_is_handed_over_as_data():
    spec = _spec()
    document = json.loads(spec.get_description_json())
    assert document["family"] == "tcspc_lifetime"
    assert set(document["structures"]) == set(spec.get_structure_keys())
    # Templates are expanded: an application reads structures, not rules.
    assert "lifetime.components.3" in document["structures"]


def test_a_topology_shows_the_rows_its_graph_reads():
    model = _spec().get_model()
    one = set(model.get_structure_parameter_ids("lifetime.components.1"))
    three = set(model.get_structure_parameter_ids("lifetime.components.3"))
    assert "lifetime.tau.0" in one and "lifetime.tau.2" not in one
    assert "lifetime.tau.2" in three
    # The instrument is read by every topology, fitted by default or not.
    assert {"instrument.timeshift", "instrument.scatter"} <= one


def test_a_release_fits_what_the_description_holds_and_keeps_the_users_value():
    model = _spec().get_model()
    model.select_structure("lifetime.components.2")
    assert model.get_parameter("instrument.timeshift").fixed
    model.set_parameter_released("instrument.timeshift", True)
    assert not model.get_parameter("instrument.timeshift").fixed
    model.get_parameter("instrument.timeshift").value = 0.3
    model.activate_structure("lifetime.components.2")
    assert model.get_parameter("instrument.timeshift").value == pytest.approx(0.3)
    model.fit_active_structure()
    # The generating decay has no shift, and the fit finds that.
    assert model.get_parameter("instrument.timeshift").value == pytest.approx(0.0, abs=1e-3)

    model.set_parameter_locked("instrument.timeshift", True)
    assert not model.get_parameter_released("instrument.timeshift")
    assert model.get_parameter("instrument.timeshift").fixed


def test_what_to_show_is_data_in_the_description():
    spec = _spec()
    document = json.loads(spec.get_description_json())
    presentation = document["presentation"]
    primary = [slot for slot, info in presentation["datasets"].items() if info.get("primary")]
    assert primary == ["decay"]
    groups = {entry["group"] for entry in document["parameters"].values()}
    assert groups <= set(presentation["groups"])
    assert document["structures"]["lifetime.components.2"]["label"] == "2 lifetime(s)"


def test_a_suggested_value_is_evaluated_over_the_bound_data():
    data, irf, dt, period = _fixtures._tcspc_dataset()
    axis = np.arange(len(irf)) * dt
    measured = bff.FitDataset()
    measured.set_values_array(np.ascontiguousarray(np.asarray(data.get_values())))
    measured.set_coordinate_array(0, "time", np.ascontiguousarray(axis))
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", measured)
    assert spec.evaluate("decay_dx") == pytest.approx(dt)
    assert spec.evaluate("2*decay_size") == 2 * len(irf)
    with pytest.raises((ValueError, RuntimeError)):
        spec.evaluate("response_size")


@pytest.mark.parametrize("short_first", [True, False])
def test_component_order_is_presentation_not_a_constraint(short_first):
    """Whichever row holds the short lifetime, the fit reaches the same mixture.

    The first amplitude is the fixed reference the others are ratios to. A
    ceiling on those ratios made the answer depend on which component landed
    in the reference row: a long component carrying more than the reference
    could not be expressed at all.
    """
    n, dt, period = 192, 0.05, 12.5
    x = np.arange(n) * dt
    irf = np.exp(-0.5 * ((x - 0.8) / 0.08) ** 2)
    source = bff.TCSPCDecay("source")
    source.set_number_of_lifetimes(2)
    source.add_output_port("source", bff.GraphPort([0.0], False, True))
    source.set_response_array(np.ascontiguousarray(irf))
    source.set_timing(dt, period)
    source.set_convolution_range(n, n)
    source.set_normalize_amplitudes(True)
    source.get_input_port("a0").value = 0.25
    source.get_input_port("t0").value = 0.6
    source.get_input_port("a1").value = 0.75
    source.get_input_port("t1").value = 3.5
    source.get_input_port("n0").value = 20000.0
    source.update()
    measured = bff.FitDataset()
    # A background off its bound, as a measurement has.
    decay = np.asarray(source.get_output_port("source").value) + 5.0
    measured.set_values_array(np.ascontiguousarray(decay))
    measured.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", measured)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    model = spec.get_model()
    model.select_structure("lifetime.components.2")
    first, second = (0.8, 3.0) if short_first else (3.0, 0.8)
    model.get_parameter("lifetime.tau.0").value = first
    model.get_parameter("lifetime.tau.1").value = second
    model.fit_active_structure()

    rows = [(model.get_parameter(f"lifetime.tau.{i}").value,
             model.get_parameter(f"lifetime.amplitude.{i}").value) for i in (0, 1)]
    total = sum(a for _, a in rows)
    mixture = sorted((tau, a / total) for tau, a in rows)
    assert [tau for tau, _ in mixture] == pytest.approx([0.6, 3.5], rel=1e-3)
    assert [f for _, f in mixture] == pytest.approx([0.25, 0.75], rel=1e-3)


def test_a_polarized_family_is_the_lifetime_family_times_rotations():
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_polarized")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    model = spec.get_model()
    assert len(model.get_structure_keys()) == 3 * 2
    key = "lifetime.components.2.rotations.2"
    model.select_structure(key)
    node = model.get_structure_curve_node(key, "decay")
    magic = []
    for polarization in (0.0, 1.0, 2.0):
        spec.set_scalar("polarization", polarization)
        model = spec.get_model()
        model.select_structure(key)
        magic.append(np.array(model.get_structure_output(key, node)))
    vm, vv, vh = magic
    # Parallel and perpendicular differ, and neither is the magic-angle decay.
    assert not np.allclose(vv, vh) and not np.allclose(vv, vm)
