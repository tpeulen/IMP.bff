"""FRET families read from descriptions: a donor quenched over distances.

Every family is `donor -> distances -> FRETSpectrumNode -> TCSPCDecay`, and
only the distance producer differs. Parity with ChiSurf's classic FRET models
is pinned on the ChiSurf side; here are the engine's own claims -- the nodes'
arithmetic, what a family publishes, and a search that finds the number of
distances the data were made from.
"""

import sys
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402

FAMILIES = ["tcspc_fret_gaussian", "tcspc_fret_discrete", "tcspc_fret_worm_like_chain",
            "tcspc_fret_saw_nu", "tcspc_fret_ising_chain", "tcspc_pddem"]


def _spec(family, data=None):
    decay, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name(family)
    spec.set_dataset("decay", data if data is not None else decay)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def _set(model, canonical, value):
    port = model.get_parameter(canonical)
    held = port.fixed
    port.fixed = False
    port.value = value
    port.fixed = held


def _curve(model):
    active = model.get_active_structure()
    return np.asarray(model.get_structure_output(active, model.get_structure_curve_node(active, "decay")))


def test_discrete_distances_are_normalised_magnitudes():
    node = bff.DiscreteDistances("d")
    node.configure('{"number_of_distances": 2}')
    node.add_output_port("d", bff.GraphPort([0.0], False, True))
    node.get_input_port("distance0").value = -40.0
    node.get_input_port("amplitude0").value = 3.0
    node.get_input_port("distance1").value = 60.0
    node.get_input_port("amplitude1").value = -1.0
    node.update()
    np.testing.assert_allclose(node.get_output_port("d").value, [0.75, 40.0, 0.25, 60.0])


def test_a_distance_axis_is_configured_as_values_or_as_a_range():
    node = bff.GaussianDistances("g")
    node.configure('{"number_of_components": 1, "axis_range": [1.0, 100.0, 3]}')
    np.testing.assert_allclose(node.get_axis(), [1.0, 10.0, 100.0])
    node = bff.PolymerDistances("p")
    node.configure('{"mode": "saw_nu", "axis": [10.0, 20.0, 35.0]}')
    np.testing.assert_allclose(node.get_axis(), [10.0, 20.0, 35.0])
    with pytest.raises(Exception):
        bff.GaussianDistances("bad").configure(
            '{"axis_range": [1.0, 100.0, 3], "axis_scale": "cubic"}')


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_builds_and_publishes_its_spectra(family):
    model = _spec(family).get_model()
    assert np.max(_curve(model)) > 0.0
    lifetimes = np.asarray(model.get_output("lifetime_spectrum"))
    distances = np.asarray(model.get_output("distance_distribution"))
    rates = np.asarray(model.get_output("fret_rates"))
    assert lifetimes.size >= 2 and lifetimes.size % 2 == 0
    # A distribution on the distance axis, or the discrete distances themselves.
    assert distances.size == (2 if family == "tcspc_fret_discrete" else 2 * 96)
    # k_FRET = 3/2 kappa2 / tau0 (R0 / r)^6, per distance.
    r0 = model.get_parameter("fret.forster_radius").value
    tau0 = model.get_parameter("fret.tau0").value
    kappa2 = model.get_parameter("fret.kappa2").value
    np.testing.assert_allclose(rates[1::2], 1.5 * kappa2 / tau0 * (r0 / distances[1::2]) ** 6, rtol=1e-12)
    np.testing.assert_allclose(rates[0::2], distances[0::2])


def test_the_search_finds_how_many_distances_made_the_data():
    truth = _spec("tcspc_fret_discrete").get_model()
    truth.select_structure("tcspc_fret_discrete.components.2")
    for canonical, value in (("distance.mean.0", 40.0), ("distance.amplitude.0", 0.5),
                             ("distance.mean.1", 60.0), ("distance.amplitude.1", 0.5),
                             ("instrument.n0", 60000.0), ("instrument.background", 0.001)):
        _set(truth, canonical, value)
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(_curve(truth)))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)

    problem = _spec("tcspc_fret_discrete", data).build()
    # Held: a donor-only fraction and a distance far beyond R0 are nearly the
    # same decay, and a search free to trade one for the other is not being
    # asked how many distances there are.
    problem.set_parameter_locked("fret.x_donly", True)
    search = bff.ModelSearch(problem)
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(30)
    config.set_dirichlet_fraction(0.0)
    config.set_seed(5)
    search.set_config(config)
    result = search.run()
    assert result.get_best_state().get_structure_key() == "tcspc_fret_discrete.components.2"
    problem.activate_state(result.get_best_state())
    np.testing.assert_allclose(
        sorted(problem.get_parameter(f"distance.mean.{i}").value for i in range(2)), [40.0, 60.0], rtol=1e-3)
