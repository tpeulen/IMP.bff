"""The TCSPC lifetime family, now read from a description.

The assertions are the ones the C++ factory had to satisfy: the same
topologies, the same canonical registry underneath all of them, a search that
finds the generating component count, and a refusal rather than a fallback
when a caller names a parameter the family does not have.  Only the way the
family is built has changed.
"""

import sys
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _problem():
    return _fixtures.tcspc_lifetime()


def _spec():
    """The description plus its measurements, before it is built."""
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def test_every_topology_reads_one_canonical_parameter_registry():
    problem = _problem()
    assert list(problem.get_structure_keys()) == [
        "lifetime.components.1",
        "lifetime.components.2",
        "lifetime.components.3",
    ]
    # Every candidate graph follows the same owner port rather than holding a
    # copy of its value, which is what makes switching topology safe.
    tau0 = problem.get_parameter("lifetime.tau.0")
    for key in problem.get_structure_keys():
        objective = problem.get_structure_objective(key)
        decay = objective.get_input_port("model").link.get_node()
        assert decay is not None
        assert decay.get_input_port("t0").link.uid == tau0.uid


def test_search_uses_native_objectives_and_activates_a_complete_topology():
    problem = _problem()
    search = bff.ModelSearch(problem)
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(40)
    config.set_dirichlet_fraction(0.0)
    config.set_seed(3)
    search.set_config(config)

    result = search.run()

    assert result.get_best_state().get_structure_key() == "lifetime.components.2"
    active = problem.get_active_objective()
    assert active is not None
    active.update()
    assert np.isfinite(np.asarray(active.get_output_port("residuals").value)).all()


def test_an_unknown_parameter_is_refused_instead_of_falling_back():
    spec = _spec()
    spec.set_parameter("chisurf.callback", 1.0, True, 0.0, 2.0)
    # A caller override that matches nothing is a caller talking about a
    # different model; building anyway would fit without it and say nothing.
    with pytest.raises((ValueError, RuntimeError)):
        spec.build()


def test_a_description_cannot_be_built_without_its_measurements():
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    # The IRF is optional since it can be modelled instead of measured
    # (`generated_response`); the decay itself cannot be done without.
    assert set(spec.get_dataset_names()) == {"decay"}
    assert set(spec.get_scalar_names()) == {"dt", "period"}
    with pytest.raises((ValueError, RuntimeError)):
        spec.build()


def test_without_a_measured_or_modelled_response_the_decay_says_so():
    """An IRF that is neither loaded nor modelled is refused where it is used."""
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", _fixtures._tcspc_dataset()[0])
    spec.set_scalar("dt", 0.05)
    spec.set_scalar("period", 12.5)
    problem = spec.build()
    with pytest.raises((ValueError, RuntimeError)) as caught:
        problem.get_initial_state()
    assert "response" in str(caught.value)


def test_the_instrument_is_supplied_by_the_caller_not_the_description():
    """A channel width belongs to a measurement, so the file cannot hold it."""
    spec = _spec()
    spec.set_scalar("dt", 0.0)
    spec.set_scalar("period", 0.0)
    with pytest.raises((ValueError, RuntimeError)):
        spec.build()
