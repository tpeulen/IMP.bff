"""The analytical FCS family, now read from a description.

Same contracts the C++ factory had to meet: the canonical registry in its
declared order, the full topology family over it, a real fit through the
native graph with no Python callback, and refusals where a description or a
measurement does not add up.
"""

import sys
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402

#: The per-point error the single-topology fixture is measured with.
ERROR = 0.002


def test_the_description_builds_the_whole_family_over_one_registry():
    problem = _fixtures.fcs_analytical()

    assert list(problem.get_parameter_ids()) == [
        "fcs.N",
        "fcs.baseline",
        "fcs.structure_parameter",
        "fcs.diffusion_time.1",
        "fcs.diffusion_fraction.1",
        "fcs.diffusion_time.2",
        "fcs.relaxation_amplitude.1",
        "fcs.relaxation_time.1",
    ]
    assert set(problem.get_structure_keys()) == {
        f"fcs.{dimension}d.{components}diff.{relaxations}relax"
        for dimension in (2, 3)
        for components in (1, 2)
        for relaxations in (0, 1)
    }

    # Each topology reads the exact same owner port through a follower. A
    # topology switch therefore cannot transfer or name-match parameters.
    owner = problem.get_parameter("fcs.N")
    for key in problem.get_structure_keys():
        objective = problem.get_structure_objective(key)
        model = objective.get_input_port("model").link.get_node()
        assert model is not None  # the problem owns the complete graph
        assert model.get_input_port("N").link.uid == owner.uid


def test_single_2d_topology_fits_without_a_python_model_callback():
    problem = _fixtures.fcs_two_dimensional_single()
    search_config = bff.ModelSearchConfig()
    search_config.set_number_of_simulations(12)
    search_config.set_dirichlet_fraction(0.0)
    search_config.set_seed(11)
    search = bff.ModelSearch(problem)
    search.set_config(search_config)

    result = search.run()

    assert result.get_best_state().get_structure_key() == "fcs.2d.1diff.0relax"

    # Recovered to better than a part in a hundred, which is a statement about
    # the fit rather than about this machine. The thresholds these replaced
    # were absolute and sat just above what one platform happened to reach --
    # the residual one with 10% headroom against a cross-platform spread of
    # the same order, which is a CI failure waiting rather than a contract.
    for name, truth in (("fcs.N", 3.2), ("fcs.baseline", 0.97),
                        ("fcs.diffusion_time.1", 0.42)):
        assert problem.get_parameter(name).value == pytest.approx(truth, rel=1e-2), name

    objective = problem.get_active_objective()
    objective.update()
    residuals = np.asarray(objective.get_output_port("residuals").value)
    # The model tracks the curve well inside the noise it was measured with:
    # the comparison that means something is against the errors, not against
    # a constant. Measured at a fifteenth of them.
    assert np.max(np.abs(residuals * ERROR)) < 0.2 * ERROR


def test_seeds_are_read_from_the_measurement_not_written_in_the_file():
    """One description, two curves, two different starting points.

    The baseline seed is the last lag of the curve itself, which still holds
    a little correlation -- the least contaminated estimate available before
    a model exists, not the true plateau. So this checks the seed tracks the
    measurement it was read from, which is the property that matters: nothing
    about either curve is written in the file.
    """
    axis = np.geomspace(1.0e-3, 20.0, 80)
    seeds = []
    for baseline, td in ((1.0, 0.7), (5.0, 3.0)):
        curve = _fixtures.fcs_curve_2d(axis, baseline=baseline, td=td)
        spec = bff.ModelSearchSpec.from_name("fcs_analytical")
        spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.01))
        problem = spec.build()
        seeds.append(
            (
                problem.get_parameter("fcs.baseline").value,
                problem.get_parameter("fcs.diffusion_time.1").value,
                curve[-1],
            )
        )
    for seeded_baseline, _, last_lag in seeds:
        assert abs(seeded_baseline - last_lag) < 1e-9
    # A four-fold slower curve must seed a longer diffusion time.
    assert seeds[0][1] < seeds[1][1]
    assert seeds[0][0] < seeds[1][0]


def test_a_description_without_its_measurement_builds_nothing():
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    assert list(spec.get_dataset_names()) == ["curve"]
    with pytest.raises((ValueError, RuntimeError)):
        spec.build()


def test_a_description_that_does_not_add_up_is_refused():
    broken = """
    {"schema": "bff.model_search.v1", "family": "broken",
     "parameters": {"a": {"initial": 1.0, "lower": 0.0, "upper": 2.0}},
     "initial_structure": "only",
     "structures": {"only": {
        "nodes": {"m": {"type": "GraphExpression",
                        "config": {"expression": "a*x"},
                        "inputs": {"a": "#a", "x": "#nonexistent"}}},
        "objective": "m", "free": ["a"]}}}
    """
    spec = bff.ModelSearchSpec.from_json(broken)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        spec.build()
    assert "nonexistent" in str(caught.value)


def test_an_unknown_node_type_names_what_is_available():
    broken = """
    {"schema": "bff.model_search.v1", "family": "broken",
     "parameters": {"a": {"initial": 1.0, "lower": 0.0, "upper": 2.0}},
     "initial_structure": "only",
     "structures": {"only": {
        "nodes": {"m": {"type": "GraphExpresion"}},
        "objective": "m", "free": ["a"]}}}
    """
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.ModelSearchSpec.from_json(broken).build()
    assert "GraphExpression" in str(caught.value)
