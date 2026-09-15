"""A description names a Python node type like a C++ one.

The C++ path stays slim: what is not a hot loop -- a report, a table a
structure derives from coordinates, a solver compiled elsewhere -- can be a
Python `GraphNode` subclass, registered by name, and then used by any
description. These pin that it builds, evaluates inside a fit, outlives the
Python references that made it, and fails loudly.
"""

import gc
import json

import numpy as np
import pytest

import IMP.bff as bff


@bff.register_node_type("test.PythonLine")
class PythonLine(bff.GraphNode):
    """slope * x, over the dataset's coordinate."""

    def evaluate(self):
        slope = self.get_input_port("slope").get_value()
        x = np.asarray(self.get_input_port("x").get_value_view(), dtype=float)
        self.get_output_port(self.get_name()).set_value_vector((slope * x).tolist())


X = np.linspace(0.0, 1.0, 21)


def _spec(node_type="test.PythonLine"):
    document = {
        "schema": "bff.model_search.v1",
        "family": "python_line",
        "datasets": ["curve"],
        "initial_structure": "line",
        "actions": [{"from": "line", "action": "fit-and-stop", "to": "line", "prior": 1.0, "terminal": True}],
        "parameters": {"line.slope": {"name": "slope", "initial": 1.0, "lower": -10.0, "upper": 10.0}},
        "structures": {
            "line": {
                "nodes": {
                    "model": {"type": node_type,
                              "inputs": {"slope": "#line.slope", "x": "@curve.x"},
                              "outputs": {"@name": "float_vector"}},
                    "objective": {"type": "FitChiSquared", "bind": {"data": "curve"},
                                  "inputs": {"model": "model"},
                                  "outputs": {"@name": "float", "residuals": "float_vector"}},
                },
                "free": ["line.slope"],
                "objective": "objective",
                "ess": "curve_size",
                "selection": "bic",
            }
        },
    }
    dataset = bff.FitDataset()
    dataset.set_values_array(np.ascontiguousarray(3.0 * X))
    dataset.set_coordinate_array(0, "x", np.ascontiguousarray(X))
    dataset.set_noise_family(bff.FIT_NOISE_FAMILY_STORED)
    dataset.set_stored_variance_array(np.full(X.size, 0.01))
    spec = bff.ModelSearchSpec.from_json(json.dumps(document))
    spec.set_dataset("curve", dataset)
    return spec


def test_a_registered_python_type_is_a_node_type():
    assert bff.GraphNodeRegistry.has_type("test.PythonLine")
    node = bff.GraphNodeRegistry.create("test.PythonLine", "made")
    assert node.get_name() == "made"


def test_a_description_builds_and_evaluates_a_python_node_after_its_references_are_gone():
    problem = _spec().build()
    gc.collect()  # nothing on the Python side holds the node any more
    problem.activate_structure("line")
    problem.get_parameter("line.slope").value = 3.0
    model = np.asarray(problem.get_structure_output("line", problem.get_structure_curve_node("line", "curve")), dtype=float)
    np.testing.assert_allclose(model, 3.0 * X)


def test_a_fit_moves_the_parameter_through_the_python_node():
    problem = _spec().build()
    gc.collect()
    problem.get_parameter("line.slope").value = 0.5
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(2)
    config.set_seed(1)
    search = bff.ModelSearch(problem)
    search.set_config(config)
    best = search.run().get_best_state()
    problem.activate_state(best)
    assert problem.get_parameter("line.slope").value == pytest.approx(3.0, rel=1e-5)


def test_a_factory_that_raises_names_the_type_and_the_error():
    def broken(name):
        raise RuntimeError("no such detector")

    bff.register_node_type("test.Broken", broken)
    with pytest.raises(ValueError, match=r"test\.Broken.*no such detector"):
        bff.GraphNodeRegistry.create("test.Broken", "x")


def test_a_factory_must_return_a_node():
    bff.register_node_type("test.NotANode", lambda name: 42)
    with pytest.raises(ValueError, match="must return an IMP.bff.GraphNode"):
        bff.GraphNodeRegistry.create("test.NotANode", "x")
