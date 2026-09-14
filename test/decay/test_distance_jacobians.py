"""The derivative of a distance distribution's weights by its parameters.

A caller that fits a distribution by Fisher scoring (BayesianDecayModel, on
its own grid) multiplies the output weights straight into its transfer
tensors, so what it needs is d(weights)/d(parameters) of exactly what the node
outputs, normalisation included, row-major. Each claim is checked against
central differences of the node's own output, on a non-uniform grid like the
one that caller owns.
"""

import numpy as np
import pytest

import IMP.bff as bff

GRID = 52.0 * np.linspace(0.3, 2.2, 128) ** 1.3        # non-uniform, in Angstrom


def _weights(node, name, values):
    for key, value in values.items():
        node.get_input_port(key).value = value
    node.update()
    return np.asarray(node.get_output_port(name).value)[0::2]


def _numeric(node, name, values, names, step=1e-6):
    columns = []
    for key in names:
        # A shape step of 1e-6 is below what `log(1 - k t) / k` resolves.
        h = (1e-4 if key.startswith("shape") else step) * max(abs(values[key]), 1.0)
        up, down = dict(values), dict(values)
        up[key] += h
        down[key] -= h
        columns.append((_weights(node, name, up) - _weights(node, name, down)) / (2 * h))
    _weights(node, name, values)
    return np.column_stack(columns)


def _gaussian(two_cloud):
    node = bff.GaussianDistances("g")
    node.set_number_of_components(2)
    node.set_distance_between_gaussians(two_cloud)
    node.set_axis_array(np.ascontiguousarray(GRID))
    node.add_output_port("g", bff.GraphPort([0.0], False, True))
    return node


@pytest.mark.parametrize("two_cloud", [False, True])
@pytest.mark.parametrize("shape", [0.0, 0.15])
def test_gaussian_weights_jacobian_is_the_derivative_of_the_output(two_cloud, shape):
    node = _gaussian(two_cloud)
    values = {"mean0": 41.0, "sigma0": 5.5, "shape0": shape, "amplitude0": 0.7,
              "mean1": 66.0, "sigma1": 9.0, "shape1": -shape, "amplitude1": 0.3}
    names = list(node.get_parameter_names())
    assert names == ["mean0", "sigma0", "shape0", "amplitude0", "mean1", "sigma1", "shape1", "amplitude1"]
    weights = _weights(node, "g", values)
    assert weights.sum() == pytest.approx(1.0)
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, len(names))
    np.testing.assert_allclose(analytic, _numeric(node, "g", values, names), rtol=1e-4, atol=1e-8)
    # The weights sum to one whatever the parameters: every column sums to zero.
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=1e-12)


def test_a_negative_mean_or_amplitude_carries_its_sign():
    node = _gaussian(False)
    values = {"mean0": -41.0, "sigma0": 5.5, "shape0": 0.0, "amplitude0": -0.7,
              "mean1": 66.0, "sigma1": 9.0, "shape1": 0.0, "amplitude1": 0.3}
    _weights(node, "g", values)
    names = list(node.get_parameter_names())
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, len(names))
    np.testing.assert_allclose(analytic, _numeric(node, "g", values, names), rtol=1e-4, atol=1e-8)


def test_a_small_shape_takes_the_series_without_a_jump():
    node = _gaussian(False)
    base = {"mean0": 41.0, "sigma0": 5.5, "amplitude0": 0.7,
            "mean1": 66.0, "sigma1": 9.0, "shape1": 0.0, "amplitude1": 0.3}
    columns = []
    for shape in (0.0, 1e-9, 1e-7):
        _weights(node, "g", {**base, "shape0": shape})
        columns.append(np.asarray(node.get_weights_jacobian()).reshape(GRID.size, 8)[:, 2])
    for column in columns[1:]:
        np.testing.assert_allclose(column, columns[0], rtol=1e-5, atol=1e-8)
    # A shape where grid points near the mean take the series and the rest the
    # closed form: one derivative across the switch.
    values = {**base, "shape0": 4e-3}
    _weights(node, "g", values)
    names = list(node.get_parameter_names())
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, 8)
    np.testing.assert_allclose(analytic, _numeric(node, "g", values, names), rtol=1e-4, atol=1e-8)


def test_discrete_weights_jacobian():
    node = bff.DiscreteDistances("d")
    node.set_number_of_distances(3)
    node.add_output_port("d", bff.GraphPort([0.0], False, True))
    values = {"distance0": 40.0, "amplitude0": 0.5, "distance1": 55.0, "amplitude1": -0.2,
              "distance2": 70.0, "amplitude2": 0.3}
    _weights(node, "d", values)
    names = list(node.get_parameter_names())
    analytic = np.asarray(node.get_weights_jacobian()).reshape(3, len(names))
    np.testing.assert_allclose(analytic, _numeric(node, "d", values, names), rtol=1e-6, atol=1e-10)


POLYMERS = [
    ("worm_like_chain", {"chain_length": 120.0, "persistence_length": 25.0, "sigma_linker": 6.0}),
    ("worm_like_chain_linker", {"chain_length": 120.0, "persistence_length": 25.0, "sigma_linker": 6.0}),
    ("saw_nu", {"r_rms": 55.0, "nu": 0.55}),
    ("ising_chain", {"n_residues": 40.0, "b_structured": 4.0, "b_unstructured": 8.0,
                     "coupling": 1.5, "field": 0.2}),
]


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode, values", POLYMERS)
def test_polymer_weights_jacobian_is_stable_in_its_step(mode, values, normalize):
    node = bff.PolymerDistances("p")
    node.set_mode(mode)
    node.set_normalize_weights(normalize)
    node.set_axis_array(np.ascontiguousarray(GRID))
    node.add_output_port("p", bff.GraphPort([0.0], False, True))
    weights = _weights(node, "p", values)
    if normalize:
        assert weights.sum() == pytest.approx(1.0)
    names = list(node.get_parameter_names())
    assert names == list(values)
    jacobian = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, len(names))
    half = np.asarray(node.get_weights_jacobian(0.5e-5)).reshape(GRID.size, len(names))
    scale = max(np.abs(jacobian).max(), 1e-12)
    assert np.abs(jacobian - half).max() < 1e-6 * scale + 1e-10
    if normalize:
        np.testing.assert_allclose(jacobian.sum(axis=0), 0.0, atol=1e-8 * scale + 1e-12)
    if mode == "ising_chain":
        assert not jacobian[:, 0].any()
    if mode == "worm_like_chain":
        assert not jacobian[:, 2].any()
    assert np.abs(jacobian).max() > 0.0
