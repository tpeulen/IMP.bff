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


def _richardson(node, name, values, names, step=1e-3):
    """Central differences at h and h/2, extrapolated: error O(h^4).

    Tight enough to see a derivative wrong by a part in 1e5, which plain
    central differences at a safe step cannot.
    """
    def central(h_rel):
        columns = []
        for key in names:
            h = h_rel * max(abs(values[key]), 1.0)
            up, down = dict(values), dict(values)
            up[key] += h
            down[key] -= h
            columns.append((_weights(node, name, up) - _weights(node, name, down)) / (2 * h))
        return np.column_stack(columns)
    extrapolated = (4.0 * central(step / 2) - central(step)) / 3.0
    _weights(node, name, values)
    return extrapolated


def _worst(analytic, reference):
    """The largest error relative to the entry.

    Floored at 1e-5 of the column's largest entry: where a derivative crosses
    zero the reference's own roundoff (~1e-15 absolute) would otherwise read
    as a relative error, while an error of a part in 1e5 anywhere that matters
    still stands out.
    """
    floor = 1e-5 * np.max(np.abs(reference), axis=0) + 1e-300
    return float(np.max(np.abs(analytic - reference) / (np.abs(reference) + floor)))


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
    reference = _richardson(node, "g", values, names)
    assert _worst(analytic, reference) < 1e-7
    # The check sees an error of a part in 1e5 in a single entry.
    planted = analytic.copy()
    j = int(np.argmax(np.abs(planted[:, 0])))
    planted[j, 0] *= 1.0 + 1e-5
    assert _worst(planted, reference) > 5e-6
    # The weights sum to one whatever the parameters: every column sums to zero.
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=1e-12)


def test_a_negative_mean_or_amplitude_carries_its_sign():
    node = _gaussian(False)
    values = {"mean0": -41.0, "sigma0": 5.5, "shape0": 0.0, "amplitude0": -0.7,
              "mean1": 66.0, "sigma1": 9.0, "shape1": 0.0, "amplitude1": 0.3}
    _weights(node, "g", values)
    names = list(node.get_parameter_names())
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, len(names))
    assert _worst(analytic, _richardson(node, "g", values, names)) < 1e-7


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
    assert _worst(analytic, _richardson(node, "g", values, names)) < 1e-7


def test_discrete_weights_jacobian():
    node = bff.DiscreteDistances("d")
    node.set_number_of_distances(3)
    node.add_output_port("d", bff.GraphPort([0.0], False, True))
    values = {"distance0": 40.0, "amplitude0": 0.5, "distance1": 55.0, "amplitude1": -0.2,
              "distance2": 70.0, "amplitude2": 0.3}
    _weights(node, "d", values)
    names = list(node.get_parameter_names())
    analytic = np.asarray(node.get_weights_jacobian()).reshape(3, len(names))
    assert _worst(analytic, _richardson(node, "d", values, names)) < 1e-7


POLYMERS = [
    ("worm_like_chain", {"chain_length": 120.0, "persistence_length": 25.0, "sigma_linker": 6.0}),
    ("worm_like_chain_linker", {"chain_length": 120.0, "persistence_length": 25.0, "sigma_linker": 6.0}),
    ("saw_nu", {"r_rms": 55.0, "nu": 0.55}),
    ("ising_chain", {"n_residues": 40.0, "b_structured": 4.0, "b_unstructured": 8.0,
                     "coupling": 1.5, "field": 0.2}),
]


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode, values", POLYMERS)
def test_polymer_weights_jacobian_is_the_derivative_of_the_output(mode, values, normalize):
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
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, len(names))
    # The residue count is an integer and the linker width unread without the
    # linker: their columns are 0 and are not differenced.
    skip = {"n_residues"} | ({"sigma_linker"} if mode == "worm_like_chain" else set())
    varied = [c for c, name in enumerate(names) if name not in skip]
    for c, name in enumerate(names):
        if name in skip:
            assert not analytic[:, c].any()
    # The best of several steps: truncation falls with the step and roundoff
    # grows with it, and where they cross depends on the kernel.
    references = [_richardson(node, "p", values, [names[c] for c in varied], step)
                  for step in (3e-3, 1e-3, 3e-4, 1e-4)]
    errors = [_worst(analytic[:, varied], reference) for reference in references]
    best = int(np.argmin(errors))
    # The Ising transform integrates an oscillating k sin(kR) phi(k) over 2000
    # points, and its finite differences bottom out near 1e-7 in roundoff
    # (smaller steps are worse, larger ones truncate); the other kernels'
    # references converge to 1e-9 or better.
    assert errors[best] < (5e-7 if mode == "ising_chain" else 1e-7)
    planted = analytic[:, varied].copy()
    j = int(np.argmax(np.abs(planted[:, 0])))
    planted[j, 0] *= 1.0 + 1e-5
    assert _worst(planted, references[best]) > 5e-6
    if normalize:
        scale = np.abs(analytic).max()
        np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=1e-10 * scale + 1e-14)


def _tabulated(threshold=0.0):
    node = bff.TabulatedDistances("t")
    node.configure('{"number_of_distributions": 2, "threshold": %r}' % threshold)
    node.add_output_port("t", bff.GraphPort([0.0], False, True))
    node.add_input_port("axis", bff.GraphPort(list(GRID)))
    node.add_input_port("distribution0", bff.GraphPort(list(np.exp(-0.5 * ((GRID - 45) / 5) ** 2))))
    node.add_input_port("distribution1", bff.GraphPort(list(3.0 * np.exp(-0.5 * ((GRID - 70) / 9) ** 2))))
    node.add_input_port("fraction0", bff.GraphPort(0.6))
    node.add_input_port("fraction1", bff.GraphPort(0.4))
    return node


def test_tabulated_distributions_mix_as_shares_of_the_ensemble():
    node = _tabulated()
    node.update()
    out = np.asarray(node.get_output_port("t").value)
    q0 = np.exp(-0.5 * ((GRID - 45) / 5) ** 2)
    q1 = np.exp(-0.5 * ((GRID - 70) / 9) ** 2)
    expected = 0.6 * q0 / q0.sum() + 0.4 * q1 / q1.sum()
    np.testing.assert_allclose(out[0::2], expected / expected.sum(), rtol=1e-12)
    np.testing.assert_allclose(out[1::2], GRID)


@pytest.mark.parametrize("threshold", [0.0, 1e-3])
def test_tabulated_weights_jacobian(threshold):
    node = _tabulated(threshold)
    values = {"fraction0": 0.6, "fraction1": -0.4}
    _weights(node, "t", values)
    names = list(node.get_parameter_names())
    assert names == ["fraction0", "fraction1"]
    analytic = np.asarray(node.get_weights_jacobian()).reshape(GRID.size, 2)
    reference = _richardson(node, "t", values, names)
    assert _worst(analytic, reference) < 1e-7
    planted = analytic.copy()
    planted.flat[int(np.argmax(np.abs(planted)))] *= 1 + 1e-5
    assert _worst(planted, reference) > 5e-6
