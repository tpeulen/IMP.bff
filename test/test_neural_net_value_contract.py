"""Independent contracts for the value-only forward path."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import IMP.bff

from test_neural_net_derivatives import _handmade_net


W1 = np.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.6]])
B1 = np.array([0.1, -0.2])
W2 = np.array([[0.7, -0.5], [0.2, 0.8]])
B2 = np.array([0.3, -0.1])


def _net(act="tanh", scaled=False):
    x_scaler = (np.array([0.1, -0.2, 0.3]), np.array([1.3, 0.7, 2.0])) if scaled else None
    y_scaler = (np.array([-0.4, 0.6]), np.array([0.8, 1.7])) if scaled else None
    return _handmade_net(W1, B1, W2, B2, act=act,
                         x_scaler=x_scaler, y_scaler=y_scaler)


def _activation(x, name):
    if name == "identity":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "softplus":
        return np.logaddexp(0.0, x)
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    if name == "sin":
        return np.sin(x)
    raise AssertionError(name)


def _reference(x, act, scaled=False):
    x = np.asarray(x, dtype=float)
    if scaled:
        x = (x - np.array([0.1, -0.2, 0.3])) / np.array([1.3, 0.7, 2.0])
    y = _activation(x @ W1.T + B1, act) @ W2.T + B2
    if scaled:
        y = y * np.array([0.8, 1.7]) + np.array([-0.4, 0.6])
    return y


def _network(x_scaler=False):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(32, 3))
    y = np.column_stack((x[:, 0] - 0.4 * x[:, 1], np.sin(x[:, 2])))
    options = IMP.bff.NeuralNetTrainOptions()
    options.hidden_layer_sizes = [8, 7]
    options.max_iter = 3
    options.early_stopping = False
    options.seed = 4
    net = IMP.bff.NeuralNet(
        IMP.bff.train_neural_net_arrays(x, y, options).get_network()
    )
    if x_scaler:
        # Training installs scalers; this branch still exercises a separate
        # scaled-input path when the implementation optimises unscaled input.
        assert net.get_n_inputs() == 3
    return net, x


@pytest.mark.parametrize("act", ["identity", "relu", "tanh", "logistic", "softplus", "silu", "sin"])
@pytest.mark.parametrize("scaled", [False, True])
def test_value_only_matches_independent_reference_and_derivative_forward(act, scaled):
    net = _net(act, scaled)
    x = np.random.default_rng(12).normal(size=(7, 3))
    before = x.copy()
    values = np.asarray(net.predict(x.ravel().tolist(), len(x))).reshape(len(x), -1)
    np.testing.assert_allclose(values, _reference(x, act, scaled), rtol=1e-12, atol=1e-12)
    direction = np.ones_like(x)
    derivative_forward = np.asarray(net.predict_derivatives(x, direction, 1)[0])
    np.testing.assert_allclose(derivative_forward, _reference(x, act, scaled), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)


def test_value_only_handles_batch_tails_and_alternating_networks_without_workspace_cross_talk():
    x = np.random.default_rng(13).normal(size=(5, 3))
    nets = [(_net("tanh", False), "tanh", False), (_net("sin", True), "sin", True)]

    def one(item):
        net, act, scaled = item
        out = []
        for n in (1, 3, 5, 2, 5):
            rows = x[:n]
            out.append(np.asarray(net.predict(rows.ravel().tolist(), n)).reshape(n, -1))
        return out, act, scaled

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, nets))
    for outputs, act, scaled in results:
        for n, value in zip((1, 3, 5, 2, 5), outputs):
            np.testing.assert_allclose(value, _reference(x[:n], act, scaled), rtol=1e-12, atol=1e-12)


def test_value_only_reuse_does_not_change_derivatives():
    net, x = _network()
    direction = np.ones_like(x)
    expected = net.predict_derivatives(x, direction, 2)
    for _ in range(4):
        net.predict(x.ravel().tolist(), len(x))
    actual = net.predict_derivatives(x, direction, 2)
    for before, after in zip(expected, actual):
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
