"""Training a dense network in IMP.bff: train_neural_net and its options.

Ported from tttrlib's NeuralNet::train (tttrlib keeps no ML code). The
document it returns is the `bff.neural_net` one IMP.bff.NeuralNet loads, so
every test here ends by evaluating the trained network through NeuralNet.
"""
import json

import numpy as np

import IMP
import IMP.test
import IMP.bff


def _regression(n=400, seed=0):
    """A smooth two-target regression on three inputs."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    Y = np.column_stack([
        np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2,
        X[:, 0] * X[:, 2] - 0.3 * X[:, 1],
    ])
    return X, Y


def _options(**kw):
    o = IMP.bff.NeuralNetTrainOptions()
    o.hidden_layer_sizes = [32, 32]
    o.max_iter = 200
    o.batch_size = 32
    o.learning_rate = 3e-3
    o.seed = 3
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _train(X, Y, opts):
    return IMP.bff.train_neural_net_with_history(
        X.ravel().tolist(), X.shape[0], X.shape[1],
        Y.ravel().tolist(), Y.shape[1], opts)


def _predict(doc, X):
    net = IMP.bff.NeuralNet(doc)
    y = np.asarray(net.predict(X.ravel().tolist(), X.shape[0]))
    return y.reshape(X.shape[0], net.get_n_outputs())


class Tests(IMP.test.TestCase):

    def test_defaults_are_sklearns(self):
        """The options default to scikit-learn's MLPRegressor"""
        o = IMP.bff.NeuralNetTrainOptions()
        self.assertEqual(list(o.hidden_layer_sizes), [256, 256, 128])
        self.assertEqual(o.activation, "relu")
        self.assertEqual(o.max_iter, 800)
        self.assertEqual(o.batch_size, 200)
        self.assertAlmostEqual(o.learning_rate, 1e-3)
        self.assertAlmostEqual(o.alpha, 1e-4)
        self.assertTrue(o.early_stopping)
        self.assertAlmostEqual(o.validation_fraction, 0.1)
        self.assertEqual(o.n_iter_no_change, 10)
        self.assertAlmostEqual(o.tol, 1e-4)

    def test_loss_decreases_and_network_fits(self):
        """Training lowers the loss and the network beats the mean"""
        X, Y = _regression()
        fit = _train(X, Y, _options(early_stopping=False))
        loss = np.asarray(fit.get_loss_curve())
        self.assertEqual(len(loss), 200)
        self.assertEqual(len(fit.get_validation_curve()), 0)
        self.assertLess(loss[-1], 0.2 * loss[0])

        pred = _predict(fit.get_network(), X)
        mse = ((pred - Y) ** 2).mean()
        baseline = ((Y - Y.mean(axis=0)) ** 2).mean()
        self.assertLess(mse, 0.1 * baseline)

    def test_document_loads_in_neuralnet(self):
        """The result is a bff.neural_net document with the stored scalers"""
        X, Y = _regression(n=120)
        doc = IMP.bff.train_neural_net(
            X.ravel().tolist(), X.shape[0], X.shape[1],
            Y.ravel().tolist(), Y.shape[1], _options(max_iter=5))
        d = json.loads(doc)
        self.assertEqual(d["format"], "bff.neural_net")
        self.assertEqual([l["n_in"] for l in d["layers"]], [3, 32, 32])
        self.assertEqual([l["activation"] for l in d["layers"]],
                         ["relu", "relu", "identity"])
        np.testing.assert_allclose(d["x_scaler"]["mean"], X.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(d["y_scaler"]["scale"], Y.std(axis=0), atol=1e-12)
        net = IMP.bff.NeuralNet(doc)
        self.assertEqual(net.get_n_inputs(), 3)
        self.assertEqual(net.get_n_outputs(), 2)
        self.assertEqual(_predict(doc, X).shape, (120, 2))

    def test_deterministic_per_seed(self):
        """The same seed gives the same network; another seed another one"""
        X, Y = _regression(n=150)
        a = _train(X, Y, _options(max_iter=20))
        b = _train(X, Y, _options(max_iter=20))
        c = _train(X, Y, _options(max_iter=20, seed=4))
        self.assertEqual(a.get_network(), b.get_network())
        self.assertEqual(list(a.get_loss_curve()), list(b.get_loss_curve()))
        self.assertNotEqual(a.get_network(), c.get_network())

    def test_early_stopping_keeps_a_validation_curve(self):
        """With early stopping the held-out curve is recorded and can stop early"""
        X, Y = _regression(n=300)
        fit = _train(X, Y, _options(max_iter=400, n_iter_no_change=3, tol=1e-2))
        n = fit.get_number_of_epochs()
        self.assertEqual(len(fit.get_validation_curve()), n)
        self.assertLess(n, 400)
        self.assertTrue(np.all(np.isfinite(fit.get_validation_curve())))

    def test_array_helper(self):
        """train_neural_net_arrays reads the shapes and accepts a 1D target"""
        X, Y = _regression(n=80)
        fit = IMP.bff.train_neural_net_arrays(X, Y[:, 0], _options(max_iter=3))
        self.assertEqual(IMP.bff.NeuralNet(fit.get_network()).get_n_outputs(), 1)

    def test_rejects_bad_input(self):
        """Inconsistent sizes and options are refused"""
        X, Y = _regression(n=20)
        with self.assertRaises(IMP.ValueException):
            IMP.bff.train_neural_net(X.ravel().tolist(), 21, 3,
                                     Y.ravel().tolist(), 2, _options())
        with self.assertRaises(IMP.ValueException):
            IMP.bff.train_neural_net(X.ravel().tolist(), 20, 3,
                                     Y.ravel().tolist(), 2, _options(batch_size=0))
        with self.assertRaises(IMP.ValueException):
            IMP.bff.train_neural_net(X.ravel().tolist(), 20, 3,
                                     Y.ravel().tolist(), 2,
                                     _options(activation="nope"))


if __name__ == '__main__':
    IMP.test.main()
