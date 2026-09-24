"""bff's network training against scikit-learn's MLPRegressor, head to head.

Section 7 of tttrlib's test_math_ab_numerics.py, which went with the network
when it moved to bff: the same data, the same hyper-parameters (Adam, ReLU,
L2 alpha, early stopping), and both must fit the function (R^2 > 0.98)
without bff falling far behind. Needs scikit-learn; skipped without it.
"""
import warnings

import numpy as np
import pytest

import IMP.bff


def test_training_reaches_sklearn_level_error():
    pytest.importorskip("sklearn")
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, size=(4000, 2))
    y = np.sin(X[:, 0]) * np.cos(0.5 * X[:, 1]) + 0.1 * X[:, 0] * X[:, 1]
    Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]

    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = [64, 64]
    opt.max_iter = 300
    opt.batch_size = 200
    opt.learning_rate = 1e-3
    opt.alpha = 1e-4
    opt.early_stopping = True
    opt.n_iter_no_change = 20
    opt.seed = 1
    net = IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(Xtr, ytr, opt).get_network())
    pred_ours = np.asarray(net.predict(Xte.ravel().tolist(), Xte.shape[0]))
    mse_ours = float(np.mean((pred_ours - yte) ** 2))

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr[:, None])
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                       learning_rate_init=1e-3, alpha=1e-4, batch_size=200, max_iter=300,
                       early_stopping=True, n_iter_no_change=20, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(xs.transform(Xtr), ys.transform(ytr[:, None]).ravel())
    pred = ys.inverse_transform(mlp.predict(xs.transform(Xte))[:, None])[:, 0]
    mse_skl = float(np.mean((pred - yte) ** 2))

    var = float(np.var(yte))
    assert mse_ours < 0.02 * var, f"bff mse {mse_ours:.3e} vs var {var:.3e}"
    assert mse_skl < 0.02 * var, f"sklearn mse {mse_skl:.3e} vs var {var:.3e}"
    assert mse_ours < 3.0 * mse_skl + 1e-4, f"bff {mse_ours:.3e} vs sklearn {mse_skl:.3e}"
