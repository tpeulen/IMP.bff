"""FP4 training: NeuralNetTrainOptions.precision = "nvfp4" / "mxfp4".

The recipe (NVIDIA's NVFP4 pretraining, arXiv 2509.25149) is
internal/MlpFp4Train.h; its pieces -- the Hadamard transform, wgrad and
dgrad against a float GEMM of the same quantised operands, determinism per
seed, stochastic rounding's unbiasedness -- are checked in C++
(cpp_snippets/test_fp4_kernels.cpp, run by test_neural_net_fp4.py). Here:
the held-out loss of FP4-trained networks against float64-trained ones on a
small regression and on the HMM surrogate's training set, determinism, and
that the trained FP4 inference network is what training evaluated.
"""

import numpy as np
import pytest

import IMP.bff


def _opts(precision, seed=1, max_iter=60, hidden=(64, 64, 64)):
    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = list(hidden)
    opt.activation = "tanh"
    opt.max_iter = max_iter
    opt.seed = seed
    opt.precision = precision
    return opt


def _regression(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    Y = np.column_stack([np.sin(X[:, 0]) * np.cos(0.5 * X[:, 1]) + 0.1 * X[:, 2] * X[:, 3],
                         np.tanh(X[:, 1] - X[:, 3])])
    return X, Y


def _mse(net, X, Y):
    pred = np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(Y.shape)
    return float(np.mean((pred - Y) ** 2))


def _train(X, Y, opt):
    return IMP.bff.train_neural_net_arrays(X, Y, opt)


# measured (okf/neural-net.md): held-out MSE of the FP4-trained network
# (W4A4 inference) over the float64-trained one on the regression below,
# nvfp4 3.1x, mxfp4 3.5x with the SplitMix64 SR stream (2026-09-24), 3.04x /
# 3.10x with the counter-based one (2026-09-25); both < 1 % of the target
# variance; the bounds leave ~1.6x room
RATIO_BOUND = {"nvfp4": 5.0, "mxfp4": 6.0}


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp4"])
def test_fp4_training_against_float64_on_a_regression(precision):
    X, Y = _regression(3000, 0)
    Xte, Yte = _regression(1000, 1)
    ref = _train(X, Y, _opts("float64"))
    fp4 = _train(X, Y, _opts(precision))
    assert ref.get_precision() == "float64" and fp4.get_precision() == precision
    assert ref.get_quantized_network() == b""
    net_ref = IMP.bff.NeuralNet(ref.get_network())
    net_master = IMP.bff.NeuralNet(fp4.get_network())
    q = IMP.bff.QuantizedNeuralNet.from_msgpack(fp4.get_quantized_network())
    assert q.get_format() == precision and q.get_quantize_activations()
    m_ref, m_master, m_q = _mse(net_ref, Xte, Yte), _mse(net_master, Xte, Yte), _mse(q, Xte, Yte)
    var = float(np.mean((Yte - Yte.mean(axis=0)) ** 2))
    print("\n%s training, 4-64-64-64-2 tanh, held-out MSE (target variance %.3g):" % (precision, var))
    print("  float64 %.4g | %s FP4 network %.4g (%.2fx) | its float64 master weights %.4g"
          " | epochs %d vs %d, bits/weight %.2f"
          % (m_ref, precision, m_q, m_q / m_ref, m_master, ref.get_number_of_epochs(),
             fp4.get_number_of_epochs(), q.get_bits_per_weight()))
    assert m_q < 0.02 * var                      # it learned
    assert m_q <= RATIO_BOUND[precision] * m_ref, (m_q, m_ref)


def test_fp4_training_is_deterministic_per_seed():
    X, Y = _regression(800, 2)
    a = _train(X, Y, _opts("nvfp4", seed=4, max_iter=5))
    b = _train(X, Y, _opts("nvfp4", seed=4, max_iter=5))
    c = _train(X, Y, _opts("nvfp4", seed=5, max_iter=5))
    assert a.get_network() == b.get_network()
    assert a.get_quantized_network() == b.get_quantized_network()
    np.testing.assert_array_equal(a.get_loss_curve(), b.get_loss_curve())
    assert a.get_network() != c.get_network()


def test_the_quantized_network_is_the_trained_forward_pass():
    """Its FP4 layers are the 2-D scaled weights fprop used and it runs W4A4;
    the first and last layers stay float64 (the paper's sensitive layers)."""
    import msgpack
    X, Y = _regression(800, 3)
    t = _train(X, Y, _opts("nvfp4", max_iter=5))
    doc = msgpack.unpackb(t.get_quantized_network(), raw=False)
    kinds = ["float64" if l.get("precision") == "float64" else "fp4" for l in doc["layers"]]
    assert kinds == ["float64", "fp4", "fp4", "float64"]
    q = IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network())
    assert q.to_msgpack() == t.get_quantized_network()
    master = IMP.bff.NeuralNet(t.get_network())
    rel = abs(_mse(q, X, Y) - _mse(master, X, Y)) / _mse(master, X, Y)
    print("\nFP4 network vs its master weights, training-set MSE: %.3g relative" % rel)
    # every layer in FP4
    o = _opts("nvfp4", max_iter=5)
    o.fp4_keep_first_layer = False
    o.fp4_keep_last_layer = False
    doc = msgpack.unpackb(_train(X, Y, o).get_quantized_network(), raw=False)
    assert all("codes" in l for l in doc["layers"])


def test_recipe_switches_and_refusals():
    X, Y = _regression(600, 4)
    for sw in ("fp4_hadamard", "fp4_stochastic_rounding"):
        o = _opts("nvfp4", max_iter=3)
        setattr(o, sw, False)
        t = _train(X, Y, o)
        assert np.isfinite(t.get_loss_curve()).all()
    with pytest.raises(ValueError):
        _train(X, Y, _opts("fp8", max_iter=1))


def _has_surrogate():
    try:
        import tttrlib  # noqa: F401
    except ImportError:
        return False
    return hasattr(IMP.bff, "HmmSurrogate")


def test_fp4_training_on_the_hmm_surrogate_set():
    if not _has_surrogate():
        pytest.skip("IMP.bff built without tttrlib: no HmmSurrogate")
    gen = lambda n, seed: IMP.bff.HmmSurrogate.generate_training_set(  # noqa: E731
        2, 2, n_samples=n, n_bursts=150, burst_len=80, mean_dt=4.0, seed=seed)
    X, Y = gen(400, 12345)
    Xte, Yte = gen(100, 12345 + 999)
    import time
    res, secs = {}, {}
    for p in ("float64", "nvfp4", "mxfp4"):
        o = _opts(p, seed=0, max_iter=200, hidden=(256, 256, 128))
        t0 = time.perf_counter()
        t = _train(X, Y, o)
        secs[p] = (time.perf_counter() - t0) / t.get_number_of_epochs()
        net = (IMP.bff.NeuralNet(t.get_network()) if p == "float64"
               else IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network()))
        pred = np.asarray(net.predict(Xte.ravel().tolist(), Xte.shape[0])).reshape(Yte.shape)
        res[p] = float(np.mean(np.abs(pred - Yte)))
    print("\nHMM surrogate (%s -> %s, 256-256-128), held-out MAE: float64 %.4f | nvfp4 %.4f | mxfp4 %.4f"
          % (X.shape, Y.shape, res["float64"], res["nvfp4"], res["mxfp4"]))
    print("  seconds an epoch: float64 %.3f | nvfp4 %.3f | mxfp4 %.3f" % (
        secs["float64"], secs["nvfp4"], secs["mxfp4"]))
    # measured: nvfp4 1.09x, mxfp4 1.08x the float64 MAE (2026-09-24);
    # 1.04x / 1.07x with the counter-based SR (2026-09-25)
    assert res["nvfp4"] <= 1.3 * res["float64"], res
    assert res["mxfp4"] <= 1.3 * res["float64"], res
