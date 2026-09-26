"""Ternary networks (BitNet b1.58, W1.58A8): QuantizedNeuralNet's ternary
formats and NeuralNetTrainOptions.precision = "ternary".

The formats, quantisers and kernels are C++ (internal/MlpTernary.h,
internal/MlpTernaryTrain.h). Checked here:

* the C++ snippet cpp_snippets/test_ternary_kernels.cpp -- quantiser
  exactness (absmean, round-half-even ties, clamp), TQ2 / TQ1 packing round
  trips, the kernel against the element-by-element reference, predict
  against float64 arithmetic on the dequantised operands, training's
  pieces, and one fingerprint for every SIMD variant (built the ways
  test_neural_net_fp4.py builds the FP4 snippet);
* the formats through the module: sizes (bits a weight), the msgpack round
  trip, the refusals;
* accuracy: post-training quantisation (PTQ) of a float-trained network
  against quantisation-aware training (QAT) with precision="ternary", on a
  64-wide regression and on the HMM surrogate's training set; determinism;
  the trained ternary network is training's forward pass;
* speeds are in bench_neural_net_fp4.py (reported, never asserted).
"""

import os
import sys

import numpy as np
import pytest

import IMP.bff

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_neural_net_fp4 import _build_and_run, _variants, _fixture  # noqa: E402

FORMATS = ["ternary", "ternary_row", "ternary_tq1", "ternary_tq1_row"]


def _batch(net, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(X.shape[0], net.get_n_outputs())


def _opts(precision, seed=1, max_iter=60, hidden=(64, 64, 64), **kw):
    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = list(hidden)
    opt.activation = "tanh"
    opt.max_iter = max_iter
    opt.seed = seed
    opt.precision = precision
    for k, v in kw.items():
        setattr(opt, k, v)
    return opt


def _regression(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    Y = np.column_stack([np.sin(X[:, 0]) * np.cos(0.5 * X[:, 1]) + 0.1 * X[:, 2] * X[:, 3],
                         np.tanh(X[:, 1] - X[:, 3])])
    return X, Y


def _mse(net, X, Y):
    return float(np.mean((_batch(net, X) - Y) ** 2))


def _train(X, Y, opt):
    return IMP.bff.train_neural_net_arrays(X, Y, opt)


# ---------------------------------------------------------------------------
# The C++ snippet, every variant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,runner,flags,want", _variants(), ids=[v[0] for v in _variants()])
def test_ternary_kernels_cpp_every_variant(name, runner, flags, want):
    run = _build_and_run("test_ternary_kernels.cpp", name, runner, flags)
    print("\n" + "\n".join(l for l in run.stdout.splitlines() if l.startswith("  ")))
    assert run.stdout.splitlines()[0] == "variant " + want
    assert run.returncode == 0, run.stdout
    assert "0 failure(s)" in run.stdout


# ---------------------------------------------------------------------------
# The formats
# ---------------------------------------------------------------------------

def _wide_net():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(256, 24))
    Y = rng.normal(size=(256, 8))
    return IMP.bff.NeuralNet(_train(X, Y, _opts("float64", max_iter=1, hidden=(256, 256, 128))).get_network())


def test_sizes_and_round_trip():
    net = _wide_net()
    X = np.random.default_rng(1).normal(size=(33, 24))
    bits = {}
    for f in FORMATS:
        q = IMP.bff.QuantizedNeuralNet(net, f)
        assert q.get_format() == f and q.get_quantize_activations()
        assert q.get_n_weights() == 24 * 256 + 256 * 256 + 256 * 128 + 128 * 8
        bits[f] = q.get_bits_per_weight()
        again = IMP.bff.QuantizedNeuralNet.from_msgpack(q.to_msgpack())
        assert again.to_msgpack() == q.to_msgpack()
        np.testing.assert_array_equal(_batch(again, X), _batch(q, X))
        assert "W4A4" not in repr(q)
    print("\nbits a weight, 24-256-256-128-8:", {k: round(v, 3) for k, v in bits.items()})
    # exactly: packed rows (padded to 4 / 5 trits) plus a float64 scale a
    # tensor (or a row); 2.002 / 1.630 / 2.393 / 2.021 bits a weight here
    shapes = [(24, 256), (256, 256), (256, 128), (128, 8)]
    n = sum(i * o for i, o in shapes)
    for f, per, row in (("ternary", 4, False), ("ternary_tq1", 5, False),
                        ("ternary_row", 4, True), ("ternary_tq1_row", 5, True)):
        nbytes = sum(o * (-(-i // per)) + 8 * (o if row else 1) for i, o in shapes)
        assert bits[f] == pytest.approx(8.0 * nbytes / n, rel=1e-12), f
    assert 2.0 <= bits["ternary"] < 2.01 and 1.6 <= bits["ternary_tq1"] < 1.64
    # storage only: TQ1 predicts exactly as TQ2
    np.testing.assert_array_equal(_batch(IMP.bff.QuantizedNeuralNet(net, "ternary_tq1"), X),
                                  _batch(IMP.bff.QuantizedNeuralNet(net, "ternary"), X))
    # quantize_activations is ignored: always int8 activations (W1.58A8)
    np.testing.assert_array_equal(_batch(IMP.bff.QuantizedNeuralNet(net, "ternary", True), X),
                                  _batch(IMP.bff.QuantizedNeuralNet(net, "ternary", False), X))


def test_refusals():
    import msgpack
    net = _wide_net()
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet(net, "ternary2")
    doc = msgpack.unpackb(IMP.bff.QuantizedNeuralNet(net, "ternary").to_msgpack(), raw=False)
    bad = dict(doc)
    bad["layers"] = [dict(l) for l in doc["layers"]]
    bad["layers"][0]["codes"] = b"\xff" * len(doc["layers"][0]["codes"])  # 2-bit code 3
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(msgpack.packb(bad, use_bin_type=True))
    bad["layers"][0]["codes"] = doc["layers"][0]["codes"][:-1]
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(msgpack.packb(bad, use_bin_type=True))
    bad["layers"][0]["codes"] = doc["layers"][0]["codes"]
    bad["layers"][0]["weight_scale"] = -1.0
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(msgpack.packb(bad, use_bin_type=True))
    with pytest.raises(ValueError):
        _train(*_regression(200, 5), _opts("ternary3", max_iter=1))


# ---------------------------------------------------------------------------
# Accuracy: post-training quantisation vs quantisation-aware training
# ---------------------------------------------------------------------------

def test_ptq_error_on_the_fixtures_is_reported():
    """Ternary PTQ of float-trained networks is coarse (that is why BitNet
    trains with the quantiser in the loop); reported, loosely bounded."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(64, 2))
    for name in ("mlp_torch_legacy.onnx", "mlp_relu_nometa.safetensors"):
        net = _fixture(name)
        exact = _batch(net, X)
        errs = {f: float(np.abs(_batch(IMP.bff.QuantizedNeuralNet(net, f), X) - exact).max() / np.abs(exact).max())
                for f in ("ternary", "ternary_row")}
        print("\n%s: ternary PTQ max error / output absmax %s" % (name, errs))
        assert all(e < 2.0 for e in errs.values())


# measured 2026-09-25 (okf/neural-net.md, "Ternary"): held-out MSE, target
# variance 0.51 -- float64 8.4e-4; ternary QAT 1.52e-3 (1.8x; every layer
# ternary: 1.86e-3, 2.2x); ternary PTQ of the float64 net 0.233 (276x),
# ternary_row PTQ 0.206. With the BitNet schedule (the default since
# 2026-09-26): QAT 1.35e-3 (1.6x), every layer ternary 2.72e-3 (3.2x)
QAT_BOUND = 3.0


def test_qat_against_float64_and_ptq_on_a_regression():
    X, Y = _regression(3000, 0)
    Xte, Yte = _regression(1000, 1)
    ref = _train(X, Y, _opts("float64"))
    ter = _train(X, Y, _opts("ternary"))
    assert ter.get_precision() == "ternary"
    net_ref = IMP.bff.NeuralNet(ref.get_network())
    q = IMP.bff.QuantizedNeuralNet.from_msgpack(ter.get_quantized_network())
    assert q.get_format() == "ternary" and q.to_msgpack() == ter.get_quantized_network()
    ptq = IMP.bff.QuantizedNeuralNet(net_ref, "ternary")
    m_ref, m_qat, m_ptq = _mse(net_ref, Xte, Yte), _mse(q, Xte, Yte), _mse(ptq, Xte, Yte)
    m_master = _mse(IMP.bff.NeuralNet(ter.get_network()), Xte, Yte)
    var = float(np.mean((Yte - Yte.mean(axis=0)) ** 2))
    print("\nternary, 4-64-64-64-2 tanh, held-out MSE (target variance %.3g): float64 %.4g | QAT %.4g (%.2fx)"
          " | PTQ of the float64 net %.4g (%.0fx) | QAT's float64 master weights evaluated in float64 %.4g"
          % (var, m_ref, m_qat, m_qat / m_ref, m_ptq, m_ptq / m_ref, m_master))
    assert m_qat < 0.01 * var
    assert m_qat <= QAT_BOUND * m_ref, (m_qat, m_ref)
    assert m_qat < 0.05 * m_ptq  # QAT is what makes ternary usable
    # every layer ternary
    allt = _train(X, Y, _opts("ternary", ternary_keep_first_layer=False, ternary_keep_last_layer=False))
    qa = IMP.bff.QuantizedNeuralNet.from_msgpack(allt.get_quantized_network())
    m_all = _mse(qa, Xte, Yte)
    print("  every layer ternary: %.4g (%.2fx), %.2f bits a weight" % (m_all, m_all / m_ref, qa.get_bits_per_weight()))
    assert m_all <= 1.5 * QAT_BOUND * m_ref


def test_qat_is_deterministic_and_is_the_trained_forward_pass():
    import msgpack
    X, Y = _regression(800, 2)
    a = _train(X, Y, _opts("ternary", seed=4, max_iter=5))
    b = _train(X, Y, _opts("ternary", seed=4, max_iter=5))
    c = _train(X, Y, _opts("ternary", seed=5, max_iter=5))
    assert a.get_network() == b.get_network() and a.get_quantized_network() == b.get_quantized_network()
    np.testing.assert_array_equal(a.get_loss_curve(), b.get_loss_curve())
    assert a.get_network() != c.get_network()
    doc = msgpack.unpackb(a.get_quantized_network(), raw=False)
    kinds = ["float64" if l.get("precision") == "float64" else "ternary" for l in doc["layers"]]
    assert kinds == ["float64", "ternary", "ternary", "float64"]
    assert doc["quantization"] == "ternary" and doc["quantize_activations"] is True
    # the ternary layers are the absmean quantiser applied to the master
    # weights (what fprop used), the kept layers the master weights; that
    # the document's predict() is training's forward pass bit for bit is
    # checked in C++ (test_ternary_kernels.cpp, with the library's GEMM)
    ptq = msgpack.unpackb(IMP.bff.QuantizedNeuralNet(IMP.bff.NeuralNet(a.get_network()), "ternary").to_msgpack(),
                          raw=False)
    master = msgpack.unpackb(a.get_network(), raw=False)
    for i, (l, p) in enumerate(zip(doc["layers"], ptq["layers"])):
        if kinds[i] == "ternary":
            assert l["codes"] == p["codes"] and l["weight_scale"] == p["weight_scale"]
        else:
            np.testing.assert_array_equal(np.asarray(l["weight"], float).ravel(),
                                          np.asarray(master["layers"][i]["weight"], float).ravel())
        np.testing.assert_array_equal(l["bias"], p["bias"])


def _has_surrogate():
    try:
        import tttrlib  # noqa: F401
    except ImportError:
        return False
    return hasattr(IMP.bff, "HmmSurrogate")


def test_qat_on_the_hmm_surrogate_set():
    if not _has_surrogate():
        pytest.skip("IMP.bff built without tttrlib: no HmmSurrogate")
    gen = lambda n, seed: IMP.bff.HmmSurrogate.generate_training_set(  # noqa: E731
        2, 2, n_samples=n, n_bursts=150, burst_len=80, mean_dt=4.0, seed=seed)
    X, Y = gen(400, 12345)
    Xte, Yte = gen(100, 12345 + 999)
    ref = IMP.bff.NeuralNet(_train(X, Y, _opts("float64", seed=0, max_iter=200, hidden=(256, 256, 128))).get_network())
    t = _train(X, Y, _opts("ternary", seed=0, max_iter=200, hidden=(256, 256, 128)))
    q = IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network())
    mae = lambda net: float(np.mean(np.abs(_batch(net, Xte) - Yte)))  # noqa: E731
    r, qat, ptq = mae(ref), mae(q), mae(IMP.bff.QuantizedNeuralNet(ref, "ternary"))
    print("\nHMM surrogate (256-256-128), held-out MAE: float64 %.4f | ternary QAT %.4f (%.2fx) | ternary PTQ %.4f (%.2fx)"
          % (r, qat, qat / r, ptq, ptq / r))
    # measured 2026-09-25: float64 0.0959, QAT 0.0939 (0.98x), PTQ 0.173 (1.81x);
    # BitNet schedule (default since 2026-09-26): QAT 0.0918 (0.96x)
    assert qat <= 1.2 * r
    assert qat < ptq


# ---------------------------------------------------------------------------
# The low-precision backward modes (ternary_backward)
# ---------------------------------------------------------------------------

BACKWARDS = ("float64", "int8_dgrad", "int8")
# measured 2026-09-26 (okf/neural-net.md, "Ternary: int8 backward"),
# held-out MSE: float64 training 8.4e-4; ternary QAT with the float64
# backward 1.35e-3, int8_dgrad 1.35e-3 (1.00x), int8 1.46e-3 (1.08x) (BitNet
# schedule; constant: 1.52e-3 / 1.57e-3 / 1.67e-3); HMM surrogate MAE
# 0.0918 / 0.0919 / 0.0921 (constant: 0.0939 / 0.0943 / 0.0935). Bound: each int8 mode within this
# factor of the float64-backward ternary result
BACKWARD_BOUND = 1.3


def test_backward_modes_on_a_regression():
    X, Y = _regression(3000, 0)
    Xte, Yte = _regression(1000, 1)
    ref = _mse(IMP.bff.NeuralNet(_train(X, Y, _opts("float64")).get_network()), Xte, Yte)
    m = {}
    for bw in BACKWARDS:
        t = _train(X, Y, _opts("ternary", ternary_backward=bw))
        m[bw] = _mse(IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network()), Xte, Yte)
    print("\nternary backward modes, 4-64-64-64-2 tanh, held-out MSE: float64 training %.4g | %s"
          % (ref, " | ".join("%s %.4g (%.2fx)" % (k, v, v / m["float64"]) for k, v in m.items())))
    for bw in ("int8_dgrad", "int8"):
        assert m[bw] <= BACKWARD_BOUND * m["float64"], (bw, m)
        assert m[bw] <= QAT_BOUND * ref


def test_backward_modes_are_deterministic_and_refused():
    X, Y = _regression(800, 2)
    for bw in ("int8_dgrad", "int8"):
        a = _train(X, Y, _opts("ternary", seed=4, max_iter=5, ternary_backward=bw))
        b = _train(X, Y, _opts("ternary", seed=4, max_iter=5, ternary_backward=bw))
        c = _train(X, Y, _opts("ternary", seed=5, max_iter=5, ternary_backward=bw))
        assert a.get_network() == b.get_network() and a.get_quantized_network() == b.get_quantized_network()
        assert a.get_network() != c.get_network()
    d = _train(X, Y, _opts("ternary", seed=4, max_iter=5))
    e = _train(X, Y, _opts("ternary", seed=4, max_iter=5, ternary_backward="float64"))
    assert d.get_network() == e.get_network()  # the default is the float64 backward
    assert IMP.bff.NeuralNetTrainOptions().ternary_backward == "float64"
    with pytest.raises(ValueError):
        _train(X, Y, _opts("ternary", max_iter=1, ternary_backward="int4"))


def test_backward_modes_on_the_hmm_surrogate_set():
    if not _has_surrogate():
        pytest.skip("IMP.bff built without tttrlib: no HmmSurrogate")
    gen = lambda n, seed: IMP.bff.HmmSurrogate.generate_training_set(  # noqa: E731
        2, 2, n_samples=n, n_bursts=150, burst_len=80, mean_dt=4.0, seed=seed)
    X, Y = gen(400, 12345)
    Xte, Yte = gen(100, 12345 + 999)
    mae = lambda net: float(np.mean(np.abs(_batch(net, Xte) - Yte)))  # noqa: E731
    ref = mae(IMP.bff.NeuralNet(_train(X, Y, _opts("float64", seed=0, max_iter=200, hidden=(256, 256, 128))).get_network()))
    m = {}
    for bw in BACKWARDS:
        t = _train(X, Y, _opts("ternary", seed=0, max_iter=200, hidden=(256, 256, 128), ternary_backward=bw))
        m[bw] = mae(IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network()))
    print("\nHMM surrogate, held-out MAE: float64 training %.4f | %s"
          % (ref, " | ".join("%s %.4f (%.2fx)" % (k, v, v / m["float64"]) for k, v in m.items())))
    for bw in ("int8_dgrad", "int8"):
        assert m[bw] <= 1.2 * m["float64"], (bw, m)
        assert m[bw] <= 1.3 * ref


# ---------------------------------------------------------------------------
# The BitNet schedule (ternary_schedule)
# ---------------------------------------------------------------------------

def test_bitnet_schedule_against_constant():
    """BitNet b1.58's two-stage recipe (higher peak learning rate with linear
    decay and a drop at the split, weight decay 0.1 -> 0, Adam beta2 0.95,
    warm-up, patience only in stage 2) against the constant schedule, each
    backward mode. Measured 2026-09-26, held-out MSE after 60 epochs:
    constant 1.52e-3 / 1.67e-3 (float64 / int8 backward), bitnet 1.35e-3 /
    1.46e-3."""
    X, Y = _regression(3000, 0)
    Xte, Yte = _regression(1000, 1)
    assert IMP.bff.NeuralNetTrainOptions().ternary_schedule == "bitnet"
    for bw in ("float64", "int8"):
        c = _train(X, Y, _opts("ternary", ternary_backward=bw, ternary_schedule="constant"))
        b = _train(X, Y, _opts("ternary", ternary_backward=bw))
        mc_, mb = (_mse(IMP.bff.QuantizedNeuralNet.from_msgpack(t.get_quantized_network()), Xte, Yte) for t in (c, b))
        print("\nbackward %s, held-out MSE: constant %.4g | bitnet %.4g (%.2fx), epochs %d / %d"
              % (bw, mc_, mb, mb / mc_, c.get_number_of_epochs(), b.get_number_of_epochs()))
        assert mb <= mc_
        assert b.get_number_of_epochs() >= 30  # early stopping waits for stage 2


def test_bitnet_schedule_options():
    X, Y = _regression(800, 2)
    a = _train(X, Y, _opts("ternary", seed=4, max_iter=6))
    b = _train(X, Y, _opts("ternary", seed=4, max_iter=6, ternary_schedule="bitnet"))
    assert a.get_network() == b.get_network()
    # every knob matters
    for kw in (dict(ternary_lr_stage1=3.0), dict(ternary_lr_stage2=2.0), dict(ternary_weight_decay=0.0),
               dict(ternary_stage_split=0.25), dict(ternary_warmup=0.2), dict(ternary_beta2=0.999)):
        assert _train(X, Y, _opts("ternary", seed=4, max_iter=6, **kw)).get_network() != a.get_network(), kw
    # float64 / FP4 training ignore it
    f = _train(X, Y, _opts("float64", seed=4, max_iter=3))
    g = _train(X, Y, _opts("float64", seed=4, max_iter=3, ternary_schedule="constant"))
    assert f.get_network() == g.get_network()
    for kw in (dict(ternary_schedule="cosine"), dict(ternary_stage_split=1.5), dict(ternary_lr_stage1=0.0),
               dict(ternary_beta2=1.0), dict(ternary_warmup=-0.1)):
        with pytest.raises(ValueError):
            _train(X, Y, _opts("ternary", max_iter=1, **kw))
