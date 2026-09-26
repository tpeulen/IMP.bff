"""ONNX export of the float neural networks.

`model_to_onnx()` in `internal/MlpCore.h` writes the same MLP subset the
reader accepts — one Gemm per layer plus its activation node — as protobuf
bytes built by hand (no ONNX library involved). `NeuralNet.to_onnx()` /
`to_onnx_file()` reach it. Weights are FLOAT: onnxruntime evaluates in
float32, so a round trip lands within float32 precision, like the float32
nets from_onnx() imports.

The files are spec ONNX: the tests run onnx.checker and onnxruntime over
bff's bytes when the packages are importable, otherwise skip — bff's own
round trips never depend on them.
"""

import math
import os
import tempfile

import msgpack
import numpy as np
import pytest

from IMP.bff import NeuralNet

ONNX_PY = None
try:
    import onnx
    import onnxruntime as ort
    ONNX_PY = True
except ImportError:
    ONNX_PY = False


def _make_net(n_in=8, hidden=(4,), n_out=2, activation="tanh"):
    layers = []
    dims = [n_in, *hidden, n_out]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        w = [math.sin(0.37 * (a * o + k) + 3.0) for o in range(b) for k in range(a)]
        bias = [0.1 * o - 0.3 for o in range(b)]
        layers.append({"n_in": a, "n_out": b,
                       "activation": activation if i + 2 < len(dims) else "identity",
                       "weight": w, "bias": bias})
    doc = {"format": "bff.neural_net", "version": 1, "layers": layers}
    return NeuralNet(msgpack.packb(doc, use_bin_type=True))


def _predict_bff(net, x):
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    _ = doc
    n = net.get_n_inputs()
    return np.array(net.predict(x.reshape(-1).tolist(), len(x) // n))


def test_onnx_round_trip_predictions():
    """bff -> ONNX -> bff: the weights land within float32 precision."""
    net = _make_net()
    net2 = NeuralNet.from_onnx(net.to_onnx())
    doc0 = msgpack.unpackb(net.to_msgpack(), raw=False)
    doc1 = msgpack.unpackb(net2.to_msgpack(), raw=False)
    assert len(doc1["layers"]) == len(doc0["layers"])
    for l0, l1 in zip(doc0["layers"], doc1["layers"]):
        assert l1["n_in"] == l0["n_in"] and l1["n_out"] == l0["n_out"]
        w0 = np.array(l0["weight"], dtype=np.float32)
        w1 = np.array(l1["weight"], dtype=np.float32)
        assert float(np.abs(w0 - w1).max()) < 1e-6


def test_onnx_bytes_not_text():
    net = _make_net()
    raw = net.to_onnx()
    assert isinstance(raw, bytes)
    assert len(raw) > 8


@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic",
                                        "softplus", "silu", "sin", "identity"])
def test_onnx_file_round_trip(activation):
    net = _make_net(n_in=32, hidden=(32,), n_out=4, activation=activation)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        p = f.name
    try:
        net.to_onnx_file(p)
        net2 = NeuralNet.from_onnx_file(p)
        rng = np.random.default_rng(42)
        x = rng.normal(size=5 * 32).astype(np.float32)
        y0 = _predict_bff(net, x)
        y1 = _predict_bff(net2, x)
        assert float(np.abs(y0 - y1).max()) < 1e-4
    finally:
        os.unlink(p)


@pytest.mark.skipif(not ONNX_PY, reason="onnx/onnxruntime not importable")
def test_onnx_runtime_runs_bff_files():
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        p = f.name
    try:
        net.to_onnx_file(p)
        onnx.checker.check_model(p)
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5, 32)).astype(np.float32)
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        # reference: manual evaluation of the doc's layers in float32
        h = x
        acts = {"relu": lambda v: np.maximum(v, 0), "tanh": np.tanh,
                "logistic": lambda v: 1 / (1 + np.exp(-v)),
                "identity": lambda v: v}
        for i, layer in enumerate(doc["layers"]):
            w = np.array(layer["weight"], dtype=np.float32).reshape(
                    layer["n_out"], layer["n_in"])
            b = np.array(layer["bias"], dtype=np.float32)
            a = layer["activation"]
            h = h @ w.T + b
            if i + 1 < len(doc["layers"]):
                h = acts[a](h)
        assert float(np.abs(got - h).max()) < 1e-4
    finally:
        os.unlink(p)
