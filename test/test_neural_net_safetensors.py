"""safetensors export of the float neural networks.

`model_to_safetensors()` in `internal/MlpCore.h` writes a PyTorch-style
`state_dict` — `blk.<i>.weight` of shape `(n_out, n_in)` plus `blk.<i>.bias`
per layer, dtype F64, the activations in `__metadata__` — reached through
`NeuralNet.to_safetensors()` / `to_safetensors_file()`. The files are spec
safetensors: when the `safetensors` package is importable the tests check
bff's bytes against it, otherwise that check is skipped — bff's own round
trips never depend on it.

safetensors is an export format for float nets; quantised nets have no
dtypes to hold them and round-trip through GGUF instead.
"""

import math
import os
import tempfile

import msgpack
import numpy as np
import pytest

from IMP.bff import NeuralNet

SAFETENSORS_PY = None
try:
    import safetensors.numpy  # noqa: F401
    SAFETENSORS_PY = True
except ImportError:
    SAFETENSORS_PY = False


def _make_net(n_in=8, hidden=(4,), n_out=2):
    layers = []
    dims = [n_in, *hidden, n_out]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        w = [math.sin(0.37 * (a * o + k) + 3.0) for o in range(b) for k in range(a)]
        bias = [0.1 * o - 0.3 for o in range(b)]
        layers.append({"n_in": a, "n_out": b,
                       "activation": "tanh" if i + 2 < len(dims) else "identity",
                       "weight": w, "bias": bias})
    doc = {"format": "bff.neural_net", "version": 1, "layers": layers}
    return NeuralNet(msgpack.packb(doc, use_bin_type=True))


def test_safetensors_round_trip():
    net = _make_net()
    net2 = NeuralNet.from_safetensors(net.to_safetensors())
    doc0 = msgpack.unpackb(net.to_msgpack(), raw=False)
    doc1 = msgpack.unpackb(net2.to_msgpack(), raw=False)
    assert doc1["layers"] == doc0["layers"]


def test_safetensors_file_round_trip():
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        p = f.name
    try:
        net.to_safetensors_file(p)
        net2 = NeuralNet.from_safetensors_file(p)
        doc0 = msgpack.unpackb(net.to_msgpack(), raw=False)
        doc1 = msgpack.unpackb(net2.to_msgpack(), raw=False)
        assert doc1["layers"] == doc0["layers"]
    finally:
        os.unlink(p)


def test_safetensors_bytes_not_text():
    net = _make_net()
    raw = net.to_safetensors()
    assert isinstance(raw, bytes)
    # the u64 length word counts the header only (little-endian)
    hlen = int.from_bytes(raw[:8], "little")
    assert 8 + hlen <= len(raw)
    assert raw[8:18] == b'{"__metada'


@pytest.mark.skipif(not SAFETENSORS_PY, reason="safetensors not importable")
def test_safetensors_py_reads_bff_files():
    from safetensors.numpy import load_file
    net = _make_net()
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        p = f.name
    try:
        net.to_safetensors_file(p)
        t = load_file(p)
        for i, layer in enumerate(doc["layers"]):
            w = t["blk.%d.weight" % i]
            b = t["blk.%d.bias" % i]
            want_w = np.array(layer["weight"], dtype=np.float64).reshape(
                    layer["n_out"], layer["n_in"])
            want_b = np.array(layer["bias"], dtype=np.float64)
            assert w.shape == want_w.shape and w.dtype == np.float64
            # bit-exact: the writer memcpys the model's doubles
            assert np.array_equal(w, want_w), "blk.%d.weight" % i
            assert np.array_equal(b, want_b), "blk.%d.bias" % i
    finally:
        os.unlink(p)
