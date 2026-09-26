"""GGUF import/export of the neural networks.

`GgufIo.h` is the std-only GGUF v3 container and its block codecs; the
mapping between bff's networks and the file lives in `internal/NetworkGguf.h`
and is reached through `NeuralNet.to_gguf()` / `from_gguf()` (any weight
type: F32, F16, BF16, F64, Q8_0, Q4_0, Q1_0, MXFP4, NVFP4, TQ1_0, TQ2_0)
and `QuantizedNeuralNet.to_gguf()` / `from_gguf()` (int8 as I8, mxfp4
natively, nvfp4 folded, ternary as TQ2_0 / TQ1_0, the absmean in metadata).

The files are spec GGUF: when the `gguf` (gguf-py) package is importable the
tests additionally check bff's bytes against gguf-py's reference
quantisers, and MLX's `mx.load` for the types it reads. The reference
checks run when gguf-py is present, otherwise they are skipped -- bff's own
round trips never depend on it.
"""

import math
import os
import tempfile

import msgpack
import numpy as np
import pytest

import IMP
import IMP.bff
from IMP.bff import NeuralNet, QuantizedNeuralNet

GGUF_PY = None
try:
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize as gguf_dequantize
    from gguf import GGUFReader
    GGUF_PY = True
except ImportError:
    GGUF_PY = False

MLX = None
try:
    import mlx.core as mx
    MLX = True
except ImportError:
    MLX = False


def _make_net(n_in=4, hidden=(8, 6), n_out=2, seed=3):
    """A small deterministic network."""
    dims = [n_in] + list(hidden) + [n_out]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        w = [math.sin(0.37 * (a * o + k) + seed) for o in range(b) for k in range(a)]
        bias = [0.1 * o - 0.3 for o in range(b)]
        layers.append({"n_in": a, "n_out": b,
                       "activation": "tanh" if i + 2 < len(dims) else "identity",
                       "weight": w, "bias": bias})
    doc = {"format": "bff.neural_net", "version": 1, "layers": layers}
    return NeuralNet(msgpack.packb(doc, use_bin_type=True))


def _batch(net, n=5):
    x = [math.sin(0.11 * r + 0.3 * c) for r in range(n) for c in range(net.get_n_inputs())]
    return x, net.predict(x, n)


def test_float_round_trip_every_type():
    """to_gguf -> from_gguf reproduces the weights for every type; F64 exactly."""
    net = _make_net()
    ref_doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    x, y0 = _batch(net)
    for t in ["f64", "f32", "f16", "bf16", "q8_0", "q4_0", "q1_0",
              "mxfp4", "nvfp4", "tq2_0", "tq1_0"]:
        data = net.to_gguf(t)
        assert isinstance(data, bytes), t
        net2 = NeuralNet.from_gguf(data)
        if t == "f64":
            doc2 = msgpack.unpackb(net2.to_msgpack(), raw=False)
            for l0, l1 in zip(ref_doc["layers"], doc2["layers"]):
                assert l0["weight"] == l1["weight"], t
        x2, y2 = _batch(net2)
        assert len(y2) == len(y0), t
        # quantisation error bounds per type: 4-bit / 1-bit codes lose real
        # precision on weights of order 1, so the predictions may drift
        tol = {"f64": 1e-12, "f32": 1e-5, "f16": 0.02, "bf16": 0.05,
               "q8_0": 0.05, "q4_0": 0.4, "q1_0": 1.0, "mxfp4": 0.4,
               "nvfp4": 0.4, "tq2_0": 0.7, "tq1_0": 0.7}[t]
        for a, b in zip(y0, y2):
            assert abs(a - b) < tol, (t, a, b)


def test_file_round_trip():
    net = _make_net()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "net.gguf")
        net.to_gguf_file(p, "q8_0")
        assert os.path.getsize(p) > 0
        net2 = NeuralNet.from_gguf_file(p)
        assert net2.get_n_inputs() == net.get_n_inputs()
        assert net2.get_n_outputs() == net.get_n_outputs()
        assert net2.get_n_layers() == net.get_n_layers()
        x, y0 = _batch(net)
        x2, y2 = _batch(net2)
        for a, b in zip(y0, y2):
            assert abs(a - b) < 0.05


def test_foreign_architecture_refused():
    data = bytes(_make_net().to_gguf("f32"))
    tag = b"bff.mlp"
    i = data.find(tag)
    assert i >= 0
    broken = data[:i] + b"llama  " + data[i + 7:]
    with pytest.raises(IMP.ValueException):
        NeuralNet.from_gguf(broken)


def test_bytes_not_text():
    net = _make_net()
    assert isinstance(net.to_gguf("f32"), bytes)
    with pytest.raises(TypeError):
        NeuralNet.from_gguf("not bytes")


def test_quantized_int8_round_trip():
    net = _make_net()
    q = QuantizedNeuralNet(net, "int8")
    q2 = QuantizedNeuralNet.from_gguf(q.to_gguf())
    assert q2.get_format() == "int8"
    assert q2.get_n_inputs() == q.get_n_inputs()
    assert q2.get_n_outputs() == q.get_n_outputs()
    assert q2.get_n_layers() == q.get_n_layers()
    assert q2.get_bits_per_weight() == q.get_bits_per_weight()
    x, _ = _batch(net)
    y0 = q.predict(x, len(x) // q.get_n_inputs())
    y1 = q2.predict(x, len(x) // q.get_n_inputs())
    for a, b in zip(y0, y1):
        assert abs(a - b) < 1e-12


def test_quantized_mxfp4_round_trip():
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    q = QuantizedNeuralNet(net, "mxfp4")
    q2 = QuantizedNeuralNet.from_gguf(q.to_gguf())
    assert q2.get_format() == "mxfp4"
    x, _ = _batch(net)
    y0 = q.predict(x, len(x) // q.get_n_inputs())
    y1 = q2.predict(x, len(x) // q.get_n_inputs())
    for a, b in zip(y0, y1):
        assert abs(a - b) < 1e-9


def test_quantized_ternary_round_trip():
    net = _make_net(n_in=64, hidden=(32,), n_out=4)
    for fmt in ["ternary", "ternary_row", "ternary_tq1", "ternary_tq1_row"]:
        q = QuantizedNeuralNet(net, fmt)
        q2 = QuantizedNeuralNet.from_gguf(q.to_gguf())
        assert q2.get_format() == fmt
        x, _ = _batch(net)
        y0 = q.predict(x, len(x) // q.get_n_inputs())
        y1 = q2.predict(x, len(x) // q.get_n_inputs())
        for a, b in zip(y0, y1):
            assert abs(a - b) < 1e-9, fmt


def test_quantized_nvfp4_round_trip():
    net = _make_net(n_in=64, hidden=(32,), n_out=4)
    q = QuantizedNeuralNet(net, "nvfp4")
    q2 = QuantizedNeuralNet.from_gguf(q.to_gguf())
    assert q2.get_format() == "nvfp4"
    assert q2.get_bits_per_weight() == q.get_bits_per_weight()
    x, _ = _batch(net)
    y0 = q.predict(x, len(x) // q.get_n_inputs())
    y1 = q2.predict(x, len(x) // q.get_n_inputs())
    # the fold into ggml's super-blocks rounds the block scales to the
    # ue4m3 grid; the import re-quantises against them, so predictions
    # drift by 4-bit quantisation noise, not by zero
    for a, b in zip(y0, y1):
        assert abs(a - b) < 0.2


def test_quantized_fp4_round_trip():
    """bff's per-row fp4 has no GGUF type: the file carries F32; the reader
    re-quantises (the F32 payload is exactly on the fp4 grid, so the codes
    and row scales come back identical) and predictions agree exactly."""
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    q = QuantizedNeuralNet(net, "fp4")
    q2 = QuantizedNeuralNet.from_gguf(q.to_gguf())
    assert q2.get_format() == "fp4"
    assert q2.get_bits_per_weight() == q.get_bits_per_weight()
    x, _ = _batch(net)
    y0 = q.predict(x, len(x) // q.get_n_inputs())
    y1 = q2.predict(x, len(x) // q.get_n_inputs())
    for a, b in zip(y0, y1):
        assert abs(a - b) < 1e-12


@pytest.mark.skipif(not GGUF_PY, reason="gguf-py not importable")
def test_gguf_py_reads_bff_files():
    """GGUFReader opens bff's files and dequantises to bff's values."""
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    for t, qt in [("f32", GGMLQuantizationType.F32),
                  ("q8_0", GGMLQuantizationType.Q8_0),
                  ("mxfp4", GGMLQuantizationType.MXFP4),
                  ("tq2_0", GGMLQuantizationType.TQ2_0),
                  ("tq1_0", GGMLQuantizationType.TQ1_0)]:
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            p = f.name
        try:
            net.to_gguf_file(p, t)
            r = GGUFReader(p)
            arch = r.fields["general.architecture"]
            assert arch.parts[arch.data[0]].tobytes().decode() == "bff.mlp", t
            wt = r.tensors[0]
            assert wt.name == "blk.0.weight", t
            assert wt.tensor_type == qt, t
            got = gguf_dequantize(np.frombuffer(wt.data.tobytes(), dtype=np.uint8), qt)
            got = np.asarray(got, dtype=np.float32).reshape(-1)
            cols = int(wt.shape[0])  # file width (padded to the block size)
            want = np.array(doc["layers"][0]["weight"], dtype=np.float32)
            want = want.reshape(32, 32)
            # rows sit at file stride `cols`; gather the real columns
            got = np.stack([got[r * cols:r * cols + 32] for r in range(32)])
            # 4-bit / ternary codes quantise; the file is correct if the
            # dequantised values sit on the code grid near the source
            bound = {"f32": 1e-6, "q8_0": 0.02, "mxfp4": 0.25,
                     "tq2_0": 0.5, "tq1_0": 0.5}[t]
            assert float(np.abs(got - want).max()) < bound, t
        finally:
            os.unlink(p)


@pytest.mark.skipif(not (GGUF_PY and MLX), reason="gguf-py or mlx not importable")
def test_mlx_loads_bff_files():
    net = _make_net(n_in=32, hidden=(32,), n_out=4)
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    want = np.array(doc["layers"][0]["weight"], dtype=np.float32).reshape(-1)
    for t in ["f32", "f16", "q8_0"]:
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            p = f.name
        try:
            net.to_gguf_file(p, t)
            w = mx.load(p, format="gguf")
            key = "blk.0.weight"
            assert key in w, t
            got = np.array(w[key], dtype=np.float32).reshape(-1)
            if got.size != want.size:
                # mlx returns raw quantised words (uint32) for types it does
                # not dequantise on load: presence + shape is the contract
                pytest.skip("mlx returns raw words for %s" % t)
            err = float(np.abs(got - want).max())
            assert err < (0.05 if t == "f16" else 1e-5), t
        except Exception as e:
            if "unsupported" in str(e).lower() or "not support" in str(e).lower():
                pytest.skip("mlx refuses %s: %s" % (t, e))
            raise
        finally:
            os.unlink(p)
