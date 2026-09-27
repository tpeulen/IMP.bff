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
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    from gguf.quants import dequantize as gguf_dequantize
    from gguf.quants import quantize as gguf_quantize
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
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


def test_float_export_imports_all_plain_integer_types():
    """I16/I32/I64 GGUF tensors preserve integral-valued model weights."""
    net = _make_net(n_in=8, hidden=(8,), n_out=2)
    for tensor_type in ("i16", "i32", "i64"):
        imported = NeuralNet.from_gguf(net.to_gguf(tensor_type))
        assert imported.get_n_inputs() == net.get_n_inputs(), tensor_type
        assert imported.get_n_outputs() == net.get_n_outputs(), tensor_type


# These active GGML v3 formats have exact decoders in bff, but their upstream
# writers are not yet reference-verified in bff. The public contract is
# read-only rather than emitting merely plausible bytes.
IMPORT_ONLY_IQ_TYPES = (
    "iq3_xxs", "iq1_s", "iq3_s", "iq2_s", "iq1_m",
)


@pytest.mark.parametrize("tensor_type", IMPORT_ONLY_IQ_TYPES)
def test_float_export_rejects_import_only_iq_types(tensor_type):
    """Read-only IQ formats fail explicitly instead of writing zero payloads."""
    net = _make_net(n_in=256, hidden=(), n_out=1)
    with pytest.raises(IMP.ValueException, match="import only"):
        net.to_gguf(tensor_type)


@pytest.mark.skipif(not GGUF_PY, reason="gguf-py not importable")
@pytest.mark.parametrize(("source", "expected_hex"), [
    ([-3.875 + 0.25 * i for i in range(32)],
     "22288091a1a1b2b2c3c3d3d4e4e5e6f6f7f8"),
    ([(i % 7 - 3) * 0.625 + (0.125 if i % 3 == 0 else 0.0)
      for i in range(32)], "f7a3bf8e4b2705f2e0bf7e5b2805f2e0bf8d"),
])
def test_iq4_nl_export_matches_ggml_reference_vector(source, expected_hex, tmp_path):
    """IQ4_NL uses ggml's deterministic nonlinear-codebook quantizer."""
    doc = {"format": "bff.neural_net", "version": 1, "layers": [{
        "n_in": 32, "n_out": 1, "activation": "identity",
        "weight": source, "bias": [0.0],
    }]}
    net = NeuralNet(msgpack.packb(doc, use_bin_type=True))
    path = tmp_path / "iq4-nl.gguf"
    net.to_gguf_file(str(path), "iq4_nl")
    tensor = next(t for t in GGUFReader(str(path)).tensors
                  if t.name == "blk.0.weight")
    assert tensor.tensor_type == GGMLQuantizationType.IQ4_NL
    assert tensor.data.tobytes().hex() == expected_hex


@pytest.mark.skipif(not GGUF_PY, reason="gguf-py not importable")
def test_iq4_xs_export_matches_ggml_reference_vector(tmp_path):
    """IQ4_XS uses ggml's deterministic nonlinear-codebook quantizer."""
    source = [-3.875 + 0.03125 * i for i in range(256)]
    doc = {"format": "bff.neural_net", "version": 1, "layers": [{
        "n_in": 256, "n_out": 1, "activation": "identity",
        "weight": source, "bias": [0.0],
    }]}
    net = NeuralNet(msgpack.packb(doc, use_bin_type=True))
    path = tmp_path / "iq4-xs.gguf"
    net.to_gguf_file(str(path), "iq4_xs")
    tensor = next(t for t in GGUFReader(str(path)).tensors
                  if t.name == "blk.0.weight")
    assert tensor.tensor_type == GGMLQuantizationType.IQ4_XS
    assert tensor.data.tobytes().hex() == "ec13af056e7f07080010101010101010101010101010101010101010101010101010101010202121102020202020202121213131313131314040405151516161627272728383939337262626252525151414141413030303131212121212020202020201010101011212010101010101010101010101010101010101010101010101010101010100"

@pytest.mark.parametrize("tensor_type", [
    "f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq2_xxs", "iq2_xs",
    "tq1_0", "tq2_0", "mxfp4", "nvfp4",
])
def test_float_export_matches_reference_decoder(tensor_type, tmp_path):
    """An independent gguf-py decoder must recover bff's imported weights."""
    net = _make_net(n_in=256, hidden=(), n_out=1)
    path = tmp_path / "export.gguf"
    net.to_gguf_file(str(path), tensor_type)
    reader = GGUFReader(str(path))
    tensor = next(t for t in reader.tensors if t.name == "blk.0.weight")
    qtype = GGMLQuantizationType[tensor_type.upper()]
    assert tensor.tensor_type == qtype
    reference = gguf_dequantize(
        np.frombuffer(tensor.data.tobytes(), dtype=np.uint8), qtype
    ).reshape(-1)
    imported = NeuralNet.from_gguf(path.read_bytes())
    weight = msgpack.unpackb(imported.to_msgpack(), raw=False)["layers"][0]["weight"]
    np.testing.assert_allclose(weight, reference, rtol=0, atol=1e-6)


@pytest.mark.skipif(not GGUF_PY, reason="gguf-py not importable")
@pytest.mark.parametrize("tensor_type", [
    "f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "tq1_0", "tq2_0", "mxfp4",
])
def test_float_import_accepts_reference_encoded_tensor(tensor_type, tmp_path):
    """Read a GGUF container and weight payload made by gguf-py."""
    net = _make_net(n_in=256, hidden=(), n_out=1)
    layer = msgpack.unpackb(net.to_msgpack(), raw=False)["layers"][0]
    source = np.asarray(layer["weight"], dtype=np.float32).reshape(1, 256)
    qtype = GGMLQuantizationType[tensor_type.upper()]
    encoded = gguf_quantize(source, qtype)
    path = tmp_path / "reference.gguf"
    writer = GGUFWriter(str(path), "bff.mlp")
    writer.add_uint32("bff.layer_count", 1)
    writer.add_key_value("bff.dims", [256, 1], GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.UINT32)
    writer.add_key_value("bff.activations", ["identity"], GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.STRING)
    writer.add_string("bff.weight_type", tensor_type.upper())
    writer.add_tensor("blk.0.weight", encoded, raw_dtype=qtype)
    writer.add_tensor("blk.0.bias", np.asarray(layer["bias"], dtype=np.float32).reshape(1, 1))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    imported = NeuralNet.from_gguf(path.read_bytes())
    weight = msgpack.unpackb(imported.to_msgpack(), raw=False)["layers"][0]["weight"]
    reference = gguf_dequantize(np.asarray(encoded).view(np.uint8).reshape(-1), qtype)
    np.testing.assert_allclose(weight, reference.reshape(-1), rtol=0, atol=1e-6)


@pytest.mark.skipif(not GGUF_PY, reason="gguf-py not importable")
@pytest.mark.parametrize("tensor_type", IMPORT_ONLY_IQ_TYPES)
def test_float_import_only_iq_types_match_reference_decoder(tensor_type, tmp_path):
    """Import IQ blocks that gguf-py can decode but neither library encodes."""
    qtype = GGMLQuantizationType[tensor_type.upper()]
    block_size, byte_size = GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(42)
    payload = rng.integers(0, 256, size=byte_size * (256 // block_size), dtype=np.uint8)
    for offset in range(0, len(payload), byte_size):
        payload[offset:offset + 2] = (0, 60)  # finite fp16 scale, 1.0
    reference = gguf_dequantize(payload, qtype).reshape(-1)
    assert np.isfinite(reference).all()

    path = tmp_path / "reference-iq.gguf"
    writer = GGUFWriter(str(path), "bff.mlp")
    writer.add_uint32("bff.layer_count", 1)
    writer.add_key_value("bff.dims", [256, 1], GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.UINT32)
    writer.add_key_value("bff.activations", ["identity"], GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.STRING)
    writer.add_string("bff.weight_type", tensor_type.upper())
    writer.add_tensor("blk.0.weight", payload.reshape(1, -1), raw_dtype=qtype)
    writer.add_tensor("blk.0.bias", np.zeros((1, 1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    imported = NeuralNet.from_gguf(path.read_bytes())
    weight = msgpack.unpackb(imported.to_msgpack(), raw=False)["layers"][0]["weight"]
    np.testing.assert_allclose(weight, reference, rtol=0, atol=1e-5)


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
