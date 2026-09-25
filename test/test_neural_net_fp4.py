"""FP4 weights for the network: QuantizedNeuralNet's fp4 / mxfp4 / nvfp4.

The formats, their recipes and the integer-SIMD kernels are C++
(internal/MlpFp4.h, internal/MlpFp4Kernels.h). Checked here:

* the C++ snippets cpp_snippets/test_fp4_kernels.cpp and test_mlp_math.cpp
  (the vectorised tanh every path shares) -- exhaustive codecs
  (E2M1, E4M3, E8M0), the block recipes, and every compiled SIMD variant
  against the generic kernel -- built four ways: NEON + dotprod, plain NEON,
  AVX2 (under Rosetta 2 on Apple Silicon, natively on x86) and the generic
  scalar code, plus AVX-512 VNNI on an x86 CPU that has it (CI runs that
  one under Intel SDE otherwise);
* an independent cross-check of the E2M1 encoding, the packing and the E4M3
  / E8M0 block scales against PyTorch's own FP4/FP8 utilities (skipped when
  PyTorch is absent);
* the error of each format against the double network on the committed
  fixtures and a trained 3x64 net, the weight sizes, W4A4 against W4A8,
  the msgpack round trip, and the refusals;
* speeds, reported and never asserted.
"""

import ast
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import msgpack
import numpy as np
import pytest

import IMP.bff

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "input", "nn")
FORMATS = ["fp4", "mxfp4", "nvfp4"]


def _batch(net, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(
        X.shape[0], net.get_n_outputs())


def _rel_err(approx, exact):
    return np.abs(approx - exact).max() / np.abs(exact).max()


def _fixture(name):
    path = os.path.join(FIXTURES, name)
    if name.endswith(".onnx"):
        return IMP.bff.NeuralNet.from_onnx_file(path)
    return IMP.bff.NeuralNet.from_safetensors_file(
        path, hidden_activation="relu" if "relu" in name else "tanh")


def _trained_3x64():
    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, size=(2000, 2))
    y = np.sin(X[:, 0]) * np.cos(0.5 * X[:, 1]) + 0.1 * X[:, 0] * X[:, 1]
    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = [64, 64, 64]
    opt.activation = "tanh"
    opt.max_iter = 40
    opt.seed = 1
    return IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(X, y, opt).get_network())


def _n_weights(net):
    return sum(net.get_layer_weights(i).size for i in range(net.get_n_layers()))


# ---------------------------------------------------------------------------
# The C++ snippet, four ways
# ---------------------------------------------------------------------------

def _has_avx512_vnni():
    """AVX-512 VNNI on this CPU (Linux /proc/cpuinfo; elsewhere: unknown, no)."""
    try:
        with open("/proc/cpuinfo") as fh:
            return "avx512_vnni" in fh.read()
    except OSError:
        return False


def _variants():
    """(name, compiler prefix, flags, expected variant) for this machine."""
    out = []
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        out.append(("neon-dotprod", [], ["-march=armv8.2-a+dotprod"], "neon-dotprod"))
        # IMPBFF_FP4_NO_DOTPROD: some toolchains (conda's clang on macOS arm64)
        # keep dotprod on under -march=armv8-a, so the plain NEON path is forced.
        out.append(("neon", [], ["-march=armv8-a", "-DIMPBFF_FP4_NO_DOTPROD"], "neon"))
        if sys.platform == "darwin" and shutil.which("arch"):
            out.append(("avx2", ["arch", "-x86_64"], ["-arch", "x86_64", "-mavx2", "-mfma"], "avx2"))
    elif machine in ("x86_64", "amd64"):
        out.append(("avx2", [], ["-mavx2", "-mfma"], "avx2"))
        if _has_avx512_vnni():
            out.append(("avx512-vnni", [], ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni"],
                        "avx512-vnni"))
    out.append(("generic", [], ["-DIMPBFF_FP4_NO_SIMD"], "generic"))
    return out


def _build_and_run(snippet, name, runner, flags):
    """Compile cpp_snippets/<snippet> with `flags` and run it (under `runner`);
    skips when the cross toolchain or the emulator is missing."""
    cxx = shutil.which("clang++") or shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    repo = os.path.dirname(HERE)
    with tempfile.TemporaryDirectory() as tmp:
        inc = os.path.join(tmp, "IMP", "bff")
        os.makedirs(inc)
        os.symlink(os.path.join(repo, "include", "internal"), os.path.join(inc, "internal"))
        exe = os.path.join(tmp, os.path.splitext(snippet)[0])
        build = subprocess.run([cxx, "-std=c++17", "-O2", *flags, "-I", tmp,
                                os.path.join(HERE, "cpp_snippets", snippet), "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            if name in ("avx2", "sse2") and runner:
                pytest.skip("no x86_64 toolchain: " + build.stderr[-300:])
            raise AssertionError(build.stderr)
        try:
            run = subprocess.run([*runner, exe], capture_output=True, text=True, timeout=600)
        except OSError as e:  # e.g. no Rosetta 2
            pytest.skip("cannot run the %s build: %s" % (name, e))
    if runner and run.returncode != 0 and "variant" not in run.stdout:
        pytest.skip("cannot run the %s build: %s" % (name, run.stderr[-300:]))
    return run


@pytest.mark.parametrize("name,runner,flags,want", _variants(), ids=[v[0] for v in _variants()])
def test_fp4_kernels_cpp_every_variant(name, runner, flags, want):
    """Codecs exhaustive, recipes, and the compiled SIMD variant bit-identical
    to the generic kernel (integer sums exact, float scaling shared)."""
    run = _build_and_run("test_fp4_kernels.cpp", name, runner, flags)
    print("\n" + "\n".join(l for l in run.stdout.splitlines() if l.startswith("  time")))
    assert run.stdout.splitlines()[0] == "variant " + want
    assert run.returncode == 0, run.stdout
    assert "0 failure(s)" in run.stdout


_MATH_VARIANT = {"neon-dotprod": "neon", "neon": "neon", "avx2": "avx2",
                 "avx512-vnni": "avx512", "generic": "generic"}


def _math_variants():
    """The FP4 builds (their MlpMath.h variant) plus x86-64's baseline build,
    which MlpMath.h runs on SSE2 (natively on x86, under Rosetta 2 on Apple
    Silicon)."""
    out = [(n, r, f, _MATH_VARIANT[w]) for n, r, f, w in _variants()]
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        out.append(("sse2", [], [], "sse2"))
    elif sys.platform == "darwin" and shutil.which("arch"):
        out.append(("sse2", ["arch", "-x86_64"], ["-arch", "x86_64"], "sse2"))
    return out


@pytest.mark.parametrize("name,runner,flags,want", _math_variants(), ids=[v[0] for v in _math_variants()])
def test_mlp_math_cpp_every_variant(name, runner, flags, want):
    """MlpMath.h (the network's tanh / sigmoid / SiLU): <= 2 ulp of libm on
    a dense grid, every exponent and the edge cases, odd, monotone, and one
    committed fingerprint for every SIMD variant."""
    run = _build_and_run("test_mlp_math.cpp", name, runner, flags)
    print("\n" + "\n".join(l for l in run.stdout.splitlines() if l.startswith("  ")))
    assert run.stdout.splitlines()[0] == "variant " + want
    assert run.returncode == 0, run.stdout
    assert "0 failure(s)" in run.stdout


def test_the_module_reports_its_kernel():
    name = IMP.bff.QuantizedNeuralNet.get_kernel_name()
    assert name in ("neon-dotprod", "neon", "avx2", "generic")
    print("\nIMP.bff FP4 kernel variant:", name)
    if platform.machine().lower() == "arm64" and sys.platform == "darwin":
        assert name == "neon-dotprod"  # every Apple Silicon core has dotprod


# ---------------------------------------------------------------------------
# PyTorch as an independent reference
# ---------------------------------------------------------------------------

def _torch_fp4_reference():
    """PyTorch's E2M1 encoder (torchao's `_f32_to_floatx_unpacked`), its
    packing (`pack_uint4`) and `to_mxfp`, taken out of
    torch/testing/_internal/common_quantized.py without importing that module
    (it pulls in test-only dependencies)."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this PyTorch has no float8_e4m3fn")
    path = os.path.join(os.path.dirname(torch.__file__), "testing", "_internal", "common_quantized.py")
    if not os.path.exists(path):
        pytest.skip("no torch/testing/_internal/common_quantized.py")
    tree = ast.parse(open(path).read())
    want = {"_n_ones", "_f32_to_floatx_unpacked", "down_size", "pack_uint4",
            "_bfloat16_to_float4_e2m1fn_x2", "to_mxfp"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
        elif isinstance(node, ast.Assign) and all(
                isinstance(t, (ast.Name, ast.Tuple)) for t in node.targets):
            names = [n.id for t in node.targets for n in ([t] if isinstance(t, ast.Name) else t.elts)]
            if any(n.startswith(("EBITS", "MBITS", "F32_", "FP4_")) for n in names):
                body.append(node)
    ns = {"torch": torch, "Tensor": torch.Tensor}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    if "_f32_to_floatx_unpacked" not in ns or "pack_uint4" not in ns:
        pytest.skip("PyTorch's FP4 helpers moved")
    return torch, ns


def _layers_of(q):
    return msgpack.unpackb(q.to_msgpack(), raw=False)["layers"]


def _decode_weights(fmt, layer):
    """A layer's weights decoded from its document (numpy, independent of C++)."""
    n_out, n_in = layer["n_out"], layer["n_in"]
    b = np.frombuffer(layer["codes"], dtype=np.uint8).reshape(n_out, -1)
    codes = np.stack([b & 15, b >> 4], axis=2).reshape(n_out, -1)[:, :n_in]
    e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    return e2m1[codes] * _block_scales(fmt, layer)


def _e4m3_decode(codes):
    """FP8 E4M3 (e4m3fn: bias 7, no infinities, S.1111.111 is NaN) in numpy,
    so decoding the stored block scales needs no torch."""
    c = np.asarray(codes, dtype=np.uint8).astype(int)
    sign = np.where(c & 0x80, -1.0, 1.0)
    e, m = (c >> 3) & 0xF, c & 0x7
    val = np.where(e == 0, np.ldexp(m / 8.0, -6), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where((c & 0x7F) == 0x7F, np.nan, sign * val)


def _block_scales(fmt, layer):
    """bff's decoded scale per (row, element) of a layer, from its document."""
    n_out, n_in, block = layer["n_out"], layer["n_in"], layer["block"]
    sc = np.frombuffer(layer["scales"], dtype=np.uint8)
    if fmt == "fp4":
        s = np.frombuffer(layer["scales"], dtype="<f4").astype(float)
        return np.repeat(s[:, None], n_in, axis=1)
    nb = len(sc) // n_out
    sc = sc.reshape(n_out, nb)
    if fmt == "mxfp4":
        d = np.ldexp(1.0, sc.astype(int) - 127)
    else:
        d = _e4m3_decode(sc) * float(layer["tensor_scale"])
    return np.repeat(d, block, axis=1)[:, :n_in]


@pytest.mark.parametrize("fmt", FORMATS)
def test_e2m1_codes_and_packing_match_pytorch(fmt):
    """Given bff's scales, every weight's E2M1 code is what PyTorch's encoder
    gives for w / scale, and bff's bytes are PyTorch's pack_uint4 order."""
    torch, ns = _torch_fp4_reference()
    net = _trained_3x64()
    q = IMP.bff.QuantizedNeuralNet(net, fmt)
    n_codes = n_bad = 0
    for i, layer in enumerate(_layers_of(q)):
        W = np.asarray(net.get_layer_weights(i))
        d = _block_scales(fmt, layer)
        x = torch.from_numpy(np.where(d > 0, W / np.where(d > 0, d, 1), 0.0)).float()
        ref = ns["_f32_to_floatx_unpacked"](x, 2, 1).numpy()
        n_out, n_in = W.shape
        kp = (n_in + 31) // 32 * 32
        padded = np.zeros((n_out, kp), dtype=np.uint8)
        padded[:, :n_in] = ref
        ref_bytes = ns["pack_uint4"](torch.from_numpy(padded)).numpy().ravel().tobytes()
        got = layer["codes"]
        assert len(got) == len(ref_bytes)
        a = np.frombuffer(got, dtype=np.uint8)
        b = np.frombuffer(ref_bytes, dtype=np.uint8)
        # compare nibble by nibble (float32 vs double quotient may differ at an exact tie)
        n_bad += int(np.sum((a & 15) != (b & 15)) + np.sum((a >> 4) != (b >> 4)))
        n_codes += 2 * a.size
    print("\n%s: %d of %d E2M1 codes differ from PyTorch's encoder" % (fmt, n_bad, n_codes))
    assert n_bad <= n_codes * 1e-4


def test_nvfp4_block_scales_match_pytorch_float8():
    """Each NVFP4 block scale is PyTorch's float8_e4m3fn cast of
    absmax_block / (6 g), g = float32(absmax / (6 * 448))."""
    torch, _ = _torch_fp4_reference()
    net = _trained_3x64()
    q = IMP.bff.QuantizedNeuralNet(net, "nvfp4")
    for i, layer in enumerate(_layers_of(q)):
        W = np.asarray(net.get_layer_weights(i))
        n_out, n_in = W.shape
        g = np.float32(np.abs(W).max() / (6 * 448))
        assert np.float32(layer["tensor_scale"]) == g
        kp = (n_in + 31) // 32 * 32
        Wp = np.zeros((n_out, kp))
        Wp[:, :n_in] = W
        amax = np.abs(Wp.reshape(n_out, kp // 16, 16)).max(axis=2)
        want = torch.from_numpy(np.where(amax > 0, amax / (6 * float(g)), 1.0)).to(
            torch.float8_e4m3fn).view(torch.uint8).numpy().ravel()
        got = np.frombuffer(layer["scales"], dtype=np.uint8)
        np.testing.assert_array_equal(got, want)


def test_mxfp4_scales_are_the_ocp_floor_rule_and_relate_to_pytorch_rceil():
    """bff: X = 2^(floor(log2 amax) - 2) (OCP MX v1.0). PyTorch's to_mxfp uses
    RCEIL, 2^ceil(log2(amax / 6)): equal when frac(log2 amax) <= log2 1.5,
    else twice bff's. Both checked block by block."""
    torch, ns = _torch_fp4_reference()
    net = _trained_3x64()
    q = IMP.bff.QuantizedNeuralNet(net, "mxfp4")
    n_same = n_twice = 0
    for i, layer in enumerate(_layers_of(q)):
        W = np.asarray(net.get_layer_weights(i))
        n_out, n_in = W.shape
        kp = (n_in + 31) // 32 * 32
        Wp = np.zeros((n_out, kp), dtype=np.float32)
        Wp[:, :n_in] = W
        amax = np.abs(Wp.reshape(n_out, kp // 32, 32).astype(float)).max(axis=2)
        e = np.frombuffer(layer["scales"], dtype=np.uint8).reshape(n_out, kp // 32).astype(int) - 127
        nz = amax > 0
        np.testing.assert_array_equal(e[nz], np.floor(np.log2(amax[nz])).astype(int) - 2)
        if "to_mxfp" in ns:
            rceil, _ = ns["to_mxfp"](torch.from_numpy(Wp), 32, "mxfp4")
            er = rceil.view(torch.uint8).numpy().astype(int) - 127
            frac = np.log2(amax[nz]) - np.floor(np.log2(amax[nz]))
            same = frac <= np.log2(1.5) + 1e-12
            np.testing.assert_array_equal(er[nz][same], e[nz][same])
            np.testing.assert_array_equal(er[nz][~same], e[nz][~same] + 1)
            n_same += int(same.sum())
            n_twice += int((~same).sum())
    print("\nmxfp4: %d blocks OCP == RCEIL, %d blocks RCEIL = 2 x OCP" % (n_same, n_twice))


def test_e4m3_codec_matches_pytorch_on_all_codes():
    """bff's E4M3 decode, through a network's scales, is PyTorch's
    float8_e4m3fn: a synthetic nvfp4 net whose blocks hit many scale codes."""
    torch, _ = _torch_fp4_reference()
    r = np.random.default_rng(3)
    n_in = 16 * 256
    # one magnitude a block, 2^-16 .. 1, so the block scales sweep the E4M3 range
    w = (r.uniform(-1, 1, size=(256, 16)) * np.exp2(r.uniform(-16, 0, size=(256, 1)))).reshape(1, n_in)
    doc = {"format": "bff.neural_net", "version": 1,
           "layers": [{"n_in": n_in, "n_out": 1, "activation": "identity",
                       "weight": w.ravel().tolist(), "bias": [0.0]}]}
    net = IMP.bff.NeuralNet(msgpack.packb(doc, use_bin_type=True))
    layer = _layers_of(IMP.bff.QuantizedNeuralNet(net, "nvfp4"))[0]
    codes = np.frombuffer(layer["scales"], dtype=np.uint8)
    assert len(set(codes.tolist())) > 30
    g = float(layer["tensor_scale"])
    amax = np.abs(w.reshape(-1, 16)).max(axis=1)
    want = torch.from_numpy(amax / (6 * g)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    np.testing.assert_array_equal(codes, want)


# ---------------------------------------------------------------------------
# Accuracy and size
# ---------------------------------------------------------------------------

FIXTURE_NAMES = ["mlp_torch_legacy.onnx", "mlp_torch_dynamo.onnx", "mlp_matmul_add.onnx",
                 "mlp_state_dict.safetensors", "mlp_relu_nometa.safetensors"]

# Bounds on the error relative to the output absmax, set ~1.5x above what was
# measured 2026-09-24 (okf/neural-net.md has the table).
FIXTURE_BOUND = {"fp4": 0.12, "mxfp4": 0.3, "nvfp4": 0.15}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fp4_error_on_the_fixtures(name):
    net = _fixture(name)
    X = np.random.default_rng(0).uniform(-2, 2, size=(512, 2))
    exact = _batch(net, X)
    line = []
    for fmt in FORMATS:
        errs = []
        for qa in (False, True):
            q = IMP.bff.QuantizedNeuralNet(net, fmt, qa)
            assert (q.get_n_inputs(), q.get_n_outputs(), q.get_n_layers()) == (
                net.get_n_inputs(), net.get_n_outputs(), net.get_n_layers())
            approx = np.asarray(q.predict(X.ravel().tolist(), X.shape[0])).reshape(exact.shape)
            errs.append(_rel_err(approx, exact))
        line.append("%s %.3g / W4A4 %.3g" % (fmt, errs[0], errs[1]))
        assert errs[0] <= FIXTURE_BOUND[fmt], (fmt, errs)
    print("\n%s: %s" % (name, "; ".join(line)))


def test_fp4_on_a_trained_net_errors_sizes_and_speed():
    net = _trained_3x64()
    rng = np.random.default_rng(5)
    Xt = rng.uniform(-2, 2, size=(256, 2))
    exact = _batch(net, Xt)
    n_w = _n_weights(net)
    err = {}
    for fmt in ["int8"] + FORMATS:
        for qa in (False, True):
            if fmt == "int8" and qa:
                continue
            q = IMP.bff.QuantizedNeuralNet(net, fmt, qa)
            assert q.get_format() == fmt and q.get_n_weights() == n_w
            approx = np.asarray(q.predict(Xt.ravel().tolist(), 256)).reshape(exact.shape)
            err[fmt, qa] = _rel_err(approx, exact)
    sizes = {fmt: IMP.bff.QuantizedNeuralNet(net, fmt).get_weight_bytes() for fmt in ["int8"] + FORMATS}
    bits = {fmt: IMP.bff.QuantizedNeuralNet(net, fmt).get_bits_per_weight() for fmt in ["int8"] + FORMATS}
    print("\ntrained 2-64-64-64-1 tanh net, %d weights (%d bytes as float64):" % (n_w, 8 * n_w))
    for fmt in ["int8"] + FORMATS:
        print("  %-5s W%sA8 error %.3g%s, %d bytes (%.1fx under float64), %.3f bits/weight" % (
            fmt, "8" if fmt == "int8" else "4", err[fmt, False],
            "" if fmt == "int8" else ", W4A4 %.3g" % err[fmt, True],
            sizes[fmt], 8 * n_w / sizes[fmt], bits[fmt]))
    # sizes: int8 one byte a weight + 8 a layer; FP4 rows padded to 32
    assert sizes["int8"] == n_w + 8 * net.get_n_layers()
    dims = [(net.get_layer_weights(i).shape) for i in range(net.get_n_layers())]
    kp = lambda n_in: (n_in + 31) // 32 * 32
    assert sizes["fp4"] == sum(o * kp(i) // 2 + 4 * o for o, i in dims)
    assert sizes["mxfp4"] == sum(o * kp(i) // 2 + o * kp(i) // 32 for o, i in dims)
    assert sizes["nvfp4"] == sum(o * kp(i) // 2 + o * kp(i) // 16 + 4 for o, i in dims)
    # 11.5x under float64 here: the 2-input layer's rows are padded to 32
    # codes; wide layers reach 64 / 4.5 = 14.2x (test below)
    assert 8 * n_w / sizes["nvfp4"] > 11
    # the weights themselves: finer blocks reconstruct better (nvfp4 < fp4)
    werr = {}
    for fmt in FORMATS:
        q = IMP.bff.QuantizedNeuralNet(net, fmt)
        num = den = 0.0
        for i, layer in enumerate(_layers_of(q)):
            W = np.asarray(net.get_layer_weights(i))
            num += float(np.sum((_decode_weights(fmt, layer) - W) ** 2))
            den += float(np.sum(W ** 2))
        werr[fmt] = np.sqrt(num / den)
    print("  weight relative RMS error: " + ", ".join("%s %.3g" % (f, werr[f]) for f in FORMATS))
    assert werr["nvfp4"] < werr["fp4"] and werr["nvfp4"] < werr["mxfp4"], werr
    # outputs: all bounded (measured 0.04 / 0.13 / 0.08 for fp4 / mxfp4 /
    # nvfp4 -- nvfp4's E4M3 block scales cost it the per-row fp4 advantage
    # on this 64-wide net, see okf/neural-net.md); A4 costs more than A8
    for fmt, bound in (("fp4", 0.08), ("mxfp4", 0.2), ("nvfp4", 0.12)):
        assert err[fmt, False] <= bound, (fmt, err)
        assert err[fmt, True] > err[fmt, False], (fmt, err)
    assert err["int8", False] <= min(err[f, False] for f in FORMATS)


def test_fp4_bits_per_weight_on_wide_layers():
    doc = {"format": "bff.neural_net", "version": 1, "layers": []}
    r = np.random.default_rng(1)
    for n_in, n_out in [(256, 256), (256, 128)]:
        doc["layers"].append({"n_in": n_in, "n_out": n_out, "activation": "tanh",
                              "weight": (r.normal(size=(n_out, n_in)) / 16).ravel().tolist(),
                              "bias": [0.0] * n_out})
    net = IMP.bff.NeuralNet(msgpack.packb(doc, use_bin_type=True))
    n_w = 256 * 256 + 256 * 128
    assert IMP.bff.QuantizedNeuralNet(net, "mxfp4").get_bits_per_weight() == pytest.approx(4.25)
    assert IMP.bff.QuantizedNeuralNet(net, "nvfp4").get_bits_per_weight() == pytest.approx(4.5 + 64.0 / n_w)
    assert IMP.bff.QuantizedNeuralNet(net, "fp4").get_bits_per_weight() == pytest.approx(
        4 + 32.0 * (256 + 128) / n_w)
    q = IMP.bff.QuantizedNeuralNet(net, "nvfp4")
    assert 8 * n_w / q.get_weight_bytes() == pytest.approx(64 / 4.5, rel=1e-3)  # ~14.2x under float64


# ---------------------------------------------------------------------------
# Documents and refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,qa", [("int8", False), ("fp4", False), ("mxfp4", False),
                                    ("nvfp4", False), ("nvfp4", True), ("mxfp4", True)])
def test_msgpack_round_trip_is_bit_exact(fmt, qa):
    net = _fixture("mlp_state_dict.safetensors")
    q = IMP.bff.QuantizedNeuralNet(net, fmt, qa)
    doc = q.to_msgpack()
    assert isinstance(doc, bytes)
    q2 = IMP.bff.QuantizedNeuralNet.from_msgpack(doc)
    assert q2.to_msgpack() == doc
    assert (q2.get_format(), q2.get_quantize_activations(), q2.get_weight_bytes()) == (
        fmt, qa or fmt == "int8", q.get_weight_bytes())
    X = np.random.default_rng(2).normal(size=(64, 2))
    a = np.asarray(q.predict(X.ravel().tolist(), 64))
    b = np.asarray(q2.predict(X.ravel().tolist(), 64))
    np.testing.assert_array_equal(a, b)
    d = msgpack.unpackb(doc, raw=False)
    assert d["format"] == "bff.quantized_neural_net" and d["quantization"] == fmt
    key = "weight" if fmt == "int8" else "codes"
    assert isinstance(d["layers"][0][key], bytes)
    assert d["layers"][0]["activation"] in ("tanh", "identity", "relu", "logistic", "silu", "softplus")


def test_refusals():
    net = _fixture("mlp_state_dict.safetensors")
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet(net, "fp8")
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet(net, "NVFP4")
    q = IMP.bff.QuantizedNeuralNet(net, "nvfp4")
    doc = q.to_msgpack()
    with pytest.raises(TypeError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(doc.decode("latin-1"))
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(b"")
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(net.to_msgpack())  # a bff.neural_net
    d = msgpack.unpackb(doc, raw=False)
    d["layers"][0]["codes"] = d["layers"][0]["codes"][:-1]
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(msgpack.packb(d, use_bin_type=True))
    d = msgpack.unpackb(doc, raw=False)
    d["quantization"] = "fp16"
    with pytest.raises(ValueError):
        IMP.bff.QuantizedNeuralNet.from_msgpack(msgpack.packb(d, use_bin_type=True))
    with pytest.raises(ValueError):
        q.predict([1.0, 2.0, 3.0], 1)
    assert "nvfp4" in repr(q)


def test_rows_do_not_depend_on_the_batch():
    net = _trained_3x64()
    X = np.random.default_rng(1).normal(size=(20, 2))
    for fmt in FORMATS:
        for qa in (False, True):
            q = IMP.bff.QuantizedNeuralNet(net, fmt, qa)
            whole = np.asarray(q.predict(X.ravel().tolist(), 20))
            one = np.asarray(q.predict(X[7].tolist(), 1))
            np.testing.assert_array_equal(one, whole[7:8])


# ---------------------------------------------------------------------------
# Speed, reported only
# ---------------------------------------------------------------------------

def _timed(f, reps=20):
    best = 1e300
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def test_speed_report_256_256_128():
    """24-256-256-128-8 tanh net (the HMM surrogate's shape) at batch 1, 32,
    256: double (MatGemm) against each format's kernels, through Python."""
    r = np.random.default_rng(0)
    dims = [24, 256, 256, 128, 8]
    doc = {"format": "bff.neural_net", "version": 1, "layers": []}
    for i in range(4):
        doc["layers"].append({"n_in": dims[i], "n_out": dims[i + 1],
                              "activation": "tanh" if i < 3 else "identity",
                              "weight": (r.normal(size=(dims[i + 1], dims[i])) / np.sqrt(dims[i])).ravel().tolist(),
                              "bias": (0.1 * r.normal(size=dims[i + 1])).tolist()})
    net = IMP.bff.NeuralNet(msgpack.packb(doc, use_bin_type=True))
    X = r.normal(size=(256, 24))
    print("\nkernel %s; 24-256-256-128-8 tanh, us per predict (best of 20)"
          % IMP.bff.QuantizedNeuralNet.get_kernel_name())
    qs = {(f, qa): IMP.bff.QuantizedNeuralNet(net, f, qa) for f in FORMATS for qa in (False, True)}
    qs["int8", False] = IMP.bff.QuantizedNeuralNet(net, "int8")
    for b in (1, 32, 256):
        xl = X[:b].ravel().tolist()
        t = {"double": _timed(lambda: net.predict(xl, b))}
        for key, q in qs.items():
            t[key] = _timed(lambda: q.predict(xl, b))
        print("  batch %3d: double %.1f | int8 %.1f | %s" % (
            b, t["double"] * 1e6, t["int8", False] * 1e6,
            " | ".join("%s W4A8 %.1f W4A4 %.1f" % (f, t[f, False] * 1e6, t[f, True] * 1e6)
                       for f in FORMATS)))
