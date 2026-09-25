"""The network as a differentiable building block, and networks from elsewhere.

Ported from tttrlib's test/python/test_neural_net.py (the network moved to
bff; tttrlib dropped it in 10858cf19), in bff's terms: msgpack documents in
place of JSON, `get_` names, managed numpy views, ValueError for every
refusal. Also covered here: the ONNX / safetensors fixtures in
test/input/nn against PyTorch's own outputs, the C++ core test
(cpp_snippets/test_mlp_core.cpp: derivatives two ways, MatGemm parity), and
the int8 deployment path (QuantizedNeuralNet).

Training (Adam) is test_neural_net_training.py's; the batch forward pass and
the accelerator are test_neural_net.py's.
"""

import json
import os
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


def _pack(doc):
    return msgpack.packb(doc, use_bin_type=True)


def _handmade_net(w1, b1, w2, b2, act="relu", x_scaler=None, y_scaler=None):
    """A 2-layer net from explicit weights, through the msgpack reader."""
    doc = {
        "format": "bff.neural_net", "version": 1,
        "layers": [
            {"n_in": w1.shape[1], "n_out": w1.shape[0], "activation": act,
             "weight": w1.ravel().tolist(), "bias": b1.tolist()},
            {"n_in": w2.shape[1], "n_out": w2.shape[0], "activation": "identity",
             "weight": w2.ravel().tolist(), "bias": b2.tolist()},
        ],
    }
    if x_scaler is not None:
        doc["x_scaler"] = {"mean": list(x_scaler[0]), "scale": list(x_scaler[1])}
    if y_scaler is not None:
        doc["y_scaler"] = {"mean": list(y_scaler[0]), "scale": list(y_scaler[1])}
    return IMP.bff.NeuralNet(_pack(doc))


def _net(dims, activation="tanh", seed=0, scalers=True):
    """A deeper net with active scalers, so the chain rule through them is
    exercised, not just the raw layers."""
    r = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        layers.append({"n_in": n_in, "n_out": n_out,
                       "activation": activation if i < len(dims) - 2 else "identity",
                       "weight": (r.normal(size=(n_out, n_in)) / np.sqrt(n_in)).ravel().tolist(),
                       "bias": r.normal(0.0, 0.2, n_out).tolist()})
    doc = {"format": "bff.neural_net", "version": 1, "layers": layers}
    if scalers:
        doc["x_scaler"] = {"mean": r.normal(size=dims[0]).tolist(),
                           "scale": r.uniform(0.5, 2.0, dims[0]).tolist()}
        doc["y_scaler"] = {"mean": r.normal(size=dims[-1]).tolist(),
                           "scale": r.uniform(0.5, 2.0, dims[-1]).tolist()}
    return IMP.bff.NeuralNet(_pack(doc))


def _batch(net, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(
        X.shape[0], net.get_n_outputs())


def _fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


_ACTS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "identity": lambda z: z,
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "softplus": lambda z: np.logaddexp(0.0, z),
    "silu": lambda z: z / (1.0 + np.exp(-z)),
    "sin": np.sin,
}


# ---------------------------------------------------------------------------
# Evaluation and shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("act", sorted(_ACTS))
def test_one_sample_matches_numpy_and_the_document_round_trips(act):
    rng = np.random.default_rng(1)
    w1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5)
    w2, b2 = rng.normal(size=(2, 5)), rng.normal(size=2)
    net = _handmade_net(w1, b1, w2, b2, act=act)
    x = rng.normal(size=3)
    expect = w2 @ _ACTS[act](w1 @ x + b1) + b2
    np.testing.assert_allclose(net.predict(x.tolist()), expect, rtol=0, atol=1e-12)
    again = IMP.bff.NeuralNet(net.to_msgpack())
    assert msgpack.unpackb(again.to_msgpack(), raw=False)["layers"][0]["activation"] == act
    assert again.get_layer_activation(0) == act
    np.testing.assert_array_equal(again.predict(x.tolist()), net.predict(x.tolist()))


def test_batch_matches_single():
    rng = np.random.default_rng(3)
    net = _handmade_net(rng.normal(size=(6, 4)), rng.normal(size=6),
                        rng.normal(size=(3, 6)), rng.normal(size=3))
    X = rng.normal(size=(10, 4))
    single = np.array([net.predict(row.tolist()) for row in X])
    np.testing.assert_allclose(_batch(net, X), single, rtol=0, atol=1e-12)


def test_introspection():
    rng = np.random.default_rng(5)
    w1, b1 = rng.normal(size=(6, 4)), rng.normal(size=6)
    w2, b2 = rng.normal(size=(3, 6)), rng.normal(size=3)
    net = _handmade_net(w1, b1, w2, b2)
    assert (net.get_n_inputs(), net.get_n_outputs(), net.get_n_layers()) == (4, 3, 2)
    assert net.get_n_parameters() == w1.size + b1.size + w2.size + b2.size
    np.testing.assert_array_equal(net.get_layer_weights(0), w1)
    np.testing.assert_array_equal(net.get_layer_bias(1), b2)
    assert net.get_layer_activation(0) == "relu"
    assert net.get_layer_activation(1) == "identity"
    assert repr(net) == "NeuralNet(4->6->3, %d parameters)" % net.get_n_parameters()
    with pytest.raises(ValueError, match="out of range"):
        net.get_layer_weights(2)


@pytest.mark.parametrize("layers,extra,needle", [
    ([{"n_in": 2, "n_out": 3, "activation": "relu", "weight": [1, 2, 3], "bias": [0, 0, 0]}],
     {}, "weight"),
    ([{"n_in": 2, "n_out": 2, "activation": "relu", "weight": [1, 2, 3, 4], "bias": [0, 0]},
      {"n_in": 5, "n_out": 1, "activation": "identity", "weight": [1] * 5, "bias": [0]}],
     {}, "expects"),
    ([{"n_in": 2, "n_out": 2, "activation": "banana", "weight": [1, 2, 3, 4], "bias": [0, 0]}],
     {}, "activation"),
    ([{"n_in": 2, "n_out": 2, "activation": "relu", "weight": [1, 2, 3, 4], "bias": [0]}],
     {}, "bias"),
    ([{"n_in": 2, "n_out": 1, "activation": "identity", "weight": [1, 1], "bias": [0]}],
     {"x_scaler": {"mean": [0, 0, 0], "scale": [1, 1, 1]}}, "x_scaler"),
])
def test_malformed_model_rejected(layers, extra, needle):
    doc = dict(format="bff.neural_net", version=1, layers=layers, **extra)
    with pytest.raises(ValueError, match=needle):
        IMP.bff.NeuralNet(_pack(doc))


def test_wrong_input_width_raises():
    net = _net([3, 4, 2])
    with pytest.raises(ValueError):
        net.predict(np.zeros(7).tolist())
    with pytest.raises(ValueError, match="columns"):
        net.predict_derivatives(np.zeros((2, 7)), order=0)
    with pytest.raises(ValueError, match="columns"):
        net.backward(np.zeros((2, 3)), np.zeros((2, 5)))
    with pytest.raises(ValueError, match="order"):
        net.predict_derivatives(np.zeros((2, 3)), np.zeros((2, 3)), order=3)
    with pytest.raises(ValueError, match="directions"):
        net.predict_derivatives(np.zeros((2, 3)), order=1)


# ---------------------------------------------------------------------------
# Derivatives (internal/MlpCore.h)
# ---------------------------------------------------------------------------

def test_parameters_round_trip_and_drive_predictions():
    net = _net([2, 8, 8, 2])
    p = net.get_parameters()
    assert p.shape == (net.get_n_parameters(),)
    # layout: layer 0 weight (row-major), layer 0 bias, layer 1 weight, ...
    w0 = net.get_layer_weights(0)
    np.testing.assert_array_equal(p[: w0.size], w0.ravel())
    x = [0.2, -0.4]
    before = net.predict(x).copy()
    net.set_parameters(p * 1.1)
    assert not np.allclose(net.predict(x), before)
    # the stored document follows the parameters
    np.testing.assert_array_equal(IMP.bff.NeuralNet(net.to_msgpack()).predict(x), net.predict(x))
    net.set_parameters(p)
    np.testing.assert_array_equal(net.predict(x), before)
    with pytest.raises(ValueError):
        net.set_parameters(p[:-1])


@pytest.mark.parametrize("act", ["tanh", "softplus", "silu", "sin"])
def test_backward_matches_finite_differences(act):
    """dL/dparams and dL/dx for an arbitrary loss L = <W, y>, through the
    stored scalers, against central differences."""
    net = _net([2, 8, 8, 2], activation=act, seed=11)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(6, 2))
    W = rng.normal(size=(6, net.get_n_outputs()))
    dparams, dx = net.backward(X, W)
    assert dx.shape == X.shape

    def loss_params(p):
        net.set_parameters(p)
        return float(np.sum(W * _batch(net, X)))

    p0 = net.get_parameters().copy()
    fd = _fd_grad(loss_params, p0)
    net.set_parameters(p0)
    np.testing.assert_allclose(dparams, fd, rtol=1e-6, atol=1e-8)
    fdx = _fd_grad(lambda xf: float(np.sum(W * _batch(net, xf.reshape(X.shape)))), X.ravel())
    np.testing.assert_allclose(dx.ravel(), fdx, rtol=1e-6, atol=1e-8)


def test_jacobian_and_hessian_match_finite_differences():
    net = _net([2, 8, 8, 2], activation="tanh", seed=12)
    x = np.array([0.3, -0.2])
    J = net.jacobian(x.tolist())
    assert J.shape == (net.get_n_outputs(), net.get_n_inputs())
    for k in range(net.get_n_outputs()):
        fd = _fd_grad(lambda xx: net.predict(xx.tolist())[k], x)
        np.testing.assert_allclose(J[k], fd, rtol=1e-6, atol=1e-8)
        H = net.hessian(x.tolist(), k)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, H.T, atol=1e-12)
        fdH = np.array([_fd_grad(lambda xx: net.jacobian(xx.tolist())[k, i], x, h=1e-5)
                        for i in range(2)])
        np.testing.assert_allclose(H, fdH, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        net.hessian(x.tolist(), 2)


def test_predict_derivatives_orders():
    net = _net([2, 8, 8, 2], activation="silu", seed=13)
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(5, 2))
    V = rng.normal(size=(5, 2))
    y0, d1, d2 = net.predict_derivatives(X, V, order=2)
    assert y0.shape == d1.shape == d2.shape == (5, 2)
    np.testing.assert_allclose(y0, _batch(net, X), atol=1e-13)
    for r in range(5):
        J = net.jacobian(X[r].tolist())
        np.testing.assert_allclose(d1[r], J @ V[r], rtol=1e-10, atol=1e-12)
        for k in range(net.get_n_outputs()):
            H = net.hessian(X[r].tolist(), k)
            np.testing.assert_allclose(d2[r, k], V[r] @ H @ V[r], rtol=1e-8, atol=1e-10)
    y_only, none1, none2 = net.predict_derivatives(X, order=0)
    assert none1 is None and none2 is None
    np.testing.assert_array_equal(y_only, y0)
    _, one, none2 = net.predict_derivatives(X, V, order=1)
    assert none2 is None
    np.testing.assert_array_equal(one, d1)


def test_backward_derivatives_matches_finite_differences():
    """The adjoint of the Taylor-augmented pass: a loss on y, J v and v^T H v,
    differentiated with respect to the parameters, inputs and directions."""
    net = _net([2, 8, 8, 2], activation="tanh", seed=14)
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, size=(4, 2))
    V = rng.normal(size=(4, 2))
    C0, C1, C2 = (rng.normal(size=(4, net.get_n_outputs())) for _ in range(3))

    def loss(p=None, Xa=None, Va=None):
        if p is not None:
            net.set_parameters(p)
        y, d1, d2 = net.predict_derivatives(X if Xa is None else Xa,
                                            V if Va is None else Va, order=2)
        return float(np.sum(C0 * y) + np.sum(C1 * d1) + np.sum(C2 * d2))

    dparams, dx, dv = net.backward_derivatives(X, V, C0, dY1=C1, dY2=C2)
    p0 = net.get_parameters().copy()
    fd = _fd_grad(lambda p: loss(p=p), p0, h=1e-5)
    net.set_parameters(p0)
    np.testing.assert_allclose(dparams, fd, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        dx.ravel(), _fd_grad(lambda xf: loss(Xa=xf.reshape(X.shape)), X.ravel(), h=1e-5),
        rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        dv.ravel(), _fd_grad(lambda vf: loss(Va=vf.reshape(V.shape)), V.ravel(), h=1e-5),
        rtol=1e-5, atol=1e-7)
    # order 0: the plain backward, dv zero
    p_a, x_a, v_a = net.backward_derivatives(X, None, C0)
    p_b, x_b = net.backward(X, C0)
    np.testing.assert_array_equal(p_a, p_b)
    np.testing.assert_array_equal(x_a, x_b)
    assert not np.any(v_a)
    with pytest.raises(ValueError, match="V must have"):
        net.backward_derivatives(X, None, C0, dY1=C1)


def test_pinn_poisson_1d():
    """End to end: a physics-informed fit of u'' = f on [0, 1], u(0) = u(1) = 0,
    with f = -pi^2 sin(pi x), so u = sin(pi x). The loss is the PDE residual at
    collocation points plus the boundary values; its gradient with respect to
    the weights comes from backward_derivatives (order 2, direction v = 1) and
    goes into L-BFGS. This is the whole reason the derivative API exists."""
    pytest.importorskip("scipy")
    from scipy.optimize import minimize

    net = _net([1, 16, 16, 1], activation="tanh", seed=0, scalers=False)
    net.set_parameters(net.get_parameters() * 0.8)
    xc = np.linspace(0.0, 1.0, 41)[:, None]
    xb = np.array([[0.0], [1.0]])
    f = -np.pi ** 2 * np.sin(np.pi * xc)
    ones = np.ones_like(xc)

    def objective(p):
        net.set_parameters(p)
        _, _, u_xx = net.predict_derivatives(xc, ones, order=2)
        ub, _, _ = net.predict_derivatives(xb, order=0)
        r = u_xx - f
        loss = np.mean(r ** 2) + 10.0 * np.mean(ub ** 2)
        g_c, _, _ = net.backward_derivatives(xc, ones, np.zeros_like(r), dY2=2 * r / r.size)
        g_b, _ = net.backward(xb, 10.0 * 2 * ub / ub.size)
        return loss, g_c + g_b

    res = minimize(objective, net.get_parameters().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 800, "ftol": 1e-14, "gtol": 1e-10})
    net.set_parameters(res.x)
    xt = np.linspace(0, 1, 101)[:, None]
    err = np.abs(_batch(net, xt).ravel() - np.sin(np.pi * xt.ravel())).max()
    assert err < 2e-2, (err, res.message)


def test_matches_sklearn_forward_pass():
    """A net converted from scikit-learn weights predicts identically."""
    pytest.importorskip("sklearn")
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1234)
    X = rng.uniform(-1.0, 1.0, size=(500, 2))
    Y = np.column_stack([np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2, X[:, 0] * X[:, 1]])
    xs, ys = StandardScaler().fit(X), StandardScaler().fit(Y)
    mlp = MLPRegressor(hidden_layer_sizes=(16, 16), activation="relu",
                       max_iter=60, random_state=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(xs.transform(X), ys.transform(Y))
    n_hidden = len(mlp.coefs_) - 1
    layers = [{"n_in": w.shape[0], "n_out": w.shape[1],
               "activation": mlp.activation if i < n_hidden else mlp.out_activation_,
               "weight": np.ascontiguousarray(w.T).ravel().tolist(),
               "bias": np.asarray(b).tolist()}
              for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_))]
    net = IMP.bff.NeuralNet(_pack({
        "format": "bff.neural_net", "version": 1,
        "x_scaler": {"mean": xs.mean_.tolist(), "scale": xs.scale_.tolist()},
        "y_scaler": {"mean": ys.mean_.tolist(), "scale": ys.scale_.tolist()},
        "layers": layers}))
    expect = ys.inverse_transform(mlp.predict(xs.transform(X)))
    np.testing.assert_allclose(_batch(net, X), expect, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Standard file formats in: ONNX and safetensors, as PyTorch writes them
# ---------------------------------------------------------------------------

def _expected():
    with open(os.path.join(FIXTURES, "expected.json")) as fh:
        e = json.load(fh)
    return {k: np.asarray(v) for k, v in e.items()}


@pytest.mark.parametrize("name,key,tol", [
    ("mlp_torch_legacy.onnx", "Y_float32", 1e-6),  # torch.onnx.export(dynamo=False): Gemm + Tanh / Sigmoid*Mul / Softplus
    ("mlp_torch_dynamo.onnx", "Y_float32", 1e-6),  # dynamo=True: + Greater/Where softplus threshold
    ("mlp_matmul_add.onnx", "Y_double", 1e-12),    # MatMul + Add, (n_in, n_out) weights, double
])
def test_from_onnx_matches_pytorch(name, key, tol):
    e = _expected()
    path = os.path.join(FIXTURES, name)
    net = IMP.bff.NeuralNet.from_onnx_file(path)
    assert (net.get_n_layers(), net.get_n_inputs(), net.get_n_outputs()) == (4, 2, 2)
    assert [net.get_layer_activation(i) for i in range(4)] == ["tanh", "silu", "softplus", "identity"]
    np.testing.assert_allclose(_batch(net, e["X"]), e[key], rtol=0, atol=tol)
    # the bytes form is the same network, from any bytes-like
    with open(path, "rb") as fh:
        raw = fh.read()
    for form in (raw, bytearray(raw), memoryview(raw)):
        np.testing.assert_array_equal(_batch(IMP.bff.NeuralNet.from_onnx(form), e["X"]),
                                      _batch(net, e["X"]))
    # a full citizen: derivatives, and stored natively as msgpack, bit for bit
    assert net.jacobian(e["X"][0].tolist()).shape == (2, 2)
    native = IMP.bff.NeuralNet(net.to_msgpack())
    np.testing.assert_array_equal(_batch(native, e["X"]), _batch(net, e["X"]))
    np.testing.assert_array_equal(native.get_parameters(), net.get_parameters())


def test_from_safetensors_matches_pytorch():
    e = _expected()
    path = os.path.join(FIXTURES, "mlp_state_dict.safetensors")
    net = IMP.bff.NeuralNet.from_safetensors_file(path)
    np.testing.assert_allclose(_batch(net, e["X"]), e["Y_double"], rtol=0, atol=1e-12)
    with open(path, "rb") as fh:
        again = IMP.bff.NeuralNet.from_safetensors(fh.read())
    np.testing.assert_array_equal(_batch(again, e["X"]), _batch(net, e["X"]))
    relu = IMP.bff.NeuralNet.from_safetensors_file(
        os.path.join(FIXTURES, "mlp_relu_nometa.safetensors"), hidden_activation="relu")
    assert relu.get_n_layers() == 2
    assert relu.get_layer_activation(0) == "relu"
    np.testing.assert_allclose(_batch(relu, e["X"]), e["Y_relu"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(_batch(IMP.bff.NeuralNet(relu.to_msgpack()), e["X"]),
                                  _batch(relu, e["X"]))


def test_bad_imports_are_refused_clearly():
    with pytest.raises(ValueError, match="ONNX"):
        IMP.bff.NeuralNet.from_onnx(b"\x00\x01not onnx")
    with pytest.raises(ValueError, match="safetensors"):
        IMP.bff.NeuralNet.from_safetensors(b"\x01")
    with pytest.raises(TypeError, match="ONNX"):
        IMP.bff.NeuralNet.from_onnx("model.onnx")
    with pytest.raises(TypeError, match="safetensors"):
        IMP.bff.NeuralNet.from_safetensors("model.safetensors")
    with pytest.raises(Exception, match="cannot open"):
        IMP.bff.NeuralNet.from_onnx_file(os.path.join(FIXTURES, "missing.onnx"))


def test_roundtrip_through_pytorch_live():
    """When PyTorch and onnx are installed: build, export and compare live, so
    a change in the exporter's graph shape shows up here before a user sees it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    nn = torch.nn
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(3, 7), nn.Sigmoid(), nn.Linear(7, 5), nn.ReLU(),
                      nn.Linear(5, 2)).double()
    X = torch.rand(9, 3, dtype=torch.float64) * 2 - 1
    with torch.no_grad():
        Y = m(X).numpy()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.onnx")
        torch.onnx.export(m, X[:1], path, dynamo=False, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamic_axes={"x": {0: "n"}, "y": {0: "n"}})
        net = IMP.bff.NeuralNet.from_onnx_file(path)
    np.testing.assert_allclose(_batch(net, X.numpy()), Y, atol=1e-12)


# ---------------------------------------------------------------------------
# The C++ core test, compiled against bff's headers alone
# ---------------------------------------------------------------------------

def test_mlp_core_cpp_checks_pass():
    """cpp_snippets/test_mlp_core.cpp: activation derivatives, the Taylor
    passes and their adjoint vs central differences and the Dual pass, the
    scaler chain rule, ONNX import, MatGemm vs PortableGemm parity, and the
    int8 path. Timings it prints are reported, not asserted."""
    if sys.platform == "win32":
        pytest.skip("standalone C++ snippet compilation not supported on Windows")
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    repo = os.path.dirname(HERE)
    with tempfile.TemporaryDirectory() as tmp:
        inc = os.path.join(tmp, "IMP", "bff")
        os.makedirs(inc)
        os.symlink(os.path.join(repo, "include", "internal"), os.path.join(inc, "internal"))
        exe = os.path.join(tmp, "test_mlp_core")
        subprocess.check_call([cxx, "-std=c++17", "-O2", "-I", tmp,
                               os.path.join(HERE, "cpp_snippets", "test_mlp_core.cpp"), "-o", exe])
        run = subprocess.run([exe, FIXTURES], capture_output=True, text=True)
    print("\n".join(line for line in run.stdout.splitlines() if line.startswith("  time")))
    assert run.returncode == 0, run.stdout
    assert "0 failure(s)" in run.stdout
    assert "mlp_torch_dynamo.onnx" in run.stdout  # the fixtures were found


def test_the_blocked_gemm_is_the_portable_arithmetic_reassociated():
    """NeuralNet runs its products through MatGemm (internal/MlpGemm.h); on a
    batch large enough for the blocked kernel its forward pass agrees with a
    numpy evaluation to 1e-12 relative."""
    net = _net([5, 64, 64, 64, 3], activation="tanh", seed=21)
    X = np.random.default_rng(22).normal(size=(256, 5))
    doc = msgpack.unpackb(net.to_msgpack(), raw=False)
    A = (X - np.asarray(doc["x_scaler"]["mean"])) / np.asarray(doc["x_scaler"]["scale"])
    for layer in doc["layers"]:
        W = np.asarray(layer["weight"]).reshape(layer["n_out"], layer["n_in"])
        A = _ACTS[layer["activation"]](A @ W.T + np.asarray(layer["bias"]))
    ref = A * np.asarray(doc["y_scaler"]["scale"]) + np.asarray(doc["y_scaler"]["mean"])
    got = _batch(net, X)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


# ---------------------------------------------------------------------------
# int8 deployment path (internal/MlpQuant.h, rebuilt from its spec)
# ---------------------------------------------------------------------------

def _rel_err(approx, exact):
    return np.abs(approx - exact).max() / np.abs(exact).max()


def _n_weights(net):
    return sum(net.get_layer_weights(i).size for i in range(net.get_n_layers()))


@pytest.mark.parametrize("name", ["mlp_torch_legacy.onnx", "mlp_torch_dynamo.onnx",
                                  "mlp_matmul_add.onnx", "mlp_state_dict.safetensors",
                                  "mlp_relu_nometa.safetensors"])
def test_int8_error_is_bounded_on_the_fixtures(name):
    path = os.path.join(FIXTURES, name)
    if name.endswith(".onnx"):
        net = IMP.bff.NeuralNet.from_onnx_file(path)
    else:
        net = IMP.bff.NeuralNet.from_safetensors_file(
            path, hidden_activation="relu" if "relu" in name else "tanh")
    X = np.random.default_rng(0).uniform(-2, 2, size=(512, 2))
    q = IMP.bff.QuantizedNeuralNet(net)
    assert (q.get_n_inputs(), q.get_n_outputs(), q.get_n_layers()) == (
        net.get_n_inputs(), net.get_n_outputs(), net.get_n_layers())
    exact = _batch(net, X)
    approx = np.asarray(q.predict(X.ravel().tolist(), X.shape[0])).reshape(exact.shape)
    assert _rel_err(approx, exact) <= 0.02, _rel_err(approx, exact)
    # one byte a weight (eight as double) and an 8-byte scale a layer
    assert q.get_weight_bytes() == _n_weights(net) + 8 * net.get_n_layers()


def test_int8_error_is_bounded_on_a_trained_net_and_weights_are_8x_smaller():
    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, size=(2000, 2))
    y = np.sin(X[:, 0]) * np.cos(0.5 * X[:, 1]) + 0.1 * X[:, 0] * X[:, 1]
    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = [64, 64, 64]
    opt.activation = "tanh"
    opt.max_iter = 40
    opt.seed = 1
    net = IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(X, y, opt).get_network())
    q = IMP.bff.QuantizedNeuralNet(net)
    n_w = _n_weights(net)
    n_scales = 8 * net.get_n_layers()                             # one double scale a layer
    assert q.get_weight_bytes() == n_w + n_scales                 # int8: one byte a weight
    assert n_w * np.dtype(float).itemsize == 8 * (q.get_weight_bytes() - n_scales)  # 8x under float64
    Xt = rng.uniform(-2, 2, size=(256, 2))
    exact = _batch(net, Xt)
    approx = np.asarray(q.predict(Xt.ravel().tolist(), 256)).reshape(exact.shape)
    err = _rel_err(approx, exact)
    assert err <= 0.02, err
    # speed, reported only (the rebuilt spec claimed ~2x the double portable
    # path at batch 256 on a 3x64 net; this double path is MatGemm's)
    xl = Xt.ravel().tolist()
    t_d = min(_timed(lambda: net.predict(xl, 256)) for _ in range(20))
    t_q = min(_timed(lambda: q.predict(xl, 256)) for _ in range(20))
    print("\nint8 %.3g relative error; batch 256, 3x64: double %.1f us, int8 %.1f us (%.2fx)"
          % (err, t_d * 1e6, t_q * 1e6, t_d / t_q))


def _timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def test_int8_rows_do_not_depend_on_the_batch():
    net = _net([3, 32, 32, 2], activation="silu", seed=5)
    q = IMP.bff.QuantizedNeuralNet(net)
    X = np.random.default_rng(1).normal(size=(20, 3))
    whole = np.asarray(q.predict(X.ravel().tolist(), 20)).reshape(20, 2)
    one = np.asarray(q.predict(X[7].tolist(), 1))
    np.testing.assert_array_equal(one, whole[7])
    with pytest.raises(ValueError):
        q.predict([1.0, 2.0], 1)
