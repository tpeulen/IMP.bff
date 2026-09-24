---
okf_version: "0.2"
---

# Dense networks in bff (`NeuralNet`, `MlpCore.h`)

bff owns the stack's neural network; tttrlib is ML-free (AGENTS.md, placement
exception, tpeulen 2026-09-23). The network started in tttrlib and moved here
in pieces: the kernels (`internal/MlpCore.h`, bff-owned since it diverged),
training (`train_neural_net`, `NeuralNetTraining.h`), and on 2026-09-24 the
rest of tttrlib's `NeuralNet` API (removed there in tttrlib `10858cf19`),
ported in bff's terms.

## What is where

| Piece | File | What it is |
|---|---|---|
| Kernels | `include/internal/MlpCore.h` | std-only: activations and their derivatives to order 3, the Taylor-augmented forward pass (`y`, `J v`, `v^T H v`), its exact adjoint, `flatten`/`unflatten`, `MlpModel` with scalers, document tree <-> model, ONNX and safetensors readers. The GEMM is a template policy (`PortableGemm` default). |
| GEMM policy | `include/internal/MlpGemm.h` | `MatGemm`: MlpCore's `nn`/`nt`/`tn` through the vendored `internal/Mat.h`'s packed, register-blocked kernels (threaded over row tiles with OpenMP). Used by `NeuralNet` and `train_neural_net`. |
| int8 path | `include/internal/MlpQuant.h` | Dynamic-range int8 inference, **rebuilt from the lost tttrlib spec** (below). |
| Documents | `include/internal/NetworkDocument.h` | The one msgpack encoder/decoder. |
| Face | `include/NeuralNet.h`, `src/NeuralNet.cpp` | `NeuralNet`, `QuantizedNeuralNet`; bindings in `pyext/include/IMP_bff.core.i`. |
| Training | `include/NeuralNetTraining.h` | Adam + explicit backprop, sklearn's defaults. |

## The API (C++ / Python)

- Construct: `NeuralNet(MsgpackBytes)`; import `from_onnx(OnnxBytes)`,
  `from_onnx_file(path)`, `from_safetensors(SafetensorsBytes,
  hidden_activation="tanh")`, `from_safetensors_file(path, ...)`. ONNX and
  safetensors are **import formats only**; `to_msgpack()` makes an imported
  network a native document (an action policy, a surrogate's net). There is no
  ONNX writer and no JSON reader.
- Shape: `get_n_inputs/outputs/layers/parameters`, `get_layer_activation(i)`,
  `get_layer_weights(i)` (`(n_out, n_in)` view), `get_layer_bias(i)`.
- Evaluate: `predict(x, n_rows)` (flat, offered to the compute backend) and
  `predict(x)` for one sample (the same path with `n_rows = 1`).
- Differentiate (CPU only): `predict_derivatives(X, V=None, order=2)` ->
  `(y, J v, v^T H v)`, higher orders `None`; `jacobian(x)`, `hessian(x, k)`;
  `backward(X, dY)` -> `(dparams, dx)`; `backward_derivatives(X, V, dY,
  dY1=None, dY2=None)` -> `(dparams, dx, dv)`; `get_parameters()` /
  `set_parameters(p)` (copy on write between copies of a network).
- Deploy: `QuantizedNeuralNet(net)`, `predict(x, n_rows)`,
  `get_weight_bytes()`.

What of tttrlib's `NeuralNet` was **not** ported: `from_json_string/file`,
`to_json_string/file` (msgpack is bff's one format: `NeuralNet(bytes)`,
`to_msgpack()`); `train`/`train_np`/`TrainOptions`/loss curves (bff already
has `train_neural_net_with_history`, `NeuralNetTrainOptions`,
`NeuralNetTraining`); `get_layers()`/`DenseLayer`/`get_x_scaler()` and the
`MlpCore.h` SWIG exposure (the kernels stay C++-only; the layer accessors
above and the document cover introspection); the `_np`/`_out` twin methods
and the `parameters` property (bff returns managed views from the one method,
no property sugar); `predict_batch` (bff's `predict(x, n_rows)` is the batch).

## Validation

- `test/test_neural_net_derivatives.py`: tttrlib's test_neural_net.py ported
  (FD checks of `backward`, `backward_derivatives`, `jacobian`, `hessian`,
  Taylor orders, a Poisson PINN, sklearn forward parity 1e-10), the fixtures
  in `test/input/nn` (PyTorch's outputs: float32 ONNX 1e-6, double 1e-12;
  safetensors 1e-12), the C++ core test, MatGemm parity, the int8 bounds.
- `test/cpp_snippets/test_mlp_core.cpp`: tttrlib's test_mlp_core.cpp ported
  (derivatives two ways: central differences and the Dual dot-product
  identity), plus MatGemm vs PortableGemm (forward orders 0-2 and backward,
  <= 1e-12 relative) and the int8 path; prints timings, asserts none.
- `test/medium_test_neural_net_sklearn.py`: training head to head with
  sklearn (tttrlib's A/B section 7).
- `test/expensive_test_neural_net_examples.py`: the three
  `examples/neural_net/` gallery scripts and their reported numbers.
- `test/bench_neural_net.py`: tttrlib's bench_nn.py (training vs sklearn,
  forward/derivative/int8 timings, surrogate vs EM).

Measured 2026-09-24 on the M-series dev box, this build **without OpenMP**
(so Mat.h's row-tile threading is off), standalone `-O2`, 3x64 tanh net:
MatGemm 2.35x the portable path at batch 256 and 2.4x at 512 (forward and
backward alike); int8 2.1x the double *portable* path at batch 256, but not
faster than the MatGemm double path the module actually uses (0.9x through
the Python binding). int8 error on the fixtures and a trained 3x64 net is
below 1 % of the output absmax; an untrained net with O(1) weights reaches
~2.4 %. `bench_neural_net.py` on the surrogate's 24-256-256-128-8 net, batch
500: int8 2.4 ms against 3.0 ms for MatGemm double (1.27x), 0.7 % error,
weights 844 kB -> 105 kB; bff training 0.46 s against sklearn's 2.2 s, same
held-out MAE (0.0997 / 0.1006).

## The design record of the lost tttrlib work

Two pieces of tttrlib MLP work were never committed and their files were
deleted; what survives is the description in a dangling stash (`04ccb86a`,
"nn-overlap", tttrlib `modules/math/README.md` and CHANGELOG). Kept here
verbatim so the record is not lost:

> **`MlpGemm.h`**: The `MatGemm` GEMM policy for `MlpCore.h`'s template seam —
> routes the batch products through `Mat.h`'s packed, register-blocked
> kernels, now threaded over row tiles (`gemm_nn_core_mt`: the
> pre-tile-history OpenMP path bypassed the micro-kernel for a per-row SAXPY,
> so `gemm_nt`/`gemm_tn` — the MLP hot path — ran on one core of an 8-core
> machine). Measured on wall clock, OpenMP builds, median over three net
> draws: 2.7× the portable path at batch 512, parity at batch 256 (the
> portable loops get `omp simd` too), and 1.7–2.6× behind Eigen — recorded
> honestly; closing that gap means a true blocked GEMM, a project of its own,
> not an adapter — a K×2 unroll of the micro-kernel was tried and measured at
> noise level (~2 %), kept but not credited as a win. Batch-1 shapes stay
> serial by threshold (packing overhead dominates a single row).

> **`MlpQuant.h`**: Dynamic-range int8 inference for the network — weights
> quantised once (per-tensor symmetric), activations requantised between
> layers, int32 accumulation. ~2× inference against the double portable path
> at batch 256 (3×64 net, measured), 8× smaller weights; error bounded by the
> activation absmax (≤ 2% on the parity fixtures), so it is a deployment
> path, not a training or derivative path. Concepts from ermig1979/Simd's
> quantised inner-product scheme, re-expressed std-only.

> **Standard formats in**: `model_from_onnx` reads the MLP subset of ONNX
> (Gemm / MatMul+Add, Relu/Tanh/Sigmoid/Softplus/Sin, the SiLU and
> softplus-threshold patterns, pass-through reshapes — what PyTorch's two
> exporters, Keras, JAX and skl2onnx emit) through a minimal protobuf
> wire-format reader, so no ONNX or protobuf library; `model_from_safetensors`
> reads a PyTorch `state_dict` (weights only, activations from `__metadata__`
> or an argument). Checked against PyTorch's own outputs on committed fixtures
> and live when PyTorch is installed. So the network need not be trained here
> at all: train anywhere, export, load.

> **A whole model, and its file format**: `MlpModel` = layers + input/output
> `StandardScaler`s; `model_predict` / `model_backward` apply the scalers and
> their chain rule so a consumer stays in physical units; `model_from_json` /
> `model_to_json` are templated on the JSON type (any nlohmann-compatible
> object) so the header stays std-only while both repositories deserialise
> the `tttrlib.neural_net` document with the nlohmann copy they vendor.

How bff realised them: `MlpGemm.h` is the `MatGemm` struct tttrlib's
`NeuralNet.cpp` carried (the one piece of code that survived), as a bff-owned
header. `MlpQuant.h` is a **reconstruction from the spec above, not the
original code**: per-tensor symmetric int8 weights, per-row dynamic int8
activations, int32 accumulation, scalers and activations in double. The
"Eigen" and "K×2 unroll" measurements were not repeated. The JSON document is
now msgpack (`NetworkDocument.h`); the tree mapping is unchanged.
