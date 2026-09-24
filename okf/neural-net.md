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
| FP4 formats | `include/internal/MlpFp4.h` | E2M1 / E4M3 / E8M0 codecs, the `fp4` / `mxfp4` / `nvfp4` block recipes, 2-D (16 x 16) weight scaling, stochastic rounding, the float reference forward pass (tests only). |
| FP4 kernels | `include/internal/MlpFp4Kernels.h` | Integer-SIMD FP4 x int8 and FP4 x FP4 dot products and GEMMs (after llama.cpp/ggml, MIT), NEON + dotprod / NEON / AVX2 / generic, and the fast forward pass. |
| FP4 training | `include/internal/MlpFp4Train.h` | NVIDIA's NVFP4 pretraining recipe (and MXFP4) on those kernels. |
| Documents | `include/internal/NetworkDocument.h` | The one msgpack encoder/decoder (`bff.quantized_neural_net` through its own small msgpack reader/writer, because the vendored nlohmann 3.6.1 has no `bin`). |
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
- Deploy: `QuantizedNeuralNet(net, format="int8", quantize_activations=False)`
  with `format` in `int8 | fp4 | mxfp4 | nvfp4`; `predict(x, n_rows)`,
  `get_format()`, `get_quantize_activations()`, `get_n_weights()`,
  `get_weight_bytes()` (scales included), `get_bits_per_weight()`,
  `to_msgpack()` / `QuantizedNeuralNet.from_msgpack(bytes)`
  (`bff.quantized_neural_net`, bit-exact), static `get_kernel_name()`.
- Train in FP4: `NeuralNetTrainOptions.precision = "nvfp4" | "mxfp4"`
  (default `"float64"`), with `fp4_keep_first_layer`, `fp4_keep_last_layer`,
  `fp4_hadamard`, `fp4_stochastic_rounding`; the result's
  `get_quantized_network()` is the trained FP4 network,
  `get_network()` its float64 master weights.

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

## FP4: formats, kernels, training (2026-09-24)

**CPUs have no FP4 arithmetic unit.** "Native" FP4 here means: the weights
(and, in W4A4 and in training, the activations and gradients) are held as
packed 4-bit E2M1 codes with their block scales, and every GEMM runs on the
codes through integer SIMD dot products -- a 16-byte table lookup turns 16
nibbles into the int8 values `2 x E2M1`, `vdotq_s32` / `maddubs` multiplies
them against int8 or other looked-up FP4 values, and the block scales are
applied afterwards in float. Nothing is dequantised to float first. The
dequantise-to-double path survives only as `predict_reference` in
`MlpFp4.h`, which the tests pin the kernels against.

### Formats (exact recipes in `MlpFp4.h`)

- **E2M1** element: 1 sign, 2 exponent bits (bias 1), 1 mantissa bit;
  magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. Round to nearest, ties to even
  (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4),
  saturating to +-6. Two codes a byte, **element 2i in the low nibble** (ONNX
  FLOAT4E2M1, PyTorch `float4_e2m1fn_x2`/`pack_uint4`). Rows padded with zero
  codes to a multiple of 32 elements.
- **E4M3** (`e4m3fn`: bias 7, no inf, NaN = S.1111.111, max 448), RNE,
  saturating; **E8M0** = 2^(code - 127), 255 NaN.
- `fp4`: one float32 scale a row, `absmax / 6`.
- `mxfp4` (OCP MX v1.0): blocks of 32, E8M0 scale
  `2^(floor(log2 absmax) - 2)` (emax of E2M1 = 2), elements clamped to +-6.
  PyTorch/torchao's default is "RCEIL" `2^ceil(log2(absmax / 6))`; the two
  agree when `frac(log2 absmax) <= log2 1.5` and RCEIL is twice the OCP scale
  otherwise (on the trained 3x64 net: 67 blocks equal, 255 twice).
- `nvfp4` (NVIDIA, Model Optimizer / arXiv 2509.25149 appendix B): blocks of
  16, `g = float32(absmax_tensor / (6 * 448))`, block scale
  `e4m3(absmax_block / (6 g))` (zero block -> e4m3(1.0)), element
  `e2m1(w / (S g))`.
- Bits a weight (scales included): mxfp4 4.25, nvfp4 4.5 (+32 a tensor), fp4
  `4 + 32 / n_in`; a layer with few inputs pays for the padding to 32 (the
  2-input first layer of the test nets).

Cross-checked against PyTorch 2.13 (`float8_e4m3fn`, `float8_e8m0fnu`,
`float4_e2m1fn_x2`, and torchao's E2M1 encoder and `pack_uint4` from
`torch/testing/_internal/common_quantized.py`, extracted by `ast` because the
module needs test-only packages): 0 of 10 304 E2M1 codes differ for each of
fp4/mxfp4/nvfp4 given bff's scales, packing byte-identical, every NVFP4 block
scale equals PyTorch's `float8_e4m3fn` cast, the E4M3 codec agrees on a
synthetic net sweeping the scale range, and bff's MXFP4 exponents equal the
OCP floor rule and relate to torch's RCEIL `to_mxfp` as stated. torchao
itself is not installed; PyTorch cannot cast to `float4_e2m1fn_x2`
("copy_kernel not implemented"), so the encoder is torchao's Python one.

### Kernels (`MlpFp4Kernels.h`)

Ported from llama.cpp/ggml (MIT; notice in the header):
`ggml_vec_dot_mxfp4_q8_0` / `ggml_vec_dot_nvfp4_q8_0` (`ggml-cpu/quants.c`,
`arch/arm/quants.c`, `arch/x86/quants.c`), `kvalues_mxfp4` (`ggml-common.h`).
ggml packs element j with j + 16 in a byte; bff keeps the standard pair
order and permutes the int8 activations within each 32-group instead, so
the weights stay ONNX/PyTorch-compatible. (Upstream llama.cpp now also has a
NEON NVFP4 kernel; bff's NVFP4 path is the same group kernel with 16-element
scales.)

- W4A8 (default): activations quantised per block to int8 (32; 16 for nvfp4),
  `d = amax / 127`, round half away from zero. W4A4 (`quantize_activations`):
  activations quantised to FP4 in the same format per row (nvfp4 with a
  per-row global scale), FP4 x FP4 kernel.
- Variants at **compile time**: `__ARM_FEATURE_DOTPROD` (neon-dotprod),
  `__ARM_NEON` (neon: `vmull_s8`/`vmlal_s8`/`vpaddlq`), `__AVX2__` (avx2:
  pshufb + `_mm256_sign_epi8` + `maddubs` + `madd`), else generic;
  `-DIMPBFF_FP4_NO_SIMD` forces generic. No run-time cpuid: under Rosetta 2
  `__builtin_cpu_supports("avx2")` is 0 while AVX2 executes. No AVX-512 VNNI
  variant. The float scaling is one shared lane-structured routine without
  contracted multiply-adds, so **every variant is bit-identical to the
  generic kernel** (checked: integer sums, `gemm_q8`, `gemm_fp4`).
- `test/test_neural_net_fp4.py` compiles `cpp_snippets/test_fp4_kernels.cpp`
  four ways and runs each: `-march=armv8.2-a+dotprod`, `-march=armv8-a`,
  `-arch x86_64 -mavx2 -mfma` under `arch -x86_64` (Rosetta 2; skipped where
  unavailable, native on x86), and `-DIMPBFF_FP4_NO_SIMD` -- all 0 failures
  on 2026-09-24. The IMP module build on this Mac compiles `neon-dotprod`
  (AppleClang's arm64 target has dotprod;
  `QuantizedNeuralNet.get_kernel_name()`). The standalone/wheel build adds
  opt-in `IMPBFF_WITH_AVX2` (x86-64: `-mavx2 -mfma`) and
  `IMPBFF_WITH_ARM_DOTPROD` (AArch64 Linux), both OFF so a wheel runs on
  every CPU of its architecture.

Speed, M-series dev box (loaded; no OpenMP in the IMP build), best of 20:

| 256 x 256 x 256 GEMM | us |
|---|---|
| double MatGemm `nt` | 1090 |
| FP4 x int8 `gemm_q8` (neon-dotprod) | 520 |
| FP4 x FP4 `gemm_fp4` (neon-dotprod) | 578 |
| FP4 x int8, generic scalar | 32 200 |
| int8-quantise the 256 x 256 activations | 215 |

Whole forward pass, 24-256-256-128-8 tanh net (the HMM surrogate's shape),
us per call (C++ snippet; "dequant" is the first, dequantise-to-double
version):

| variant | batch | double MatGemm | nvfp4 dequant | nvfp4 W4A8 | nvfp4 W4A4 | fp4 W4A8 | mxfp4 W4A8 |
|---|---|---|---|---|---|---|---|
| neon-dotprod | 1 | 28.4 | 428 | 21.0 | 24.4 | 11.8 | 21.1 |
| neon-dotprod | 32 | 410 | 812 | 328 | 432 | 329 | 337 |
| neon-dotprod | 256 | 2901 | 3307 | 2568 | 3395 | 2642 | 2643 |
| neon (no dotprod) | 256 | 2906 | 3313 | 3306 | 4104 | 3369 | 3383 |
| generic | 256 | 2921 | 3267 | 53 308 | 56 215 | 54 257 | 53 769 |
| avx2 under Rosetta 2 | 256 | 7948 | 8813 | 12 648 | 14 064 | 12 598 | 12 580 |

The AVX2 row is Rosetta's translation of AVX2 -- a **correctness** run, not
representative of x86 speed. At batch 256 the GEMMs are ~2x faster than
double but the pass is dominated by the 164 k `tanh` evaluations and the
per-call activation quantisation, so the whole pass gains only 1.1x (W4A8);
at batch 1 FP4 is 1.4-2.4x faster than double. Through the Python binding
(`test_speed_report_256_256_128`), batch 256: double 3032 us, int8 2441,
nvfp4 W4A8 2722, W4A4 3513; batch 1: double 32, int8 11, fp4 14, nvfp4 24.

### Accuracy and size (2026-09-24)

Max error relative to the output absmax, W4A8 / W4A4:

| net | fp4 | mxfp4 | nvfp4 | int8 |
|---|---|---|---|---|
| fixture nets (tanh; 4 files, one net) | 0.093 / 0.31 | 0.175 / 0.30 | 0.089 / 0.31 | < 0.02 |
| `mlp_relu_nometa.safetensors` | 0.020 / 0.11 | 0.182 / 0.26 | 0.018 / 0.096 | |
| trained 2-64-64-64-1 tanh | 0.040 / 0.22 | 0.134 / 0.31 | 0.080 / 0.24 | 0.009 |

Weight relative RMS error on the trained net: fp4 0.110, mxfp4 0.109,
nvfp4 0.101 -- nvfp4's finer blocks reconstruct the weights best, but its
E4M3 block scales (3 mantissa bits) do not always translate into a smaller
output error than a per-row float32 scale; mxfp4 is consistently worst
(power-of-two scales, the OCP floor rule saturating [6, 8) X). So "nvfp4 <=
fp4" holds for the weights and on the fixtures, not on every net's outputs.
W4A4 costs 2-4x the W4A8 error: FP4 activations are coarse, which is why
training keeps the first layer (physical inputs) in float64. Sizes on the
trained net (8384 weights, 67 kB as float64): int8 8.4 kB (8.0x), fp4
5.9 kB (11.3x), mxfp4 5.5 kB (12.3x), nvfp4 5.8 kB (11.5x) -- the padded
2-input layer; on 256-wide layers mxfp4 is 4.25 and nvfp4 4.50 bits a
weight, 15.1x and 14.2x under float64.

### FP4 training (`MlpFp4Train.h`, `precision = "nvfp4" | "mxfp4"`)

Follows arXiv 2509.25149 ("Pretraining Large Language Models with NVFP4"):
fprop, dgrad and wgrad of every FP4 layer on the FP4 x FP4 kernel (4.1);
float64 master weights and Adam state; weights 2-D 16 x 16 scaled so W and
W^T quantise identically (4.3; `quantize_2d` / `transpose_2d`); a 16 x 16
random Hadamard transform with one fixed sign vector on the wgrad inputs
only (4.2); stochastic rounding for the gradient operand of dgrad and wgrad,
RNE for weights and activations (4.4), SR draws seeded by `options.seed`;
first and last layer kept in float64 by default (4.1: "a few sensitive
linear layers in higher precision ... majority at the end"). Deviations: the
global NVFP4 scale of activation/gradient operands is per operand row, not
per tensor (batch-independent, and inference W4A4 reproduces training's
fprop exactly -- checked bit-identical); accumulation in float, loss and
bias in double. MXFP4 training: E8M0 blocks of 32, 32 x 32 weight tiles,
no global scale, same Hadamard (16) and SR.

The result's `get_quantized_network()` is a `bff.quantized_neural_net`
(W4A4) whose FP4 layers hold exactly the 2-D scaled weights fprop used and
whose kept layers are float64 (`"precision": "float64"` layer entries).

Checked (C++): Hadamard orthogonal to 1e-16 and `(T dZ)^T (T A) = dZ^T A`;
wgrad and dgrad equal a float64 GEMM of the same quantised operands to
7e-8; SR deterministic per seed and unbiased (mean of 2e5 roundings within
3e-3); one FP4 step bit-reproducible; FP4 gradient cosine 0.98 to the
float64 one on a random 6-64-48-32-2 net. Measured (Python):

| task | float64 | nvfp4 | mxfp4 |
|---|---|---|---|
| 4-64-64-64-2 tanh regression, held-out MSE (var 0.51) | 8.4e-4 | 2.6e-3 (3.1x) | 3.0e-3 (3.5x) |
| HMM surrogate set (400 x 24 -> 8), 256-256-128, held-out MAE | 0.0959 | 0.1046 (1.09x) | 0.1036 (1.08x) |
| seconds an epoch (surrogate) | 0.011 | 0.020 | 0.017 |

FP4 training is slower than float64 here: the batches (200) are small, and
every step re-quantises weights (2-D, plus the transpose), activations,
gradients and the Hadamard-transformed wgrad operands on one thread. The
point of the path is fidelity to the hardware recipe and a network that
deploys in FP4, not CPU training speed.
