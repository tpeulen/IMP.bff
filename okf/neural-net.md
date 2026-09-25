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
| Elementwise math | `include/internal/MlpMath.h` | Vectorised `tanh`, `exp`, sigmoid, SiLU (NEON / SSE2 / AVX2 / AVX-512 / generic, bit-identical), used by every path (below, "Fast tanh"). |
| Kernels | `include/internal/MlpCore.h` | std-only: activations and their derivatives to order 3, the Taylor-augmented forward pass (`y`, `J v`, `v^T H v`), its exact adjoint, `flatten`/`unflatten`, `MlpModel` with scalers, document tree <-> model, ONNX and safetensors readers. The GEMM is a template policy (`PortableGemm` default). |
| GEMM policy | `include/internal/MlpGemm.h` | `MatGemm`: MlpCore's `nn`/`nt`/`tn` through the vendored `internal/Mat.h`'s packed, register-blocked kernels (threaded over row tiles with OpenMP). Used by `NeuralNet` and `train_neural_net`. |
| int8 path | `include/internal/MlpQuant.h` | Dynamic-range int8 inference, **rebuilt from the lost tttrlib spec** (below). |
| FP4 formats | `include/internal/MlpFp4.h` | E2M1 / E4M3 / E8M0 codecs, the `fp4` / `mxfp4` / `nvfp4` block recipes, 2-D (16 x 16) weight scaling, stochastic rounding, the float reference forward pass (tests only). |
| FP4 kernels | `include/internal/MlpFp4Kernels.h` | Integer-SIMD FP4 x int8 and FP4 x FP4 dot products and GEMMs (after llama.cpp/ggml, MIT), register-blocked micro-kernels for NEON + dotprod / NEON / AVX2 / AVX-512 VNNI / generic, the vectorised quantisers, and the fast forward pass. |
| FP4 training | `include/internal/MlpFp4Train.h` | NVIDIA's NVFP4 pretraining recipe (and MXFP4) on those kernels. |
| Ternary | `include/internal/MlpTernary.h`, `MlpTernaryTrain.h` | BitNet b1.58 (W1.58A8): absmean ternary weights, int8 absmax activations, 2-bit (TQ2_0-style) and base-3 (TQ1_0-style) packings, ternary x int8 SIMD kernels on the FP4 kernels' variants, and BitNet's quantisation-aware training (below, "Ternary"). |
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
  with `format` in `int8 | fp4 | mxfp4 | nvfp4 | ternary | ternary_row |
  ternary_tq1 | ternary_tq1_row`; `predict(x, n_rows)`,
  `get_format()`, `get_quantize_activations()`, `get_n_weights()`,
  `get_weight_bytes()` (scales included), `get_bits_per_weight()`,
  `to_msgpack()` / `QuantizedNeuralNet.from_msgpack(bytes)`
  (`bff.quantized_neural_net`, bit-exact), static `get_kernel_name()`.
- Train in FP4: `NeuralNetTrainOptions.precision = "nvfp4" | "mxfp4"`
  (default `"float64"`), with `fp4_keep_first_layer`, `fp4_keep_last_layer`,
  `fp4_hadamard`, `fp4_stochastic_rounding`; the result's
  `get_quantized_network()` is the trained FP4 network,
  `get_network()` its float64 master weights.
- Train ternary: `precision = "ternary"` (BitNet b1.58 QAT), with
  `ternary_keep_first_layer`, `ternary_keep_last_layer` (both default true);
  `get_quantized_network()` is the `"ternary"` document.

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
`arch/arm/quants.c`, `arch/x86/quants.c`), `kvalues_mxfp4` (`ggml-common.h`):
the 16-byte table lookup of `2 x E2M1` and integer dot products, scales
applied after in float. Since the second pass (below) the dot products run
as a register-blocked GEMM rather than row-by-row.

- W4A8 (default): activations quantised per block to int8 (32; 16 for nvfp4),
  `d = amax / 127`, round half away from zero. W4A4 (`quantize_activations`):
  activations quantised to FP4 in the same format per row (nvfp4 with a
  per-row global scale); the kernel takes their codes' values `2 x E2M1`
  (exact int8 in [-12, 12]).
- Variants at **compile time**: `__ARM_FEATURE_DOTPROD` (neon-dotprod),
  `__ARM_NEON` (neon), `__AVX512F__ && __AVX512BW__ && __AVX512VNNI__`
  (avx512-vnni), `__AVX2__` (avx2), else generic; `-DIMPBFF_FP4_NO_SIMD`
  forces generic. No run-time cpuid: under Rosetta 2
  `__builtin_cpu_supports("avx2")` is 0 while AVX2 executes. **Every
  variant is bit-identical to the generic kernel, under any compiler
  flags** (see the design below; before, g++ with `-mfma` contracted the
  float combine into FMAs and an AVX2 build differed from a generic one --
  measured on cordeshub, 52 of 163 golden lines).
- `test/test_neural_net_fp4.py` compiles `cpp_snippets/test_fp4_kernels.cpp`
  with `-march=armv8.2-a+dotprod`, `-march=armv8-a`, `-arch x86_64 -mavx2
  -mfma` under `arch -x86_64` (Rosetta 2; native on x86),
  `-DIMPBFF_FP4_NO_SIMD`, and on an x86 CPU with VNNI `-mavx512f
  -mavx512bw -mavx512vl -mavx512vnni`. CI (`fp4_kernels_x86` in
  `.github/workflows/ci.yml`) builds generic / AVX2 / AVX-512 on
  ubuntu-latest and runs the AVX-512 one natively when the runner has
  `avx512_vnni`, else under Intel SDE (`sde64 -spr`). The IMP module build
  on this Mac compiles `neon-dotprod`
  (`QuantizedNeuralNet.get_kernel_name()`). The standalone/wheel build adds
  opt-in `IMPBFF_WITH_AVX2` (x86-64: `-mavx2 -mfma`) and
  `IMPBFF_WITH_ARM_DOTPROD` (AArch64 Linux), both OFF so a wheel runs on
  every CPU of its architecture.
- **AVX-512 status**: validated on real hardware (cordeshub, Xeon Silver
  4416+, Sapphire Rapids; g++ 11.4): the snippet built
  `-mavx512f -mavx512bw -mavx512vl -mavx512vnni` and `-march=sapphirerapids`
  reports `avx512-vnni`, 0 failures, and a golden run (quantisers, GEMMs,
  predict on four nets, four FP4 training steps on three nets x two
  formats x three recipe configurations) equal to the generic build's.
  Not runnable on the arm64 Mac (Rosetta has no AVX-512); there it is only
  compiled (`clang++ -arch x86_64 -mavx512f -mavx512bw -mavx512vnni -c`).

### Performance design (second pass, 2026-09-24)

Profiled first (C++ harness, `sample`): the old GEMM re-did the table lookup
of every weight for every activation row (row x 4 rows dot products), and
around it the quantisation was scalar with `ilogb`/`ldexp`/`fmod` per
block scale, allocations per call, and the Hadamard a scalar 16 x 16
product per tile. What changed, each step checked bit-identical against the
old headers (golden hashes on the Mac's four variants and cordeshub's four,
and the Python-trained networks' msgpack bytes):

- **Micro-kernel**: the right operand (weights; wgrad's transformed
  activations) is repacked once into tiles of `kNR` rows: per 16-wide
  sub-block, two chunks whose low/high nibbles are k-quads `2p`/`2p + 1`
  of every tile row. One lookup of a chunk is the operand of a lane-wise
  4-byte dot: `vdotq_laneq_s32` (NEON dotprod, 4 x 4 tile; plain NEON
  emulates it with `vmull_s8` + `vpaddlq_s16`), `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16` (AVX2, 2 x 8), `_mm512_dpbusd_epi32` (AVX-512 VNNI,
  4 x 16). The x86 u8 x s8 instructions get the left operand offset by 128
  (`a ^ 0x80`) and subtract `128 x` the row's sub-block sum, precomputed at
  packing (ggml's unsigned/signed handling; the sums stay exact). The float
  part is `combine()` per output lane: `acc[s % 4] += float(sum) *
  (sa * sw)`, then `(a0 + a1) + (a2 + a3)`; the product goes through an
  empty asm (`IMPBFF_FP4_KEEP`) so no compiler fuses it into an FMA.
  QuantizedNeuralNet packs its layers once (`kern::prepare`), predict keeps
  thread-local buffers, and the bias is added in the GEMM's store.
- **Quantisers**: absmax, the E2M1 threshold network, the int8 rounding and
  stochastic rounding run on 2 / 4 / 8 doubles with the scalar reference's
  exact IEEE operations (`std::min`/`std::max` NaN rules via compare-select
  or x86 `min_pd` operand order). On NEON the quotient is narrowed to float32
  with **round-to-odd** (`FCVTXN`): a double and its round-to-odd float
  compare alike with every float of even last mantissa bit, which all E2M1
  thresholds are, so the codes stay exact on 4 lanes. Power-of-two scales
  (every mxfp4 block) multiply by the exact reciprocal instead of dividing
  (the same exact product, rounded once). E4M3 encode has a bit-level fast
  path (`e4m3_encode_reference` is the old code; 4e5 random values and every
  midpoint compared in the snippet); scale tables are hoisted; block scales
  are computed in a separate pass so the core overlaps them; the nvfp4 row
  scale comes from the block absmaxes.
- **Stochastic rounding kept its stream** in this pass (SplitMix64, draws
  in order); the third pass below replaced it.
- **Training**: `quantize_2d_packed` quantises each FP4 layer's weights once
  a step straight into both packed operands, W (fprop) and W^T (dgrad), with
  the 2-D tile scales -- no `transpose_2d` re-pack; the snippet checks it
  against `quantize_2d` / `transpose_2d`. The Hadamard transform is fused
  with the transpose (`rht_transpose`), register-blocked over 8 rows of T x 4
  (8 on AVX-512) columns with FMAs (`t x` is exact, `t = +-1/4`, so the FMA
  rounds as `acc + t x` does; the previous clang arm64 build contracted it
  too); all buffers live in the workspace.
- Unchanged by choice: `tanh` (shared by both paths, libm's; replaced on
  2026-09-25, "Fast tanh" below), MatGemm (the
  float64 baseline; AVX2 on x86 even with AVX-512), threading (none in the
  IMP build; the FP4 GEMM has the same OpenMP opportunity as MatGemm).

### Training, third pass (2026-09-25): counter-based SR, fast operand quantiser

**Every FP4-trained network changed** (by decision: "switch"); inference
(W4A8 / W4A4 predict, QuantizedNeuralNet, fprop's activations and the 2-D
weights) is bit-identical to before -- checked on the Python golden hashes
of every predict variant and by the snippet comparing `quantize_left` with
the MlpFp4.h recipe.

- **Generator** (`SrKey`, `MlpFp4.h`): counter-based, no state carried from
  draw to draw. A key `(a, b)` (two 32-bit words) is `fmix64` (SplitMix64's
  output function) chained over (seed, step, layer, operand) -- operand 0
  is dgrad's dZ, 1 is wgrad's `(T dZ)^T`; `train::SrStream` advances the
  step once a backward pass. Word n is `mix32(mix32(n ^ a) ^ b)`, `mix32`
  a bijective 2-multiply xorshift-multiply mix (Wellons' lowbias32
  constants `0x21F0AAAD`, `0x735A2D97`): two keyed rounds, an
  Even-Mansour-style keyed permutation of the counter, so two operands'
  streams are unrelated permutations, not shifted copies of one sequence.
  Each element takes **16 bits**: element j of a 16-element group g uses
  word `8 g + (j & 7)`, low half for j < 8, high half otherwise, the
  element index being `row * padded_cols + k` -- a 4 / 8 / 16-lane vector
  of 32-bit words serves whole lanes of elements, and all lanes are 32-bit
  multiplies (`vmulq_u32`, `_mm256_mullo_epi32`), which NEON has and
  SplitMix64's 64-bit ones lack. Why 16 bits suffice: SR only needs
  `P(up) = fraction of the step`; with a centred 16-bit draw that holds to
  2^-17 of a step for every value (asserted exactly over all 65536 draws),
  far below E2M1's own step. Tested: that exact bound; the mean of 2^16 and
  2^22 roundings of six values within 4 sigma (+2^-17) of x (worst 0.44 /
  0.53 of the bound); chi-square of the top byte over 2^20 draws (248, 255
  dof); lag-1 correlation 2.6e-3 (bound 4.9e-3); other operand / step keys
  give other words. Scratch checks (not committed) over 8 keys and 2^22
  draws: lag-1, same-word low/high halves and lag-16 correlations all
  within 1.7 sigma.
- **Element rule for training-only operands** (dgrad's and wgrad's
  gradients, wgrad's transformed activations; `kern::quantize_train`):
  block scales exactly as before (same codes), elements
  `q = float(x * (1 / d))` -- one division a block instead of one an
  element, the value narrowed to float -- then round-to-nearest-even on
  the float (`e2m1_encode_f`) or SR on float bits (`e2m1_encode_sr16`):
  `y = min(|q| + 2, (|q| + 2) / 2 + 3)` puts the E2M1 grid on 2-mantissa-
  bit boundaries ([0, 2) -> [2, 4), [2, 4) -> [4, 6), [4, 6] -> [6, 7]),
  the draw `16 + (u << 5)` is added below the kept bits, and the code is
  `(bits >> 21) - 512` saturated to 7. NaN codes as +-0. On NEON the carry
  and the RNE threshold compares run in 16-bit lanes (8 at a time).
  fprop's activations and the weights keep the exact rule (inference must
  reproduce training's forward), now through the same row quantiser
  (`quantize_rows_fast`, a 16-element group at a time, division on vector
  lanes and round-to-odd narrowing as before).
- **Hadamard as butterflies**: `x_j s_j / 4` (exact) then four add/sub
  stages, 2 (NEON) / 4 (AVX2) / 8 (generic) columns in registers, fully
  unrolled -- 2x faster than the second pass's 16 x 16 FMA product on the
  Mac, 3.4x under Rosetta AVX2. Adds only, so every build rounds alike
  (`rht_rows` is the same code).
- **Block scales**: an unrolled 16-element absmax; nvfp4's E4M3 codes,
  decoded scales and half scales two blocks a NEON op (the bit-level
  rounding of `e4m3_encode`, the value rebuilt from the code bits), the
  scalar code outside the normal range; 2-D weight tile scales hoisted;
  `e4m3_encode_reference` is out of line (cold) so the fast path inlines.
- **GEMM**: the micro-kernel tiles are always inlined into `gemm_packed`
  (K = 64 GEMMs 2x faster: the call per 4 x 4 tile dominated) and the
  store/bias loop has constant trip counts; right-operand packing is SWAR.
- **Not reused across operands**: dZ goes into dgrad row-wise and into
  wgrad transposed and Hadamard-transformed; A into fprop row-wise and
  into wgrad transformed -- different values, so nothing is quantised once
  for two GEMMs with the recipe's Hadamard on.
- Bit-identical across variants is now also one committed number:
  `kTrainingFingerprint` in the snippet hashes the training quantiser (SR
  and RNE, both formats) and three steps of an all-FP4 ReLU net (no libm,
  no float64 GEMM); generic, neon, neon-dotprod, AVX2 (Rosetta) and the
  x86 generic build on the Mac, and generic / AVX2 / AVX-512 VNNI /
  `-march=sapphirerapids` (g++ 11.4) on cordeshub all give
  `06b331c979c13fc5`. Since the fast tanh (below) the fingerprint also runs
  three steps of an all-FP4 **tanh** net and is `64407c22cd039ecf` on the
  same nine builds.

### Speed gates (2026-09-24)

`test/bench_neural_net_fp4.py` (through the module, median of 5 runs;
single-threaded both paths). bff has **no float32 network path**, so float64
is the only float baseline. The x86 numbers are from an equivalent C++
harness (the gate code paths without Python; median of 9) on cordeshub,
**under a load average of ~18-20 on 24 cores from another user** -- expect
noise of 10-20 %. The Mac was also loaded (load average 8-13 on 8 cores).

G1, inference, us a call, 24-256-256-128-8 (the -15 net, the 3-state
surrogate, is within a few % of it; full tables in the script output):

| machine / variant | batch | float64 | fp4 W4A8 | mxfp4 W4A8 | nvfp4 W4A8 | nvfp4 W4A4 | before: nvfp4 W4A8 / W4A4 |
|---|---|---|---|---|---|---|---|
| M1 Pro, neon-dotprod (Python) | 1 | 31.7 | 10.2 | 10.3 | 10.0 | 10.5 | 24 / 24 (Python) |
| M1 Pro, neon-dotprod (Python) | 32 | 460 | 276 | 276 | 275 | 283 | |
| M1 Pro, neon-dotprod (Python) | 256 | 3210 | 2194 | 2189 | 2193 | 2273 | 2722 / 3513 (Python) |
| M1 Pro, neon-dotprod (C++) | 1 | 28.4 | 7.8 | 7.9 | 7.8 | 8.2 | 21.7 / 24.6 |
| M1 Pro, neon-dotprod (C++) | 32 | 417 | 241 | 236 | 238 | 246 | 336 / 432 |
| M1 Pro, neon-dotprod (C++) | 256 | 2970 | 1906 | 1916 | 1888 | 1983 | 2630 / 3411 |
| Xeon 4416+, avx2 (C++) | 1 | 28.2 | 13.0 | 12.8 | 13.5 | 13.9 | 41.2 / 45.4 |
| Xeon 4416+, avx2 (C++) | 32 | 977 | 625 | 619 | 628 | 655 | 896 / 999 |
| Xeon 4416+, avx2 (C++) | 256 | 5865 | 4919 | 4943 | 5041 | 5324 | 7379 / 8500 |
| Xeon 4416+, avx512-vnni (C++) | 1 | 27.5 | 12.5 | 12.7 | 12.4 | 12.9 | (no variant) |
| Xeon 4416+, avx512-vnni (C++) | 32 | 990 | 586 | 582 | 586 | 607 | |
| Xeon 4416+, avx512-vnni (C++) | 256 | 5984 | 4641 | 4642 | 4646 | 4910 | |

Every FP4 cell is below float64 (worst FP4/float64: 0.71 Mac, 0.91 x86).
At batch 256 both paths are dominated by the 164 k `tanh` calls (glibc's
cost ~20 ns each on the Xeon, 3.3 ms of the 5.9); the FP4 GEMMs themselves:
256^3 `gemm_packed` 320 us on the Mac (old kernel 520, MatGemm 1070).
"Before" is the old kernels (the first FP4 commit) in the same harness;
before, x86 FP4 at batch 256 was 1.1-1.3x *slower* than float64.

G2, training, ms an epoch (Python, batch 200, every recipe feature on:
2-D weight scales, Hadamard on wgrad, stochastic rounding, first/last layer
float64; 40 / 20 / 20 epochs, early stopping patience disabled):

| shape | float64 | nvfp4 | mxfp4 | nvfp4 / f64 | mxfp4 / f64 | before (nvfp4 / mxfp4, s an epoch) |
|---|---|---|---|---|---|---|
| surrogate 400 x 24 -> 8, 256-256-128 | 11.03 | 8.22 | 8.12 | **0.75** | **0.74** | 0.020 / 0.017 vs 0.011 |
| regression 3000 x 4 -> 2, 64-64-64 | 12.52 | 13.85 | 12.81 | 1.11 | 1.02 | |
| tiny 2000 x 2 -> 1, 16-16 (no gate) | 0.81 | 1.25 | 1.14 | 1.53 | 1.40 | |

The same step (weights quantisation + forward + backward, C++, bs 200),
us, before -> after:

| net | M1 float64 | M1 nvfp4 | M1 mxfp4 | Xeon avx2 f64 | avx2 nvfp4 | avx2 mxfp4 | avx512 f64 | avx512 nvfp4 | avx512 mxfp4 |
|---|---|---|---|---|---|---|---|---|---|
| 24-256-256-128-8 | 5270 | 9882 -> 3830 (0.73) | 8278 -> 3626 (0.69) | 9085 | 19723 -> 7263 (0.80) | 18493 -> 6668 (0.73) | 9711 | 6266 (0.65) | 5707 (0.59) |
| 4-64-64-64-2 | 729 | 2426 -> 875 (1.20) | 2024 -> 820 (1.12) | 1336 | 4190 -> 1745 (1.31) | 3708 -> 1577 (1.18) | 1390 | 1599 (1.15) | 1448 (1.04) |
| 4-96-96-96-2 | 1354 | 3859 -> 1415 (1.04) | 3254 -> 1317 (0.97) | 2329 | 6772 -> 2749 (1.18) | 6241 -> 2488 (1.07) | 2468 | 2471 (1.00) | 2256 (0.91) |
| 4-128-128-128-2 | 2138 | 5287 -> 1977 (0.92) | 4453 -> 1893 (0.89) | 3474 | 9230 -> 3968 (1.14) | 8418 -> 3532 (1.02) | 3777 | 3392 (0.90) | 3071 (0.81) |
| 2-16-16-1 | 75 | 246 -> 121 (1.60) | 197 -> 111 (1.48) | 146 | 406 -> 258 (1.77) | 370 -> 228 (1.56) | 147 | 210 (1.43) | 190 (1.29) |

**G2 holds on the surrogate shape and does not on the 64-wide
regression.** An FP4 layer of width n quantises about 4 x batch x n values
a step (A for fprop; dZ for dgrad with stochastic rounding; the
Hadamard-transformed dZ^T and A^T for wgrad, one stochastically rounded),
at 1.5-3 ns each here, against 3 x batch x n^2 multiply-adds that float64
does at ~15 a ns: the FP4 GEMMs save ~2 ns of float64 time per quantised
value at n = 64 and ~8 at n = 256. The crossover is n ~ 96 (M1), ~128
(AVX2), ~96 (AVX-512); below it the recipe's quantisation, not the
arithmetic, is the cost. The tiny net is fixed per-step overhead (a few
hundred values a layer, eight quantiser calls a step). Stochastic rounding
(~1 ns a draw for the scalar SplitMix64, 26 k draws a 64-wide layer-step;
replaced by the third pass below)
and the division by the non-power-of-two nvfp4 scale are the two costs
left that exactness pins; a different, vectorisable random stream would
change every trained network and was not taken.

### Speed gates, third pass (2026-09-25)

G2, training, ms an epoch, `test/bench_neural_net_fp4.py` (M1 Pro,
neon-dotprod, module build, median of 5; load average 4.9 at the start,
9-11 over the previous 15 min), before (second pass, table above) -> now:

| shape | float64 | nvfp4 | mxfp4 | nvfp4 / f64 | mxfp4 / f64 |
|---|---|---|---|---|---|
| surrogate 400 x 24 -> 8, 256-256-128 | 8.73 | 5.38 | 5.24 | 0.75 -> **0.62** | 0.74 -> **0.60** |
| regression 3000 x 4 -> 2, 64-64-64 | 9.53 | 8.69 | 8.44 | 1.11 -> **0.91** | 1.02 -> **0.89** |
| tiny 2000 x 2 -> 1, 16-16 (no gate) | 0.63 | 0.80 | 0.76 | 1.53 -> 1.27 | 1.40 -> 1.20 |

The training step in C++ (weights quantisation + forward + backward,
bs 200, us; `before` is HEAD 48fafb16's headers in the same binary run,
FP4 / float64 in brackets). Mac: M1 Pro, load 3.5. x86: cordeshub Xeon
Silver 4416+, g++ 11.4 `-O3`, **load average 15-17 on 24 cores from
another user** (expect 10-20 % noise; the float64 column moves by that
much between runs):

| net | M1 f64 | M1 nvfp4 before -> now | M1 mxfp4 before -> now | avx2 f64 | avx2 nvfp4 | avx2 mxfp4 | avx512 f64 | avx512 nvfp4 | avx512 mxfp4 |
|---|---|---|---|---|---|---|---|---|---|
| 24-256-256-128-8 | 4350 | 2990 (0.73) -> 2419 (0.56) | 2871 (0.70) -> 2375 (0.55) | 9132 | 7304 (0.81) -> 7318 (0.80) | 6669 (0.74) -> 6289 (0.69) | 9738 | 6305 (0.61) -> 5621 (0.58) | 5759 (0.56) -> 5224 (0.54) |
| 4-64-64-64-2 | 567 | 677 (1.20) -> 538 (0.95) | 630 (1.12) -> 516 (0.91) | 1366 | 1749 (1.32) -> 1510 (1.11) | 1590 (1.20) -> 1422 (1.04) | 1393 | 2114 (1.54) -> 1391 (1.00) | 1846 (1.34) -> 1288 (0.92) |
| 4-96-96-96-2 | 1040 | 1113 (1.06) -> 860 (0.83) | 1040 (0.99) -> 839 (0.81) | 2347 | 2754 (1.19) -> 2447 (1.04) | 2506 (1.08) -> 2281 (0.97) | 2391 | 2481 (0.86) -> 2200 (0.92) | 2373 (0.82) -> 2006 (0.84) |
| 4-128-128-128-2 | 1640 | 1549 (0.93) -> 1246 (0.76) | 1457 (0.87) -> 1226 (0.75) | 3574 | 5114 (1.16) -> 3409 (0.95) | 3740 (0.85) -> 3169 (0.89) | 3652 | 3604 (0.92) -> 3029 (0.83) | 3013 (0.77) -> 2743 (0.75) |
| 2-16-16-1 | 59 | 96 (1.61) -> 77 (1.30) | 88 (1.48) -> 76 (1.28) | 162 | 236 (1.64) -> 203 (1.25) | 226 (1.57) -> 240 (1.48) | 164 | 206 (1.43) -> 182 (1.11) | 193 (1.34) -> 175 (1.07) |

(x86 "before" is the old headers' binary in the same session, whose own
float64 column read 1329-1377 / 2316-2889 / 3934-4411 -- the load.)

**The gate is met on the surrogate shape (0.60-0.62 on the Mac) and is not
met on the 64-wide regression**: 0.89-0.91 an epoch on the Mac (target
<= 0.8), 0.91-1.11 a step on x86. Where the 64-wide step goes now (M1,
two FP4 layers, per step): the FP4 GEMMs ~95 us (6 x 14-17 us; their float
combine -- convert, scale, add per 16-element sub-block and output -- is
as much work as the int8 dot products at K = 64), quantisation ~120 us
(fprop exact 2 x 14, dgrad SR 2 x 16, wgrad SR 2 x 14 + RNE 2 x 12, weights
2 x 6), Hadamard 2 x 9 -- ~235 us against float64's ~250 us for the same
three GEMMs of both layers; the rest of the step (~300 us: `tanh`, 190 us
of it, and the float64 first/last layers) is shared. So at n = 64 FP4
arithmetic and float64 arithmetic cost the same and the ratio is set by
the shared part; the crossover moved from n ~ 96-128 to n ~ 64, and
width 96 is now 0.81-0.83 on the Mac. The tiny net's remaining overhead is
per-call fixed work (eight quantiser calls, block-scale setup, 2-D weight
packing of 16 x 16 tiles) on a few hundred values -- 77 vs 59 us.

G1, inference, is unchanged in speed (worst FP4 / float64 0.67 on the Mac
at batch 256; Python table from the bench script: batch 1 7.1-7.6 us vs
24.8, batch 32 206-214 vs 352-387, batch 256 1623-1729 vs 2534-2833) and
bit-identical; x86 (C++, load 15-17): avx2 12-13.5 / 629-812 / 5093-5418
us vs float64 27-28 / 984-1008 / 6064-6518, avx512-vnni 12.3-13.8 /
585-614 / 4684-4934 vs 26-28 / 977-1003 / 5905-6150.

### Fast tanh (2026-09-25): `MlpMath.h`

libm's `tanh` was the largest shared cost (above: 190 us of a 64-wide
training step; at batch 256 most of the 24-256-256-128-8 forward pass on the
Xeon, where glibc's costs ~20 ns a call). **By decision ("go"), every
float64 result that goes through tanh, the sigmoid or SiLU changes by
rounding**: one vectorised implementation now serves every path -- MlpCore's
batch forward (`act_apply`) and single-sample `act_value<double>`
(`predict_scalar`), int8 and FP4 inference, FP4 training -- so a
network's activations are the same bits on every path and every SIMD
variant. The derivatives MlpCore's Taylor passes need are formulas of the
value (`tanh' = 1 - t^2`, `tanh'' = -2 t tanh'`, `tanh''' = -2 tanh' (1 -
3 t^2)`), so they follow; `1 - t^2` is now computed with the square rounded
first on every variant (`one_minus_sq_n`), which makes FP4 training of tanh
networks bit-identical across variants too. Softplus's and SiLU's
derivatives take their sigmoid from the same code. Softplus's value (needs
`log1p`) and `sin` stay libm's: no default activation, and a vector
`log1p` / `sin` with argument reduction is a project of its own. Dual.h
(vendored, SHA-checked) keeps `std::tanh` for forward-mode values; the
tests comparing it with the reverse pass use tolerances far above 2 ulp.

**Algorithm** (after Cephes `tanh.c`, fdlibm's `exp` reduction; notices in
the header): `|x| < 0.625`: `x + x (z P(z) / Q(z))`, `z = x^2`, Cephes'
coefficients; above: `1 - 2 / (e^(2|x|) + 1)` with the argument clamped to
40. The two branches share one division (`num / den` selected per lane).
`exp`: Cody-Waite `n ln2_hi + n ln2_lo`, `n` rounded with the 1.5 x 2^52
shifter, degree-13 Taylor polynomial by Estrin's scheme (the Horner chain
was latency-bound: 4.2 -> 3.3 ns an element on the M1), `2^n` from
exponent bits (two factors for the sigmoid's full range [-746, 710]). The
sign is put back from `x`: exactly odd, `tanh(-0) = -0`, `tanh(+-inf) =
+-1`, NaN passes through. Variants: NEON (2 lanes), **SSE2** (2 lanes, the
x86-64 baseline: a wheel without `IMPBFF_WITH_AVX2` still vectorises), AVX2
(4), AVX-512F (8), generic scalar (also every loop's tail); same operations
in the same order, products kept apart from adds by an empty asm
(`IMPBFF_MATH_KEEP`, the FP4 convention), selected at compile time
(`IMPBFF_MATH_NO_SIMD` or `IMPBFF_FP4_NO_SIMD` force generic).

**Accuracy** (`test/cpp_snippets/test_mlp_math.cpp`, run by
`test_neural_net_fp4.py` for every variant and by CI's `fp4_kernels_x86`):
probes are a 4e6-point grid on [-20, 20], 8 mantissas of every exponent
2^-1074 .. 2^110 (both signs), 9 x 80 k consecutive doubles around 0.3,
0.625, 1, 19, 20, 354, 709, 710 and 745, 1e6 uniform on [-30, 30], 2e5
random bit patterns, and the edge cases.

| | tanh | sigmoid | SiLU |
|---|---|---|---|
| vs Apple libm (arm64; and x86_64 under Rosetta) | 1 ulp | 2 ulp | 2 ulp |
| vs glibc 2.35 (cordeshub) | 2 ulp (at x = -0.99958) | 2 ulp | 2 ulp |
| vs the true value (x86 `tanhl`, 64-bit mantissa) | 1 ulp | | |

(sigmoid / SiLU: against `1 / (1 + exp(-z))` with libm's `exp`, on results
in the normal range; subnormal results round twice.) Also asserted: odd
bit for bit, `tanh(x) == x` for `|x| < 1e-8`, the edge cases, monotone on
the grid and on the consecutive-double sweeps across the branch point and
saturation, vector lanes == the scalar code, and one fingerprint of every
output, `3f4f1d48f024ef7f`, on the Mac (neon, generic, and under Rosetta
AVX2, SSE2, generic) and cordeshub (generic, SSE2, AVX2, AVX-512,
`-march=sapphirerapids`, g++ 11.4).

Numerics tests after the change: sklearn forward parity (1e-10) passes;
the PyTorch fixtures (tanh, SiLU and softplus layers) are off by
6.3e-8 (float32 ONNX, tolerance 1e-6) and 1.7e-16 (double ONNX and
safetensors, tolerance 1e-12) -- no tolerance needed changing; FD and
Dual-identity derivative checks, FP4 accuracy bounds and the int8 bounds
pass. The FP4 training fingerprint changed (a tanh net was added to it,
`64407c22cd039ecf`, equal on all nine builds); the FP4 cross-variant
golden tests are unchanged.

**Speed**, ns an element (1e6 values on [-4, 4], best of 5):

| machine / variant | tanh | libm tanh | sigmoid | libm `1/(1+exp(-z))` |
|---|---|---|---|---|
| M1 Pro, neon | 3.3 | 8.3 | 2.8 | 6.5 |
| M1 Pro, generic | 9.0 | 8.3 | 8.9 | 6.4 |
| Xeon 4416+, avx512 | 2.6 | 19.7 | 2.2-2.8 | 5.5-6.5 |
| Xeon 4416+, avx2 | 3.9 | 19.7 | 3.0 | 5.8 |
| Xeon 4416+, sse2 (baseline) | 7.4 | 19.6 | 6.1 | 5.8 |
| Xeon 4416+, generic | 10.9 | 19.7 | 9.7 | 4.9 |

The generic scalar code is slower than libm's `exp` for the sigmoid (and
on par with Apple's `tanh`); no build selects it unless forced (x86-64
takes SSE2 at least, AArch64 NEON).

**Gates** (C++ harness, the old and new headers in the same session, us;
Mac load 5-7, cordeshub load 17-20 on 24 cores from another user, so x86
cells carry 10-20 % noise):

G1, inference, 24-256-256-128-8, before -> after:

| machine / variant | batch | float64 | fp4 W4A8 | nvfp4 W4A8 | nvfp4 W4A4 |
|---|---|---|---|---|---|
| M1 Pro, neon-dotprod | 1 | 22.2 -> 22.1 | 6.0 -> 5.6 | 6.1 -> 5.5 | 5.9 -> 5.5 |
| M1 Pro, neon-dotprod | 32 | 336 -> 293 | 182 -> 144 | 180 -> 142 | 178 -> 142 |
| M1 Pro, neon-dotprod | 256 | 2308 -> 1988 | 1473 -> 1165 | 1439 -> 1131 | 1440 -> 1128 |
| Xeon 4416+, avx2 | 1 | 34.3 -> 22.1 | 12.2 -> 7.1 | 12.7 -> 7.2 | 13.2 -> 7.9 |
| Xeon 4416+, avx2 | 32 | 970 -> 555 | 625 -> 204 | 633 -> 213 | 657 -> 234 |
| Xeon 4416+, avx2 | 256 | 5866 -> 2538 | 5074 -> 1650 | 5106 -> 1719 | 5357 -> 1929 |
| Xeon 4416+, avx512-vnni | 1 | 28.6 -> 21.0 | 13.6 -> 4.8 | 14.5 -> 4.9 | 14.5 -> 5.4 |
| Xeon 4416+, avx512-vnni | 32 | 999 -> 553 | 595 -> 130 | 591 -> 135 | 609 -> 148 |
| Xeon 4416+, avx512-vnni | 256 | 5897 -> 2323 | 4706 -> 1043 | 4756 -> 1080 | 4945 -> 1241 |
| Xeon 4416+, x86-64 baseline (sse2 math, generic FP4) | 1 | 32.9 -> 28.9 | 61.3 -> 58.7 | 61.2 -> 57.7 | 64.0 -> 59.9 |
| Xeon 4416+, x86-64 baseline | 32 | 1356 -> 1008 | 2331 -> 2015 | 2329 -> 2019 | 2391 -> 2094 |
| Xeon 4416+, x86-64 baseline | 256 | 8920 -> 6074 | 18630 -> 16031 | 18801 -> 16212 | 19417 -> 16737 |

(The baseline row is a build without `-m` flags, what a wheel without
`IMPBFF_WITH_AVX2` runs: MlpMath's SSE2 path, but the FP4 kernel's generic
scalar code -- FP4 there is 2.6x *slower* than float64, before and after;
its training step 64-wide: float64 2048 -> 1313 us.)

Python (`bench_neural_net_fp4.py`, M1, load 5): float64 25.2 / 318 / 2096
us at batch 1 / 32 / 256 (third pass: 24.8 / 352-387 / 2534-2833), FP4
7.2-7.5 / 152-161 / 1207-1303 (7.1-7.6 / 206-214 / 1623-1729); worst FP4 /
float64 0.30 / 0.51 / 0.62.

G2, training step (bs 200, weights quantisation + forward + backward), us,
before -> after (FP4 / float64 after):

| net | M1 f64 | M1 nvfp4 | M1 mxfp4 | avx2 f64 | avx2 nvfp4 | avx2 mxfp4 | avx512 f64 | avx512 nvfp4 | avx512 mxfp4 |
|---|---|---|---|---|---|---|---|---|---|
| 24-256-256-128-8 | 4179 -> 3943 | 2449 -> 2181 (0.55) | 2389 -> 2152 (0.55) | 9066 -> 6801 | 6695 -> 4234 (0.62) | 6226 -> 3823 (0.56) | 9519 -> 6733 | 5591 -> 3120 (0.46) | 5227 -> 2721 (0.40) |
| 4-64-64-64-2 | 578 -> 501 | 541 -> 460 (0.92) | 521 -> 438 (0.87) | 1352 -> 640 | 1490 -> 757 (1.18) | 1395 -> 681 (1.06) | 1385 -> 591 | 1401 -> 623 (1.05) | 1296 -> 528 (0.89) |
| 4-96-96-96-2 | 1060 -> 936 | 885 -> 739 (0.79) | 855 -> 720 (0.77) | 2328 -> 1244 | 2417 -> 1313 (1.06) | 2253 -> 1184 (0.95) | 2379 -> 1256 | 2198 -> 1027 (0.82) | 1997 -> 885 (0.70) |
| 4-128-128-128-2 | 1674 -> 1498 | 1273 -> 1061 (0.71) | 1213 -> 1028 (0.69) | 3569 -> 2131 | 3364 -> 1907 (0.89) | 3141 -> 1802 (0.85) | 3683 -> 2044 | 3087 -> 1473 (0.72) | 2824 -> 1269 (0.62) |
| 2-16-16-1 | 60 -> 51 | 76 -> 68 (1.33) | 73 -> 62 (1.21) | 160 -> 64 | 201 -> 112 (1.76) | 187 -> 102 (1.60) | 150 -> 52 | 170 -> 88 (1.68) | 170 -> 73 (1.39) |

Python G2 (M1, ms an epoch, median of 5, load 5): surrogate float64 8.46,
nvfp4 4.81 (0.57), mxfp4 4.62 (0.55); 64-wide regression 8.22 / 7.46
(0.91) / 6.59 (0.80); tiny 0.51 / 0.68 / 0.62.

**Where the 64-wide FP4 gate lands: 0.80-0.91 on the Mac (epoch), 0.87-0.92
a step; 0.89-1.18 on x86** -- not the <= 0.8 target, and the fast tanh
cannot deliver it: tanh is shared, so removing ~40-60 % of the shared part
makes both paths faster by the same microseconds and moves the ratio
*toward* FP4-arithmetic / float64-arithmetic at n = 64, which the third
pass measured at ~0.94 (M1) and ~1.1 (x86). What the fast tanh does buy is
absolute: the 64-wide float64 step 2.1x (x86) and 1.15x (M1) faster, FP4
2.0x / 1.18x, and x86 inference at batch 256 2.3x (float64) and 3.1-4.5x
(FP4). The remaining FP4 cost at n = 64 is the recipe's quantisation (four
operands a layer-step, two of them stochastically rounded) and the FP4
GEMMs' float combine.

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
RNE for weights and activations (4.4), SR draws counter-based, keyed by
`options.seed`, the step, the layer and the operand;
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
7e-8; SR deterministic per key and unbiased (the counter-based draws of the
third pass: P(up) exact to 2^-17 over all draws, means of 2^16 / 2^22
roundings within 4 sigma); one FP4 step bit-reproducible and one training
fingerprint for every variant; FP4 gradient cosine 0.98 to the
float64 one on a random 6-64-48-32-2 net. Measured (Python):

| task | float64 | nvfp4 | mxfp4 |
|---|---|---|---|
| 4-64-64-64-2 tanh regression, held-out MSE (var 0.51), SplitMix64 SR (to 2026-09-24) | 8.4e-4 | 2.6e-3 (3.1x) | 3.0e-3 (3.5x) |
| the same, counter-based SR + training's element rule (2026-09-25) | 8.4e-4 | 2.57e-3 (3.04x) | 2.62e-3 (3.10x) |
| HMM surrogate set (400 x 24 -> 8), 256-256-128, held-out MAE, SplitMix64 SR | 0.0959 | 0.1046 (1.09x) | 0.1036 (1.08x) |
| the same, counter-based SR (2026-09-25) | 0.0959 | 0.1002 (1.04x) | 0.1023 (1.07x) |
| seconds an epoch (surrogate), first pass | 0.011 | 0.020 | 0.017 |
| ms an epoch (surrogate), second pass (bench script) | 11.0 | 8.2 | 8.1 |

New and old accuracy agree within one seed's noise (the bounds in
`test_neural_net_fp4_training.py` -- 5x / 6x the float64 MSE, 1.3x the
MAE -- are unchanged). FP4 training's speed: see the third-pass gates
above -- 0.60-0.62x float64 an epoch on the surrogate shape, 0.89-0.91x at
64 wide, and slower on the tiny net.

## Ternary: BitNet b1.58 (2026-09-25)

`internal/MlpTernary.h` (formats, quantisers, kernels, forward pass) and
`internal/MlpTernaryTrain.h` (quantisation-aware training). Sources: Ma et
al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
(arXiv 2402.17764) and its "Training Tips, Code and FAQ" (microsoft/unilm,
`weight_quant`, `activation_quant`, BitLinear); bitnet.cpp's I2_S idea
(microsoft/BitNet, MIT, no code copied); ggml's TQ1_0 / TQ2_0
(ggml-org/llama.cpp, MIT: the base-3 fixed-point byte, the `q + 1` codes
with `sum q a = sum (q + 1) a - sum a`). The MIT notices are in the header.

### Format and quantisers

- Weights: `gamma = max(mean |W|, 1e-5)` over the whole matrix (absmean,
  per tensor), `q = clamp(round(W / gamma), -1, 1)`, value `q gamma`
  (computed as `q / (1 / gamma)`, BitNet's form). `round` is
  round-half-to-even (torch.round, `nearbyint` in the default mode):
  `W / gamma = +-0.5 -> 0`, `+-1.5 -> +-2 -> +-1`; NaN -> 0. The absmean
  is summed in a fixed order (8 interleaved partial sums) on every build.
  `_row` formats: one absmean a weight row (a PTQ option; BitNet is per
  tensor).
- Activations: every ternary layer's input per row (token) to int8,
  `s_a = 127 / max(max |x|, 1e-5)`, `q_a = clamp(round(x s_a), -128, 127)`,
  ties to even, NaN -> 0 -- W1.58A8 always (`quantize_activations` is
  ignored). No RMSNorm before the quantiser (BitLinear has one; bff's
  layer is `W x + b` like the float net).
- Arithmetic: `z = ((sum_k (q + 1) q_a - sum_k q_a) / s_a) * gamma + b`,
  the integer sum exact in int32, each double product rounded on its own
  (the FP4 kernels' unfused combine), so every SIMD variant gives the
  generic bits.
- Storage: `ternary` / `ternary_row` 2 bits a weight, four codes `q + 1` a
  byte (element `4 i + j` in bits `2 j`, rows padded to 4; code 3 is
  refused on load). `ternary_tq1` / `ternary_tq1_row` five trits a byte,
  `ceil(v 256 / 243)` with `v = sum_n (q_n + 1) 3^(4 - n)`, decoded as
  `((byte 3^n) mod 256) 3 >> 8` (ggml's form; element order within the row
  is bff's, rows padded to 5). Only the storage differs: `_tq1` predicts
  bit-identically to its 2-bit twin. Document: `bff.quantized_neural_net`,
  `"quantization": "ternary..."`, layer `"codes"` (bin) and `"weight_scale"`
  (float64) or `"weight_scales"` (bin of n_out little-endian float64);
  kept layers `"precision": "float64"` + `"weight"`. Bit-exact round trip.
- Kernels: weights repacked once into tiles of `kNR` rows, 16-wide
  sub-blocks whose byte `4 o + j` holds the codes of elements `4 l + j`
  (`l = 0..3`) in bits `2 l`; `(chunk >> 2 l) & 3` is directly the operand
  of NEON dotprod `vdotq_laneq_s32` (plain NEON emulates it), AVX2
  `maddubs` (u8 x s8, |sum| <= 2048, no int16 saturation), AVX-512 VNNI
  `dpbusd`; generic unpacks a sub-block and runs int8 dot products. Same
  compile-time selection and switches as FP4 (`IMPBFF_FP4_NO_SIMD`,
  `IMPBFF_FP4_NO_DOTPROD`); `get_kernel_name()` reports it.

### Training (`precision = "ternary"`)

BitNet's recipe on bff's MLP: float64 master weights and Adam state; every
step each ternary layer's weights are quantised (absmean, per tensor) and
packed once; fprop is `Q(A) Q(W)^T + b` on the ternary x int8 kernel --
the same quantisers and kernel as `QuantizedNeuralNet`, so the result's
`get_quantized_network()` (ternary layers with fprop's trits and scale,
kept layers float64) predicts training's forward pass bit for bit
(checked in C++ with the library's MatGemm and the portable GEMM).
Backward is the straight-through estimator of both quantisers
(BitLinear's `x + (quant(x) - x).detach()`): the layer is treated as the
linear map of the dequantised operands `Â = q_a / s_a`, `Ŵ = q gamma`,
with the quantisers as the identity, so `dW = dZ^T Â` (reaching every
master weight, clipped ones included -- no clipping mask), `dA = dZ Ŵ`,
`db = sum_rows dZ`, and nothing flows into `gamma` or `s_a`. dgrad and
wgrad run in float64 (MatGemm) on those operands: BitNet keeps gradients
in high precision. First and last layer float64 by default
(`ternary_keep_first_layer` / `ternary_keep_last_layer`; BitNet keeps its
embedding and head in high precision). No randomness beyond
train_neural_net's: deterministic per seed.

### Accuracy and size (measured 2026-09-25, M1 Pro)

| task | float64 | ternary QAT | ternary QAT, every layer | ternary PTQ | ternary_row PTQ | nvfp4 PTQ (W4A8) | int8 PTQ |
|---|---|---|---|---|---|---|---|
| 4-64-64-64-2 tanh regression, held-out MSE (var 0.51) | 8.45e-4 | 1.52e-3 (1.80x) | 1.86e-3 (2.20x) | 0.233 (276x) | 0.206 (244x) | 1.19e-2 (14x) | 8.60e-4 (1.02x) |
| HMM surrogate set (400 x 24 -> 8), 256-256-128, held-out MAE | 0.0959 | 0.0939 (0.98x) | | 0.173 (1.81x) | | | |

**Post-training ternary quantisation of a float-trained net is not
usable** (276x the MSE); quantisation-aware training is what makes
ternary work: 1.8x float64's MSE on the regression (FP4 training: 3.0x)
and float64's MAE on the surrogate set. Fixture nets under PTQ: max error
0.41-0.51 of the output absmax per tensor, 0.12-0.40 per row. Test bounds
(`test_neural_net_ternary.py`): QAT <= 3x float64 (every layer ternary
4.5x), QAT < 5 % of PTQ, surrogate QAT <= 1.2x float64 and < PTQ.

Sizes, 24-256-256-128-8 (107 k weights): ternary 2.002 bits a weight
(32x under float64), ternary_tq1 1.630 (39x; the 24-wide rows pad to 25),
ternary_row 2.393 and ternary_tq1_row 2.021 (a float64 scale a row costs
2.7 bits a weight on the 24-wide input layer). The QAT regression net
(8576 weights) with its float64 first and last layer: 4.79 bits a weight,
5.1 kB against 68.6 kB float64; every layer ternary 2.03 bits.

### Speed gates (2026-09-25)

G1, inference, us a predict() call, 24-256-256-128-8 (float64 = MatGemm):

| machine / variant | batch | float64 | nvfp4 W4A8 | ternary | ternary / float64 |
|---|---|---|---|---|---|
| M1 Pro, neon-dotprod (Python, load 7.0) | 1 | 26.1 | 7.7 | 7.8 | 0.30 |
| M1 Pro, neon-dotprod (Python) | 32 | 300 | 156 | 145 | 0.48 |
| M1 Pro, neon-dotprod (Python) | 256 | 2052 | 1251 | 1199 | 0.58 |
| M1 Pro, neon-dotprod (C++, load 5.5) | 1 | 27.9 | 5.7 | 5.6 | 0.20 |
| M1 Pro, neon-dotprod (C++) | 32 | 278 | 140 | 128 | 0.46 |
| M1 Pro, neon-dotprod (C++) | 256 | 1914 | 1100 | 1019 | 0.53 |
| Xeon 4416+, avx2 (C++, load 15-17) | 1 | 20.5 | 7.3 | 5.7 | 0.28 |
| Xeon 4416+, avx2 (C++) | 32 | 561 | 209 | 164 | 0.29 |
| Xeon 4416+, avx2 (C++) | 256 | 2506 | 1662 | 1191 | 0.48 |
| Xeon 4416+, avx512-vnni (C++, load 15-17) | 1 | 19.4 | 5.0 | 6.1 | 0.31 |
| Xeon 4416+, avx512-vnni (C++) | 32 | 516 | 130 | 129 | 0.25 |
| Xeon 4416+, avx512-vnni (C++) | 256 | 2410 | 1050 | 928 | 0.39 |

256 x 256 x 256 ternary x int8 GEMM (snippet): M1 dotprod 268 us, plain
NEON 698 us; Xeon AVX2 294 us, AVX-512 VNNI 231 us (portable double GEMM
8.9 / 18.6 ms). Activation quantisation is 10-20 % of it (M1 44 us, Xeon
48-53 us).

G2, training. Python, M1 Pro (neon-dotprod, load 7.0, median of 5), ms an
epoch, batch 200, first/last layer float64:

| shape | float64 | ternary | ternary / float64 |
|---|---|---|---|
| surrogate 400 x 24 -> 8, 256-256-128 | 8.02 | 6.57 | **0.82** |
| regression 3000 x 4 -> 2, 64-64-64 | 7.50 | 6.92 | **0.92** |
| tiny 2000 x 2 -> 1, 16-16 (no gate) | 0.51 | 0.53 | 1.03 |

The step in C++ (quantise weights + forward + backward, bs 200, us;
ternary / float64 in brackets; Xeon under load 14-17 from another user):

| net | M1 f64 | M1 ternary | M1 every layer | avx2 f64 | avx2 ternary | avx512 f64 | avx512 ternary |
|---|---|---|---|---|---|---|---|
| 24-256-256-128-8 | 3669 | 3142 (0.86) | 3176 (0.87) | 6123 | 5334 (0.87) | 6495 | 5449 (0.84) |
| 4-64-64-64-2 | 473 | 474 (1.00) | 458 (0.97) | 637 | 658 (1.03) | 648 | 657 (1.01) |
| 4-128-128-128-2 | 1514 | 1315 (0.87) | 1294 (0.85) | 2079 | 2084 (1.00) | 2108 | 2103 (1.00) |
| 2-16-16-1 | 50 | 49 (1.00) | 56 (1.14) | 62 | 67 (1.07) | 55 | 60 (1.09) |

**Inference is the win (0.2-0.6x float64, 0.25-0.5x on the Xeon); ternary
training costs about what float64 does** (0.82-0.92 an epoch on the Mac,
0.84-1.03 a step on the Xeon). Only fprop runs on the integer kernel --
dgrad and wgrad are float64 GEMMs by the recipe, two thirds of float64's
arithmetic -- and the per-step overheads (absmean, trits and repacking of
the weights, the int8 activations and `Â` for wgrad) eat most of the fprop
saving below ~128 wide. An int8 dgrad on the ternary kernel would be
faster but quantises dZ, which BitNet does not; not taken.

Checked: `cpp_snippets/test_ternary_kernels.cpp` (quantiser ties / clamp /
NaN, SIMD activation rounding == scalar, TQ2 / TQ1 round trips incl. all
243 five-trit strings, kernel == element-by-element reference on 150
shapes and saturating operands, predict == float64 arithmetic on the
dequantised operands to 4e-16, to_model() == training's forward pass, STE
gradient cosine 0.86 to float64's on a random all-ternary net, one
fingerprint `4f6b5c542ab55f36` over kernel outputs, predict and three
training steps) is bit-identical on the Mac (neon-dotprod, neon, AVX2 under
Rosetta, generic) and cordeshub (generic, AVX2, AVX-512 VNNI,
`-march=sapphirerapids`); pytest runs the matrix and CI job
`fp4_kernels_x86` builds and runs it generic / AVX2 / AVX-512 (native or
Intel SDE). `test_neural_net_ternary.py`: sizes (exact bytes), msgpack
round trips, refusals, determinism per seed, the trained ternary layers
== the quantiser on the master weights, accuracy bounds above.
`test/bench_neural_net_fp4.py --ternary` prints the Python gates.
