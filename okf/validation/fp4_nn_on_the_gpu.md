---
type: validation
title: "FP4 networks on the GPU: decoded per tile, as fast as f32, and the bit trick Metal breaks"
description: QuantizedNeuralNet (fp4 / mxfp4 / nvfp4) now runs through the compute door with its weights kept on the device as 4-bit codes. The first kernel decoded a code per multiply-add and was 1.3-1.8x slower than f32; a tiled kernel that decodes each weight tile once into workgroup memory runs FP4 as fast as f32 (sometimes faster) and the f32 path 1.5-1.9x faster than the untiled kernel on 128/256-wide nets (64 rows x 64 outputs a workgroup, 256 threads, the input tile k-major so both tiles fill 16 384 bytes exactly). W4A32, so 2-8e-7 from the dequantised network against the CPU W4A8's 1e-2. The E2M1-to-f32 bit trick (the analogue of int4's 0x6400) decodes 0.5 as 0 on Metal, which flushes subnormals; an integer decode is exact.
resource: /Users/tpeulen/dev/imp.bff
tags: [validation, imp.bff, performance, gpu, webgpu, neural-net, fp4]
timestamp: '2026-10-03T00:00:00Z'
---
# FP4 networks on the GPU

[nn_on_the_gpu.md](nn_on_the_gpu.md) put the float network through the
compute door. Quantized networks never reached it: `QuantizedNeuralNet::predict`
always ran the CPU's integer FP4 kernels, and the float door converted,
transposed and uploaded every weight on every call.

## What changed

- **ABI 3** (`include/ComputeBackend.h`): `mlp_upload` / `mlp_run` /
  `mlp_release`, a network kept on the device. Per layer f32 (doubles in) or
  FP4: `Fp4Tensor`'s codes as they are, plus the decoded scale of every
  16-element sub-block as f32, so fp4's row scale, mxfp4's E8M0 and nvfp4's
  E4M3 x tensor scale are one array on the device.
  `get_compute_backend_generation()` tags a handle with the backend it came
  from; a reload drops it (leaked, never freed into the wrong backend).
- **`QuantizedNeuralNet`** uploads on its first GPU-sized batch and keeps the
  handle (`mutable`, under a mutex; copies start without one).
  `get_last_backend()` says where a call ran. int8, ternary and W4A4
  (`quantize_activations`) stay on the CPU.
- **`gpu/mlp.wgsl`** is now tiled, for f32 and FP4 alike; the old
  `mlp_forward` is upload + run + release on it.

## The kernel

A workgroup of 256 threads owns 64 rows x 64 outputs and walks the
contraction 32 at a time: the input tile and the weight tile in workgroup
memory, a 4 x 4 register block a thread (rows tr + 16 i, outputs to + 16 j).
The input tile is stored k-major, transposed, so the two tiles fill 16 384
bytes -- WebGPU's default workgroup-storage limit -- exactly; the row-major
tile needs a pad column against bank conflicts and does not fit. FP4 codes
are decoded and scaled on the way into workgroup memory -- once per
workgroup, not once per row. A layer's outputs are padded to 64, the batch's
inputs to 32 and its rows to 64, with zeros, so the inner loop checks
nothing; the batch itself crosses unpadded (`x_stride`, `y_stride`),
because for a 4-input, 2-output network the padding was 8x the bytes and
made the narrow net 25 % slower than the old kernel until it was removed.
The shape is measured, not assumed
(`benchmark/gpu_mlp_tile_tuning.py`): 8 x 8 register blocks and vec4 loads
were slower on the M1 Pro than 4 x 4 with 256 threads.

## The decoder

E2M1 is `s.ee.m`. The int4 trick -- `half(0x6400 | n) - 1024` -- has an FP4
analogue: shift `ee.m` into f32's two lowest exponent bits and top mantissa
bit, the sign to bit 31, and multiply by 2^126 to move the bias from 1 to 127.
It is exact in IEEE arithmetic, and **wrong on Metal**: code 1 (0.5) is an f32
subnormal before the multiply, WGSL allows subnormals to be flushed, and Metal
flushes them -- 0.5 and -0.5 decode as 0. Checked over all 16 codes:

| decoder | M1 Pro (Metal) |
|---|---|
| bit trick, x 2^126 | 0.5 -> 0, -0.5 -> -0 |
| `f32(select((2+m) << e, 2m, e == 0)) * 0.25`, sign by `select` | exact |
| 16-entry `const` table | exact |

The integer decode is used; it measured ~5 % ahead of the table in the
untiled kernel and the difference vanishes once decoding is per tile.

## Accuracy

Relative to the dequantised network in double (the reference the device
should reproduce), 30 001 rows, `test_fp4_on_the_gpu_is_the_decoded_network_in_f32`:
2-8e-7 for fp4 / mxfp4 / nvfp4, on a 5-70-40-37-33-50-45-3 net with every
activation and on a trained 2-64x3-1 net with scalers. The CPU's W4A8 is
about 1e-2 from the same reference (its activations are int8). So the GPU is
the *more* accurate path; the two agree to the CPU's error, not bit for bit.

## Speed

M1 Pro, through `predict()` from Python (list in, view out), minimum of 5;
weights resident for FP4, uploaded per call for f32. Prototype:
`benchmark/gpu_mlp_fp4_wgsl.py`.

| net | rows | f32 GPU, untiled | f32 GPU, 32x32 tile | f32 GPU, 64x64 tile | FP4 GPU, 64x64 tile | f64 CPU | fp4 CPU |
|---|---|---|---|---|---|---|---|
| 4-64-64-2 | 10k | 4.7 | 3.6 | 3.7 | 3.4 | 8.4 | 8.5 |
| 4-64-64-2 | 400k | 68.7 | 67.7 | 68.3 | 52.9 | 353 | 331 |
| 4-128x3-1 | 3k | — | 3.1 | 3.0 | 1.4 | 10.2 | 7.5 |
| 4-128x3-1 | 400k | 212.5 | 142.4 | 114.8 | 96.4 | 1400 | 1169 |
| 32-256x3-8 | 10k | 25.7 | 18.8 | 17.7 | 17.5 | 138 | 62 |
| 32-256x3-8 | 400k | 963.8 | 633.1 | 505.2 | 482.7 | 6872 | 2931 |

(ms; the untiled and 32x32 columns from HEAD's plugin rebuilt against ABI 3.)
The 64x64 tile is ~224 GFLOP/s on the widest net (f32, 400k rows), against
~180 at 32x32 and ~120 untiled; M1 Pro's f32 peak is ~5 TFLOP/s. The shape
came from a sweep (`benchmark/gpu_mlp_tile_tuning.py`): larger register
blocks (8 x 8) and vec4 loads were slower, so the remaining gap is where it
is -- the shader is simple enough to stay memory-tile-bound.

**Untiled FP4 was slower than f32** (1.3-1.8x): with a decode per weight per
row the kernel is ALU-bound, and these weights fit in cache, so 4 bits buy no
bandwidth. Tiling is what made FP4 free.

## When the device takes a batch

f32: at least 16 M multiply-accumulates, as before. A network with FP4
weights additionally needs 1 000 rows and 32 M: the CPU's FP4 kernels are
about twice its double ones, and a wide net with few rows leaves the GPU
idle -- 32-256x3-8 at 300 rows (42 M) is 2.9 ms on the GPU against 1.9 ms,
4-128x3-1 at 1 000 rows (33 M) a tie, 4-64-64-2 at 10 000 rows (45 M) 2.2 ms
against 8.1 ms.

## What this does not change

[is_the_gpu_backend_worth_it.md](is_the_gpu_backend_worth_it.md) still holds:
no caller in the library sends batches this large (the HMM surrogate sends
one row, the MCTS policy tens). The FP4 path is ready for the per-voxel
network of PRD-115 and for users evaluating surrogates on large grids from
Python.
