---
type: validation
title: "FP4 networks on the GPU: decoded per tile, as fast as f32, and the bit trick Metal breaks"
description: QuantizedNeuralNet (fp4 / mxfp4 / nvfp4) now runs through the compute door with its weights kept on the device as 4-bit codes. The first kernel decoded a code per multiply-add and was 1.3-1.8x slower than f32; a tiled kernel that decodes each weight tile once into workgroup memory runs FP4 as fast as f32 (sometimes faster) and the f32 path 1.4-1.6x faster than before on 128/256-wide nets. W4A32, so 2-8e-7 from the dequantised network against the CPU W4A8's 1e-2. The E2M1-to-f32 bit trick (the analogue of int4's 0x6400) decodes 0.5 as 0 on Metal, which flushes subnormals; an integer decode is exact.
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

A workgroup of 64 threads owns 32 rows x 32 outputs and walks the contraction
32 at a time: the input tile and the weight tile in workgroup memory, a 4 x 4
register block a thread. FP4 codes are decoded and scaled on the way into
workgroup memory -- once per workgroup, not once per row. Widths and row
counts are padded to 32 with zeros, so the inner loop checks nothing; the
batch itself crosses unpadded (`x_stride`, `y_stride`), because for a
4-input, 2-output network the padding was 8x the bytes and made the narrow
net 25 % slower than the old kernel until it was removed.

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

| net | rows | f32 GPU, old kernel | f32 GPU, tiled | FP4 GPU, tiled | f64 CPU | fp4 CPU |
|---|---|---|---|---|---|---|
| 4-64-64-2 | 10k | 4.7 | 3.6 | 2.2 | 9.0 | 8.1 |
| 4-64-64-2 | 400k | 68.7 | 67.7 | 56.9 | 345 | 329 |
| 4-128x3-1 | 3k | — | 3.1 | 1.5 | 9.5 | 7.2 |
| 4-128x3-1 | 400k | 212.5 | 142.4 | 131.8 | 1314 | 972 |
| 32-256x3-8 | 10k | 25.7 | 18.8 | 17.2 | 110 | 61 |
| 32-256x3-8 | 400k | 963.8 | 633.1 | 603.6 | 6406 | 3277 |

(ms; old-kernel column from HEAD's plugin rebuilt against ABI 3.) The tiled
kernel is ~180 GFLOP/s on the widest net (f32, 400k rows), against ~120
before; M1 Pro's f32 peak is ~5 TFLOP/s, so 8 x 8 register blocks and vector loads are the next
step.

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
