---
okf_version: "0.2"
---

# Portable CPU neural-network tuning (2026-10-03)

## Scope and invariant

This validation covers the CPU forward paths used by `NeuralNet` and
`QuantizedNeuralNet`: float64 `MatGemm` and the packed FP4 kernels. The
generic scalar, Apple NEON/NEON-dotprod, x86 AVX2, and AVX-512 VNNI paths
remain the supported implementations. No GPU file or third-party dependency
was changed.

The accepted change is in `include/internal/MlpFp4Kernels.h`: when the input
scaler is inactive, the FP4 forward pass borrows the caller's read-only input
for its first layer instead of copying the complete batch into thread-local
scratch. Scaled inputs still take the existing copy-and-transform path; later
layers still use the existing scratch buffer. The integer kernels, packing,
scales, activation arithmetic, and output values are unchanged.

## Measurement method

`/Users/tpeulen/.hermes/cache/scratch/bff-gpu/bench_cpp.py`, CPU-only with
`IMP_BFF_GPU=off`, through the Python `predict()` API, min of five calls after
the script's warm-up. M1 Pro arm64, Release `-O3 -DNDEBUG`. The benchmark's
`f64 cpu / fp4 cpu` columns are reported below in milliseconds.

## Baseline vs accepted result

Baseline was measured at HEAD `cdca25403` before the source edit. Final was
measured after the edit and a successful rebuild. The f64 path is unchanged;
its movement is run-to-run load noise, not an intended result.

| network | rows | baseline f64 / fp4 | final f64 / fp4 |
|---|---:|---:|---:|
| 4-64-64-2 | 300 | 0.26 / 0.24 | 0.25 / 0.23 |
| 4-64-64-2 | 1,000 | 0.85 / 0.77 | 0.83 / 0.76 |
| 4-64-64-2 | 3,000 | 2.60 / 2.38 | 2.46 / 2.30 |
| 4-64-64-2 | 10,000 | 8.81 / 8.15 | 9.46 / 8.02 |
| 4-64-64-2 | 100,000 | 94.11 / 85.57 | 88.79 / 85.14 |
| 4-64-64-2 | 400,000 | 376.89 / 349.26 | 347.70 / 338.29 |
| 4-128-128-128-1 | 300 | 0.97 / 0.72 | 1.02 / 0.72 |
| 4-128-128-128-1 | 1,000 | 3.24 / 2.44 | 3.10 / 2.58 |
| 4-128-128-128-1 | 3,000 | 9.88 / 7.43 | 9.75 / 7.33 |
| 4-128-128-128-1 | 10,000 | 35.35 / 25.46 | 34.84 / 26.54 |
| 4-128-128-128-1 | 100,000 | 359.00 / 265.63 | 340.24 / 261.08 |
| 4-128-128-128-1 | 400,000 | 1,304.93 / 969.90 | 1,326.37 / 986.09 |
| 32-256-256-256-8 | 300 | 3.13 / 1.79 | 3.16 / 1.86 |
| 32-256-256-256-8 | 1,000 | 10.37 / 6.03 | 10.88 / 7.32 |
| 32-256-256-256-8 | 3,000 | 31.54 / 18.46 | 32.90 / 18.12 |
| 32-256-256-256-8 | 10,000 | 118.83 / 78.59 | 112.30 / 62.21 |
| 32-256-256-256-8 | 100,000 | 1,166.68 / 650.52 | 1,081.39 / 640.67 |
| 32-256-256-256-8 | 400,000 | 5,624.29 / 3,290.72 | 6,969.44 / 3,543.09 |

The focused same-process run gave a clearer input-copy signal: final FP4
times were 8.133, 25.249, and 61.053 ms at 10,000 rows for the three nets,
and 338.207, 970.496, and 2,503.439 ms at 400,000 rows. The prescribed
whole-table run is retained as the official before/after record because its
f64 and FP4 timings are collected in one common script, but large-batch
figures should be re-run on a quiet machine before treating small percentage
differences as durable.

## Verification

After the accepted edit:

- `ninja` completed and rebuilt the bff test snippets, library, executable,
  and Python extension.
- The requested CPU-only gate passed: **168 passed, 71 skipped** in 93.77 s.
- The standalone `test_mlp_core` and `test_fp4_kernels` executables passed;
  the FP4 PyTorch parity test passed **3/3**.
- The committed FP4 fingerprint remained `64407c22cd039ecf`.

## Tried and reverted

An attempt to add a borrowing mode to `MlpCore::forward` and remove the
float64 `model_predict` input copy caused a segmentation fault in both the
Python FP4 training test and the standalone `test_mlp_core` executable. It
was reverted before the accepted benchmark and is not part of the final
diff. The safe FP4-only borrowing change does not touch the float64 core.

## Resume points

- Revisit float64 input-copy elimination only with a separate value-only
  forward routine that cannot invalidate the workspace required by backward;
  the first shared-workspace attempt is unsafe.
- A quiet-machine, CPU-time-clock benchmark and x86 CI measurements are still
  needed before changing MatGemm blocking or architecture-specific tile
  sizes. No AVX2/AVX-512 code was changed or locally executed.
- The shared board claim could not be posted from this session because
  `okf/agent-board.md` is a symlink into `../tttrlib`, outside the permitted
  writable roots.
