"""FP4 speed gates: inference and training against float64, through the module.

Run with the IMP environment (not collected by pytest):

    python test/bench_neural_net_fp4.py [--reps N] [--quick]

Prints markdown tables (median over repeated runs, microseconds a predict()
call; seconds an epoch) for okf/neural-net.md:

* G1, inference: `NeuralNet.predict` (float64) against
  `QuantizedNeuralNet.predict` for fp4 / mxfp4 / nvfp4, W4A8 and W4A4, at
  batch 1, 32 and 256 on the 24-256-256-128-8 tanh net (the HMM surrogate
  with 2 states x 2 streams) and 24-256-256-128-15 (3 states x 2 streams).
  bff has no float32 network path; float64 is the only float baseline.
* G2, training: `train_neural_net_arrays` epoch time, float64 against the
  NVFP4 / MXFP4 recipe with every feature on (2-D weight scales, Hadamard
  on wgrad, stochastic rounding, first and last layer float64), batch 200:
  the surrogate's training shape (400 x 24 -> 8, hidden 256-256-128), a
  64-wide 3-hidden-layer regression (3000 x 4 -> 2), and a tiny
  2-16-16-1 net where fixed per-step costs dominate (reported, not a gate).

Both paths run on one thread in the IMP build (no OpenMP there); the
kernel variant is `QuantizedNeuralNet.get_kernel_name()`; the machine's
load average is printed with the results (record it with the numbers).
Timings only: nothing is asserted. Recorded results: okf/neural-net.md,
"Speed gates" (2026-09-24) and "Speed gates, third pass" (2026-09-25).
"""

import argparse
import os
import platform
import statistics
import time

import numpy as np

import IMP.bff

FORMATS = ("fp4", "mxfp4", "nvfp4")


def _median_us(fn, reps, inner):
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - t0) / inner * 1e6)
    return statistics.median(out)


def _net(n_out, seed=1):
    """A tanh 24-256-256-128-n_out network (one epoch of training on noise:
    the weights only need realistic magnitudes)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 24))
    Y = rng.normal(size=(256, n_out))
    o = IMP.bff.NeuralNetTrainOptions()
    o.hidden_layer_sizes = [256, 256, 128]
    o.activation = "tanh"
    o.max_iter = 1
    o.early_stopping = False
    return IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(X, Y, o).get_network())


def bench_inference(reps):
    rows = []
    for n_out in (8, 15):
        net = _net(n_out)
        qs = {(f, qa): IMP.bff.QuantizedNeuralNet(net, f, qa) for f in FORMATS for qa in (False, True)}
        for b in (1, 32, 256):
            X = np.random.default_rng(b).normal(size=(b, 24)).ravel().tolist()
            inner = max(1, 2000 // (b * 8))
            t64 = _median_us(lambda: net.predict(X, b), reps, inner)
            cells = []
            for f in FORMATS:
                for qa in (False, True):
                    q = qs[(f, qa)]
                    cells.append(_median_us(lambda: q.predict(X, b), reps, inner))
            rows.append((n_out, b, t64, cells))
    print("\n### G1 inference, us a predict() call (median of %d)\n" % reps)
    hdr = " | ".join("%s %s" % (f, a) for f in FORMATS for a in ("W4A8", "W4A4"))
    print("| net | batch | float64 | %s | worst FP4 / float64 |" % hdr)
    print("|---|---|---|%s|---|" % "|".join("---" for _ in range(6)))
    for n_out, b, t64, cells in rows:
        print("| 24-256-256-128-%d | %d | %.1f | %s | %.2f |" % (
            n_out, b, t64, " | ".join("%.1f" % c for c in cells), max(cells) / t64))
    return rows


def _train_opts(precision, hidden, epochs):
    o = IMP.bff.NeuralNetTrainOptions()
    o.hidden_layer_sizes = list(hidden)
    o.activation = "tanh"
    o.max_iter = epochs
    o.n_iter_no_change = 10 ** 6  # every run does all epochs
    o.seed = 0
    o.precision = precision
    return o


def _epoch_seconds(X, Y, precision, hidden, epochs, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        t = IMP.bff.train_neural_net_arrays(X, Y, _train_opts(precision, hidden, epochs))
        out.append((time.perf_counter() - t0) / t.get_number_of_epochs())
    return statistics.median(out)


def bench_training(reps, quick):
    rng = np.random.default_rng(0)
    cases = [
        ("surrogate 400x24->8, 256-256-128", rng.normal(size=(400, 24)), rng.normal(size=(400, 8)),
         (256, 256, 128), 10 if quick else 40),
        ("regression 3000x4->2, 64-64-64", rng.uniform(-2, 2, size=(3000, 4)), rng.normal(size=(3000, 2)),
         (64, 64, 64), 5 if quick else 20),
        ("tiny 2000x2->1, 16-16 (no gate)", rng.uniform(-2, 2, size=(2000, 2)), rng.normal(size=(2000, 1)),
         (16, 16), 5 if quick else 20),
    ]
    rows = []
    for name, X, Y, hidden, epochs in cases:
        t = {p: _epoch_seconds(X, Y, p, hidden, epochs, reps) for p in ("float64", "nvfp4", "mxfp4")}
        rows.append((name, t))
    print("\n### G2 training, ms an epoch (median of %d runs; batch 200)\n" % reps)
    print("| shape | float64 | nvfp4 | mxfp4 | nvfp4 / float64 | mxfp4 / float64 |")
    print("|---|---|---|---|---|---|")
    for name, t in rows:
        print("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            name, 1e3 * t["float64"], 1e3 * t["nvfp4"], 1e3 * t["mxfp4"],
            t["nvfp4"] / t["float64"], t["mxfp4"] / t["float64"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("machine %s, kernel variant %s" % (platform.machine(), IMP.bff.QuantizedNeuralNet.get_kernel_name()))
    if hasattr(os, "getloadavg"):
        print("load average %.2f %.2f %.2f" % os.getloadavg())
    bench_inference(a.reps)
    bench_training(a.reps, a.quick)


if __name__ == "__main__":
    main()
