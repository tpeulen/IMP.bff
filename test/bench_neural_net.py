"""NeuralNet / HMM-surrogate benchmark: bff's network against scikit-learn and EM.

Ported from tttrlib's benchmarks/bench_nn.py (the network moved to bff).
Not collected by pytest -- the name is bench_, not test_. Run it directly:

    <build>/setup_environment.sh python test/bench_neural_net.py [--out results.jsonl]

It answers the questions the surrogate design rests on, on the real
implementation:

1. How does train_neural_net() compare with ``sklearn.MLPRegressor`` in wall
   clock and in held-out accuracy, on identical data?
2. What does a forward pass cost -- double (MatGemm), int8
   (QuantizedNeuralNet), and the derivative passes a physics-informed fit
   makes?
3. How fast is the surrogate (feature extraction + network) against the
   Baum-Welch EM it replaces, and how much accuracy does it give up?

Sections 1 and 3 simulate with ``IMP.bff.HmmSurrogate`` and fit with
``tttrlib.HMM``, so they need an IMP.bff built with tttrlib; without it only
section 2 runs, on a synthetic regression. A benchmark, not a test: nothing
here asserts.
"""

import argparse
import json
import time

import numpy as np

import IMP.bff

# ---- problem definition ----------------------------------------------------
N_STATES = 2
N_STREAMS = 2
N_BURSTS = 150
BURST_LEN = 80
MEAN_DT = 4.0
SEED = 12345
N_TRAIN = 400
HIDDEN = [256, 256, 128]

_RECORDS = []


def timeit(f, repeat=5, warmup=1):
    """Best, median and all wall-clock times of `f()` in seconds."""
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts)), ts


def bench(name, f, repeat=5, warmup=1, n_items=None, unit="items", **extra):
    best, median, _ = timeit(f, repeat, warmup)
    rate = "" if not n_items else f"   {n_items / best:,.0f} {unit}/s"
    print(f"  {name:<34} best {best * 1e3:9.3f} ms  median {median * 1e3:9.3f} ms{rate}")
    _RECORDS.append(dict(name=name, best_s=best, median_s=median, n_items=n_items,
                         unit=unit, **extra))
    return best


def _train_options(max_iter=200):
    opt = IMP.bff.NeuralNetTrainOptions()
    opt.hidden_layer_sizes = HIDDEN
    opt.max_iter = max_iter
    opt.batch_size = 200
    opt.learning_rate = 1e-3
    opt.early_stopping = True
    opt.seed = SEED
    return opt


def _net(X, Y, opt):
    return IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(X, Y, opt).get_network())


def _predict(net, X):
    return np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(
        X.shape[0], net.get_n_outputs())


def _has_surrogate():
    try:
        import tttrlib  # noqa: F401
    except ImportError:
        return False
    return hasattr(IMP.bff, "HmmSurrogate")


def _simulate_dataset(e_lo, e_hi, k, rng):
    """A two-state kinetic dataset loaded into a tttrlib HMM engine."""
    import tttrlib
    times, streams = [], []
    for _ in range(N_BURSTS):
        t = np.concatenate([[0], np.cumsum(rng.poisson(MEAN_DT, BURST_LEN - 1) + 1)]).astype(np.int64)
        s = np.empty(BURST_LEN, dtype=np.int32)
        state = rng.random() < 0.5
        for j in range(BURST_LEN):
            if j and rng.random() < k:
                state = not state
            s[j] = rng.random() < (e_hi if state else e_lo)
        times.append(t)
        streams.append(s)
    engine = tttrlib.HMM()
    engine.set_bursts(
        tttrlib.VectorVectorInt64([tttrlib.VectorInt64(t.tolist()) for t in times]),
        tttrlib.VectorVectorInt32([tttrlib.VectorInt32(s.tolist()) for s in streams]),
        N_STREAMS)
    return engine


def training(X, Y, Xte, Yte):
    print("1. training: bff vs scikit-learn, identical data")
    opt = _train_options()
    bench("bff train_neural_net", lambda: _net(X, Y, opt), repeat=3, warmup=0,
          n_items=len(X), unit="samples", hidden=HIDDEN)
    net = _net(X, Y, opt)
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn not installed; skipped")
        return net
    import warnings
    xs, ys = StandardScaler().fit(X), StandardScaler().fit(Y)

    def fit_sklearn():
        m = MLPRegressor(hidden_layer_sizes=tuple(HIDDEN), activation="relu",
                         max_iter=opt.max_iter, early_stopping=True, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(xs.transform(X), ys.transform(Y) if Y.shape[1] > 1 else ys.transform(Y).ravel())
        return m

    bench("sklearn MLPRegressor.fit", fit_sklearn, repeat=3, warmup=0,
          n_items=len(X), unit="samples", hidden=HIDDEN)
    mlp = fit_sklearn()
    pred = mlp.predict(xs.transform(Xte)).reshape(len(Xte), -1)
    mae_bff = float(np.abs(_predict(net, Xte) - Yte).mean())
    mae_skl = float(np.abs(ys.inverse_transform(pred) - Yte).mean())
    print(f"  held-out MAE   bff={mae_bff:.5f}   sklearn={mae_skl:.5f}")
    _RECORDS.append(dict(name="heldout_mae", bff=mae_bff, sklearn=mae_skl))
    return net


def forward(net, X):
    print("2. forward and derivative passes, batch %d, %s" % (len(X), HIDDEN))
    xl = X.ravel().tolist()
    n = len(X)
    q = IMP.bff.QuantizedNeuralNet(net)
    bench("predict (double, MatGemm)", lambda: net.predict(xl, n), n_items=n, unit="rows")
    bench("predict (int8)", lambda: q.predict(xl, n), n_items=n, unit="rows")
    V = np.ones_like(X)
    bench("predict_derivatives order 2", lambda: net.predict_derivatives(X, V, 2),
          n_items=n, unit="rows")
    dY = np.ones((n, net.get_n_outputs()))
    bench("backward", lambda: net.backward(X, dY), n_items=n, unit="rows")
    bench("backward_derivatives order 2",
          lambda: net.backward_derivatives(X, V, dY, dY, dY), n_items=n, unit="rows")
    exact = _predict(net, X)
    approx = np.asarray(q.predict(xl, n)).reshape(exact.shape)
    err = float(np.abs(approx - exact).max() / np.abs(exact).max())
    n_w = sum(net.get_layer_weights(i).size for i in range(net.get_n_layers()))
    print(f"  int8: {err:.2e} relative error, weights {8 * n_w} -> {q.get_weight_bytes()} bytes")
    _RECORDS.append(dict(name="int8_error", relative=err))


def surrogate_vs_em(net):
    print("3. surrogate vs Baum-Welch EM")
    surrogate = IMP.bff.HmmSurrogate(net.to_msgpack(), N_STATES, N_STREAMS)
    rng = np.random.default_rng(SEED)
    engine = _simulate_dataset(0.25, 0.75, 0.02, rng)
    n_ph = engine.get_n_photons()
    t_feat = bench("extract features", lambda: IMP.bff.HmmSurrogate.features(engine),
                   repeat=7, n_items=n_ph, unit="photons")
    t_sur = bench("surrogate predict", lambda: surrogate.predict_hmm(engine),
                  repeat=7, n_items=n_ph, unit="photons")
    t_em = bench("EM fit", lambda: engine.fit(N_STATES, 1, 500, 1e-7, SEED, True, False),
                 repeat=3, warmup=0, n_items=n_ph, unit="photons")
    print(f"  surrogate is {t_em / t_sur:.1f}x faster than EM "
          f"({t_sur * 1e3:.2f} ms vs {t_em * 1e3:.1f} ms; features {t_feat * 1e3:.2f} ms)")
    # Accuracy against ground truth, per trial. Read it carefully: EM maximises
    # likelihood, which is not closeness to the truth on finite data -- on
    # barely separated states the surface is flat and the MLE drifts, while the
    # surrogate falls back toward its training prior. An averaged "win" for
    # the surrogate is dominated by those hard trials; do not quote it as
    # "more accurate than EM".
    rows = []
    for trial in range(8):
        r = np.random.default_rng(SEED + 7000 + trial)
        e_lo, e_hi = sorted(r.uniform(0.15, 0.85, 2))
        eng = _simulate_dataset(e_lo, e_hi, 0.02, r)
        truth = np.array([e_lo, e_hi])
        m_em = eng.fit(N_STATES, 1, 500, 1e-7, SEED, True, False)
        e_sur = float(np.abs(np.sort(surrogate.predict_hmm(eng).obs_np[:, 1]) - truth).mean())
        e_em = float(np.abs(np.sort(m_em.obs_np[:, 1]) - truth).mean())
        rows.append({"separation": float(e_hi - e_lo), "surrogate": e_sur, "em": e_em})
    print(f"  {'separation':>10} {'surrogate':>10} {'EM':>10}")
    for row in sorted(rows, key=lambda d: d["separation"]):
        print(f"  {row['separation']:>10.3f} {row['surrogate']:>10.4f} {row['em']:>10.4f}")
    n_em = sum(1 for row in rows if row["em"] < row["surrogate"])
    print(f"  EM more accurate in {n_em}/{len(rows)} trials")
    _RECORDS.append(dict(name="fret_error", trials=rows, em_better=n_em, speedup=t_em / t_sur))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="append the records to this JSON-lines file")
    args = parser.parse_args()

    if _has_surrogate():
        gen = lambda n, seed: IMP.bff.HmmSurrogate.generate_training_set(  # noqa: E731
            N_STATES, N_STREAMS, n_samples=n, n_bursts=N_BURSTS, burst_len=BURST_LEN,
            mean_dt=MEAN_DT, seed=seed)
        print("0. training-set generation")
        bench("HmmSurrogate.generate_training_set", lambda: gen(N_TRAIN, SEED), repeat=3,
              warmup=0, n_items=N_TRAIN, unit="datasets")
        X, Y = gen(N_TRAIN, SEED)
        Xte, Yte = gen(100, SEED + 999)
    else:
        print("IMP.bff without tttrlib: a synthetic regression stands in for the surrogate's")
        rng = np.random.default_rng(SEED)
        X = rng.uniform(-2, 2, size=(N_TRAIN, 6))
        Y = np.column_stack([np.sin(X[:, 0]) * np.cos(X[:, 1]), X[:, 2] * X[:, 3] - X[:, 4]])
        Xte = rng.uniform(-2, 2, size=(100, 6))
        Yte = np.column_stack([np.sin(Xte[:, 0]) * np.cos(Xte[:, 1]), Xte[:, 2] * Xte[:, 3] - Xte[:, 4]])
    print(f"  training set: X{X.shape} Y{Y.shape}")
    net = training(X, Y, Xte, Yte)
    forward(net, np.ascontiguousarray(np.tile(Xte, (5, 1))))
    if _has_surrogate():
        surrogate_vs_em(net)
    if args.out:
        with open(args.out, "a") as fh:
            for rec in _RECORDS:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
