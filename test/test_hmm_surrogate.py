"""IMP.bff.HmmSurrogate -- the amortised neural estimator for H2MM.

Moved here from tttrlib (tttrlib stays ML-free; learned models live in
imp.bff even when their inputs are photons), with its tests. The feature
extractor must reproduce the NumPy reference below exactly: a surrogate
trained in Python has to be executable here and vice versa. The places a
naive port silently diverges are all pinned: histogram bin-edge handling and
density normalisation, quantile interpolation, and population-versus-sample
standard deviation.

Skipped when IMP.bff was built without tttrlib (no HmmSurrogate) or tttrlib's
Python module is missing.
"""
import json
import os
import tempfile
import unittest

import numpy as np

import IMP
import IMP.test
import IMP.bff

try:
    import tttrlib
except ImportError:
    tttrlib = None

HAVE = hasattr(IMP.bff, "HmmSurrogate") and tttrlib is not None
SKIP = "IMP.bff built without tttrlib, or tttrlib's Python module missing"

_AC_LAGS = np.array([1, 2, 4, 8, 16, 32], dtype=np.int64)
_WINDOW = 12


def _reference_features(times, streams_list, n_streams):
    """Feature vector computed the way the NumPy/numba implementation does."""
    streams = np.concatenate([np.asarray(s, dtype=np.int64) for s in streams_list])
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in streams_list])]).astype(np.int64)

    scale = 1.0 / max(n_streams - 1, 1)
    n = streams.shape[0]
    mu = float((streams * scale).sum() / n) if n else 0.0

    # The same sliding two-pointer sum as the kernel; a fresh slice.mean()
    # per photon accumulates float error differently and moves edge values.
    loc = np.empty(n)
    for b in range(offsets.shape[0] - 1):
        s, e = int(offsets[b]), int(offsets[b + 1])
        a = s
        c = e if s + _WINDOW + 1 > e else s + _WINDOW + 1
        wsum = 0.0
        for k in range(a, c):
            wsum += streams[k] * scale
        for j in range(s, e):
            na = s if j - _WINDOW < s else j - _WINDOW
            nc = e if j + _WINDOW + 1 > e else j + _WINDOW + 1
            while a < na:
                wsum -= streams[a] * scale
                a += 1
            while c < nc:
                wsum += streams[c] * scale
                c += 1
            loc[j] = wsum / (c - a)

    ac = np.zeros(_AC_LAGS.shape[0])
    for li, lag in enumerate(_AC_LAGS):
        num, cnt = 0.0, 0
        for b in range(offsets.shape[0] - 1):
            s, e = int(offsets[b]), int(offsets[b + 1])
            m = e - s
            if m > lag:
                x = streams[s:e] * scale - mu
                num += float((x[: m - lag] * x[lag:]).sum()) / (m - lag)
                cnt += 1
        ac[li] = num / cnt if cnt else 0.0

    feats = [mu]
    hist, _ = np.histogram(loc, bins=10, range=(0.0, 1.0), density=True)
    feats += [float(v) for v in hist]
    feats += [float(v) for v in np.quantile(loc, [0.1, 0.25, 0.5, 0.75, 0.9])]
    feats += [float(v) for v in ac]

    dts = []
    for t in times:
        t = np.asarray(t, dtype=np.int64)
        if t.size > 1:
            dts.append(np.diff(t))
    dt = np.concatenate(dts) if dts else np.zeros(1)
    feats += [float(dt.mean()), float(dt.std())]
    return np.asarray(feats, dtype=np.float64)


def _simulate(n_bursts=40, burst_len=60, n_streams=2, seed=0, mean_gap=4):
    """A small two-state-ish dataset as (times, streams) lists."""
    rng = np.random.default_rng(seed)
    times, streams = [], []
    for _ in range(n_bursts):
        gaps = rng.poisson(mean_gap, burst_len - 1) + 1
        t = np.concatenate([[0], np.cumsum(gaps)]).astype(np.int64)
        state = rng.random(burst_len) < 0.5
        e = np.where(state, 0.75, 0.25)
        s = (rng.random(burst_len) < e).astype(np.int32)
        if n_streams > 2:
            s = rng.integers(0, n_streams, burst_len).astype(np.int32)
        times.append(t)
        streams.append(s)
    return times, streams


def _hmm(times, streams, n_streams):
    engine = tttrlib.HMM()
    engine.set_bursts(
        tttrlib.VectorVectorInt64([tttrlib.VectorInt64(np.asarray(t).tolist()) for t in times]),
        tttrlib.VectorVectorInt32([tttrlib.VectorInt32(np.asarray(s).tolist()) for s in streams]),
        n_streams)
    return engine


def _random_model(n, p, rng):
    obs = rng.dirichlet(np.ones(p), size=n)
    obs = obs[np.argsort(-obs[:, 0])]
    trans = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                trans[i, j] = 10.0 ** rng.uniform(-2.8, -1.3)
        trans[i, i] = 1.0 - (trans[i].sum() - trans[i, i])
    prior = np.full(n, 1.0 / n)
    return prior, trans, obs


_TRAINED = {}


def _train_small(n_states=2, n_streams=2, seed=0):
    """A small surrogate; cached, training is the slow part of this file."""
    key = (n_states, n_streams, seed)
    if key not in _TRAINED:
        opt = IMP.bff.NeuralNetTrainOptions()
        opt.hidden_layer_sizes = [64, 64]
        opt.max_iter = 150
        opt.batch_size = 32
        opt.learning_rate = 3e-3
        _TRAINED[key] = IMP.bff.HmmSurrogate.train(
            n_states, n_streams, 200, 40, 60, 4.0, opt, seed)
    return _TRAINED[key]


@unittest.skipUnless(HAVE, SKIP)
class Tests(IMP.test.TestCase):

    # --- feature parity: the whole risk of the port -------------------------

    def test_features_match_numpy_reference(self):
        """Features equal the NumPy reference to 1e-12"""
        for n_bursts, burst_len, n_streams, seed, mean_gap in [
                (40, 60, 2, 0, 4),
                (10, 200, 2, 1, 7),
                (100, 15, 2, 2, 2),   # bursts shorter than the window and most lags
                (25, 80, 3, 3, 4),    # more than two streams
                (5, 40, 4, 4, 11),
                (3, 5, 2, 5, 1)]:     # every burst shorter than every lag but 1, 2, 4
            times, streams = _simulate(n_bursts, burst_len, n_streams, seed, mean_gap)
            expect = _reference_features(times, streams, n_streams)
            got = IMP.bff.HmmSurrogate.features_from_bursts(times, streams, n_streams)
            self.assertEqual(got.shape, (IMP.bff.HmmSurrogate.N_FEATURES,))
            np.testing.assert_allclose(got, expect, rtol=0, atol=1e-12)
            # the same through a tttrlib.HMM's CSR layout
            via_hmm = IMP.bff.HmmSurrogate.features(_hmm(times, streams, n_streams))
            np.testing.assert_allclose(via_hmm, expect, rtol=0, atol=1e-12)

    def test_feature_count_is_stable(self):
        """24 = 1 mean + 10 histogram + 5 quantiles + 6 lags + 2 gap statistics"""
        self.assertEqual(IMP.bff.HmmSurrogate.N_FEATURES, 24)
        self.assertEqual(IMP.bff.HmmSurrogate.FEATURES_VERSION, 1)

    def test_features_are_permutation_invariant(self):
        """Reordering bursts does not change the summary"""
        rng = np.random.default_rng(11)
        times, streams = _simulate(30, 50, 2, seed=8)
        order = rng.permutation(len(times))
        a = IMP.bff.HmmSurrogate.features_from_bursts(times, streams, 2)
        b = IMP.bff.HmmSurrogate.features_from_bursts(
            [times[i] for i in order], [streams[i] for i in order], 2)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    # --- encode / decode ------------------------------------------------------

    def test_encode_decode_round_trip(self):
        """decode(encode(m)) returns the same model, up to the log10 floor"""
        for n, p in [(2, 2), (3, 2), (2, 3), (4, 3)]:
            rng = np.random.default_rng(4)
            prior, trans, obs = _random_model(n, p, rng)
            vec = IMP.bff.HmmSurrogate.encode_arrays(
                prior.tolist(), trans.ravel().tolist(), obs.ravel().tolist())
            self.assertEqual(len(vec), IMP.bff.HmmSurrogate.n_targets(n, p))
            back = IMP.bff.HmmSurrogate.decode_arrays(vec, n, p)
            np.testing.assert_allclose(np.reshape(back.get_obs(), (n, p)), obs,
                                       rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(np.reshape(back.get_trans(), (n, n)), trans,
                                       rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(back.get_prior(), prior, rtol=1e-9, atol=1e-9)

    def test_decode_produces_valid_stochastic_model(self):
        """Even from noise, decode returns normalised rows"""
        rng = np.random.default_rng(6)
        n, p = 3, 2
        vec = rng.normal(size=IMP.bff.HmmSurrogate.n_targets(n, p))
        m = IMP.bff.HmmSurrogate.decode_arrays(vec.tolist(), n, p)
        obs = np.reshape(m.get_obs(), (n, p))
        trans = np.reshape(m.get_trans(), (n, n))
        np.testing.assert_allclose(obs.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(m.get_prior()), 1.0, atol=1e-12)
        self.assertTrue((trans >= 0).all() and (obs >= 0).all())

    def test_decode_rejects_wrong_length(self):
        """A target vector of the wrong length is refused"""
        with self.assertRaises(IMP.ValueException):
            IMP.bff.HmmSurrogate.decode_arrays([0.1, 0.2], 3, 2)

    def test_encode_is_label_invariant(self):
        """Permuting state labels does not change the encoding"""
        rng = np.random.default_rng(13)
        prior, trans, obs = _random_model(3, 2, rng)
        perm = np.array([2, 0, 1])

        def enc(pr, tr, ob):
            return np.asarray(IMP.bff.HmmSurrogate.encode_arrays(
                pr.tolist(), tr.ravel().tolist(), ob.ravel().tolist()))

        np.testing.assert_allclose(
            enc(prior, trans, obs),
            enc(prior[perm], trans[np.ix_(perm, perm)], obs[perm]),
            rtol=0, atol=1e-12)

    # --- training and estimation -----------------------------------------------

    def test_generate_training_set_shapes(self):
        """The simulated training set has the documented shape and varies"""
        X, Y = IMP.bff.HmmSurrogate.generate_training_set(
            2, 2, n_samples=12, n_bursts=20, burst_len=40, seed=3)
        self.assertEqual(X.shape, (12, IMP.bff.HmmSurrogate.N_FEATURES))
        self.assertEqual(Y.shape, (12, IMP.bff.HmmSurrogate.n_targets(2, 2)))
        self.assertTrue(np.isfinite(X).all() and np.isfinite(Y).all())
        self.assertGreater(X.std(axis=0).max(), 0)
        X2, _ = IMP.bff.HmmSurrogate.generate_training_set(
            2, 2, n_samples=12, n_bursts=20, burst_len=40, seed=3)
        np.testing.assert_array_equal(X, X2)

    def test_training_produces_usable_surrogate(self):
        """A trained surrogate has the right shape and predicts a valid model"""
        s = _train_small()
        self.assertEqual(s.get_n_states(), 2)
        self.assertEqual(s.get_n_streams(), 2)
        self.assertEqual(s.get_features_version(), IMP.bff.HmmSurrogate.FEATURES_VERSION)
        self.assertEqual(s.get_net().get_n_inputs(), IMP.bff.HmmSurrogate.N_FEATURES)
        self.assertEqual(s.get_net().get_n_outputs(), IMP.bff.HmmSurrogate.n_targets(2, 2))
        self.assertGreater(len(s.get_loss_curve()), 0)

        times, streams = _simulate(40, 60, 2, seed=77)
        m = s.predict_model(times, streams)
        self.assertIsInstance(m, tttrlib.HmmModel)
        np.testing.assert_allclose(m.obs_np.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(m.trans_np.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(m.n_phot, sum(len(t) for t in times))

        # the same estimate from a loaded tttrlib.HMM
        m2 = s.predict_hmm(_hmm(times, streams, 2))
        np.testing.assert_allclose(m2.obs_np, m.obs_np, rtol=0, atol=1e-14)

    def test_predicted_model_is_accepted_by_tttrlib(self):
        """tttrlib's HMM scores the prediction and EM polishes it without loss"""
        s = _train_small()
        times, streams = _simulate(40, 60, 2, seed=78)
        engine = _hmm(times, streams, 2)
        start = s.predict_hmm(engine)
        ev = engine.evaluate(start)
        self.assertTrue(np.isfinite(ev.loglik))
        fit = engine.optimize(start, 200)
        self.assertTrue(np.isfinite(fit.loglik))
        self.assertGreaterEqual(fit.loglik, ev.loglik - 1e-6)

    def test_surrogate_beats_a_constant_guess(self):
        """A trained surrogate tracks the true emission better than a fixed guess"""
        s = _train_small(seed=5)
        rng = np.random.default_rng(31)
        err_net, err_const = [], []
        for trial in range(6):
            e_lo, e_hi = sorted(rng.uniform(0.15, 0.85, 2))
            times, streams = [], []
            for _ in range(60):
                t = np.concatenate([[0], np.cumsum(rng.poisson(4, 79) + 1)]).astype(np.int64)
                state = rng.random(80) < 0.5
                e = np.where(state, e_hi, e_lo)
                streams.append((rng.random(80) < e).astype(np.int32))
                times.append(t)
            pred = np.sort(np.reshape(
                s.predict_from_bursts([t.tolist() for t in times],
                                      [x.tolist() for x in streams]).get_obs(),
                (2, 2))[:, 1])
            truth = np.sort([e_lo, e_hi])
            err_net.append(np.abs(pred - truth).mean())
            err_const.append(np.abs(np.array([0.5, 0.5]) - truth).mean())
        self.assertLess(np.mean(err_net), np.mean(err_const))

    def test_predict_rejects_wrong_stream_count(self):
        """A dataset with another stream count is refused"""
        s = _train_small()
        times, streams = _simulate(10, 40, 3, seed=9)
        with self.assertRaises(IMP.ValueException) as cm:
            s.predict_hmm(_hmm(times, streams, 3))
        self.assertIn("n_streams", str(cm.exception))

    # --- serialisation -----------------------------------------------------------

    def test_json_round_trip(self):
        """from_json_string(to_json_string()) predicts identically"""
        s = _train_small()
        times, streams = _simulate(20, 50, 2, seed=44)
        back = IMP.bff.HmmSurrogate.from_json_string(s.to_json_string())
        np.testing.assert_allclose(back.predict_model(times, streams).obs_np,
                                   s.predict_model(times, streams).obs_np,
                                   rtol=0, atol=1e-12)
        self.assertEqual(len(back.get_loss_curve()), 0)

    def test_json_file_round_trip_and_shape(self):
        """The file holds a bff.hmm_surrogate carrying a bff.neural_net"""
        s = _train_small()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "surrogate.json")
            s.to_json_file(path)
            with open(path) as fh:
                doc = json.load(fh)
            self.assertEqual(doc["format"], "bff.hmm_surrogate")
            self.assertEqual((doc["n_states"], doc["n_streams"]), (2, 2))
            self.assertEqual(doc["net"]["format"], "bff.neural_net")
            IMP.bff.HmmSurrogate.from_json_file(path)
            self.assertEqual(json.loads(s.get_net_json())["format"], "bff.neural_net")

    def test_stale_features_version_is_rejected(self):
        """A model built for another feature layout fails loudly"""
        doc = json.loads(_train_small().to_json_string())
        doc["features_version"] = IMP.bff.HmmSurrogate.FEATURES_VERSION + 1
        with self.assertRaises(IMP.ValueException) as cm:
            IMP.bff.HmmSurrogate.from_json_string(json.dumps(doc))
        self.assertIn("features_version", str(cm.exception))

    def test_wrong_format_is_rejected(self):
        """Another document tag -- tttrlib's old one included -- is refused"""
        for tag in ("bff.neural_net", "tttrlib.hmm_surrogate"):
            doc = json.loads(_train_small().to_json_string())
            doc["format"] = tag
            with self.assertRaises(IMP.ValueException):
                IMP.bff.HmmSurrogate.from_json_string(json.dumps(doc))

    def test_net_output_width_must_match_state_count(self):
        """A net whose output width disagrees with n_states is refused"""
        doc = json.loads(_train_small().to_json_string())
        doc["n_states"] = 3
        with self.assertRaises(IMP.ValueException) as cm:
            IMP.bff.HmmSurrogate.from_json_string(json.dumps(doc))
        self.assertIn("outputs", str(cm.exception))


if __name__ == '__main__':
    IMP.test.main()
