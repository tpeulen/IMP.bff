"""The photon-by-photon forward filter and its exact gradient (arXiv:2608.21061).

* against a brute-force matrix-exponential recursion (paper Eq. 10/14) on a
  small grid;
* in the two-state limit (M = 2), against a hand-computed Markov-modulated
  Poisson likelihood with closed-form 2x2 exponentials;
* the adjoint gradient against central finite differences, for every
  parameter (knots, log D, log brightnesses, log backgrounds).
"""

import math

import numpy as np
import pytest

import IMP.bff as bff


def _random_traces(rng, n_traces, n_photons, n_channels=2, rate=5.0):
    times, chans, offs = [], [], [0]
    for m in range(n_traces):
        n = int(rng.integers(1, n_photons + 1)) if m else n_photons
        t = np.cumsum(rng.exponential(1.0 / rate, n))
        times.append(t)
        chans.append(rng.integers(0, n_channels, n))
        offs.append(offs[-1] + n)
    return np.concatenate(times), np.concatenate(chans).astype(np.int32), np.asarray(offs, np.int32)


def _model(m=12, k=6, batch=3, seed=0, n_traces=5, n_photons=40):
    rng = np.random.default_rng(seed)
    model = bff.FRETLandscapeModel(4.0, 8.0, m, k, 6.0)
    model.set_batch_size(batch)
    t, c, o = _random_traces(rng, n_traces, n_photons)
    model.set_photons(t, c, o)
    theta = model.pack_parameters(list(rng.normal(0, 1.5, k)), 0.4, [3.0, 2.5], [0.4, 0.7])
    return model, np.asarray(theta), (t, c, o)


def _brute_force(model, theta, data):
    from scipy.linalg import expm
    t, c, o = data
    m = model.get_n_grid()
    u = np.asarray(model.get_landscape(theta))
    d = model.get_diffusion(theta)
    q = np.asarray(bff.sqra_generator(u, d, model.get_grid_spacing())).reshape(m, m)
    lam = np.asarray(model.get_rates(theta)).reshape(model.get_n_channels(), m)
    kill = np.diag(lam.sum(axis=0))
    pi = np.exp(-u) / np.exp(-u).sum()
    total = 0.0
    for j in range(len(o) - 1):
        tt, cc = t[o[j]:o[j + 1]], c[o[j]:o[j + 1]]
        alpha = lam[cc[0]] * pi
        for n in range(1, len(tt)):
            alpha = lam[cc[n]] * (expm((q - kill) * (tt[n] - tt[n - 1])) @ alpha)
        total += math.log(alpha.sum())
    return total


def test_forward_filter_matches_brute_force_matrix_exponential():
    pytest.importorskip("scipy")
    model, theta, data = _model()
    ref = _brute_force(model, theta, data)
    assert model.log_likelihood(theta) == pytest.approx(ref, rel=1e-10, abs=1e-8)
    per = np.asarray(model.trace_log_likelihoods(theta))
    assert per.sum() == pytest.approx(ref, rel=1e-10, abs=1e-8)
    # batch size does not change the answer
    model.set_batch_size(1)
    assert model.log_likelihood(theta) == pytest.approx(ref, rel=1e-10, abs=1e-8)
    model.set_batch_size(64)
    np.testing.assert_allclose(model.trace_log_likelihoods(theta), per, rtol=1e-11)


def _expm2(a, t):
    """exp(a t) for a 2x2 matrix, closed form."""
    m = 0.5 * (a[0, 0] + a[1, 1])
    b = a - m * np.eye(2)
    delta = math.sqrt(0.25 * (a[0, 0] - a[1, 1]) ** 2 + a[0, 1] * a[1, 0])
    sh = math.sinh(delta * t) / delta if delta > 0 else t
    return math.exp(m * t) * (math.cosh(delta * t) * np.eye(2) + sh * b)


def test_two_state_limit_is_a_markov_modulated_poisson_process():
    """M = 2: two cells, one hop rate each way -- Gopich & Szabo's MMPP."""
    rng = np.random.default_rng(11)
    x0, x1 = 4.5, 7.0
    model = bff.FRETLandscapeModel(x0, x1, 2, 2, 6.0)
    t, c, o = _random_traces(rng, 3, 60, rate=6.0)
    model.set_photons(t, c, o)
    u1, u2, d = 0.3, -0.4, 0.8
    amp, bg = [4.0, 3.5], [0.3, 0.6]
    theta = model.pack_parameters([u1, u2], d, amp, bg)
    # by hand: the hop rates and the channel rates of the two states
    h = x1 - x0
    k12 = d / h ** 2 * math.exp(-(u2 - u1) / 2)   # 1 -> 2
    k21 = d / h ** 2 * math.exp(-(u1 - u2) / 2)   # 2 -> 1
    K = np.array([[-k12, k21], [k12, -k21]])
    e = [1 / (1 + (x0 / 6.0) ** 6), 1 / (1 + (x1 / 6.0) ** 6)]
    fr = [(0.97, 0.08), (0.03, 0.92)]
    lam = np.array([[amp[ch] * (fr[ch][0] * (1 - ei) + fr[ch][1] * ei) + bg[ch] for ei in e]
                    for ch in range(2)])
    p = np.array([k21, k12]) / (k12 + k21)
    ref = 0.0
    for j in range(len(o) - 1):
        tt, cc = t[o[j]:o[j + 1]], c[o[j]:o[j + 1]]
        alpha = lam[cc[0]] * p
        for n in range(1, len(tt)):
            alpha = lam[cc[n]] * (_expm2(K - np.diag(lam.sum(0)), tt[n] - tt[n - 1]) @ alpha)
        ref += math.log(alpha.sum())
    assert model.log_likelihood(theta) == pytest.approx(ref, rel=1e-11)


@pytest.mark.parametrize("batch", [1, 4])
def test_gradient_matches_finite_differences(batch):
    model, theta, _ = _model(m=14, k=6, batch=batch, seed=3)
    fg = np.asarray(model.log_likelihood_and_gradient(theta))
    assert fg[0] == pytest.approx(model.log_likelihood(theta), rel=1e-12)
    g = fg[1:]
    np.testing.assert_allclose(g, model.log_likelihood_gradient(theta), rtol=1e-12)
    eps = 1e-5
    fd = np.empty_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fd[i] = (model.log_likelihood(tp) - model.log_likelihood(tm)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())
    # the likelihood sees only energy differences: a uniform knot shift is flat
    assert abs(g[:model.get_n_knots()].sum()) < 1e-7 * np.abs(g).max()


def test_gradient_on_a_stiff_grid():
    """Paper-like scales: D/h^2 in the thousands, tens of photons per ms."""
    rng = np.random.default_rng(5)
    model = bff.FRETLandscapeModel(3.75, 8.75, 60, 10, 6.0)
    t, c, o = _random_traces(rng, 3, 300, rate=30.0)
    model.set_photons(t, c, o)
    knots = np.asarray(model.get_knots())
    mu = 4.0 * (((knots - 5.9) / 1.4) ** 2 - 1) ** 2 - 0.5 * (knots - 5.9) / 1.4
    theta = np.asarray(model.pack_parameters(list(mu), 1.5, [24.0, 24.0], [1.6, 4.0]))
    g = np.asarray(model.log_likelihood_gradient(theta))
    eps = 1e-5
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fd = (model.log_likelihood(tp) - model.log_likelihood(tm)) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-5 * np.abs(g).max()), i
