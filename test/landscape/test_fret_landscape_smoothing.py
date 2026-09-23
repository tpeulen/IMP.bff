"""Posterior over the hidden distance (arXiv:2608.21061 Appendix D):
smoothing marginals and forward-filter backward-sampling."""

import numpy as np
import pytest

import IMP.bff as bff


def _setup():
    rng = np.random.default_rng(2)
    model = bff.FRETLandscapeModel(4.0, 8.0, 10, 5, 6.0)
    n = 25
    t = np.cumsum(rng.exponential(0.2, n))
    c = rng.integers(0, 2, n).astype(np.int32)
    model.set_photons(t, c, np.array([0, n], np.int32))
    theta = np.asarray(model.pack_parameters([1.0, -1.0, 1.5, -0.5, 1.0], 0.6, [5.0, 5.0],
                                             [0.5, 0.8]))
    return model, theta, t, c


def test_marginals_match_brute_force():
    from scipy.linalg import expm
    model, theta, t, c = _setup()
    m = model.get_n_grid()
    u = np.asarray(model.get_landscape(theta))
    q = np.asarray(bff.sqra_generator(u, model.get_diffusion(theta),
                                      model.get_grid_spacing())).reshape(m, m)
    lam = np.asarray(model.get_rates(theta)).reshape(2, m)
    a = q - np.diag(lam.sum(0))
    pi = np.exp(-u) / np.exp(-u).sum()
    n = len(t)
    alpha = [lam[c[0]] * pi]
    for k in range(1, n):
        alpha.append(lam[c[k]] * (expm(a * (t[k] - t[k - 1])) @ alpha[-1]))
    chi = [None] * n
    chi[-1] = np.ones(m)
    for k in range(n - 1, 0, -1):
        chi[k - 1] = expm(a * (t[k] - t[k - 1])).T @ (lam[c[k]] * chi[k])
    ref = np.array([al * ch / (al @ ch) for al, ch in zip(alpha, chi)])
    got = np.asarray(model.posterior_marginals(theta, 0)).reshape(n, m)
    np.testing.assert_allclose(got, ref, atol=1e-10)
    np.testing.assert_allclose(model.posterior_mean_trajectory(theta, 0),
                               ref @ np.asarray(model.get_grid()), atol=1e-9)


def test_backward_samples_follow_the_marginals():
    model, theta, t, _ = _setup()
    n, m = len(t), model.get_n_grid()
    x = np.asarray(model.get_grid())
    marg = np.asarray(model.posterior_marginals(theta, 0)).reshape(n, m)
    draws = np.array([model.sample_trajectory(theta, 0, s) for s in range(1500)])
    assert set(np.unique(draws)) <= set(x)
    assert np.array_equal(draws[3], model.sample_trajectory(theta, 0, 3))
    idx = np.searchsorted(x, draws)
    for k in (0, n // 2, n - 1):
        freq = np.bincount(idx[:, k], minlength=m) / len(draws)
        assert np.abs(freq - marg[k]).max() < 0.05
    # pairs, not just marginals: the lag-one joint matches the brute force
    from scipy.linalg import expm
    _, _, t, c = _setup()
    u = np.asarray(model.get_landscape(theta))
    q = np.asarray(bff.sqra_generator(u, model.get_diffusion(theta),
                                      model.get_grid_spacing())).reshape(m, m)
    lam = np.asarray(model.get_rates(theta)).reshape(2, m)
    a = q - np.diag(lam.sum(0))
    pi = np.exp(-u) / np.exp(-u).sum()
    alpha = lam[c[0]] * pi
    for k in range(1, 6):
        alpha = lam[c[k]] * (expm(a * (t[k] - t[k - 1])) @ alpha)
    chi = np.ones(m)
    for k in range(n - 1, 6, -1):
        chi = expm(a * (t[k] - t[k - 1])).T @ (lam[c[k]] * chi)
    s6 = expm(a * (t[6] - t[5]))
    joint = alpha[:, None] * s6.T * (lam[c[6]] * chi)[None, :]
    joint /= joint.sum()
    emp = np.zeros((m, m))
    np.add.at(emp, (idx[:, 5], idx[:, 6]), 1.0)
    emp /= len(draws)
    assert np.abs(emp - joint).max() < 0.05
