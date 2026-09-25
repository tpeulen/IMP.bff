"""Priors (arXiv:2608.21061 Sec. III D) and the L-BFGS fit of the log posterior."""

import numpy as np
import pytest

import IMP.bff as bff


def _model(seed=0):
    rng = np.random.default_rng(seed)
    model = bff.FRETLandscapeModel(4.0, 8.0, 30, 8, 6.0)
    offs, times, chans = [0], [], []
    for _ in range(6):
        n = 150
        times.append(np.cumsum(rng.exponential(1 / 20.0, n)))
        chans.append(rng.integers(0, 2, n))
        offs.append(offs[-1] + n)
    model.set_photons(np.concatenate(times), np.concatenate(chans).astype(np.int32),
                      np.asarray(offs, np.int32))
    theta = np.asarray(model.pack_parameters(list(rng.normal(0, 1, 8)), 1.0, [12.0, 10.0],
                                             [1.5, 3.0]))
    return model, theta


def test_prior_gradient_and_precision():
    model, theta = _model()
    model.set_roughness_weight(0.02)
    model.set_anchor_sigma(0.7)
    model.set_background_prior([1.6, 4.0], [0.16, 0.4])
    p = len(theta)
    g = np.asarray(model.log_prior_gradient(theta))
    h = np.asarray(model.prior_precision(theta)).reshape(p, p)
    eps = 1e-6
    for i in range(p):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fd = (model.log_prior(tp) - model.log_prior(tm)) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=1e-8)
        fdh = -(np.asarray(model.log_prior_gradient(tp)) -
                np.asarray(model.log_prior_gradient(tm))) / (2 * eps)
        np.testing.assert_allclose(h[i], fdh, rtol=1e-5, atol=1e-6)
    # the paper's forms, by hand
    hs = model.get_knots()[1] - model.get_knots()[0]
    mu = theta[:8]
    rough = 0.02 * np.sum(((mu[2:] - 2 * mu[1:-1] + mu[:-2]) / hs ** 2) ** 2)
    anch = 0.5 * (mu.mean() / 0.7) ** 2
    beta = np.exp(theta[-2:])
    r = beta / np.array([1.6, 4.0])
    gam = np.sum((np.array([1.6, 4.0]) / np.array([0.16, 0.4])) ** 2 * (r - np.log(r)))
    assert model.log_prior(theta) == pytest.approx(-(rough + anch + gam), rel=1e-12)
    lp = model.log_posterior(theta)
    assert lp == pytest.approx(model.log_likelihood(theta) + model.log_prior(theta), rel=1e-12)


@pytest.mark.skipif(
    not bff.PhotonExperiment.get_available(), reason="built without tttrlib"
)
def test_fit_climbs_to_a_stationary_point():
    model, theta = _model(1)
    model.set_background_prior([1.6, 4.0], [0.16, 0.4])
    opts = bff.FRETLandscapeFitOptions()
    # random photons carry no landscape: D drifts along a flat valley, which
    # takes a few hundred iterations to settle
    opts.max_iterations = 1500
    opts.patience = 50
    opts.min_delta = 1e-9
    fit = model.fit(theta, opts)
    assert fit.get_log_posterior() > model.log_posterior(theta)
    th = np.asarray(fit.get_theta())
    assert fit.get_log_posterior() == pytest.approx(model.log_posterior(th), rel=1e-12)
    g0 = np.abs(model.log_posterior_gradient(theta)).max()
    g1 = np.abs(model.log_posterior_gradient(th)).max()
    assert g1 < 1e-3 * g0
    assert fit.get_status() in ("patience", "converged")
    hist = np.asarray(fit.get_history())
    assert np.all(np.diff(hist) >= 0)


@pytest.mark.skipif(
    not bff.PhotonExperiment.get_available(), reason="built without tttrlib"
)
def test_fixed_parameters_stay_put():
    model, theta = _model(2)
    opts = bff.FRETLandscapeFitOptions()
    opts.max_iterations = 20
    opts.fixed = [model.get_diffusion_index()]
    fit = model.fit(theta, opts)
    th = np.asarray(fit.get_theta())
    assert th[model.get_diffusion_index()] == theta[model.get_diffusion_index()]
    assert np.any(th != theta)
