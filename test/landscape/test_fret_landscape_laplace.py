"""Laplace uncertainties from the empirical Fisher information (arXiv:2608.21061 Sec. III E)."""

import numpy as np
import pytest

import IMP.bff as bff


def _setup():
    model = bff.FRETLandscapeModel(3.75, 8.75, 40, 10, 6.0)
    k = np.asarray(model.get_knots())
    z = (k - 5.9) / 1.4
    theta = np.asarray(model.pack_parameters(list(3.0 * (z ** 2 - 1) ** 2 - 0.5 * z), 1.5,
                                             [24.0, 24.0], [1.6, 4.0]))
    model.set_photons(model.simulate(theta, 16, 60.0, 1e-3, 4))
    model.set_background_prior([1.6, 4.0], [0.16, 0.4])
    return model, theta


def test_scores_sum_to_the_gradient():
    model, theta = _setup()
    p = model.get_n_parameters()
    s = np.asarray(model.trace_scores(theta)).reshape(model.get_n_traces(), p)
    np.testing.assert_allclose(s.sum(0), model.log_likelihood_gradient(theta),
                               rtol=1e-9, atol=1e-9 * np.abs(s).max())


def test_laplace_is_fisher_plus_prior():
    model, theta = _setup()
    p, kk = model.get_n_parameters(), model.get_n_knots()
    lap = model.laplace(theta)
    s = np.asarray(model.trace_scores(theta)).reshape(-1, p)
    h = s.T @ s + np.asarray(model.prior_precision(theta)).reshape(p, p)
    np.testing.assert_allclose(np.asarray(lap.get_precision()).reshape(p, p), h, rtol=1e-10)
    cov = np.asarray(lap.get_covariance()).reshape(p, p)
    np.testing.assert_allclose(cov @ h, np.eye(p), atol=1e-6)
    # Eq. 22 with the centred basis
    phi = np.asarray(model.get_basis()).reshape(-1, kk)
    pc = phi - phi.mean(0)
    band = np.sqrt(np.einsum("ik,kl,il->i", pc, cov[:kk, :kk], pc))
    np.testing.assert_allclose(lap.get_landscape_sigma(), band, rtol=1e-8)
    u = np.asarray(model.get_landscape(theta))
    np.testing.assert_allclose(lap.get_landscape(), u - u.mean(), atol=1e-12)
    # delta method on log D
    d = model.get_diffusion(theta)
    assert lap.get_diffusion() == pytest.approx(d)
    assert lap.get_diffusion_sigma() == pytest.approx(d * np.sqrt(cov[kk, kk]))
    # the barrier: global minimum and top between the two wells
    x = np.asarray(model.get_grid())
    assert abs(lap.get_x_min() - 7.3) < 0.3 and abs(lap.get_x_barrier() - 5.9) < 0.3
    i0 = np.argmin(np.abs(x - lap.get_x_min()))
    i1 = np.argmin(np.abs(x - lap.get_x_barrier()))
    assert lap.get_barrier() == pytest.approx(u[i1] - u[i0], abs=1e-9)
    assert 0 < lap.get_barrier_sigma() < 5
    # a uniform shift of the knots changes neither the band nor the barrier
    shifted = theta.copy()
    shifted[:kk] += 3.0
    lap2 = model.laplace(shifted)
    np.testing.assert_allclose(lap2.get_landscape_sigma(), lap.get_landscape_sigma(), rtol=1e-6)
    assert lap2.get_barrier() == pytest.approx(lap.get_barrier(), abs=1e-9)
    # explicit positions
    lap3 = model.laplace(theta, 4.5, 5.9)
    assert lap3.get_x_min() == 4.5 and lap3.get_x_barrier() == 5.9
