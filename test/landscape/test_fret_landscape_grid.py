"""The SqRA discretisation of a 1-D landscape (arXiv:2608.21061, Sec. III A).

Checks the three properties the likelihood rests on: columns of Q sum to zero,
the Boltzmann weights are its exact stationary state on the grid, and the
similarity transform with Pi^(1/2) turns Q - Lambda into a symmetric
tridiagonal matrix with constant off-diagonal D/h^2. Plus the exposed
tridiagonal eigensolver and the natural cubic spline.
"""

import numpy as np
import pytest

import IMP.bff as bff


def _landscape(m=40, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(3.75, 8.75, m)
    return x, 3.0 * np.sin(1.3 * x) + 0.5 * rng.standard_normal(m)


def test_columns_sum_to_zero_and_boltzmann_is_stationary():
    x, u = _landscape()
    m, h, d = len(u), x[1] - x[0], 1.5
    q = np.asarray(bff.sqra_generator(u, d, h)).reshape(m, m)
    np.testing.assert_allclose(q.sum(axis=0), 0.0, atol=1e-9 * np.abs(q).max())
    pi = np.asarray(bff.sqra_stationary_distribution(u))
    np.testing.assert_allclose(pi, np.exp(-u) / np.exp(-u).sum(), rtol=1e-12)
    assert np.abs(q @ pi).max() < 1e-12 * np.abs(q).max()
    # detailed balance, entry by entry
    flux = q * pi[None, :]
    np.testing.assert_allclose(flux, flux.T, atol=1e-12 * np.abs(flux).max())
    # tridiagonal, reflecting
    assert np.count_nonzero(np.triu(q, 2)) == 0 and np.count_nonzero(np.tril(q, -2)) == 0


def test_symmetrised_killed_generator():
    x, u = _landscape(25, seed=3)
    m, h, d = len(u), x[1] - x[0], 0.7
    kill = np.linspace(2.0, 9.0, m)
    q = np.asarray(bff.sqra_generator(u, d, h)).reshape(m, m)
    pi = np.asarray(bff.sqra_stationary_distribution(u))
    a = np.diag(pi ** -0.5) @ (q - np.diag(kill)) @ np.diag(pi ** 0.5)
    np.testing.assert_allclose(a, a.T, atol=1e-10 * np.abs(a).max())
    np.testing.assert_allclose(np.diag(a, 1), d / h ** 2, rtol=1e-12)
    diag = np.asarray(bff.sqra_symmetric_diagonal(u, d, h, kill))
    np.testing.assert_allclose(diag, np.diag(a), rtol=1e-12)


def test_tridiagonal_eigensolver_matches_numpy():
    rng = np.random.default_rng(7)
    for m in (1, 2, 3, 17, 120):
        dg = rng.standard_normal(m) * 5
        off = rng.standard_normal(m - 1)
        es = bff.symmetric_tridiagonal_eigen(dg, off)
        vals = np.asarray(es.get_values())
        vecs = np.asarray(es.get_vectors()).reshape(m, m)
        t = np.diag(dg) + np.diag(off, 1) + np.diag(off, -1)
        np.testing.assert_allclose(vals, np.sort(np.linalg.eigvalsh(t))[::-1], atol=1e-10)
        np.testing.assert_allclose(vecs @ np.diag(vals) @ vecs.T, t, atol=1e-10)
        np.testing.assert_allclose(vecs.T @ vecs, np.eye(m), atol=1e-10)
        assert np.all(np.diff(vals) <= 0)


def test_natural_cubic_spline_matches_scipy():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    sp = bff.NaturalCubicSpline(3.75, 8.75, 12)
    knots = np.asarray(sp.get_knots())
    mu = np.cos(knots) * 2 + 0.1 * knots
    x = np.linspace(3.75, 8.75, 301)
    ref = scipy_interp.CubicSpline(knots, mu, bc_type="natural")
    np.testing.assert_allclose(sp.evaluate(mu, x), ref(x), atol=1e-12)
    np.testing.assert_allclose(sp.derivative(mu, x), ref(x, 1), atol=1e-10)
    phi = np.asarray(sp.get_basis(x)).reshape(len(x), 12)
    np.testing.assert_allclose(phi @ mu, ref(x), atol=1e-12)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-12)  # partition of unity
    np.testing.assert_allclose(np.asarray(sp.get_basis(knots)).reshape(12, 12), np.eye(12), atol=1e-12)
    dphi = np.asarray(sp.get_basis_derivative(x)).reshape(len(x), 12)
    np.testing.assert_allclose(dphi @ mu, ref(x, 1), atol=1e-10)
