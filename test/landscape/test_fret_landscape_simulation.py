"""Synthetic photon streams (arXiv:2608.21061 Sec. III F) and the data-driven
initial guess (Sec. III C)."""

import numpy as np
import pytest

import IMP.bff as bff


def _two_well(model, depth=4.5, tilt=0.5):
    k = np.asarray(model.get_knots())
    z = (k - 5.9) / 1.4
    return list(depth * (z ** 2 - 1) ** 2 - tilt * z)


def _model():
    model = bff.FRETLandscapeModel(3.75, 8.75, 40, 12, 6.0)
    theta = model.pack_parameters(_two_well(model), 1.5, [24.0, 24.0], [1.6, 4.0])
    return model, np.asarray(theta)


def test_simulation_is_deterministic_and_has_the_model_rates():
    model, theta = _model()
    a = model.simulate(theta, 12, 100.0, 1e-3, 7)
    b = model.simulate(theta, 12, 100.0, 1e-3, 7)
    np.testing.assert_array_equal(a.get_times(), b.get_times())
    np.testing.assert_array_equal(a.get_channels(), b.get_channels())
    c = model.simulate(theta, 12, 100.0, 1e-3, 8)
    assert c.get_n_photons() != a.get_n_photons()
    t = np.asarray(a.get_times())
    o = np.asarray(a.get_offsets())
    assert o[0] == 0 and o[-1] == len(t) and a.get_n_traces() == 12
    for m in range(12):
        tm = t[o[m]:o[m + 1]]
        assert np.all(np.diff(tm) >= 0) and tm[0] >= 0 and tm[-1] <= 100.0
    # equilibrium rates: <lambda_c>_pi on the grid
    pi = np.asarray(bff.sqra_stationary_distribution(model.get_landscape(theta)))
    lam = np.asarray(model.get_rates(theta)).reshape(2, -1)
    expect = lam @ pi * 100.0 * 12
    counts = np.bincount(np.asarray(a.get_channels()), minlength=2)
    # dominated by the slow well populations; a loose, many-sigma bound
    np.testing.assert_allclose(counts, expect, rtol=0.2)
    assert abs(counts.sum() - expect.sum()) < 0.03 * expect.sum()


def test_initial_guess_finds_the_wells():
    model, theta = _model()
    model.set_photons(model.simulate(theta, 20, 100.0, 1e-3, 3))
    model.set_background_prior([1.6, 4.0], [0.16, 0.4])
    guess = model.initial_guess()
    th0 = np.asarray(guess.get_theta())
    assert guess.get_bin_width() in (0.5, 1.0, 2.0, 4.0)
    assert len(guess.get_bin_scores()) == 4 and len(guess.get_diffusion_scores()) == 7
    np.testing.assert_allclose(model.get_backgrounds(th0), [1.6, 4.0])
    np.testing.assert_allclose(model.get_amplitudes(th0), [24.0, 24.0], rtol=0.15)
    x = np.asarray(model.get_grid())
    u = np.asarray(model.get_landscape(th0))
    left, right = x < 5.9, x >= 5.9
    assert abs(x[left][np.argmin(u[left])] - 4.5) < 0.5
    assert abs(x[right][np.argmin(u[right])] - 7.3) < 0.5
    assert model.log_likelihood(th0) > model.log_likelihood(
        model.pack_parameters([0.0] * 12, 1.5, [24.0, 24.0], [1.6, 4.0]))
