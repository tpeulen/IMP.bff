"""Parameters, gradient, priors, fit and Laplace of a FRET network model."""

import numpy as np
import pytest

import IMP.bff as bff
from test_fret_network_filter import _dyes, _instrument, _data


def _net(landscape=False, nb=8, seed=0, two=False):
    if landscape:
        hp = bff.FRETHiddenProcess(40.0, 70.0, 12, 5)
        hp.set_landscape([2.0, 0.0, 1.0, -0.5, 1.5])
        hp.set_diffusion(300.0)
    else:
        hp = bff.FRETHiddenProcess(2)
        hp.set_rate(0, 1, 6.0)
        hp.set_rate(1, 0, 4.0)
    net = bff.FRETNetworkModel(hp)
    rng = np.random.default_rng(seed)
    for p in range(2 if two else 1):
        d, a = _dyes(True)
        m = bff.FRETMeasurement("p%d" % p, d, a, 55.0)
        m.set_instrument(_instrument(nb))
        if landscape:
            m.set_distance_map([45.0, 60.0 + 5 * p, 50.0], 1.0)
        else:
            m.set_state_distance(0, 45.0, [-3.0, 0.0, 3.0], [0.25, 0.5, 0.25])
            m.set_state_distance(1, 70.0)
        net.add_measurement(m, _data(rng, nb, n_seg=3, n_ph=15))
    return net


def _fd(f, theta, h=1e-6):
    g = np.empty_like(theta)
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        g[j] = (f(tp) - f(tm)) / (2 * h)
    return g


@pytest.mark.parametrize("landscape,arrival", [(False, 1), (False, 0), (True, 1)])
def test_gradient_matches_finite_differences(landscape, arrival):
    net = _net(landscape, two=landscape)
    for i in range(net.get_n_measurements()):
        net.set_arrival_model(i, arrival)
    # free some photophysics and instrument parameters too
    for name in ["p0.donor.lifetime[bright]", "p0.acceptor.quantum_yield[bright]",
                 "p0.gain[r]", "p0.background[g]", "p0.forster_radius",
                 "p0.acceptor.initial[bleached]"]:
        net.set_parameter_free(name, True)
    theta = np.asarray(net.get_theta())
    g = np.asarray(net.log_likelihood_gradient(theta))
    fd = _fd(net.log_likelihood_at, theta)
    np.testing.assert_allclose(g, fd, rtol=2e-5, atol=2e-5 * np.abs(fd).max())
    s = np.asarray(net.segment_scores(theta)).reshape(-1, len(theta))
    np.testing.assert_allclose(s.sum(0), g, rtol=1e-9, atol=1e-9 * np.abs(g).max())
    assert s.shape[0] == sum(net.get_data(i).get_n_segments()
                             for i in range(net.get_n_measurements()))


def test_priors_and_structure_prior_from_a_path():
    net = _net(True, two=True)
    net.set_roughness_weight(0.3)
    net.set_anchor_sigma(0.8)
    net.set_map_roughness_weight(0.01)
    net.set_map_prior_from_path(1, [40.0, 55.0, 70.0], [44.0, 62.0, 48.0], 2.0)
    assert net.get_parameter_value("p1.map[1]") == pytest.approx(62.0)
    assert net.get_parameter_value("p1.map[0]") == pytest.approx(44.0)
    net.set_parameter_prior("hidden.D", np.log(300.0), 0.5)
    theta = np.asarray(net.get_theta())
    theta += np.random.default_rng(1).normal(0, 0.1, len(theta))
    np.testing.assert_allclose(net.log_prior_gradient(theta), _fd(net.log_prior, theta),
                               rtol=1e-5, atol=1e-6)
    assert net.log_posterior(theta) == pytest.approx(
        net.log_likelihood_at(theta) + net.log_prior(theta))


def test_fit_and_laplace():
    net = _net(False, seed=2)
    # few photons: weak log-normal priors keep the posterior proper
    for name in net.get_free_parameter_names():
        if "rate" in name:
            net.set_parameter_prior(name, np.log(net.get_parameter_value(name)), 1.0)
        else:
            net.set_parameter_prior(name, net.get_parameter_value(name), 5.0)
    theta0 = np.asarray(net.get_theta())
    opts = bff.FRETLandscapeFitOptions()
    opts.max_iterations = 60
    opts.patience = 20
    opts.min_delta = 1e-6
    fit = net.fit(theta0, opts)
    th = np.asarray(fit.get_theta())
    assert fit.get_log_posterior() >= net.log_posterior(theta0)
    assert np.abs(net.log_posterior_gradient(th)).max() < 1e-2 * max(
        1.0, np.abs(net.log_posterior_gradient(theta0)).max())
    lap = net.laplace(th)
    n = len(th)
    H = np.asarray(lap.get_precision()).reshape(n, n)
    np.testing.assert_allclose(np.asarray(lap.get_covariance()).reshape(n, n) @ H, np.eye(n),
                               atol=1e-6)
    names = list(lap.get_names())
    k = names.index("hidden.rate[0->1]")
    assert lap.get_natural_sigmas()[k] == pytest.approx(lap.get_values()[k] * lap.get_sigmas()[k])


def test_state_distance_from_label_clouds():
    rng = np.random.default_rng(0)
    def cloud(c, n=400):
        p = rng.normal(0, 4.0, (n, 3)) + c
        return np.column_stack([p, np.ones(n)]).ravel()
    d, a = _dyes(False)
    m = bff.FRETMeasurement("p", d, a, 55.0)
    axis = np.linspace(0, 120, 121)
    mean = m.set_state_distance_from_clouds(0, cloud([0, 0, 0]), cloud([50, 0, 0]), axis)
    ref = bff.cloud_model_distance(cloud([0, 0, 0]), cloud([50, 0, 0]),
                                   bff.PROBE_PAIR_DISTANCE_MEAN, 55.0)
    assert mean == pytest.approx(ref, abs=1.0)
    assert m.get_state_means()[0] == pytest.approx(mean)
