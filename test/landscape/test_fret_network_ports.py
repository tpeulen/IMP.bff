"""Signal brightness, reversible (population, flux) conformers and microtime
groups of a FRET network model: what the PyTorch prototype
(prototypes/fret_network_landscape) needed from the C++ to fit there."""

import numpy as np
import pytest

import IMP.bff as bff
from test_fret_network_filter import _dyes, _instrument, _data
from test_fret_network_fit import _fd, _net


def _three_states():
    hp = bff.FRETHiddenProcess(3)
    for (i, j, k) in [(0, 1, 3.0), (1, 0, 1.5), (1, 2, 2.0), (2, 1, 4.0), (0, 2, 0.5),
                      (2, 0, 0.25)]:
        hp.set_rate(i, j, k)
    return hp


# --- brightness ----------------------------------------------------------------------


def test_brightness_scales_the_signal_not_the_background():
    net = _net(nb=8)
    m = net.get_measurement(0)
    hp = net.get_process()
    e1 = np.asarray(m.get_emission(hp))
    m.set_brightness(0.3)
    e2 = np.asarray(m.get_emission(hp))
    ins = m.get_instrument()
    n = m.get_n_states(hp)
    bg = np.concatenate([ins.get_background(c) * np.asarray(ins.get_background_density(c))
                         for c in range(ins.get_n_channels())])
    bg = np.repeat(bg, n)
    np.testing.assert_allclose(e2 - bg, 0.3 * (e1 - bg), rtol=1e-12, atol=1e-15)
    # one scale for every channel: the gains set equal to b give the same table
    m2 = net.get_measurement(0)
    for c in range(ins.get_n_channels()):
        ins.set_gain(c, 0.3)
    m2.set_instrument(ins)
    np.testing.assert_allclose(np.asarray(m2.get_emission(hp)), e2, rtol=1e-12)


def test_brightness_is_a_fixed_log_parameter_until_freed():
    net = _net(nb=8)
    names = list(net.get_parameter_names())
    assert "p0.brightness" in names
    assert "p0.brightness" not in net.get_free_parameter_names()
    net.set_parameter_free("p0.brightness", True)
    net.set_parameter_value("p0.brightness", 0.4)
    theta = np.asarray(net.get_theta())
    k = list(net.get_free_parameter_names()).index("p0.brightness")
    assert theta[k] == pytest.approx(np.log(0.4))
    g = np.asarray(net.log_likelihood_gradient(theta))
    np.testing.assert_allclose(g, _fd(net.log_likelihood_at, theta), rtol=2e-5,
                               atol=2e-5 * np.abs(g).max())


# --- reversible process --------------------------------------------------------------


def test_reversible_keeps_the_chain_and_balances_it():
    hp = _three_states()
    pi0 = np.asarray(hp.get_stationary())
    K0 = np.asarray(hp.get_generator()).reshape(3, 3)
    hp.set_reversible(True)
    assert hp.get_is_reversible()
    assert list(hp.get_parameter_names()) == [
        "hidden.population[0]", "hidden.population[1]", "hidden.population[2]",
        "hidden.flux[0-1]", "hidden.flux[0-2]", "hidden.flux[1-2]"]
    K = np.asarray(hp.get_generator()).reshape(3, 3)  # K[target, source]
    pi = np.asarray(hp.get_stationary())
    np.testing.assert_allclose(pi, pi0, rtol=1e-10)
    np.testing.assert_allclose(K.sum(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(K @ pi, 0.0, atol=1e-12)
    flux = K * pi[None, :]  # j -> i flux
    np.testing.assert_allclose(flux, flux.T, rtol=1e-12)  # detailed balance
    # the converted fluxes symmetrise the original chain's (non-reversible) ones
    F0 = K0 * pi0[None, :]
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        assert hp.get_flux(i, j) == pytest.approx(0.5 * (F0[i, j] + F0[j, i]))


def test_reversible_rates_from_populations_and_fluxes():
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 1.0)
    hp.set_rate(1, 0, 1.0)
    hp.set_reversible(True)
    hp.set_population(0, 3.0)
    hp.set_population(1, 1.0)
    hp.set_flux(1, 0, 0.6)
    pi = np.array([0.75, 0.25])
    np.testing.assert_allclose(hp.get_stationary(), pi)
    assert hp.get_rate(0, 1) == pytest.approx(0.6 / 0.75)
    assert hp.get_rate(1, 0) == pytest.approx(0.6 / 0.25)
    with pytest.raises(Exception):
        hp.set_rate(0, 1, 2.0)
    # the parameters round-trip, and only ratios of populations matter
    v = np.asarray(hp.get_parameter_values())
    hp.set_parameter_values(list(v * [2.0, 2.0, 1.0]))
    assert hp.get_rate(0, 1) == pytest.approx(0.6 / 0.75)
    hp.set_reversible(False)
    assert list(hp.get_parameter_names()) == ["hidden.rate[0->1]", "hidden.rate[1->0]"]
    assert hp.get_rate(1, 0) == pytest.approx(0.6 / 0.25)


def test_reversible_needs_both_directions():
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 1.0)
    with pytest.raises(Exception):
        hp.set_reversible(True)


@pytest.mark.parametrize("arrival", [1, 0])
def test_reversible_model_gradient_and_likelihood(arrival):
    hp = _three_states()
    hp.set_reversible(True)
    net = bff.FRETNetworkModel(hp)
    rng = np.random.default_rng(3)
    d, a = _dyes(True)
    m = bff.FRETMeasurement("p0", d, a, 55.0)
    m.set_instrument(_instrument(8))
    for k, r in enumerate([42.0, 55.0, 70.0]):
        m.set_state_distance(k, r)
    net.add_measurement(m, _data(rng, 8, n_seg=3, n_ph=15))
    net.set_arrival_model(0, arrival)
    free = list(net.get_free_parameter_names())
    assert free[:6] == list(hp.get_parameter_names())
    theta = np.asarray(net.get_theta())
    g = np.asarray(net.log_likelihood_gradient(theta))
    np.testing.assert_allclose(g, _fd(net.log_likelihood_at, theta), rtol=2e-5,
                               atol=2e-5 * np.abs(g).max())
    # the same chain as plain rates scores the same
    plain = _three_states()
    plain.set_reversible(True)
    plain.set_reversible(False)
    net2 = bff.FRETNetworkModel(plain)
    net2.add_measurement(net.get_measurement(0), net.get_data(0))
    net2.set_arrival_model(0, arrival)
    assert net2.log_likelihood() == pytest.approx(net.log_likelihood(), rel=1e-10)
    # scaling every population is a gauge: the likelihood does not move
    t2 = theta.copy()
    t2[:3] += 0.7
    assert net.log_likelihood_at(t2) == pytest.approx(net.log_likelihood_at(theta), rel=1e-10)


# --- microtime groups ----------------------------------------------------------------


def _pulse_groups(nb):
    return [0] * (nb // 2) + [1] * (nb // 2)


def test_groups_sum_the_fine_emission():
    nb = 16
    hp, m = _setup_pie(nb)
    e_fine = np.asarray(m.get_emission(hp)).reshape(2, nb, -1)
    ins = m.get_instrument()
    ins.set_microtime_groups(_pulse_groups(nb))
    assert ins.get_n_photon_bins() == 2 and ins.get_n_bins() == nb
    m.set_instrument(ins)
    e = np.asarray(m.get_emission(hp)).reshape(2, 2, -1)
    np.testing.assert_allclose(e[:, 0], e_fine[:, :nb // 2].sum(1), rtol=1e-12)
    np.testing.assert_allclose(e[:, 1], e_fine[:, nb // 2:].sum(1), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(m.get_detection_rates(hp)).reshape(2, -1),
                               e_fine.sum(1), rtol=1e-12)
    np.testing.assert_allclose(ins.group_bins(list(range(nb))),
                               [sum(range(nb // 2)), sum(range(nb // 2, nb))])
    with pytest.raises(Exception):
        ins.set_microtime_groups([0, 2] * (nb // 2))  # group 1 has no bin


def _setup_pie(nb):
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 2.0)
    hp.set_rate(1, 0, 3.0)
    d, a = _dyes(True)
    m = bff.FRETMeasurement("p0", d, a, 55.0)
    exc = bff.PhotophysicsCrosstalkMatrix(["dp", "ap"], ["donor", "acceptor"],
                                          [50.0, 0.5, 0.0, 30.0])
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
                                         [0.4, 0.02, 0.05, 0.5])
    ins = bff.FRETInstrument(exc, em, nb, 0.5)
    ins.set_background(0, 0.3)
    ins.set_background(1, 0.7)
    t = np.arange(nb)
    for p in range(2):
        irf = np.exp(-0.5 * ((t - (1 + p * nb // 2)) / 0.7) ** 2)
        for c in range(2):
            ins.set_irf(p, c, irf)
    m.set_instrument(ins)
    m.set_state_distance(0, 45.0)
    m.set_state_distance(1, 65.0)
    return hp, m


def test_grouped_likelihood_is_the_coarsened_photons():
    nb = 16
    hp, m = _setup_pie(nb)
    o = bff.FRETSimulationOptions()
    o.n_molecules, o.duration, o.seed = 6, 3.0, 4
    data = bff.simulate_fret_measurement(hp, m, o)
    fine = bff.FRETNetworkModel(hp)
    fine.add_measurement(m, data)
    _, coarse_m = _setup_pie(nb)
    ins = coarse_m.get_instrument()
    ins.set_microtime_groups(_pulse_groups(nb))
    coarse_m.set_instrument(ins)
    groups = np.asarray(_pulse_groups(nb))
    coarse_data = bff.FRETPhotonData(
        list(data.get_macrotimes()), list(data.get_channels()),
        [int(groups[b]) for b in data.get_microtimes()],
        list(data.get_segment_starts()), list(data.get_segment_stops()))
    coarse = bff.FRETNetworkModel(hp)
    coarse.add_measurement(coarse_m, coarse_data)
    for net in (fine, coarse):
        net.set_arrival_model(0, 1)
    # a one-photon segment scores sum_s start_s F[row](s): the coarse factor is
    # the fine one summed over the photon's group
    one = lambda d: bff.FRETPhotonData([0.0], [d[1]], [d[2]], [0], [0])  # noqa: E731
    for c, b in [(0, 3), (1, 5), (1, 12)]:
        f = [bff.FRETNetworkModel(hp) for _ in range(2)]
        f[0].add_measurement(m, one((0, c, b)))
        f[1].add_measurement(coarse_m, one((0, c, int(groups[b]))))
        start = np.asarray(m.get_start(hp))
        e = np.asarray(m.get_emission(hp)).reshape(2, nb, -1)
        lam = e.sum((0, 1))
        w = start * lam / (start @ lam)
        g = groups == groups[b]
        assert f[1].log_likelihood() == pytest.approx(np.log(w @ (e[c, g].sum(0) / lam)),
                                                      rel=1e-10)
        assert f[0].log_likelihood() == pytest.approx(np.log(w @ (e[c, b] / lam)), rel=1e-10)
    assert np.isfinite(fine.log_likelihood()) and np.isfinite(coarse.log_likelihood())
    theta = np.asarray(coarse.get_theta())
    g = np.asarray(coarse.log_likelihood_gradient(theta))
    np.testing.assert_allclose(g, _fd(coarse.log_likelihood_at, theta), rtol=2e-5,
                               atol=2e-5 * np.abs(g).max())
    # the simulator on a grouped instrument reports coarse bins
    sim = bff.simulate_fret_measurement(hp, coarse_m, o)
    assert set(sim.get_microtimes()) <= {0, 1}
