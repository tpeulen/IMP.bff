"""The photon-by-photon filter of a FRET network measurement: segments,
arrival models, uniformization, per-segment outputs."""

import math

import numpy as np
import pytest

import IMP.bff as bff


def _dyes(bleach=True):
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    d.add_state("dark", 0.0, 0.0, 4.0)
    d.set_rate("bright", "dark", 3.0, True)
    d.set_rate("dark", "bright", 20.0)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 1.0, 0.6, 3.0, 1.0)
    a.add_state("dark", 0.0, 0.0, 3.0, 1.0)
    a.set_rate("bright", "dark", 5.0, True)
    a.set_rate("dark", "bright", 30.0)
    if bleach:
        a.add_state("bleached", 0.0, 0.0, 3.0, 0.0)
        a.set_rate("bright", "bleached", 2.0, True)
        a.set_initial([0.7, 0.1, 0.2])
    return d, a


def _instrument(nb):
    exc = bff.PhotophysicsCrosstalkMatrix(["dp", "ap"], ["donor", "acceptor"],
                                          [60.0, 1.0, 0.0, 25.0])
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
                                         [0.4, 0.03, 0.05, 0.45])
    ins = bff.FRETInstrument(exc, em, nb, 0.25)
    ins.set_background(0, 0.5)
    ins.set_background(1, 0.8)
    if nb > 1:
        t = np.arange(nb)
        for p in range(2):
            irf = np.exp(-0.5 * ((t - (2 + p * nb // 2)) / 1.0) ** 2)
            for c in range(2):
                ins.set_irf(p, c, irf)
    return ins


def _data(rng, nb, n_seg=4, n_ph=25, rate=40.0):
    t, c, b, st, sp = [], [], [], [], []
    for k in range(n_seg):
        st.append(len(t))
        tt = np.cumsum(rng.exponential(1.0 / rate, n_ph)) + 10.0 * k
        t += list(tt)
        c += list(rng.integers(0, 2, n_ph))
        b += list(rng.integers(0, nb, n_ph))
        sp.append(len(t) - 1)
    return bff.FRETPhotonData(t, c, b, st, sp)


def _setup(nb=16, bleach=True, seed=0):
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 6.0)
    hp.set_rate(1, 0, 4.0)
    d, a = _dyes(bleach)
    m = bff.FRETMeasurement("p", d, a, 55.0)
    m.set_instrument(_instrument(nb))
    m.set_state_distance(0, 45.0, [-3.0, 0.0, 3.0], [0.25, 0.5, 0.25])
    m.set_state_distance(1, 70.0)
    data = _data(np.random.default_rng(seed), nb)
    net = bff.FRETNetworkModel(hp)
    net.add_measurement(m, data)
    return net, hp, m, data


def _brute(hp, m, data, conditional, detection, joint=False):
    from scipy.linalg import expm
    n = m.get_n_states(hp)
    ins = m.get_instrument()
    C, nb = ins.get_n_channels(), ins.get_n_bins()
    Q = np.asarray(m.get_generator(hp)).reshape(n, n)
    E = np.asarray(m.get_emission(hp)).reshape(C, nb, n)
    lam = E.sum((0, 1))
    A = Q if conditional else Q - np.diag(lam)
    F = E / lam if conditional else E
    pi = np.asarray(m.get_joint_stationary(hp) if joint else m.get_start(hp))
    if detection:
        start = pi * lam / (pi @ lam) if conditional else pi / (pi @ lam)
    else:
        start = pi
    t, ch, mb = map(np.asarray, (data.get_macrotimes(), data.get_channels(), data.get_microtimes()))
    out = []
    for s0, s1 in zip(data.get_segment_starts(), data.get_segment_stops()):
        al = start * F[ch[s0], mb[s0] if nb > 1 else 0]
        ll = math.log(al.sum())
        al /= al.sum()
        for i in range(s0 + 1, s1 + 1):
            al = F[ch[i], mb[i] if nb > 1 else 0] * (expm(A * (t[i] - t[i - 1])) @ al)
            ll += math.log(al.sum())
            al /= al.sum()
        out.append(ll)
    return np.array(out)


@pytest.mark.parametrize("conditional,detection", [(True, True), (False, True), (False, False),
                                                   (True, False)])
def test_filter_matches_expm_on_a_bleaching_scheme(conditional, detection):
    pytest.importorskip("scipy")
    net, hp, m, data = _setup()
    net.set_arrival_model(0, bff.FRET_ARRIVAL_CONDITIONAL if conditional else bff.FRET_ARRIVAL_FULL)
    net.set_detection_weighted_start(0, detection)
    ref = _brute(hp, m, data, conditional, detection)
    np.testing.assert_allclose(net.segment_log_likelihoods(0), ref, rtol=1e-11, atol=1e-10)
    assert net.log_likelihood() == pytest.approx(ref.sum(), rel=1e-11)


def test_long_gaps_are_chunked():
    pytest.importorskip("scipy")
    net, hp, m, _ = _setup()
    data = bff.FRETPhotonData([0.0, 0.003, 2.5, 2.51, 9.0], [0, 1, 1, 0, 1], [3, 9, 1, 12, 5],
                              [0], [4])
    net = bff.FRETNetworkModel(hp)
    net.add_measurement(m, data)
    net.set_arrival_model(0, bff.FRET_ARRIVAL_FULL)
    ref = _brute(hp, m, data, False, True)
    np.testing.assert_allclose(net.segment_log_likelihoods(0), ref, rtol=1e-10)


def test_conditional_is_full_minus_the_arrival_term():
    """With the same Lambda_tot in every state the arrival process carries no
    information about the state: full = conditional + sum(log L - L tau)."""
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 6.0)
    hp.set_rate(1, 0, 4.0)
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 1.0, 4.0)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 0.0, 1.0, 3.0, 1.0)
    m = bff.FRETMeasurement("p", d, a, 55.0)
    exc = bff.PhotophysicsCrosstalkMatrix(["dp"], ["donor", "acceptor"], [50.0, 0.0])
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"], [0.9, 0.1, 0.2, 0.8])
    m.set_instrument(bff.FRETInstrument(exc, em, 8, 0.5))
    m.set_state_distance(0, 45.0)
    m.set_state_distance(1, 70.0)
    lam = np.asarray(m.get_detection_rates(hp)).reshape(2, -1).sum(0)
    np.testing.assert_allclose(lam, lam[0], rtol=1e-12)
    data = _data(np.random.default_rng(1), 8)
    net = bff.FRETNetworkModel(hp)
    net.add_measurement(m, data)
    cond = np.asarray(net.segment_log_likelihoods(0))
    net.set_arrival_model(0, bff.FRET_ARRIVAL_FULL)
    full = np.asarray(net.segment_log_likelihoods(0))
    t = np.asarray(data.get_macrotimes())
    L = lam[0]
    for k, (s0, s1) in enumerate(zip(data.get_segment_starts(), data.get_segment_stops())):
        gaps = np.diff(t[s0:s1 + 1])
        arrival = len(gaps) * math.log(L) - L * gaps.sum()
        assert full[k] == pytest.approx(cond[k] + arrival, rel=1e-11)


def test_joint_stationary_start_equals_the_product_start_without_bleaching():
    net, hp, m, data = _setup(bleach=False)
    a = np.asarray(net.segment_log_likelihoods(0))
    net.set_joint_stationary_start(0, True)
    np.testing.assert_allclose(net.segment_log_likelihoods(0), a, rtol=1e-10)


def test_occupancies_and_posteriors_match_brute_force():
    pytest.importorskip("scipy")
    from scipy.linalg import expm
    net, hp, m, data = _setup(nb=8, seed=4)
    n = m.get_n_states(hp)
    ins = m.get_instrument()
    E = np.asarray(m.get_emission(hp)).reshape(2, 8, n)
    lam = E.sum((0, 1))
    Q = np.asarray(m.get_generator(hp)).reshape(n, n)
    F = E / lam
    pi = np.asarray(m.get_start(hp))
    start = pi * lam / (pi @ lam)
    t, ch, mb = map(np.asarray, (data.get_macrotimes(), data.get_channels(), data.get_microtimes()))
    occ = np.asarray(net.segment_occupancies(0, "joint")).reshape(-1, n)
    hid = np.asarray(net.segment_occupancies(0, "hidden")).reshape(-1, 2)
    acc = np.asarray(net.segment_occupancies(0, "acceptor")).reshape(-1, 3)
    np.testing.assert_allclose(occ.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(hid, occ.reshape(-1, 2, 2, 3).sum((2, 3)), atol=1e-12)
    np.testing.assert_allclose(acc, occ.reshape(-1, 2, 2, 3).sum((1, 2)), atol=1e-12)
    s0, s1 = data.get_segment_starts()[1], data.get_segment_stops()[1]
    idx = range(s0, s1 + 1)
    fw = []
    al = start * F[ch[s0], mb[s0]]
    fw.append(al / al.sum())
    for i in list(idx)[1:]:
        al = F[ch[i], mb[i]] * (expm(Q * (t[i] - t[i - 1])) @ fw[-1])
        fw.append(al / al.sum())
    bw = [np.ones(n)]
    for i in reversed(list(idx)[1:]):
        b = expm(Q * (t[i] - t[i - 1])).T @ (F[ch[i], mb[i]] * bw[0])
        bw.insert(0, b / b.max())
    post = np.array([f * b / (f @ b) for f, b in zip(fw, bw)])
    np.testing.assert_allclose(np.asarray(net.photon_posteriors(0, 1)).reshape(-1, n), post,
                               atol=1e-10)
    # time fractions: integrate alpha(t) o beta(t) over each gap on a fine grid
    tot = np.zeros(n)
    for k, i in enumerate(list(idx)[1:]):
        tau = t[i] - t[i - 1]
        g = np.linspace(0, tau, 401)
        bend = F[ch[i], mb[i]] * bw[k + 1]
        vals = np.array([(expm(Q * x) @ fw[k]) * (expm(Q * (tau - x)).T @ bend) for x in g])
        z = bend @ (expm(Q * tau) @ fw[k])
        tot += np.trapezoid(vals, g, axis=0) / z
    tot /= t[s1] - t[s0]
    np.testing.assert_allclose(occ[1], tot, atol=2e-6)


def test_reduces_to_the_landscape_model():
    """No microtimes, one-state dyes, identity map, plain start, full arrival:
    exactly FRETLandscapeModel's likelihood (paper Eq. 14)."""
    rng = np.random.default_rng(3)
    lm = bff.FRETLandscapeModel(3.75, 8.75, 30, 8, 6.0)
    t, c, o = [], [], [0]
    for k in range(4):
        n = 80
        t += list(np.cumsum(rng.exponential(1 / 25.0, n)))
        c += list(rng.integers(0, 2, n))
        o.append(o[-1] + n)
    lm.set_photons(t, np.asarray(c, np.int32), np.asarray(o, np.int32))
    mu = list(rng.normal(0, 1.5, 8))
    theta = lm.pack_parameters(mu, 1.5, [24.0, 20.0], [1.6, 4.0])
    hp = bff.FRETHiddenProcess(3.75, 8.75, 30, 8)
    hp.set_landscape(mu)
    hp.set_diffusion(1.5)
    d = bff.FRETDye("donor")
    d.add_state("on", 1.0, 1.0, 4.0)
    a = bff.FRETDye("acceptor")
    a.add_state("on", 0.0, 1.0, 3.0, 1.0)
    m = bff.FRETMeasurement("p", d, a, 6.0)
    m.set_distance_map([3.75, 8.75])
    exc = bff.PhotophysicsCrosstalkMatrix(["dp"], ["donor", "acceptor"], [1.0, 0.0])
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
                                         [24.0 * 0.97, 20.0 * 0.03, 24.0 * 0.08, 20.0 * 0.92])
    ins = bff.FRETInstrument(exc, em, 1, 1.0)
    ins.set_background(0, 1.6)
    ins.set_background(1, 4.0)
    m.set_instrument(ins)
    data = bff.FRETPhotonData(t, c, [], o[:-1], [x - 1 for x in o[1:]])
    net = bff.FRETNetworkModel(hp)
    net.add_measurement(m, data)
    net.set_arrival_model(0, bff.FRET_ARRIVAL_FULL)
    net.set_detection_weighted_start(0, False)
    np.testing.assert_allclose(net.segment_log_likelihoods(0), lm.trace_log_likelihoods(theta),
                               rtol=1e-11)
