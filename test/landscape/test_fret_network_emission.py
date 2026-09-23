"""Per-state emission of a FRET network measurement: detection rates and
microtime densities (excited-state kinetics x light path x IRF), integrated
over each hidden state's distance distribution."""

import numpy as np
import pytest

import IMP.bff as bff


def _dyes():
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    d.add_state("quenched", 1.0, 0.3, 1.5)
    d.set_rate("bright", "quenched", 1.0)
    d.set_rate("quenched", "bright", 1.0)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 1.0, 0.6, 3.0, 1.0)
    a.add_state("dark", 0.0, 0.0, 3.0, 1.0)      # absorbs, does not emit
    a.add_state("bleached", 0.0, 0.0, 3.0, 0.0)  # neither
    a.set_rate("bright", "dark", 1.0)
    a.set_rate("dark", "bright", 1.0)
    a.set_rate("bright", "bleached", 0.1)
    a.set_initial([1.0, 0.0, 0.0])
    return d, a


def _instrument(n_bins=1, pie=False):
    pulses = ["donor_pulse", "acceptor_pulse"] if pie else ["donor_pulse"]
    ex = [50.0, 0.5] + ([0.0, 30.0] if pie else [])
    exc = bff.PhotophysicsCrosstalkMatrix(pulses, ["donor", "acceptor"], ex)
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["green", "red"],
                                         [0.4, 0.02, 0.05, 0.5])
    ins = bff.FRETInstrument(exc, em, n_bins, 0.064)
    ins.set_background(0, 0.3)
    ins.set_background(1, 0.7)
    if n_bins > 1:
        t = np.arange(n_bins)
        for p in range(len(pulses)):
            centre = 8 + p * n_bins // 2
            irf = np.exp(-0.5 * ((t - centre) / 2.0) ** 2)
            for c in range(2):
                ins.set_irf(p, c, irf)
    return ins


def _setup(n_bins=1, pie=False, offsets=None):
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 1.0)
    hp.set_rate(1, 0, 1.0)
    d, a = _dyes()
    m = bff.FRETMeasurement("p", d, a, 55.0)
    m.set_instrument(_instrument(n_bins, pie))
    m.set_state_distance(0, 45.0)
    if offsets is None:
        m.set_state_distance(1, 65.0)
    else:
        m.set_state_distance(1, 65.0, offsets[0], offsets[1])
    return hp, m


def test_detection_rates_are_the_steady_state_yields():
    hp, m = _setup()
    n = m.get_n_states(hp)
    lam = np.asarray(m.get_detection_rates(hp)).reshape(2, n)
    tau0 = 4.0
    for s in range(n):
        h, d, a = m.get_state(hp, s)
        r = [45.0, 65.0][h]
        tauD, qyD = [(4.0, 0.8), (1.5, 0.3)][d]
        tauA, qyA, excA, acc = [(3.0, 0.6, 1.0, 1.0), (3.0, 0.0, 0.0, 1.0), (3.0, 0.0, 0.0, 0.0)][a]
        kt = acc * (55.0 / r) ** 6 / tau0
        kD = 1 / tauD
        pD, pA = 50.0, 0.5 * excA
        donor = pD * qyD * kD / (kD + kt)             # donor photons
        acc_ph = (pD * kt / (kD + kt) + pA) * qyA      # sensitised + direct
        ref = [0.4 * donor + 0.05 * acc_ph + 0.3, 0.02 * donor + 0.5 * acc_ph + 0.7]
        np.testing.assert_allclose(lam[:, s], ref, rtol=1e-12)


def _brute_decay(m, hp, s, channel, pulse, n_bins, dt, sub=40, periods=6):
    """m^T exp(K t) p0 by scipy expm, periodic sum, box IRF, bin integrals."""
    from scipy.linalg import expm
    h, d, a = m.get_state(hp, s)
    r = [45.0, 65.0][h]
    ins = m.get_instrument()
    tauD, qyD = [(4.0, 0.8), (1.5, 0.3)][d]
    tauA, qyA, excA, acc = [(3.0, 0.6, 1.0, 1.0), (3.0, 0.0, 0.0, 1.0), (3.0, 0.0, 0.0, 0.0)][a]
    kt = acc * (55.0 / r) ** 6 / 4.0
    K = np.array([[-1 / tauD - kt, 0.0], [kt, -1 / tauA]])
    p0 = np.array([ins.get_excitation(pulse, 0), ins.get_excitation(pulse, 1) * excA])
    mm = np.array([ins.get_emission(0, channel) * qyD / tauD,
                   ins.get_emission(1, channel) * qyA / tauA])
    period = n_bins * dt
    fine = dt / sub
    t = np.arange(periods * n_bins * sub) * fine
    w, v = np.linalg.eig(K)
    coef = np.linalg.solve(v, p0)
    decay = np.real(((mm @ v) * coef) @ np.exp(np.outer(w, t)))
    decay[0] *= 0.5                                     # trapezoid: causal conv
    per = decay.reshape(periods, -1).sum(0)            # periodic, fine grid of one period
    irf = np.asarray(ins.get_irf(pulse, channel))
    irf_fine = np.repeat(irf, sub) / sub                # uniform within a bin
    conv = np.real(np.fft.ifft(np.fft.fft(per) * np.fft.fft(irf_fine))) * fine
    return conv.reshape(n_bins, sub).sum(1) / sub * sub * 1.0


def test_microtime_densities_match_brute_force():
    pytest.importorskip("scipy")
    nb = 128
    hp, m = _setup(nb, pie=True)
    n = m.get_n_states(hp)
    e = np.asarray(m.get_emission(hp)).reshape(2, nb, n)
    lam = np.asarray(m.get_detection_rates(hp)).reshape(2, n)
    np.testing.assert_allclose(e.sum(1), lam, rtol=1e-12)
    ins = m.get_instrument()
    for s in (0, 3, 7, 10):
        for c in range(2):
            ref = sum(_brute_decay(m, hp, s, c, p, nb, 0.064) for p in range(2))
            ref = ref + ins.get_background(c) / nb
            np.testing.assert_allclose(e[c, :, s], ref, rtol=1e-4, atol=1e-6 * ref.max())
    # microtime-off rates equal the microtime-on totals
    _, m1 = _setup(1, pie=True)
    np.testing.assert_allclose(np.asarray(m1.get_detection_rates(hp)).reshape(2, n), lam,
                               rtol=1e-10)


def test_pie_tells_a_bleached_acceptor_from_low_fret():
    nb = 128
    hp, m = _setup(nb, pie=True)
    n = m.get_n_states(hp)
    e = np.asarray(m.get_emission(hp)).reshape(2, nb, n)
    idx = lambda h, d, a: (h * 2 + d) * 3 + a
    window = slice(nb // 2 + 4, nb)          # after the acceptor pulse
    red_pie_bright = e[1, window, idx(1, 0, 0)].sum()
    red_pie_bleached = e[1, window, idx(1, 0, 2)].sum()
    red_pie_dark = e[1, window, idx(1, 0, 1)].sum()
    # without the acceptor pulse, the same window holds only the donor
    # pulse's tail and background: a bleached or dark acceptor adds nothing
    _, m0 = _setup(nb, pie=False)
    e0 = np.asarray(m0.get_emission(hp)).reshape(2, nb, n)
    assert red_pie_bright > 5 * red_pie_bleached
    assert red_pie_bleached == pytest.approx(e0[1, window, idx(1, 0, 2)].sum(), rel=1e-9)
    assert red_pie_dark == pytest.approx(e0[1, window, idx(1, 0, 1)].sum(), rel=1e-9)
    # a dark-but-accepting acceptor still quenches the donor; a bleached one does not
    green = e[0].sum(0)
    assert green[idx(0, 0, 2)] > 2 * green[idx(0, 0, 1)]


def test_distance_distribution_is_a_mixture_of_point_states():
    nb = 64
    offs = ([-5.0, 0.0, 7.0], [0.2, 0.5, 0.3])
    hp, m = _setup(nb, offsets=offs)
    n = m.get_n_states(hp)
    mix = np.asarray(m.get_emission(hp)).reshape(2, nb, n)
    ref = 0
    for o, w in zip(*offs):
        _, mp = _setup(nb)
        mp.set_state_distance(1, 65.0 + o)
        ref = ref + w * np.asarray(mp.get_emission(hp)).reshape(2, nb, n)
    np.testing.assert_allclose(mix, ref, rtol=1e-12, atol=1e-14)
    # zero width reproduces the point model exactly
    hp, m0 = _setup(nb, offsets=([0.0], [1.0]))
    _, mp = _setup(nb)
    np.testing.assert_array_equal(m0.get_emission(hp), mp.get_emission(hp))
