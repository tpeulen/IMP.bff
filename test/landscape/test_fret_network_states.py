"""The factored state of a FRET network measurement (FRETNetwork.h): hidden
process x donor x acceptor, amalgamated by KineticNetwork."""

import numpy as np
import pytest

import IMP.bff as bff


def _dyes(bleach=False, pet_in=None):
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    d.add_state("dark", 0.0, 0.0, 4.0)
    d.add_state("quenched", 1.0, 0.2, 1.0)
    d.set_rate("bright", "dark", 0.5, True)
    d.set_rate("dark", "bright", 2.0)
    if pet_in is None:
        d.set_rate("bright", "quenched", 1.5)
        d.set_rate("quenched", "bright", 3.0)
    else:
        d.set_rate("bright", "quenched", 1.5, False, pet_in)
        d.set_rate("quenched", "bright", 3.0, False, pet_in)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 0.1, 0.7, 3.0, 1.0)
    a.add_state("dark", 0.0, 0.0, 3.0, 1.0)
    a.set_rate("bright", "dark", 1.0, True)
    a.set_rate("dark", "bright", 4.0)
    if bleach:
        a.add_state("bleached", 0.0, 0.0, 3.0, 0.0)
        a.set_rate("bright", "bleached", 0.05, True)
        a.set_initial([0.8, 0.1, 0.1])
    return d, a


def _gen(n, pairs):
    k = np.zeros((n, n))
    for s, t, r in pairs:
        k[t, s] += r
    return k - np.diag(k.sum(0))


def test_joint_generator_is_the_kronecker_sum():
    hp = bff.FRETHiddenProcess(3)
    hp.set_rate(0, 1, 2.0)
    hp.set_rate(1, 0, 1.0)
    hp.set_rate(1, 2, 0.5)
    hp.set_rate(2, 1, 0.3)
    d, a = _dyes()
    m = bff.FRETMeasurement("p", d, a, 50.0)
    m.set_power(2.0)
    for k, r in enumerate([40.0, 50.0, 60.0]):
        m.set_state_distance(k, r)
    kh = np.asarray(hp.get_generator()).reshape(3, 3)
    kd = _gen(3, [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 1.5), (2, 0, 3.0)])  # light rate x power 2
    ka = _gen(2, [(0, 1, 2.0), (1, 0, 4.0)])
    ref = (np.kron(np.kron(kh, np.eye(3)), np.eye(2)) + np.kron(np.kron(np.eye(3), kd), np.eye(2))
           + np.kron(np.eye(9), ka))
    n = m.get_n_states(hp)
    assert n == 18
    np.testing.assert_allclose(np.asarray(m.get_generator(hp)).reshape(n, n), ref, atol=1e-12)
    assert list(m.get_state(hp, 13)) == [2, 0, 1]
    # product start == joint stationary without bleaching and without feedback
    np.testing.assert_allclose(m.get_start(hp), m.get_joint_stationary(hp), atol=1e-10)


def test_dye_rates_conditioned_on_the_hidden_state():
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 1.0)
    hp.set_rate(1, 0, 1.0)
    d, a = _dyes(pet_in=1)
    m = bff.FRETMeasurement("p", d, a)
    m.set_state_distance(0, 40.0)
    m.set_state_distance(1, 60.0)
    n = m.get_n_states(hp)
    g = np.asarray(m.get_generator(hp)).reshape(n, n)
    idx = lambda h, dd, aa: (h * 3 + dd) * 2 + aa
    assert g[idx(0, 2, 0), idx(0, 0, 0)] == 0.0      # no PET in conformer 0
    assert g[idx(1, 2, 0), idx(1, 0, 0)] == 1.5      # PET in conformer 1
    np.testing.assert_allclose(g.sum(0), 0, atol=1e-12)
    start = np.asarray(m.get_start(hp)).reshape(2, 3, 2)
    assert start[0, 2].sum() == 0.0 and start[1, 2].sum() > 0.0   # stationary per conformer


def test_bleaching_start_and_landscape_hidden_process():
    hp = bff.FRETHiddenProcess(4.0, 8.0, 12, 5)
    hp.set_landscape([2.0, 0.0, 1.5, -0.5, 2.0])
    hp.set_diffusion(1.2)
    d, a = _dyes(bleach=True)
    m = bff.FRETMeasurement("p", d, a, 6.0)
    m.set_distance_map([4.0, 8.0], 0.3)
    n = m.get_n_states(hp)
    assert n == 12 * 9
    g = np.asarray(m.get_generator(hp)).reshape(n, n)
    kh = np.asarray(bff.sqra_generator(hp.get_landscape(), 1.2, 4.0 / 11)).reshape(12, 12)
    off = ~np.eye(12, dtype=bool)
    np.testing.assert_allclose(g[::9, ::9][off], kh[off], atol=1e-10)  # hidden hops, dyes fixed
    js = np.asarray(m.get_joint_stationary(hp)).reshape(12, 3, 3)
    assert js[:, :, 2].sum() == pytest.approx(1.0)    # bleaching absorbs everything
    st = np.asarray(m.get_start(hp)).reshape(12, 3, 3)
    np.testing.assert_allclose(st.sum((1, 2)), hp.get_stationary(), atol=1e-12)
    np.testing.assert_allclose(st.sum((0, 1)), [0.8, 0.1, 0.1], atol=1e-12)
    # a linear map with a Gaussian spread: five Gauss-Hermite nodes about r(q)
    q = hp.get_coordinates()
    rw = np.asarray(m.get_state_distances(hp, 5)).reshape(-1, 2)
    assert rw.shape == (5, 2)
    assert np.sum(rw[:, 0] * rw[:, 1]) == pytest.approx(q[5])
    assert np.sum((rw[:, 0] - q[5]) ** 2 * rw[:, 1]) == pytest.approx(0.09)


def test_parameter_round_trip():
    hp = bff.FRETHiddenProcess(2)
    hp.set_rate(0, 1, 1.0)
    hp.set_rate(1, 0, 2.0)
    d, a = _dyes()
    m = bff.FRETMeasurement("p", d, a)
    m.set_state_distance(0, 40.0)
    m.set_state_distance(1, 60.0, [-2.0, 2.0], [1.0, 1.0])
    names = list(m.get_parameter_names(hp))
    vals = np.asarray(m.get_parameter_values(hp))
    assert len(names) == len(vals) == len(m.get_parameter_transforms(hp)) == len(m.get_parameter_kinds(hp))
    assert "p.donor.rate[bright->dark]|light" in names
    vals[names.index("p.mean[1]")] = 55.0
    m.set_parameter_values(hp, vals)
    np.testing.assert_allclose(np.asarray(m.get_state_distances(hp, 1)).reshape(-1, 2),
                               [[53.0, 0.5], [57.0, 0.5]])
    assert list(hp.get_parameter_names()) == ["hidden.rate[0->1]", "hidden.rate[1->0]"]
