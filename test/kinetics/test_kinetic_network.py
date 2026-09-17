"""Kinetic schemes composed from factors: CIMs and their amalgamation (PRD-150).

The contract, in the order a convention flip would break it:

* a generator is ``K[target, source]``, columns summing to zero, and the joint
  index is C-order over the variables in declaration order -- pinned on a
  two-state chain and on ``kron(K_x, I) + kron(I, K_y)``;
* the amalgamated generator of real schemes equals the matrix written out by
  hand: conformation x photophysics for FCS (and ``fcs_bunching_factor`` gives
  the same curve from either), and chisurf's polarised smFRET simulator
  scheme (conformation x emission mode, two clocks, tttrlib's transposed
  source -> target layout);
* aGrUM's own amalgamation and exact inference agree, through the recorded
  fixture of ``test/kinetics/agrum_ctbn_reference.py``.
"""

import json
import pathlib

import numpy as np
import pytest
from scipy.linalg import expm

import IMP.bff as bff

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "data" / "ctbn_agrum_fixture.json").read_text()
)


def _square(flat):
    flat = np.asarray(flat)
    n = int(round(np.sqrt(flat.size)))
    return flat.reshape(n, n)


# --- conventions -------------------------------------------------------------


def test_generator_is_target_source_with_zero_column_sums():
    net = bff.KineticNetwork()
    net.add_variable("X", 2)
    net.set_rate("X", 0, 1, 3.0)  # 0 -> 1
    net.set_rate("X", 1, 0, 0.5)  # 1 -> 0
    k = _square(net.get_generator())
    np.testing.assert_array_equal(k, [[-3.0, 0.5], [3.0, -0.5]])
    np.testing.assert_allclose(k.sum(axis=0), 0.0, atol=0)
    # dp/dt = K p: from state 0 the population of 1 rises as 3/(3.5) (1 - e^{-3.5 t})
    p = bff.kinetic_transient_distribution(k.ravel(), [1.0, 0.0], 0.4)
    assert p[1] == pytest.approx(3.0 / 3.5 * (1.0 - np.exp(-3.5 * 0.4)), rel=1e-13)


def test_joint_index_is_c_order_and_the_kronecker_sum():
    rng = np.random.default_rng(1)
    kx = rng.uniform(0.1, 2.0, (2, 2))
    ky = rng.uniform(0.1, 2.0, (3, 3))
    for k in (kx, ky):
        np.fill_diagonal(k, 0.0)
        np.fill_diagonal(k, -k.sum(axis=0))
    net = bff.KineticNetwork()
    net.add_variable("X", 2)
    net.add_variable("Y", 3)
    for name, k in (("X", kx), ("Y", ky)):
        for t in range(k.shape[0]):
            for s in range(k.shape[0]):
                if s != t:
                    net.set_rate(name, s, t, k[t, s])
    expected = np.kron(kx, np.eye(3)) + np.kron(np.eye(2), ky)
    np.testing.assert_allclose(_square(net.get_generator()), expected, rtol=1e-14, atol=1e-15)
    assert net.get_state_index([1, 2]) == np.ravel_multi_index((1, 2), (2, 3)) == 5
    assert list(net.get_states(4)) == [1, 1]


def test_simultaneous_changes_are_zero_and_rates_are_parent_conditioned():
    net = bff.KineticNetwork()
    net.add_variable("C", 2)
    net.add_variable("P", 2)
    net.add_arc("C", "P")
    net.set_rate("C", 0, 1, 1.0)
    net.set_rate("P", 0, 1, 7.0, [0])
    net.set_rate("P", 0, 1, 11.0, [1])
    k = _square(net.get_generator())
    idx = net.get_state_index
    assert k[idx([1, 1]), idx([0, 0])] == 0.0  # both change in one event
    assert k[idx([0, 1]), idx([0, 0])] == 7.0  # P given C = 0
    assert k[idx([1, 1]), idx([1, 0])] == 11.0  # P given C = 1
    assert k[idx([1, 0]), idx([0, 0])] == 1.0


def test_rates_set_before_an_arc_hold_for_every_parent_state():
    net = bff.KineticNetwork()
    net.add_variable("C", 3)
    net.add_variable("P", 2)
    net.set_rate("P", 1, 0, 4.0)
    net.add_arc("C", "P")
    assert [net.get_rate("P", 1, 0, [c]) for c in range(3)] == [4.0, 4.0, 4.0]
    assert net.get_rate("P", 1, 1, [2]) == -4.0  # the diagonal is derived


def test_refusals():
    cim = bff.KineticIntensityMatrix("X", 2)
    with pytest.raises(ValueError):
        cim.set_rate(0, 0, 1.0)  # the diagonal is derived, not set
    with pytest.raises(ValueError):
        cim.set_rate(0, 1, -1.0)
    with pytest.raises(ValueError):
        cim.amalgamate(bff.KineticIntensityMatrix("X", 2))  # shared variable
    with pytest.raises(ValueError):
        bff.KineticIntensityMatrix("X", 2, ["Z"], [2]).get_generator()  # still conditional


# --- amalgamation as aGrUM defines it -----------------------------------------


def _random_cim(rng, name, n, parents=(), parent_states=()):
    cim = bff.KineticIntensityMatrix(name, n, list(parents), list(parent_states))
    for c in range(cim.get_number_of_parent_configurations()):
        for s in range(n):
            for t in range(n):
                if s != t:
                    cim.set_rate(s, t, float(rng.uniform(0.1, 3.0)), c)
    return cim


def test_amalgamation_is_associative_and_keeps_unresolved_parents():
    rng = np.random.default_rng(7)
    a = _random_cim(rng, "A", 2, ["B"], [3])
    b = _random_cim(rng, "B", 3, ["A", "D"], [2, 2])
    d = _random_cim(rng, "D", 2, ["B"], [3])
    ab = a.amalgamate(b)
    assert list(ab.get_variables()) == ["A", "B"]
    assert list(ab.get_parents()) == ["D"]  # B resolved inside, D not yet
    left = _square(ab.amalgamate(d).get_generator())
    right = _square(a.amalgamate(b.amalgamate(d)).get_generator())
    np.testing.assert_allclose(left, right, rtol=1e-14, atol=1e-14)
    # extracting D = 1 from the pair equals building the pair on B given D = 1
    b_given_d1 = b.extract(["D"], [1])
    np.testing.assert_allclose(
        _square(ab.extract(["D"], [1]).get_generator()),
        _square(a.amalgamate(b_given_d1).get_generator()),
        rtol=1e-14,
        atol=1e-14,
    )


def test_two_clocks_add_because_amalgamation_is_linear():
    """K = K_dark + k_exc K_exc (FCS.h; k_nrad / k_rad in tttrlib's simulator)."""
    rng = np.random.default_rng(3)

    def build(scale):
        net = bff.KineticNetwork()
        net.add_variable("C", 2)
        net.add_variable("P", 3)
        net.add_arc("C", "P")
        return net

    dark, exc, total = build(0), build(0), build(0)
    k_exc = 2.5
    for c in range(2):
        for s, t in ((1, 0), (1, 2), (2, 0)):
            r = float(rng.uniform(0.5, 2.0))
            dark.set_rate("P", s, t, r, [c])
            total.set_rate("P", s, t, r, [c])
        sigma = float(rng.uniform(0.5, 2.0))
        exc.set_rate("P", 0, 1, sigma, [c])
        total.set_rate("P", 0, 1, k_exc * sigma, [c])
    for net in (dark, total):
        net.set_rate("C", 0, 1, 0.3)
        net.set_rate("C", 1, 0, 0.1)
    np.testing.assert_allclose(
        np.asarray(dark.get_generator()) + k_exc * np.asarray(exc.get_generator()),
        total.get_generator(),
        rtol=1e-14,
        atol=1e-14,
    )


# --- real schemes, written out by hand ----------------------------------------

# A PET/PIFE-type FCS scheme: a two-state conformation (open, closed) and the
# rhodamine singlet-triplet photophysics (chisurf fcs_saturation_calc
# rhodamine_3state.json), the S1 decay quenched in the closed state. SI units.
K_OC, K_CO = 800.0, 300.0  # conformational exchange, Hz
K_S1 = (2.5e8, 1.0e9)  # S1 -> S0 in open, closed
K_ISC, K_T = 2.5e6, 5.0e5  # S1 -> T, T -> S0
SIGMA = 1.0  # S0 -> S1 excitation cross section (times k_exc)


def _pet_networks():
    dark, exc = bff.KineticNetwork(), bff.KineticNetwork()
    for net in (dark, exc):
        net.add_variable("conformation", 2)  # open, closed
        net.add_variable("photophysics", 3)  # S0, S1, T
        net.add_arc("conformation", "photophysics")
    dark.set_rate("conformation", 0, 1, K_OC)
    dark.set_rate("conformation", 1, 0, K_CO)
    for c in range(2):
        dark.set_rate("photophysics", 1, 0, K_S1[c], [c])
        dark.set_rate("photophysics", 1, 2, K_ISC, [c])
        dark.set_rate("photophysics", 2, 0, K_T, [c])
        exc.set_rate("photophysics", 0, 1, SIGMA, [c])
    return dark, exc


def _pet_by_hand():
    """The 6-state K[target, source], one transition at a time."""
    o_s0, o_s1, o_t, c_s0, c_s1, c_t = range(6)
    dark = np.zeros((6, 6))
    exc = np.zeros((6, 6))

    def rate(m, source, target, k):
        m[target, source] += k
        m[source, source] -= k

    for o, c in ((o_s0, c_s0), (o_s1, c_s1), (o_t, c_t)):
        rate(dark, o, c, K_OC)
        rate(dark, c, o, K_CO)
    rate(dark, o_s1, o_s0, K_S1[0])
    rate(dark, c_s1, c_s0, K_S1[1])
    for s1, t, s0 in ((o_s1, o_t, o_s0), (c_s1, c_t, c_s0)):
        rate(dark, s1, t, K_ISC)
        rate(dark, t, s0, K_T)
    rate(exc, o_s0, o_s1, SIGMA)
    rate(exc, c_s0, c_s1, SIGMA)
    return dark, exc


def test_conformation_times_photophysics_equals_the_hand_written_fcs_scheme():
    dark, exc = _pet_networks()
    hand_dark, hand_exc = _pet_by_hand()
    k_dark = _square(dark.get_generator())
    k_exc = _square(exc.get_generator())
    np.testing.assert_allclose(k_dark, hand_dark, rtol=1e-15, atol=1e-6)
    np.testing.assert_array_equal(k_exc, hand_exc)
    np.testing.assert_allclose(k_dark.sum(axis=0), 0.0, atol=1e-6)  # ~1e9 Hz entries

    # The FCS forward model consumes either one identically.
    tau = np.logspace(-9, -1, 60)
    brightness = [0.0, 1.0, 0.0, 0.0, 0.3, 0.0]  # S1 bright, dimmer when closed
    x_factored = bff.fcs_bunching_factor(tau, 5e5, k_dark.ravel(), k_exc.ravel(), 6, brightness)
    x_hand = bff.fcs_bunching_factor(tau, 5e5, hand_dark.ravel(), hand_exc.ravel(), 6, brightness)
    np.testing.assert_allclose(x_factored, x_hand, rtol=1e-12)
    assert np.ptp(x_hand) > 0.05  # the scheme bunches: the comparison can fail


def test_stationary_and_transient_distributions():
    dark, exc = _pet_networks()
    k = _square(dark.get_generator()) + 5e5 * _square(exc.get_generator())
    p = np.asarray(bff.kinetic_stationary_distribution(k.ravel()))
    assert p.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(k @ p, 0.0, atol=1e-12 * np.abs(k).max())
    conformation = np.asarray(dark.get_marginal(p, "conformation"))
    np.testing.assert_allclose(conformation, [K_CO / (K_OC + K_CO), K_OC / (K_OC + K_CO)], rtol=1e-9)
    # stiff: 1e9 Hz decay beside 1e3 Hz exchange, long after both relaxed
    p0 = np.zeros(6)
    p0[0] = 1.0
    late = bff.kinetic_transient_distribution(k.ravel(), p0, 0.05)
    np.testing.assert_allclose(late, p, rtol=1e-8, atol=1e-15)
    early = bff.kinetic_transient_distribution(k.ravel(), p0, 3e-9)
    np.testing.assert_allclose(early, expm(k * 3e-9) @ p0, rtol=1e-10, atol=1e-15)
    with pytest.raises(ValueError):  # two closed classes: not unique
        bff.kinetic_stationary_distribution(np.zeros(4))


def _smfret_polarized_by_hand(rate_matrix_hz, shares, speeds):
    """chisurf core/fluorescence/burst/simulate.py, SmfretParameters
    exchange_matrix_ms (polarized branch) and photoselection_matrix, as
    written there: species 2 * state + mode, row-major source -> target."""
    n = rate_matrix_hz.shape[0]
    full = np.zeros((2 * n, 2 * n))
    for source in range(n):
        for target in range(n):
            for mode in range(2):
                full[2 * target + mode, 2 * source + mode] = rate_matrix_hz[target, source] * 1e-3
    k_nrad = full.T.ravel()
    matrix = np.zeros((2 * n, 2 * n))
    for index in range(n):
        par, perp = 2 * index, 2 * index + 1
        matrix[par, perp] = speeds[index] * (1.0 - shares[index])
        matrix[perp, par] = speeds[index] * shares[index]
    return k_nrad, matrix.ravel()


def test_conformation_times_emission_mode_equals_the_smfret_simulator_scheme():
    rates_hz = np.array([[0.0, 150.0, 20.0], [400.0, 0.0, 60.0], [5.0, 90.0, 0.0]])  # K[target, source]
    shares = [0.61, 0.55, 0.52]  # parallel share of each conformation's emission
    speeds = [4000.0, 2500.0, 1800.0]  # photoselection rate, per state
    hand_nrad, hand_rad = _smfret_polarized_by_hand(rates_hz, shares, speeds)

    nrad, rad = bff.KineticNetwork(), bff.KineticNetwork()
    for net in (nrad, rad):
        net.add_variable("conformation", 3)
        net.add_variable("mode", 2)  # parallel, perpendicular
        net.add_arc("conformation", "mode")
    for s in range(3):
        for t in range(3):
            if s != t:
                nrad.set_rate("conformation", s, t, rates_hz[t, s] * 1e-3)
        rad.set_rate("mode", 0, 1, speeds[s] * (1.0 - shares[s]), [s])
        rad.set_rate("mode", 1, 0, speeds[s] * shares[s], [s])

    def engine_layout(net):
        """tttrlib SimSystem: row-major source -> target, diagonal ignored."""
        k = _square(net.get_generator()).T.copy()
        np.fill_diagonal(k, 0.0)
        return k.ravel()

    np.testing.assert_allclose(engine_layout(nrad), hand_nrad, rtol=1e-15, atol=0)
    np.testing.assert_allclose(engine_layout(rad), hand_rad, rtol=1e-15, atol=0)
    # Not the untransposed matrix: the comparison can fail.
    assert not np.allclose(_square(nrad.get_generator()).ravel(), hand_nrad)


# --- A/B against aGrUM ---------------------------------------------------------


def _network_from_fixture(scheme):
    net = bff.KineticNetwork()
    for v in scheme["variables"]:
        net.add_variable(v["name"], v["states"])
    for parent, child in scheme["arcs"]:
        net.add_arc(parent, child)
    for r in scheme["rates"]:
        values = [r["parents"][p] for p in net.get_parents(r["variable"])]
        net.set_rate(r["variable"], r["source"], r["target"], r["rate"], values)
    return net


def _agrum_index(net, states, order):
    """aGrUM toMatrix: names sorted, the first sorted name incremented fastest."""
    names = list(net.get_variables())
    sizes = dict(zip(names, net.get_variable_states()))
    index, stride = 0, 1
    for name in order:
        index += states[names.index(name)] * stride
        stride *= sizes[name]
    return index


@pytest.mark.parametrize("name", sorted(FIXTURE["schemes"]))
def test_amalgamation_and_inference_match_agrum(name):
    scheme = FIXTURE["schemes"][name]
    net = _network_from_fixture(scheme)
    q = np.asarray(scheme["agrum_matrix_row_from_col_to"])
    order = scheme["agrum_index_order_little_endian"]
    n = net.get_number_of_states()
    perm = [_agrum_index(net, list(net.get_states(i)), order) for i in range(n)]
    assert sorted(perm) == list(range(n))

    k = _square(net.get_generator())
    # K[target, source] (bff) is Q[from, to] (aGrUM) transposed and re-indexed.
    np.testing.assert_allclose(k, q[np.ix_(perm, perm)].T, rtol=1e-13, atol=1e-13)

    # SimpleInference.makeInference(t): uniform start, marginal of each variable.
    p = bff.kinetic_transient_distribution(k.ravel(), np.full(n, 1.0 / n), scheme["inference_time"])
    for variable, expected in scheme["agrum_posteriors"].items():
        np.testing.assert_allclose(net.get_marginal(p, variable), expected, rtol=1e-11, atol=1e-14)
