"""Excited-state kinetics between two chromophores that transfer both ways.

m^T exp(K t) p0 is the contract: the components and the decay form must be it,
at a degeneracy too, and the one-way case (donor-acceptor FRET) is its k_BA = 0
special case. The light path comes from PhotophysicsCrosstalkMatrix.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import IMP.bff as bff

TIME = np.linspace(0.0, 25.0, 400)


def _reference(tau_a, tau_b, k_ab, k_ba, p0, m, time=TIME):
    K = np.array([[-1.0 / tau_a - k_ab, k_ba], [k_ab, -1.0 / tau_b - k_ba]])
    return np.array([m @ expm(K * t) @ p0 for t in time])


def _from_components(parts, time=TIME):
    parts = np.asarray(parts).reshape(2, 3)
    out = np.zeros_like(time)
    for amplitude, rate, kind in parts:
        out += amplitude * (time if kind == bff.TRANSFER_T_EXPONENTIAL else 1.0) * np.exp(-rate * time)
    return out


CASES = [
    (4.0, 2.5, 0.3, 0.1, (0.9, 0.1), (0.2, 0.8)),     # both directions
    (4.0, 1.0, 0.5, 0.0, (1.0, 0.0), (0.0, 1.0)),     # FRET: sensitised acceptor
    (4.0, 1.0, 0.5, 0.0, (1.0, 0.0), (1.0, 0.0)),     # FRET: quenched donor
    (3.0, 3.0, 0.0, 0.0, (0.5, 0.5), (1.0, 1.0)),     # no transfer, equal lifetimes
    (4.0, 1.0 / (0.25 + 0.75), 0.75, 0.0, (1.0, 0.0), (0.0, 1.0)),  # 1/tauD + k = 1/tauA exactly
]


@pytest.mark.parametrize("case", CASES)
def test_components_and_decay_are_the_matrix_exponential(case):
    tau_a, tau_b, k_ab, k_ba, p0, m = case
    reference = _reference(tau_a, tau_b, k_ab, k_ba, np.array(p0), np.array(m))
    parts = bff.transfer_pair_components(tau_a, tau_b, k_ab, k_ba, *p0, *m)
    np.testing.assert_allclose(_from_components(parts), reference, rtol=1e-10, atol=1e-12)
    decay = bff.transfer_pair_decay(TIME, tau_a, tau_b, k_ab, k_ba, *p0, *m)
    np.testing.assert_allclose(decay, reference, rtol=1e-10, atol=1e-12)


def test_the_degeneracy_is_a_t_exponential_term():
    parts = np.asarray(bff.transfer_pair_components(4.0, 1.0, 0.75, 0.0, 1.0, 0.0, 0.0, 1.0)).reshape(2, 3)
    assert parts[1, 2] == bff.TRANSFER_T_EXPONENTIAL
    # Just off it, two exponentials again -- and still exact, without a regularised divisor.
    near = np.asarray(bff.transfer_pair_components(4.0, 1.0, 0.75 + 1e-6, 0.0, 1.0, 0.0, 0.0, 1.0)).reshape(2, 3)
    assert (near[:, 2] == bff.TRANSFER_EXPONENTIAL).all()
    np.testing.assert_allclose(_from_components(near),
                               _reference(4.0, 1.0, 0.75 + 1e-6, 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
                               rtol=1e-6, atol=1e-10)


def test_one_way_transfer_is_the_donor_acceptor_closed_form():
    tau_d, tau_a, k = 4.0, 1.5, 0.6
    parts = np.asarray(bff.transfer_pair_components(tau_d, tau_a, k, 0.0, 1.0, 0.0, 1.0, 0.0)).reshape(2, 3)
    donor = _from_components(parts)
    np.testing.assert_allclose(donor, np.exp(-(1 / tau_d + k) * TIME), rtol=1e-12)
    parts = np.asarray(bff.transfer_pair_components(tau_d, tau_a, k, 0.0, 1.0, 0.0, 0.0, 1.0)).reshape(2, 3)
    kd, ka = 1 / tau_d + k, 1 / tau_a
    acceptor = k / (kd - ka) * (np.exp(-ka * TIME) - np.exp(-kd * TIME))
    np.testing.assert_allclose(_from_components(parts), acceptor, rtol=1e-10, atol=1e-14)


def _chisurf_pddem_rates():
    try:
        from chisurf.core.fluorescence.tcspc.tcspc import pddem_rates
    except Exception:
        pytest.skip("ChiSurf is not importable here")
    return pddem_rates


def test_the_chisurf_mode_reproduces_chisurf_pddem():
    pddem_rates = _chisurf_pddem_rates()
    a = np.array([0.6, 3.8, 0.4, 1.1])
    b = np.array([0.7, 4.2, 0.3, 0.9])
    rates = np.array([0.2, 0.05, 0.5, 0.4, 0.3, 2.5])
    px, pm, pure = np.array([0.98, 0.02]), np.array([0.02, 0.98]), np.array([0.1, 0.05])
    fab = np.array([1.0, 0.3])
    expected = pddem_rates(a, b, rates[1::2][:, None] * fab[None, :], px, pm, pure, rates[0::2])
    excitation = bff.PhotophysicsCrosstalkMatrix(["pulse"], ["A", "B"], list(px))
    emission = bff.PhotophysicsCrosstalkMatrix(["A", "B"], ["channel"], list(pm))
    populations = bff.transfer_populations_from_pure_fractions(*pure)
    got = np.asarray(bff.transfer_kinetics_spectrum(a, b, rates, fab[0], fab[1], populations, excitation,
                                                   emission, "pulse", "channel", "A", "B",
                                                   bff.TRANSFER_CHISURF_REGULARISED))
    # ChiSurf drops components below 1e-10; the fixed layout keeps them at 0.
    def decay(spectrum):
        return np.sum(spectrum[0::2, None] * np.exp(-TIME[None, :] / spectrum[1::2, None]), axis=0)
    kept = got[np.repeat(np.abs(got[0::2]) > 1e-10, 2)]
    np.testing.assert_allclose(np.sort(kept[1::2]), np.sort(expected[1::2]), rtol=1e-12)
    np.testing.assert_allclose(decay(got.reshape(-1, 2).ravel()), decay(expected), rtol=1e-12)
    exact = np.asarray(bff.transfer_kinetics_spectrum(a, b, rates, fab[0], fab[1], populations, excitation,
                                                     emission, "pulse", "channel"))
    np.testing.assert_allclose(decay(exact), decay(expected), rtol=1e-7)


def test_the_crosstalk_mixing_jacobian():
    M = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]])
    sources = np.array([[3.0, 1.0], [0.5, 2.0]])          # (n_sources, n_items)
    jac = np.asarray(bff.crosstalk_apply_mixing_jacobian(list(M.ravel()), 2, 3, list(sources.ravel())))
    n_out, n_params = 3 * 2, 2 * 3 + 2 * 2
    jac = jac.reshape(n_out, n_params)
    names = list(bff.crosstalk_apply_mixing_parameter_names(2, 3, 2))
    assert len(names) == n_params and names[0] == "M[0,0]" and names[-1] == "sources[1,1]"
    params = np.concatenate([M.ravel(), sources.ravel()])

    def f(x):
        return np.asarray(bff.crosstalk_apply_mixing(list(x[:6]), 2, 3, list(x[6:])))
    numeric = np.column_stack([(f(params + h) - f(params - h)) / 2e-6
                               for h in np.eye(n_params) * 1e-6])
    np.testing.assert_allclose(jac, numeric, atol=1e-9)
    shares = np.asarray(bff.crosstalk_row_shares(list(M.ravel()), 2, 3)).reshape(2, 3)
    np.testing.assert_allclose(shares, M / M.sum(axis=1, keepdims=True))


@pytest.mark.parametrize("case", CASES[:3])
def test_the_components_jacobian_is_the_derivative_of_the_components(case):
    tau_a, tau_b, k_ab, k_ba, p0, m = case
    at = np.array([tau_a, tau_b, k_ab, k_ba, *p0, *m], dtype=float)
    jac = np.asarray(bff.transfer_pair_components_jacobian(*at)).reshape(6, 8)

    def f(x):
        return np.asarray(bff.transfer_pair_components(*x))

    def central(h):
        cols = []
        for j in range(8):
            step = h * max(abs(at[j]), 1.0)
            e = np.zeros(8)
            e[j] = step
            cols.append((f(at + e) - f(at - e)) / (2 * step))
        return np.column_stack(cols)
    # Richardson at several steps, the best one: truncation falls with the
    # step and roundoff grows with it.
    references = [(4 * central(h / 2) - central(h)) / 3 for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)]
    floor = 1e-5 * np.max(np.abs(references[0])) + 1e-12
    errors = [np.max(np.abs(jac - r) / (np.abs(r) + floor)) for r in references]
    reference = references[int(np.argmin(errors))]
    assert min(errors) < 1e-7
    planted = jac.copy()
    planted.flat[int(np.argmax(np.abs(planted)))] *= 1 + 1e-5
    assert np.max(np.abs(planted - reference) / (np.abs(reference) + floor)) > 5e-6


def _transfer_node(kinds):
    node = bff.PhotophysicsTransferKineticsNode("transfer")
    node.build_ports()
    node.add_output_port("transfer", bff.GraphPort([0.0], False, True))
    if kinds:
        node.add_output_port("kinds", bff.GraphPort([0.0], False, True))
    for key, value in (("spectrum_a", [1.0, 4.0]), ("spectrum_b", [1.0, 1.0]), ("rates", [1.0, 0.75])):
        node.get_input_port(key).set_values_array(np.asarray(value, dtype=float))
    for key, value in (("f_ab", 1.0), ("f_ba", 0.0), ("pure_a", 0.0), ("pure_b", 0.0),
                       ("excitation_pulse_A", 1.0), ("excitation_pulse_B", 0.0),
                       ("emission_A_channel", 0.0), ("emission_B_channel", 1.0)):
        node.get_input_port(key).value = value
    node.update()
    return node


def test_at_a_degeneracy_the_decay_convolves_the_exact_t_exponential():
    """1/tauA + k = 1/tauB: the node flags the t e^{-kt} term, TCSPCDecay
    convolves it, and the unconvolved periodic curve is the matrix exponential
    summed over the pulses -- no split exponentials, no eps."""
    node = _transfer_node(kinds=True)
    spectrum = np.asarray(node.get_output_port("transfer").value, dtype=float)
    kinds = np.asarray(node.get_output_port("kinds").value, dtype=float)
    assert kinds.size == spectrum.size // 2 and kinds.sum() == 1.0

    dt, period, n = 0.04, 12.5, 400
    decay = bff.TCSPCDecay("decay")
    decay.set_number_of_lifetimes(1)
    decay.add_output_port("decay", bff.GraphPort([0.0], False, True))
    decay.set_response_array(np.r_[1.0, np.zeros(n - 1)])
    decay.set_timing(dt, period)
    decay.set_convolution_range(n, n)
    decay.set_convolve(False)
    decay.set_spectrum_from_port(True)
    decay.get_input_port("lifetime_spectrum").set_values_array(spectrum)
    decay.add_input_port("spectrum_kinds", bff.GraphPort([0.0]))
    decay.get_input_port("spectrum_kinds").set_values_array(kinds)
    decay.update()
    got = np.asarray(decay.get_output_port("decay").value, dtype=float)

    t = np.arange(n) * dt
    want = sum(_reference(4.0, 1.0, 0.75, 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), time=t + j * period)
               for j in range(40))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)


def test_without_a_kinds_output_the_degeneracy_is_split_as_before():
    spectrum = np.asarray(_transfer_node(kinds=False).get_output_port("transfer").value, dtype=float)
    exact = np.asarray(_transfer_node(kinds=True).get_output_port("transfer").value, dtype=float)
    assert spectrum.size == exact.size
