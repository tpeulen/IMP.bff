"""Public Fast-NPS network-evaluator pins for the native IMP.bff API.

These cover the dynamic-regime likelihood core of Fast-NPS
(``junk/Fast-NPS-CPC-2017/samplers/logLikelihood.cpp`` and
``FRETeval.cpp``): the multi-dye oriented FRET efficiency, the convolved
polynomial efficiency with Fast-NPS's orientation grids, and the
multi-measurement log-likelihood accumulator.

No physics is duplicated: the oriented branch reuses the module's
committed ``wobbling_kappa2`` route (already pinned equal to Fast-NPS's
dep-factor closed form), and the likelihood reuses the committed
``normal_log_density`` normalization. Fast-NPS repeats its 12-term
polynomial five times in its source; BFF has one implementation.
"""

import math
import sys

import IMP
import IMP.bff as bff
import pytest


def _config(dyes):
    """[x, y, z, m, phi] rows, Fast-NPS's ``config`` layout."""
    return [list(dye) for dye in dyes]


def _dye(dep=0.0, iso=False, dist_conv=False):
    dye = bff.NPSNetworkDye()
    dye.dep = dep
    dye.iso = iso
    dye.dist_conv = dist_conv
    return dye


def _meas(dye1=0, dye2=1, fret=False, e_avg=0.0, e_err=1.0, r_iso6=50.0 ** 6,
          ta=False, r_t_avg=0.0, r_t_err=1.0, coeffs=None):
    meas = bff.NPSMeasurement()
    meas.dye1 = dye1
    meas.dye2 = dye2
    meas.fret_active = fret
    meas.e_avg = e_avg
    meas.e_err = e_err
    meas.r_iso6 = r_iso6
    meas.ta_active = ta
    meas.r_t_avg = r_t_avg
    meas.r_t_err = r_t_err
    if coeffs is not None:
        meas.set_eff_conv_coeff(coeffs)
    return meas


# ---------------------------------------------------------------------------
# Network-oriented FRET efficiency (logLikelihood.cpp lines 152-165)
# ---------------------------------------------------------------------------

def test_network_fret_matches_the_committed_scalar_api_on_random_configs():
    """The peak check, pinned: the network evaluator and the committed
    Dale-Eisinger/wobbling scalar leaf must agree to float noise."""
    import random

    rng = random.Random(1729)
    for _ in range(50):
        m1 = rng.uniform(-1, 1)
        m2 = rng.uniform(-1, 1)
        p1 = rng.uniform(-math.pi, math.pi)
        p2 = rng.uniform(-math.pi, math.pi)
        ra1 = rng.uniform(0.05, 0.39)
        ra2 = rng.uniform(0.05, 0.39)
        r = rng.uniform(20, 120)
        z = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        config = _config([
            (0.0, 0.0, 0.0, m1, p1),
            (r * math.sqrt(1 - z * z) * math.cos(ph),
             r * math.sqrt(1 - z * z) * math.sin(ph), r * z, m2, p2),
        ])
        dyes = [_dye(math.sqrt(ra1 / 0.4)), _dye(math.sqrt(ra2 / 0.4))]
        r_iso = rng.uniform(30, 80)

        network = bff.nps_network_fret_efficiency(
            config, dyes, 0, 1, r_iso ** 6)
        dye1 = bff.NPSDirectDye()
        dye1.x, dye1.y, dye1.z = 0.0, 0.0, 0.0
        dye1.m, dye1.phi = m1, p1
        dye1.steady_state_anisotropy = ra1
        dye2 = bff.NPSDirectDye()
        dye2.x, dye2.y, dye2.z = config[1][0], config[1][1], config[1][2]
        dye2.m, dye2.phi = m2, p2
        dye2.steady_state_anisotropy = ra2
        scalar = bff.nps_direct_fret_efficiency(dye1, dye2, r_iso)

        assert network == pytest.approx(scalar, rel=1e-12, abs=1e-14)


def test_network_fret_is_isotropic_half_at_riso():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    assert bff.nps_network_fret_efficiency(
        config, [_dye(), _dye()], 0, 1, 50.0 ** 6) == pytest.approx(
        0.5, rel=1e-12, abs=1e-14)


def test_network_fret_is_exactly_zero_beyond_150_angstroms():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 150.0000001, 0.0, 0.0)])
    assert bff.nps_network_fret_efficiency(
        config, [_dye(), _dye()], 0, 1, 50.0 ** 6) == 0.0


def test_network_fret_evaluates_the_inclusive_150_angstrom_cutoff():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 150.0, 0.0, 0.0)])
    # Isotropic: avg_k2 = 2/3, avgR6 = (2/3)*1.5*Riso6 = Riso6. The
    # evaluator takes r_iso6 and recovers the sixth root, so the exact
    # relation is checked to float noise (the pow roundtrip costs a bit).
    expected = 1.0 / (1.0 + (150.0 / 50.0) ** 6)
    assert bff.nps_network_fret_efficiency(
        config, [_dye(), _dye()], 0, 1, 50.0 ** 6) == pytest.approx(
        expected, rel=1e-12, abs=1e-14)


def test_network_fret_rejects_coincident_dyes():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0, 0.0, 0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye(), _dye()], 0, 1, 50.0 ** 6)


def test_network_fret_rejects_invalid_dye_states():
    config = _config([(0.0, 0.0, 0.0, 1.5, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye(), _dye()], 0, 1, 50.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye(1.5), _dye()], 0, 1, 50.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            _config([(0.0, 0.0, 0.0, 0.0, 0.0)]), [_dye(), _dye()],
            0, 1, 50.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye()], 0, 1, 50.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye(), _dye()], 0, 5, 50.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, [_dye(), _dye()], 0, 1, 0.0)


# ---------------------------------------------------------------------------
# Network transfer anisotropy (logLikelihood.cpp line 218; dep == q identity)
# ---------------------------------------------------------------------------

def test_network_ta_matches_the_committed_scalar_api():
    import random

    rng = random.Random(7)
    for _ in range(50):
        m1, m2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        p1 = rng.uniform(-math.pi, math.pi)
        p2 = rng.uniform(-math.pi, math.pi)
        ra1, ra2 = rng.uniform(0.05, 0.39), rng.uniform(0.05, 0.39)
        config = _config([(1.0, 2.0, 3.0, m1, p1),
                          (4.0, 5.0, 6.0, m2, p2)])
        dyes = [_dye(math.sqrt(ra1 / 0.4)), _dye(math.sqrt(ra2 / 0.4))]
        network = bff.nps_network_transfer_anisotropy(config, dyes, 0, 1)
        dye1 = bff.NPSDirectDye()
        dye1.m, dye1.phi = m1, p1
        dye1.steady_state_anisotropy = ra1
        dye2 = bff.NPSDirectDye()
        dye2.m, dye2.phi = m2, p2
        dye2.steady_state_anisotropy = ra2
        scalar = bff.nps_direct_transfer_anisotropy(dye1, dye2)
        assert network == pytest.approx(scalar, rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------
# Convolved polynomial efficiency (logLikelihood.cpp lines 147, 172-188).
# Fast-NPS repeats the same 12-term polynomial five times; BFF has one.
# ---------------------------------------------------------------------------

def test_convolved_efficiency_evaluates_the_polynomial_by_horner():
    coeffs = [0.0] * 12
    coeffs[2] = 1.0  # E(d) = d^2
    assert bff.nps_convolved_efficiency(3.0, coeffs) == pytest.approx(
        9.0, rel=1e-12, abs=1e-14)


def test_convolved_efficiency_is_exactly_zero_beyond_150_angstroms():
    coeffs = [1.0] + [0.0] * 11
    assert bff.nps_convolved_efficiency(150.1, coeffs) == 0.0


def test_convolved_efficiency_evaluates_the_inclusive_cutoff():
    coeffs = [0.5] + [0.0] * 11
    assert bff.nps_convolved_efficiency(150.0, coeffs) == 0.5


def test_convolved_efficiency_rejects_wrong_coefficient_counts():
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency(10.0, [1.0] * 11)
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency(10.0, [1.0] * 13)


def test_convolved_efficiency_rejects_non_finite_inputs():
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency(float("nan"), [1.0] + [0.0] * 11)
    bad = [1.0] + [0.0] * 11
    bad[3] = float("inf")
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency(10.0, bad)


# ---------------------------------------------------------------------------
# Fast-NPS orientation grid rows (logLikelihood.cpp lines 175, 186).
# The reference uses the literal 3.142, not pi; slots that would leave the
# table at the exact boundaries (phi = pi, m = -1) clamp, where the
# reference would index out of bounds.
# ---------------------------------------------------------------------------

def test_orientation_row_index_reproduces_the_reference_layout():
    # rowInd = 125*int(5/3.142*phi1) + 25*int(2.5*(cth1+1))
    #        +   5*int(5/3.142*phi2) +    int(2.5*(cth2+1)),  cth = -m.
    m1, p1, m2, p2 = 0.0, math.pi / 2, 0.0, 0.0
    cth1 = cth2 = -m1
    expected = (125 * int(5 / 3.142 * p1) + 25 * int(2.5 * (cth1 + 1.0))
                + 5 * int(5 / 3.142 * p2) + int(2.5 * (cth2 + 1.0)))
    assert bff.nps_orientation_row_index(m1, p1, m2, p2) == expected


def test_orientation_row_index_is_zero_at_the_grid_origin():
    # phi = 0, m = +1 (cth = -1): every slot is 0.
    assert bff.nps_orientation_row_index(1.0, 0.0, 1.0, 0.0) == 0


@pytest.mark.parametrize("m,phi", [
    (-1.0, math.pi),   # phi slot would be 5 (past the 625-entry grid)
    (0.0, math.pi),    # same, through the phi1 slot
    (1.0, math.pi),
    (-1.0, -0.1),      # negative phi would go negative
    (1.0, 3.0),        # cos slot at cth = -1 is 0; at m = 1 clamp catches 5
])
def test_orientation_row_index_clamps_to_the_reference_grid(m, phi):
    index = bff.nps_orientation_row_index(m, phi, m, phi)
    assert 0 <= index < 625


def test_single_orientation_row_index_reproduces_the_reference_layout():
    # rowInd = 5*int(5/3.142*phi) + int(2.5*(cth+1)), cth = -m
    # (logLikelihood.cpp line 186).
    m, phi = 0.0, math.pi / 2
    cth = -m
    expected = 5 * int(5 / 3.142 * phi) + int(2.5 * (cth + 1.0))
    assert bff.nps_single_orientation_row_index(m, phi) == expected
    # phi = pi clamps the azimuth slot to 4 (the reference would floor to
    # 5 and index row 25 of a 25-row table, out of bounds).
    assert bff.nps_single_orientation_row_index(1.0, math.pi) == 5 * 4 + 0


# ---------------------------------------------------------------------------
# Convolved regimes through the network evaluator
# (logLikelihood.cpp lines 168-190: iso-pair row 0, 625-row pair grid,
# 25-row single-dye grid)
# ---------------------------------------------------------------------------

def test_network_fret_iso_pair_uses_the_first_coefficient_row():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 60.0, 0.0, 0.0)])
    coeffs = [[0.75] + [0.0] * 11]
    dyes = [_dye(0.0, iso=True, dist_conv=True),
            _dye(0.0, iso=True, dist_conv=True)]
    assert bff.nps_network_fret_efficiency(
        config, dyes, 0, 1, 50.0 ** 6, coeffs) == 0.75


def test_network_fret_non_iso_pair_uses_the_625_row_grid():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 60.0, 0.0, 0.0)])
    row = bff.nps_orientation_row_index(0.0, 0.0, 0.0, 0.0)
    assert row == 25 * int(2.5) + int(2.5)  # 25*2 + 2 = 52
    coeffs = [[0.0] * 12 for _ in range(625)]
    coeffs[row][0] = 0.3
    dyes = [_dye(0.5, dist_conv=True), _dye(0.5, dist_conv=True)]
    assert bff.nps_network_fret_efficiency(
        config, dyes, 0, 1, 50.0 ** 6, coeffs) == 0.3


def test_network_fret_mixed_pair_uses_the_25_row_grid():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 60.0, 0.0, 0.0)])
    row = bff.nps_single_orientation_row_index(0.0, 0.0)
    assert row == 5 * 0 + int(2.5)  # 2
    coeffs = [[0.0] * 12 for _ in range(25)]
    coeffs[row][0] = 0.45
    dyes = [_dye(0.5, dist_conv=True), _dye(0.0, iso=True)]
    assert bff.nps_network_fret_efficiency(
        config, dyes, 0, 1, 50.0 ** 6, coeffs) == 0.45


def test_network_fret_convolved_without_coefficients_raises():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 60.0, 0.0, 0.0)])
    dyes = [_dye(0.0, iso=True, dist_conv=True),
            _dye(0.0, iso=True, dist_conv=True)]
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(config, dyes, 0, 1, 50.0 ** 6, [])
    # A table too short for the regime's grid raises as well: a 1-row
    # table cannot serve the 625-row non-iso pair grid.
    pair_dyes = [_dye(0.5, dist_conv=True), _dye(0.5, dist_conv=True)]
    with pytest.raises(IMP.ValueException):
        bff.nps_network_fret_efficiency(
            config, pair_dyes, 0, 1, 50.0 ** 6, [[0.5] + [0.0] * 11])


# ---------------------------------------------------------------------------
# Multi-measurement log likelihood (logLikelihood.cpp lines 197-228),
# accumulated with the committed normal_log_density.
# ---------------------------------------------------------------------------

def test_log_likelihood_is_the_normalized_gaussian_sum():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    meas = [_meas(fret=True, e_avg=0.5, e_err=0.1)]
    # E = 0.5 exactly, so the residual is 0 and only the normalization
    # remains: -0.5*log(2*pi) - log(0.1).
    expected = -0.5 * math.log(2 * math.pi) - math.log(0.1)
    assert bff.nps_network_log_likelihood(
        config, [_dye(), _dye()], meas) == pytest.approx(
        expected, rel=1e-12, abs=1e-14)


def test_log_likelihood_accumulates_fret_and_ta():
    config = _config([(0.0, 0.0, 0.0, -1.0, 0.0),
                      (0.0, 0.0, 50.0, -1.0, 0.0)])
    meas = [_meas(fret=True, e_avg=0.6, e_err=0.05,
                  ta=True, r_t_avg=0.4, r_t_err=0.02)]
    # Aligned z dyes, m = -1: cth = 1, cThT = 1.
    # TA = (1*1/5)(3*1 - 1) = 0.4; residual 0.
    # FRET: kappa_x2 = (1 - 3*1*1)^2 = 4; dep = 1 each:
    # avg_k2 = 4; E = 6 Riso6 / (6 Riso6 + r^6).
    e = 6.0 * 50.0 ** 6 / (6.0 * 50.0 ** 6 + 50.0 ** 6)
    expected = (
        -0.5 * ((e - 0.6) / 0.05) ** 2 - 0.5 * math.log(2 * math.pi)
        - math.log(0.05)
        - 0.5 * math.log(2 * math.pi) - math.log(0.02)
    )
    dyes = [_dye(1.0), _dye(1.0)]
    got = bff.nps_network_log_likelihood(config, dyes, meas)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_log_likelihood_matches_the_scalar_ta_api_on_random_configs():
    import random

    rng = random.Random(11)
    for _ in range(25):
        m1, m2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        p1 = rng.uniform(-math.pi, math.pi)
        p2 = rng.uniform(-math.pi, math.pi)
        ra1, ra2 = rng.uniform(0.05, 0.39), rng.uniform(0.05, 0.39)
        config = _config([(0.0, 0.0, 0.0, m1, p1),
                          (10.0, 0.0, 0.0, m2, p2)])
        dyes = [_dye(math.sqrt(ra1 / 0.4)), _dye(math.sqrt(ra2 / 0.4))]
        meas = [_meas(ta=True, r_t_avg=0.0, r_t_err=0.1)]
        log_l = bff.nps_network_log_likelihood(config, dyes, meas)
        dye1 = bff.NPSDirectDye()
        dye1.m, dye1.phi = m1, p1
        dye1.steady_state_anisotropy = ra1
        dye2 = bff.NPSDirectDye()
        dye2.m, dye2.phi = m2, p2
        dye2.steady_state_anisotropy = ra2
        ta = bff.nps_direct_transfer_anisotropy(dye1, dye2)
        expected = -0.5 * (ta / 0.1) ** 2 - 0.5 * math.log(2 * math.pi) \
            - math.log(0.1)
        assert log_l == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_log_likelihood_without_active_terms_is_zero():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    assert bff.nps_network_log_likelihood(
        config, [_dye(), _dye()], [_meas()]) == 0.0


def test_log_likelihood_accumulates_over_multiple_measurements():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    normalization = -0.5 * math.log(2 * math.pi)
    meas = [_meas(fret=True, e_avg=0.5, e_err=0.1),
            _meas(dye1=1, dye2=0, fret=True, e_avg=0.5, e_err=0.2)]
    expected = (normalization - math.log(0.1)) \
        + (normalization - math.log(0.2))
    assert bff.nps_network_log_likelihood(
        config, [_dye(), _dye()], meas) == pytest.approx(
        expected, rel=1e-12, abs=1e-14)


def test_log_likelihood_rejects_invalid_measurement_indices():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye(), _dye()], [_meas(fret=True, dye2=2)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye(), _dye()], [_meas(ta=True, dye1=-1)])


def test_log_likelihood_rejects_non_positive_or_non_finite_errors():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye(), _dye()], [_meas(fret=True, e_err=0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye(), _dye()], [_meas(ta=True, r_t_err=-1.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye(), _dye()],
            [_meas(fret=True, e_err=float("nan"))])


def test_log_likelihood_rejects_mismatched_config_and_dye_lengths():
    config = _config([(0.0, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 50.0, 0.0, 0.0)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            config, [_dye()], [_meas(fret=True)])
    with pytest.raises(IMP.ValueException):
        bff.nps_network_log_likelihood(
            _config([(0.0, 0.0, 0.0, 0.0, 0.0)]),
            [_dye(), _dye()], [_meas(ta=True)])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
