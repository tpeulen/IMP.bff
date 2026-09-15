"""A donor quenched by acceptors at a density, in 1, 2 or 3 dimensions.

The laws differ in one exponent (d/6) and a half-gamma prefactor, which are
easy to transcribe wrongly without any smoke test noticing, so both are pinned
separately, and the efficiency at C = C0 against an independent quadrature.
"""

import math

import numpy as np
import pytest

import IMP.bff as bff

#: scipy.integrate.quad over [0, inf), confirmed by a dense trapezoid rule.
EFFICIENCY_AT_C0 = {3: 0.72381, 2: 0.67222, 1: 0.64159}


@pytest.mark.parametrize("dimension, volume", [(3, 4 / 3 * math.pi * 50.0 ** 3), (2, math.pi * 50.0 ** 2), (1, 100.0)])
def test_the_characteristic_density_is_one_acceptor_per_forster_volume(dimension, volume):
    assert bff.acceptor_characteristic_density(50.0, dimension) == pytest.approx(1.0 / volume, rel=1e-14)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_reduced_density_is_a_half_gamma(dimension):
    assert bff.acceptor_reduced_density(1.7, dimension) == pytest.approx(
        0.5 * math.gamma(1 - dimension / 6) * 1.7, rel=1e-14)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_stretch_exponent_is_d_over_six(dimension):
    t = np.array([0.5, 2.0, 8.0])
    factor = np.asarray(bff.acceptor_quenching_factor(t, 4.0, 1.0, dimension))
    eta = bff.acceptor_reduced_density(1.0, dimension)
    np.testing.assert_allclose(-np.log(factor) / (2 * eta), (t / 4.0) ** (dimension / 6), rtol=1e-13)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_efficiency_at_the_characteristic_density(dimension):
    # The trapezoid rule converges slowly where t^(d/6) has an infinite slope
    # at 0, most slowly in one dimension: 1.2e-4 there at the default sampling.
    assert bff.acceptor_transfer_efficiency(1.0, dimension) == pytest.approx(EFFICIENCY_AT_C0[dimension], abs=2e-4)


def test_folding_adds_the_preceding_pulses_quenched_at_their_own_times():
    spectrum = [1.0, 3.0]
    single = np.asarray(bff.acceptor_density_decay(spectrum, 64, 0.1, 4.0, 0.8, 2))
    folded = np.asarray(bff.acceptor_density_decay(spectrum, 64, 0.1, 4.0, 0.8, 2, 10.0, 2))
    t = np.arange(64) * 0.1
    expected = sum(np.exp(-(t + k * 10.0) / 3.0) * np.asarray(
        bff.acceptor_quenching_factor(t + k * 10.0, 4.0, 0.8, 2)) for k in range(3))
    np.testing.assert_allclose(single, expected[:] - sum(np.exp(-(t + k * 10.0) / 3.0) * np.asarray(
        bff.acceptor_quenching_factor(t + k * 10.0, 4.0, 0.8, 2)) for k in (1, 2)), rtol=1e-13)
    np.testing.assert_allclose(folded, expected, rtol=1e-13)


def test_an_unsupported_dimension_is_refused():
    with pytest.raises(Exception):
        bff.acceptor_reduced_density(1.0, 4)
