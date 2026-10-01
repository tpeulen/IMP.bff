"""The direct-FRET likelihood term of the NPS Bayesian structural tracer.

``nps_isotropic_direct_score`` is the ``bff_structural_direct`` likelihood:
two label positions, an additive efficiency bias ``b_E`` and
``eta_R = log(R_iso / 1 A)``, scored as ``-log N(E_obs | E_iso + b_E, sigma)``.
It takes plain values -- whatever the sampler holds -- so the core builds
without IMP. It replaced ``NPSIsotropicFRETEfficiencyRestraint``, which read
the same values from IMP particles; every scoring assertion of that class's
tests is kept here against the function.
"""

import math
import sys

import IMP
import IMP.bff as bff
import pytest

DONOR = [0.0, 0.0, 0.0]
ACCEPTOR = [0.0, 0.0, 50.0]
ETA = math.log(50.0)


def score(donor=DONOR, acceptor=ACCEPTOR, bias=0.0, eta=ETA, obs=0.5, sigma=0.1):
    return bff.nps_isotropic_direct_score(donor, acceptor, bias, eta, obs, sigma)


def efficiency(donor=DONOR, acceptor=ACCEPTOR, eta=ETA):
    return bff.nps_isotropic_direct_efficiency(donor, acceptor, eta)


def test_scores_the_synthetic_reference_state():
    assert efficiency() == pytest.approx(0.5, rel=1e-12, abs=1e-14)
    assert score() == pytest.approx(-bff.normal_log_density(0.5, 0.5, 0.1),
                                    rel=1e-12, abs=1e-14)


def test_joint_mode_likelihood_pin():
    """The likelihood component of the former tracer's synthetic joint mode."""
    assert score() == pytest.approx(-1.383646559789373, rel=1e-12)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def test_positions_must_have_three_coordinates(bad):
    with pytest.raises(IMP.ValueException):
        efficiency(donor=bad)
    with pytest.raises(IMP.ValueException):
        score(acceptor=bad)


@pytest.mark.parametrize("obs", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_observations(obs):
    with pytest.raises(IMP.ValueException):
        score(obs=obs)


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), float("-inf"), 0.0, -0.1])
def test_rejects_invalid_scales(sigma):
    with pytest.raises(IMP.ValueException):
        score(sigma=sigma)


def test_accepts_finite_observation_outside_unit_range():
    """The raw Gaussian is a declared local approximation, not clipped."""
    assert score(obs=1.25) == pytest.approx(-bff.normal_log_density(1.25, 0.5, 0.1),
                                            rel=1e-12, abs=1e-14)


def test_position_bias_and_log_radius_all_change_the_score():
    # 50 -> 100 A changes the prediction 0.5 -> 1/65.
    far = [0.0, 0.0, 100.0]
    assert efficiency(acceptor=far) == pytest.approx(1.0 / 65.0, rel=1e-12)
    assert score(acceptor=far) == pytest.approx(
        -bff.normal_log_density(0.5, 1.0 / 65.0, 0.1), rel=1e-12)
    # The observation mean tracks b_E exactly.
    assert score(bias=0.2) == pytest.approx(-bff.normal_log_density(0.5, 0.7, 0.1), rel=1e-12)
    # eta_R moves the efficiency on the logistic curve.
    assert efficiency(eta=math.log(100.0)) == pytest.approx(
        1.0 / (1.0 + (50.0 / 100.0) ** 6), rel=1e-12)


@pytest.mark.parametrize("eta", [1000.0, -1000.0])
def test_extreme_finite_log_radius_stays_finite(eta):
    e = efficiency(eta=eta)
    assert math.isfinite(e) and 0.0 <= e <= 1.0
    assert math.isfinite(score(eta=eta))


def test_coincident_positions_use_the_isotropic_unit_limit():
    assert efficiency(acceptor=DONOR) == 1.0


def test_beyond_the_compatibility_cutoff_is_zero():
    assert efficiency(acceptor=[0.0, 0.0, 151.0]) == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(donor=[float("nan"), 0.0, 0.0]),
    dict(eta=float("nan")),
    dict(bias=float("inf")),
])
def test_unscorable_state_scores_plus_infinity(kwargs):
    """Every reachable sampler state has a score: never a throw."""
    assert score(**kwargs) == float("inf")


def test_overflowing_finite_separation_scores_plus_infinity():
    big = sys.float_info.max
    assert score(donor=[-big, 0.0, 0.0], acceptor=[big, 0.0, 0.0]) == float("inf")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
