"""Public Bayesian posterior tracer tests for the native IMP.bff NPS port.

The restraint under test is the thin BFF likelihood bridge of the
`bff_structural_direct` tracer: `NPSIsotropicFRETEfficiencyRestraint` reads
two *existing* live `IMP.core.XYZ` label-site particles and two *existing*
`IMP.isd.Nuisance` particles (additive efficiency bias ``b_E`` and
``eta_R = log(R_iso / 1 A)``) at score time.  It owns no coordinates, no
hierarchy, no probe/AV state, no graph port, no prior and no sampler: one
`IMP.core.MonteCarlo` owns the entire posterior transaction.
"""

import math
import sys

import IMP
import IMP.bff as bff
import IMP.isd
import pytest


def _xyz(model, x, y, z, name):
    p = IMP.Particle(model, name)
    d = IMP.core.XYZ.setup_particle(p)
    d.set_coordinates(IMP.algebra.Vector3D(x, y, z))
    return d


def _nuisance(model, value, name):
    return IMP.isd.Nuisance.setup_particle(
        IMP.Particle(model, name), value)


def _baseline_fixture():
    """The synthetic reference state: sites at 0 and 50 A, E_obs = 0.5."""
    model = IMP.Model()
    donor = _xyz(model, 0.0, 0.0, 0.0, "donor")
    acceptor = _xyz(model, 0.0, 0.0, 50.0, "acceptor")
    bias = _nuisance(model, 0.0, "bias")
    log_r_iso = _nuisance(model, math.log(50.0), "log_r_iso")
    return model, donor, acceptor, bias, log_r_iso


def test_isotropic_direct_nps_restraint_scores_existing_xyzs():
    """The first public vertical slice: score live existing label sites."""
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()

    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)

    assert restraint.get_model_efficiency() == pytest.approx(
        0.5, rel=1e-12, abs=1e-14)
    assert restraint.get_observation_mean() == pytest.approx(
        0.5, rel=1e-12, abs=1e-14)
    expected = -bff.normal_log_density(0.5, 0.5, 0.1)
    assert restraint.evaluate(False) == pytest.approx(
        expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("which", ["donor", "acceptor"])
def test_isotropic_direct_nps_restraint_rejects_non_xyz_endpoints(which):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    plain = IMP.Particle(model, "plain")
    endpoints = {"donor": plain, "acceptor": acceptor} if which == "donor" \
        else {"donor": donor, "acceptor": plain}
    with pytest.raises(IMP.ValueException):
        bff.NPSIsotropicFRETEfficiencyRestraint(
            model, endpoints["donor"], endpoints["acceptor"],
            bias, log_r_iso, 0.5, 0.1)


@pytest.mark.parametrize("which", ["bias", "log_r_iso"])
def test_isotropic_direct_nps_restraint_rejects_non_nuisance_particles(which):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    plain = IMP.Particle(model, "plain")
    nuisances = {"bias": plain, "log_r_iso": log_r_iso} if which == "bias" \
        else {"bias": bias, "log_r_iso": plain}
    with pytest.raises(IMP.ValueException):
        bff.NPSIsotropicFRETEfficiencyRestraint(
            model, donor, acceptor, nuisances["bias"],
            nuisances["log_r_iso"], 0.5, 0.1)


@pytest.mark.parametrize("obs", [float("nan"), float("inf"), float("-inf")])
def test_isotropic_direct_nps_restraint_rejects_non_finite_observations(obs):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    with pytest.raises(IMP.ValueException):
        bff.NPSIsotropicFRETEfficiencyRestraint(
            model, donor, acceptor, bias, log_r_iso, obs, 0.1)


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"),
                                   float("-inf"), 0.0, -0.1])
def test_isotropic_direct_nps_restraint_rejects_invalid_scales(sigma):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    with pytest.raises(IMP.ValueException):
        bff.NPSIsotropicFRETEfficiencyRestraint(
            model, donor, acceptor, bias, log_r_iso, 0.5, sigma)


def test_isotropic_direct_nps_restraint_accepts_finite_observation_outside_unit_range():
    """The raw Gaussian is a declared local approximation, not clipped."""
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()

    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 1.25, 0.1)

    expected = -bff.normal_log_density(1.25, 0.5, 0.1)
    assert restraint.evaluate(False) == pytest.approx(
        expected, rel=1e-12, abs=1e-14)


def test_live_xyz_bias_and_log_radius_all_change_the_score_and_inputs_are_complete():
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)

    names = sorted(p.get_name() for p in restraint.get_inputs())
    assert names == ["acceptor", "bias", "donor", "log_r_iso"]

    # Live coordinate: 50 -> 100 A changes the prediction 0.5 -> 1/65.
    acceptor.set_coordinates(IMP.algebra.Vector3D(0.0, 0.0, 100.0))
    assert restraint.get_model_efficiency() == pytest.approx(
        1.0 / 65.0, rel=1e-12)
    assert restraint.evaluate(False) == pytest.approx(
        -bff.normal_log_density(0.5, 1.0 / 65.0, 0.1), rel=1e-12)

    # Live bias: the observation mean tracks b_E exactly.
    acceptor.set_coordinates(IMP.algebra.Vector3D(0.0, 0.0, 50.0))
    bias.set_nuisance(0.2)
    assert restraint.get_observation_mean() == pytest.approx(0.7, rel=1e-12)
    assert restraint.evaluate(False) == pytest.approx(
        -bff.normal_log_density(0.5, 0.7, 0.1), rel=1e-12)
    bias.set_nuisance(0.0)

    # Live log radius: eta_R moves the efficiency on the logistic curve.
    log_r_iso.set_nuisance(math.log(100.0))
    assert restraint.get_model_efficiency() == pytest.approx(
        1.0 / (1.0 + (50.0 / 100.0) ** 6), rel=1e-12)


def test_isotropic_direct_nps_restraint_rejects_derivative_evaluation():
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    with pytest.raises(IMP.UsageException):
        restraint.evaluate(True)


@pytest.mark.parametrize("eta", [1000.0, -1000.0])
def test_extreme_finite_log_radius_stays_finite(eta):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    log_r_iso.set_nuisance(eta)
    efficiency = restraint.get_model_efficiency()
    assert math.isfinite(efficiency)
    assert 0.0 <= efficiency <= 1.0
    assert math.isfinite(restraint.evaluate(False))


def test_coincident_finite_endpoints_use_the_isotropic_unit_limit():
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    acceptor.set_coordinates(IMP.algebra.Vector3D(0.0, 0.0, 0.0))
    assert restraint.get_model_efficiency() == 1.0
    assert restraint.get_observation_mean() == 1.0


@pytest.mark.parametrize("setup", [
    # Non-finite live endpoint coordinate.
    lambda m, d, a, b, e: d.set_coordinates(IMP.algebra.Vector3D(
        float("nan"), 0.0, 0.0)),
    # Note: non-finite nuisance values cannot be probed through any public
    # IMP API -- IMP.Model's attribute table itself rejects them, so the
    # restraint's defensive +inf branches for them are unreachable from
    # Python.  Only the coordinate path is live-testable.
])
def test_unscorable_live_state_scores_plus_infinity(setup):
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    setup(model, donor, acceptor, bias, log_r_iso)
    assert restraint.evaluate(False) == float("inf")
    assert restraint.get_observation_mean() == float("inf")


def test_overflowing_finite_separation_scores_plus_infinity():
    model, donor, acceptor, bias, log_r_iso = _baseline_fixture()
    restraint = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    big = sys.float_info.max
    donor.set_coordinates(IMP.algebra.Vector3D(-big, 0.0, 0.0))
    acceptor.set_coordinates(IMP.algebra.Vector3D(big, 0.0, 0.0))
    assert restraint.evaluate(False) == float("inf")


# ---------------------------------------------------------------------------
# The one-transaction Bayesian posterior: structural anchor + likelihood +
# proper Gaussian priors, sampled by exactly one IMP.core.MonteCarlo.
# ---------------------------------------------------------------------------

#: Regression pins for the synthetic joint mode; never scientific defaults.
PIN_LIKELIHOOD = -1.383646559789373
PIN_BIAS_PRIOR = -2.993084472223473
PIN_LOG_RADIUS_PRIOR = -2.076793740349318
PIN_TOTAL = -6.453524772362164

#: Structural anchor spring constant, dimensionless kT=1 convention.
K_ANCHOR = 1.0


def _marker(model, position, name):
    p = _xyz(model, position[0], position[1], position[2], name)
    return p, IMP.algebra.Vector3D(*position)


def _posterior_fixture():
    """A proper synthetic posterior whose joint mode is the start state.

    The acceptor and two non-collinear markers form one existing rigid
    body; the donor is fixed in the global frame.  S_anchor is the sum of
    three harmonic distance-to-reference restraints on the body members,
    which confines every translation and rotation the RigidBodyMover
    proposes.  The likelihood and the two ISD Gaussian priors complete
    the score; all of it is sampled by one IMP.core.MonteCarlo.
    """
    model = IMP.Model()

    donor = _xyz(model, 0.0, 0.0, 0.0, "donor")
    acceptor = _xyz(model, 0.0, 0.0, 50.0, "acceptor")
    marker1 = _xyz(model, 10.0, 0.0, 50.0, "marker1")
    marker2 = _xyz(model, 0.0, 10.0, 50.0, "marker2")

    references = {
        acceptor: IMP.algebra.Vector3D(0.0, 0.0, 50.0),
        marker1: IMP.algebra.Vector3D(10.0, 0.0, 50.0),
        marker2: IMP.algebra.Vector3D(0.0, 10.0, 50.0),
    }

    rigid_body = IMP.core.RigidBody.setup_particle(
        IMP.Particle(model, "acceptor_body"), [acceptor, marker1, marker2])
    rigid_body.set_coordinates_are_optimized(True)

    bias = _nuisance(model, 0.0, "bias")
    log_r_iso = _nuisance(model, math.log(50.0), "log_r_iso")

    likelihood = bff.NPSIsotropicFRETEfficiencyRestraint(
        model, donor, acceptor, bias, log_r_iso, 0.5, 0.1)
    bias_prior = IMP.isd.GaussianRestraint(0.0, bias, 0.02)
    log_radius_prior = IMP.isd.GaussianRestraint(
        math.log(50.0), log_r_iso, 0.05)

    anchor = []
    for member, reference in references.items():
        score = IMP.core.DistanceToSingletonScore(
            IMP.core.Harmonic(0.0, K_ANCHOR), reference)
        anchor.append(IMP.core.SingletonRestraint(model, score, member))

    scoring_function = IMP.core.RestraintsScoringFunction(
        [*anchor, likelihood, bias_prior, log_radius_prior],
        "nps_posterior")

    return {
        "model": model,
        "donor": donor,
        "acceptor": acceptor,
        "marker1": marker1,
        "marker2": marker2,
        "rigid_body": rigid_body,
        "bias": bias,
        "log_r_iso": log_r_iso,
        "likelihood": likelihood,
        "bias_prior": bias_prior,
        "log_radius_prior": log_radius_prior,
        "anchor": anchor,
        "scoring_function": scoring_function,
    }


def _movers(fixture, translation=2.0, rotation=0.1, nuisance=0.05):
    model = fixture["model"]
    bias = fixture["bias"]
    log_r_iso = fixture["log_r_iso"]
    rigid_body = fixture["rigid_body"]

    # The IMP prerequisites: an attribute a mover touches must be flagged
    # optimized first, or the mover refuses to construct/propose.
    bias.set_nuisance_is_optimized(True)
    log_r_iso.set_nuisance_is_optimized(True)
    assert rigid_body.get_coordinates_are_optimized()

    return [
        IMP.core.RigidBodyMover(model, rigid_body, translation, rotation),
        IMP.core.NormalMover(
            model, bias.get_particle_index(),
            [IMP.isd.Nuisance.get_nuisance_key()], nuisance),
        IMP.core.NormalMover(
            model, log_r_iso.get_particle_index(),
            [IMP.isd.Nuisance.get_nuisance_key()], nuisance),
    ]


def test_posterior_scores_at_the_joint_synthetic_mode():
    fixture = _posterior_fixture()

    assert fixture["likelihood"].evaluate(False) == pytest.approx(
        PIN_LIKELIHOOD, rel=1e-12)
    assert fixture["bias_prior"].evaluate(False) == pytest.approx(
        PIN_BIAS_PRIOR, rel=1e-12)
    assert fixture["log_radius_prior"].evaluate(False) == pytest.approx(
        PIN_LOG_RADIUS_PRIOR, rel=1e-12)
    assert sum(r.evaluate(False) for r in fixture["anchor"]) == \
        pytest.approx(0.0, abs=1e-12)
    assert fixture["scoring_function"].evaluate(False) == pytest.approx(
        PIN_TOTAL, rel=1e-12)


def _read_coordinates(model, xyz):
    """Read member coordinates refreshed through any lazy frame writes."""
    model.update()
    return tuple(xyz.get_coordinates())


def _assert_coordinates_restored(actual, expected):
    """Rigid-body rollback restores member XYZ only to float roundoff."""
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want, abs=1e-9)


def test_rigid_body_mover_propose_and_reject_restores_live_state():
    fixture = _posterior_fixture()
    model = fixture["model"]
    rigid_body_mover = _movers(fixture)[0]

    before_xyz = _read_coordinates(model, fixture["acceptor"])
    before_efficiency = fixture["likelihood"].get_model_efficiency()
    before_score = fixture["likelihood"].evaluate(False)

    rigid_body_mover.propose()
    model.update()
    assert fixture["likelihood"].get_model_efficiency() != before_efficiency
    rigid_body_mover.reject()

    _assert_coordinates_restored(
        _read_coordinates(model, fixture["acceptor"]), before_xyz)
    assert fixture["likelihood"].get_model_efficiency() == before_efficiency
    assert fixture["likelihood"].evaluate(False) == before_score


def test_one_monte_carlo_transaction_rejects_and_restores_everything():
    # First, prove the rejection at kT=0 is genuinely forced: replay the
    # exact seeded proposal sequence by hand and check it scores higher.
    probe = _posterior_fixture()
    probe_movers = _movers(probe)
    initial = probe["scoring_function"].evaluate(False)
    IMP.random_number_generator.seed(1729)
    for mover in probe_movers:
        mover.propose()
    proposed = probe["scoring_function"].evaluate(False)
    assert proposed > initial
    for mover in reversed(probe_movers):
        mover.reject()
    assert probe["scoring_function"].evaluate(False) == pytest.approx(
        initial, rel=1e-9)

    # Now the real transaction: same seed, same movers, one MC step at
    # kT=0 must reject and restore every structural and nuisance value.
    run = _posterior_fixture()
    run_movers = _movers(run)
    IMP.random_number_generator.seed(1729)

    before = {
        "acceptor": _read_coordinates(run["model"], run["acceptor"]),
        "marker1": _read_coordinates(run["model"], run["marker1"]),
        "marker2": _read_coordinates(run["model"], run["marker2"]),
        "bias": run["bias"].get_nuisance(),
        "log_r_iso": run["log_r_iso"].get_nuisance(),
    }
    components_before = {
        "likelihood": run["likelihood"].evaluate(False),
        "bias_prior": run["bias_prior"].evaluate(False),
        "log_radius_prior": run["log_radius_prior"].evaluate(False),
        "anchor": sum(r.evaluate(False) for r in run["anchor"]),
    }

    mc = IMP.core.MonteCarlo(run["model"])
    mc.set_scoring_function(run["scoring_function"])
    for mover in run_movers:
        mc.add_mover(mover)
    mc.set_kt(0.0)
    mc.set_return_best(False)
    mc.set_score_moved(False)
    mc.optimize(1)

    assert mc.get_number_of_proposed_steps() == 1
    assert mc.get_number_of_accepted_steps() == 0

    _assert_coordinates_restored(
        _read_coordinates(run["model"], run["acceptor"]),
        before["acceptor"])
    _assert_coordinates_restored(
        _read_coordinates(run["model"], run["marker1"]),
        before["marker1"])
    _assert_coordinates_restored(
        _read_coordinates(run["model"], run["marker2"]),
        before["marker2"])
    assert run["bias"].get_nuisance() == before["bias"]
    assert run["log_r_iso"].get_nuisance() == before["log_r_iso"]
    assert run["likelihood"].evaluate(False) == pytest.approx(
        components_before["likelihood"], rel=1e-12, abs=1e-15)
    assert run["bias_prior"].evaluate(False) == pytest.approx(
        components_before["bias_prior"], rel=1e-12, abs=1e-15)
    assert run["log_radius_prior"].evaluate(False) == pytest.approx(
        components_before["log_radius_prior"], rel=1e-12, abs=1e-15)
    assert sum(r.evaluate(False) for r in run["anchor"]) == pytest.approx(
        components_before["anchor"], abs=1e-12)
    assert run["scoring_function"].evaluate(False) == pytest.approx(
        PIN_TOTAL, rel=1e-12)


def test_seeded_posterior_runs_are_reproducible():
    outcomes = []
    for _ in range(2):
        fixture = _posterior_fixture()
        movers = _movers(fixture)
        mc = IMP.core.MonteCarlo(fixture["model"])
        mc.set_scoring_function(fixture["scoring_function"])
        for mover in movers:
            mc.add_mover(mover)
        mc.set_kt(1.0)
        mc.set_return_best(False)
        mc.set_score_moved(False)
        IMP.random_number_generator.seed(1729)
        mc.optimize(50)
        outcomes.append((
            tuple(fixture["acceptor"].get_coordinates()),
            tuple(fixture["marker1"].get_coordinates()),
            fixture["bias"].get_nuisance(),
            fixture["log_r_iso"].get_nuisance(),
            fixture["scoring_function"].evaluate(False),
            mc.get_number_of_proposed_steps(),
            mc.get_number_of_accepted_steps(),
        ))
    assert outcomes[0] == outcomes[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
