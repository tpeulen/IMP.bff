"""IMP structural modelling bridge: native scoring, particle forces and weights.

These are deliberately IMP-layer tests. The standalone collection gate excludes
this file; a layer build must actually optimize particles through the restraint.
"""
import gc

import IMP.algebra
import IMP.atom
import IMP.core
import numpy as np
import pytest

import IMP
from IMP import bff


def flat(values):
    return np.asarray(values, dtype=float).ravel().tolist()


def make_options():
    options = bff.SMLMLikelihoodOptions()
    options.roi_min = [-20, -20, -20]
    options.roi_max = [20, 20, 20]
    options.background_fraction = 0.03
    options.intrinsic_sigma = 0.15
    options.cutoff_sigma = 20
    return options


def make_particles(model, coordinates):
    particles, decorators = [], []
    for point in coordinates:
        particle = IMP.Particle(model)
        xyz = IMP.core.XYZ.setup_particle(particle, IMP.algebra.Vector3D(*point))
        xyz.set_coordinates_are_optimized(True)
        particles.append(particle)
        decorators.append(xyz)
    return particles, decorators


def measured_cloud(weights=(1, 3, 2)):
    return bff.SMLMIndex(flat([[0.2, 0.1, 0.3], [1, 0.4, -0.2], [0.5, -0.1, 0.2]]),
                         flat([[0.6, 0.8, 0.5], [0.9, 0.4, 0.7], [0.5, 0.6, 0.9]]),
                         list(weights))


def particle_coordinates(decorators):
    return [list(xyz.get_coordinates()) for xyz in decorators]


def finite_difference(restraint, decorators, epsilon=1e-6):
    numerical = []
    for xyz in decorators:
        for axis in range(3):
            coordinate = xyz.get_coordinate(axis)
            xyz.set_coordinate(axis, coordinate+epsilon)
            upper = restraint.evaluate(False)
            xyz.set_coordinate(axis, coordinate-epsilon)
            lower = restraint.evaluate(False)
            xyz.set_coordinate(axis, coordinate)
            numerical.append((upper-lower)/(2*epsilon))
    return np.asarray(numerical).reshape(-1, 3)


@pytest.mark.parametrize("sum_score", [False, True])
def test_forward_restraint_matches_native_core_score_and_particle_gradients(sum_score):
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.1, -0.2, 0.1], [0.8, 0.3, -0.1]])
    observations, options, weights = measured_cloud(), make_options(), [2, 3]
    # This diagnostic option must never disable an IMP derivative request.
    options.compute_model_gradient = False
    restraint = bff.SMLMRestraint(model, particles, observations, options, weights, sum_score)
    options.compute_model_gradient = True
    expected = bff.SMLMPointModel(flat(particle_coordinates(xyzs)), weights).evaluate(observations, options)
    sf = IMP.core.RestraintsScoringFunction([restraint])
    score = sf.evaluate(True)
    scale = expected.total_observation_weight if sum_score else 1
    assert score == pytest.approx(-expected.weighted_log_likelihood if sum_score else expected.mean_nll,
                                  rel=3e-13)
    observed_gradient = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])
    expected_gradient = np.asarray(expected.model_gradient).reshape(-1, 3)*scale
    np.testing.assert_allclose(observed_gradient, expected_gradient, rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(observed_gradient, finite_difference(restraint, xyzs),
                               rtol=2e-6, atol=4e-10)
    assert restraint.get_score_mode() == "forward_likelihood"
    assert restraint.get_uses_sum_score() == sum_score
    assert restraint.get_number_of_localizations() == 3
    np.testing.assert_allclose(restraint.get_model_weights(), [0.4, 0.6])
    assert set(restraint.get_particle_indexes()) == {p.get_index() for p in particles}
    assert {IMP.Particle.get_from(p).get_index() for p in restraint.get_inputs()} == {
        p.get_index() for p in particles}


def test_sum_score_scales_mean_gradient_by_total_observation_weight_once():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.2, -0.1, 0.3]])
    measured, options = measured_cloud((2, 0, 5)), make_options()
    mean = bff.SMLMRestraint(model, particles, measured, options)
    summed = bff.SMLMRestraint(model, particles, measured, options, [], True)
    mean_score = mean.evaluate(True)
    mean_gradient = np.asarray(xyzs[0].get_derivatives())
    assert summed.evaluate(True) == pytest.approx(7*mean_score, rel=3e-13)
    np.testing.assert_allclose(xyzs[0].get_derivatives(), 7*mean_gradient, rtol=3e-13)


def test_native_restraint_and_nested_set_weights_scale_score_and_accumulator_once():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.1, -0.2, 0.1], [0.8, 0.3, -0.1]])
    restraint = bff.SMLMRestraint(model, particles, measured_cloud(), make_options())
    base_score = restraint.evaluate(True)
    base_gradient = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])
    restraint.set_weight(1.7)
    inner = IMP.RestraintSet(model, 2.5)
    inner.add_restraint(restraint)
    outer = IMP.RestraintSet(model, 0.4)
    outer.add_restraint(inner)
    sf = IMP.core.RestraintsScoringFunction([outer])
    factor = 1.7*2.5*0.4
    assert sf.evaluate(True) == pytest.approx(factor*base_score, rel=3e-13)
    observed = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])
    np.testing.assert_allclose(observed, factor*base_gradient, rtol=3e-13, atol=1e-14)


def test_observation_and_option_ownership_survives_original_python_objects():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.2, -0.1, 0.3]])
    source, options, weights = measured_cloud(), make_options(), [2.0]
    restraint = bff.SMLMRestraint(model, particles, source, options, weights)
    score = restraint.evaluate(True)
    gradient = list(xyzs[0].get_derivatives())
    options.roi_min = [100, 100, 100]
    options.background_fraction = 0.9
    options.intrinsic_sigma = 100
    weights[0] = 0
    del source, options, weights
    gc.collect()
    assert restraint.evaluate(True) == score
    np.testing.assert_allclose(xyzs[0].get_derivatives(), gradient, rtol=1e-14)


def test_live_emitter_coordinates_change_score_and_alignment_improves_it():
    model = IMP.Model()
    initial = np.array([1.3, -0.8, 0.5])
    particles, xyzs = make_particles(model, [initial])
    measured = bff.SMLMIndex([0, 0, 0], [0.5])
    options = make_options()
    options.intrinsic_sigma = 0
    restraint = bff.SMLMRestraint(model, particles, measured, options)
    initial_score = restraint.evaluate(False)
    xyzs[0].set_coordinates(IMP.algebra.Vector3D(*(initial*0.5)))
    intermediate = restraint.evaluate(False)
    xyzs[0].set_coordinates(IMP.algebra.Vector3D(0, 0, 0))
    final = restraint.evaluate(True)
    assert final < intermediate < initial_score
    np.testing.assert_allclose(xyzs[0].get_derivatives(), [0, 0, 0], atol=1e-14)


def test_imp_native_optimizer_minimizes_the_structural_restraint():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.7, -0.4, 0.3]])
    measured = bff.SMLMIndex([0, 0, 0], [0.5])
    options = make_options()
    options.intrinsic_sigma = 0
    restraint = bff.SMLMRestraint(model, particles, measured, options)
    sf = IMP.core.RestraintsScoringFunction([restraint])
    initial_score = sf.evaluate(False)
    optimizer = IMP.core.ConjugateGradients(model)
    optimizer.set_scoring_function(sf)
    optimizer.optimize(60)
    assert sf.evaluate(False) < initial_score
    np.testing.assert_allclose(xyzs[0].get_coordinates(), [0, 0, 0], atol=1e-5)


def test_rigid_body_member_forces_feed_native_rigid_optimization():
    model = IMP.Model()
    target = np.array([[-2, -1, 0], [2, 0, 0], [0, 2, 1]])
    offset = np.array([0.4, -0.3, 0.2])
    particles, xyzs = make_particles(model, target+offset)
    body_particle = IMP.Particle(model)
    body = IMP.core.RigidBody.setup_particle(body_particle, particles)
    body.set_coordinates_are_optimized(True)
    measured = bff.SMLMIndex(flat(target), [0.3]*3)
    options = make_options()
    options.intrinsic_sigma = 0
    restraint = bff.SMLMRestraint(model, particles, measured, options)
    sf = IMP.core.RestraintsScoringFunction([restraint])
    initial = sf.evaluate(False)
    optimizer = IMP.core.ConjugateGradients(model)
    optimizer.set_scoring_function(sf)
    optimizer.optimize(100)
    assert sf.evaluate(False) < initial
    # No internal emitter motion: they are carried by one structural rigid body.
    np.testing.assert_allclose(particle_coordinates(xyzs), target, atol=3e-4)


def test_particle_radius_and_atomic_mass_never_replace_observation_noise_or_model_weights():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.1, -0.2, 0.1], [0.8, 0.3, -0.1]])
    for particle in particles:
        IMP.core.XYZR.setup_particle(particle, 0.3)
        IMP.atom.Mass.setup_particle(particle, 12)
    restraint = bff.SMLMRestraint(model, particles, measured_cloud(), make_options())
    first_score = restraint.evaluate(True)
    first_gradient = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])
    IMP.core.XYZR(particles[0]).set_radius(150)
    IMP.core.XYZR(particles[1]).set_radius(0.0001)
    IMP.atom.Mass(particles[0]).set_mass(1e6)
    IMP.atom.Mass(particles[1]).set_mass(0.01)
    assert restraint.evaluate(True) == first_score
    np.testing.assert_allclose([list(xyz.get_derivatives()) for xyz in xyzs], first_gradient,
                               rtol=1e-14)
    np.testing.assert_allclose(restraint.get_model_weights(), [0.5, 0.5])


def test_gaussian_overlap_mode_matches_core_full_covariance_score_and_gradient():
    model = IMP.Model()
    particles, xyzs = make_particles(model, [[0.1, -0.2, 0.1], [0.8, 0.3, -0.1]])
    covariance = np.array([[[0.2, 0.07, 0.04], [0.07, 0.3, -0.02], [0.04, -0.02, 0.15]],
                           [[0.4, -0.1, 0.03], [-0.1, 0.5, 0.1], [0.03, 0.1, 0.3]]])
    weights = [2, 3]
    source = measured_cloud()
    restraint = bff.SMLMRestraint(model, particles, source, flat(covariance), weights, 20, 1e-10)
    expected = bff.SMLMGaussianOverlap(source).evaluate(flat(particle_coordinates(xyzs)),
                                                       flat(covariance), weights, 20, 1e-10)
    assert restraint.evaluate(True) == pytest.approx(expected.score, rel=3e-13)
    analytic = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])
    np.testing.assert_allclose(analytic, np.asarray(expected.gradient).reshape(-1, 3),
                               rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(analytic, finite_difference(restraint, xyzs),
                               rtol=2e-6, atol=4e-10)
    assert restraint.get_score_mode() == "gaussian_overlap"
    assert not restraint.get_uses_sum_score()
    del source
    gc.collect()
    assert restraint.evaluate(False) == pytest.approx(expected.score, rel=3e-13)


def test_rotated_rigid_body_overlap_keeps_anisotropic_covariances_in_observation_frame():
    model = IMP.Model()
    initial = np.array([[-0.6, -0.1, 0.2], [0.4, 0.8, -0.3], [1.2, -0.5, 0.7]])
    particles, xyzs = make_particles(model, initial)
    body = IMP.core.RigidBody.setup_particle(IMP.Particle(model), particles)
    body.set_coordinates_are_optimized(True)
    covariances = np.array([
        [[0.04, 0.03, 0], [0.03, 0.7, 0.02], [0, 0.02, 0.1]],
        np.diag([0.9, 0.05, 0.2]),
        [[0.15, -0.04, 0.02], [-0.04, 0.09, 0], [0.02, 0, 0.6]],
    ])
    weights, cutoff, background = [2, 3, 1], 20, 1e-10
    source = measured_cloud()
    restraint = bff.SMLMRestraint(model, particles, source, flat(covariances),
                                 weights, cutoff, background)
    sf = IMP.core.RestraintsScoringFunction([restraint])

    # Move the body AFTER constructing the restraint. Only centres follow its
    # frame; the supplied anisotropic covariance tensors stay in the lab frame.
    angle = np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = IMP.algebra.get_rotation_about_axis(IMP.algebra.Vector3D(0, 0, 1), angle)
    translation = np.array([0.3, -0.2, 0.1])
    IMP.core.transform(body, IMP.algebra.Transformation3D(
        rotation, IMP.algebra.Vector3D(*translation)))
    score = sf.evaluate(True)
    centres = np.asarray(particle_coordinates(xyzs))
    np.testing.assert_allclose(centres, initial @ rotation_matrix.T + translation, atol=1e-13)
    analytic = np.asarray([list(xyz.get_derivatives()) for xyz in xyzs])

    overlap = bff.SMLMGaussianOverlap(source)
    expected = overlap.evaluate(flat(centres), flat(covariances), weights, cutoff, background)
    expected_gradient = np.asarray(expected.gradient).reshape(-1, 3)
    assert score == pytest.approx(expected.score, rel=3e-13)
    np.testing.assert_allclose(analytic, expected_gradient, rtol=3e-13, atol=1e-14)
    rotated_covariances = rotation_matrix @ covariances @ rotation_matrix.T
    body_frame_score = overlap.evaluate(flat(centres), flat(rotated_covariances),
                                        weights, cutoff, background).score
    assert abs(score - body_frame_score) > 1e-4

    # The body's torque consists only of the member-centre force pullback:
    # fixed observation-frame covariances introduce no covariance-rotation term.
    inverse_rotation = body.get_reference_frame().get_transformation_to().get_rotation().get_inverse()
    local_gradient = np.asarray([
        list(inverse_rotation.get_rotated(IMP.algebra.Vector3D(*gradient)))
        for gradient in expected_gradient
    ])
    local_centres = np.asarray([
        list(IMP.core.RigidMember(particle).get_internal_coordinates())
        for particle in particles
    ])
    np.testing.assert_allclose(body.get_torque(), np.cross(local_centres, local_gradient).sum(axis=0),
                               rtol=3e-13, atol=1e-14)

    # Independent XYZ probes let each member coordinate vary without the rigid
    # body's score state restoring it during an evaluation. The scored objective
    # and fixed covariances are identical at the rotated member centres.
    probe_particles, probe_xyzs = make_particles(model, centres)
    probe = bff.SMLMRestraint(model, probe_particles, source, flat(covariances),
                             weights, cutoff, background)
    assert probe.evaluate(False) == pytest.approx(score, rel=3e-13)
    np.testing.assert_allclose(analytic, finite_difference(probe, probe_xyzs),
                               rtol=2e-6, atol=4e-10)


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "foreign_model", "missing_xyz"])
def test_constructor_rejects_invalid_emitter_particles(invalid):
    model = IMP.Model()
    particles, _ = make_particles(model, [[0, 0, 0]])
    if invalid == "empty":
        particles = []
    elif invalid == "duplicate":
        particles *= 2
    elif invalid == "foreign_model":
        other_model = IMP.Model()
        particles, _ = make_particles(other_model, [[0, 0, 0]])
    elif invalid == "missing_xyz":
        particles = [IMP.Particle(model)]
    with pytest.raises((ValueError, RuntimeError), match="SMLM"):
        bff.SMLMRestraint(model, particles, measured_cloud(), make_options())


@pytest.mark.parametrize("invalid", ["empty_data", "zero_data_weight", "model_weight_length",
                                      "negative_model_weight", "zero_model_weight", "roi",
                                      "background", "leaf"])
def test_constructor_validates_measured_data_and_likelihood_configuration(invalid):
    model = IMP.Model()
    particles, _ = make_particles(model, [[0, 0, 0]])
    measured, options, weights, leaf = measured_cloud(), make_options(), [], 16
    if invalid == "empty_data":
        measured = bff.SMLMIndex([], [])
    elif invalid == "zero_data_weight":
        measured = measured_cloud((0, 0, 0))
    elif invalid == "model_weight_length":
        weights = [1, 2]
    elif invalid == "negative_model_weight":
        weights = [-1]
    elif invalid == "zero_model_weight":
        weights = [0]
    elif invalid == "roi":
        options.roi_min = [5, 5, 5]
    elif invalid == "background":
        options.background_fraction = 1
    elif invalid == "leaf":
        leaf = 0
    with pytest.raises((ValueError, RuntimeError), match="SMLM"):
        bff.SMLMRestraint(model, particles, measured, options, weights, False, leaf)


@pytest.mark.parametrize("covariance", [[0]*8, [-1, 0, 0, 0, 1, 0, 0, 0, 1],
                                       [1, 0.1, 0, 0, 1, 0, 0, 0, 1],
                                       [float("nan")]+[0]*8])
def test_gaussian_overlap_constructor_uses_native_full_covariance_validation(covariance):
    model = IMP.Model()
    particles, _ = make_particles(model, [[0, 0, 0]])
    with pytest.raises((ValueError, RuntimeError)):
        bff.SMLMRestraint(model, particles, measured_cloud(), covariance)
