"""Forward discrete emitter likelihood: Wu et al., PMC9834062, Eq. 3/4/16.

Reference computations are intentionally brute force, with each observation's
own precision. Synthetic rigid fits use an asymmetric model to avoid symmetry
ambiguities; noise-free replicas make the expected optimum identifiable.
"""
import math

import numpy as np
import pytest

from IMP import bff


def flat(values):
    return np.asarray(values, dtype=float).ravel().tolist()


def observations(points, sigmas, weights=None):
    return bff.SMLMIndex(flat(points), flat(sigmas),
                         [] if weights is None else flat(weights))


def model(points, weights=None, leaf_size=2):
    return bff.SMLMPointModel(flat(points), [] if weights is None else flat(weights), leaf_size)


def options(background=0.03, intrinsic=0, cutoff=6, bounds=(-20, 20)):
    out = bff.SMLMLikelihoodOptions()
    out.roi_min = [bounds[0]] * 3
    out.roi_max = [bounds[1]] * 3
    out.background_fraction = background
    out.intrinsic_sigma = intrinsic
    out.cutoff_sigma = cutoff
    return out


def pose(rotation=None, translation=(0, 0, 0)):
    return flat(np.column_stack((np.eye(3) if rotation is None else rotation, translation)))


def rotation_vector(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def brute_pdf(points, model_weights, measured, sigmas, configuration, transform=None):
    points, measured, sigmas = map(np.asarray, (points, measured, sigmas))
    if transform is not None:
        transform = np.asarray(transform).reshape(3, 4)
        points = points @ transform[:, :3].T + transform[:, 3]
    model_weights = np.asarray(model_weights, dtype=float)
    model_weights = model_weights / model_weights.sum()
    effective = np.sqrt(sigmas ** 2 + configuration.intrinsic_sigma ** 2)
    displacement = (measured[:, None, :] - points[None, :, :]) / effective[:, None, :]
    q2 = np.sum(displacement ** 2, axis=2)
    coefficient = 1 / ((2 * np.pi) ** 1.5 * effective.prod(axis=1))
    signal = ((np.exp(-q2/2) * (q2 <= configuration.cutoff_sigma ** 2)) @ model_weights) * coefficient
    volume = np.prod(np.asarray(configuration.roi_max) - configuration.roi_min)
    pdf = (1-configuration.background_fraction)*signal + configuration.background_fraction/volume
    return pdf, signal


def test_paper_discrete_mixture_uses_own_heterogeneous_observation_precision():
    rng = np.random.default_rng(731)
    emitters = rng.normal(size=(113, 3)) * (2, 0.5, 1)
    measured = rng.normal(size=(71, 3)) * (1, 1, 2)
    sigmas = rng.uniform(0.15, 1.6, measured.shape)
    emitter_weights = rng.uniform(0, 3, len(emitters))
    emitter_weights[::9] = 0
    observation_weights = rng.uniform(0, 4, len(measured))
    observation_weights[::13] = 0
    rotation = rotation_vector([0.3, -0.1, 0.2])
    transform = pose(rotation, (0.3, 0.4, -0.1))
    cloud = observations(measured, sigmas, observation_weights)
    for leaf in (1, 7, 1000):
        point_model = model(emitters, emitter_weights, leaf)
        for cutoff in (1.3, 6):
            configuration = options(intrinsic=0.25, cutoff=cutoff)
            expected, signal = brute_pdf(emitters, emitter_weights, measured, sigmas,
                                         configuration, transform)
            result = point_model.evaluate(cloud, configuration, transform)
            np.testing.assert_allclose(result.pdf, expected, rtol=3e-13, atol=2e-16)
            np.testing.assert_allclose(result.model_pdf, signal, rtol=3e-13, atol=2e-16)
            np.testing.assert_allclose(result.log_pdf, np.log(expected), rtol=3e-13)
            assert result.log_likelihood == pytest.approx(np.log(expected).sum(), rel=2e-13)
            weighted = np.sum(observation_weights * np.log(expected))
            assert result.weighted_log_likelihood == pytest.approx(weighted, rel=2e-13)
            assert result.mean_nll == pytest.approx(-weighted/observation_weights.sum(), rel=2e-13)
            assert result.localization_count == len(measured)
            assert result.total_observation_weight == pytest.approx(observation_weights.sum())
            expected_fraction = (1-configuration.background_fraction)*signal/expected
            np.testing.assert_allclose(result.signal_fraction, expected_fraction, atol=2e-14)


def test_widening_only_one_observation_precision_changes_only_its_pdf():
    emitters = [[-1, 0, 0], [1, 0, 0]]
    measured = [[0.1, 0.2, 0], [0.1, 0.2, 0]]
    first = observations(measured, [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]])
    second = observations(measured, [[1, 2, 0.4], [0.2, 0.3, 0.4]])
    point_model = model(emitters)
    one = point_model.evaluate(first, options())
    two = point_model.evaluate(second, options())
    assert one.pdf[0] != pytest.approx(two.pdf[0])
    assert one.pdf[1] == pytest.approx(two.pdf[1], rel=1e-14)
    assert one.pdf[0] == pytest.approx(one.pdf[1])


def test_intrinsic_sigma_is_independent_variance_added_in_quadrature():
    sigma, intrinsic = np.array([[0.3, 0.4, 0.5], [0.7, 0.1, 0.2]]), 0.6
    measured = [[0.1, -0.2, 0.3], [0.5, 0.2, 0.1]]
    point_model = model([[0, 0, 0]])
    first = point_model.evaluate(observations(measured, sigma), options(intrinsic=intrinsic))
    second = point_model.evaluate(observations(measured, np.sqrt(sigma ** 2 + intrinsic ** 2)), options())
    np.testing.assert_allclose(first.pdf, second.pdf, rtol=2e-14)
    np.testing.assert_allclose(first.pose_gradient, second.pose_gradient, rtol=2e-14)


def test_normalized_contamination_mixture_and_explicit_roi_volume():
    configuration = options(background=0.2, bounds=(-10, 10))
    configuration.roi_min = [-10, -8, -6]
    configuration.roi_max = [10, 8, 6]
    assert configuration.get_roi_volume() == pytest.approx(20*16*12)
    point_model = model([[0, 0, 0]])
    result = point_model.evaluate(observations([[0, 0, 0], [9, 7, 5]], [0.4, 0.4]), configuration)
    peak = 1 / ((2*np.pi)**1.5 * 0.4**3)
    assert result.pdf[0] == pytest.approx(0.8*peak + 0.2/(20*16*12), rel=1e-14)
    assert result.pdf[1] == pytest.approx(0.2/(20*16*12), rel=1e-14)
    assert result.model_pdf[1] == 0
    assert result.signal_fraction[1] == 0
    assert result.supported_localizations == 1
    # Midpoint integration of the normalized Gaussian and uniform components.
    configuration = options(background=0.2, bounds=(-4, 4))
    axis = np.arange(-3.95, 4, 0.1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    configuration.compute_model_gradient = False
    integrated = point_model.evaluate(observations(grid, np.full(len(grid), 0.4)), configuration)
    assert np.sum(integrated.pdf) * 0.1**3 == pytest.approx(1, abs=8e-8)


def test_geometric_search_preserves_broad_anisotropic_observation_support():
    emitters = [[-100, 0, 0], [-120, 2, 0], [100, 0, 0], [150, 0, 0]]
    measured, sigmas = [[0, 0, 0]], [[30, 0.1, 0.2]]
    point_model = model(emitters, leaf_size=1)
    configuration = options(bounds=(-200, 200), background=0, cutoff=6)
    result = point_model.evaluate(observations(measured, sigmas), configuration)
    expected, _ = brute_pdf(emitters, [1]*4, measured, sigmas, configuration)
    assert result.pdf[0] > 0
    np.testing.assert_allclose(result.pdf, expected, rtol=1e-13)


def test_analytic_rigid_pose_and_model_gradients_against_finite_difference():
    emitters = np.array([[-0.8, 0.2, 0.3], [0.5, -0.7, 0.8], [0.9, 0.5, -0.1]])
    measured = [[0.1, -0.2, 0.4], [0.8, 0.4, 0.1], [-0.6, 0.1, 0.5]]
    sigmas = [[0.6, 0.8, 0.5], [0.9, 0.4, 0.7], [0.5, 0.6, 0.9]]
    cloud = observations(measured, sigmas, [1, 3, 2])
    weights = [1, 2, 3]
    rotation = rotation_vector([0.2, -0.1, 0.3])
    translation = np.array([0.1, 0.2, -0.2])
    configuration = options(intrinsic=0.1, cutoff=20)
    point_model = model(emitters, weights)
    result = point_model.evaluate(cloud, configuration, pose(rotation, translation))
    epsilon = 1e-6
    numerical_pose = []
    for k in range(6):
        plus_translation, minus_translation = translation.copy(), translation.copy()
        plus_rotation, minus_rotation = rotation.copy(), rotation.copy()
        if k < 3:
            plus_translation[k] += epsilon
            minus_translation[k] -= epsilon
        else:
            omega = np.zeros(3)
            omega[k-3] = epsilon
            plus_rotation = rotation_vector(omega) @ rotation
            minus_rotation = rotation_vector(-omega) @ rotation
        plus = point_model.evaluate(cloud, configuration, pose(plus_rotation, plus_translation)).mean_nll
        minus = point_model.evaluate(cloud, configuration, pose(minus_rotation, minus_translation)).mean_nll
        numerical_pose.append((plus-minus)/(2*epsilon))
    np.testing.assert_allclose(result.pose_gradient, numerical_pose, rtol=1e-6, atol=4e-10)
    numerical_model = []
    for k in range(emitters.size):
        plus, minus = emitters.copy(), emitters.copy()
        plus.ravel()[k] += epsilon
        minus.ravel()[k] -= epsilon
        upper = model(plus, weights).evaluate(cloud, configuration, pose(rotation, translation)).mean_nll
        lower = model(minus, weights).evaluate(cloud, configuration, pose(rotation, translation)).mean_nll
        numerical_model.append((upper-lower)/(2*epsilon))
    np.testing.assert_allclose(result.model_gradient, numerical_model, rtol=1e-6, atol=5e-10)
    information = np.asarray(result.pose_information).reshape(6, 6)
    np.testing.assert_allclose(information, information.T, atol=1e-14)
    assert np.linalg.eigvalsh(information).min() > -1e-12
    configuration.compute_model_gradient = False
    cheap = point_model.evaluate(cloud, configuration, pose(rotation, translation))
    assert len(cheap.model_gradient) == 0
    np.testing.assert_allclose(cheap.pose_gradient, result.pose_gradient)
    assert cheap.mean_nll == result.mean_nll


def test_zero_background_zero_support_is_infinite_without_fabricated_floor():
    point_model = model([[0, 0, 0]])
    cloud = observations([[9, 9, 9]], [0.1])
    result = point_model.evaluate(cloud, options(background=0))
    assert list(result.pdf) == [0]
    assert result.mean_nll == math.inf
    assert result.log_likelihood == -math.inf
    assert result.weighted_log_likelihood == -math.inf
    assert list(result.pose_gradient) == [0]*6
    assert list(result.model_gradient) == [0]*3
    fit = bff.fit_smlm_likelihood_rigid(cloud, point_model, options(background=0))
    assert not fit.converged
    assert fit.iterations == 0


def test_log_space_retains_finite_log_likelihood_when_pdf_underflows():
    # A large physical unit makes the Gaussian coefficient subnormal, while
    # its logarithm remains finite. No arbitrary floor can replace that PDF.
    point_model = model([[0, 0, 0]])
    cloud = observations([[0, 0, 0]], [1e110])
    configuration = options(background=0)
    result = point_model.evaluate(cloud, configuration)
    assert result.pdf[0] == 0
    expected = -1.5*np.log(2*np.pi)-3*np.log(1e110)
    assert result.log_pdf[0] == pytest.approx(expected, rel=1e-14)
    assert result.mean_nll == pytest.approx(-expected)
    assert result.supported_localizations == 1


def test_model_weights_normalized_once_and_owned():
    coordinates, weights = [-1., 0, 0, 1., 0, 0], [1., 3.]
    point_model = bff.SMLMPointModel(coordinates, weights)
    coordinates[0], weights[0] = 100, 0
    assert list(point_model.get_coordinates()) == [-1, 0, 0, 1, 0, 0]
    assert list(point_model.get_weights()) == [0.25, 0.75]
    assert point_model.get_number_of_points() == 2
    huge = model([[0, 0, 0], [1, 0, 0]], [1e308, 1e308])
    assert list(huge.get_weights()) == [0.5, 0.5]


def test_observation_weights_affect_mean_nll_but_not_actual_logL_or_count():
    point_model = model([[0, 0, 0]])
    measured = [[0.1, 0, 0], [1.2, 0, 0]]
    sigmas = [0.2, 0.4]
    one = point_model.evaluate(observations(measured, sigmas), options())
    two = point_model.evaluate(observations(measured, sigmas, [4, 0]), options())
    assert one.log_likelihood == two.log_likelihood
    assert one.localization_count == two.localization_count == 2
    assert two.mean_nll == pytest.approx(-two.log_pdf[0])
    assert one.mean_nll != pytest.approx(two.mean_nll)
    zero = point_model.evaluate(observations(measured, sigmas, [0, 0]), options())
    assert zero.mean_nll == zero.weighted_log_likelihood == 0
    assert zero.log_likelihood == one.log_likelihood
    assert list(zero.pose_gradient) == [0]*6
    empty = point_model.evaluate(observations([], []), options())
    assert empty.mean_nll == empty.log_likelihood == 0
    assert empty.localization_count == 0
    assert list(empty.pdf) == []


@pytest.mark.parametrize("unit_scale", [1.0, 100.0])
def test_rigid_fit_recovers_translation_and_rotation_with_heterosigma(unit_scale):
    emitters = unit_scale * np.array([[-2, -1, -0.5], [2, -0.5, 0],
                                      [0.5, 2, 0.3], [-0.8, 0.4, 2]])
    true_rotation = rotation_vector([0.12, -0.09, 0.15])
    true_translation = unit_scale * np.array([0.3, -0.2, 0.15])
    measured = np.repeat(emitters @ true_rotation.T + true_translation, 4, axis=0)
    sigma = unit_scale * np.tile([[0.15, 0.25, 0.35], [0.3, 0.2, 0.2],
                                  [0.2, 0.25, 0.2], [0.25, 0.2, 0.3]], (4, 1))
    point_model, cloud = model(emitters), observations(measured, sigma)
    configuration = options(background=0.02, bounds=(-20*unit_scale, 20*unit_scale))
    fit_options = bff.SMLMLikelihoodFitOptions()
    fit_options.max_rotation_step = 0.05
    fit_options.max_translation_step = 0.1*unit_scale
    fit_options.tolerance = 1e-9
    fitted = bff.fit_smlm_likelihood_rigid(cloud, point_model, configuration, [], fit_options)
    assert fitted.converged
    assert 0 < fitted.iterations <= fit_options.max_iterations
    assert fitted.likelihood.mean_nll < fitted.initial_mean_nll
    observed = np.asarray(fitted.transform).reshape(3, 4)
    np.testing.assert_allclose(observed[:, :3], true_rotation, atol=3e-7)
    np.testing.assert_allclose(observed[:, 3], true_translation, atol=3e-7*unit_scale)
    np.testing.assert_allclose(observed[:, :3].T @ observed[:, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(observed[:, :3]) == pytest.approx(1, abs=1e-12)
    direct = point_model.evaluate(cloud, configuration, fitted.transform)
    assert direct.mean_nll == fitted.likelihood.mean_nll
    assert len(fitted.likelihood.model_gradient) == emitters.size


def test_rigid_fit_requires_initial_overlap_and_honors_zero_iterations():
    point_model = model([[0, 0, 0], [1, 1, 1]])
    cloud = observations([[10, 10, 10]], [0.1])
    result = bff.fit_smlm_likelihood_rigid(cloud, point_model, options())
    assert not result.converged
    assert result.iterations == 0
    assert result.likelihood.supported_localizations == 0
    assert list(result.transform) == pose()
    fit_options = bff.SMLMLikelihoodFitOptions()
    fit_options.max_iterations = 0
    initial = pose(rotation_vector([0.1, 0.2, 0.3]), (0.2, -0.1, 0.1))
    result = bff.fit_smlm_likelihood_rigid(observations([[0, 0, 0]], [0.2]),
                                         point_model, options(), initial, fit_options)
    assert result.iterations == 0
    assert not result.converged
    assert result.likelihood.mean_nll == result.initial_mean_nll
    assert list(result.transform) == initial


def test_aic_and_aicc_use_summed_logL_actual_count_and_integer_free_parameters():
    log_likelihood, count, parameters = -123.4, 57, 6
    aic = 2*parameters-2*log_likelihood
    aicc = aic + 2*parameters*(parameters+1)/(count-parameters-1)
    assert bff.smlm_aic(log_likelihood, parameters) == pytest.approx(aic)
    assert bff.smlm_aicc(log_likelihood, count, parameters) == pytest.approx(aicc)
    for n in (0, parameters, parameters+1):
        assert bff.smlm_aicc(log_likelihood, n, parameters) == math.inf
    assert bff.smlm_aicc(log_likelihood, parameters+2, parameters) == \
        pytest.approx(aic+2*parameters*(parameters+1))
    assert bff.smlm_aic(-math.inf, parameters) == math.inf
    assert bff.smlm_aicc(-math.inf, count, parameters) == math.inf


@pytest.mark.parametrize("coordinates,weights,leaf", [
    ([], [], 16), ([0, 0], [], 16), ([0, 0, float("nan")], [], 16),
    ([0, 0, 0], [0], 16), ([0, 0, 0], [-1], 16),
    ([0, 0, 0], [float("inf")], 16), ([0, 0, 0], [1, 2], 16),
    ([0, 0, 0], [], 0),
])
def test_point_model_rejects_invalid_inputs(coordinates, weights, leaf):
    with pytest.raises((ValueError, RuntimeError)):
        bff.SMLMPointModel(coordinates, weights, leaf)


@pytest.mark.parametrize("field,value", [
    ("roi_min", []), ("roi_min", [20, 20, 20]),
    ("roi_max", [float("inf"), 20, 20]), ("roi_max", [-20, 20, 20]),
    ("background_fraction", -0.1), ("background_fraction", 1),
    ("background_fraction", float("nan")), ("intrinsic_sigma", -1),
    ("intrinsic_sigma", float("inf")), ("cutoff_sigma", 0),
    ("cutoff_sigma", float("nan")),
])
def test_likelihood_rejects_invalid_options(field, value):
    configuration = options()
    setattr(configuration, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        model([[0, 0, 0]]).evaluate(observations([[0, 0, 0]], [0.2]), configuration)


def test_outside_roi_is_rejected_instead_of_silently_filtering_even_zero_weight_rows():
    point_model = model([[0, 0, 0]])
    for weight in (0, 1):
        with pytest.raises((ValueError, RuntimeError)):
            point_model.evaluate(observations([[21, 0, 0]], [0.2], [weight]), options())


@pytest.mark.parametrize("transform", [
    [1]*11, pose(np.diag([-1, 1, 1])), pose(np.diag([2, 1, 1])),
    pose(translation=(float("nan"), 0, 0)),
])
def test_likelihood_rejects_improper_poses(transform):
    with pytest.raises((ValueError, RuntimeError)):
        model([[0, 0, 0]]).evaluate(observations([[0, 0, 0]], [0.2]), options(), transform)


@pytest.mark.parametrize("field,value", [
    ("max_iterations", -1), ("max_translation_step", 0),
    ("max_rotation_step", float("inf")), ("tolerance", -1), ("max_backtracks", 0),
])
def test_rigid_fit_rejects_invalid_iteration_and_step_limits(field, value):
    fit_options = bff.SMLMLikelihoodFitOptions()
    setattr(fit_options, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        bff.fit_smlm_likelihood_rigid(observations([[0, 0, 0]], [0.2]),
                                     model([[0, 0, 0]]), options(), [], fit_options)


@pytest.mark.parametrize("logL,count,parameters", [
    (float("nan"), 10, 3), (float("inf"), 10, 3), (-2, -1, 3), (-2, 10, -1),
])
def test_aicc_rejects_invalid_numbers(logL, count, parameters):
    with pytest.raises((ValueError, RuntimeError)):
        bff.smlm_aicc(logL, count, parameters)
