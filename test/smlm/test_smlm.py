"""Native 3D localization search, likelihoods and selective particle density.

The reference kernels deliberately enumerate all points/voxels; the native
implementation must match them while using spatial/support pruning.
"""

import math

import numpy as np
import pytest

from IMP import bff


def flat(values):
    return np.asarray(values, dtype=float).ravel().tolist()


def make_index(coordinates, sigmas, weights=None, ids=None, leaf=2):
    return bff.SMLMIndex(flat(coordinates), flat(sigmas),
                         [] if weights is None else flat(weights),
                         [] if ids is None else list(ids), leaf)


def transform(rotation=None, translation=(0, 0, 0)):
    return flat(np.column_stack((np.eye(3) if rotation is None else rotation,
                                 np.asarray(translation))))


def brute_density(points, sigmas, weights, queries, cutoff=4, normalize=True):
    points, sigmas, queries = map(np.asarray, (points, sigmas, queries))
    weights = np.asarray(weights, dtype=float)
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    differences = (queries[:, None, :] - points[None, :, :]) / sigmas
    q2 = np.sum(differences ** 2, axis=2)
    kernels = np.exp(-0.5 * q2) / ((2 * np.pi) ** 1.5 * sigmas.prod(axis=1))
    return (kernels * (q2 <= cutoff ** 2) * weights).sum(axis=1)


def average(index, ids, transforms=None, shape=(15, 17, 19),
            origin=(-3.5, -4.0, -4.5), spacing=(0.5, 0.5, 0.5), options=None):
    args = (index, list(ids), [] if transforms is None else flat(transforms),
            list(shape), flat(origin), flat(spacing))
    return bff.average_smlm_particles(*args) if options is None else \
        bff.average_smlm_particles(*args, options)


def voxel_points(shape, origin, spacing):
    # x-fast result storage, matching MRC storage. Numpy view is (nz, ny, nx).
    z, y, x = np.indices(tuple(reversed(shape)))
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())) * spacing + origin


def test_tree_kde_matches_brute_force_with_heterogeneous_anisotropic_sigmas():
    rng = np.random.default_rng(2301)
    points = rng.normal(size=(127, 3)) * (2, 6, 0.5)
    sigmas = rng.uniform(0.1, 1.8, size=points.shape)
    weights = rng.uniform(0, 4, len(points))
    weights[::11] = 0
    queries = rng.normal(size=(55, 3)) * (3, 9, 1)
    for leaf in (1, 7, 1000):
        index = make_index(points, sigmas, weights, leaf=leaf)
        for cutoff in (1.2, 4.0, 6.0):
            for normalize in (False, True):
                expected = brute_density(points, sigmas, weights, queries,
                                         cutoff, normalize)
                observed = index.evaluate_density(flat(queries), cutoff, normalize)
                np.testing.assert_allclose(observed, expected, rtol=3e-13, atol=1e-16)


def test_sigma_support_extends_far_outside_coordinate_bounds():
    points = np.array([[0, 0, 0], [100, 0, 0], [110, 0, 0], [120, 0, 0]])
    sigmas = np.array([[50, 0.1, 0.2], [0.1, 3, 4], [0.1, 3, 4], [0.1, 3, 4]])
    query = np.array([[-150, 0.02, 0.1]])
    index = make_index(points, sigmas, leaf=1)
    expected = brute_density(points, sigmas, np.ones(4), query)
    assert expected[0] > 0
    np.testing.assert_allclose(index.evaluate_density(flat(query)), expected, rtol=1e-13)


def test_radius_and_nearest_match_brute_force_with_stable_ties():
    rng = np.random.default_rng(33)
    points = rng.normal(size=(103, 3))
    points[:4] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    index = make_index(points, np.ones_like(points), leaf=1)
    for query in ([0, 0, 0], [3, -1, 0.2]):
        distances = np.linalg.norm(points - query, axis=1)
        for radius in (0, 0.5, 1, 20):
            assert list(index.radius_search(query, radius)) == \
                np.flatnonzero(distances <= radius).tolist()
        for k in (1, 5, 500):
            for limit in (-1, 0.5, 1, 20):
                expected = sorted(range(len(points)), key=lambda i: (distances[i], i))
                if limit >= 0:
                    expected = [i for i in expected if distances[i] <= limit]
                assert list(index.nearest_search(query, k, limit)) == expected[:k]
    tied = make_index(points[:4], np.ones((4, 3)), leaf=1)
    assert list(tied.nearest_search([0, 0, 0], 3)) == [0, 1, 2]


def test_score_gradient_is_weighted_mean_log_density_derivative():
    points = np.array([[0, 0, 0], [1, 0.3, -0.5], [-0.4, 1, 0.2]])
    sigmas = np.array([[0.8, 1, 1.2], [1.1, 0.7, 0.9], [0.7, 0.9, 0.6]])
    weights = np.array([2, 3, 1])
    queries = np.array([[0.2, 0.1, -0.1], [0.5, 0.1, 0.3]])
    qweights = [1, 3]
    index = make_index(points, sigmas, weights)
    background = 0.02
    result = index.evaluate_score(flat(queries), qweights, 6, background)
    densities = brute_density(points, sigmas, weights, queries, 6)
    assert result.score == pytest.approx(-np.average(np.log(densities + background),
                                                    weights=qweights), rel=1e-13)
    np.testing.assert_allclose(result.densities, densities, rtol=1e-13)
    epsilon = 1e-6
    numerical = []
    for i in range(queries.size):
        plus, minus = flat(queries), flat(queries)
        plus[i] += epsilon
        minus[i] -= epsilon
        numerical.append((index.evaluate_score(plus, qweights, 6, background).score -
                          index.evaluate_score(minus, qweights, 6, background).score) /
                         (2 * epsilon))
    np.testing.assert_allclose(result.gradient, numerical, rtol=3e-6, atol=3e-10)


def test_background_floor_and_zero_mass_queries_are_stable():
    index = make_index([[0, 0, 0]], [1])
    result = index.evaluate_score([100, 100, 100], [], 4, 1e-250)
    assert result.score == pytest.approx(-math.log(1e-250))
    assert list(result.gradient) == [0, 0, 0]
    empty = make_index([], [])
    assert list(empty.radius_search([0, 0, 0], 2)) == []
    assert list(empty.nearest_search([0, 0, 0])) == []
    assert list(empty.evaluate_density([0, 0, 0])) == [0]
    assert empty.evaluate_score([]).score == 0
    zero = make_index([[0, 0, 0]], [1], [0])
    assert list(zero.evaluate_density([0, 0, 0])) == [0]
    assert zero.evaluate_score([0, 0, 0]).score == pytest.approx(-math.log(1e-12))
    unweighted = index.evaluate_score([0.2, 0.1, 0.1], [0])
    assert unweighted.score == 0
    assert list(unweighted.gradient) == [0, 0, 0]


def test_owned_input_arrays_and_isotropic_expansion():
    points, sigmas, weights, ids = [1., 2., 3.], [0.5], [2.], [7]
    index = bff.SMLMIndex(points, sigmas, weights, ids)
    points[0], sigmas[0], weights[0], ids[0] = 200, 20, 0, 70
    assert list(index.get_coordinates()) == [1, 2, 3]
    assert list(index.get_sigmas()) == [0.5, 0.5, 0.5]
    assert list(index.get_weights()) == [2]
    assert list(index.get_particle_ids()) == [7]
    assert list(index.get_particle_localizations(7)) == [0]
    assert list(index.get_particle_localizations(-1)) == []


@pytest.mark.parametrize("coordinates,sigmas,weights,ids,leaf", [
    ([0, 0], [1], [], [], 16),  # 2D rejected
    ([0, 0, float("nan")], [1], [], [], 16),
    ([0, 0, 0], [], [], [], 16),
    ([0, 0, 0], [0], [], [], 16),
    ([0, 0, 0], [-1], [], [], 16),
    ([0, 0, 0], [float("inf")], [], [], 16),
    ([0, 0, 0], [1, 1], [], [], 16),
    ([0, 0, 0], [1], [-1], [], 16),
    ([0, 0, 0], [1], [float("nan")], [], 16),
    ([0, 0, 0], [1], [1, 1], [], 16),
    ([0, 0, 0], [1], [], [1, 2], 16),
    ([0, 0, 0], [1], [], [], 0),
])
def test_index_rejects_invalid_inputs(coordinates, sigmas, weights, ids, leaf):
    with pytest.raises((ValueError, RuntimeError)):
        bff.SMLMIndex(coordinates, sigmas, weights, ids, leaf)


@pytest.mark.parametrize("operation", [
    lambda i: i.radius_search([0, 0], 1),
    lambda i: i.radius_search([0, 0, 0], -1),
    lambda i: i.radius_search([0, 0, 0], float("inf")),
    lambda i: i.nearest_search([0, 0, 0], 0),
    lambda i: i.nearest_search([0, 0, float("nan")]),
    lambda i: i.evaluate_density([0, 0]),
    lambda i: i.evaluate_density([0, 0, 0], 0),
    lambda i: i.evaluate_score([0, 0, 0], [1, 2]),
    lambda i: i.evaluate_score([0, 0, 0], [-1]),
    lambda i: i.evaluate_score([0, 0, 0], [], 4, 0),
])
def test_search_and_score_reject_invalid_arguments(operation):
    with pytest.raises((ValueError, RuntimeError)):
        operation(make_index([[0, 0, 0]], [1]))


def test_provided_rigid_transforms_align_translated_rotated_particles():
    points = np.array([[-1, 0, 0], [1, 0.3, 0], [0, -0.4, 0.8]])
    # 90-degree rotation keeps transformed diagonal sigmas exactly permuted.
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([3, -2, 1])
    second = points @ rotation.T + translation
    sigma = np.tile([0.35, 0.5, 0.6], (3, 1))
    second_sigma = sigma[:, [1, 0, 2]]
    index = make_index(np.concatenate((points, second, [[20, 20, 20], [0, 0, 0]])),
                       np.concatenate((sigma, second_sigma, [[1, 1, 1], [1, 1, 1]])),
                       ids=[4]*3 + [9]*3 + [23, -1])
    result = average(index, [4, 9], [transform(),
                                    transform(rotation.T, -rotation.T @ translation)])
    queries = voxel_points((15, 17, 19), (-3.5, -4, -4.5), (0.5, 0.5, 0.5))
    expected = brute_density(points, sigma, np.ones(3), queries)
    np.testing.assert_allclose(result.values, expected, rtol=5e-13, atol=2e-16)
    np.testing.assert_allclose(result.half1, expected, rtol=5e-13, atol=2e-16)
    np.testing.assert_allclose(result.half2, expected, rtol=5e-13, atol=2e-16)
    assert list(result.selected_ids) == [4, 9]
    assert result.selected_localizations == 6
    assert list(result.localization_counts) == [3, 3]
    assert list(result.half1_particle_ids) == [4]
    assert list(result.half2_particle_ids) == [9]
    assert result.half1_localizations == result.half2_localizations == 3


def test_full_covariance_orientation_includes_offdiagonal_terms():
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    sigma = np.array([1.2, 0.2, 0.3])
    index = make_index([[0, 0, 0]], [sigma], ids=[5])
    shape, origin, spacing = (25, 25, 9), (-3, -3, -1), (0.25, 0.25, 0.25)
    result = average(index, [5], [transform(rotation)], shape, origin, spacing)
    queries = voxel_points(shape, origin, spacing)
    # Reference offsets transformed back into the source principal axes.
    expected = brute_density([[0, 0, 0]], [sigma], [1], queries @ rotation)
    np.testing.assert_allclose(result.values, expected, rtol=3e-13, atol=1e-15)
    grid = np.asarray(result.values).reshape(tuple(reversed(shape)))
    assert grid[4, 16, 16] > 100 * grid[4, 8, 16]
    voxel_volume = np.prod(spacing)
    assert np.sum(result.values) * voxel_volume == pytest.approx(1, abs=0.006)
    mass = np.sum(result.values)
    mean = np.sum(queries * np.asarray(result.values)[:, None], axis=0) / mass
    covariance = ((queries - mean).T * np.asarray(result.values)) @ (queries - mean) / mass
    expected_covariance = rotation @ np.diag(sigma ** 2) @ rotation.T
    np.testing.assert_allclose(covariance, expected_covariance, rtol=0.04, atol=0.01)


def test_x_fast_order_and_anisotropic_spacing_on_asymmetric_grid():
    shape, origin, spacing = (7, 4, 3), (1.1, -2.3, 7), (0.4, 0.7, 1.2)
    centre = np.array(origin) + np.array(spacing) * (4, 1, 2)
    sigma = [[0.15, 0.3, 0.45]]
    index = make_index([centre], sigma, ids=[7])
    result = average(index, [7], shape=shape, origin=origin, spacing=spacing)
    expected = brute_density([centre], sigma, [1], voxel_points(shape, origin, spacing))
    np.testing.assert_allclose(result.values, expected, rtol=1e-13, atol=1e-14)
    assert np.argmax(result.values) == 4 + 7 * (1 + 4 * 2)
    assert list(result.grid_shape) == list(shape)
    assert list(result.origin) == list(origin)
    assert list(result.spacing) == list(spacing)


def test_particle_vs_localization_weight_normalization_and_half_maps():
    index = make_index([[-1, 0, 0], [1, 0, 0], [1, 0, 0], [10, 0, 0], [0, 0, 0]],
                       np.full((5, 3), 0.4), [2, 3, 5, 50, 100], [2, 8, 8, 9, -1])
    shape, origin, spacing = (17, 3, 3), (-2, -0.25, -0.25), (0.25, 0.25, 0.25)
    queries = voxel_points(shape, origin, spacing)
    left = brute_density([[-1, 0, 0]], [[0.4]*3], [1], queries)
    right = brute_density([[1, 0, 0]], [[0.4]*3], [1], queries)
    equal = average(index, [2, 8], shape=shape, origin=origin, spacing=spacing)
    np.testing.assert_allclose(equal.values, (left + right) / 2, atol=2e-15)
    np.testing.assert_allclose(equal.half1, left, atol=2e-15)
    np.testing.assert_allclose(equal.half2, right, atol=2e-15)
    options = bff.SMLMAverageOptions()
    options.normalize_particles = False
    weighted = average(index, [2, 8], shape=shape, origin=origin, spacing=spacing,
                       options=options)
    np.testing.assert_allclose(weighted.values, (2*left + 8*right)/10, atol=2e-15)
    np.testing.assert_allclose(weighted.half1, left, atol=2e-15)
    np.testing.assert_allclose(weighted.half2, right, atol=2e-15)
    options.normalize_density = False
    summed = average(index, [2, 8], shape=shape, origin=origin, spacing=spacing,
                     options=options)
    np.testing.assert_allclose(summed.values, 2*left + 8*right, atol=2e-14)
    np.testing.assert_allclose(summed.half1, 2*left, atol=2e-14)
    np.testing.assert_allclose(summed.half2, 8*right, atol=2e-14)


def test_quality_gates_exclude_particles_before_normalization_and_half_assignment():
    index = make_index(np.zeros((7, 3)), np.ones((7, 3)),
                       [1, 1, 1, 1, 1, 0, 0], [3, 3, 8, 9, 9, 10, 10])
    options = bff.SMLMAverageOptions()
    options.min_localizations = options.max_localizations = 2
    options.use_score_gate = True
    options.min_score, options.max_score = 0.5, 1.0
    options.particle_scores = [0.9, 0.9, 0.1, 0.9]
    result = average(index, [3, 8, 9, 10], options=options)
    reference = average(index, [3])
    assert list(result.selected_ids) == [3]
    assert list(result.rejected_ids) == [8, 9, 10]
    assert list(result.half1_particle_ids) == [3]
    assert list(result.half2_particle_ids) == []
    np.testing.assert_allclose(result.values, reference.values)
    np.testing.assert_allclose(result.half1, reference.values)
    assert np.count_nonzero(result.half2) == 0
    options.min_score = 0.95
    empty = average(index, [3, 8, 9, 10], options=options)
    assert list(empty.selected_ids) == []
    assert list(empty.rejected_ids) == [3, 8, 9, 10]
    assert np.count_nonzero(empty.values) == 0


def test_half_maps_split_particles_not_localizations():
    points = [[-1, 0, 0]] * 3 + [[0, 0, 0]] * 2 + [[1, 0, 0]] * 4
    index = make_index(points, np.full((9, 3), 0.4), ids=[4]*3 + [7]*2 + [8]*4)
    result = average(index, [4, 7, 8])
    first = average(index, [4, 8])
    second = average(index, [7])
    assert list(result.half1_particle_ids) == [4, 8]
    assert list(result.half2_particle_ids) == [7]
    assert result.half1_localizations == 7
    assert result.half2_localizations == 2
    np.testing.assert_allclose(result.half1, first.values)
    np.testing.assert_allclose(result.half2, second.values)
    np.testing.assert_allclose(result.values,
                               (2*np.asarray(first.values) + np.asarray(second.values))/3)


@pytest.mark.parametrize("ids,transforms,shape,origin,spacing", [
    ([], [], (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1, 1], [], (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([-1], [], (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([9], [], (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], [1]*11, (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], transform(np.diag([2, 1, 1])), (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], transform(np.diag([-1, 1, 1])), (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], transform(translation=(float("nan"), 0, 0)), (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], [], (0, 2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], [], (2, 2), (0, 0, 0), (1, 1, 1)),
    ([1], [], (2, 2, 2), (0, 0, 0), (0, 1, 1)),
    ([1], [], (2, 2, 2), (0, 0, float("inf")), (1, 1, 1)),
])
def test_average_rejects_invalid_selection_rigid_transforms_and_grids(
        ids, transforms, shape, origin, spacing):
    index = make_index([[0, 0, 0]], [1], ids=[1])
    with pytest.raises((ValueError, RuntimeError)):
        average(index, ids, transforms, shape, origin, spacing)


@pytest.mark.parametrize("field,value", [
    ("cutoff_sigma", -1), ("min_localizations", -1),
    ("max_localizations", -1), ("use_score_gate", True),
    ("particle_scores", [1, 2]), ("particle_scores", [float("nan")]),
])
def test_average_rejects_invalid_quality_options(field, value):
    index = make_index([[0, 0, 0]], [1], ids=[1])
    options = bff.SMLMAverageOptions()
    setattr(options, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        average(index, [1], options=options)


def test_grid_clipping_preserves_lost_mass_and_remote_kernels_are_skipped():
    index = make_index([[0, 0, 0], [1e100, -1e100, 0]], [1, 1], ids=[1, 2])
    result = average(index, [1, 2], shape=(11, 11, 11),
                     origin=(0, 0, 0), spacing=(0.2, 0.2, 0.2))
    expected = brute_density([[0, 0, 0]], [[1, 1, 1]], [1],
                             voxel_points((11, 11, 11), (0, 0, 0), (0.2, 0.2, 0.2))) / 2
    np.testing.assert_allclose(result.values, expected, atol=1e-15)
    assert np.sum(result.values) * 0.2**3 < 0.15
    assert np.count_nonzero(result.half2) == 0


def test_native_translation_refinement_improves_score_and_preserves_rotation():
    points = np.array([[-2, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 3]])
    index = make_index(points, np.full((4, 3), 0.25))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    offset = np.array([0.3, -0.2, 0.15])
    moving = (points + offset) @ rotation
    options = bff.SMLMRegistrationOptions()
    options.max_translation_step = 0.15
    options.max_iterations = 100
    options.tolerance = 1e-7
    result = bff.refine_smlm_translation(index, flat(moving), transform(rotation), [], options)
    observed_transform = np.asarray(result.transform).reshape(3, 4)
    np.testing.assert_allclose(observed_transform[:, :3], rotation)
    np.testing.assert_allclose(observed_transform[:, 3], -offset, atol=2e-6)
    assert result.score < result.initial_score
    assert result.iterations > 0
    assert result.converged
    assert result.score == pytest.approx(index.evaluate_score(flat(points)).score, abs=1e-9)
    options.max_iterations = 0
    unchanged = bff.refine_smlm_translation(index, flat(moving), transform(rotation), [], options)
    assert unchanged.iterations == 0
    assert unchanged.score == unchanged.initial_score
    assert list(unchanged.transform) == transform(rotation)


def test_translation_refinement_does_not_claim_background_plateau_convergence():
    index = make_index([[0, 0, 0]], [0.5])
    result = bff.refine_smlm_translation(index, [100, 100, 100])
    assert not result.converged
    assert result.iterations == 0
    assert list(result.transform) == transform()


@pytest.mark.parametrize("field,value", [
    ("max_iterations", -1), ("max_translation_step", 0),
    ("tolerance", float("nan")), ("cutoff_sigma", 0), ("background", 0),
])
def test_registration_rejects_invalid_options(field, value):
    index = make_index([[0, 0, 0]], [0.5])
    options = bff.SMLMRegistrationOptions()
    setattr(options, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        bff.refine_smlm_translation(index, [0.1, 0.2, 0.1], [], [], options)
