"""Exact Gaussian-pair overlap through a persistent owned localization tree."""

import gc
import math
import time

import numpy as np
import pytest

from IMP import bff


def flat(values):
    return np.asarray(values, dtype=float).ravel().tolist()


def make_overlap(points, sigmas, weights=None, leaf=2):
    index = bff.SMLMIndex(flat(points), flat(sigmas),
                         [] if weights is None else flat(weights), [], leaf)
    return bff.SMLMGaussianOverlap(index)


def brute_overlap(points, sigmas, data_weights, centres, covariances,
                  model_weights, cutoff=6, background=1e-12):
    """Enumerate all Gaussian pairs using full covariance sums, without a tree."""
    points, sigmas, centres, covariances = map(
        np.asarray, (points, sigmas, centres, covariances))
    data_weights = np.asarray(data_weights, dtype=float)
    model_weights = np.asarray(model_weights, dtype=float)
    if data_weights.sum() > 0:
        data_weights = data_weights / data_weights.sum()
    if model_weights.sum() > 0:
        model_weights = model_weights / model_weights.sum()
    component_overlaps = np.zeros(len(centres))
    derivative = np.zeros_like(centres, dtype=float)
    for j, (centre, covariance) in enumerate(zip(centres, covariances)):
        for point, sigma, weight in zip(points, sigmas, data_weights):
            covariance_sum = covariance + np.diag(sigma ** 2)
            delta = centre - point
            inverse_delta = np.linalg.solve(covariance_sum, delta)
            mahalanobis_squared = delta @ inverse_delta
            if mahalanobis_squared > cutoff ** 2:
                continue
            kernel = (math.exp(-0.5 * mahalanobis_squared) /
                      math.sqrt((2 * math.pi) ** 3 * np.linalg.det(covariance_sum)))
            contribution = weight * kernel
            component_overlaps[j] += contribution
            derivative[j] -= model_weights[j] * contribution * inverse_delta
    overlap = model_weights @ component_overlaps
    return (overlap, -math.log(overlap + background), component_overlaps,
            -derivative / (overlap + background))


@pytest.mark.parametrize("leaf", [1, 7, 1000])
@pytest.mark.parametrize("cutoff", [1.2, 3.0, 6.0])
def test_pair_covariance_sum_matches_brute_force(leaf, cutoff):
    rng = np.random.default_rng(6107)
    points = rng.normal(size=(83, 3)) * [2, 4, 0.5]
    sigmas = rng.uniform(0.08, 2, points.shape)
    data_weights = rng.uniform(0, 5, len(points))
    data_weights[::9] = 0
    centres = rng.normal(size=(19, 3)) * [3, 5, 1]
    matrices = rng.normal(size=(len(centres), 3, 3))
    covariances = matrices @ matrices.transpose(0, 2, 1)
    covariances[0] = 0
    covariances[1] = np.outer([1, 2, -0.5], [1, 2, -0.5])
    model_weights = rng.uniform(0, 3, len(centres))
    model_weights[::5] = 0
    scorer = make_overlap(points, sigmas, data_weights, leaf)
    observed = scorer.evaluate(flat(centres), flat(covariances),
                               flat(model_weights), cutoff, 0.002)
    overlap, score, components, gradient = brute_overlap(
        points, sigmas, data_weights, centres, covariances, model_weights,
        cutoff, 0.002)
    assert observed.overlap == pytest.approx(overlap, rel=4e-12, abs=1e-16)
    assert observed.score == pytest.approx(score, rel=2e-13)
    np.testing.assert_allclose(observed.component_overlaps, components,
                               rtol=4e-12, atol=2e-16)
    np.testing.assert_allclose(np.asarray(observed.gradient).reshape(-1, 3),
                               gradient, rtol=8e-11, atol=3e-14)


def test_rotated_covariance_support_cannot_be_pruned_as_diagonal_ellipsoid():
    # Along the correlated long axis, a diagonal-only Mahalanobis test would
    # discard this pair even though the true covariance puts it inside 3 sigma.
    points = np.array([[0, 0, 0], [100, -100, 0], [-100, 100, 0]])
    sigmas = np.full(points.shape, 0.1)
    centres = np.array([[25, 25, 0]])
    covariance = np.array([[[100, 99, 0], [99, 100, 0], [0, 0, 0.04]]])
    expected = brute_overlap(points, sigmas, np.ones(3), centres,
                             covariance, [1], 3)[0]
    assert expected > 0
    observed = make_overlap(points, sigmas, leaf=1).evaluate(
        flat(centres), flat(covariance), [], 3)
    assert observed.overlap == pytest.approx(expected, rel=3e-13)


def test_broad_observation_kernel_reaches_outside_node_coordinate_bounds():
    points = np.array([[0, 0, 0], [100, 0, 0], [110, 0, 0], [120, 0, 0]])
    sigmas = np.array([[50, 0.1, 0.2], [0.1, 3, 4], [0.1, 3, 4], [0.1, 3, 4]])
    centres = np.array([[-150, 0.02, 0.1]])
    covariances = np.array([np.diag([0.3, 0.1, 0.5])])
    expected = brute_overlap(points, sigmas, np.ones(4), centres, covariances,
                             [1], 4)[0]
    assert expected > 0
    observed = make_overlap(points, sigmas, leaf=1).evaluate(
        flat(centres), flat(covariances), [], 4)
    assert observed.overlap == pytest.approx(expected, rel=2e-13)


def test_analytic_score_gradient_matches_finite_difference_for_full_covariance():
    points = np.array([[0, 0, 0], [1, 0.3, -0.5], [-0.4, 1, 0.2]])
    sigmas = np.array([[0.8, 1, 1.2], [1.1, 0.7, 0.9], [0.7, 0.9, 0.6]])
    centres = np.array([[0.2, 0.1, -0.1], [0.5, 0.1, 0.3]])
    covariances = np.array([[[0.4, 0.15, -0.05], [0.15, 0.3, 0.08],
                            [-0.05, 0.08, 0.2]], np.diag([0.1, 0.2, 0.3])])
    scorer = make_overlap(points, sigmas, [2, 3, 1])
    result = scorer.evaluate(flat(centres), flat(covariances), [1, 3], 6, 0.02)
    numerical = []
    epsilon = 1e-6
    for k in range(centres.size):
        plus, minus = flat(centres), flat(centres)
        plus[k] += epsilon
        minus[k] -= epsilon
        numerical.append((scorer.evaluate(plus, flat(covariances), [1, 3], 6, 0.02).score -
                          scorer.evaluate(minus, flat(covariances), [1, 3], 6, 0.02).score) /
                         (2 * epsilon))
    np.testing.assert_allclose(result.gradient, numerical, rtol=3e-6, atol=3e-10)


def test_zero_model_covariance_is_point_kde_and_global_log_not_mean_log():
    points = [[0, 0, 0], [2, 0, 0]]
    sigmas = [[0.8, 1, 1.2], [1.1, 0.7, 0.9]]
    centres = [[0.1, 0.2, -0.1], [0.8, 0.1, 0.3]]
    index = bff.SMLMIndex(flat(points), flat(sigmas), [2, 3])
    result = bff.SMLMGaussianOverlap(index).evaluate(flat(centres), [0] * 18,
                                                   [1, 3], 6, 0.01)
    densities = np.asarray(index.evaluate_density(flat(centres), 6))
    np.testing.assert_allclose(result.component_overlaps, densities, rtol=3e-13)
    assert result.overlap == pytest.approx(np.average(densities, weights=[1, 3]))
    assert result.score == pytest.approx(-math.log(result.overlap + 0.01))
    assert abs(result.score - index.evaluate_score(flat(centres), [1, 3], 6, 0.01).score) > 1e-5


def test_normalized_weight_rescaling_and_component_splitting_invariance():
    points = np.array([[0, 0, 0], [1, 2, 0.5]])
    sigmas = np.array([[0.8, 1, 1.2], [1.1, 0.7, 0.9]])
    centres = np.array([[0.1, 0.2, -0.1], [0.8, 0.1, 0.3]])
    covariances = np.array([np.diag([0.2, 0.4, 0.1]), np.diag([0.5, 0.1, 0.3])])
    reference = make_overlap(points, sigmas, [2, 3]).evaluate(
        flat(centres), flat(covariances), [4, 7])
    scaled = make_overlap(points, sigmas, [20, 30]).evaluate(
        flat(centres), flat(covariances), [400, 700])
    split_data = make_overlap(np.repeat(points, 2, axis=0),
                              np.repeat(sigmas, 2, axis=0), [1, 1, 1.5, 1.5]).evaluate(
                                  flat(centres), flat(covariances), [4, 7])
    split_model = make_overlap(points, sigmas, [2, 3]).evaluate(
        flat(np.repeat(centres, 2, axis=0)),
        flat(np.repeat(covariances, 2, axis=0)), [2, 2, 3.5, 3.5])
    for result in (scaled, split_data, split_model):
        assert result.overlap == pytest.approx(reference.overlap, rel=3e-14)
        assert result.score == pytest.approx(reference.score, rel=3e-14)
    np.testing.assert_allclose(scaled.gradient, reference.gradient, rtol=3e-14)
    np.testing.assert_allclose(split_data.gradient, reference.gradient, rtol=3e-14)
    combined = np.asarray(split_model.gradient).reshape(2, 2, 3).sum(axis=1)
    np.testing.assert_allclose(combined.ravel(), reference.gradient, rtol=3e-14)


def test_model_weight_normalization_cannot_overflow():
    scorer = make_overlap([[0, 0, 0]], [1])
    centres = [0.2, 0.1, 0.3, 1, 0, 0]
    covariances = flat([np.eye(3), np.eye(3)])
    expected = scorer.evaluate(centres, covariances, [1, 1])
    observed = scorer.evaluate(centres, covariances, [np.finfo(float).max] * 2)
    assert observed.overlap == expected.overlap
    np.testing.assert_allclose(observed.gradient, expected.gradient)


def test_finite_overlap_plus_finite_background_cannot_overflow_score():
    target, background = 1e308, 1e308
    sigma = math.exp((-1.5 * math.log(2 * math.pi) - math.log(target)) / 3)
    scorer = make_overlap([[0, 0, 0]], [sigma])
    result = scorer.evaluate([0, 0, 0], [0] * 9, [], 6, background)
    assert math.isfinite(result.overlap)
    assert result.overlap == pytest.approx(target, rel=5e-13)
    expected_score = -math.log(result.overlap) - math.log1p(background / result.overlap)
    assert result.score == pytest.approx(expected_score, rel=2e-14)
    assert list(result.gradient) == [0, 0, 0]


@pytest.mark.parametrize("scale,background", [(1e100, 1e-310), (1e-100, 1e-12)])
def test_scaled_cholesky_avoids_variance_determinant_overflow_or_underflow(scale, background):
    points = np.array([[0, 0, 0]])
    centres = np.array([[0.1, -0.2, 0.3]])
    covariance = np.array([np.diag([0.2, 0.4, 0.1])])
    unit = make_overlap(points, [[0.8, 1, 1.2]]).evaluate(flat(centres), flat(covariance))
    observed = make_overlap(points * scale, np.array([[0.8, 1, 1.2]]) * scale).evaluate(
        flat(centres * scale), flat(covariance * scale ** 2), [], 6, background)
    expected_overlap = math.exp(math.log(unit.overlap) - 3 * math.log(scale))
    assert observed.overlap == pytest.approx(expected_overlap, rel=5e-13)
    ratio = expected_overlap / max(expected_overlap, background)
    expected_score = (-math.log(max(expected_overlap, background)) -
                      math.log(ratio + background / max(expected_overlap, background)))
    assert observed.score == pytest.approx(expected_score, rel=3e-14)
    assert np.all(np.isfinite(observed.gradient))
    attenuation = ratio / (ratio + background / max(expected_overlap, background))
    expected_gradient = (np.asarray(unit.gradient) * ((unit.overlap + 1e-12) / unit.overlap)
                         * attenuation) / scale
    np.testing.assert_allclose(observed.gradient, expected_gradient, rtol=6e-13, atol=0)


def test_unrepresentable_overlap_is_rejected_instead_of_returning_infinity():
    scorer = make_overlap([[0, 0, 0]], [1e-150])
    with pytest.raises((ValueError, RuntimeError)):
        scorer.evaluate([0, 0, 0], [0] * 9)


def test_owned_tree_survives_source_destruction_and_input_mutation():
    points, sigmas, weights = [0.0, 0.0, 0.0], [1.0], [2.0]
    index = bff.SMLMIndex(points, sigmas, weights)
    scorer = bff.SMLMGaussianOverlap(index)
    baseline = scorer.evaluate([0.2, 0.1, 0.3], flat(np.eye(3)))
    del index
    gc.collect()
    points[0], sigmas[0], weights[0] = 200, 20, 0
    assert scorer.get_number_of_localizations() == 1
    for _ in range(100):
        observed = scorer.evaluate([0.2, 0.1, 0.3], flat(np.eye(3)))
        assert observed.overlap == baseline.overlap
        np.testing.assert_array_equal(observed.gradient, baseline.gradient)


@pytest.mark.parametrize("data_zero,model_zero", [(True, False), (False, True), (True, True)])
def test_zero_mass_background_only_has_zero_score_gradient(data_zero, model_zero):
    scorer = make_overlap([[0, 0, 0]], [1], [0 if data_zero else 1])
    result = scorer.evaluate([0.2, 0.1, 0.3], flat(np.eye(3)),
                             [0 if model_zero else 1], 6, 1e-250)
    assert result.overlap == 0
    assert result.score == pytest.approx(-math.log(1e-250))
    assert list(result.gradient) == [0, 0, 0]


def test_empty_mixtures_and_remote_support_are_background_only():
    for scorer, centres, covariances in (
        (make_overlap([], []), [0, 0, 0], flat(np.eye(3))),
        (make_overlap([[0, 0, 0]], [1]), [], []),
        (make_overlap([[0, 0, 0]], [1]), [100, 100, 100], flat(np.eye(3))),
    ):
        result = scorer.evaluate(centres, covariances, [], 6, 0.01)
        assert result.overlap == 0
        assert result.score == pytest.approx(-math.log(0.01))
        assert not np.any(result.gradient)


def test_inclusive_mahalanobis_cutoff_and_zero_weight_gradient():
    scorer = make_overlap([[0, 0, 0]], [1])
    result = scorer.evaluate([6, 0, 0, 6.001, 0, 0], [0] * 18, [1, 0], 6)
    expected = math.exp(-18) / (2 * math.pi) ** 1.5
    np.testing.assert_allclose(result.component_overlaps, [expected, 0], rtol=2e-14)
    assert result.overlap == pytest.approx(expected, rel=2e-14)
    assert list(result.gradient)[3:] == [0, 0, 0]


@pytest.mark.parametrize("covariance", [
    [[1, 0.2, 0], [0.1, 1, 0], [0, 0, 1]],  # asymmetric
    np.diag([1, -0.001, 1]),
    [[1, 2, 0], [2, 1, 0], [0, 0, 1]],       # indefinite with positive diagonal
    np.diag([1, float("nan"), 1]),
    np.diag([1, float("inf"), 1]),
])
@pytest.mark.parametrize("weight", [0, 1])
def test_invalid_covariance_is_rejected_even_for_zero_weight(covariance, weight):
    scorer = make_overlap([[0, 0, 0]], [1])
    with pytest.raises((ValueError, RuntimeError)):
        scorer.evaluate([0, 0, 0], flat(covariance), [weight])


def test_semidefinite_covariance_and_roundoff_tolerance_are_explicit():
    scorer = make_overlap([[0, 0, 0]], [1])
    rank_one = np.outer([1, 1, 0], [1, 1, 0])
    assert scorer.evaluate([0.1, 0.2, 0.3], flat(rank_one)).overlap > 0
    near_psd = np.diag([1.0, 1.0, -1e-13])
    near_symmetric = np.eye(3)
    near_symmetric[0, 1] = 1e-13
    clamped = scorer.evaluate([0.1, 0.2, 0.3], flat(near_psd))
    expected = scorer.evaluate([0.1, 0.2, 0.3], flat(np.diag([1, 1, 0])))
    assert clamped.overlap == pytest.approx(expected.overlap, rel=2e-14)
    assert scorer.evaluate([0.1, 0.2, 0.3], flat(near_symmetric)).overlap > 0
    with pytest.raises((ValueError, RuntimeError)):
        scorer.evaluate([0, 0, 0], flat(np.diag([1e-20, -1e-23, 1e-20])))


@pytest.mark.parametrize("coordinates,covariances,weights,cutoff,background", [
    ([0, 0], [0] * 9, [], 6, 1e-12),
    ([0, 0, float("nan")], [0] * 9, [], 6, 1e-12),
    ([0, 0, 0], [0] * 3, [], 6, 1e-12),
    ([0, 0, 0], [0] * 9, [1, 2], 6, 1e-12),
    ([0, 0, 0], [0] * 9, [-1], 6, 1e-12),
    ([0, 0, 0], [0] * 9, [float("inf")], 6, 1e-12),
    ([0, 0, 0], [0] * 9, [], 0, 1e-12),
    ([0, 0, 0], [0] * 9, [], float("inf"), 1e-12),
    ([0, 0, 0], [0] * 9, [], 6, 0),
    ([0, 0, 0], [0] * 9, [], 6, float("nan")),
])
def test_invalid_arguments_are_rejected(coordinates, covariances, weights, cutoff, background):
    with pytest.raises((ValueError, RuntimeError)):
        make_overlap([[0, 0, 0]], [1]).evaluate(coordinates, covariances, weights,
                                              cutoff, background)


def test_unit_conversion_transforms_overlap_score_and_gradient_consistently():
    points = np.array([[0, 0, 0], [1, 2, 0.5]])
    sigmas = np.array([[0.8, 1, 1.2], [1.1, 0.7, 0.9]])
    centres = np.array([[0.1, 0.2, -0.1], [0.8, 0.1, 0.3]])
    covariances = np.array([np.diag([0.2, 0.4, 0.1]), np.diag([0.5, 0.1, 0.3])])
    a, background = 10.0, 0.01
    baseline = make_overlap(points, sigmas, [2, 3]).evaluate(
        flat(centres), flat(covariances), [4, 7], 6, background)
    scaled = make_overlap(points * a, sigmas * a, [2, 3]).evaluate(
        flat(centres * a), flat(covariances * a ** 2), [4, 7], 6, background / a ** 3)
    assert scaled.overlap == pytest.approx(baseline.overlap / a ** 3, rel=3e-14)
    assert scaled.score == pytest.approx(baseline.score + 3 * math.log(a), rel=3e-14)
    np.testing.assert_allclose(scaled.gradient, np.asarray(baseline.gradient) / a,
                               rtol=4e-14, atol=1e-16)


def test_persistent_100k_tree_reused_for_100_evaluations(capsys):
    # No timing threshold: this checks persistent ownership/reuse and supplies
    # reproducible benchmark timings without making machine speed a test oracle.
    count = 100_000
    axis = np.arange(count, dtype=float) * 10
    points = np.column_stack((axis, np.zeros(count), np.zeros(count)))
    start = time.perf_counter()
    scorer = make_overlap(points, np.full(count, 0.2), leaf=16)
    construction_seconds = time.perf_counter() - start
    ids = np.array([0, 101, 2003, 40007, 80003, 99999])
    centres = flat(points[ids])
    covariances = flat(np.tile(np.eye(3) * 0.09, (len(ids), 1, 1)))
    expected = ((2 * math.pi) ** -1.5 / 0.13 ** 1.5) / count
    start = time.perf_counter()
    for _ in range(100):
        result = scorer.evaluate(centres, covariances)
        assert result.overlap == pytest.approx(expected, rel=5e-13)
        np.testing.assert_allclose(result.component_overlaps, expected, rtol=5e-13)
        assert not np.any(result.gradient)
    evaluation_seconds = time.perf_counter() - start
    with capsys.disabled():
        print(f"100k Gaussian tree: construction={construction_seconds:.4f}s, "
              f"100 six-component evaluations={evaluation_seconds:.4f}s")
