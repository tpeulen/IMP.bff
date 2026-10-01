"""S2 RED tests: the cloud-based position prior.

Fast-NPS positions a dye site's prior on hard box grids (isInBox / box
adjacency from a sampled accessible volume). The bff replacement keeps the
accessible volume's own point cloud as the prior: a weighted Gaussian
kernel density estimate whose bandwidth is Scott's rule on the cloud, a
log-density the samplers can use directly, a config seed for the network
evaluator's [x, y, z, m, phi] rows, and the prior as a score
(nps_cloud_prior_score) with the conventions of nps_isotropic_direct_score:
+inf (never a throw) for an unscorable position, ValueException for
misconfiguration. Plain values in, so the core builds without IMP.
"""
import math

import numpy as np
import pytest

import IMP
import IMP.bff as bff


def _cloud_gaussian(mean, n_points=1500, seed=3):
    rng = np.random.default_rng(seed)
    pts = rng.normal(mean, 3.0, size=(n_points, 3))
    return np.hstack([pts, np.full((n_points, 1), 1.0 / n_points)]).ravel()


def _cloud_uniform(radius=15.0, n_points=1200, seed=5):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-radius, radius, size=(n_points * 2, 3))
    pts = pts[np.einsum("ij,ij->i", pts, pts) < radius * radius][:n_points]
    return np.hstack([pts, np.full((len(pts), 1), 1.0 / len(pts))]).ravel()


# ---------------------------------------------------------------------------
# bandwidth: Scott's rule on the weighted cloud
# ---------------------------------------------------------------------------

def test_bandwidth_is_scotts_rule_on_the_weighted_spread():
    # A known cloud: two points on the x axis, equal weights. The weighted
    # RMS spread is exactly 5, Scott's factor for n=2 is 2^(-1/5).
    cloud = np.array([0.0, 0, 0, 0.5,
                      10.0, 0, 0, 0.5])
    h = bff.nps_cloud_bandwidth(cloud)
    assert h == pytest.approx(5.0 * 2.0 ** (-0.2), rel=1e-12)


def test_bandwidth_is_strictly_positive_and_finite_for_real_clouds():
    h = bff.nps_cloud_bandwidth(_cloud_gaussian([0.0, 0.0, 0.0]))
    assert math.isfinite(h) and h > 0.0


def test_bandwidth_rejects_degenerate_clouds():
    empty = []
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_bandwidth(empty)
    # all weights zero -> no spread to estimate
    zero_w = np.zeros((4, 4))
    zero_w[:, :3] = 1.0
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_bandwidth(zero_w.ravel())
    # ragged (not a multiple of 4)
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_bandwidth([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# log prior: weighted Gaussian KDE log-density from the cloud weights
# ---------------------------------------------------------------------------

def test_log_prior_peaks_at_the_weighted_mean():
    cloud = _cloud_gaussian([8.0, -4.0, 2.0])
    mean = bff.points_weighted_mean(cloud)
    h = bff.nps_cloud_bandwidth(cloud)
    lp_mean = bff.nps_cloud_log_prior(cloud, *mean)
    lp_off = bff.nps_cloud_log_prior(
        cloud, mean[0] + 2.0 * h, mean[1], mean[2])
    assert lp_mean > lp_off


def test_log_prior_differences_are_shift_invariant():
    """A translated cloud shifts the log-density exactly by translation:
    logp(x | cloud + t) - logp(y | cloud + t) == logp(x - t | cloud) - ...
    (exact, because every kernel moves with its point)."""
    cloud = _cloud_uniform()
    a, b = (1.0, 2.0, 3.0), (-4.0, 5.0, 0.5)
    t = (7.0, -3.0, 11.0)
    shifted = np.asarray(cloud).reshape(-1, 4).copy()
    shifted[:, 0] += t[0]
    shifted[:, 1] += t[1]
    shifted[:, 2] += t[2]
    d_orig = (bff.nps_cloud_log_prior(cloud, *a) -
              bff.nps_cloud_log_prior(cloud, *b))
    d_shift = (bff.nps_cloud_log_prior(shifted.ravel(),
                                       a[0] + t[0], a[1] + t[1], a[2] + t[2]) -
               bff.nps_cloud_log_prior(shifted.ravel(),
                                       b[0] + t[0], b[1] + t[1], b[2] + t[2]))
    assert d_shift == pytest.approx(d_orig, abs=1e-9)


def test_log_prior_is_finite_everywhere_and_largest_inside_the_cloud():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0])
    assert math.isfinite(bff.nps_cloud_log_prior(cloud, 0.0, 0.0, 0.0))
    # far tail: finite but much lower
    lp_far = bff.nps_cloud_log_prior(cloud, 100.0, 100.0, 100.0)
    lp_in = bff.nps_cloud_log_prior(cloud, 0.0, 0.0, 0.0)
    assert math.isfinite(lp_far)
    assert lp_in - lp_far > 10.0


def test_log_prior_rejects_bad_input():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=50)
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_log_prior(cloud, float("nan"), 0.0, 0.0)
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_log_prior(cloud, float("inf"), 0.0, 0.0)
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_log_prior([1.0, 2.0], 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# config seeding: a cloud becomes an evaluator config row
# ---------------------------------------------------------------------------

def test_config_from_cloud_is_the_weighted_mean_with_isotropic_angles():
    cloud = _cloud_gaussian([3.0, 4.0, 5.0])
    row = bff.nps_config_from_cloud(cloud)
    assert len(row) == 5
    mean = bff.points_weighted_mean(cloud)
    assert row[0] == pytest.approx(mean[0])
    assert row[1] == pytest.approx(mean[1])
    assert row[2] == pytest.approx(mean[2])
    # isotropic angles: m = 0 (dep 1), phi = 0
    assert row[3] == 0.0 and row[4] == 0.0


def test_seeded_config_feeds_the_network_evaluator():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=400)
    row1 = bff.nps_config_from_cloud(cloud)
    row2 = list(row1)
    row2[0] += 60.0
    config = [row1, row2]
    dyes = []
    for _ in range(2):
        dye = bff.NPSNetworkDye()
        dye.dep = 1.0
        dye.iso = True
        dye.dist_conv = False  # direct branch: config positions are the model
        dyes.append(dye)
    e = bff.nps_network_fret_efficiency(
        config, dyes, 0, 1, 55.0 ** 6)
    assert 0.0 < e < 1.0


# ---------------------------------------------------------------------------
# the score: -scale * log prior, what a sampler minimises
# ---------------------------------------------------------------------------

def test_score_is_minus_the_cloud_log_prior():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=200)
    assert bff.nps_cloud_prior_score(cloud, 1.0, 2.0, 3.0) == pytest.approx(
        -bff.nps_cloud_log_prior(cloud, 1.0, 2.0, 3.0), rel=1e-12)
    assert bff.nps_cloud_prior_score(cloud, 1.0, 2.0, 3.0, 2.5) == pytest.approx(
        -2.5 * bff.nps_cloud_log_prior(cloud, 1.0, 2.0, 3.0), rel=1e-12)


def test_score_rejects_bad_configuration():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=100)
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(IMP.ValueException):
            bff.nps_cloud_prior_score(cloud, 0.0, 0.0, 0.0, scale)
    with pytest.raises(IMP.ValueException):
        bff.nps_cloud_prior_score([1.0, 2.0], 0.0, 0.0, 0.0)


def test_score_maps_an_unscorable_position_to_infinity():
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=100)
    assert bff.nps_cloud_prior_score(cloud, float("nan"), 0.0, 0.0) == float("inf")


def test_metropolis_on_the_score_pulls_the_site_into_the_cloud():
    """A seeded Metropolis walk on the score alone, started outside the
    cloud, ends near the weighted mean -- what a hard box could only
    approximate by construction."""
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=600)
    rng = np.random.default_rng(11)
    x = np.array([30.0, 30.0, 30.0])
    s = bff.nps_cloud_prior_score(cloud, *x)
    kt = 0.5
    for _ in range(2000):
        y = x + rng.normal(0.0, 2.0, 3)
        t = bff.nps_cloud_prior_score(cloud, *y)
        if t <= s or rng.random() < math.exp(-(t - s) / kt):
            x, s = y, t
    mean = np.asarray(bff.points_weighted_mean(cloud))
    assert float(np.linalg.norm(x - mean)) < 5.0 * bff.nps_cloud_bandwidth(cloud)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
