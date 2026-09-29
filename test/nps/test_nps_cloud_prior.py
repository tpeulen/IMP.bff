"""S2 RED tests: the cloud-based position prior.

Fast-NPS positions a dye site's prior on hard box grids (isInBox / box
adjacency from a sampled accessible volume). The bff replacement keeps the
accessible volume's own point cloud as the prior: a weighted Gaussian
kernel density estimate whose bandwidth is Scott's rule on the cloud, a
log-density the samplers can use directly, a config seed for the network
evaluator's [x, y, z, m, phi] rows, and a thin IMP restraint so
IMP.core.MonteCarlo + Movers can sample a live label site under it — the
same conventions as the committed NPSIsotropicFRETEfficiencyRestraint:
+inf (never a throw) for unscorable live state, Monte-Carlo-only, and
ValueException at construction for misconfiguration.
"""
import math

import numpy as np
import pytest

import IMP
import IMP.core
import IMP.algebra
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
# the restraint: IMP.core.MonteCarlo samples a live label site under the prior
# ---------------------------------------------------------------------------

def _restraint(model, particle, cloud, scale=1.0):
    return bff.NPSCloudPositionPriorRestraint(
        model, particle, cloud, scale)


def test_restraint_scores_minus_the_cloud_log_prior():
    model = IMP.Model()
    p = IMP.Particle(model)
    xyz = IMP.core.XYZ.setup_particle(
        p, IMP.algebra.Vector3D(1.0, 2.0, 3.0))
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=200)
    r = _restraint(model, p, cloud)
    score = r.unprotected_evaluate(None)
    assert score == pytest.approx(
        -bff.nps_cloud_log_prior(cloud, 1.0, 2.0, 3.0), rel=1e-12)


def test_restraint_rejects_derivatives_and_bad_construction():
    model = IMP.Model()
    p = IMP.Particle(model)
    IMP.core.XYZ.setup_particle(p, IMP.algebra.Vector3D(0.0, 0.0, 0.0))
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=100)
    r = _restraint(model, p, cloud)
    with pytest.raises(IMP.UsageException):
        r.unprotected_evaluate(IMP.DerivativeAccumulator())
    # a particle without XYZ is refused at construction
    q = IMP.Particle(model)
    with pytest.raises(IMP.ValueException):
        bff.NPSCloudPositionPriorRestraint(model, q, cloud, 1.0)
    # a non-positive bandwidth scale is refused
    with pytest.raises(IMP.ValueException):
        bff.NPSCloudPositionPriorRestraint(model, p, cloud, 0.0)
    with pytest.raises(IMP.ValueException):
        bff.NPSCloudPositionPriorRestraint(model, p, cloud, float("nan"))
    # a ragged cloud is refused
    with pytest.raises(IMP.ValueException):
        bff.NPSCloudPositionPriorRestraint(model, p, [1.0, 2.0], 1.0)


def test_restraint_maps_unscorable_state_to_infinity():
    model = IMP.Model()
    p = IMP.Particle(model)
    xyz = IMP.core.XYZ.setup_particle(
        p, IMP.algebra.Vector3D(0.0, 0.0, 0.0))
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=100)
    r = _restraint(model, p, cloud)
    xyz.set_coordinates(IMP.algebra.Vector3D(float("nan"), 0.0, 0.0))
    assert r.unprotected_evaluate(None) == float("inf")


def test_monte_carlo_with_a_ball_mover_pulls_the_site_into_the_cloud():
    """The Mover route: one IMP.core.MonteCarlo owning the transaction, a
    BallMover proposing positions, the cloud prior as the sole restraint.
    Starting outside the cloud, the chain must move the site to the
    neighbourhood of the weighted mean — what a hard box could only
    approximate by construction."""
    model = IMP.Model()
    p = IMP.Particle(model)
    xyz = IMP.core.XYZ.setup_particle(
        p, IMP.algebra.Vector3D(30.0, 30.0, 30.0))
    cloud = _cloud_gaussian([0.0, 0.0, 0.0], n_points=600)
    r = _restraint(model, p, cloud)
    sf = IMP.core.RestraintsScoringFunction([r])
    xyz.set_coordinates_are_optimized(True)
    mc = IMP.core.MonteCarlo(model)
    mc.set_scoring_function(sf)
    mc.set_kt(0.5)
    mover = IMP.core.BallMover(model, p, 4.0)
    mover.set_was_used(True)
    mc.add_mover(mover)
    mc.optimize(2000)
    mean = bff.points_weighted_mean(cloud)
    h = bff.nps_cloud_bandwidth(cloud)
    final = np.array([xyz.get_coordinate(i) for i in range(3)])
    dist = float(np.linalg.norm(final - np.asarray(mean)))
    # a converged chain sits within a few bandwidths of the cloud's mean
    assert dist < 5.0 * h


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
