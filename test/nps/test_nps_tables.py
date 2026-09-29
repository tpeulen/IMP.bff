"""S1 tests: generating NPS convolved-efficiency tables from bff states.

The Fast-NPS originals (modelMCSimulation.cpp + polynomialFit.cpp) fit raw
degree-11 powers of distances 1..150 A — a Vandermonde system with condition
number ~1e20 — and their non-iso MC branches normalize the dipole-vs-axis
cosine by the mean distance instead of the sampled one. The bff table
generator keeps the table layout (1 row iso/iso, 25 mixed, 625 pair grid,
12 power-basis coefficients each, the layout nps_network_fret_efficiency
consumes) but estimates on the label clouds themselves and fits stably.

Oracle: for the spherical 20-A accessible volume the original used, the
iso/iso curve must match brute-force MC of
E = <1/(1 + dist^6/Riso6)> within MC error, and the evaluated table must
reproduce the curve far more accurately than a naive raw-power fit of the
same data would.

Runtime budget: CTest runs this file under a hard 5 s timeout
(IMPAddTests.cmake), so the brute-force references are vectorized over
distances and cached at module scope, and the heavy table generators get
explicit (smaller) sample budgets with fixed seeds. Tolerances are unchanged.
"""
import math

import numpy as np
import pytest

import IMP
import IMP.bff as bff


def _cloud_on_sphere(radius, n_points, seed):
    """Uniform points in a sphere, as the flat (n, 4) cloud States takes."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-radius, radius, size=(n_points * 2, 3))
    inside = np.einsum("ij,ij->i", pts, pts) < radius * radius
    pts = pts[inside][:n_points]
    weights = np.full((len(pts), 1), 1.0 / len(pts))
    return np.hstack([pts, weights]).ravel()


def _brute_force_iso_curve(r_iso6, radius=20.0, n_pairs=120000, seed=7):
    """The iso/iso convolved curve by direct MC: E(d) = <1/(1+r^6/Riso6)>.

    Pairs are drawn independently from each cloud — the same joint
    distribution the Fast-NPS shuffle sampler and the bff estimator use.
    One pooled pair draw serves all 150 distances (each distance's
    estimator stays unbiased; draws are only correlated across distances).
    Distances are evaluated in blocks to bound the broadcast temporary.
    """
    rng = np.random.default_rng(seed)
    n_pts = 4000
    pts = rng.uniform(-radius, radius, size=(n_pts * 2, 3))
    inside = np.einsum("ij,ij->i", pts, pts) < radius * radius
    pts = pts[inside][:n_pts]
    i1 = rng.integers(0, n_pts, size=n_pairs)
    i2 = rng.integers(0, n_pts, size=n_pairs)
    v = pts[i1] - pts[i2]
    curve = np.empty(150)
    ds = np.arange(1, 151, dtype=float)
    ex = np.array([1.0, 0.0, 0.0])
    for start in range(0, 150, 30):
        block = ds[start:start + 30]
        ov = v[None, :, :] - block[:, None, None] * ex
        r6 = np.einsum("ijk,ijk->ij", ov, ov) ** 3
        curve[start:start + 30] = np.mean(1.0 / (1.0 + r6 / r_iso6), axis=1)
    return curve


@pytest.fixture(scope="module")
def iso_curve():
    """The brute-force iso/iso reference, computed once for the module."""
    return _brute_force_iso_curve(55.0 ** 6)


@pytest.fixture(scope="module")
def pair_grid():
    """One 625-row table (dep1 = dep2 = 0.9), shared by the layout and
    dye-exchange tests — table generation is the expensive step here, and
    the exchange symmetry holds on any equal-dep table."""
    cloud = _cloud_on_sphere(20.0, 2000, seed=13)
    table = bff.nps_convolved_efficiency_table(
        cloud, cloud, 0.9, 0.9, 55.0 ** 6, 3000, 11)
    return cloud, table


# ---------------------------------------------------------------------------
# iso/iso: one row
# ---------------------------------------------------------------------------

def test_table_iso_iso_produces_one_row_of_twelve():
    cloud = _cloud_on_sphere(20.0, 4000, seed=1)
    table = bff.nps_convolved_efficiency_table(
        cloud, cloud, 1.0, 1.0, 55.0 ** 6)
    assert len(table) == 1
    assert len(table[0]) == 12
    assert all(math.isfinite(c) for c in table[0])


def test_table_iso_iso_matches_brute_force_curve_within_mc_error(iso_curve):
    cloud = _cloud_on_sphere(20.0, 6000, seed=3)
    r_iso6 = 55.0 ** 6
    table = bff.nps_convolved_efficiency_table(cloud, cloud, 1.0, 1.0, r_iso6)
    truth = iso_curve
    # Evaluate the fitted table at the same 150 distances.
    xs = np.arange(1, 151, dtype=float)
    fitted = np.polyval(table[0][::-1], xs)
    # MC noise of the brute-force reference is ~1e-3; the degree-11 fit
    # tracks the curve everywhere to under 1e-2 (its knee representation
    # limit in the power basis — the raw-power fit of the original misses
    # by orders of magnitude more).
    assert np.max(np.abs(fitted - truth)) < 1e-2


def test_table_fit_is_more_stable_than_a_raw_power_fit(iso_curve):
    """The reason the original tables were BS: cond(Vandermonde) ~1e20.

    Same cloud, same model: fit in a scaled variable, express in the same
    power basis, and the reconstruction error at the sampled points must
    stay far below what a direct raw-power fit of identical data gives.
    """
    cloud = _cloud_on_sphere(20.0, 6000, seed=3)
    r_iso6 = 55.0 ** 6
    table = bff.nps_convolved_efficiency_table(cloud, cloud, 1.0, 1.0, r_iso6)
    xs = np.arange(1, 151, dtype=float)
    fitted = np.polyval(table[0][::-1], xs)

    truth = iso_curve
    # A degree-11 fit of the same truth in UNSCALED raw powers — exactly
    # the original's system (cond ~1e20): numpy's lstsq on V(d).
    raw = np.linalg.lstsq(
        np.vander(xs, 12, increasing=True), truth, rcond=None)[0]
    raw_err = np.max(np.abs(np.polyval(raw[::-1], xs) - truth))
    our_err = np.max(np.abs(fitted - truth))
    assert our_err < raw_err
    assert our_err < 1e-2


def test_table_efficiency_is_bounded_and_monotone_decreasing():
    cloud = _cloud_on_sphere(20.0, 5000, seed=5)
    r_iso6 = 55.0 ** 6
    table = bff.nps_convolved_efficiency_table(cloud, cloud, 1.0, 1.0, r_iso6)
    xs = np.arange(1, 151, dtype=float)
    e = np.polyval(table[0][::-1], xs)
    assert np.all((e >= -5e-3) & (e <= 1.0 + 5e-3))
    # Monotone decreasing to within the fit's representation noise (the
    # tail sits at E ~ 3e-3, so its fit wiggle is a few 1e-4).
    assert np.all(np.diff(e) < 1e-3)


# ---------------------------------------------------------------------------
# mixed: 25 rows indexed by the single-dye grid
# ---------------------------------------------------------------------------

def _brute_force_grid_row(cloud, dep, regime, phi1, polar1, phi2=0,
                          polar2=0, r_iso6=55.0 ** 6, seed=5,
                          n_pairs=80000):
    """Independent MC truth for one orientation row of a convolved table.

    regime="mixed" uses the one-grid-dye factor (2 - dep)/3 + dep c1^2;
    regime="grid" uses the two-dye form with both dyes at the same dep.
    Slot dipoles are the mid-slot angles of the committed grid layout.
    Distances are evaluated in blocks to bound the broadcast temporary.
    """
    pts = cloud.reshape(-1, 4)[:, :3]
    m1 = -1.0 + 2.0 * (polar1 + 0.5) / 5.0
    p1 = (phi1 + 0.5) * 3.142 / 5.0
    d1 = np.array([math.sqrt(1 - m1 * m1) * math.cos(p1),
                   math.sqrt(1 - m1 * m1) * math.sin(p1), m1])
    d2 = np.array([0.0, 0.0, 1.0])  # placeholder; read only in "grid" regime
    c_th_t = 0.0
    if regime == "grid":
        m2 = -1.0 + 2.0 * (polar2 + 0.5) / 5.0
        p2 = (phi2 + 0.5) * 3.142 / 5.0
        d2 = np.array([math.sqrt(1 - m2 * m2) * math.cos(p2),
                       math.sqrt(1 - m2 * m2) * math.sin(p2), m2])
        c_th_t = float(d1 @ d2)
    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, len(pts), n_pairs)
    i2 = rng.integers(0, len(pts), n_pairs)
    v = pts[i1] - pts[i2]
    curve = np.empty(150)
    ds = np.arange(1, 151, dtype=float)
    ex = np.array([1.0, 0.0, 0.0])
    for start in range(0, 150, 25):
        block = ds[start:start + 25]
        ov = v[None, :, :] - block[:, None, None] * ex
        r2 = np.einsum("ijk,ijk->ij", ov, ov)
        r = np.sqrt(r2)
        c1 = np.einsum("ijk,k->ij", ov, d1) / r
        if regime == "mixed":
            k2 = (2.0 - dep) / 3.0 + c1 * c1 * dep
        else:
            c2 = np.einsum("ijk,k->ij", ov, d2) / r
            kx2 = (c_th_t - 3.0 * c1 * c2) ** 2
            k2 = (kx2 * dep * dep + (2.0 - 2.0 * dep) / 3.0 +
                  c1 * c1 * dep * (1.0 - dep) +
                  c2 * c2 * dep * (1.0 - dep))
        avg_r6 = k2 * 1.5 * r_iso6
        curve[start:start + 25] = np.mean(
            avg_r6 / (avg_r6 + r2 * r2 * r2), axis=1)
    return curve


def test_table_mixed_regime_produces_twenty_five_rows():
    cloud = _cloud_on_sphere(20.0, 3000, seed=11)
    table = bff.nps_convolved_efficiency_table(
        cloud, cloud, 0.9, 1.0, 55.0 ** 6)
    assert len(table) == 25
    assert all(len(row) == 12 for row in table)
    # Row layout: m = 1, phi = 0 is slot (0, 0) -> row 0.
    assert bff.nps_single_orientation_row_index(1.0, 0.0) == 0
    # Every slot carries its own dipole: rows must differ across slots.
    assert not np.allclose(table[0], table[7])
    # Row 7 = (phi slot 1, polar slot 2): an independent brute-force
    # average over the same cloud with the row's slot dipole.
    truth = _brute_force_grid_row(
        cloud, dep=0.9, regime="mixed", phi1=1, polar1=2,
        r_iso6=55.0 ** 6, seed=5)
    xs = np.arange(1, 151, dtype=float)
    fitted = np.polyval(np.asarray(table[7])[::-1], xs)
    assert np.max(np.abs(fitted - truth)) < 5e-3


# ---------------------------------------------------------------------------
# non-iso/non-iso: the 625-row pair grid
# ---------------------------------------------------------------------------

def test_table_pair_grid_produces_six_hundred_twenty_five_rows(pair_grid):
    cloud, table = pair_grid
    assert len(table) == 625
    assert all(len(row) == 12 for row in table)
    # Layout agreement: index (m1=1, phi1=0, m2=1, phi2=0) is row 0 and
    # the evaluator's index maps into the produced table.
    assert bff.nps_orientation_row_index(1.0, 0.0, 1.0, 0.0) == 0
    # Row 38 decodes as (phi1=0, polar1=1, phi2=2, polar2=3): brute-force
    # the same row with the two-dye orientation factor.
    truth = _brute_force_grid_row(
        cloud, dep=0.9, regime="grid", phi1=0, polar1=1,
        phi2=2, polar2=3, r_iso6=55.0 ** 6, seed=5)
    xs = np.arange(1, 151, dtype=float)
    fitted = np.polyval(np.asarray(table[38])[::-1], xs)
    assert np.max(np.abs(fitted - truth)) < 5e-3


def test_table_pair_grid_is_symmetric_under_dye_exchange(pair_grid):
    """E(1<->2) must give the same curve for the transposed orientation slot."""
    _, table = pair_grid
    xs = np.arange(1, 151, dtype=float)
    # slot (m=1, phi=0) for dye1 and (m=0, phi=pi/2) for dye2 vs swapped:
    i1 = bff.nps_orientation_row_index(1.0, 0.0, 0.0, math.pi / 2)
    i2 = bff.nps_orientation_row_index(0.0, math.pi / 2, 1.0, 0.0)
    e1 = np.polyval(table[i1][::-1], xs)
    e2 = np.polyval(table[i2][::-1], xs)
    assert np.max(np.abs(e1 - e2)) < 5e-3


# ---------------------------------------------------------------------------
# integration with the committed evaluator
# ---------------------------------------------------------------------------

def test_generated_table_feeds_the_network_evaluator(iso_curve):
    cloud = _cloud_on_sphere(20.0, 6000, seed=19)
    r_iso6 = 55.0 ** 6
    table = bff.nps_convolved_efficiency_table(
        cloud, cloud, 1.0, 1.0, r_iso6, 40000, 19)

    config = [[0.0, 0.0, 0.0, 0.0, 0.0],
              [70.0, 0.0, 0.0, 0.0, 0.0]]
    iso_dyes = []
    for _ in range(2):
        dye = bff.NPSNetworkDye()
        dye.dep = 1.0
        dye.iso = True
        dye.dist_conv = True
        iso_dyes.append(dye)
    e = bff.nps_network_fret_efficiency(
        config, iso_dyes, 0, 1, r_iso6, table)
    assert 0.0 < e < 1.0
    # At 70 A site separation the 20-A AVs spread pairs wide; the table
    # value must match a direct random-pair average at the same geometry.
    truth = iso_curve[69]
    assert e == pytest.approx(truth, abs=0.02)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_table_rejects_bad_clouds_and_parameters():
    cloud = _cloud_on_sphere(20.0, 500, seed=23)
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency_table(
            cloud, cloud, 1.5, 1.0, 55.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency_table(
            cloud, cloud, 1.0, float("nan"), 55.0 ** 6)
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency_table(cloud, cloud, 1.0, 1.0, 0.0)
    with pytest.raises(IMP.ValueException):
        bff.nps_convolved_efficiency_table(
            [1.0, 2.0], cloud, 1.0, 1.0, 55.0 ** 6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
