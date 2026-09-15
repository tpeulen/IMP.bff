"""PRD-143 A4.4 (#16): the B-spline basis of a Bayesian P-spline, in C++.

`bayesian_pspline_basis` builds the design matrix the distance distribution is
expanded in: cubic (or any degree) B-splines on equally spaced knots extended past
the domain. Checked against scipy's `BSpline.design_matrix` on the same knots (the
reference ucfret's prototype used), for several sizes and degrees; the rows are a
partition of unity and non-negative; and a clamped knot sequence -- the
interpolation construction the P-spline deliberately avoids -- does not give it.

Written 2026-09-15 (ucfret prompt 452).
"""

import json

import numpy as np
import pytest

from bayesian_cxx import run_driver

scipy_interpolate = pytest.importorskip("scipy.interpolate")

CASES = [(128, 25, 3), (64, 10, 3), (200, 40, 2), (33, 8, 1)]

DRIVER = r"""
#include <IMP/bff/BayesianPSpline.h>
#include <cstdio>
#include <string>
int main() {
  const std::size_t cases[][3] = {%s};
  std::printf("[");
  for (std::size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    const auto B = IMP::bff::bayesian_pspline_basis(cases[c][0], cases[c][1], int(cases[c][2]));
    std::printf("%%s[", c ? "," : "");
    for (std::size_t i = 0; i < B.size(); ++i) std::printf("%%s%%.17g", i ? "," : "", B[i]);
    std::printf("]");
  }
  std::printf("]\n");
  return 0;
}
""" % ", ".join("{%d, %d, %d}" % c for c in CASES)


def reference(n_points, n_coef, degree, clamped=False):
    n_int = n_coef - degree
    h = 1.0 / n_int
    if clamped:
        knots = np.concatenate([np.zeros(degree), np.arange(n_int + 1) * h, np.ones(degree)])
    else:
        knots = np.arange(-degree, n_int + degree + 1) * h
    x = np.linspace(0.0, 1.0, n_points)
    lo, hi = (0.0, 1.0 - 1e-12) if clamped else (knots[degree], knots[-degree - 1] - 1e-12)
    return scipy_interpolate.BSpline.design_matrix(np.clip(x, lo, hi), knots, degree).toarray()


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_equals_scipy_design_matrix(result):
    for (n, k, d), flat in zip(CASES, result):
        B = np.asarray(flat).reshape(n, k)
        assert np.abs(B - reference(n, k, d)).max() < 1e-13, (n, k, d)


def test_partition_of_unity_and_non_negative(result):
    for (n, k, d), flat in zip(CASES, result):
        B = np.asarray(flat).reshape(n, k)
        assert B.min() >= 0.0
        assert np.abs(B.sum(1) - 1.0).max() < 1e-13


def test_clamped_knots_are_a_different_basis(result):
    n, k, d = CASES[0]
    B = np.asarray(result[0]).reshape(n, k)
    assert np.abs(B - reference(n, k, d, clamped=True)).max() > 1e-2
