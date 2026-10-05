"""NNLS and BVLS in C++, against scipy as the oracle.

`bff.nnls` and `bff.bvls` are what lets ChiSurf drop `scipy.optimize.nnls` and
`scipy.optimize.lsq_linear`. Agreement is judged on the **objective**, not on
`x`: a rank-deficient `A` has a whole face of minimisers and two correct
solvers may stop on different points of it. Where the minimiser is unique
(full column rank) `x` itself is compared.
"""

import unittest

import numpy as np

from IMP import bff

try:
    import scipy.optimize as so
except ImportError:  # pragma: no cover - scipy is the oracle, test-only
    so = None


def _problems(n_problems=300, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(n_problems):
        m = int(rng.integers(1, 40))
        n = int(rng.integers(2, 25))
        a = rng.normal(size=(m, n))
        rank_deficient = k % 3 == 0
        ill = k % 5 == 0
        if rank_deficient:
            a[:, 0] = a[:, 1]
        if ill:
            a = a @ np.diag(np.logspace(0, 8, n))
        b = rng.normal(size=m) * 10.0
        yield a, b, rank_deficient or ill or m < n, rng


class TestNNLS(unittest.TestCase):
    def test_scipy_docstring_example(self):
        a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        r = bff.nnls(a, np.array([2.0, 1.0, 1.0]))
        np.testing.assert_allclose(r.x, [1.5, 1.0])
        self.assertAlmostEqual(r.rnorm, np.sqrt(0.5))
        self.assertTrue(r.get_success())
        r = bff.nnls(a, np.array([-1.0, -1.0, -1.0]))
        np.testing.assert_array_equal(r.x, [0.0, 0.0])
        self.assertAlmostEqual(r.rnorm, np.sqrt(3.0))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bff.nnls(np.ones((3, 2)), np.ones(4))

    @unittest.skipIf(so is None, "scipy is the oracle")
    def test_matches_scipy(self):
        for a, b, degenerate, _ in _problems():
            n = a.shape[1]
            xs, rs = so.nnls(a, b, maxiter=50 * n)
            r = bff.nnls(a, b, maxiter=50 * n)
            self.assertTrue(r.get_success())
            self.assertTrue(np.all(r.x >= 0.0))
            # Objective parity; an exact fit (rnorm 0) differs by round-off.
            self.assertLessEqual(r.rnorm - rs, 1e-9 * max(1.0, np.linalg.norm(b)))
            if not degenerate:
                np.testing.assert_allclose(r.x, xs, rtol=1e-9, atol=1e-9 * max(1, np.abs(xs).max()))

    def test_kkt_conditions(self):
        """Dual feasibility: no zero variable could lower the residual."""
        for a, b, _, _ in _problems(100, seed=1):
            r = bff.nnls(a, b, maxiter=50 * a.shape[1])
            w = a.T @ (b - a @ r.x)
            scale = np.abs(a).sum(axis=0).max() * max(1.0, np.linalg.norm(b))
            self.assertLessEqual(w[r.x == 0].max(initial=0.0), 1e-9 * scale)
            np.testing.assert_allclose(w[r.x > 0], 0.0, atol=1e-7 * scale)

    def test_iteration_limit_reports_status(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(30, 20))
        b = rng.normal(size=30)
        r = bff.nnls(a, b, maxiter=1)
        self.assertEqual(r.status, 1)
        self.assertFalse(r.get_success())


class TestBVLS(unittest.TestCase):
    @unittest.skipIf(so is None, "scipy is the oracle")
    def test_matches_scipy(self):
        for a, b, degenerate, rng in _problems(seed=2):
            n = a.shape[1]
            lb = rng.uniform(-2.0, 0.0, n)
            ub = lb + rng.uniform(0.1, 3.0, n)
            ub[::4] = np.inf
            ref = so.lsq_linear(a, b, bounds=(lb, ub), method="bvls")
            r = bff.bvls(a, b, lb, ub)
            self.assertTrue(r.get_success())
            self.assertTrue(np.all(r.x >= lb) and np.all(r.x <= ub))
            cost = 0.5 * r.rnorm**2
            self.assertLessEqual(cost - ref.cost, 1e-9 * max(1.0, ref.cost))
            if not degenerate:
                np.testing.assert_allclose(r.x, ref.x, rtol=1e-7, atol=1e-7)

    def test_zero_to_inf_is_nnls(self):
        for a, b, _, _ in _problems(60, seed=4):
            n = a.shape[1]
            r1 = bff.nnls(a, b, maxiter=50 * n)
            r2 = bff.bvls(a, b, np.zeros(n), np.full(n, np.inf))
            self.assertLessEqual(abs(r1.rnorm - r2.rnorm), 1e-9 * max(1.0, np.linalg.norm(b)))

    def test_unbounded_is_lstsq(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(20, 6))
        b = rng.normal(size=20)
        r = bff.bvls(a, b, np.full(6, -np.inf), np.full(6, np.inf))
        np.testing.assert_allclose(r.x, np.linalg.lstsq(a, b, rcond=None)[0], rtol=1e-10)
        np.testing.assert_array_equal(r.active_mask, 0)

    def test_active_mask(self):
        a = np.eye(3)
        b = np.array([-1.0, 0.5, 5.0])
        r = bff.bvls(a, b, np.zeros(3), np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(r.x, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(r.active_mask, [-1, 0, 1])

    def test_bad_bounds_raise(self):
        with self.assertRaises(ValueError):
            bff.bvls(np.eye(2), np.ones(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
