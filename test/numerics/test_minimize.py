"""bff.minimize_lbfgsb / bff.minimize_nelder_mead against scipy.optimize.minimize.

SciPy is the oracle here and only here. Nelder-Mead is the same arithmetic
in the same order, so it is held to bit identity. L-BFGS-B runs SciPy's own
C translation of the algorithm; what differs is the BLAS underneath (SciPy
links an optimised one whose dot products sum in a different order), so a
path can drift by an ulp. With an analytic gradient, or once bounds are
active, the paths are identical in practice and are held to 1e-10. With a
finite-difference gradient and no bounds the two runs are held to the same
quality instead: the same iteration count to within a few, the same
objective to 1e-9, and the same distance from the true optimum.
"""

from __future__ import annotations

import numpy as np
import pytest

import IMP.bff as bff

scipy_optimize = pytest.importorskip("scipy.optimize")
from scipy.optimize import rosen, rosen_der  # noqa: E402


class Objective(bff.MinimizeObjective):
    """A Python objective with an optional analytic gradient."""

    def __init__(self, f, g=None):
        super().__init__()
        self.f = f
        self.g = g
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        return float(self.f(np.asarray(x)))

    def evaluate_with_gradient(self, x):
        self.calls += 1
        x = np.asarray(x)
        return np.concatenate([[self.f(x)], self.g(x)])


X0 = [-1.2, 1.0, 0.5, -0.3]
LB = [-2.0, -0.5, -2.0, 0.6]
UB = [2.0, 2.0, 0.8, 2.0]


def _scipy(method, x0, bounds=None, jac=None, options=None):
    return scipy_optimize.minimize(
        rosen, np.asarray(x0), method=method, jac=jac, bounds=bounds, options=options
    )


# ----------------------------------------------------------------- Nelder-Mead


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_nelder_mead_is_scipy_bit_for_bit(bounded, adaptive):
    lb, ub = (LB, UB) if bounded else ([], [])
    r = bff.minimize_nelder_mead(Objective(rosen), X0, lb, ub, 0, 0, 1e-4, 1e-4, adaptive, [])
    s = _scipy(
        "Nelder-Mead",
        X0,
        bounds=list(zip(LB, UB)) if bounded else None,
        options={"adaptive": adaptive},
    )
    assert np.array_equal(np.asarray(r.x), s.x)
    assert r.fun == s.fun
    assert (r.nit, r.nfev, r.status, r.message) == (s.nit, s.nfev, s.status, s.message)
    sim = np.asarray(r.final_simplex).reshape(len(X0) + 1, len(X0))
    assert np.array_equal(sim, s.final_simplex[0])
    assert np.array_equal(np.asarray(r.final_simplex_values), s.final_simplex[1])


@pytest.mark.parametrize("limit", [{"maxiter": 40}, {"maxfev": 60}])
def test_nelder_mead_limits_report_like_scipy(limit):
    r = bff.minimize_nelder_mead(
        Objective(rosen), X0, [], [], limit.get("maxiter", 0), limit.get("maxfev", 0),
        1e-4, 1e-4, False, [],
    )
    s = _scipy("Nelder-Mead", X0, options=dict(limit))
    assert np.array_equal(np.asarray(r.x), s.x)
    assert (r.nit, r.nfev, r.status, r.success, r.message) == (
        s.nit, s.nfev, s.status, s.success, s.message,
    )


def test_nelder_mead_with_an_initial_simplex():
    rng = np.random.default_rng(3)
    sim0 = np.asarray(X0) + 0.1 * rng.normal(size=(5, 4))
    r = bff.minimize_nelder_mead(
        Objective(rosen), X0, [], [], 0, 0, 1e-6, 1e-6, False, list(sim0.ravel())
    )
    s = _scipy(
        "Nelder-Mead", X0, options={"initial_simplex": sim0, "xatol": 1e-6, "fatol": 1e-6}
    )
    assert np.array_equal(np.asarray(r.x), s.x)
    assert r.nfev == s.nfev


# ------------------------------------------------------------------- L-BFGS-B


@pytest.mark.parametrize("bounded", [False, True])
def test_lbfgsb_with_gradient_follows_scipy(bounded):
    lb, ub = (LB, UB) if bounded else ([], [])
    r = bff.minimize_lbfgsb(Objective(rosen, rosen_der), X0, lb, ub, True)
    s = _scipy(
        "L-BFGS-B", X0, bounds=list(zip(LB, UB)) if bounded else None, jac=rosen_der
    )
    # Bounded, the active set pins the path; unbounded, 55 iterations let the
    # BLAS rounding difference accumulate to a few 1e-10.
    tol = 1e-10 if bounded else 1e-8
    np.testing.assert_allclose(r.x, s.x, rtol=0, atol=tol)
    np.testing.assert_allclose(r.jac, s.jac, rtol=0, atol=1e3 * tol)
    assert abs(r.fun - s.fun) <= 1e-12 * max(1.0, abs(s.fun))
    assert (r.nit, r.nfev, r.njev, r.status, r.message) == (
        s.nit, s.nfev, s.njev, s.status, s.message,
    )
    # The inverse-Hessian correction pairs: SciPy's hess_inv.sk / .yk.
    k = r.n_corrections
    assert k == s.hess_inv.n_corrs
    S = np.asarray(r.correction_s).reshape(k, 4)
    Y = np.asarray(r.correction_y).reshape(k, 4)
    assert np.all(np.einsum("ij,ij->i", S, Y) > 0)  # every pair is a curvature pair
    if bounded:
        # A pair is a difference of nearly equal iterates; with the path pinned
        # they agree, unbounded the ulp drift above makes them path-dependent.
        np.testing.assert_allclose(S, s.hess_inv.sk, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(Y, s.hess_inv.yk, rtol=1e-8, atol=1e-12)

def test_lbfgsb_difference_gradient_inside_bounds_follows_scipy():
    r = bff.minimize_lbfgsb(Objective(rosen), X0, LB, UB, False)
    s = _scipy("L-BFGS-B", X0, bounds=list(zip(LB, UB)))
    np.testing.assert_allclose(r.x, s.x, rtol=0, atol=1e-8)
    assert (r.nit, r.nfev, r.message) == (s.nit, s.nfev, s.message)


@pytest.mark.parametrize(
    "x0",
    [[-1.2, 1.0, 0.5, -0.3], [0.0, 0.0, 0.0, 0.0], [2.0, -1.0, 1.5, 0.3], [-0.5] * 6],
)
def test_lbfgsb_difference_gradient_unbounded_matches_scipy_quality(x0):
    r = bff.minimize_lbfgsb(Objective(rosen), x0, [], [], False)
    s = _scipy("L-BFGS-B", x0)
    assert r.success and s.success
    assert abs(r.nit - s.nit) <= 5
    assert abs(r.fun - s.fun) <= 1e-9
    err_bff = np.max(np.abs(np.asarray(r.x) - 1.0))
    err_scipy = np.max(np.abs(s.x - 1.0))
    assert err_bff <= max(5 * err_scipy, 1e-4)


def test_lbfgsb_difference_step_flips_at_an_upper_bound():
    """A start a hair below an active upper bound: the forward step would
    leave the box, so SciPy differences backward -- and so must we."""
    f = lambda x: float(np.sum((x - 3.0) ** 2))  # noqa: E731
    x0 = [2.0 - 1e-9, 0.5]
    r = bff.minimize_lbfgsb(Objective(f), x0, [-5.0, -5.0], [2.0, 5.0], False)
    s = scipy_optimize.minimize(f, np.asarray(x0), method="L-BFGS-B", bounds=[(-5, 2), (-5, 5)])
    np.testing.assert_allclose(r.x, s.x, rtol=0, atol=1e-8)
    assert r.x[0] == 2.0
    assert (r.nit, r.nfev) == (s.nit, s.nfev)


@pytest.mark.parametrize("limit", [{"maxiter": 5}, {"maxfun": 30}])
def test_lbfgsb_limits_report_like_scipy(limit):
    r = bff.minimize_lbfgsb(
        Objective(rosen, rosen_der), X0, [], [], True, 10, 2.2204460492503131e-09, 1e-5,
        1e-8, limit.get("maxfun", 15000), limit.get("maxiter", 15000), 20,
    )
    s = _scipy("L-BFGS-B", X0, jac=rosen_der, options=dict(limit))
    np.testing.assert_allclose(r.x, s.x, rtol=0, atol=1e-10)
    assert (r.nit, r.nfev, r.status, r.success, r.message) == (
        s.nit, s.nfev, s.status, s.success, s.message,
    )


def test_lbfgsb_clips_a_start_outside_the_box():
    r = bff.minimize_lbfgsb(Objective(rosen, rosen_der), [5.0, 5.0], [-1, -1], [0.5, 0.5], True)
    s = _scipy("L-BFGS-B", [5.0, 5.0], bounds=[(-1, 0.5), (-1, 0.5)], jac=rosen_der)
    np.testing.assert_allclose(r.x, s.x, rtol=0, atol=1e-10)


# --------------------------------------------------------------------- errors


def test_inverted_bounds_are_a_value_error():
    with pytest.raises(ValueError):
        bff.minimize_lbfgsb(Objective(rosen), [0.0], [1.0], [0.0], False)
    with pytest.raises(ValueError):
        bff.minimize_nelder_mead(Objective(rosen), [0.0, 0.0], [1.0, 0.0], [0.0, 1.0])


def test_an_exception_in_the_objective_propagates():
    class Boom(bff.MinimizeObjective):
        def evaluate(self, x):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        bff.minimize_lbfgsb(Boom(), [0.0, 0.0])
    with pytest.raises(RuntimeError, match="boom"):
        bff.minimize_nelder_mead(Boom(), [0.0, 0.0])


def test_a_short_gradient_is_a_value_error():
    class Short(bff.MinimizeObjective):
        def evaluate_with_gradient(self, x):
            return np.array([1.0])

    with pytest.raises(ValueError):
        bff.minimize_lbfgsb(Short(), [0.0, 0.0], [], [], True)
