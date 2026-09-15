"""Maximum entropy on a linear problem: `maxent_solve` and `maxent_invert`.

The engine under MaxEntSpectrum, public for inversions that are not a decay
(FCS, DEER and 2D FLC in ChiSurf, ucfret). None of the checks reads the
engine's arithmetic: the returned point is tested against the objective the
header documents, with SciPy's L-BFGS-B as the independent minimiser (the
objective is convex, so KKT at the point is sufficient and there is one answer
to agree on).

The engine came from tttrlib, which no longer carries maximum entropy. Its
answers on fixed problems, recorded before the move
(`data/maxent_tttrlib_reference.json`), are the floor: this engine's objective
is never above them.
"""

import json
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

scipy_optimize = pytest.importorskip("scipy.optimize")

REFERENCE = json.loads((pathlib.Path(__file__).parent / "data" / "maxent_tttrlib_reference.json").read_text())


def _bounded_min(f, jac, x0, lb):
    return scipy_optimize.minimize(f, x0, jac=jac, bounds=[(lb, None)] * len(x0), method="L-BFGS-B",
                                   options=dict(ftol=1e-16, gtol=1e-14, maxiter=200000, maxfun=2000000))


def _lifetime_problem(seed=3, n_rows=200, n=25, sigma=0.01):
    """Two lifetimes planted on a log-spaced grid, in the `1/2 p^T H p - g^T p + c` form."""
    rng = np.random.default_rng(seed)
    tau = np.geomspace(0.1, 10, n)
    t = np.linspace(0, 20, n_rows)
    A = np.exp(-t[:, None] / tau[None, :])
    p_true = np.zeros(n)
    p_true[8] = 1.0
    p_true[17] = 0.5
    b = A @ p_true + rng.standard_normal(n_rows) * sigma
    w = np.full(n_rows, 1.0 / sigma ** 2)
    H = 2 * (A * w[:, None]).T @ A
    g0 = 2 * A.T @ (w * b)
    c = float(w @ (b * b))
    return A, b, H, g0, c


def _Q(H, g0, c, m, nu):
    def Q(p):
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.sum((1.0 - np.log(p / m)) * p) - m.sum()
        return 0.5 * p @ H @ p - g0 @ p + c - 0.5 * nu * S

    def dQ(p):
        with np.errstate(divide="ignore", invalid="ignore"):
            return H @ p - g0 + 0.5 * nu * np.log(p / m)
    return Q, dQ


def _solve(H, g0, m, c, nu, max_iter=5000, tol=1e-8):
    r = bff.maxent_solve(H.ravel().tolist(), g0.tolist(), m.tolist(), c, nu, max_iter, tol, 1e-12)
    return np.asarray(r.get_amplitudes()), r


@pytest.mark.parametrize("nu", [0.1, 1.0, 10.0, 100.0])
def test_the_solution_is_the_bounded_minimiser_of_the_documented_objective(nu):
    _, _, H, g0, c = _lifetime_problem()
    m = np.full(25, 0.05)
    p, _ = _solve(H, g0, m, c, nu)
    Q, dQ = _Q(H, g0, c, m, nu)
    g = dQ(p)
    scale = np.abs(H @ p).max() + np.abs(g0).max()
    free = p > 1e-9
    assert np.abs(g[free]).max() / scale < 1e-9
    if (~free).any():
        # Relative to the same scale as the free check: a coordinate whose true
        # optimum is the floor has a gradient that is zero to rounding there
        # (3.8e-17 of the scale at nu = 1 on macOS, a hair below zero on Linux).
        assert g[~free].min() > -1e-9 * scale, "a clamped amplitude wants to move up"
    assert _bounded_min(Q, dQ, p.copy(), 1e-12).fun >= Q(p) - 1e-9 * abs(Q(p))
    assert Q(p) <= _bounded_min(Q, dQ, m.copy(), 1e-12).fun + 1e-9 * abs(Q(p))


def test_the_two_lifetimes_come_back():
    _, _, H, g0, c = _lifetime_problem()
    p, _ = _solve(H, g0, np.full(25, 0.05), c, 0.1)
    assert set(np.argsort(p)[-2:].tolist()) == {8, 17}
    assert abs(p[7:10].sum() - 1.0) < 0.1
    assert abs(p[16:19].sum() - 0.5) < 0.1


@pytest.mark.parametrize("nu", [0.05, 0.3, 1.0])
def test_inversion_minimises_the_weighted_objective_against_the_prior(nu):
    """`sum w (Ax - b)^2 - nu^2 S(x; prior)`: the translation onto the engine's form included."""
    A, b, *_ = _lifetime_problem(seed=9, sigma=0.02)
    n_rows, n = A.shape
    rng = np.random.default_rng(4)
    w = 0.5 + rng.random(n_rows)
    m = 0.5 + rng.random(n)
    x = np.asarray(bff.maxent_invert(A.ravel().tolist(), b.tolist(), w.tolist(), m.tolist(), nu, n_rows, n,
                                     20000, 1e-9).get_amplitudes())

    def Q(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.sum((1.0 - np.log(x / m)) * x) - m.sum()
        r = A @ x - b
        return r @ (w * r) - nu * nu * S

    def dQ(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * A.T @ (w * (A @ x - b)) + nu * nu * np.log(x / m)

    assert np.all(x >= 1e-12)
    free = x > 1e-9
    assert np.abs(dQ(x)[free]).max() / (np.abs(2 * A.T @ (w * (A @ x))).max() + 1.0) < 1e-8
    assert Q(x) == pytest.approx(_bounded_min(Q, dQ, m.copy(), 1e-12).fun, abs=1e-7 * max(1.0, abs(Q(x))))


def test_empty_weights_and_prior_are_ones():
    A, b, *_ = _lifetime_problem(seed=9, sigma=0.02)
    n_rows, n = A.shape
    given = bff.maxent_invert(A.ravel().tolist(), b.tolist(), [1.0] * n_rows, [1.0] * n, 0.3, n_rows, n)
    default = bff.maxent_invert(A.ravel().tolist(), b.tolist(), [], [], 0.3, n_rows, n)
    np.testing.assert_array_equal(given.get_amplitudes(), default.get_amplitudes())


def test_shapes_that_disagree_are_refused():
    with pytest.raises(ValueError):
        bff.maxent_invert([1.0, 2.0, 3.0], [1.0], [], [], 0.1, 1, 2)
    with pytest.raises(ValueError):
        bff.maxent_solve([1.0, 0.0, 0.0], [1.0, 1.0], [0.5, 0.5], 0.0, 0.1)


def _synthetic(n_col, n_pt, seed, nu, max_iter, tol=1e-4):
    rng = np.random.default_rng(seed)
    basis = np.abs(rng.normal(size=(n_pt, n_col)))
    p_star = np.abs(rng.normal(size=n_col))
    p_star /= p_star.sum()
    y = basis @ p_star + 0.01 * rng.normal(size=n_pt)
    H = (2.0 / n_pt) * (basis.T @ basis)
    g0 = (2.0 / n_pt) * (y @ basis)
    return bff.maxent_solve(H.ravel().tolist(), g0.tolist(), [1.0 / n_col] * n_col, float(y @ y / n_pt), nu,
                            max_iter, tol, 1e-12)


def test_running_out_of_iterations_is_success_but_not_convergence():
    result = _synthetic(129, 900, 2, 1e-6, 300)
    assert result.iterations == 300
    assert result.gradient_angle > 1e-2
    assert not result.converged
    assert result.success


def test_converged_is_exactly_the_stopping_test():
    for n_col, n_pt, seed, nu in ((129, 900, 2, 1e-6), (8, 50, 11, 1e-2), (16, 200, 12, 1e-4)):
        result = _synthetic(n_col, n_pt, seed, nu, 300)
        assert result.converged == (result.gradient_angle <= 1e-4)
    assert _synthetic(8, 50, 11, 1e-2, 300).converged


def test_a_clamped_amplitude_is_released_when_the_objective_wants_it_back():
    """The bounded QP is KKT-correct. Here the unbounded solve sends every
    coordinate below the bound; clamped all together, the first one's gradient
    is negative, so it belongs back off the bound. The old sweep, which never
    released, returned all three at the bound (Q = 0 against -0.0363). The
    early-stop amplitudes are that QP, unregularised."""
    C = np.array([[1.24, 0.28, -0.75], [0.28, 6.29, 1.36], [-0.75, 1.36, 2.52]])
    d = np.array([-0.3, 1.5, 2.0])
    r = bff.maxent_solve(C.ravel().tolist(), (-d).tolist(), [1.0 / 3] * 3, 0.0, 0.0, 1)
    x = np.asarray(r.get_early_stop_amplitudes())
    ref = _bounded_min(lambda x: 0.5 * x @ C @ x + d @ x, lambda x: C @ x + d, np.full(3, 0.5), 1e-12)
    np.testing.assert_allclose(x, ref.x, atol=1e-8)
    np.testing.assert_allclose(x, [0.3 / 1.24, 0.0, 0.0], atol=1e-10)


def _reference_inversion_problem(seed):
    rng = np.random.default_rng(seed)
    t = np.logspace(-3, 1, 60)
    rates = np.logspace(-1, 3, 30)
    A = np.exp(-np.outer(t, rates))
    x_true = np.exp(-0.5 * ((np.log10(rates) - 1.2) / 0.25) ** 2)
    b0 = A @ x_true
    sigma = 0.01 * b0.max() + 0.02 * b0
    b = b0 + rng.normal(0.0, sigma)
    return A, b, 1.0 / sigma ** 2, 0.5 + rng.random(rates.size)


def _reference_normal_problem(seed):
    rng = np.random.default_rng(seed)
    n, k = 200, 40
    t = np.arange(n) * 0.05
    taus = np.linspace(0.3, 6.0, k)
    F = np.exp(-np.outer(t, 1.0 / taus))
    p_true = np.exp(-0.5 * ((taus - 2.5) / 0.3) ** 2)
    y = rng.poisson(F @ p_true / (F @ p_true).max() * 1e4) + 1.0
    s = np.sqrt(y)
    Fw, yw = F / s[:, None], y / s
    H = (2.0 / n) * (Fw.T @ Fw)
    g0 = (2.0 / n) * (yw @ Fw)
    return H, g0, np.full(k, 1.0 / k), float(yw @ yw / n)


@pytest.mark.parametrize("case", REFERENCE["normal_equations"], ids=lambda c: f"seed{c['seed']}-nu{c['nu']:g}-it{c['max_iter']}")
def test_never_above_tttrlibs_answer_on_its_own_problems(case):
    H, g0, m, c = _reference_normal_problem(case["seed"])
    p, r = _solve(H, g0, m, c, case["nu"], case["max_iter"], 1e-4)
    Q, _ = _Q(H, g0, c, m, case["nu"])
    assert Q(p) <= Q(np.asarray(case["p"])) + 1e-12 * max(1.0, abs(Q(p)))
    assert r.objective <= case["objective"] + 1e-12 * max(1.0, abs(case["objective"]))


@pytest.mark.parametrize("case", REFERENCE["inversion"], ids=lambda c: f"seed{c['seed']}-nu{c['nu']:g}")
def test_inversion_never_above_tttrlibs_answer(case):
    A, b, w, m = _reference_inversion_problem(case["seed"])
    nu = case["nu"]
    x = np.asarray(bff.maxent_invert(A.ravel().tolist(), b.tolist(), w.tolist(), m.tolist(), nu, *A.shape,
                                     500, 1e-8).get_amplitudes())

    def Q(x):
        S = np.sum((1.0 - np.log(np.maximum(x, 1e-300) / m)) * x) - m.sum()
        r = A @ x - b
        return r @ (w * r) - nu * nu * S

    assert Q(x) <= Q(np.asarray(case["x"])) + 1e-10 * max(1.0, abs(Q(x)))
